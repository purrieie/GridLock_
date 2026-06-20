"""
export_csvs.py
==============
STANDALONE exporter. Reads ONLY the trajectory CSV produced by
trajectory_collector.py and writes clean, separate, Excel-friendly CSV files
into output/reports/ so you can open and read them instead of squinting at the
terminal. It touches nothing else -- no model, no video, no existing code.

WHAT IT WRITES (into output/reports/):
  motorbikes_<stamp>.csv ........ one row per real motorbike (ghosts removed),
                                  with summary stats per bike.
  parking_<stamp>.csv ........... illegal-parking candidates (from is_stationary).
  wrongside_<stamp>.csv ......... wrong-side driving (needs --expected-heading).

USAGE
-----
    python export_csvs.py --csv output/trajectory_data_XXXX.csv

    # to also get the wrong-side file, give the legal flow direction:
    python export_csvs.py --csv output/trajectory_data_XXXX.csv --expected-heading 270

heading convention (from README): 0=right, 90=down, 180=left, 270=up.
For the sample Test.mp4 traffic flows upward, so use 270.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Keep thresholds in sync with the shared config when reachable; otherwise use
# sensible local defaults so this runs from anywhere.
try:
    import config as _cfg
    STATIONARY_MIN_FRAMES = getattr(_cfg, "STATIONARY_MIN_FRAMES", 30)
except Exception:
    STATIONARY_MIN_FRAMES = 30

GHOST_MIN_FRAMES = 5                       # tracks shorter than this = junk
PARKED_MIN_FRAMES = STATIONARY_MIN_FRAMES * 3   # sustained parking requirement
HEADING_TOLERANCE = 75.0
WRONGSIDE_MIN_FRACTION = 0.6
WRONGSIDE_MIN_SPEED = 15.0


def load(path):
    if not os.path.exists(path):
        sys.exit(f"[error] CSV not found: {path}")
    return pd.read_csv(path)


def infer_fps(df):
    t, f = df["timestamp_sec"].max(), df["frame_index"].max()
    return round(f / t, 3) if t and t > 0 else 30.0


def drop_ghosts(df, ghost_min_frames=GHOST_MIN_FRAMES):
    lengths = df.groupby("tracker_id").size()
    real_ids = lengths[lengths >= ghost_min_frames].index
    return df[df["tracker_id"].isin(real_ids)].copy()


# ---- 1. motorbikes only -----------------------------------------------------

def motorbikes_table(real_df, fps, parking_ids=None, wrongside_ids=None):
    """One row per real motorbike. The `violations` column lists what each bike
    did (e.g. 'wrong_side', 'illegal_parking') or 'none' if it was clean, and
    `is_violation` is a simple 1/0 flag for easy filtering/sorting in Excel."""
    parking_ids = parking_ids or set()
    wrongside_ids = wrongside_ids or set()
    moto = real_df[real_df["vehicle_class"] == "motorcycle"]
    rows = []
    for tid, g in moto.groupby("tracker_id"):
        g = g.sort_values("frame_index")
        moving = g[g["speed_px_sec"] > 1.0]
        dx, dy = g["bottom_center_x"].diff(), g["bottom_center_y"].diff()

        vios = []
        if int(tid) in wrongside_ids:
            vios.append("wrong_side")
        if int(tid) in parking_ids:
            vios.append("illegal_parking")
        violation_str = ", ".join(vios) if vios else "none"

        rows.append({
            "tracker_id": int(tid),
            "is_violation": 1 if vios else 0,
            "violations": violation_str,
            "first_seen_sec": round(float(g["timestamp_sec"].min()), 2),
            "last_seen_sec": round(float(g["timestamp_sec"].max()), 2),
            "dwell_sec": round(float(g["timestamp_sec"].max() - g["timestamp_sec"].min()), 2),
            "frames_visible": int(len(g)),
            "avg_speed_px_sec": round(float(moving["speed_px_sec"].mean()), 1) if len(moving) else 0.0,
            "max_speed_px_sec": round(float(g["speed_px_sec"].max()), 1),
            "path_length_px": round(float(np.nansum(np.hypot(dx, dy))), 1),
            "max_frames_tracked": int(g["frames_since_first_seen"].max()),
        })
    # violations first (is_violation desc), then longest-seen
    return pd.DataFrame(rows).sort_values(["is_violation", "dwell_sec"], ascending=[False, False])


# ---- 2. illegal parking -----------------------------------------------------

def parking_table(real_df, fps):
    out = []
    for tid, g in real_df.groupby("tracker_id"):
        stat = int(g["is_stationary"].sum())
        if stat >= PARKED_MIN_FRAMES:
            gs = g[g["is_stationary"] == 1]
            out.append({
                "tracker_id": int(tid),
                "vehicle_class": g["vehicle_class"].mode().iat[0],
                "stationary_sec": round(stat / fps, 1),
                "from_sec": round(float(gs["timestamp_sec"].min()), 2),
                "to_sec": round(float(gs["timestamp_sec"].max()), 2),
                "location_x": round(float(gs["bottom_center_x"].median()), 1),
                "location_y": round(float(gs["bottom_center_y"].median()), 1),
            })
    return pd.DataFrame(out).sort_values("stationary_sec", ascending=False) if out else pd.DataFrame()


# ---- 3. wrong-side ----------------------------------------------------------

def _ang_diff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def wrongside_table(real_df, expected_heading):
    if expected_heading is None:
        return None
    opp = (expected_heading + 180.0) % 360.0
    out = []
    for tid, g in real_df.groupby("tracker_id"):
        mv = g[(g["speed_px_sec"] >= WRONGSIDE_MIN_SPEED) & (g["heading_deg"] >= 0)]
        if len(mv) < 5:
            continue
        h = mv["heading_deg"].to_numpy()
        wrong = (_ang_diff(h, opp) < _ang_diff(h, expected_heading)) & (_ang_diff(h, expected_heading) > HEADING_TOLERANCE)
        frac = wrong.mean()
        if frac >= WRONGSIDE_MIN_FRACTION:
            out.append({
                "tracker_id": int(tid),
                "vehicle_class": g["vehicle_class"].mode().iat[0],
                "wrong_fraction": round(float(frac), 2),
                "median_heading_deg": round(float(np.median(h)), 1),
                "expected_heading_deg": expected_heading,
                "from_sec": round(float(mv["timestamp_sec"].min()), 2),
                "to_sec": round(float(mv["timestamp_sec"].max()), 2),
            })
    return pd.DataFrame(out).sort_values("wrong_fraction", ascending=False) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Export clean per-category CSVs from a trajectory CSV.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--expected-heading", type=float, default=None,
                    help="Legal flow direction in deg to enable the wrong-side file (Test.mp4 = 270).")
    ap.add_argument("--out-dir", default=None, help="Default: <csv folder>/reports")
    ap.add_argument("--ghost-min-frames", type=int, default=GHOST_MIN_FRAMES,
                    help="Min frames a track must last to count as real. "
                         "Lower = more vehicles shown but more junk. Default 5; "
                         "try 2 or 3 to surface flickery motorbikes.")
    args = ap.parse_args()

    df = load(args.csv)
    fps = infer_fps(df)
    real_df = drop_ghosts(df, args.ghost_min_frames)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.csv) or ".", "reports")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    written = []

    # compute violations first so we can flag them on the motorbikes table
    park = parking_table(real_df, fps)
    ws = wrongside_table(real_df, args.expected_heading)

    parking_ids = set(park["tracker_id"]) if len(park) else set()
    wrongside_ids = set(ws["tracker_id"]) if (ws is not None and len(ws)) else set()

    moto = motorbikes_table(real_df, fps, parking_ids, wrongside_ids)
    p = os.path.join(out_dir, f"motorbikes_{stamp}.csv")
    moto.to_csv(p, index=False); written.append((p, len(moto)))

    p = os.path.join(out_dir, f"parking_{stamp}.csv")
    park.to_csv(p, index=False); written.append((p, len(park)))

    if ws is not None:
        p = os.path.join(out_dir, f"wrongside_{stamp}.csv")
        ws.to_csv(p, index=False); written.append((p, len(ws)))

    print("\nExported CSVs:")
    for path, n in written:
        print(f"  {path}  ({n} rows)")
    if ws is None:
        print("  (wrong-side skipped -- pass --expected-heading 270 to include it)")
    print("\nOpen these in Excel / Numbers. Done.\n")


if __name__ == "__main__":
    main()