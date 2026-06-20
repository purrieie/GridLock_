"""
trajectory_analytics.py
=======================
STANDALONE analysis layer. Reads ONLY the trajectory CSV produced by
trajectory_collector.py. It never loads YOLO, never opens the video, never
re-runs the tracker. All the expensive perception work was already done and
written to the CSV -- this module is pure, cheap, deterministic rule-checking
on top of that data. That separation is the whole point of the architecture:
GPU work happens once, analysis runs in milliseconds and can be re-run with
different thresholds as many times as you like.

WHAT IT DERIVES (everything below comes from the CSV alone):
  1. Scene-level analytics .... counts, class mix, duration, density over time
  2. Ghost filtering .......... drops YOLO 2-3 frame hallucinations
  3. Per-vehicle summary ...... trajectory length, dwell time, speed profile
  4. Illegal parking .......... from the is_stationary flag + dwell duration
  5. Wrong-side driving ....... from heading_deg vs. expected flow direction
  6. Stop-line violation ...... from bottom_center crossing a line ROI

WHAT IT CANNOT DERIVE (needs the pixels, not the trajectory):
  - Helmet / seatbelt / triple-riding -> see rider_safety_layer.py
  - License plate text (OCR)
  Anything about what's INSIDE a box is out of scope here by design.

USAGE
-----
    python trajectory_analytics.py --csv output/trajectory_data_XXXX.csv

    # optional: write per-vehicle + violation tables back out as CSVs
    python trajectory_analytics.py --csv <file> --write-reports

    # optional: feed ROIs for wrong-side / stop-line checks (see --help)
    python trajectory_analytics.py --csv <file> \
        --stop-line "100,800,1900,800" \
        --expected-heading 90 --heading-tolerance 75

Edit the CONFIG block below to tune thresholds, or pass them on the CLI.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Try to import the project's shared config so thresholds stay in sync with
# the collector. If it isn't importable (e.g. run from another folder), fall
# back to local defaults -- this module must work standalone.
try:
    import config as _cfg
    GHOST_MIN_FRAMES = 5
    STATIONARY_MIN_FRAMES = getattr(_cfg, "STATIONARY_MIN_FRAMES", 30)
except Exception:
    _cfg = None
    GHOST_MIN_FRAMES = 5
    STATIONARY_MIN_FRAMES = 30


# ---------------------------------------------------------------------------
# CONFIG (CLI flags override these)
# ---------------------------------------------------------------------------

# A track shorter than this many frames is treated as a detection artifact
# ("ghost") and excluded from all violation logic and most stats.
DEFAULT_GHOST_MIN_FRAMES = GHOST_MIN_FRAMES

# Illegal-parking: a vehicle must hold is_stationary==1 for at least this many
# total frames to count as "parked" rather than just "stopped at a light".
# STATIONARY_MIN_FRAMES (from config) already gates when the flag turns on;
# this is an ADDITIONAL dwell requirement on top of it.
DEFAULT_PARKED_MIN_FRAMES = STATIONARY_MIN_FRAMES * 3   # ~3s extra beyond flag

# Wrong-side: by default disabled (needs a known expected flow direction).
# Pass --expected-heading to enable. heading_deg convention (from README):
#   0=right, 90=down, 180=left, 270=up  (screen space)
DEFAULT_HEADING_TOLERANCE = 75.0   # +/- degrees considered "correct direction"

# Wrong-side debounce: ignore brief heading flips (turning, jitter). Vehicle
# must point the wrong way for at least this fraction of its moving frames.
DEFAULT_WRONGSIDE_MIN_FRACTION = 0.6
# ...and have at least this much real motion (px/sec) to be judged at all.
DEFAULT_WRONGSIDE_MIN_SPEED = 15.0


# ---------------------------------------------------------------------------
# Loading / cleaning
# ---------------------------------------------------------------------------

def load_csv(path):
    if not os.path.exists(path):
        sys.exit(f"[error] CSV not found: {path}")
    df = pd.read_csv(path)
    required = {
        "tracker_id", "frame_index", "timestamp_sec", "vehicle_class",
        "bottom_center_x", "bottom_center_y", "speed_px_sec", "heading_deg",
        "is_stationary", "frames_since_first_seen",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[error] CSV is missing expected columns: {sorted(missing)}")
    return df


def infer_fps(df):
    """fps = frames / seconds, read straight from the CSV's own timeline."""
    t = df["timestamp_sec"].max()
    f = df["frame_index"].max()
    if t and t > 0:
        return round(f / t, 3)
    return 30.0


def split_ghosts(df, ghost_min_frames):
    """Return (real_df, ghost_ids). A ghost is any track whose TOTAL visible
    length is below ghost_min_frames -- YOLO hallucinations that pop for a few
    frames then vanish. Everything downstream uses real_df only."""
    track_len = df.groupby("tracker_id").size()
    ghost_ids = set(track_len[track_len < ghost_min_frames].index)
    real_df = df[~df["tracker_id"].isin(ghost_ids)].copy()
    return real_df, ghost_ids


# ---------------------------------------------------------------------------
# 1. Scene-level analytics
# ---------------------------------------------------------------------------

def scene_summary(df, real_df, ghost_ids, fps):
    duration = float(df["timestamp_sec"].max())
    n_frames = int(df["frame_index"].max()) + 1

    # class mix counted by UNIQUE real vehicle, not by row
    first_class = real_df.groupby("tracker_id")["vehicle_class"].first()
    class_mix = first_class.value_counts().to_dict()

    # peak concurrent objects in any single frame (real only)
    per_frame = real_df.groupby("frame_index")["tracker_id"].nunique()
    peak_concurrent = int(per_frame.max()) if len(per_frame) else 0
    peak_frame = int(per_frame.idxmax()) if len(per_frame) else -1

    return {
        "duration_sec": round(duration, 2),
        "fps": fps,
        "total_frames": n_frames,
        "unique_tracks_raw": int(df["tracker_id"].nunique()),
        "ghosts_dropped": len(ghost_ids),
        "real_vehicles": int(real_df["tracker_id"].nunique()),
        "class_mix": class_mix,
        "peak_concurrent": peak_concurrent,
        "peak_concurrent_at_sec": round(peak_frame / fps, 2) if peak_frame >= 0 else None,
        "avg_concurrent": round(float(per_frame.mean()), 1) if len(per_frame) else 0,
    }


# ---------------------------------------------------------------------------
# 2. Per-vehicle summary
# ---------------------------------------------------------------------------

def per_vehicle_summary(real_df, fps):
    rows = []
    for tid, g in real_df.groupby("tracker_id"):
        g = g.sort_values("frame_index")
        moving = g[g["speed_px_sec"] > 1.0]
        # path length = sum of step distances between consecutive bottom-centers
        dx = g["bottom_center_x"].diff()
        dy = g["bottom_center_y"].diff()
        path_len = float(np.nansum(np.hypot(dx, dy)))
        rows.append({
            "tracker_id": int(tid),
            "vehicle_class": g["vehicle_class"].mode().iat[0],
            "first_seen_sec": round(float(g["timestamp_sec"].min()), 2),
            "last_seen_sec": round(float(g["timestamp_sec"].max()), 2),
            "dwell_sec": round(float(g["timestamp_sec"].max() - g["timestamp_sec"].min()), 2),
            "frames_visible": int(len(g)),
            "path_length_px": round(path_len, 1),
            "avg_speed_px_sec": round(float(moving["speed_px_sec"].mean()), 1) if len(moving) else 0.0,
            "max_speed_px_sec": round(float(g["speed_px_sec"].max()), 1),
            "stationary_frames": int(g["is_stationary"].sum()),
        })
    return pd.DataFrame(rows).sort_values("dwell_sec", ascending=False)


# ---------------------------------------------------------------------------
# 3. Illegal parking  (from is_stationary)
# ---------------------------------------------------------------------------

def detect_illegal_parking(real_df, fps, parked_min_frames):
    """A vehicle is flagged if its is_stationary frames meet/exceed the dwell
    requirement. is_stationary already encodes 'slower than threshold for
    STATIONARY_MIN_FRAMES consecutive frames' per the collector, so here we
    just require sustained parking on top of that."""
    out = []
    for tid, g in real_df.groupby("tracker_id"):
        stat_frames = int(g["is_stationary"].sum())
        if stat_frames >= parked_min_frames:
            gs = g[g["is_stationary"] == 1]
            out.append({
                "tracker_id": int(tid),
                "vehicle_class": g["vehicle_class"].mode().iat[0],
                "stationary_frames": stat_frames,
                "stationary_sec": round(stat_frames / fps, 1),
                "location_x": round(float(gs["bottom_center_x"].median()), 1),
                "location_y": round(float(gs["bottom_center_y"].median()), 1),
                "from_sec": round(float(gs["timestamp_sec"].min()), 2),
                "to_sec": round(float(gs["timestamp_sec"].max()), 2),
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 4. Wrong-side driving  (from heading_deg)
# ---------------------------------------------------------------------------

def _angular_diff(a, b):
    """Smallest absolute difference between two angles in degrees (0..180)."""
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def detect_wrong_side(real_df, expected_heading, tolerance,
                      min_fraction, min_speed):
    """Flag vehicles whose heading_deg points opposite the expected traffic
    flow for a sustained fraction of their moving frames. Requires the caller
    to supply expected_heading (the legal direction for this camera/lane).

    heading convention (README): 0=right, 90=down, 180=left, 270=up.
    A vehicle is 'wrong-side' when its heading is closer to the OPPOSITE of
    expected_heading than to expected_heading itself (beyond tolerance)."""
    if expected_heading is None:
        return None  # disabled

    opposite = (expected_heading + 180.0) % 360.0
    out = []
    for tid, g in real_df.groupby("tracker_id"):
        moving = g[(g["speed_px_sec"] >= min_speed) & (g["heading_deg"] >= 0)]
        if len(moving) < 5:
            continue
        h = moving["heading_deg"].to_numpy()
        # frame counts as wrong-side if nearer to opposite than to expected
        d_exp = _angular_diff(h, expected_heading)
        d_opp = _angular_diff(h, opposite)
        wrong = (d_opp < d_exp) & (d_exp > tolerance)
        frac = wrong.mean()
        if frac >= min_fraction:
            out.append({
                "tracker_id": int(tid),
                "vehicle_class": g["vehicle_class"].mode().iat[0],
                "wrong_fraction": round(float(frac), 2),
                "median_heading_deg": round(float(np.median(h)), 1),
                "expected_heading_deg": expected_heading,
                "moving_frames": int(len(moving)),
                "from_sec": round(float(moving["timestamp_sec"].min()), 2),
                "to_sec": round(float(moving["timestamp_sec"].max()), 2),
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 5. Stop-line violation  (bottom_center crossing a line ROI)
# ---------------------------------------------------------------------------

def detect_stop_line(real_df, line, fps):
    """line = (x1,y1,x2,y2) defining a stop line in pixel space. A crossing is
    recorded when a vehicle's bottom_center moves from one side of the line to
    the other between consecutive frames. This is a geometric event only; it
    does NOT know the signal phase -- combine with red-light timing elsewhere."""
    if line is None:
        return None
    (lx1, ly1, lx2, ly2) = line

    def side(px, py):
        # sign of cross product (line direction) x (point - line_start)
        return np.sign((lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1))

    out = []
    for tid, g in real_df.groupby("tracker_id"):
        g = g.sort_values("frame_index")
        s = side(g["bottom_center_x"].to_numpy(), g["bottom_center_y"].to_numpy())
        changes = np.where(np.diff(s) != 0)[0]
        if len(changes):
            i = changes[0]  # first crossing
            row = g.iloc[i + 1]
            out.append({
                "tracker_id": int(tid),
                "vehicle_class": g["vehicle_class"].mode().iat[0],
                "crossed_at_sec": round(float(row["timestamp_sec"]), 2),
                "crossed_at_frame": int(row["frame_index"]),
                "x": round(float(row["bottom_center_x"]), 1),
                "y": round(float(row["bottom_center_y"]), 1),
                "speed_px_sec": round(float(row["speed_px_sec"]), 1),
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def _hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_report(scene, veh_df, parking_df, wrong_df, stopline_df):
    _hr("SCENE SUMMARY")
    print(f"  Duration            : {scene['duration_sec']} s "
          f"({scene['total_frames']} frames @ {scene['fps']} fps)")
    print(f"  Tracks (raw)        : {scene['unique_tracks_raw']}")
    print(f"  Ghosts dropped      : {scene['ghosts_dropped']}")
    print(f"  Real vehicles       : {scene['real_vehicles']}")
    print(f"  Peak concurrent     : {scene['peak_concurrent']} "
          f"(at {scene['peak_concurrent_at_sec']} s)")
    print(f"  Avg concurrent      : {scene['avg_concurrent']}")
    print(f"  Class mix           :")
    for k, v in sorted(scene["class_mix"].items(), key=lambda x: -x[1]):
        print(f"      {k:<12}: {v}")

    _hr("PER-VEHICLE SUMMARY (top 10 by dwell)")
    cols = ["tracker_id", "vehicle_class", "first_seen_sec", "last_seen_sec",
            "dwell_sec", "frames_visible", "path_length_px",
            "avg_speed_px_sec", "max_speed_px_sec"]
    with pd.option_context("display.max_rows", 10, "display.width", 120):
        print(veh_df[cols].head(10).to_string(index=False))

    _hr("ILLEGAL PARKING CANDIDATES (from is_stationary)")
    if len(parking_df):
        print(parking_df.to_string(index=False))
    else:
        print("  none flagged at current dwell threshold")

    _hr("WRONG-SIDE DRIVING (from heading_deg)")
    if wrong_df is None:
        print("  disabled -- pass --expected-heading <deg> to enable")
    elif len(wrong_df):
        print(wrong_df.to_string(index=False))
    else:
        print("  none flagged")

    _hr("STOP-LINE CROSSINGS (geometric event only, no signal phase)")
    if stopline_df is None:
        print("  disabled -- pass --stop-line x1,y1,x2,y2 to enable")
    elif len(stopline_df):
        print(stopline_df.to_string(index=False))
    else:
        print("  no crossings detected")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_line(s):
    if not s:
        return None
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        sys.exit("[error] --stop-line must be 'x1,y1,x2,y2'")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(description="Derive analytics + violations from a trajectory CSV.")
    ap.add_argument("--csv", required=True, help="Path to trajectory_data_*.csv")
    ap.add_argument("--ghost-min-frames", type=int, default=DEFAULT_GHOST_MIN_FRAMES)
    ap.add_argument("--parked-min-frames", type=int, default=DEFAULT_PARKED_MIN_FRAMES)
    ap.add_argument("--expected-heading", type=float, default=None,
                    help="Legal flow direction in deg (0=right,90=down,180=left,270=up). Enables wrong-side check.")
    ap.add_argument("--heading-tolerance", type=float, default=DEFAULT_HEADING_TOLERANCE)
    ap.add_argument("--wrongside-min-fraction", type=float, default=DEFAULT_WRONGSIDE_MIN_FRACTION)
    ap.add_argument("--wrongside-min-speed", type=float, default=DEFAULT_WRONGSIDE_MIN_SPEED)
    ap.add_argument("--stop-line", type=str, default=None, help="'x1,y1,x2,y2' pixel coords")
    ap.add_argument("--write-reports", action="store_true",
                    help="Also write per-vehicle + violation tables next to the input CSV")
    args = ap.parse_args()

    df = load_csv(args.csv)
    fps = infer_fps(df)
    real_df, ghost_ids = split_ghosts(df, args.ghost_min_frames)

    scene = scene_summary(df, real_df, ghost_ids, fps)
    veh_df = per_vehicle_summary(real_df, fps)
    parking_df = detect_illegal_parking(real_df, fps, args.parked_min_frames)
    wrong_df = detect_wrong_side(real_df, args.expected_heading,
                                 args.heading_tolerance,
                                 args.wrongside_min_fraction,
                                 args.wrongside_min_speed)
    stopline_df = detect_stop_line(real_df, _parse_line(args.stop_line), fps)

    print_report(scene, veh_df, parking_df, wrong_df, stopline_df)

    if args.write_reports:
        base = os.path.splitext(args.csv)[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        veh_df.to_csv(f"{base}__vehicles_{stamp}.csv", index=False)
        if len(parking_df):
            parking_df.to_csv(f"{base}__parking_{stamp}.csv", index=False)
        if wrong_df is not None and len(wrong_df):
            wrong_df.to_csv(f"{base}__wrongside_{stamp}.csv", index=False)
        if stopline_df is not None and len(stopline_df):
            stopline_df.to_csv(f"{base}__stopline_{stamp}.csv", index=False)
        print(f"[ok] reports written next to {args.csv}")


if __name__ == "__main__":
    main()