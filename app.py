"""
app.py  --  GridLock Web UI
===========================
A Flask wrapper around the EXISTING GridLock scripts. It does NOT reimplement
detection or tracking -- it shells out to trajectory_collector.py exactly the
way you do on the command line, then reads the resulting CSV and applies the
stop-line + direction the user drew on screen.

FLOW
----
  1. User uploads an MP4.
  2. We grab frame #1 and show it on an HTML canvas (camera is static, so one
     frame's geometry is valid for the whole clip).
  3. User draws: a STOP LINE (2 points) and a DIRECTION ARROW (2 points telling
     us the legal flow direction). These are pixel coords in the ORIGINAL video
     resolution (the frontend rescales for us).
  4. We run trajectory_collector.py on the uploaded video (your code, untouched).
  5. We read the produced CSV and compute:
        - stop-line crossings   (geometry the user drew)
        - wrong-side driving    (heading vs the user's arrow direction)
        - illegal parking       (is_stationary)
     and return them as JSON for the results page + downloadable CSVs.

Run:
    pip install flask opencv-python pandas numpy
    python app.py
    # open http://127.0.0.1:5000
"""

import os
import sys
import json
import subprocess
import glob
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, render_template

try:
    import cv2
except Exception:
    sys.exit("[error] pip install opencv-python")

# project paths -------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "ui_uploads")
FRAME_DIR = os.path.join(HERE, "ui_frames")
OUTPUT_DIR = os.path.join(HERE, "output")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
for d in (UPLOAD_DIR, FRAME_DIR, OUTPUT_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"),
            static_folder=os.path.join(HERE, "ui_frames"), static_url_path="/frames")

# in-memory store of the last uploaded video per session (simple, single-user)
STATE = {}


# ---------------------------------------------------------------------------
# helpers that mirror your export_csvs.py logic (kept here so the UI is
# self-contained; your scripts stay untouched)
# ---------------------------------------------------------------------------

def _ang_diff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def arrow_to_heading(x1, y1, x2, y2):
    """Convert the user's drawn arrow (tail->head) into a heading in degrees
    using the SAME convention as the trajectory CSV: 0=right, 90=down,
    180=left, 270=up (screen/pixel space, y grows downward)."""
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return float(ang % 360.0)


def _point_in_poly(px, py, poly):
    """Ray-casting point-in-polygon. poly = [(x,y),...]. Returns bool array
    if px,py are arrays, or bool if scalars."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > py) != (yj > py)) & \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def compute_violations(csv_path, zone, expected_heading, ghost_min_frames=5):
    """zone = [[x,y],[x,y],...] polygon in ORIGINAL pixel coords (or None).
    A 'zone violation' = a vehicle that was INSIDE the zone at some point and
    later moved OUTSIDE it -- i.e. it crossed out of the zone (e.g. rolled past
    a stop box / crossed the stop region)."""
    df = pd.read_csv(csv_path)
    lengths = df.groupby("tracker_id").size()
    real_ids = lengths[lengths >= ghost_min_frames].index
    df = df[df["tracker_id"].isin(real_ids)].copy()

    fps = (df["frame_index"].max() / df["timestamp_sec"].max()
           if df["timestamp_sec"].max() else 30.0)

    # --- zone crossing (inside -> outside) ---
    crossings = []
    if zone and len(zone) >= 3:
        poly = [(float(x), float(y)) for x, y in zone]
        for tid, g in df.groupby("tracker_id"):
            g = g.sort_values("frame_index")
            ins = _point_in_poly(g["bottom_center_x"].to_numpy(),
                                 g["bottom_center_y"].to_numpy(), poly)
            # find a transition from inside (True) to outside (False)
            was_inside = False
            for k in range(len(ins)):
                if ins[k]:
                    was_inside = True
                elif was_inside and not ins[k]:
                    row = g.iloc[k]
                    crossings.append({
                        "tracker_id": int(tid),
                        "vehicle_class": g["vehicle_class"].mode().iat[0],
                        "exited_zone_at_sec": round(float(row["timestamp_sec"]), 2),
                        "exit_x": round(float(row["bottom_center_x"]), 1),
                        "exit_y": round(float(row["bottom_center_y"]), 1),
                        "speed_px_sec": round(float(row["speed_px_sec"]), 1),
                    })
                    break  # one flag per vehicle

    # write CSVs for download
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    # zone coordinates file (always written when a zone exists)
    if zone and len(zone) >= 3:
        zp = os.path.join(REPORT_DIR, f"ui_zone_coords_{stamp}.csv")
        pd.DataFrame(
            [{"point_index": i, "x": int(round(x)), "y": int(round(y))}
             for i, (x, y) in enumerate(zone)]
        ).to_csv(zp, index=False)
        paths["zone"] = os.path.basename(zp)

    if crossings:
        p = os.path.join(REPORT_DIR, f"ui_zone_violations_{stamp}.csv")
        pd.DataFrame(crossings).to_csv(p, index=False)
        paths["zone_violations"] = os.path.basename(p)

    return {
        "summary": {
            "real_vehicles": int(df["tracker_id"].nunique()),
            "zone_violations": len(crossings),
        },
        "zone_violations": crossings,
        "zone_coords": zone,
        "analytics": _build_analytics(df, crossings, fps),
        "files": paths,
    }


def _build_analytics(df, crossings, fps):
    """Derive chart-ready aggregates from the (already ghost-filtered) df.
    Everything here comes straight from the trajectory CSV."""
    # class breakdown by unique vehicle
    by_first = df.groupby("tracker_id")["vehicle_class"].first()
    class_counts = by_first.value_counts().to_dict()

    duration = float(df["timestamp_sec"].max()) if len(df) else 0.0

    # traffic flow: count of NEW vehicles entering per time bucket
    bucket = max(2.0, round(duration / 12.0))  # ~12 buckets
    firsts = df.groupby("tracker_id")["timestamp_sec"].min()
    edges = np.arange(0, duration + bucket, bucket)
    flow_labels, flow_values = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        n = int(((firsts >= lo) & (firsts < hi)).sum())
        flow_labels.append(f"{int(lo)}s")
        flow_values.append(n)

    # speed distribution (moving only), in px/sec, bucketed
    mv = df[df["speed_px_sec"] > 5]["speed_px_sec"].to_numpy()
    speed_bins = [0, 25, 50, 100, 200, 400, 1e9]
    speed_labels = ["0-25", "25-50", "50-100", "100-200", "200-400", "400+"]
    speed_values = []
    for i in range(len(speed_bins) - 1):
        speed_values.append(int(((mv >= speed_bins[i]) & (mv < speed_bins[i + 1])).sum()))

    # violations by vehicle class
    viol_by_class = {}
    for c in crossings:
        viol_by_class[c["vehicle_class"]] = viol_by_class.get(c["vehicle_class"], 0) + 1

    # headline KPIs
    total = int(df["tracker_id"].nunique())
    viol = len(crossings)
    rate = round(100.0 * viol / total, 1) if total else 0.0
    peak = max(flow_values) if flow_values else 0

    return {
        "kpis": {
            "vehicles": total,
            "violations": viol,
            "violation_rate": rate,
            "duration_sec": round(duration, 1),
            "peak_inflow": peak,
        },
        "class_breakdown": class_counts,
        "flow": {"labels": flow_labels, "values": flow_values},
        "speed": {"labels": speed_labels, "values": speed_values},
        "violations_by_class": viol_by_class,
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Receive the MP4, save it, extract frame #1, return its URL + dimensions
    so the canvas can size itself correctly."""
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "no file"}), 400
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vid_path = os.path.join(UPLOAD_DIR, f"upload_{stamp}.mp4")
    f.save(vid_path)

    cap = cv2.VideoCapture(vid_path)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return jsonify({"error": "could not read video"}), 400
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_name = f"frame_{stamp}.jpg"
    cv2.imwrite(os.path.join(FRAME_DIR, frame_name), frame)

    STATE["video"] = vid_path
    return jsonify({
        "frame_url": f"/frames/{frame_name}",
        "width": W, "height": H,
    })


def _find_ffmpeg():
    """Locate an ffmpeg binary without requiring a system install.
    imageio-ffmpeg ships its own ffmpeg binary via pip, so if it's installed
    we use that. Falls back to a system ffmpeg on PATH if present."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    import shutil
    return shutil.which("ffmpeg")


def _to_browser_mp4(src_path):
    """Re-encode an mp4 to H.264 so browsers can play it inline. Tries, in
    order: (1) bundled/system ffmpeg via imageio-ffmpeg, (2) returns the source
    unchanged if nothing is available (download still works). The zone video
    itself is already written with an H.264-capable writer below, so this is
    mainly for the collector's mp4v output."""
    base, _ = os.path.splitext(src_path)
    out = base + "_web.mp4"
    ff = _find_ffmpeg()
    if ff:
        try:
            subprocess.run(
                [ff, "-y", "-i", src_path, "-vcodec", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                check=True, capture_output=True,
            )
            return out
        except Exception:
            pass
    return src_path


def make_zone_video(input_video, csv_path, zone, crossing_ids, ghost_min_frames=5):
    """Render a NEW video that draws the user's zone polygon on every frame and
    highlights vehicles: red box if that tracker_id ever exits the zone
    (a violation), otherwise a thin grey box. Returns the output path."""
    df = pd.read_csv(csv_path)
    lengths = df.groupby("tracker_id").size()
    real_ids = set(lengths[lengths >= ghost_min_frames].index)

    # index rows by frame for fast per-frame lookup
    by_frame = {}
    for _, r in df.iterrows():
        if int(r["tracker_id"]) not in real_ids:
            continue
        by_frame.setdefault(int(r["frame_index"]), []).append(r)

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_out = os.path.join(OUTPUT_DIR, f"zone_overlay_{stamp}.mp4")
    # try browser-playable codecs first; fall back to mp4v if unavailable
    writer = None
    for codec in ("avc1", "H264", "h264"):
        w = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*codec), fps, (W, H))
        if w.isOpened():
            writer = w
            break
        w.release()
    used_mp4v = False
    if writer is None:
        writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        used_mp4v = True

    poly_np = (np.array(zone, dtype=np.int32).reshape(-1, 1, 2)
               if zone and len(zone) >= 3 else None)
    viol = set(crossing_ids)

    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1

        # draw the zone polygon (semi-transparent red fill + outline)
        if poly_np is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly_np], (0, 0, 200))
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [poly_np], True, (0, 0, 220), 2)

        # draw vehicles present in this frame
        for r in by_frame.get(fi, []):
            tid = int(r["tracker_id"])
            x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            is_viol = tid in viol
            col = (0, 0, 230) if is_viol else (150, 150, 150)
            th = 3 if is_viol else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, th)
            if is_viol:
                cv2.putText(frame, f"VIOLATION #{tid}", (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 230), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    # if we already wrote H.264, it's browser-ready; otherwise try to convert
    if used_mp4v:
        return _to_browser_mp4(raw_out)
    return raw_out


@app.route("/process", methods=["POST"])
def process():
    """Run the user's geometry through the existing collector + violation logic."""
    data = request.get_json(force=True)
    zone = data.get("zone")                  # [[x,y],...] polygon in original px
    arrow = data.get("arrow")                # [x1,y1,x2,y2] tail->head
    ghost = int(data.get("ghost_min_frames", 5))

    vid_path = STATE.get("video")
    if not vid_path or not os.path.exists(vid_path):
        return jsonify({"error": "no uploaded video in session"}), 400

    # 1) run YOUR collector, untouched, exactly like the CLI
    before = set(glob.glob(os.path.join(OUTPUT_DIR, "trajectory_data_*.csv")))
    try:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "trajectory_collector.py"),
             "--video", vid_path, "--output-dir", OUTPUT_DIR],
            check=True, cwd=HERE,
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"collector failed: {e}"}), 500

    after = set(glob.glob(os.path.join(OUTPUT_DIR, "trajectory_data_*.csv")))
    new_csv = sorted(after - before)
    if not new_csv:
        # fall back to newest CSV if naming differed
        all_csv = sorted(after, key=os.path.getmtime)
        if not all_csv:
            return jsonify({"error": "collector produced no CSV"}), 500
        csv_path = all_csv[-1]
    else:
        csv_path = new_csv[-1]

    # 2) derive heading from arrow, apply geometry
    expected_heading = None
    if arrow and len(arrow) == 4:
        expected_heading = arrow_to_heading(*arrow)

    result = compute_violations(csv_path, zone, expected_heading, ghost)
    result["csv"] = os.path.basename(csv_path)
    result["expected_heading"] = round(expected_heading, 1) if expected_heading is not None else None

    # 3) videos: (a) the collector's annotated output, re-encoded for browser;
    #            (b) a NEW zone-overlay video highlighting violators
    videos = {}

    # collector's annotated_output_<stamp>.mp4 lives alongside the CSV stamp
    stamp = os.path.basename(csv_path).replace("trajectory_data_", "").replace(".csv", "")
    annotated = os.path.join(OUTPUT_DIR, f"annotated_output_{stamp}.mp4")
    if os.path.exists(annotated):
        web_annot = _to_browser_mp4(annotated)
        videos["annotated"] = os.path.basename(web_annot)

    # new zone-overlay video
    crossing_ids = [c["tracker_id"] for c in result["zone_violations"]]
    try:
        zone_vid = make_zone_video(vid_path, csv_path, zone, crossing_ids, ghost)
        videos["zone"] = os.path.basename(zone_vid)
    except Exception as e:
        videos["zone_error"] = str(e)

    result["videos"] = videos
    return jsonify(result)


@app.route("/download/<path:fname>")
def download(fname):
    return send_from_directory(REPORT_DIR, fname, as_attachment=True)


@app.route("/video/<path:fname>")
def video(fname):
    """Serve a generated video from output/ for inline <video> playback."""
    return send_from_directory(OUTPUT_DIR, fname, as_attachment=False)


@app.route("/video-download/<path:fname>")
def video_download(fname):
    return send_from_directory(OUTPUT_DIR, fname, as_attachment=True)


if __name__ == "__main__":
    print("GridLock UI -> http://127.0.0.1:5000")
    app.run(debug=True, port=5000)