"""
rider_safety_layer.py
=====================
SECOND-LAYER detector for the two violations that the trajectory CSV alone
can NEVER answer, because they depend on what's INSIDE the box, not where the
box moves: HELMET non-compliance and TRIPLE-RIDING.

WHY THIS IS A SEPARATE LAYER (and why that kills the noise)
----------------------------------------------------------
If you just run a person/helmet model over every full frame, you get a flood
of false positives: pedestrians on the footpath, people in cars, reflections,
distant blobs -- none of which are motorcycle riders. That's the "noise" you
were seeing.

This module does the opposite. It is CROP-TARGETED and TRAJECTORY-GATED:

  1. It reads the CSV first and only looks at frames where a MOTORCYCLE
     actually exists, and only inside an expanded crop around that motorcycle.
     Pedestrians and car-occupants elsewhere in the frame are never even seen
     by the model -> massive noise reduction for free.

  2. It only judges a motorcycle that the tracker has held for
     >= TRIPLE_RIDING_MIN_FRAMES consecutive frames (the README's debounce
     window). A 2-3 frame ghost motorcycle can't trigger a flag.

  3. A violation is only RAISED after it persists across a debounce window of
     its own, so a single bad frame (someone's head briefly occluded, a helmet
     mistaken for hair for one frame) does not produce a flag.

This module DOES open the video and DOES run a model -- but selectively, on a
tiny fraction of pixels. It writes its OWN output CSV. It does not touch
trajectory_collector.py, config.py, or the existing trajectory CSV.

DETECTION BACKEND (best-accuracy with graceful fallback)
--------------------------------------------------------
Set HELMET_MODEL_PATH below to a fine-tuned helmet/no-helmet .pt for best
accuracy. If that file is absent, the module automatically falls back to a
COCO 'person'-counting heuristic using the same base model the collector
already uses (yolo11m.pt) -- so it RUNS regardless, just with helmet detection
downgraded to "rider present?" until you drop in proper weights.

USAGE
-----
    python rider_safety_layer.py \
        --video Test.mp4 \
        --csv output/trajectory_data_XXXX.csv

    # point at a fine-tuned helmet model for best results:
    python rider_safety_layer.py --video Test.mp4 --csv <file> \
        --helmet-model weights/helmet_yolo.pt
"""

import argparse
import os
import sys
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import cv2
except Exception:
    sys.exit("[error] opencv-python required: pip install opencv-python")

try:
    from ultralytics import YOLO
except Exception:
    sys.exit("[error] ultralytics required: pip install ultralytics")

# Reuse the project's shared config when available (keeps base model in sync).
try:
    import config as _cfg
    BASE_MODEL_PATH = getattr(_cfg, "YOLO_MODEL_PATH", "yolo11m.pt")
    IMAGE_SIZE = getattr(_cfg, "IMAGE_SIZE", 1280)
    CONF = getattr(_cfg, "CONFIDENCE_THRESHOLD", 0.2)
except Exception:
    _cfg = None
    BASE_MODEL_PATH = "yolo11m.pt"
    IMAGE_SIZE = 1280
    CONF = 0.2


# ---------------------------------------------------------------------------
# CONFIG  (CLI flags override)
# ---------------------------------------------------------------------------

# Optional fine-tuned helmet model. If this path doesn't exist, we fall back
# to person-counting on the base model. Expected classes if you supply one:
#   class names containing "helmet" (compliant) and "no_helmet"/"nohelmet"/
#   "head" (violation). Adjust HELMET_CLASS_NAMES / NOHELMET_CLASS_NAMES below.
HELMET_MODEL_PATH = "weights/helmet_yolo.pt"
HELMET_CLASS_NAMES = {"helmet", "with_helmet", "helmet_on"}
NOHELMET_CLASS_NAMES = {"no_helmet", "nohelmet", "without_helmet", "head", "bare_head"}

# A motorcycle must be tracked at least this many frames before we judge it
# (matches the README's triple-riding debounce window).
TRIPLE_RIDING_MIN_FRAMES = 15

# Persons-on-one-motorcycle count that constitutes triple riding.
TRIPLE_RIDING_PERSON_COUNT = 3

# How far to expand the motorcycle box (fraction of w/h) so riders' heads /
# the third passenger are inside the crop. Riders sit ABOVE the bike, so we
# expand upward most.
CROP_EXPAND_TOP = 1.2     # 120% of box height added above
CROP_EXPAND_SIDE = 0.35   # 35% of box width added each side
CROP_EXPAND_BOTTOM = 0.15

# A violation is only emitted once it holds for this many of the sampled
# motorcycle frames (debounce against single-frame flukes).
VIOLATION_PERSIST_FRAMES = 8

# To save compute you can sample every Nth frame that contains a motorcycle.
# 1 = judge every motorcycle frame; 3 = every 3rd. Persistence counts are in
# SAMPLED frames, so keep PERSIST modest if you raise this.
FRAME_SAMPLE_STRIDE = 2

# Person/helmet confidence inside the crop.
RIDER_CONF = 0.25

# COCO ids for the fallback heuristic.
COCO_PERSON = 0
COCO_MOTORCYCLE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_motorcycle_frames(csv_path):
    """From the CSV, build: {frame_index: [ (tracker_id, x1,y1,x2,y2,
    frames_since_first_seen), ... ]} for motorcycles only. This is what makes
    the layer cheap -- we know exactly which frames and which boxes to crop."""
    df = pd.read_csv(csv_path)
    moto = df[df["vehicle_class"] == "motorcycle"].copy()
    by_frame = defaultdict(list)
    for _, r in moto.iterrows():
        by_frame[int(r["frame_index"])].append((
            int(r["tracker_id"]),
            float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
            int(r.get("frames_since_first_seen", 0)),
        ))
    fps = df["timestamp_sec"].max() and df["frame_index"].max() / df["timestamp_sec"].max() or 30.0
    return by_frame, round(fps, 3), int(df["frame_index"].max())


def expand_crop(x1, y1, x2, y2, W, H):
    w, h = x2 - x1, y2 - y1
    nx1 = max(0, int(x1 - CROP_EXPAND_SIDE * w))
    nx2 = min(W, int(x2 + CROP_EXPAND_SIDE * w))
    ny1 = max(0, int(y1 - CROP_EXPAND_TOP * h))
    ny2 = min(H, int(y2 + CROP_EXPAND_BOTTOM * h))
    return nx1, ny1, nx2, ny2


def load_helmet_model(helmet_path):
    """Return (model, mode). mode is 'helmet' if a real helmet model loaded,
    else 'fallback' meaning we person-count on the base model."""
    if helmet_path and os.path.exists(helmet_path):
        try:
            m = YOLO(helmet_path)
            print(f"[ok] helmet model loaded: {helmet_path}  (best-accuracy mode)")
            return m, "helmet"
        except Exception as e:
            print(f"[warn] could not load helmet model ({e}); falling back")
    print(f"[info] no helmet model at '{helmet_path}'. Falling back to "
          f"person-count heuristic on {BASE_MODEL_PATH}.")
    return YOLO(BASE_MODEL_PATH), "fallback"


def classify_crop(model, mode, crop):
    """Return (n_persons, n_no_helmet) for one motorcycle crop.
    - helmet mode: counts heads with/without helmet from the dedicated model.
    - fallback mode: counts COCO persons; no_helmet is unknown -> returns -1."""
    res = model.predict(crop, imgsz=640, conf=RIDER_CONF, verbose=False)[0]
    names = res.names
    n_persons = 0
    n_no_helmet = 0
    if res.boxes is None or len(res.boxes) == 0:
        return 0, (0 if mode == "helmet" else -1)
    for b in res.boxes:
        cid = int(b.cls[0])
        cname = str(names.get(cid, cid)).lower()
        if mode == "helmet":
            if cname in HELMET_CLASS_NAMES or cname in NOHELMET_CLASS_NAMES:
                n_persons += 1
                if cname in NOHELMET_CLASS_NAMES:
                    n_no_helmet += 1
        else:
            if cid == COCO_PERSON:
                n_persons += 1
    if mode == "fallback":
        n_no_helmet = -1  # unknown without a helmet model
    return n_persons, n_no_helmet


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def run(video_path, csv_path, helmet_path, out_dir):
    by_frame, fps, max_frame = load_motorcycle_frames(csv_path)
    if not by_frame:
        print("[info] no motorcycles in CSV -> no rider-safety checks to run.")
        return

    model, mode = load_helmet_model(helmet_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # per-motorcycle rolling evidence for debounce
    persons_window = defaultdict(lambda: deque(maxlen=VIOLATION_PERSIST_FRAMES))
    nohelmet_window = defaultdict(lambda: deque(maxlen=VIOLATION_PERSIST_FRAMES))
    flagged_triple = set()
    flagged_helmet = set()
    events = []

    frame_idx = -1
    target_frames = set(by_frame.keys())
    print(f"[run] scanning {len(target_frames)} motorcycle-bearing frames "
          f"(of {max_frame+1}) in {mode} mode...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx not in target_frames:
            continue
        if frame_idx % FRAME_SAMPLE_STRIDE != 0:
            continue

        for (tid, x1, y1, x2, y2, seen) in by_frame[frame_idx]:
            if seen < TRIPLE_RIDING_MIN_FRAMES:
                continue  # debounce: tracker hasn't held this bike long enough
            cx1, cy1, cx2, cy2 = expand_crop(x1, y1, x2, y2, W, H)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            n_persons, n_no_helmet = classify_crop(model, mode, crop)

            persons_window[tid].append(n_persons)
            if mode == "helmet":
                nohelmet_window[tid].append(max(0, n_no_helmet))

            ts = round(frame_idx / fps, 2)

            # --- triple riding ---
            if (tid not in flagged_triple
                    and len(persons_window[tid]) == VIOLATION_PERSIST_FRAMES
                    and min(persons_window[tid]) >= TRIPLE_RIDING_PERSON_COUNT):
                flagged_triple.add(tid)
                events.append({
                    "tracker_id": tid, "violation": "triple_riding",
                    "frame_index": frame_idx, "timestamp_sec": ts,
                    "persons_on_bike": int(min(persons_window[tid])),
                    "detail": f">= {TRIPLE_RIDING_PERSON_COUNT} riders for "
                              f"{VIOLATION_PERSIST_FRAMES} sampled frames",
                })

            # --- helmet (only meaningful in helmet mode) ---
            if (mode == "helmet" and tid not in flagged_helmet
                    and len(nohelmet_window[tid]) == VIOLATION_PERSIST_FRAMES
                    and min(nohelmet_window[tid]) >= 1):
                flagged_helmet.add(tid)
                events.append({
                    "tracker_id": tid, "violation": "helmet_noncompliance",
                    "frame_index": frame_idx, "timestamp_sec": ts,
                    "persons_on_bike": int(persons_window[tid][-1]),
                    "detail": f"bare head present for {VIOLATION_PERSIST_FRAMES} "
                              f"sampled frames",
                })

    cap.release()

    # -------- output --------
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"rider_safety_violations_{stamp}.csv")
    ev_df = pd.DataFrame(events)
    if len(ev_df):
        ev_df = ev_df.sort_values(["timestamp_sec", "tracker_id"])
        ev_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 64)
    print("RIDER-SAFETY LAYER RESULT")
    print("=" * 64)
    print(f"  mode                : {mode}")
    print(f"  motorcycles judged  : {len(persons_window)}")
    print(f"  triple-riding flags : {len(flagged_triple)}")
    if mode == "helmet":
        print(f"  helmet flags        : {len(flagged_helmet)}")
    else:
        print(f"  helmet flags        : n/a (supply --helmet-model for this)")
    if len(ev_df):
        print("\n" + ev_df.to_string(index=False))
        print(f"\n[ok] written: {out_csv}")
    else:
        print("\n  no violations met the persistence threshold.")
    print()


def main():
    ap = argparse.ArgumentParser(description="Second-layer helmet + triple-riding detector (crop-targeted, low-noise).")
    ap.add_argument("--video", required=True, help="Source video (same one the CSV was made from)")
    ap.add_argument("--csv", required=True, help="trajectory_data_*.csv from the collector")
    ap.add_argument("--helmet-model", default=HELMET_MODEL_PATH,
                    help="Fine-tuned helmet .pt for best accuracy (optional)")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()
    run(args.video, args.csv, args.helmet_model, args.out_dir)


if __name__ == "__main__":
    main()