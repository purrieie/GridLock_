"""
trajectory_collector.py
------------------------
FOUNDATION MODULE: Vehicle Trajectory Data Collection

PURPOSE
-------
This is the base data-collection layer for the automated traffic violation
system. It does exactly one job: take a traffic video as
input, detect every vehicle/person in every frame using YOLO, assign each
detection a PERSISTENT identity across frames using BoT-SORT, and record
the full position history of every tracked object into a structured CSV.

This CSV is the "ground truth" data source that every later violation
module (red-light, triple-riding, wrong-side driving, speed estimation,
risk scoring, etc.) will read from. None of those modules re-run detection
or tracking themselves -- they all consume this CSV. This keeps the
expensive GPU work (detection + tracking) decoupled from the cheap logic
work (violation rule-checking), which is what makes the overall system
scalable.

OUTPUTS
-------
1. A CSV file with one row per (vehicle, frame) observation. Schema below.
2. A separate annotated video file with bounding boxes + tracker IDs drawn,
   for visual sanity-checking. This is NOT used by downstream modules --
   it exists purely so a human can confirm tracking quality looks correct.

CSV SCHEMA (one row = one detection in one frame)
---------------------------------------------------
| Column          | Type  | Description                                          |
|-----------------|-------|------------------------------------------------------|
| tracker_id       | int   | Persistent unique ID assigned by BoT-SORT. Stays the  |
|                  |       | same for a given vehicle across its entire time in    |
|                  |       | frame, even through brief occlusion.                  |
| frame_index      | int   | Zero-based index of the video frame this row is from. |
| timestamp_sec    | float | Time in seconds from video start, computed as         |
|                  |       | frame_index / video_fps. This is the SOURCE OF TRUTH  |
|                  |       | for "when" -- accurate regardless of processing speed,|
|                  |       | because it's derived from frame position and the      |
|                  |       | video's own FPS metadata, not wall-clock processing   |
|                  |       | time.                                                  |
| vehicle_class    | str   | Human-readable class name (e.g. "car", "motorcycle",  |
|                  |       | "person"), mapped from the YOLO class ID.             |
| class_id         | int   | Raw YOLO class ID (kept alongside vehicle_class so     |
|                  |       | downstream code can filter numerically if needed).    |
| confidence       | float | YOLO's detection confidence score for this box,        |
|                  |       | 0.0-1.0. Lets later modules discard low-confidence    |
|                  |       | rows if needed without re-running the model.          |
| x1, y1, x2, y2   | float | Bounding box top-left and bottom-right pixel           |
|                  |       | coordinates, in the ORIGINAL video's pixel space.      |
| bottom_center_x  | float | x-coordinate of the box's bottom-center point.         |
| bottom_center_y  | float | y-coordinate of the box's bottom-center point. This    |
|                  |       | point (where the vehicle's tires meet the road) is     |
|                  |       | the standard reference point used for ROI/zone checks  |
|                  |       | (stop-line crossing, lane occupancy, etc.) in the      |
|                  |       | violation modules you'll build next -- it's more       |
|                  |       | accurate for "is this vehicle in this zone" logic than |
|                  |       | the box centroid, since centroid drifts upward for     |
|                  |       | tall vehicles (trucks/buses).                          |
| box_width        | float | Width of the bounding box in pixels. Useful later for  |
|                  |       | scale-based distance/speed estimation, since apparent  |
|                  |       | vehicle size changes with distance from camera.        |
| box_height       | float | Height of the bounding box in pixels. Same use case    |
|                  |       | as box_width.                                          |

DESIGN NOTE ON SMOOTHING / DOWNSAMPLING
----------------------------------------
This module intentionally writes ONE ROW PER FRAME PER DETECTION, with raw
(unsmoothed) coordinates. This is deliberate, not an oversight: smoothing
or downsampling here would throw away information you might need later and
can't get back. Trajectory smoothing (e.g. moving average, Kalman) and
downsampling (e.g. keep every 3rd point) are CONSUMER-SIDE concerns -- they
belong in the analysis module that reads this CSV, not in the collection
module that produces it. Keep this module "dumb and complete"; keep
intelligence in the modules built on top of it.

HOW TO RUN
----------
    python trajectory_collector.py --video path/to/input_video.mp4

All model/threshold settings are NOT in this file -- see config.py.
"""

import argparse
import csv
import os
from datetime import datetime
from collections import defaultdict

import cv2
from ultralytics import YOLO

import config


class OneEuroFilter:
    """
    One-Euro Filter: A simple first-order low-pass filter with adaptive
    smoothing based on velocity. Reduces jitter while preserving sharp
    movements. See: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        Initialize the One-Euro Filter.

        Args:
            min_cutoff (float): Minimum cutoff frequency in Hz. Lower values
                create more smoothing.
            beta (float): Velocity-dependent cutoff multiplier. Higher values
                make the filter more responsive to fast movements.
            d_cutoff (float): Derivative cutoff frequency in Hz. Reduces
                noise in velocity estimation.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.filtered_value = None
        self.filtered_velocity = 0.0
        self.last_time = None

    def _smoothing_factor(self, cutoff, delta_time):
        """
        Compute the alpha smoothing factor for a given cutoff frequency
        and time step.
        """
        if delta_time <= 0:
            return 1.0
        tau = 1.0 / (2 * 3.14159265359 * cutoff)
        return 1.0 / (1.0 + tau / delta_time)

    def filter(self, raw_value, timestamp):
        """
        Apply One-Euro filtering to a scalar value.

        Args:
            raw_value (float): The raw unfiltered measurement.
            timestamp (float): Current timestamp (seconds). Used to compute
                the time delta since the last update.

        Returns:
            float: The smoothed/filtered value.
        """
        if self.filtered_value is None:
            self.filtered_value = raw_value
            self.filtered_velocity = 0.0
            self.last_time = timestamp
            return raw_value

        delta_time = timestamp - self.last_time
        if delta_time <= 0:
            return self.filtered_value

        # Estimate raw velocity
        raw_velocity = (raw_value - self.filtered_value) / delta_time

        # Smooth the velocity estimate using a low-pass filter with d_cutoff
        alpha_velocity = self._smoothing_factor(self.d_cutoff, delta_time)
        self.filtered_velocity = (
            alpha_velocity * raw_velocity
            + (1.0 - alpha_velocity) * self.filtered_velocity
        )

        # Adaptively set cutoff based on velocity magnitude
        cutoff = self.min_cutoff + self.beta * abs(self.filtered_velocity)

        # Smooth the value using the adaptive cutoff frequency
        alpha = self._smoothing_factor(cutoff, delta_time)
        self.filtered_value = (
            alpha * raw_value + (1.0 - alpha) * self.filtered_value
        )

        self.last_time = timestamp
        return self.filtered_value


class EMAFilter:
    """
    EMA Filter: Exponential Moving Average - A straightforward, aggressive
    smoothing filter that removes jitter by averaging measurements with
    exponential decay weights. Simpler than momentum-based filtering with
    direct control over smoothing strength.
    """

    def __init__(self, alpha=0.85):
        """
        Initialize the EMA Filter.

        Args:
            alpha (float): Smoothing factor (0.0-1.0). Higher values give
                more weight to recent measurements and historical average,
                resulting in stronger smoothing.
                - 0.95: Very aggressive smoothing
                - 0.85: Recommended for strong jitter removal
                - 0.7: Moderate smoothing
                - 0.5: Light smoothing
        """
        self.alpha = alpha
        self.filtered_value = None

    def filter(self, raw_value):
        """
        Apply exponential moving average smoothing to a scalar value.

        Args:
            raw_value (float): The raw unfiltered measurement.

        Returns:
            float: The smoothed/filtered value.
        """
        if self.filtered_value is None:
            self.filtered_value = raw_value
            return raw_value

        # EMA formula: smoothed = alpha * raw + (1 - alpha) * previous_smoothed
        self.filtered_value = (
            self.alpha * raw_value
            + (1.0 - self.alpha) * self.filtered_value
        )

        return self.filtered_value


def ensure_output_dir(path):
    """Create the output directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def build_output_paths(output_dir):
    """
    Build timestamped output file paths so repeated runs never overwrite
    a previous run's results.

    Returns:
        tuple(csv_path, video_path)
    """
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        output_dir, f"{config.CSV_FILENAME_PREFIX}_{run_stamp}.csv"
    )
    video_path = os.path.join(
        output_dir, f"{config.VIDEO_FILENAME_PREFIX}_{run_stamp}.mp4"
    )
    return csv_path, video_path


def get_class_name(class_id):
    """
    Map a raw YOLO class ID to a human-readable name using the
    TARGET_CLASSES dict in config.py. Falls back to the raw ID (as a
    string) if the model returns a class we haven't named, so the
    pipeline never crashes on an unexpected class.
    """
    return config.TARGET_CLASSES.get(class_id, f"unknown_class_{class_id}")


def process_video(video_path, output_dir=None):
    """
    Run YOLO + BoT-SORT over the input video frame-by-frame, write every
    detection's data to a CSV, and write an annotated copy of the video
    for visual inspection.

    Args:
        video_path (str): Path to the input traffic video file.
        output_dir (str, optional): Override for config.OUTPUT_DIR.

    Returns:
        str: Path to the generated CSV file.
    """
    output_dir = output_dir or config.OUTPUT_DIR
    ensure_output_dir(output_dir)
    csv_path, annotated_video_path = build_output_paths(output_dir)

    # ---- Load model -----------------------------------------------------
    model = YOLO(config.YOLO_MODEL_PATH)

    # ---- Read source video metadata --------------------------------------
    # We need the video's own FPS to compute accurate timestamps. We do NOT
    # use wall-clock time during processing, because processing speed
    # (frames-per-second the script can chew through) has nothing to do
    # with the actual playback timing of the footage. Using frame_index /
    # source_fps guarantees the timestamp column reflects the video's real
    # timeline regardless of how fast/slow this script runs on your machine.
    probe_capture = cv2.VideoCapture(video_path)
    if not probe_capture.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    source_fps = probe_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(probe_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(probe_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe_capture.release()

    if not source_fps or source_fps <= 0:
        # Some video containers report FPS as 0 due to bad metadata.
        # Fall back to a sane default rather than dividing by zero later,
        # but warn loudly since timestamps will be inaccurate in this case.
        print(
            "WARNING: Source video reported invalid FPS metadata. "
            "Falling back to 30.0 FPS for timestamp calculation. "
            "Timestamps may be inaccurate -- verify your video file."
        )
        source_fps = 30.0

    # ---- Set up annotated video writer ------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    video_writer = cv2.VideoWriter(
        annotated_video_path, fourcc, source_fps, (frame_width, frame_height)
    )

    # ---- Set up CSV writer --------------------------------------------------
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "tracker_id",
            "frame_index",
            "timestamp_sec",
            "vehicle_class",
            "class_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "bottom_center_x",
            "bottom_center_y",
            "box_width",
            "box_height",
        ]
    )

    # ---- Initialize One-Euro Filters per tracker_id ----------------------
    # Each tracked object gets its own set of 4 filters (one per coordinate:
    # x1, y1, x2, y2) to smooth bounding box jitter independently.
    tracker_filters = defaultdict(
        lambda: {
            "x1": {
                "one_euro": OneEuroFilter(
                    config.ONE_EURO_MIN_CUTOFF,
                    config.ONE_EURO_BETA,
                    config.ONE_EURO_D_CUTOFF,
                ),
                "ema": EMAFilter(config.EMA_ALPHA),
            },
            "y1": {
                "one_euro": OneEuroFilter(
                    config.ONE_EURO_MIN_CUTOFF,
                    config.ONE_EURO_BETA,
                    config.ONE_EURO_D_CUTOFF,
                ),
                "ema": EMAFilter(config.EMA_ALPHA),
            },
            "x2": {
                "one_euro": OneEuroFilter(
                    config.ONE_EURO_MIN_CUTOFF,
                    config.ONE_EURO_BETA,
                    config.ONE_EURO_D_CUTOFF,
                ),
                "ema": EMAFilter(config.EMA_ALPHA),
            },
            "y2": {
                "one_euro": OneEuroFilter(
                    config.ONE_EURO_MIN_CUTOFF,
                    config.ONE_EURO_BETA,
                    config.ONE_EURO_D_CUTOFF,
                ),
                "ema": EMAFilter(config.EMA_ALPHA),
            },
        }
    )

    frame_index = 0
    target_class_ids = list(config.TARGET_CLASSES.keys())

    print(f"Processing video: {video_path}")
    print(f"Source FPS: {source_fps:.2f} | Resolution: {frame_width}x{frame_height}")

    # ---- Main per-frame loop ------------------------------------------------
    # model.track() with persist=True and a BoT-SORT tracker config handles
    # detection (YOLO) and identity persistence (BoT-SORT) in one call.
    # stream=True makes this memory-efficient for long videos, since it
    # yields one frame's results at a time instead of loading everything
    # into memory at once.
    results_stream = model.track(
        source=video_path,
        tracker=config.TRACKER_CONFIG,
        conf=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        classes=target_class_ids,
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in results_stream:
        timestamp_sec = frame_index / source_fps
        annotated_frame = result.orig_img.copy()

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            # boxes.id is None on frames where the tracker hasn't yet
            # confirmed any tracks (e.g. very first frame) -- guard against
            # that so we don't crash on an empty/uninitialized frame.
            tracker_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            xyxy_coords = boxes.xyxy.cpu().tolist()

            for tracker_id, class_id, confidence, (x1, y1, x2, y2) in zip(
                tracker_ids, class_ids, confidences, xyxy_coords
            ):
                vehicle_class = get_class_name(class_id)

                # ---- Apply One-Euro Filter for jitter removal ----
                if config.ENABLE_ONE_EURO_FILTER:
                    filters = tracker_filters[tracker_id]
                    x1 = filters["x1"]["one_euro"].filter(x1, timestamp_sec)
                    y1 = filters["y1"]["one_euro"].filter(y1, timestamp_sec)
                    x2 = filters["x2"]["one_euro"].filter(x2, timestamp_sec)
                    y2 = filters["y2"]["one_euro"].filter(y2, timestamp_sec)

                # ---- Apply EMA Filter for additional smoothing ----
                if config.ENABLE_EMA_FILTER:
                    filters = tracker_filters[tracker_id]
                    x1 = filters["x1"]["ema"].filter(x1)
                    y1 = filters["y1"]["ema"].filter(y1)
                    x2 = filters["x2"]["ema"].filter(x2)
                    y2 = filters["y2"]["ema"].filter(y2)

                bottom_center_x = (x1 + x2) / 2.0
                bottom_center_y = y2  # bottom edge of the box
                box_width = x2 - x1
                box_height = y2 - y1

                csv_writer.writerow(
                    [
                        tracker_id,
                        frame_index,
                        round(timestamp_sec, 4),
                        vehicle_class,
                        class_id,
                        round(confidence, 4),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                        round(bottom_center_x, 2),
                        round(bottom_center_y, 2),
                        round(box_width, 2),
                        round(box_height, 2),
                    ]
                )

                # ---- Draw annotation for visual sanity-check video ----
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    2,
                )
                label = f"ID:{tracker_id} {vehicle_class} {confidence:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), max(int(y1) - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                # Mark the bottom-center reference point used by ROI logic
                # in later modules -- visualizing it now helps you confirm
                # it lines up correctly with where tires meet the road.
                cv2.circle(
                    annotated_frame,
                    (int(bottom_center_x), int(bottom_center_y)),
                    4,
                    (0, 0, 255),
                    -1,
                )

        video_writer.write(annotated_frame)
        frame_index += 1

        if frame_index % 100 == 0:
            print(f"  Processed {frame_index} frames...")

    # ---- Cleanup ------------------------------------------------------------
    csv_file.close()
    video_writer.release()

    print(f"\nDone. Processed {frame_index} total frames.")
    print(f"CSV written to:   {csv_path}")
    print(f"Video written to: {annotated_video_path}")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle Trajectory Data Collection (YOLO + BoT-SORT)"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input traffic video file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory (default: {config.OUTPUT_DIR}, set in config.py)",
    )
    args = parser.parse_args()

    process_video(video_path=args.video, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
