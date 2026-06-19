"""
config.py
----------
Centralized configuration for the Vehicle Trajectory Data Collection module.

WHY THIS FILE EXISTS SEPARATELY:
Every model path, threshold, and tracker setting lives here instead of being
hardcoded inside the main script. This means future modules (red-light
detector, triple-riding detector, wrong-side detector) can import this same
config file and stay in sync with whatever model/thresholds you're using,
without touching the core tracking logic.

When you add new violation modules later, add their settings here too,
in their own clearly marked section at the bottom.
"""

# Prefix for your new isolated black canvas trajectory video
BLACK_CANVAS_FILENAME_PREFIX = "isolated_trajectory_map"
# Track whether you want to render the black canvas video stream
SAVE_BLACK_CANVAS_VIDEO = True

# ---------------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------------

# Path to the YOLO weights file used for vehicle / person detection.
# Use a pretrained checkpoint (e.g. "yolo11m.pt") to auto-download from
# Ultralytics, or point this to your own fine-tuned weights file.
YOLO_MODEL_PATH = "yolo11m.pt"

# Minimum detection confidence. Detections below this are discarded by
# YOLO/the tracker before they ever reach our code.
CONFIDENCE_THRESHOLD = 0.2

# (Non-Max Suppression) to merge overlapping duplicate boxes.
IOU_THRESHOLD = 0.3 #A lower IOU_THRESHOLD means more aggressive rejection and discarding of overlapping bounding boxes

# Which YOLO class IDs we care about for this project.
# These indices match the default COCO dataset class list that pretrained
# YOLOv8 models ship with. Update this dict if you fine-tune on a custom
# dataset with different class indices.
TARGET_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
}

IMAGE_SIZE = 1280  # 640 YOLOv8 default image size. Change if you fine-tune on a different size.
# ---------------------------------------------------------------------------
# TRACKER SETTINGS (BoT-SORT)
# ---------------------------------------------------------------------------

# Ultralytics ships BoT-SORT as a YAML tracker config. "botsort.yaml" is the
# built-in default. If you later want to tune BoT-SORT's internal
# parameters (e.g. re-identification thresholds), copy that YAML into this
# project folder, edit it, and point this variable to your local copy.
TRACKER_CONFIG = "custom_botsort.yaml"  # botsort.yaml, bytetrack.yaml is preferred

# ---------------------------------------------------------------------------
# OUTPUT SETTINGS
# ---------------------------------------------------------------------------

# Folder where the trajectory CSV and annotated video get written.
OUTPUT_DIR = "output"

# CSV filename (timestamped automatically in the main script so repeated
# runs don't overwrite each other).
CSV_FILENAME_PREFIX = "trajectory_data"

# Annotated video filename prefix.
VIDEO_FILENAME_PREFIX = "annotated_output"

# Codec used to write the annotated video. "mp4v" is widely compatible.
VIDEO_CODEC = "mp4v"


# EMA alpha used specifically for velocity smoothing (vx, vy).
# This alpha is intentionally HIGHER than position EMA alpha because:
# velocity is a derivative (delta/dt), so it amplifies noise much more than
# raw position does. You want quick response to real speed changes but
# suppression of frame-to-frame noise spikes.
#
# Formula: smoothed = alpha * raw + (1-alpha) * smoothed_prev
# So: lower alpha = MORE smoothing but MORE lag (bad for braking detection)
#     higher alpha = LESS smoothing but LESS lag
#
# still show forward motion 2-3 seconds after a vehicle stopped.
# 0.25 is a reasonable starting point: filters single-frame spikes
# while still responding to real acceleration/braking within ~4 frames.
VELOCITY_EMA_ALPHA = 0.25

# ---------------------------------------------------------------------------
# ONE-EURO FILTER SETTINGS (Jitter Removal)
# ---------------------------------------------------------------------------

# Enable One-Euro Filter to smooth bounding box coordinates and remove jitter
# from both the CSV data and the annotated video.
ENABLE_ONE_EURO_FILTER = True

# Minimum cutoff frequency for the One-Euro Filter (Hz).
# Lower values = more smoothing. Default: 1.0 Hz. Use smaller values for more smoothing.
ONE_EURO_MIN_CUTOFF = 0.25

# Velocity-dependent cutoff multiplier. Controls how much the filter
# responds to fast movements. Higher = more responsive to velocity changes.
# Default: 0.007. Increase for less smoothing on fast movements.
ONE_EURO_BETA = 0.015

# Derivative cutoff frequency (Hz). Reduces high-frequency noise in velocity.
# Default: 1.0 Hz. Lower = more velocity smoothing.
ONE_EURO_D_CUTOFF = 1.0

# ---------------------------------------------------------------------------
# EMA FILTER SETTINGS True(Exponential Moving Average - Jitter Smoothing)
# ---------------------------------------------------------------------------

# Enable EMA Filter for additional trajectory smoothing.
# Works alongside One-Euro Filter for aggressive jitter removal.
ENABLE_EMA_FILTER = False

# EMA smoothing factor (0.0 to 1.0).
# Formula: smoothed = alpha * raw + (1-alpha) * smoothed_previous
# LOWER alpha = more smoothing, more lag (e.g. 0.05 = very heavy smoothing)
# HIGHER alpha = less smoothing, more responsive (e.g. 0.9 = almost raw)
# Recommended starting point: 0.3 (moderate smoothing, ~3-4 frame lag)
EMA_ALPHA = 0.01

# ---------------------------------------------------------------------------
# DERIVED COLUMN THRESHOLDS
# ---------------------------------------------------------------------------

# Speed (in pixels/sec) below which a vehicle is considered stationary.
# Used to set the is_stationary flag in the CSV. Tune based on your
# camera's pixel-to-real-world scale -- at a typical intersection camera
# distance, 8 px/sec is roughly equivalent to <1 km/h real-world speed.
STATIONARY_SPEED_THRESHOLD_PX_SEC = 8.0

# Number of consecutive frames a vehicle must be stationary before
# is_stationary is set to True. Prevents a momentarily stopped vehicle
# (e.g., slowing for a speed bump) from being flagged as parked.
# At 30fps: 30 frames = 1 second, 90 frames = 3 seconds.
STATIONARY_MIN_FRAMES = 30

# ---------------------------------------------------------------------------
# RESERVED FOR FUTURE MODULES
# ---------------------------------------------------------------------------
# Add settings for red-light detection, triple-riding detection, wrong-side
# detection, ROI polygons, etc. here as those modules are built. Keeping
# them in this same file means every module reads from one shared source
# of truth instead of duplicating constants.
