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

# ---------------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------------

# Path to the YOLO weights file used for vehicle / person detection.
# Use a pretrained checkpoint (e.g. "yolo11m.pt") to auto-download from
# Ultralytics, or point this to your own fine-tuned weights file.
YOLO_MODEL_PATH = "yolo11m.pt"

# Minimum detection confidence. Detections below this are discarded by
# YOLO/the tracker before they ever reach our code.
CONFIDENCE_THRESHOLD = 0.25

# Intersection-over-Union threshold used by YOLO's internal NMS
# (Non-Max Suppression) to merge overlapping duplicate boxes.
IOU_THRESHOLD = 0.3

# Which YOLO class IDs we care about for this project.
# These indices match the default COCO dataset class list that pretrained
# YOLOv8 models ship with. Update this dict if you fine-tune on a custom
# dataset with different class indices.
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ---------------------------------------------------------------------------
# TRACKER SETTINGS (BoT-SORT)
# ---------------------------------------------------------------------------

# Ultralytics ships BoT-SORT as a YAML tracker config. "botsort.yaml" is the
# built-in default. If you later want to tune BoT-SORT's internal
# parameters (e.g. re-identification thresholds), copy that YAML into this
# project folder, edit it, and point this variable to your local copy.
TRACKER_CONFIG = " bytetrack.yaml" # botsort.yaml ,bytetrack.yaml is prefered

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

# ---------------------------------------------------------------------------
# ONE-EURO FILTER SETTINGS (Jitter Removal)
# ---------------------------------------------------------------------------

# Enable One-Euro Filter to smooth bounding box coordinates and remove jitter
# from both the CSV data and the annotated video.
ENABLE_ONE_EURO_FILTER = False

# Minimum cutoff frequency for the One-Euro Filter (Hz).
# Lower values = more smoothing. Default: 1.0 Hz. Use smaller values for more smoothing.
ONE_EURO_MIN_CUTOFF = 0.5

# Velocity-dependent cutoff multiplier. Controls how much the filter
# responds to fast movements. Higher = more responsive to velocity changes.
# Default: 0.007. Increase for less smoothing on fast movements.
ONE_EURO_BETA = 0.05

# Derivative cutoff frequency (Hz). Reduces high-frequency noise in velocity.
# Default: 1.0 Hz. Lower = more velocity smoothing.
ONE_EURO_D_CUTOFF = 1.0

# ---------------------------------------------------------------------------
# EMA FILTER SETTINGS (Exponential Moving Average - Jitter Smoothing)
# ---------------------------------------------------------------------------

# Enable EMA Filter for additional trajectory smoothing.
# Works alongside One-Euro Filter for aggressive jitter removal.
ENABLE_EMA_FILTER = True

# EMA smoothing factor (0.0 to 1.0). Controls how much weight is given to
# new measurements vs. historical average.
# lower values (0.8-0.95) = more smoothing, more lag.
# higher values (0.3-0.5) = more responsive to changes.
# Recommended: 0.85 for significant jitter removal. Default: 0.85
EMA_ALPHA = 0.85

# ---------------------------------------------------------------------------
# RESERVED FOR FUTURE MODULES
# ---------------------------------------------------------------------------
# Add settings for red-light detection, triple-riding detection, wrong-side
# detection, ROI polygons, etc. here as those modules are built. Keeping
# them in this same file means every module reads from one shared source
# of truth instead of duplicating constants.
