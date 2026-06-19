# Vehicle Trajectory Data Collection — Foundation Module

This is the base data-collection layer of the traffic violation detection
system. It does ONE job: turn a traffic video into a structured CSV
containing every detected vehicle's position history over time, plus an
annotated video for visual verification.

Every violation-detection module you build after this (red-light,
triple-riding, wrong-side driving, speed estimation, risk scoring) should
read FROM this CSV. None of them should re-run YOLO or the tracker
themselves. This keeps the expensive GPU work in one place and the cheap
rule-checking logic in another, which is what makes the system scalable.

## Files

| File                      | Purpose                                                          |
|---------------------------|-------------------------------------------------------------------|
| `config.py`                | All model paths, thresholds, and tracker settings. Edit this, not the main script, when tuning behavior. |
| `trajectory_collector.py`  | The main module. Run this to process a video.                    |
| `requirements.txt`         | Python dependencies.                                              |

## Setup

```bash
pip install -r requirements.txt
```

The first time you run this, `ultralytics` will auto-download the YOLO
weights file named in `config.py` (`YOLO_MODEL_PATH`). If you have your own
fine-tuned `.pt` file, just point `YOLO_MODEL_PATH` at it instead.

## Usage

```bash
python trajectory_collector.py --video path/to/your_video.mp4
```

Optional: override the output directory (default is `output/`, set in
`config.py`):

```bash
python trajectory_collector.py --video path/to/your_video.mp4 --output-dir my_results/
```

Each run produces two timestamped files so repeated runs never overwrite
each other:

```
output/trajectory_data_20260617_143200.csv
output/annotated_output_20260617_143200.mp4
```

## CSV Schema

One row = one detected object, in one frame.

| Column           | Type  | Meaning                                                                 |
|-------------------|-------|--------------------------------------------------------------------------|
| `tracker_id`        | int   | Persistent ID from BoT-SORT. Same vehicle = same ID across its time in frame, including through brief occlusion. |
| `frame_index`       | int   | Zero-based frame number.                                                  |
| `timestamp_sec`     | float | `frame_index / video_fps`. Accurate regardless of how fast the script runs, since it's derived from the video's own FPS metadata, not wall-clock processing time. |
| `vehicle_class`     | str   | e.g. `"car"`, `"motorcycle"`, `"person"`, `"bus"`, `"truck"`.              |
| `class_id`          | int   | Raw YOLO class ID (COCO indices by default — see `config.py` to remap if you fine-tune on a custom dataset). |
| `confidence`        | float | YOLO detection confidence, 0.0–1.0.                                       |
| `x1, y1, x2, y2`    | float | Bounding box corners in pixel space (top-left, bottom-right).             |

| `bottom_center_x/y` | float | Bottom-center point of the box (where tires meet road). **Use this point for ROI/zone-crossing checks** (stop-line, lane polygons) — it's more accurate than the box centroid for tall vehicles like trucks/buses, whose centroid sits noticeably above ground level. |

| `box_width/height`  | float | Box dimensions in pixels. Keep these — apparent vehicle size shrinks with distance from camera, so this is useful later for scale-based speed/   distance estimation.  

|velocity_x_px_sec,velocity_y_px_sex| float | velcotiy of each id in x and y coordinates directly being derived from bottom_center_x/y coordinates over time .
|speed_px_sec| float | calculated using velocity_x_px_sec and velocity_y_px_sex .

|heading_deg| float |  (degrees(atan2(vy, vx)) + 360) % 360
0° = moving right across the screen
90° = moving downward on screen
180° = moving left across the screen
270° = moving upward on screen

|is_stationary| bool |It's a 0 or 1 flag. 1 means this vehicle has been moving slower than STATIONARY_SPEED_THRESHOLD_PX_SEC for at least STATIONARY_MIN_FRAMES consecutive frames , usefull attribute for illegal parking traffic voilation .

|frames_sinze_first_scene| float | how many frames this tracker_id has been continuously visible since it first appeared , using in detecting ghost ,
Ghost detection — YOLO sometimes hallucinates a bounding box for 2–3 frames then it disappears. That's not a real vehicle, it's a detection artifact. Filter them out before any violation check:
pythonreal_vehicles = df[df['frames_since_first_seen'] >= 5]
Triple-riding debounce — your rule is "3 persons on motorcycle for 15 consecutive frames before flagging." You need to confirm the motorcycle itself has been tracked for at least 15 frames, otherwise you could flag a motorcycle that appeared in frame 1 and by frame 3 coincidentally had 3 person boxes near it. The check becomes:
pythonif overlap_condition and row['frames_since_first_seen'] >= 15:
    flag_triple_riding(tracker_id)

-----------------------------------------------------------------------------------------------------------------------------------------------

**Timestamps come from frame position, not wall-clock time.** `timestamp_sec`
is calculated as `frame_index / source_video_fps`, read directly from the
video file's own metadata. This is what makes it accurate — it reflects
the video's real internal timeline, not how fast your machine happened to
process it.

**The annotated video is for humans, not code.** It exists so you can
visually confirm tracking quality (no ID switching, boxes look correct,
bottom-center dot lines up with tires) before trusting the CSV for
violation logic. Don't build any downstream module to read the video file;
they should all read the CSV.

## Adding new columns later

You mentioned wanting this to be easy to extend. To add a new column (e.g.
lane ID, homography-corrected real-world coordinates):

1. Add the column name to the `csv_writer.writerow([...])` header line in
   `trajectory_collector.py`.
2. Compute the value inside the per-detection loop (where `bottom_center_x`
   etc. are computed) and add it to the corresponding data row in the same
   order.

Because both the header and data rows are explicit lists in one place, this
is a two-line change, not a refactor.

## Suggested next steps (per earlier discussion)

1. Run this on a sample video and watch the annotated output — confirm IDs
   stay stable and don't switch when vehicles briefly occlude each other.
2. Build a small analysis script that reads the CSV, groups by
   `tracker_id`, and reconstructs each vehicle's trajectory as a polyline.
   Apply smoothing/downsampling THERE.
3. Layer violation logic (red-light ROI check, direction-vector check for
   wrong-side driving, etc.) on top of that trajectory data.


Data processing flow 
'''

                                    |
                                    ▼
                        [Raw Bounding Box from YOLO]
                                    │
                     
                                    ▼
 
    [Layer 1: One-Euro Filter] ──► Smooths positions *just* enough to track cleanly
 
                                    │ (Keeps boxes responsive; zero trailing lag)
                     
                                    ▼
 
    [Calculate Velocity Math] ──► v = Δx / Δt (Amplifies sub-pixel noise)
 
                                    │
                     
                                    ▼
 
    [Layer 2: Velocity Filter] ──► Dedicated low-alpha EMA filter wipes out
 
                                    │ high-frequency derivative spikes directly
                     
                                    ▼
 
                        [Final CSV & Stable Vectors]
'''
