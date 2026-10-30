> For current end-to-end commands, including combined-dataset updates and
> model training, see **README_START_HERE.md**. This file describes the
> annotation sidecar and compatibility details.

# DetZero temporal annotations: integration and compatibility

## Components

Four scripts belong in the **same directory**: `run_detzero_sequence.py`,
`improve_detzero_annotations.py`, `temporal_annotations.py`, and `camera_motion.py`. The runner now discovers a sibling `DetZero/` tree and an ancestor
MMDetection3D config checkout; `MMDET3D_ROOT` and `DETZERO_ROOT` may override.
For the complete bundled source tree, follow `README_START_HERE.md`.

The recorder, DetZero changes, and tests are included in this package. Do not
mix files from an older overlay archive with the current checkout.

## Run: new recording

1. Run your existing recorder `decode / synchronize / export` workflow. The
   upgraded export preserves `arc_{f,sl,sr,r}_timestamp_ns`, the synchronization
   target and clock in `manifest.csv`; no new recorder command-line flags needed.
2. Run DetZero from the location of your existing script:

```bash
python run_detzero_sequence.py /path/to/synchronized_dual_images \
  --stage all --calibration /path/to/calibration.json
```

3. Enhance with camera YOLO (using your own existing weights):

```bash
python improve_detzero_annotations.py /path/to/synchronized_dual_images \
  --weights /path/to/best.pt \
  --calibration /path/to/calibration.json
```

Output examples:

```
SEQUENCE/detzero_output/
  results.json                     # unchanged DetZero classic result JSON
  results_temporal.json            # new per-frame timestamps, poses, source
  results_fisheye3dod/
    frame_map.json
    label_3d/
      000000.json                  # classic labels: bounding_box/class/name/id
      000001.json
  tracking_results.json
  tracking_results_temporal.json
  tracking_results_fisheye3dod/   # generated when --stage track
  improved_results.json
  improved_results_temporal.json
  improved_results_fisheye3dod/
    label_3d/...
```

`--skip-classic-labels` skips the extra per-frame Fisheye3DOD-style JSON
files, but keeps sidecars. Stage `detect` exports
`detection_results_temporal.json` and stage `track` exports
`tracking_results_temporal.json`. Stage `refine` and `all` export
`results_temporal.json`; `render` never rewrites annotations.

## Re-export from existing DetZero JSON (no inference)

```bash
python temporal_annotations.py /path/to/synchronized_dual_images \
  --results /path/to/synchronized_dual_images/detzero_output/improved_results.json \
  --tracks /path/to/synchronized_dual_images/detzero_output/tracks.pkl \
  --stage camera_improved \
  --calibration /path/to/calibration.json
```

No GPU, detector or Ultralytics dependency is needed for this command.

**Warning:** For recordings exported before camera timestamps were retained,
the metadata can reconstruct camera packet time from
`camera_sync_target_ns + arc_*_delta_ns` *only if both are present*. Otherwise
the camera timestamp is `null` and marked `unavailable`. It never guesses an
exposure timestamp. Re-run the updated recorder `export` from the existing
synchronization.csv (with `--overwrite` if needed) to recover those columns.

## Temporal sidecar v1 schema

```json
{
  "schema": "detzero_temporal_v1",
  "coordinate_convention": "right-handed x-forward, y-left, z-up; center-origin boxes in CLIDAR",
  "calibration_path": ".../calibration.json",
  "stage": "camera_improved",
  "frames": [
    {
      "frame_id": 0,
      "acquisition_timestamp_ns": 1800000000,
      "recorder_timestamp_ns": 1800000900,
      "lidar_transport_delay_ns": 900,
      "timestamp_clock": "lidar_sensor",
      "ego_pose": {
        "odometry_index": 0,
        "reference_timestamp_ns": 1800000000,
        "alignment_delta_ns": 25000000,
        "alignment_status": "...",
        "T_world_BASELINK": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        "T_world_CLIDAR": [[1,0,0,2.5],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
      },
      "cameras": {
        "ARC_F": {
          "timestamp_ns": 1800000005,
          "timestamp_origin": "manifest",
          "clock": "camera_recorder_packet",
          "synchronization_clock": "sensor",
          "delta_to_sync_target_ns": 5,
          "image": "images/arc_f/1800000900.jpg"
        }
      },
      "labels": [
        {
          "object_index": 0,
          "track_id": 21,
          "class": "Car",
          "box_frame": "CLIDAR",
          "source": "camera_hold",
          "source_detail": "camera_track_hold",
          "camera": "ARC_F",
          "camera_packet_timestamp_ns": 1800000005,
          "camera_packet_minus_lidar_acquisition_ns": 5
        }
      ]
    }
  ]
}
```

The example omits other optional provenance and camera entries for brevity.
Actual sidecars contain all four cameras. Matrices are row-major 4x4.

`source` values: `lidar`, `camera_hold`, `tracker_prediction`, `unknown`.
`camera_hold` can itself carry `lidar_support_points` from LiDAR frustum
correction; the *annotation continuity* nevertheless comes from the camera.
A YOLO proposal with real LiDAR clustering is `lidar`, with
`source_detail: "yolo_lidar"`.

`timestamp_clock: "recorder_fallback"` means **no proven LiDAR acquisition
timestamp** was available. `camera_recorder_packet` is a transport/index
clock, *not* a verified image exposure clock. Source clocks must be compared
or calibrated before using the cross-sensor time differences as physical
exposure offsets.

The pose orientation/frame interpretation uses the existing recorder
assumption of `T_world_BASELINK`, and is not a substitute for verifying the
localization frame's true geographical convention. The CLIDAR transformation
is `T_world_CLIDAR = T_world_BASELINK @ T_BASELINK_CLIDAR`.

## Fisheye3DOD / ROSA compatibility

The DetZero `results.json` format remains a **list of frames**, with
`frame_id`, `timestamp_ns`, `lidar_sensor_timestamp_ns` and `objects` (with
`track_id`, `name`, `score`, `box`). No new required fields are added.

The per-frame Fisheye3DOD labels are a JSON list whose entries have exactly
`bounding_box`, `class`, `name`, `id`, as expected by
`tools/fisheye3dod_converter.py` from phoenixfury/Fisheye3DOD. The original
converter defines six classes. `ConstructionVehicle` is exported as `Van`
*only in the classic `class` field*, while `name` and the temporal sidecar
retain `ConstructionVehicle`. The classic `id` is the source track ID.

Camera postprocessing classifies each sufficiently observed vehicle track
from its median LiDAR length/width/height (at least three non-held frames).
It uses the configured Car, Van, small/rigid Truck, and Bus size profiles,
with tolerances for the observed CenterPoint box bias. The chosen class is
applied to every frame of that track. A rigid truck and bus can have the same
dimensions, so an existing LiDAR Truck/Bus vote is retained in that overlap;
size alone cannot distinguish their body styles. The report records size
relabels, overlapping profiles, and tracks skipped because their dimensions
jump too much to support a reliable size-based relabel.

Camera confirmation also checks projected-box scale. A tiny 2D detection
inside a much larger 3D projection cannot validate that vehicle or sustain a
camera-held prediction when the projection area exceeds the detection area
by more than 5× and their IoU is below 0.12. This prevents roadside small
cars from confirming phantom bus-sized LiDAR boxes; the applied thresholds
are recorded in `improved_results_report.json`.

The generated `label_3d/` folder is an **annotation-format adapter, not a
complete model-ready dataset**. Both Fisheye3DOD and ROSA still need their
image filenames, `ImageSets-2hz` scene/vehicle manifests, camera calibration
and sensor metadata / converted `.pkl` info files. ROSA reuses the
Fisheye3DOD data pipeline. Do not rename the LiDAR-frame boxes to BASELINK
without transforming them. ROSA's official camera path requires its own
OCam/ray calibration; a KB4 calibration is not an automatically interchangeable
ROSA model.

These temporal sidecars are provenance-rich labels, not an activation switch
for ROSA's temporal instance bank. The incremental combined-dataset importer
reads acquisition timestamps and LiDAR poses from the synchronized manifest
and odometry, writes them into each sample, and splits each BAG chronologically
with links confined to the split. ROSA's adapter now consumes these fields.
However the current ROSA instance bank still rejects
`num_temp_instances > 0`; temporal slots require model implementation before
motion-compensated temporal training can be claimed.

## Validation performed

```bash
PYTHONPATH=<patch>/integration_scripts pytest -q <patch>/tests
# 10 passed (including CPU-only DetZero import stubs)

PYTHONPATH=<recorder_pipeline>/src:<recorder_pipeline>/tests pytest -q <recorder_pipeline>/tests
# 35 passed
```

A full GPU run and physically meaningful calibration check were **not**
performed here, because the supplied uploads do not include a complete
recorded sequence or the required CenterPoint/YOLO/DetZero checkpoints.

## Important limitations not fixed in this patch

The updated tracking filters and velocity postprocessor use acquisition-time
intervals. Association gates remain configured for the camera-2Hz profile;
their distance thresholds have not been calibrated across every gap length.
Camera frames also lack verified exposure timestamps and per-image ego poses,
so true rolling-shutter and cross-camera temporal corrections are out of scope.
See `AUDIT.md` for a prioritized backlog.
