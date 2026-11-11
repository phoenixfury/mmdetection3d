# DetZero — one-folder sensor-clock pipeline

**Integrated source package**: recorder extraction → CenterPoint/DetZero → improved annotations → six class-colored videos → native-camera export → combined Fisheye3DOD + ROSA dataset. Works standalone or unpacked at `mmdetection3d/tools/detzero_temporal_pipeline/`.

## Folder layout

```text
detzero_temporal_pipeline_complete/
├── integration_scripts/
│   ├── pipeline.py                             # single entry point / preflight
│   ├── run_detzero_sequence.py                # CenterPoint + DetZero
│   ├── improve_detzero_annotations.py         # YOLO, class repair, camera hold
│   ├── render_improved_detzero.py              # 4 fisheye MP4s + BEV + mosaic
│   ├── render_combined_projection_audit.py     # one random 10%-per-BAG audit MP4
│   ├── temporal_annotations.py                # timestamps, poses, provenance
│   ├── camera_motion.py                       # camera-time ego compensation
│   ├── build_combined_dataset.py              # canonical fresh dataset builder
│   ├── append_improved_detzero_rosa_dataset.py # in-place incremental import
│   ├── build_chronological_combined_dataset.py # NEW dataset preserving old split
│   └── build_improved_detzero_rosa_dataset.py # shared dataset helper functions
├── recorder_pipeline/                        # raw decoder + sync + export
├── DetZero/                                  # included tracking/refining sources
├── external/                                 # add the OCam preparer here
│   └── prepare_fisheye3dod_inference.py      # NOT SUPPLIED
├── assets/
│   ├── calibration/
│   │   └── calib_results.txt                 # NOT SUPPLIED: OCam 800x800 model
│   ├── centerpoint_nus_voxel0075_dcn.pth     # NOT SUPPLIED
│   ├── yolo11m-seg.pt                        # NOT SUPPLIED
│   └── checkpoints_det3zero/                 # OPTIONAL if enabling PRM
├── runs/                                     # default BAG conversion/outputs
├── datasets/                                 # combined datasets
└── tests/
```

This bundle includes the requested vehicle-rig calibration at
`assets/calibration/calibration.json`. It does **not** include user BAG data or
pretrained weights. For `mmdetection3d` CenterPoint inference, the external
repository and its config remain required.

## 1. Install / prepare

From your working **mmdetection3d Python environment** (the one with CUDA, PyTorch, mmdet3d, MMEngine, YOLO Ultralytics, OpenCV, SciPy, scikit-learn, and OCamCamera), check imports:

```bash
python -c 'import cv2,numpy,torch,mmengine,ultralytics,sklearn; from ocamcamera import OcamCamera; import mmdet3d; print("Python deps OK")'
```

Keep `ocamcamera.py` importable (for example from the original Fisheye3DOD code). **Do not blindly install the historical `DetZero/requirements.txt`** on a modern Python environment: its NumPy pin is outdated and conflicts with newer mm3d stacks. The build and runtime requirements are documented in this file. For compressed PCL LiDAR, build from the active environment:

```bash
conda run --no-capture-output -n openmmlab env CC=/usr/bin/cc CXX=/usr/bin/c++ \
  cmake --fresh -S recorder_pipeline -B recorder_pipeline/build \
  -DCMAKE_BUILD_TYPE=Release -DRECORDER_PROTOBUF_PROVIDER=auto
conda run --no-capture-output -n openmmlab \
  cmake --build recorder_pipeline/build --parallel 4
conda run --no-capture-output -n openmmlab \
  ctest --test-dir recorder_pipeline/build --output-on-failure
```

Requires a C++ toolchain, PCL and Protobuf headers/libraries, along with FFmpeg for the camera decoder. If raw `.bin` clouds are available, select `--extract-args '--lidar-encoding raw'` instead.

`auto` prefers a coherent Protobuf CMake package in the active Conda
environment. Use `system` only when the system `protoc`, headers, and runtime
have matching versions. `--fresh` clears stale compiler/toolchain selections.
The CMake project supports both the native compiler and Conda's compiler
wrapper; when PCL is installed system-wide it explicitly supplies the host
Boost include path and OpenNI library directory required by that wrapper.

**Physically validate** `T_BASELINK_CLIDAR`, all `lidar_to_camera` transforms, native KB4 intrinsics, clock domain and image orientation for each rig. Dataset images preserve their original resolution and pixels. `calib_results.txt` is only needed when explicitly requesting the legacy `--image-mode canonical-ocam`; do not substitute it for the native recorder camera calibration.

The camera preparer must be supplied as `external/prepare_fisheye3dod_inference.py` or found at `MMDET_ROOT/tools/custom_data/prepare_fisheye3dod_inference.py`. It creates `fisheye3dod_inference_2hz/preparation.json`, native images and `fisheye3dod_infos_test.pkl`. It is **not** part of the submitted files and is deliberately not replaced with guessed geometry.

## 2. Preview the planned workflow and missing files

```bash
export MMDET3D_ROOT=/path/to/mmdetection3d
export BAG=/path/to/BAG_2026_09_30_10_10_54
python integration_scripts/pipeline.py --bag "$BAG" \
  --sequence-name 30_09 \
  --mmdet3d-root "$MMDET3D_ROOT" \
  --combined-output "$PWD/datasets/combined_improved_detzero_rosa" \
  --check
```

This prints selected assets, missing inputs, CUDA availability, and exact
commands. For commands without executing, replace `--check` with `--dry-run`.

Defaults in this package:

```text
rig calibration   assets/calibration/calibration.json
CenterPoint       MMDET3D_ROOT/checkpoints/centerpoint_nus_voxel0075_dcn.pth
YOLO              MMDET3D_ROOT/checkpoints/yolo/*.pt (exactly one .pt)
DetZero refiners  MMDET3D_ROOT/checkpoints_det3zero (optional)
OCam model        MMDET3D_ROOT/data/Fisheye3DODdataset/calib_results.txt
OCam preparer     MMDET3D_ROOT/tools/custom_data/prepare_fisheye3dod_inference.py
```

The vendored calibration is an exact copy of the requested September 16 file
(SHA-256 `28e2d6a39169e6bb44f6af2a8b410b204526270fbed5e87655728b9953e383d7`).
Use `--calibration` for a different vehicle or sensor mount.

## 3. Run the whole pipeline

```bash
python integration_scripts/pipeline.py --bag "$BAG" \
  --sequence-name 30_09 \
  --mmdet3d-root "$MMDET3D_ROOT" \
  --combined-output "$PWD/datasets/combined_improved_detzero_rosa"
```

Validated-bag command (choose `create` for a new combined root or `append` for
an existing compatible root):

```bash
cd /home/mostafa/mmdetection3d/tools/detzero_temporal_pipeline_complete
conda run --no-capture-output -n openmmlab python integration_scripts/pipeline.py \
  --bag /home/mostafa/mmdetection3d/data/Custom/BAG_2026_11_06_10_07_06 \
  --sequence-name 06_11 --mmdet3d-root /home/mostafa/mmdetection3d \
  --combined-output /path/to/combined_improved_detzero_rosa \
  --dataset-mode append
```

All paths are explicit and the entry point works from any working directory. Unless overridden, it uses:

1. **Extract**: raw camera streams to `runs/30_09/converted/`, synchronized `dataset/manifest.csv`, native JPEG images, float32 XYZI PCD, odometry and `odometry_poses.npy`. The default `--camera-sync-clock auto` compares LiDAR acquisition and recorder/read clocks. A sustained set of exact acquisition-time camera matches selects the sensor clock even when a recorder backlog creates more read-time matches. Old queued LiDAR history outside the actual camera interval is discarded; it is never shifted forward to fill the video. Read time and fixed-offset alignment are only fallbacks for a genuinely incomparable device clock. PCL LiDAR chunks decode in parallel (`--lidar-workers 4` by default); lower it to `1` on a constrained machine or raise it when CPU/storage bandwidth is available. Both original timestamps and the clock decision remain in the manifests.
2. **DetZero**: `dataset/detzero_output/results.json`, tracker outputs and temporal sidecars. The Waymo position refiner is **disabled by default**; supply `--enable-position-refiner` and real checkpoints to enable.
3. **Improve**: `dataset/detzero_output/improved_results.json`, improved temporal sidecars and report. YOLO class checking and camera-time compensation are enabled.
4. **Video**: `dataset/detzero_output/improved_videos/`, one MP4 per camera (`arc_f`, `arc_sl`, `arc_sr`, `arc_r`), `lidar_bev.mp4`, `all_cameras_plus_bev.mp4`, plus a JSON report. The **same BGR class mapping** is applied on camera image and LiDAR BEV. `[hold]` labels identify extrapolated camera-track-hold boxes.
5. **Prepare cameras**: native-resolution fisheye templates with each camera's calibrated KB4 matrix. `--sample-rate 0` preserves every already-synchronized sample. Pass `--prepare-args '--image-mode canonical-ocam'` only for a legacy model that explicitly requires the 800×800 OCam lens.
6. **Dataset**: new combined dataset `annotations/fisheye3dod_infos_{all,train,val}.pkl`, byte-identical ROSA/Fisheye3DOD views, shared images, 300 ROSA anchors and summary. Uses per-BAG acquisition-time 90/10 tail split with 5-second purge by default.
7. **Audit**: `combined_projection_audit_10pct.mp4` in the combined dataset root. It contains one deterministic random 10% sample from every BAG, grouped by BAG and chronological within each BAG, with four projected native-camera views plus LiDAR BEV. The exact frame indices and seed are recorded beside it in `combined_projection_audit_10pct.json`. Use `--audit-fraction` or `--audit-seed` to change sampling, or `--stop-after dataset` to skip this stage.

Use `--stop-after visualize` to run only steps 1–4, without any OCam conversion/dataset builder. Use `--start-at detzero`, `improve`, or `visualize` to resume from existing outputs. Pass overrides such as `--converted /data/BAG_converted`, `--dataset /data/BAG_converted/dataset`, `--calibration /data/calibration.json`, `--camera-sync-clock sensor`, `--fps 2`, `--video-limit 100`, `--video-stride 2`, or `--yolo-weights /weights/my.pt` as needed.

## 4. Adding more bags to the combined dataset

When the output dataset does not exist, `--dataset-mode create` (the default) creates it and never silently overwrites. For another BAG:

```bash
python integration_scripts/pipeline.py --bag /path/to/NEW_BAG \
  --sequence-name 17_09 --mmdet3d-root "$MMDET3D_ROOT" \
  --combined-output "$PWD/datasets/combined_improved_detzero_rosa" \
  --dataset-mode append
```

`append` changes the combined dataset in place and recomputes train/val outputs, including training-only anchors. Back up the existing combined dataset first if you need exact benchmark reproducibility. To **preserve an existing benchmark's train/val tokens**, prefer creating a **new** combined dataset:

```bash
python integration_scripts/pipeline.py --bag /path/to/NEW_BAG \
  --sequence-name 17_09 --mmdet3d-root "$MMDET3D_ROOT" \
  --dataset-mode extend \
  --base-dataset "$PWD/datasets/combined_improved_detzero_rosa" \
  --combined-output "$PWD/datasets/combined_plus_17_09"
```

This calls `build_chronological_combined_dataset.py`. The original benchmark's split remains fixed. The new BAG gets a chronological tail holdout, or pass explicit source-frame boundaries through `--dataset-args`. Ensure at least three synced frames and adequate train/val separation.

## 5. Visualize an already-improved BAG without rerunning inference

```bash
python integration_scripts/render_improved_detzero.py \
  /path/to/BAG_converted/dataset \
  --calibration /path/to/BAG_converted/calibration.json \
  --fps 2 --bev-range 60
```

To audit synchronization before running either detector, render the native
camera streams beside the raw LiDAR BEV with `--no-annotations`. This mode does
not read, filter, or draw any boxes:

```bash
python integration_scripts/render_improved_detzero.py \
  /path/to/BAG_converted/dataset \
  --calibration /path/to/calibration.json --no-annotations \
  --output /path/to/BAG_converted/dataset/sync_diagnostic_videos
```

The overlay transforms LiDAR-time 3D boxes to the selected camera packet time using pose SLERP where valid, and draws visible 3D box edges with the **native KB4** camera projection. Moving-object velocity compensation applies only when the improved object explicitly contains `velocity_world`; otherwise only ego motion is compensated. No LiDAR scan deskew or ground-truth exposure timestamps are claimed. On invalid/unbracketed camera times it falls back to the LiDAR-time projection, reported in `visualization_report.json`. All four cameras and BEV use the same 3D-box class colors.

## 6. Tests

```bash
python -m pytest -q tests/
```

Real-bag validation used `BAG_2026_11_06_10_07_06`. Its recorder/read order
contained 249 reversals relative to LiDAR acquisition time because delayed
clouds arrived from a growing queue. Auto synchronization now selects sensor
time (10,252 valid four-camera candidates versus 10,410 on read time), yielding
1,157 exported frames with strictly increasing acquisition timestamps and
camera deltas below 20 ms. CenterPoint, DetZero, YOLO post-processing, all six
videos, the OCam inference PKL, and the combined train/validation dataset were
regenerated successfully from that corrected timeline.
