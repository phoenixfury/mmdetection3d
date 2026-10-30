# Custom recorder conversion and calibration checks

> The commands below document the legacy per-stage conversion tools. For the
> current sensor-clock BAG → DetZero → postprocessing → combined dataset →
> Fisheye3DOD/ROSA workflow, use
> [README_START_HERE.md](../detzero_temporal_pipeline/README_START_HERE.md).

These tools convert the four Fisheye3DOD surround cameras (`ARC_F`, `ARC_R`,
`ARC_SL`, `ARC_SR`) and the Ouster/CLIDAR stream in this recorder format.

The camera files have an `.h264` suffix but are MP4 containers. Their paired
`.idx` files contain one little-endian uint64 nanosecond timestamp per frame.
The LiDAR stream is a protobuf `PointCloud` whose data field uses PCL octree
compression. Decoded `.bin` files contain float32 `[x, y, z, intensity]`; the
compression contains geometry but no intensity, so the fourth value is zero.

Build and run a small conversion first:

```bash
make -C tools/custom_data

python tools/custom_data/convert_cameras.py \
  data/Custom/BAG_2026_09_30_10_10_54 \
  data/Custom/BAG_2026_09_30_10_10_54_converted \
  --limit 10

tools/custom_data/decode_lidar \
  data/Custom/BAG_2026_09_30_10_10_54 \
  data/Custom/BAG_2026_09_30_10_10_54_converted \
  --limit 10
```

Extract calibration, synchronize to LiDAR, and create projection overlays:

```bash
python tools/custom_data/extract_calibration.py \
  data/Custom/BAG_2026_09_30_10_10_54 \
  data/Custom/BAG_2026_09_30_10_10_54_converted/calibration.json

python tools/custom_data/synchronize.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted \
  --tolerance-ms 20

python tools/custom_data/project_lidar.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted \
  data/Custom/BAG_2026_09_30_10_10_54_converted/calibration.json \
  --samples 1
```

Remove `--limit` to convert the complete recording. Use `--sample-rate 2` with
`synchronize.py` if a 2 Hz inference set is desired.

Camera frames that FFmpeg cannot decode are skipped by default. The converter
uses each packet's original presentation timestamp, so a damaged frame does not
shift the index timestamps assigned to later frames. Pass
`--corrupt-policy error` to stop at the first chunk containing frame loss.

`GENERIC_CAMERA_MODEL` does not state its exact projection equation. The
validation default treats the four ARC coefficients as OpenCV's fisheye model.
The overlay is the integrity check for this assumption and for the extrinsic
direction. `project_lidar.py --projection pinhole` provides a comparison.

The bag's extrinsics message omits `CLIDAR`. By default, calibration extraction
uses the rounded CLIDAR-to-BASELINK matrix recovered from the predecessor
conversation's `static_transforms.json` result. Pass the original full-precision
matrix as 16 row-major values with `--lidar-to-base` when it is available.
Use `--lidar-height-offset 0.3156` to test an additional vertical correction
without altering the camera poses.

Optimize each camera's pitch independently over several synchronized samples:

```bash
python tools/custom_data/optimize_camera_pitch.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted \
  data/Custom/BAG_2026_09_30_10_10_54_converted/calibration.json \
  --samples 60 --min-pitch -3.6 --max-pitch 2.6 --step 0.2 \
  --output data/Custom/BAG_2026_09_30_10_10_54_converted/calibration_pitch.json
```

The optimizer applies pitch around each camera optical frame's local x axis. It
matches projected LiDAR depth discontinuities above the road to strong image
edges and removes the per-image-row edge-density bias. Use disjoint sample
blocks (`--start-sample`) to check that an optimum generalizes before accepting
it. A correction can also be applied explicitly while extracting calibration,
for example `--camera-pitch ARC_F=-0.5`; repeat the option per camera.

For manual six-degree-of-freedom adjustment, launch the interactive editor with
the unmodified initial calibration:

```bash
python tools/custom_data/manual_camera_extrinsics.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted \
  data/Custom/BAG_2026_09_30_10_10_54_converted/calibration.json \
  --camera ARC_F --sample-index 0 \
  --output data/Custom/BAG_2026_09_30_10_10_54_converted/calibration_manual.json
```

The window displays the initial and adjusted projections side by side. `dx`,
`dy`, and `dz` are BASELINK-frame translation corrections. Local `Rx`, `Ry`,
and `Rz` are optical-camera-frame rotations; `Rx` moves the projection in the
pitch direction. Adjustments are retained while switching cameras. Use several
sample indices before saving, then click **Save calibration**. The initial JSON
is never overwritten unless that exact path is explicitly selected in the save
dialog. The preview defaults to `--point-stride 2 --point-radius 2` so movement
is easier to see. The interface defaults to `--ui-scale 1.4 --font-size 16`;
increase `--font-size` directly on a high-DPI display. Numeric entry boxes next
to the sliders allow exact adjustments; press Enter after typing a value. Each
local rotation control covers -20 to +20 degrees.

Prepare annotation-free input for the pretrained Fisheye3DOD BEVDet model:

```bash
/home/mostafa/.conda/envs/openmmlab/bin/python \
  tools/custom_data/prepare_fisheye3dod_inference.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted \
  data/Custom/BAG_2026_09_30_10_10_54_converted/calibration.json

/home/mostafa/.conda/envs/openmmlab/bin/python tools/test.py \
  projects/Fisheye3DOD/configs/custom_fisheye_bevdet.py \
  projects/Fisheye3DOD/weights/fisheye_bevdet.pth
```

Preparation defaults to 2 Hz and remaps each physical OpenCV-fisheye camera
into the pretrained model's 800x800 OCamCalib lens geometry. This preserves the
released model's learned spherical sampling geometry while using the manual
per-camera extrinsics from `calibration.json`.

Create LiDAR-verified training pseudo-labels from the bag's
`FusedObjectList` topic:

```bash
/home/mostafa/.conda/envs/openmmlab/bin/python \
  tools/custom_data/prepare_fused_groundtruth.py \
  data/Custom/BAG_2026_09_30_10_10_54 \
  data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_inference_2hz \
  data/Custom/BAG_2026_09_30_10_10_54_converted/calibration.json
```

The tool decodes the authoritative `object_fusion.proto` field layout,
transforms kEgo boxes into CLIDAR, recovers unknown classes from track history
or conservative dimensions, and rejects boxes without local LiDAR support. It
writes train/validation info files, an audit CSV, a summary, and BEV overlays
under `groundtruth/`. These are pseudo-labels and should be manually reviewed
before they are treated as final ground truth.

Fine-tune from the released checkpoint with:

```bash
/home/mostafa/.conda/envs/openmmlab/bin/python tools/train.py \
  projects/Fisheye3DOD/configs/custom_fisheye_bevdet_train.py
```

Manually review and edit the automatic labels with synchronized camera and
LiDAR views:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
/home/mostafa/.conda/envs/openmmlab/bin/python \
  tools/custom_data/manual_3d_annotation.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_inference_2hz
```

Click anywhere inside a BEV footprint to select it, drag it to move it, and
edit its center, dimensions, yaw, or class in the right panel. **Add box** uses
four approximate footprint-corner clicks and fits a strict rotated rectangle.
Boxes propagate into each unsaved next frame, including their configurable
per-frame `delta X`, `delta Y`, and `delta yaw`; new automatic tracks are merged
in. An explicitly deleted track is recorded and is not resurrected later.
`Ctrl+S` saves the current frame and Space saves and advances. Each frame is
atomically saved as a separate JSON in `manual_annotations/frames`, so the
session resumes without losing completed work. **Export training PKLs** writes
manual all/train/validation files while retaining the established temporal
split. Unsaved frames retain the automatic fused/LiDAR annotations.

After exporting, train with the manual annotation config:

```bash
/home/mostafa/.conda/envs/openmmlab/bin/python tools/train.py \
  projects/Fisheye3DOD/configs/custom_fisheye_bevdet_train_manual.py
```

Convert a downloaded CaliperGT scene export back into Fisheye3DOD/MMDetection3D
info files and fine-tune on the human-reviewed labels:

```bash
/home/mostafa/.conda/envs/openmmlab/bin/python \
  tools/custom_data/import_caliper_annotations.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_inference_2hz/scene_caliperai_scene_2026-10-19T09-41-49

CUDA_VISIBLE_DEVICES=0 /home/mostafa/.conda/envs/openmmlab/bin/python \
  tools/train.py \
  projects/Fisheye3DOD/configs/custom_fisheye_bevdet_train_caliper.py
```

For native KB4 training, preserve the exact annotated token list while linking
the original-resolution synchronized camera frames (the annotations themselves
are copied unchanged):

```bash
/home/mostafa/.conda/envs/openmmlab/bin/python \
  tools/custom_data/prepare_fisheye3dod_inference.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted \
  data/Custom/BAG_2026_09_30_10_10_54_converted/calibration.json \
  --output data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_native_2hz \
  --image-mode native \
  --template-info data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_inference_2hz/fisheye3dod_infos_test.pkl \
  --prune

cp -al \
  data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_inference_2hz/caliper_annotations \
  data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_native_2hz/
```

The config initializes model weights from
`projects/Fisheye3DOD/weights/fisheye_bevdet.pth` and starts a new optimizer
schedule. Caliper box centers and yaw are already in the exported LiDAR frame,
so no coordinate transform is applied. Empty frames are retained in the `all`
file but excluded from training and validation; contiguous 40-second blocks are
used for the default 80/20 temporal split.

Run inference and create combined four-camera/LiDAR visualizations for an
inclusive range of prepared frame indices:

```bash
MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES=0 \
/home/mostafa/.conda/envs/openmmlab/bin/python \
  tools/custom_data/infer_visualize_frame_range.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_inference_2hz \
  work_dirs/custom_fisheye_bevdet_train_manual_355/epoch_20.pth \
  --start-frame 500 --end-frame 600
```

The frame bounds are zero-based and inclusive. The output contains the subset
info, raw predictions, one combined PNG per frame, an MP4, and a JSON summary.

Export the synchronized raw fisheye images, manually corrected calibration,
and binary PCD point clouds as a CaliperGT scene bundle:

```bash
/home/mostafa/.conda/envs/openmmlab/bin/python \
  tools/custom_data/export_caliper_scene.py \
  data/Custom/BAG_2026_09_30_10_10_54_converted/fisheye3dod_inference_2hz \
  calibration_manual.json \
  --zip
```

The generated `caliperai_scene/` uses zero-padded, position-aligned names in
`lidar/` and `cameras/{front,left,right,rear}/`. Images retain their native
1920x1536 OpenCV-fisheye geometry, represented as `kannala_brandt` in Caliper's
calibration format. Images are hard-linked when possible to avoid duplication.
