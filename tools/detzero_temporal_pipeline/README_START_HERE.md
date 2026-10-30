# BAG → DetZero → shared Fisheye3DOD/ROSA dataset

Run from the `mmdetection3d` repository root. This is the current sensor-clock
workflow. The recorder keeps native fisheye images; the 800×800 OCam images
are generated separately. Use the installed CUDA environment and GPU.

## 0. Prerequisites

```bash
conda run -n openmmlab pip install -e tools/detzero_temporal_pipeline/recorder_pipeline
test -f checkpoints/centerpoint_nus_voxel0075_dcn.pth
test -d checkpoints_det3zero
export BAG=/path/to/BAG_YYYY_MM_DD_HH_MM_SS
export CONVERTED=/path/to/BAG_YYYY_MM_DD_HH_MM_SS_converted
export DATASET="$CONVERTED/dataset"
export YOLO_WEIGHTS=/path/to/yolo11m-seg.pt
export RECORDER="$PWD/tools/detzero_temporal_pipeline/recorder_pipeline"
```

For compressed PCL LiDAR, build the decoder once if it is not already
available (PCL and Protobuf development packages are required):

```bash
cmake -S "$RECORDER" -B "$RECORDER/build" -DCMAKE_BUILD_TYPE=Release \
  -DRECORDER_PROTOBUF_PROVIDER=system
cmake --build "$RECORDER/build" --parallel 4
```

## 1. Decode and synchronize the BAG

```bash
conda run --no-capture-output -n openmmlab recorder-pipeline all \
  "$BAG" --output "$CONVERTED" \
  --lidar-decoder "$RECORDER/build/decode_lidar" \
  --sample-rate 2 --tolerance-ms 20 --camera-sync-clock sensor
```

If the BAG lacks the CLIDAR→BASELINK extrinsic, supply the **measured**
row-major 4×4 matrix with `--lidar-to-base` (16 numbers). Do not borrow a
different rig's matrix without validation. For an existing converted BAG,
`--overwrite` is needed to replace generated data; back up manual edits first.
Verify `manifest.csv`: `camera_sync_clock=sensor`, valid
`lidar_sensor_timestamp_ns`, camera timestamps/deltas and `odometry_index`.
LiDAR acquisition time—not file/read time—is the tracking/dataset clock.
Camera timestamps are packet/index timestamps, not verified exposure times;
inspect projected boxes on several moving objects before accepting a BAG.

## 2. DetZero and postprocessing

```bash
conda run --no-capture-output -n openmmlab python \
  tools/detzero_temporal_pipeline/integration_scripts/run_detzero_sequence.py \
  "$DATASET" --stage all --calibration "$CONVERTED/calibration.json" \
  --tracking-profile camera-2hz --tracking-filter ab3dmot

conda run --no-capture-output -n openmmlab python \
  tools/detzero_temporal_pipeline/integration_scripts/improve_detzero_annotations.py \
  "$DATASET" --weights "$YOLO_WEIGHTS" \
  --calibration "$CONVERTED/calibration.json" --max-distance 60
```

Baseline: `detzero_output/results.json`. Improved labels:
`detzero_output/improved_results.json`, with temporal provenance in
`improved_results_temporal.json`. Current postprocessing confirms vehicles and
vulnerable road users by camera, classifies vehicles by dimensions, preserves
track classes, supports camera-held tracks, and rejects tiny camera detections
swallowed by huge projected boxes. `--keep-unconfirmed` is only for diagnosis.
Inspect `improved_results_report.json` and comparison video; calibration or
YOLO errors can still remove real objects. See
[README_INTEGRATION.md](README_INTEGRATION.md) for the sidecar schema.

```bash
conda run --no-capture-output -n openmmlab python \
  tools/custom_data/render_detzero_postprocess_comparison.py "$DATASET" \
  --postprocessed "$DATASET/detzero_output/improved_results.json"
```

## 3. Generate the 800×800 OCam images and info PKL

```bash
conda run --no-capture-output -n openmmlab python \
  tools/custom_data/prepare_fisheye3dod_inference.py \
  "$DATASET" "$CONVERTED/calibration.json" \
  --sync "$DATASET/manifest.csv" \
  --output "$DATASET/fisheye3dod_inference_2hz" --sample-rate 0
```

`--sample-rate 0` retains all frames already selected at 2 Hz. Re-selecting
at 2 Hz can lose frames from timestamp jitter. The incremental importer checks
that annotation, manifest and OCam template tokens match exactly.

## 4. Append a BAG to the combined training dataset

```bash
export COMBINED="$PWD/data/Custom/combined_improved_detzero_rosa"
conda run --no-capture-output -n openmmlab python \
  tools/custom_data/append_improved_detzero_rosa_dataset.py \
  --dataset "$COMBINED" --add-sequence 17_09 "$DATASET" --dry-run
```

Remove `--dry-run` after inspecting the reported sequence paths and counts.
Future bags need only `--add-sequence UNIQUE_NAME DATASET`; the script links
only new images and republishes the relatively small info PKLs, preserving
existing image assets. Anchor refitting rules are described below.

**One-time migration:** the current combined dataset's 16/30 September entries
were made with read-time camera synchronization. The importer intentionally
rejects appending to them. First run steps 2–3 on both `dataset_temporal`
sensor-clock sources, then replace both while adding the new BAG:

```bash
conda run --no-capture-output -n openmmlab python \
  tools/custom_data/append_improved_detzero_rosa_dataset.py \
  --dataset "$COMBINED" \
  --replace-sequence 16_09 data/Custom/BAG_2026_09_16_09_58_56_converted/dataset_temporal \
  --replace-sequence 30_09 data/Custom/BAG_2026_09_30_10_10_54_converted/dataset_temporal \
  --add-sequence 17_09 "$DATASET" --dry-run
```

Check the dry run, then repeat without `--dry-run`. Back up the current PKLs
if you need reproducible old metrics: the train/validation membership changes,
so old scores are not directly comparable. Replacement uses new filenames;
old linked image files are left recoverable, not deleted.

The new split is **per BAG, in LiDAR acquisition-time order**: earliest 80%
of elapsed time for train, latest 20% validation, with a 5 s training-side
purge gap. Gap samples
remain in `all` but not train/val. Scene and prev/next links stop at split
boundaries and recording gaps over 4 s. All samples carry `acquisition_timestamp_ns` and
`T_world_CLIDAR`; train/val PKLs are globally ordered by acquisition time.
ROSA now reads the real timestamp/pose, but its instance bank still rejects
`num_temp_instances > 0`. Chronological metadata does **not** yet enable
temporal slots; leave them disabled. Migration, replacements and changes to
the split automatically refit ROSA anchors using only the new training split.
Later appends preserve these training-only anchors unless `--refit-anchors`
is supplied. Old custom-trained checkpoints may already have seen frames
that move into validation: start from external pretraining for a clean
benchmark on the new split.

## 5. Train and evaluate each camera network

Use the same combined root on the server; change only its absolute path.
Images in the PKLs are relative to the dataset root, so the PKLs need no
path rewrite. Run from the repository root (not `projects/ROSA`).

Fisheye3DOD, two GPUs:

```bash
DUAL_FISHEYE_ROOT="$COMBINED" CUDA_VISIBLE_DEVICES=0,1 \
conda run --no-capture-output -n fisheye3d python -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29502 tools/train.py \
  projects/Fisheye3DOD/configs/custom_fisheye_bevdet_train_dual_export.py \
  --launcher pytorch --work-dir work_dirs/fisheye3dod_combined_2gpu
```

ROSA, two GPUs, from the class-remapped KITTI-360 initialization:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$PWD:$PWD/projects/ROSA:$PWD/projects/Fisheye3DOD" \
FISHEYE3DOD_OFFICIAL_ROOT="$PWD/projects/Fisheye3DOD" \
conda run --no-capture-output -n fisheye3d python \
  projects/ROSA/tools/train_balanced_fisheye.py \
  --recipe retrain --gpus 2 --data-root "$COMBINED" \
  --auto-batch --fast --work-dir projects/ROSA/work_dirs/combined_retrain
```

For adaptation of an existing compatible ROSA model, use
`--recipe baseline-adapt --init-checkpoint /path/to/rosa_best.pth` and a new
work directory. Use `--resume` only for the same recipe and dataset split.

Fisheye3DOD held-out validation metrics:

```bash
DUAL_FISHEYE_ROOT="$COMBINED" CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n fisheye3d python tools/test.py \
  projects/Fisheye3DOD/configs/custom_fisheye_bevdet_train_dual_export.py \
  /path/to/fisheye3dod_best.pth --work-dir work_dirs/fisheye3dod_eval
```

Fisheye3DOD synchronized-BAG inference and four-camera/LiDAR video (set an
end frame below the prepared info PKL's frame count):

```bash
DUAL_FISHEYE_ROOT="$COMBINED" CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n fisheye3d python \
  tools/custom_data/infer_visualize_frame_range.py \
  "$DATASET/fisheye3dod_inference_2hz" /path/to/fisheye3dod_best.pth \
  --config projects/Fisheye3DOD/configs/custom_fisheye_bevdet_train_dual_export.py \
  --start-frame 0 --end-frame 199
```

ROSA synchronized-BAG inference with four cameras and LiDAR video:

```bash
ROSA_DATA_ROOT="$COMBINED" CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD:$PWD/projects/ROSA:$PWD/projects/Fisheye3DOD" \
FISHEYE3DOD_OFFICIAL_ROOT="$PWD/projects/Fisheye3DOD" \
conda run --no-capture-output -n fisheye3d python \
  tools/custom_data/infer_rosa_synchronized_bag.py \
  "$DATASET/fisheye3dod_inference_2hz" /path/to/rosa_best.pth
```

For ROSA validation metrics, use `projects/ROSA/tools/run.py evaluate` with
the trained run's config and checkpoint, `ROSA_DATA_ROOT="$COMBINED"` and
the same `PYTHONPATH`/official-root variables. Its class/NMS/anchor settings
must match that checkpoint.
