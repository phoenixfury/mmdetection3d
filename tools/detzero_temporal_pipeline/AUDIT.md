# DetZero + recorder pipeline: code audit

Review basis: uploaded `DetZero(1).zip`, `recorder_pipeline.zip`,
`run_detzero_sequence(1).py`, and `improve_detzero_annotations.py` (30 October
2026). Checked against the public converter in
`phoenixfury/Fisheye3DOD` and ROSA's stated Fisheye3DOD reuse. **Static code
review + focused unit tests**; not an accuracy benchmark on physical recordings.

## Changes included (fixed or mitigated)

| Impact | Location | Observed issue | Fix in patch |
|---|---|---|---|
| Critical | `improve_detzero_annotations.py:main()` | Loads BASELINK `odometry_poses.npy` directly and uses it to transform CLIDAR 3D boxes. Ego rotation introduces incorrect lever-arm compensation for camera-track holds. | Both scripts read per-manifest `odometry_index`, calculate `T_world_CLIDAR = T_world_BASELINK @ T_BASELINK_CLIDAR` in one shared module. |
| High | `recorder_pipeline/.../outputs.py:export()` | `sync.py` writes `arc_*_timestamp_ns`, `camera_sync_clock` and target, but final `manifest.csv` drops them. | Preserve all original camera recorder packet timestamps, synchronization reference clock and target. |
| High | `improve_detzero_annotations.py:predicted_hold_object()` | Forces held object's local Z to the most recent local Z, even when ego elevation changes. | Hold world-frame height constant for planar prediction, then transform to current LiDAR; test covers ego-height changes. |
| High | Both DetZero result exporters | Cannot tell real LiDAR hits, camera-continued boxes, and tracker coasts from classic outputs. | Versioned per-frame `_temporal.json` sidecars aligned to immutable frame/object indexes; no mutation of classic result structure. |
| High | `improve_detzero_annotations.py:run_yolo()` | YOLO cache keyed only by frame number and camera; changing weights, input images, thresholds or image size produces silent reuse. | Input + model/stat + parameter fingerprint sidecar. Pre-existing caches without metadata are intentionally invalidated. |
| High | `run_detzero_sequence.py:detect()` | Cached CenterPoint predictions depend on model/manifest state but previous key checked only checkpoint *path*, label schema, and score thresholds. | Include model configuration content hash, checkpoint size/mtime and per-cloud timestamp/file stat signature. |
| High | `run_detzero_sequence.py:check_frame_boxes()` | Merely prints invalid boxes/pose and proceeds, allowing NaNs to reach association, crop, and refiner kernels. | Fail-fast validation of finite proper matrices, box dimensions, and score arrays. Detector drops nonfinite/invalid boxes. |
| Medium | `run_detzero_sequence.py:stabilize_tracks()` | Yaw interpolated against **frame IDs** instead of real sampling time; irregular acquisitions are misweighted. | Interpolate against actual acquisition time array. |
| Medium | `DetZero/refining/detzero_refine/datasets/dataset.py:load_track_infos()` | Python `str.strip(chars)` used as a prefix/suffix removal operation; it corrupts certain sequence tokens. Empty `seq_ids` later causes divide-by-zero. | Explicit `startswith`/`endswith` slicing and empty-sequence failure. |
| Medium | 10 files across `DetZero/{tracking,refining,utils}` | `np.int` and `np.bool` removed in modern NumPy; runner monkey patch hides incompatibilities only within its own process. | Replace deprecated aliases with `int`/`np.bool_` in those source files. |
| Medium | `run_detzero_sequence.py:main()` | Frame timestamps converted to float64 before epoch offset subtraction, giving needless large-epoch quantization. | Integer nanosecond subtraction before conversion to floating seconds. |
| Medium | `improve_detzero_annotations.py:main()` | Calibration default points only at sequence parent, while runner also checks sequence itself. | Use both candidates consistently. |
| Medium | Annotation format adapter | Fisheye3DOD expects per-frame `label_3d` list with four specific fields and six classes, which existing DetZero JSON does not directly provide. | Export optional classic-compatible `label_3d/<frame_id>.json` plus `frame_map.json`, preserving six-class fallback without losing original class in sidecar. |

## Outstanding / priority backlog

### P0 — correctness and data validity

**1. Variable-step time tracking (not fixed).**
`run_detzero_sequence.py:track()` sets `FILTER.DELTA_T = 0.5` and
`post_process.velocity_optimize` divides displacement by 0.5; the Kalman
transition in `DetZero/.../kalman_filter/kalman_filter.py` builds its `F`
matrix with the same fixed step. Splitting only at `dt <= 0` or `dt > 4 s`
means a valid 1.9-second acquisition gap is treated as a 0.5-second step.
Track speed, association gates, uncertainty and coast distance are inconsistent.
**Recommendation:** pass `timestamp_ns` through tracker frame data; make
`predict(dt)` / `F(dt)` / `Q(dt)` dynamic, use actual `Δt` for velocity, and
express `death_age` in seconds. Benchmark IDF1/HOTA, ID switches and motion
residuals across dropped-scan intervals.

**2. Camera clock does not equal exposure clock.**
`recorder_pipeline/src/recorder_pipeline/camera.py` derives image time from
camera packet `.idx`; the LiDAR clock is the embedded
`PointCloud.timestamp`, and localization clock is
`EgoPoseFused.timestamp`. Packet clock offsets and image sensor latency are
not established. Camera metadata in the sidecar deliberately says
`camera_recorder_packet`. **Recommendation:** capture camera hardware
SOF/exposure timestamps if available; calibrate clock domains and publish
`exposure_midpoint_timestamp_ns` plus uncertainty separately. Do not train
motion models assuming these are directly comparable.

**3. Incomplete pose/frame verification.**
The recorder `odometry.py` itself notes that the fused localization `.proto`
does not define geographical frame convention, orientation hand, or
measurement timestamp units. The current pipeline assumes
`Rz(yaw)Ry(pitch)Rx(roll)` and `T_world_BASELINK` while CLIDAR calibration
must be measured. **Recommendation:** verify a static object while ego turns,
check camera re-projection before/after and assert a known point's transformed
position. Reject uncalibrated CLIDAR for temporal exports.

**4. Invalid or inconsistent tracks from DetZero post-processing.**
Array length mismatches, NaN geometry and duplicate frame IDs can flow into
`prepare_refining_data()` or `refiner_outputs`, especially after track merges.
New export checks detect many of these, but does not repair track identities.
**Recommendation:** add a tracker invariant validator at the *native* and
post-merge boundaries checking unique sample index, aligned fields, finite
scores/boxes, valid dimensions, and unique track ID per frame.

### P1 — temporal annotation quality

**5. Camera-confirmed holding is not true camera 3D estimation.**
`predicted_hold_object()` uses a tracker global coast or constant velocity,
then checks a 2D fisheye YOLO box. Where LiDAR support is absent, geometry
is extrapolated and the source is `camera_hold`, not a new LiDAR measurement.
**Recommendation:** store uncertainty covariance, `last_real_lidar_time_ns`,
source-specific confidence, hold duration in **seconds**, and limit such
labels in training (lower weights / short horizon). Distinguish observed
with-LiDAR-support vs pure camera continuation.

**6. Asynchronous 2D/3D projection.**
`match_frame()` compares LiDAR 3D boxes projected with static extrinsics to a
nearest camera packet, without object/ego interpolation to that camera's true
exposure time. **Recommendation:** once exposure clocks are validated,
interpolate `T_world_camera(t_cam)` and predict object pose at `t_cam`, then
project rather than comparing geometry at `t_lidar` to an older/newer image.

**7. Projection model limits.**
`improve_detzero_annotations.py:kb4_project()` assumes a four-coefficient
Kannala–Brandt-style angle polynomial, hardcodes a 110-degree angular cutoff,
and approximates partially visible 3D boxes using only valid projected
corners. The recorder calibration warns that `GENERIC_CAMERA_MODEL` is not
verified as physically identical to this projection. **Recommendation:**
project edges clipped to the field of view; validate against observed
checkerboards; provide a separate OCam model for the Fisheye3DOD/ROSA
training camera geometry.

**8. Camera validation thresholds may suppress valid objects.**
`match_frame()` retains one matched camera per 3D hypothesis and
`--keep-unconfirmed` is off by default. Objects at seams, occluded in
fisheye images, or at long range may be systematically removed.
**Recommendation:** aggregate evidence across visible cameras, add an
`unconfirmed_lidar` status rather than dropping labels by default, and compare
class-conditional precision/recall against human-reviewed frames.

**9. Class taxonomy is not exactly model taxonomy.**
The CenterPoint mapping collapses `Trailer -> Truck`,
`Motorcycle/Bicycle -> Cyclist`, while Fisheye3DOD's six labels do not include
`ConstructionVehicle`. Classic export maps that class to `Van` for this
project's six-class taxonomy, but keeps
original name and a sidecar record. **Recommendation:** make mappings explicit
in a versioned dataset taxonomy table, and audit per-class distributions.

### P2 — performance and maintainability

**10. Memory and I/O pressure.**
`read_binary_pcd()` reads whole PCD files and `prepare_refining_data()` keeps
an unbounded `cloud_cache` of frame arrays. Detector cache pickles the full
completed prefix on **every frame**, creating O(N²) cumulative serialization
writes. **Recommendation:** use mmap/fromfile for binary PCD payloads; use a
bounded frame LRU; checkpoint detections every 50–100 frames plus at end, or
write per-frame atomic records and a resumable index.

**11. Model deployment assumptions.**
`detect()` runs on `cuda:0`, DetZero overlap merging uses direct `.cuda()`,
and PRM remains Waymo-trained. **Recommendation:** parameterize device,
shard frame detection across GPUs if needed, and profile tracker/Refiner memory
separately. Treat PRM as an explicitly evaluated domain-transfer option, not
as automatically better pseudo-GT.

**12. Dataset freshness and reproducibility.**
YOLO fingerprints now record weight and image stat metadata, while the main
detector cache includes checkpoint file stats and config hash. These are
pragmatic invalidation keys, **not content hashes of all model/image bytes**;
files altered while preserving file size and modification time can escape
invalidation. Track/refiner outputs are still reused across stages without
full dependency manifests. **Recommendation:** persist a run manifest with
Git commit SHAs, model SHA256, calibration SHA256, thresholds, dataset UUID,
backend version, and explicit file lineage for every stage.

**13. Recorder's LiDAR scans are not deskewed.**
One ego pose per scan is exported. LiDAR point-wise measurement offsets are
not available; motion within a revolution deforms both source geometry and
camera projection. **Recommendation:** if native Ouster column/packet timing
can be preserved, deskew the cloud before any pseudo-label generation.

**14. Recorder output atomicity is per-file only.**
Existing `outputs.py` publishes manifests after payload files, but a crash
leaves orphaned point clouds/images and can leave partial stage products.
**Recommendation:** export into a temporary sequence directory and atomically
publish a validated dataset manifest; write a provenance checksum file.

## Suggested validation on BAG_2026_09_30_10_10_54

1. Re-export the recorder dataset from synchronized input with the updated
   `outputs.py`, verify each `arc_*_timestamp_ns` is present and check the
   distribution of camera packet minus LiDAR acquisition times.
2. Run one scene with ego turns; compare projected static box center errors
   using `T_world_BASELINK` alone versus `T_world_CLIDAR`, using CLIDAR
   extrinsics measured for that exact rig.
3. Export tracking hits-only and camera-improved sidecars; compare label
   counts by `source`, `hold_age_frames`, camera, distance and class.
4. Compare `lidar`-only labels against the camera-held subset using human
   inspection and LiDAR support. Do not treat camera-held predictions as full
   ground truth for AP/NDS measurements.
5. Inject variable Δt (0.5, 1, 2 s) and one reversed timestamp; check
   tracking reset behavior, speed biases and resulting sidecar timestamps.
6. Add a training adapter that consumes `frame_map.json`, camera calibration
   and images to generate Fisheye3DOD `ImageSets-2hz` plus `fisheye3dod_infos`.
   Keep source flags in a separate metainfo/sidecar until the dataset loader
   explicitly supports training weights or temporal samples.
