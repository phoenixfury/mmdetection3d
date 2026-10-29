# Bug audit — v2 baseline to v3

## Corrected and covered by Python tests

| Impact | Baseline issue | v3 resolution | Test / verification |
|---|---|---|---|
| Critical | `load_fused_odometry()` required `common.proto` and a raw `adas.common.EgoPose`, although neither was supplied. This prevented fused-only export. | Parse the shipped `bag.TopicData` plus `adas.localization.EgoPoseFused` directly. Raw fallback is unavailable until its schema is provided. | Synthetic TopicData/EgoPoseFused and end-to-end tests |
| Critical | PointCloud decoded as PCL octree unconditionally, but proto allows raw float32 data. | Explicit `--lidar-encoding pcl|raw`, raw XYZ and XYZI validators, typed PointCloud parsing. | Synthetic raw XYZ/XYZI tests; PCL still requires system validation |
| High | Original loader sorted pose samples by `.idx` recorder timestamp, although synchronization used `pose.timestamp`. | Sort by measurement clock; preserve `.idx` and both Sample timestamp fields for diagnosis. | Out-of-order transport timestamps test basis |
| High | Interpolator silently held endpoints outside coverage and across arbitrarily long gaps if the nearest measurement was close enough. | Explicit two-sided temporal gap check and no extrapolation. | Outside coverage and gap rejection tests |
| High | Sampling/rate limit measured recorder write time even when camera matching used LiDAR acquisition time. | Match and subsample in the same selected clock domain. | Two-scan synthetic time discrepancy test |
| High | Some outputs silently reused existing files during a rerun without confirming contents were the same. | Fail if destinations exist unless `--overwrite` was specified; atomically update individual files/manifests. | Rerun-collision tests |
| High | Fused/raw alignment was using unverified raw message fields; frame convention could silently be wrong. | Remove schema-less fallback until `common.proto` is available. | Explicit availability checks |
| Medium | Calibration extraction blocked the whole native pipeline whenever CLIDAR extrinsics were absent; older script could inject an approximate hardcoded transform. | Missing CLIDAR does not block camera calibration; never inject a guessed transform; preserve provenance. | End-to-end test omits CLIDAR |
| Medium | Export input paths could escape the output root via malformed manifests. | Reject absolute and parent-traversal file paths. | Unsafe-path test |
| Medium | Raw intensity handling was implicit. | Preserve fourth float only when raw XYZI; zero-fill XYZ and PCL PointXYZ, and mark in manifest. | Raw tests and field-format checks |
| Medium | Source C++ decoder manually parsed envelopes instead of consuming user-specified `.proto` type. | CMake generates `bag.pb.*` and `lidar.pb.*`; C++ decoder uses typed messages. | Requires PCL/Protobuf toolchain build |

## Known limitations / external validation needed

1. **PCL codec compatibility:** the supplied proto does not define the binary compression algorithm. C++ continues to use the previous PCL octree decoder, which has not been compiled here or validated against a real Ouster recording.
2. **Camera packet timestamps:** FFmpeg logic maps decoded PTS to recorder packet PTS; raw elementary H264 files lacking PTS are rejected. Compare sample overlays and timestamps on real recordings, especially videos with B-frames or damaged frames.
3. **Odometry time units:** EgoPoseFused.timestamp is an int64 without unit declaration. Prior implementation assumed milliseconds. CLI supports `ms`, `us`, `ns`; user must verify which is correct.
4. **Odometry coordinate frame:** localization proto does not specify whether positions and angles are world/ENU/NED, whether angles are radians, nor yaw handedness. Export uses a documented `T_world_base` convention provisionally; validate frames against real sensor data.
5. **LiDAR raw schema ambiguity:** `PointCloud.data` has neither a codec tag nor a point-channel specification. User must choose raw field count 3/4 and encoding. Any other custom structure needs an adapter.
6. **Motion compensation:** interpolation provides one pose at each scan acquisition timestamp; it does not deskew individual LiDAR points using firing times, and does not by itself transform CLIDAR into BASELINK. Per-point timing and a verified extrinsic are needed for full deskew.
7. **Memory consumption:** raw LiDAR and fused odometry now stream `bag.Sample` messages from disk, but the PCL C++ decoder still deserializes an entire `bag.TopicData` chunk; very large PCL chunks may need a validated streaming implementation.
8. **Atomicity:** payload files are written individually and manifests only at completion. An interrupted stage can leave orphan files. Rerunning with `--overwrite` (or a fresh destination) recovers; no cross-stage transaction is provided.
9. **Calibration projection assumptions:** `GENERIC_CAMERA_MODEL` was previously treated as an OpenCV fisheye approximation. Native export does not remap cameras; trained-model OCam remapping should be isolated in a separate adapter, verified per camera.
10. **Real recordings / build validation:** synthetic Python end-to-end pass is not evidence of correct native PCL decompression, physical calibration, or real camera packet association. Run on representative real bag data and inspect overlays/point counts/time offsets.

## v0.4 performance changes

See `PERFORMANCE.md`. The raw LiDAR Python converter now avoids whole-chunk
parsing and redundant XYZI buffer copies. The C++ PCL writer batches float32
output, with a separate byte-level native C++ test. Camera FFmpeg jobs can run
independently in parallel and stage output on the destination filesystem.
Hard-linked dataset JPEGs remain explicit opt-in because source and export
then share underlying bytes. All timestamp pairing and CSV field contracts
remain unchanged. Real PCL compressed decoding is still unverified.
