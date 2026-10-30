# Multi-sensor recorder pipeline (v0.4.2)

Decode the four ARC surround-view cameras, Ouster `PointCloud` messages, camera
calibration, and `adas.localization.EgoPoseFused` from a folder recorder into
**timestamped JPEGs + LiDAR XYZI/PCD + synchronized odometry**.

This repository contains the **exact schemas supplied by the user**:
`proto/bag.proto`, `proto/localization.proto`, `proto/lidar.proto`. Checked-in
Python protobuf classes are available in `src/recorder_pipeline/protos`; CMake
generates C++ classes directly from the `.proto` files at build time.

## Ubuntu 22.04 setup

```bash
sudo apt update
sudo apt install -y python3-venv ffmpeg libpcl-dev libprotobuf-dev protobuf-compiler cmake build-essential
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
# Only needed for compressed PCL-octree point clouds:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRECORDER_PROTOBUF_PROVIDER=system
cmake --build build --parallel 4
pytest -q
```

`ffmpeg` and `ffprobe` executables must be on `PATH`. PCL, `protoc`, and the
compiled `decode_lidar` executable are **only necessary** when decoding the
PCL-compressed LiDAR variant. Python raw LiDAR decoding needs no C++ build.

## Protobuf + Abseil linker fix (v0.4.2)

Ubuntu 22.04 PCL/VTK is designed to work with the Ubuntu Protobuf runtime. A
hand-installed `/usr/local/lib/libprotobuf.a` (for example Protobuf 6.30.1)
may get selected by CMake instead. This gives undefined symbols involving
`absl::lts_20250127` when the matching Abseil libraries are not linked, and
can also mix incompatible headers, `protoc`, and runtimes.

The default `RECORDER_PROTOBUF_PROVIDER=system` forces Ubuntu's
`/usr/bin/protoc`, `/usr/include/google/protobuf`, and
`/usr/lib/<multiarch>/libprotobuf.so` **as a consistent toolchain**. It also
adds an isolated include overlay to prevent `/usr/local/include` from
silently shadowing the selected packaged headers. The build checks that the
compiler version matches the linked library, and enables experimental
proto3 `optional` compilation for Ubuntu's protoc 3.12 when needed.

```bash
sudo apt update
sudo apt install -y libprotobuf-dev protobuf-compiler
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DRECORDER_PROTOBUF_PROVIDER=system
cmake --build build -j4
ctest --test-dir build --output-on-failure
# Confirm the selected versions/paths in the CMake output or:
grep -E '(Protobuf_LIBRARY|Protobuf_PROTOC_EXECUTABLE|Protobuf_INCLUDE_DIR):' build/CMakeCache.txt
```

When building against a complete custom Protobuf 6 installation, select
`-DRECORDER_PROTOBUF_PROVIDER=config -DProtobuf_DIR=/path/to/protobuf/cmake`.
This **requires** a working upstream CMake package exporting
`protobuf::libprotobuf`, `protobuf::protoc`, and all its Abseil dependencies.
Use `-DRECORDER_PROTOBUF_IMPORT_ROOT=/path/to/include` if its timestamp.proto
location cannot be detected. Do **not** patch this by linking Ubuntu's old
`absl` packages to the Protobuf 6 archive: ABI namespaces must match.

## Performance controls (v0.4)

The four video streams are decoded concurrently by default (`--camera-workers
4`, `--ffmpeg-threads 1`). Their manifest ordering and packet-PTS timestamps
are unchanged, and temporary JPEGs are staged on the output filesystem.
Use `--camera-workers 1` for legacy single-stream-at-a-time operation or
`--camera-workers 2` when storage throughput/RAM is limited.

`--link-images` (on `all` or `export`) is an **opt-in** faster dataset export
that creates hard links to decoded native JPEGs when possible, falling back
to regular copies. Hard-linked files share their underlying bytes: **do not
modify source or exported JPEGs in place**. Without this flag, copies are used
as before. `--timings` prints wall-time per processing stage. See
[PERFORMANCE.md](PERFORMANCE.md) for synthetic speed measurements, regression
results and detailed limitations.

## Commands

```bash
# End-to-end, PCL-compressed LiDAR (default), native camera JPEG and fused odometry
recorder-pipeline all /home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56  \
    --lidar-decoder ./build/decode_lidar \
    --camera-workers 4 \
    --timings

# Uncompressed LiDAR format: PointCloud.data contains 4 little-endian float32s/point
recorder-pipeline all /data/BAG --output /data/converted \
  --lidar-encoding raw --raw-components 4 \
  --odometry-dir /data/BAG/path/to/FusedOdometry \
  --sample-rate 2 --tolerance-ms 20

# XYZ-only raw LiDAR: pads intensity with 0 (do NOT use for XYZI payloads)
recorder-pipeline lidar /data/BAG --output /data/converted \
  --lidar-encoding raw --raw-components 3

# Individual stages, each independently rerunnable
recorder-pipeline camera /data/BAG --output /data/converted
recorder-pipeline lidar /data/BAG --output /data/converted --lidar-encoding pcl \
  --lidar-decoder ./build/decode_lidar
recorder-pipeline calibration /data/BAG --output /data/converted
recorder-pipeline sync /data/BAG --output /data/converted --sample-rate 2 --camera-sync-clock sensor
recorder-pipeline export /data/BAG --output /data/converted \
  --odometry-dir /data/BAG/path/to/FusedOdometry
```

The positional `BAG` argument on `sync` and `export` is for CLI consistency;
these stages operate on manifests in `OUTPUT`. Use `--overwrite` to rerun a
stage against preexisting data; the default prevents silently reusing a stale
result. For testing use `--limit 10`. To export without odometry, omit
`--odometry-dir`. Export requires synchronized LiDAR and all four cameras.

`--pose-timestamp-unit ms` is the default and reproduces the previous pipeline's
assumption; **the provided `.proto` does not define its units**. Verify your
recording before changing it to `us` or `ns`.

Export skips synchronized samples inside genuine localization outages by
default and writes their timestamps and errors to `dataset/skipped_odometry.csv`.
Use `--missing-odometry-policy error` for fail-fast operation. A one-sided raw
pose is only held at a segment boundary when it is within
`--max-odometry-delta-ms`; longer gaps are never fabricated.

## Layout

```text
converted/
  images/arc_f/<recorder_ns>.jpg
  images/arc_r/<recorder_ns>.jpg
  images/arc_sl/<recorder_ns>.jpg
  images/arc_sr/<recorder_ns>.jpg
  lidar/<recorder_ns>.bin                # float32 little-endian XYZI
  camera_manifest.csv
  lidar_manifest.csv                     # stores sensor and recorder timestamps, encoding
  calibration.json                      # camera intrinsics/extrinsics; CLIDAR optional
  synchronization.csv                   # LiDAR sensor time to camera frame joins
  dataset/
    images/arc_f/<recorder_ns>.jpg      # also other three cameras
    lidar_pcd/<recorder_ns>.pcd         # binary PCD v0.7 (same XYZI floats)
    manifest.csv
    odometry.csv                        # if --odometry-dir specified
    odometry_poses.npy                  # N x 4 x 4 T_world_base (assumed frame)
    metadata.json
```

## Processing and correctness guarantees

1. Camera MP4/H264 chunks are probed for original packet PTS. FFmpeg `showinfo`
   PTS maps each decoded frame to the correct `.idx` value. **Elementary H264
   streams without PTS are deliberately rejected** rather than silently
   assigning wrong frame times (particularly with B-frames).
2. The `.bin` LiDAR topic and `.idx` are decoded as `bag.TopicData.samples`.
   `samples[].data` is parsed as `PointCloud`, matching the exact proto.
   The raw backend streams individual `bag.Sample` fields instead of loading
   entire protobuf chunks in memory.
3. LiDAR encoding must be selected explicitly. `PointCloud.data` has no tag
   indicating raw vs PCL octree; merely checking byte length is unsafe. Raw
   XYZ/XYZI decoding rejects unexpected byte lengths. PCL decoding uses the
   `pcl::io::OctreePointCloudCompression<pcl::PointXYZ>` backend.
4. The recorder `.idx` (read/write time) and embedded PointCloud `timestamp`
   (acquisition time) are exported **separately**. By default, synchronization
   uses acquisition time; camera manifest timestamps still come from camera
   `.idx`. Rate limiting uses the same selected matching clock.
5. Fused odometry uses `bag.TopicData.samples[].data` -> `EgoPoseFused`.
   The pose measurement `timestamp` determines interpolation order; neither
   sample transport timestamps nor `.idx` order is substituted for it.
   `Sample.timestamp`, `Sample.read_timestamp`, and `.idx` are retained for
   diagnostics in `odometry.csv`.
6. Odometry interpolation uses shortest-path periodic Euler-angle interpolation
   and linear translation/velocity, with configurable tolerance to both
   bracketing records. **No extrapolation** outside fused coverage and no
   filling large gaps without a verified fallback source. At a raw-fallback
   segment boundary, the nearest raw pose may be held only when it is within
   the same configured tolerance; the output source records that hold.
7. All point clouds stay in `CLIDAR` coordinates. `calibration.json` is
   emitted even if the bag omits the CLIDAR extrinsic, but in that case
   `lidar_to_camera` is empty rather than using an approximate hard-coded
   transform. For a known T_BASELINK_CLIDAR pass `--lidar-to-base` and 16
   row-major values.
8. Writes use temporary files and atomic replacement where practical;
   manifests are published after payload extraction succeeds. This is
   **not** a single atomic transaction across all output files. A failed
   run may leave unreferenced payloads. Use a fresh output directory for
   calibration changes or partial/corrupt recordings.

## Frame and intensity caveats

- **PCL compression path:** original `PointXYZ` has only XYZ; exported intensity
  is zero. Do not interpret this as measured reflectivity.
- **Raw four-component path:** the fourth float is preserved as recorded.
  Confirm its physical meaning/scale in the source recorder; the proto does not
  specify it.
- **Raw three-component path:** fourth float is padded with zero.
- **Pose conventions:** the matrix uses `Rz(yaw)Ry(pitch)Rx(roll)` and columns
  `[x,y,z,1]`. The `.proto` contains no reference frame name and no angle
  unit; verify frame and radians. Transform the cloud with the calibrated
  `T_BASELINK_CLIDAR` before using ego poses for LiDAR motion compensation.

## Raw odometry fallback

The supplied protos define `adas.localization.EgoPoseFused` but do **not**
define the sibling `adas.common.EgoPose` message formerly imported from
`common.proto`. The previous raw fallback had an undefined field layout and
assumed it could be used. It has been removed rather than silently decoding
another message as raw odometry. Supply `common.proto` to implement this as a
separate, schema-verified adapter later.

## Architecture

| Component | Single responsibility |
| --- | --- |
| `bag_io.py` | Lightweight recorder wire-format parser for calibration |
| `protos/*_pb2.py` | Typed protobuf message classes for known schemas |
| `camera.py` | PTS-preserving FFmpeg decoding of H264/MP4 |
| `lidar.py` | Strict decoding of raw XYZ/XYZI PointCloud payloads |
| `cpp/decode_lidar.cpp` | Decoding of PCL-octree PointCloud payloads |
| `calibration.py` | Calibrations in BASELINK and transform provenance |
| `sync.py` | Bounded, nearest-neighbor temporal association |
| `odometry.py` | Typed pose decoding and interpolation |
| `storage.py` | Binary XYZI -> PCD conversion |
| `outputs.py` | Native dataset materialization and metadata |
| `cli.py` | Orchestrates distinct pipeline stages |

This is a SOLID-oriented refactor (single-purpose components, replaceable
backends, explicit dependencies), not a guarantee that every conceivable
SOLID rule has been mechanically enforced.

## Validation

`pytest -q` exercises synthetic `bag.TopicData` records, actual typed
`EgoPoseFused` values, shortest-path yaw interpolation, temporal gaps,
raw LiDAR XYZ/XYZI, JPEG/PCD/CSV/NPY export, and path validation.

The PCL C++ binary cannot be compiled/tested without a machine that has
`libpcl-dev` + `protoc` installed. The tests **do not** validate proprietary
real-bag recordings or sensor calibration conventions. Full sensor overlay
checks should be completed before model training or ego-motion compensation.

### Raw odometry fallback (v0.4.6)

If fused odometry stops before the LiDAR timestamps or develops gaps, optionally
supply the sibling raw topic and **the original `adas.common.EgoPose` schema**.
Raw and fused pose clocks are interpreted with the same `--pose-timestamp-unit`.

```bash
recorder-pipeline export BAG CONVERTED \
  --odometry-dir BAG/__engineering__adcu__debug__ZMQ__Localization__FusedOdometry \
  --raw-odometry-dir BAG/__engineering__adcu__debug__ZMQ__Localization__Odometry \
  
  --overwrite --timings
```

`--proto-dir` is optional; the bundled `common.proto` defines
`adas.common.EgoPose`. It is not included because it has not been supplied.
This adapter verifies the historically observed fields
`timestamp`, `pose.position`, and a **scalar yaw** `pose.orientation`.
If the real schema differs, decoding fails explicitly rather than guessing.
The raw trajectory is aligned using the nearest overlapping fused boundary;
this is a continuity heuristic, not a guarantee of globally correct localization.
Check pose frames and units. Output `odometry.csv` marks rows `fused` or
`aligned_raw`, and both paths emit the same CSV columns.

Output already created by an incomplete export may be overwritten during
`export`. The source camera JPGs and original LiDAR binary files are not
re-decoded by the export command.

### Raw odometry schema (v0.4.7)

The user-provided `adas.common.EgoPose` schema is bundled in `proto/common.proto` and
`src/recorder_pipeline/schemas/common.proto`; raw odometry fallback needs `protoc`
installed but no custom `--proto-dir`. Raw status `kNotAvailable=0` is excluded;
`kAvailable=1` and `kAvailableReduced=2` are accepted. Raw pose has only XY/yaw,
so z/roll/pitch are borrowed from the fused alignment anchor and raw velocity
is not estimated. Existing camera and LiDAR conversions are reused by the export command.
