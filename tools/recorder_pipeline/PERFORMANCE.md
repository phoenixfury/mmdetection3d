# Performance optimizations and regression evidence (v0.4)

This release prioritizes **preserving timestamp associations, wire-format decoding,
byte formats and error behavior**. It intentionally does NOT guess the LiDAR
compression variant, downsample camera frames implicitly, change JPEG quality,
or silently re-use preexisting outputs.

## Changes

| Stage | Bottleneck in v0.3 | v0.4 change | Correctness control |
|---|---|---|---|
| Camera | Four camera streams decoded sequentially | `--camera-workers 4` launches independent FFmpeg decoders; `--ffmpeg-threads 1` avoids excessive per-process threading | Per-sensor manifest order preserved; exact PTS-to-index mapping is unchanged |
| Camera | JPEGs staged in OS `/tmp`; moves to output could copy between filesystems | Temporary files created **inside** the target camera directory; rename stays on destination filesystem | Same FFmpeg commands, JPEG quality, PTS matching and manifests; JPEG byte equality regression tested |
| Raw LiDAR | Loading and parsing entire `bag.TopicData` chunks eagerly | Stream length-delimited `bag.Sample` messages from `.bin` and validate exact `.idx` count | Generated protobuf `Sample` and `PointCloud` parsers; timestamps, duplicates, lengths and finite XYZ still checked |
| Raw LiDAR | Duplicate memory copies of already-valid little-endian XYZI | Pass immutable protobuf byte buffer directly to file writer | Strict row count and XYZ finite check; raw byte comparison |
| Raw LiDAR | Accumulate all CSV manifest rows in memory | Write CSV incrementally to a temporary manifest; publish with rename at end | CSV schema, ordering and commit protocol preserved |
| PCL binary decoder | Four little-endian output writes per point | Batch serialize 16,384 XYZ points into contiguous XYZI chunks | Shared `cpp/xyzi_writer.hpp`; standalone C++ test of exact output, signed zero and crossing batch boundaries |
| Dataset export | Copy identical native JPEGs into a second directory | Optional `--link-images` creates hard links when supported (else copies) | **Opt-in only**, same bytes; do not edit source/destination content in-place afterward |
| Profiling | Hard to tell which stage is slow | `--timings` logs wall time per stage | No data-path changes |

The C++ PCL decoder still deserializes a whole protobuf chunk into memory and
its `decodePointCloud` step is not parallelized. Those are possible remaining
bottlenecks, but more invasive changes need real compressed test vectors and
realistic profiling before changing their behavior.

## Measured synthetic benchmarks

**Environment:** test container using `/mnt/data` filesystem, local CPU and
FFmpeg; timings include Python interpreter startup, I/O and subprocess setup.
Values are small-sample averages, not a promise of equivalent speedups on a
user's Ubuntu machine. v0.3 is the baseline archive, v0.4 is the optimized copy.

| Benchmark | v0.3 | v0.4 | Ratio / note |
|---|---:|---:|---|
| Camera, 4 x 160 frames, 640×360, JPEG 95 | 1.883 s | 1.285 s | 1.47× faster with four parallel decoders |
| Raw XYZI, 1,000 scans × 2,000 points (30.54 MiB protobuf chunk) | 1.002 s | 0.943 s | 1.06× faster; peak Python RSS ~159.6 MB → ~97.3 MB |
| Export 300 samples with 4 camera images/sample, ~120 KiB/JPEG | 1.449 s | 0.831 s | 1.74× faster with opt-in hard linking |

Equivalence: all **640 camera JPEGs and `camera_manifest.csv`** compared
byte-for-byte; raw LiDAR manifests and representative binary points compared;
300-frame copy-versus-hardlink exports produced identical JPG/PCD bytes.
The PCL C++ batch writer passed its native standalone byte-level test;
**the complete PCL decode step has not been built or benchmarked here**.

## Run and tune

```bash
# Safe defaults: four independent FFmpeg workers; one decoder thread per worker
recorder-pipeline all BAG CONVERTED \
  --lidar-encoding raw --raw-components 4 \
  --camera-workers 4 --ffmpeg-threads 1 --timings \
  --sample-rate 2 --tolerance-ms 20

# If you explicitly accept shared image inodes, add:
#   --link-images

# Compare throughput on your own machine:
recorder-pipeline camera BAG CAM_OUTPUT --camera-workers 1 --timings
recorder-pipeline camera BAG CAM_OUTPUT_PARALLEL --camera-workers 4 --timings
```

For the last two commands, use separate output directories. Increasing the
number of workers is not always faster: on HDDs, slow network mounts and
low-RAM machines, 2 workers can outperform 4. Avoid oversubscription; start
with `4 × 1` or `2 × 2` for four independent streams, then measure.

## Validation commands

```bash
python -m pytest -q
# This native test does not require PCL or protoc:
g++ -std=c++17 -O2 -Icpp tests/test_xyzi_writer.cpp -o /tmp/test_xyzi_writer
/tmp/test_xyzi_writer

# On a machine with PCL and the Protobuf C++ SDK:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Limitations and safeguards

- The `--sample-rate` option still filters **after** sensor decoding. This is
  deliberate: skipping frames before correct PTS mapping and nearest-neighbor
  synchronization would require an extra validated preselection pass.
  Therefore low-rate datasets still pay the cost of decoding all camera frames
  and point clouds. Sparse extraction could be a separate, optional feature.
- Multiple camera workers use more **aggregate** CPU/RAM than a single worker.
- Hard links are opt-in: modifying the bytes through either path changes the
  other path too. In-place edits must not be done on a linked dataset.
- A process crash can leave orphan payload files: per-file replacement and
  manifest publication are atomic, but the entire pipeline is not a global
  transaction. Prefer clean output directories for production conversions.
- The exact C++ PCL compressed data and proprietary real bag layout require
  validation on the deployment machine. The user-provided protobuf schema does
  not itself identify the compression algorithm or time units.
