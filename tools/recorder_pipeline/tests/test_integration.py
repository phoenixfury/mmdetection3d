"""Local smoke test of the complete raw pipeline; needs ffmpeg and ffprobe."""
import csv
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from recorder_pipeline.cli import main
from recorder_pipeline.config import CAMERA_DIRS, EXTRINSICS_DIR, INTRINSICS_DIR
from recorder_pipeline.protos import bag_pb2, lidar_pb2
from test_core import create_odometry, make_fused, len_field, encode_varint


def num_field(number, value):
    return encode_varint(number << 3) + encode_varint(value)


def packed(number, values):
    return len_field(number,struct.pack('<'+str(len(values))+'d',*values))


def write_topic(folder, payloads, timestamps):
    folder.mkdir(parents=True)
    t=bag_pb2.TopicData()
    for payload in payloads:
        t.samples.add().data=payload
    (folder/'chunk-000.bin').write_bytes(t.SerializeToString())
    (folder/'chunk-000.idx').write_bytes(struct.pack('<'+'Q'*len(timestamps),*timestamps))



@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="needs ffmpeg and ffprobe",
)
def test_cli_all_raw_end_to_end(tmp_path):
    bag = tmp_path / "BAG"
    bag.mkdir()

    # The CLI should automatically derive this directory.
    converted = tmp_path / "BAG_converted"

    times = [1_000_000_000, 1_500_000_000]

    # ---------------------------------------------------------
    # Create four synthetic camera recordings
    # ---------------------------------------------------------
    for camera, dirname in CAMERA_DIRS.items():
        folder = bag / dirname
        folder.mkdir(parents=True)

        movie = folder / "000.h264"

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "lavfi",
                "-i", "testsrc2=size=32x32:rate=2:duration=1",
                "-c:v", "libx264",
                "-bf", "0",
                "-pix_fmt", "yuv420p",
                "-f", "mp4",
                "-y", str(movie),
            ],
            check=True,
        )

        movie.with_suffix(".idx").write_bytes(
            struct.pack("<QQ", *times)
        )

    # ---------------------------------------------------------
    # Create synthetic raw LiDAR recordings
    # ---------------------------------------------------------
    cloud_payloads = [
        lidar_pb2.PointCloud(
            timestamp=timestamp,
            rows=2,
            data=struct.pack(
                "<8f", 1, 2, 3, 4, 5, 6, 7, 8
            ),
        ).SerializeToString()
        for timestamp in times
    ]

    # Recorder timestamp differs from sensor acquisition time.
    write_topic(
        bag / "ouster__points",
        cloud_payloads,
        [timestamp + 5_000_000 for timestamp in times],
    )

    # ---------------------------------------------------------
    # Create calibration records
    # ---------------------------------------------------------
    extrinsics = []
    intrinsics = []

    for i in range(4):
        identity = np.eye(4).reshape(-1)

        extrinsics.append(
            num_field(1, i)
            + num_field(2, 28)
            + packed(3, identity)
        )

        K = np.array(
            [
                [10, 0, 16],
                [0, 10, 16],
                [0, 0, 1],
            ]
        ).reshape(-1)

        intrinsics.append(
            num_field(1, i)
            + packed(2, K)
            + packed(3, [0, 0, 0, 0])
            + num_field(4, 32)
            + num_field(5, 32)
            + len_field(6, b"GENERIC_CAMERA_MODEL")
        )

    write_topic(
        bag / EXTRINSICS_DIR,
        [b"".join(len_field(1, m) for m in extrinsics)],
        [1],
    )

    write_topic(
        bag / INTRINSICS_DIR,
        [b"".join(len_field(1, m) for m in intrinsics)],
        [1],
    )

    # ---------------------------------------------------------
    # Create FusedOdometry in its actual BAG topic directory
    # ---------------------------------------------------------
    fused = (
        bag
        / "__engineering__adcu__debug__ZMQ__Localization__FusedOdometry"
    )

    create_odometry(
        fused,
        [
            make_fused(timestamp, x, 0.1)
            for timestamp, x in [
                (900, 1.0),
                (1100, 2.0),
                (1400, 3.0),
                (1600, 4.0),
            ]
        ],
        [1, 2, 3, 4],
    )

    # ---------------------------------------------------------
    # Execute the pipeline
    #
    # Only the BAG path is supplied.
    # Output and odometry paths must be discovered automatically.
    # ---------------------------------------------------------
    main(
        [
            "all",
            str(bag),
            "--lidar-encoding", "raw",
            "--raw-components", "4",
            "--sample-rate", "2",
            "--tolerance-ms", "20",
        ]
    )

    # ---------------------------------------------------------
    # Verify automatically generated output structure
    # ---------------------------------------------------------
    assert converted.is_dir()

    assert (converted / "calibration.json").is_file()
    assert (converted / "camera_manifest.csv").is_file()
    assert (converted / "lidar_manifest.csv").is_file()
    assert (converted / "synchronization.csv").is_file()

    # Verify synchronized dataset and odometry.
    assert (converted / "dataset" / "odometry.csv").is_file()

    assert (
        converted / "dataset" / "lidar_pcd" / "1005000000.pcd"
    ).is_file()

    # Verify the number and shape of odometry pose matrices.
    poses = np.load(
        converted / "dataset" / "odometry_poses.npy"
    )

    assert poses.shape == (2, 4, 4)

    # Verify synchronization produced two records.
    with (converted / "synchronization.csv").open(
        newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 2

    # Verify both LiDAR acquisition timestamps are preserved.
    assert [
        int(row["lidar_sensor_timestamp_ns"])
        for row in rows
    ] == times
