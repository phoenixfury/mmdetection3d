
"""Single command-line interface for the binary bag conversion pipeline.

The BAG path is the only required argument.

Automatically discovered paths:
    Output: BAG_NAME_converted
    Fused:  BAG/__engineering__adcu__debug__ZMQ__Localization__FusedOdometry
    Raw:    BAG/__engineering__adcu__debug__ZMQ__Localization__Odometry

Examples:
    recorder-pipeline all /data/BAG --timings
    recorder-pipeline export /data/BAG --overwrite --timings
    recorder-pipeline camera /data/BAG --camera-workers 4
    recorder-pipeline sync /data/BAG --sample-rate 2
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

from . import calibration, lidar, outputs, sync
from .config import CAMERA_DIRS


FUSED_ODOMETRY_TOPIC = (
    "__engineering__adcu__debug__ZMQ__Localization__FusedOdometry"
)

RAW_ODOMETRY_TOPIC = (
    "__engineering__adcu__debug__ZMQ__Localization__Odometry"
)


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved input/output paths for one recorder BAG."""

    bag: Path
    output: Path
    fused_odometry: Optional[Path]
    raw_odometry: Optional[Path]

    @classmethod
    def from_bag(
        cls,
        bag: Path,
        output: Optional[Path] = None,
        fused_odometry: Optional[Path] = None,
        raw_odometry: Optional[Path] = None,
    ) -> "PipelinePaths":
        bag = bag.expanduser().resolve()

        if not bag.is_dir():
            raise FileNotFoundError(
                "BAG directory does not exist: {}".format(bag)
            )

        if output is None:
            output = bag.with_name("{}_converted".format(bag.name))

        output = output.expanduser().resolve()

        fused = (
            fused_odometry.expanduser().resolve()
            if fused_odometry is not None
            else bag / FUSED_ODOMETRY_TOPIC
        )

        raw = (
            raw_odometry.expanduser().resolve()
            if raw_odometry is not None
            else bag / RAW_ODOMETRY_TOPIC
        )

        # Explicit overrides must exist; automatically discovered
        # directories may be absent in some recordings.
        if fused_odometry is not None and not fused.is_dir():
            raise FileNotFoundError(
                "FusedOdometry directory does not exist: {}".format(fused)
            )

        if raw_odometry is not None and not raw.is_dir():
            raise FileNotFoundError(
                "Odometry directory does not exist: {}".format(raw)
            )

        return cls(
            bag=bag,
            output=output,
            fused_odometry=fused if fused.is_dir() else None,
            raw_odometry=raw if raw.is_dir() else None,
        )


def camera_stage(
    bag: Path,
    output: Path,
    limit: int,
    overwrite: bool,
    image_format: str,
    jpeg_quality: int,
    corrupt_policy: str,
    workers: int = 4,
    ffmpeg_threads: int = 1,
) -> int:
    """Decode camera streams concurrently and write a single manifest."""
    from .camera import decode_camera

    if workers < 1 or ffmpeg_threads < 1:
        raise ValueError(
            "camera workers and FFmpeg threads must be positive"
        )

    args = argparse.Namespace(
        bag=bag,
        output=output,
        cameras=list(CAMERA_DIRS),
        format=image_format,
        jpeg_quality=jpeg_quality,
        limit=limit or None,
        corrupt_policy=corrupt_policy,
        overwrite=overwrite,
        ffmpeg_threads=ffmpeg_threads,
        quiet_progress=(workers > 1),
    )

    manifest = output / "camera_manifest.csv"

    if manifest.exists() and not overwrite:
        raise FileExistsError(
            "{} exists; use --overwrite".format(manifest)
        )

    def decode(sensor: str):
        return decode_camera(args, sensor)

    rows = []

    # Executor.map preserves camera order regardless of completion order.
    with ThreadPoolExecutor(
        max_workers=min(workers, len(CAMERA_DIRS))
    ) as pool:
        for sensor, images in zip(
            CAMERA_DIRS,
            pool.map(decode, CAMERA_DIRS),
        ):
            rows.extend(images)
            print("{}: decoded {} frames".format(sensor, len(images)))

    if not rows:
        raise ValueError("No camera frames decoded")

    temporary = output / "camera_manifest.tmp.csv"

    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)

        os.replace(str(temporary), str(manifest))

    finally:
        if temporary.exists():
            temporary.unlink()

    return len(rows)


def lidar_stage(
    bag: Path,
    output: Path,
    limit: int,
    overwrite: bool,
    encoding: str,
    components: int,
    executable: str,
) -> Optional[int]:
    """Decode raw or PCL-compressed LiDAR recordings."""

    if encoding == "raw":
        return lidar.decode_raw(
            bag,
            output,
            limit,
            overwrite,
            components,
        )

    command = [
        executable,
        str(bag),
        str(output),
    ]

    if limit:
        command.extend(["--limit", str(limit)])

    if overwrite:
        command.append("--overwrite")

    subprocess.run(command, check=True)

    return None


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    commands = (
        "all",
        "camera",
        "lidar",
        "calibration",
        "sync",
        "export",
    )

    for command in commands:
        stage = subparsers.add_parser(
            command,
            help="Run the {} stage".format(command),
        )

        # Only required path.
        stage.add_argument(
            "bag",
            type=Path,
            help="Source BAG directory",
        )

        stage.add_argument(
            "--output",
            type=Path,
            default=None,
            help=(
                "Converted output directory "
                "(default: BAG_NAME_converted)"
            ),
        )

        stage.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Maximum samples (0 = unlimited)",
        )

        stage.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing generated files",
        )

        stage.add_argument(
            "--timings",
            action="store_true",
            help="Print elapsed time for each stage",
        )

        if command in ("all", "camera"):
            stage.add_argument(
                "--image-format",
                choices=("jpg", "png"),
                default="jpg",
            )

            stage.add_argument(
                "--jpeg-quality",
                type=int,
                default=95,
            )

            stage.add_argument(
                "--corrupt-policy",
                choices=("skip", "error"),
                default="skip",
            )

            stage.add_argument(
                "--camera-workers",
                type=int,
                default=4,
                help="Parallel camera decoders",
            )

            stage.add_argument(
                "--ffmpeg-threads",
                type=int,
                default=1,
                help="Decoder threads per FFmpeg process",
            )

        if command in ("all", "lidar"):
            stage.add_argument(
                "--lidar-encoding",
                choices=("pcl", "raw"),
                default="pcl",
                help="PointCloud.data encoding",
            )

            stage.add_argument(
                "--raw-components",
                type=int,
                choices=(3, 4),
                default=4,
            )

            stage.add_argument(
                "--lidar-decoder",
                default="decode_lidar",
                help="Path to compiled C++ LiDAR decoder",
            )

        if command in ("all", "calibration"):
            stage.add_argument(
                "--lidar-to-base",
                type=float,
                nargs=16,
                help="Verified T_BASELINK_CLIDAR (row-major 4x4)",
            )

            stage.add_argument(
                "--camera-pitch",
                action="append",
                default=[],
                help="Camera pitch correction",
            )

        if command in ("all", "sync"):
            stage.add_argument(
                "--tolerance-ms",
                type=float,
                default=20.0,
            )

            stage.add_argument(
                "--sample-rate",
                type=float,
                default=2.0,
            )

            stage.add_argument(
                "--camera-sync-clock",
                choices=("sensor", "read"),
                default="sensor",
            )

        if command in ("all", "export"):
            stage.add_argument(
                "--link-images",
                action="store_true",
                help="Hard-link native JPGs when possible",
            )

            stage.add_argument(
                "--odometry-dir",
                type=Path,
                default=None,
                help="Override automatically discovered FusedOdometry",
            )

            stage.add_argument(
                "--raw-odometry-dir",
                type=Path,
                default=None,
                help="Override automatically discovered raw Odometry",
            )

            stage.add_argument(
                "--proto-dir",
                type=Path,
                default=None,
                help="Optional protobuf schema override",
            )

            stage.add_argument(
                "--dataset-output",
                type=Path,
                default=None,
                help="Dataset directory (default: OUTPUT/dataset)",
            )

            stage.add_argument(
                "--max-odometry-delta-ms",
                type=float,
                default=200.0,
            )

            stage.add_argument(
                "--pose-timestamp-unit",
                choices=("ms", "us", "ns"),
                default="ms",
            )

            stage.add_argument(
                "--missing-odometry-policy",
                choices=("skip", "error"),
                default="skip",
                help=(
                    "Skip synchronized samples in genuine odometry outages "
                    "and record them in skipped_odometry.csv (default: skip)"
                ),
            )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Run one stage or the complete pipeline."""

    args = create_parser().parse_args(argv)

    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")

    jpeg_quality = getattr(args, "jpeg_quality", 95)

    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be 1..100")

    # Resolve all paths in one place.
    paths = PipelinePaths.from_bag(
        bag=args.bag,
        output=args.output,
        fused_odometry=getattr(args, "odometry_dir", None),
        raw_odometry=getattr(args, "raw_odometry_dir", None),
    )

    bag = paths.bag
    output = paths.output

    output.mkdir(parents=True, exist_ok=True)

    print("BAG:     {}".format(bag))
    print("OUTPUT:  {}".format(output))

    if args.command in ("all", "export"):
        print("FUSED:   {}".format(
            paths.fused_odometry or "Not found"
        ))
        print("RAW:     {}".format(
            paths.raw_odometry or "Not found"
        ))

        if paths.fused_odometry is None:
            raise FileNotFoundError(
                "FusedOdometry directory was not found in BAG: {}"
                .format(bag / FUSED_ODOMETRY_TOPIC)
            )

    stage_start = perf_counter()

    # ---------------------------------------------------------
    # Camera decoding
    # ---------------------------------------------------------
    if args.command in ("all", "camera"):
        count = camera_stage(
            bag=bag,
            output=output,
            limit=args.limit,
            overwrite=args.overwrite,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            corrupt_policy=args.corrupt_policy,
            workers=args.camera_workers,
            ffmpeg_threads=args.ffmpeg_threads,
        )

        print("Camera frames: {}".format(count))

        if args.timings:
            print(
                "[timing] camera: {:.3f} s".format(
                    perf_counter() - stage_start
                )
            )

        stage_start = perf_counter()

    # ---------------------------------------------------------
    # LiDAR decoding
    # ---------------------------------------------------------
    if args.command in ("all", "lidar"):
        count = lidar_stage(
            bag=bag,
            output=output,
            limit=args.limit,
            overwrite=args.overwrite,
            encoding=args.lidar_encoding,
            components=args.raw_components,
            executable=args.lidar_decoder,
        )

        if count is not None:
            print("Raw LiDAR point clouds: {}".format(count))

        if args.timings:
            print(
                "[timing] lidar: {:.3f} s".format(
                    perf_counter() - stage_start
                )
            )

        stage_start = perf_counter()

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------
    if args.command in ("all", "calibration"):
        calibration.extract(
            bag,
            output / "calibration.json",
            args.lidar_to_base,
            args.camera_pitch,
        )

        print("Calibration written")

        if args.timings:
            print(
                "[timing] calibration: {:.3f} s".format(
                    perf_counter() - stage_start
                )
            )

        stage_start = perf_counter()

    # ---------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------
    if args.command in ("all", "sync"):
        count = sync.synchronize(
            output,
            output / "synchronization.csv",
            args.tolerance_ms,
            args.sample_rate,
            args.camera_sync_clock,
            args.overwrite,
        )

        print("Synchronized samples: {}".format(count))

        if args.timings:
            print(
                "[timing] sync: {:.3f} s".format(
                    perf_counter() - stage_start
                )
            )

        stage_start = perf_counter()

    # ---------------------------------------------------------
    # Dataset export and odometry
    # ---------------------------------------------------------
    if args.command in ("all", "export"):
        dataset_output = (
            args.dataset_output.expanduser().resolve()
            if args.dataset_output is not None
            else output / "dataset"
        )

        count = outputs.export(
            output,
            dataset_output,
            paths.fused_odometry,
            args.max_odometry_delta_ms,
            args.overwrite,
            args.limit,
            args.pose_timestamp_unit,
            args.link_images,
            paths.raw_odometry,
            args.proto_dir,
            args.missing_odometry_policy,
        )

        print("Exported dataset frames: {}".format(count))

        if args.timings:
            print(
                "[timing] export: {:.3f} s".format(
                    perf_counter() - stage_start
                )
            )


if __name__ == "__main__":
    main()
