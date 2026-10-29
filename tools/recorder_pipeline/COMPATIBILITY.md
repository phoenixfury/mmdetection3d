# v0.4.4 compatibility patch

Fixes failures observed under Ubuntu 22.04, FFmpeg 4.4.2 and Conda Python 3.8:

- Replaces `-fps_mode passthrough` (unsupported by FFmpeg 4.4) with `-vsync 0` for passthrough/no-duplicate frame output. Newer FFmpeg releases may emit a deprecation warning for `-vsync`; this option is retained for 4.4 compatibility.
- Replaces `Path.is_relative_to` (Python >=3.9) with `Path.relative_to` inside try/except, checking resolved paths to protect against symlink escapes.
- Declares Python >=3.8 in package metadata.
- Adds path-security regression tests.

## Verify

```bash
python -m pip install -e '.[test]'
python -m compileall -q src tests
pytest -q
```

Native FFmpeg 4.4 and Python 3.8 execution should also be verified on the target Ubuntu machine.

## v0.4.5 repeated fused odometry measurements

A localization publisher can emit more than one `EgoPoseFused` with the same
`timestamp`. The fused odometry reader now groups samples by measurement time,
retains one row, and warns with the total number of collapsed samples and the
number of groups that disagree in pose or status. For disagreements the row
with the greatest recorder index timestamp is retained; read timestamp breaks
ties, then latest encountered row. This is a data selection policy, not proof
of which pose is physically correct. Review warnings and compare recordings
when they occur. It does not interpolate between duplicate timestamps.
