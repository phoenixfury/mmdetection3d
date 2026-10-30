"""Compatibility regressions for Python 3.8 and Ubuntu FFmpeg 4.4."""
from pathlib import Path

import pytest

from recorder_pipeline.outputs import _inside


def test_inside_accepts_relative_path(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    assert _inside(source, "lidar/42.bin") == source / "lidar/42.bin"


@pytest.mark.parametrize("unsafe", ["../escape", "../../escape", "/tmp/escape", "sub/../../../escape"])
def test_inside_rejects_escape(tmp_path, unsafe):
    with pytest.raises(ValueError, match="unsafe"):
        _inside(tmp_path, unsafe)


def test_inside_rejects_symlink_escape(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (inside / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        _inside(inside, "link/file.jpg")
