"""Focused checks for chronological pseudo-label dataset updates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from append_improved_detzero_rosa_dataset import assign_chronological_split


def test_chronological_split_keeps_scenes_and_links_inside_partition():
    samples = []
    for sequence, start in (("a", 1_000_000_000_000),
                            ("b", 2_000_000_000_000)):
        for index in reversed(range(30)):
            samples.append({
                "token": str(start + index * 500_000_000 + 100_000_000),
                "metainfo": {
                    "sequence_id": sequence,
                    "acquisition_timestamp_ns": start + index * 500_000_000,
                },
            })
    train, val, purged = assign_chronological_split(samples, .2, 2.)
    assert purged == 8
    assert len(train) + len(val) + purged == len(samples)
    assert [item["sample_idx"] for item in samples] == list(range(60))
    for partition in (samples, train, val):
        times = [item["metainfo"]["acquisition_timestamp_ns"] for item in partition]
        assert times == sorted(times)
    by_token = {item["token"]: item for item in samples}
    for item in samples:
        meta = item["metainfo"]
        for neighbor in (meta["prev_id"], meta["next_id"]):
            if neighbor:
                other = by_token[neighbor]["metainfo"]
                assert other["scene_name"] == meta["scene_name"]
                assert other["temporal_split"] == meta["temporal_split"]
    for sequence in ("a", "b"):
        sequence_train = [item for item in train if item["metainfo"]["sequence_id"] == sequence]
        sequence_val = [item for item in val if item["metainfo"]["sequence_id"] == sequence]
        gap = (sequence_val[0]["metainfo"]["acquisition_timestamp_ns"] -
               sequence_train[-1]["metainfo"]["acquisition_timestamp_ns"])
        assert gap > 2_000_000_000


def test_split_uses_elapsed_time_and_breaks_recording_gaps():
    # Most frames occur near the beginning: 80% of frames is not 80% of time.
    seconds = [0, .5, 1, 1.5, 2, 2.5, 3, 9, 10, 11]
    samples = [{"token": str(index), "metainfo": {
        "sequence_id": "bag",
        "acquisition_timestamp_ns": 1_000_000_000_000 + round(second * 1e9),
    }} for index, second in enumerate(seconds)]
    train, val, _ = assign_chronological_split(samples, .2, 0.)
    assert [item["token"] for item in val] == ["7", "8", "9"]

    # With a later cutoff the long recording gap lies inside training.
    train, val, _ = assign_chronological_split(samples, .05, 0.)
    assert samples[6]["metainfo"]["next_id"] == ""
    assert samples[7]["metainfo"]["prev_id"] == ""
    assert samples[6]["metainfo"]["scene_name"] != samples[7]["metainfo"]["scene_name"]
