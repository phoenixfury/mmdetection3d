"""Check stage ordering and switching between create, append and extend."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration_scripts'))
from pipeline import arguments, commands


def test_one_pass_stages(tmp_path):
    bag = tmp_path / 'BAG_SAMPLE'; bag.mkdir()
    opts = arguments(['--bag', str(bag), '--sequence-name', 'sample',
                      '--mmdet3d-root', str(tmp_path),
                      '--combined-output', str(tmp_path / 'combined'),
                      '--start-at', 'extract'])
    output, data = commands(opts)
    assert [name for name, _, _ in output] == [
        'extract', 'detzero', 'improve', 'visualize', 'prepare', 'dataset', 'audit']
    assert '--camera-sync-clock auto' in ' '.join(output[0][1])
    dataset = next(line for name, line, _ in output if name == 'dataset')
    assert '--source-calibration' in dataset
    assert 'build_combined_dataset.py' in dataset[1]
    assert 'render_combined_projection_audit.py' in output[-1][1][1]


def test_resume_at_visualize(tmp_path):
    opts = arguments(['--start-at', 'visualize', '--stop-after', 'visualize',
                      '--dataset', str(tmp_path / 'converted/dataset')])
    output, data = commands(opts)
    assert [name for name, _, _ in output] == ['visualize']
    assert not data['preflight']['yolo_weights']['required_now']
    assert data['preflight']['improved_results']['required_now']


def test_append_and_extend(tmp_path):
    combined = tmp_path / 'combined'; combined.mkdir()
    args = arguments(['--start-at', 'dataset', '--stop-after', 'dataset',
                      '--sequence-name', 'extra', '--combined-output', str(combined),
                      '--dataset-mode', 'append'])
    cmds, _ = commands(args)
    assert 'append_improved_detzero_rosa_dataset.py' in cmds[-1][1][1]
    args = arguments(['--start-at', 'dataset', '--stop-after', 'dataset',
                      '--sequence-name', 'extra', '--combined-output', str(tmp_path / 'new'),
                      '--base-dataset', str(combined), '--dataset-mode', 'extend'])
    cmds, _ = commands(args)
    assert 'build_chronological_combined_dataset.py' in cmds[-1][1][1]
    assert '--base' in cmds[-1][1]
    with pytest.raises(FileExistsError):
        commands(arguments(['--start-at', 'dataset', '--combined-output', str(combined)]))
