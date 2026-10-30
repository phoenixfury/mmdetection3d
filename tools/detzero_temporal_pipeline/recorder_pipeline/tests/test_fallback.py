import math
import pytest
from recorder_pipeline.odometry import AlignedOdometry, FusedOdometrySeries


def row(ns, x, y=0., yaw=0.):
    return dict(measurement_ns=ns, recorder_index_ns=ns,
                read_timestamp_ns=None, x=x, y=y, z=0.,
                roll=0., pitch=0., yaw=yaw, vx=0., vy=0., vz=0.,
                yaw_rate=0., status=1)


def test_raw_fallback_before_and_after_coverage_and_internal_gap():
    fused = FusedOdometrySeries([row(1000, 100.), row(1100, 101.),
                                 row(5000, 140.), row(5100, 141.)], max_delta_ns=100)
    raw = FusedOdometrySeries([row(t, t/100.) for t in range(0, 6100, 100)], max_delta_ns=100)
    aligned = AlignedOdometry(fused, raw)
    assert aligned.at(1000)['source'] == 'fused'
    assert aligned.at(500)['source'] == 'aligned_raw'
    assert aligned.at(1500)['source'] == 'aligned_raw'
    assert aligned.at(5500)['source'] == 'aligned_raw'
    assert aligned.at(1500)['x'] == pytest.approx(105.)


def test_no_raw_coverage_or_anchor_fails_explicitly():
    fused = FusedOdometrySeries([row(1000, 1.), row(1100, 2.)], max_delta_ns=100)
    raw = FusedOdometrySeries([row(3000, 3.), row(3100, 4.)], max_delta_ns=100)
    with pytest.raises(ValueError, match='no shared raw/fused anchor'):
        AlignedOdometry(fused, raw).at(3050)


def test_yaw_and_translation_alignment():
    fused = FusedOdometrySeries([row(1000, 10., 20., math.pi/2)], max_delta_ns=200)
    raw = FusedOdometrySeries([row(1000, 0., 0., 0.), row(1100, 1., 0., 0.)], max_delta_ns=200)
    result = AlignedOdometry(fused, raw).at(1100)
    assert result['x'] == pytest.approx(10.)
    assert result['y'] == pytest.approx(21.)
    assert result['yaw'] == pytest.approx(math.pi/2)


def test_bounded_raw_hold_at_start_of_valid_segment():
    # A stale raw sample can precede the real stream by hours.  A target just
    # before the first current sample has only one useful bracket and should
    # be held only when that current sample is within the configured bound.
    fused = FusedOdometrySeries([row(1000, 100.), row(1100, 101.)],
                                max_delta_ns=100)
    raw = FusedOdometrySeries([row(0, -10.), row(900, 9.), row(1000, 10.),
                               row(1100, 11.)], max_delta_ns=100)
    result = AlignedOdometry(fused, raw).at(850)
    assert result['source'] == 'aligned_raw_boundary_hold'
    assert result['x'] == pytest.approx(99.)


def test_bounded_raw_hold_still_rejects_distant_measurement():
    fused = FusedOdometrySeries([row(1000, 100.), row(1100, 101.)],
                                max_delta_ns=100)
    raw = FusedOdometrySeries([row(0, -10.), row(900, 9.), row(1000, 10.)],
                              max_delta_ns=100)
    with pytest.raises(ValueError, match='bounded hold unavailable'):
        AlignedOdometry(fused, raw).at(750)
