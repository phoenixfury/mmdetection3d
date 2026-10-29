"""Regression coverage for the user-supplied adas.common.EgoPose mapping."""
import math
import struct

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from recorder_pipeline.bag_io import BagFormatError
from recorder_pipeline.odometry import AlignedOdometry, FusedOdometrySeries
from recorder_pipeline.raw_odometry import load_raw_odometry
from recorder_pipeline.protos import bag_pb2


def raw_message_class():
    """Build the exact relevant common.proto wire schema without needing protoc."""
    f = descriptor_pb2.FileDescriptorProto()
    f.name = 'common.proto'
    f.package = 'adas.common'
    f.syntax = 'proto3'

    def field(msg, name, number, kind, type_name=None):
        v = msg.field.add()
        v.name, v.number, v.label, v.type = name, number, 1, kind
        if type_name:
            v.type_name = type_name

    point = f.message_type.add()
    point.name = 'Point2D'
    field(point, 'x', 1, 1)
    field(point, 'y', 2, 1)
    pose = f.message_type.add()
    pose.name = 'Pose2D'
    field(pose, 'position', 1, 11, '.adas.common.Point2D')
    field(pose, 'orientation', 2, 1)
    raw = f.message_type.add()
    raw.name = 'EgoPose'
    field(raw, 'status', 1, 14, '.adas.common.AvailabilityStatus')
    field(raw, 'timestamp', 2, 3)
    field(raw, 'pose', 3, 11, '.adas.common.Pose2D')
    status = f.enum_type.add()
    status.name = 'AvailabilityStatus'
    for name, number in [('kNotAvailable', 0), ('kAvailable', 1), ('kAvailableReduced', 2)]:
        value = status.value.add()
        value.name, value.number = name, number
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(f.SerializeToString())
    descriptor = pool.FindMessageTypeByName('adas.common.EgoPose')
    return message_factory.GetMessageClass(descriptor)


def write_raw_chunk(folder, cls, specs):
    folder.mkdir(parents=True)
    topic = bag_pb2.TopicData()
    for ts, x, yaw, status in specs:
        msg = cls(timestamp=ts, status=status)
        msg.pose.position.x = x
        msg.pose.position.y = 0.
        msg.pose.orientation = yaw
        sample = topic.samples.add()
        sample.data = msg.SerializeToString()
    (folder/'chunk.bin').write_bytes(topic.SerializeToString())
    (folder/'chunk.idx').write_bytes(struct.pack('<' + 'Q'*len(specs), *range(1, len(specs)+1)))


def fused_row(ns, x):
    return dict(measurement_ns=ns, recorder_index_ns=ns, read_timestamp_ns=None,
                x=x, y=0., z=2., roll=0., pitch=0., yaw=0., vx=0., vy=0.,
                vz=0., yaw_rate=0., status=1)


def test_real_common_proto_fields_and_status(tmp_path, monkeypatch):
    from recorder_pipeline import raw_odometry
    cls = raw_message_class()
    monkeypatch.setattr(raw_odometry, 'load_message_class', lambda proto_dir: cls)
    folder = tmp_path/'Odometry'
    write_raw_chunk(folder, cls, [
        (1000, 99., 0., 0),  # must not become an alignment anchor
        (1100, 1., .2, 1),
        (1200, 2., .4, 2),
    ])
    poses = load_raw_odometry(folder)
    assert [r['measurement_ns'] for r in poses] == [1_100_000_000, 1_200_000_000]
    assert poses[0]['x'] == 1.
    assert poses[0]['yaw'] == pytest.approx(.2)
    assert poses[0]['status'] == 1
    assert poses[1]['status'] == 2
    fused = FusedOdometrySeries([fused_row(1_100_000_000, 10.)], max_delta_ns=120_000_000)
    raw = FusedOdometrySeries(poses, max_delta_ns=120_000_000)
    aligned = AlignedOdometry(fused, raw).at(1_200_000_000)
    assert aligned['source'] == 'aligned_raw'
    assert aligned['z'] == pytest.approx(2.)
    assert aligned['x'] == pytest.approx(10. + math.cos(-.2))


def test_all_raw_measurements_not_available(tmp_path, monkeypatch):
    from recorder_pipeline import raw_odometry
    cls = raw_message_class()
    monkeypatch.setattr(raw_odometry, 'load_message_class', lambda proto_dir: cls)
    folder = tmp_path/'Odometry'
    write_raw_chunk(folder, cls, [(1000, 5., 0., 0)])
    with pytest.raises(BagFormatError, match='no available raw odometry'):
        load_raw_odometry(folder)


def test_bundled_schema_is_packaged():
    from recorder_pipeline import raw_odometry
    schema = raw_odometry.Path(raw_odometry.__file__).resolve().parent / 'schemas/common.proto'
    source = schema.read_text()
    assert 'package adas.common;' in source
    assert 'message EgoPose' in source
