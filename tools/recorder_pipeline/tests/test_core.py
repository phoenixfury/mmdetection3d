import csv
import math
import struct

import numpy as np
import pytest

from recorder_pipeline.bag_io import BagFormatError, read_index, parse_fields, iter_record_payloads
from recorder_pipeline.sync import nearest, synchronize
from recorder_pipeline.odometry import (
    FusedOdometrySeries, load_fused_odometry, interpolate_angle, pose_matrix)
from recorder_pipeline.protos import bag_pb2, localization_pb2, lidar_pb2
from recorder_pipeline.lidar import decode_raw


def encode_varint(value):
    buf = bytearray()
    while value >= 128:
        buf.append((value & 127) | 128)
        value >>= 7
    buf.append(value)
    return bytes(buf)


def len_field(number, payload):
    return encode_varint(number << 3 | 2) + encode_varint(len(payload)) + payload


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_fused(timestamp, x, yaw, status=1):
    pose = localization_pb2.EgoPoseFused(timestamp=timestamp, status=status)
    pose.position.x = x
    pose.orientation.yaw = yaw
    pose.velocity.x = 2.
    return pose


def create_odometry(folder, poses, index_times):
    folder.mkdir(parents=True, exist_ok=True)
    topic = bag_pb2.TopicData()
    topic.timestamp.seconds = 100
    for pose in poses:
        sample = topic.samples.add()
        sample.timestamp.seconds = 1_700_000_000
        sample.read_timestamp.seconds = 1_700_000_001
        sample.data = pose.SerializeToString()
    (folder/'part-000.bin').write_bytes(topic.SerializeToString())
    (folder/'part-000.idx').write_bytes(struct.pack('<'+'Q'*len(index_times),*index_times))


def test_wire_envelope_compatible_with_bag_proto(tmp_path):
    payload = len_field(3, b'data')
    record = len_field(2, payload)
    chunk = len_field(1, b'metadata')+len_field(2,record)+len_field(2,record)
    file = tmp_path/'part.bin'
    file.write_bytes(chunk)
    assert list(iter_record_payloads(file)) == [payload,payload]


def test_oversized_varint():
    with pytest.raises(BagFormatError):
        parse_fields(b'\x80'*11)
    with pytest.raises(BagFormatError):
        parse_fields(b'\xff'*9+b'\x02')


def test_index_little_endian(tmp_path):
    path = tmp_path/'data.idx'
    path.write_bytes(struct.pack('<QQ',1,2**50))
    assert read_index(path) == [1,2**50]


def test_nearest_tie_chooses_earlier():
    rows=[{'timestamp_ns':'90'},{'timestamp_ns':'110'}]
    assert nearest(rows,[90,110],100)==(rows[0],-10)


def test_sync_uses_sensor_clock_and_rate(tmp_path):
    write_csv(tmp_path/'lidar_manifest.csv', [
        dict(timestamp_ns=100_000_000, sensor_timestamp_ns=10_000_000, path='lidar/a.bin'),
        dict(timestamp_ns=101_000_000, sensor_timestamp_ns=610_000_000, path='lidar/b.bin'),
    ])
    write_csv(tmp_path/'camera_manifest.csv', [
        dict(sensor=camera,timestamp_ns=t,path=f'images/{camera}/{t}.jpg')
        for camera in ('ARC_F','ARC_R','ARC_SL','ARC_SR')
        for t in (10_000_000, 610_000_000)
    ])
    assert synchronize(tmp_path,tmp_path/'s.csv',tolerance_ms=1,sample_rate=2)==2
    with (tmp_path/'s.csv').open() as file:
        assert len(list(csv.DictReader(file))) == 2
    with pytest.raises(ValueError):
        synchronize(tmp_path,tmp_path/'b.csv',tolerance_ms=1,sample_rate=0,camera_sync_clock='read')


def test_decode_fused_topic_and_interpolate(tmp_path):
    folder = tmp_path/'FusedOdometry'
    create_odometry(folder,[make_fused(1000,1.,3.1),make_fused(1100,3.,-3.1)],
                    [9_000_000_000, 9_000_000_100])
    samples=load_fused_odometry(folder)
    assert samples[0]['measurement_ns'] == 1_000_000_000
    assert samples[0]['sample_timestamp_ns'] == 1_700_000_000_000_000_000
    assert samples[0]['recorder_index_ns'] == 9_000_000_000
    series=FusedOdometrySeries(samples,max_delta_ns=100_000_000)
    middle=series.at(1_050_000_000)
    assert middle['x'] == pytest.approx(2.)
    assert abs(abs(middle['yaw'])-math.pi) < 1.e-4
    assert middle['source']=='fused'
    assert pose_matrix(middle).shape == (4,4)
    with pytest.raises(ValueError,match='outside'):
        series.at(990_000_000)
    with pytest.raises(ValueError,match='outside'):
        series.at(1_110_000_000)


def test_reject_large_pose_gap(tmp_path):
    folder=tmp_path/'FusedOdometry'
    create_odometry(folder,[make_fused(1000,1.,0),make_fused(2000,2.,0)], [1,2])
    series=FusedOdometrySeries(load_fused_odometry(folder),max_delta_ns=200_000_000)
    with pytest.raises(ValueError, match='gap'):
        series.at(1_500_000_000)


def test_duplicate_pose_measurement_latest_record_wins(tmp_path):
    folder = tmp_path/'FusedOdometry'
    create_odometry(folder, [make_fused(1000, 1, 0), make_fused(1000, 2, 0),
                             make_fused(1100, 3, 0)], [10, 20, 30])
    with pytest.warns(RuntimeWarning, match='conflicting pose'):
        samples = load_fused_odometry(folder)
    assert len(samples) == 2
    assert samples[0]['x'] == 2
    assert FusedOdometrySeries(samples).at(1_050_000_000)['x'] == pytest.approx(2.5)


def test_duplicate_identical_measurement_is_collapsed(tmp_path):
    folder = tmp_path/'FusedOdometry'
    create_odometry(folder, [make_fused(1000, 1, 0), make_fused(1000, 1, 0)], [10, 20])
    with pytest.warns(RuntimeWarning, match='0 timestamp groups had conflicting'):
        samples = load_fused_odometry(folder)
    assert len(samples) == 1
    assert samples[0]['recorder_index_ns'] == 20


def test_raw_lidar_xyz_and_intensity(tmp_path):
    bag = tmp_path/'bag'
    folder=bag/'ouster__points'
    folder.mkdir(parents=True)
    t=bag_pb2.TopicData()
    msg=lidar_pb2.PointCloud(timestamp=3000, rows=2, data=struct.pack('<8f',1,2,3,10,4,5,6,20))
    t.samples.add().data = msg.SerializeToString()
    (folder/'000.bin').write_bytes(t.SerializeToString())
    (folder/'000.idx').write_bytes(struct.pack('<Q',4000))
    output=tmp_path/'converted'
    assert decode_raw(bag,output,components=4)==1
    arr=np.fromfile(output/'lidar/4000.bin',dtype='<f4').reshape(-1,4)
    assert arr.tolist()==[[1,2,3,10],[4,5,6,20]]
    with pytest.raises(BagFormatError,match='raw bytes'):
        decode_raw(bag,tmp_path/'wrong',components=3)


def test_raw_xyz_populates_zero_intensity(tmp_path):
    folder=tmp_path/'bag/ouster__points'
    folder.mkdir(parents=True)
    t=bag_pb2.TopicData()
    t.samples.add().data=lidar_pb2.PointCloud(
        timestamp=123,rows=1,data=struct.pack('<3f',1,2,3)).SerializeToString()
    (folder/'x.bin').write_bytes(t.SerializeToString())
    (folder/'x.idx').write_bytes(struct.pack('<Q',500))
    assert decode_raw(tmp_path/'bag', tmp_path/'out', components=3)==1
    assert np.fromfile(tmp_path/'out/lidar/500.bin',dtype='<f4').tolist()==[1,2,3,0]
