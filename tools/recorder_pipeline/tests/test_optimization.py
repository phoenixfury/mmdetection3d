"""Optimized code paths must preserve the existing wire and output contracts."""
import csv
import hashlib
import os
import struct
import shutil

import numpy as np
import pytest

from recorder_pipeline.bag_io import BagFormatError, iter_bag_topic_samples
from recorder_pipeline.lidar import decode_raw
from recorder_pipeline.outputs import export
from recorder_pipeline.cli import camera_stage
from recorder_pipeline.protos import bag_pb2, lidar_pb2
from recorder_pipeline.config import CAMERA_DIRS
from test_core import len_field
from test_outputs import create_sync


def test_streaming_topic_retains_wire_order_index_and_unknown_fields(tmp_path):
    folder = tmp_path/'topic'; folder.mkdir()
    expected = [bytes(range(16)), bytes(range(32))]
    topic = bag_pb2.TopicData()
    topic.timestamp.seconds = 42
    for data in expected:
        sample = topic.samples.add()
        sample.data = data
        sample.timestamp.seconds = 123
    # Unknown fields should be skipped (forward-compatible protobuf).
    (folder/'data.bin').write_bytes(topic.SerializeToString() + len_field(99,b'future'))
    (folder/'data.idx').write_bytes(struct.pack('<QQ',800,900))
    decoded = list(iter_bag_topic_samples(folder))
    assert [(timestamp, sample.data, i) for timestamp,sample,_,i in decoded] == [
        (800,expected[0],0), (900,expected[1],1)]
    (folder/'data.idx').write_bytes(struct.pack('<Q',800))
    with pytest.raises(BagFormatError,match='more TopicData samples'):
        list(iter_bag_topic_samples(folder))


def test_raw_xyzi_is_identical_at_bit_level(tmp_path):
    folder = tmp_path/'bag/ouster__points'; folder.mkdir(parents=True)
    # This checks signed zero and a nontrivial intensity bit pattern are not
    # silently changed by a cast or array recreation.
    data = struct.pack('<8f', -0.0, 1.0, 7.5, 123.125, 3.25, 0., 1.e-6, -2.5)
    topic = bag_pb2.TopicData()
    topic.samples.add().data = lidar_pb2.PointCloud(rows=2,timestamp=10,data=data).SerializeToString()
    (folder/'0.bin').write_bytes(topic.SerializeToString())
    (folder/'0.idx').write_bytes(struct.pack('<Q',123))
    assert decode_raw(tmp_path/'bag',tmp_path/'converted') == 1
    assert (tmp_path/'converted/lidar/123.bin').read_bytes() == data
    with (tmp_path/'converted/lidar_manifest.csv').open(newline='') as stream:
        assert list(csv.DictReader(stream))[0]['sensor_timestamp_ns'] == '10'


def test_native_hardlinks_are_opt_in(tmp_path):
    converted = tmp_path/'converted'; converted.mkdir()
    create_sync(converted)
    output = tmp_path/'linked'
    assert export(converted, output, link_images=True) == 1
    src = converted/'arc_f.jpg'; dst = output/'images/arc_f/42.jpg'
    assert src.read_bytes() == dst.read_bytes()
    assert src.stat().st_ino == dst.stat().st_ino
    # Without opt-in files have their own inode.
    assert export(converted,tmp_path/'copied') == 1
    assert src.stat().st_ino != (tmp_path/'copied/images/arc_f/42.jpg').stat().st_ino


@pytest.mark.skipif(not (shutil.which('ffprobe') and shutil.which('ffmpeg')),
                    reason='FFmpeg required')
@pytest.mark.parametrize('b_frames', [0, 2])
def test_camera_workers_keep_manifest_order_and_jpeg_content(tmp_path, b_frames):
    import subprocess
    bag = tmp_path/'BAG'
    timestamps = list(range(1_000, 13_000, 1_000))
    for camera, topic in CAMERA_DIRS.items():
        directory = bag/topic; directory.mkdir(parents=True)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi',
                        '-i','testsrc2=size=64x48:rate=12:duration=1',
                        '-c:v','libx264','-bf',str(b_frames),'-pix_fmt','yuv420p',
                        '-f','mp4','-y',str(directory/'000.h264')],check=True)
        (directory/'000.idx').write_bytes(struct.pack('<'+'Q'*len(timestamps),*timestamps))
    out_seq, out_par = tmp_path/'sequential', tmp_path/'parallel'
    assert camera_stage(bag,out_seq,0,False,'jpg',95,'error',workers=1) == 4*len(timestamps)
    assert camera_stage(bag,out_par,0,False,'jpg',95,'error',workers=4) == 4*len(timestamps)
    assert (out_seq/'camera_manifest.csv').read_bytes() == (out_par/'camera_manifest.csv').read_bytes()
    for cam in CAMERA_DIRS:
        for ts in timestamps:
            rel = f'images/{cam.lower()}/{ts}.jpg'
            assert (out_seq/rel).read_bytes() == (out_par/rel).read_bytes()
