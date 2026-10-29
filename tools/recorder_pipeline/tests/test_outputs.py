import csv
import struct
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from recorder_pipeline.outputs import export
from test_core import create_odometry, make_fused


def create_sync(converted, lidar_sensor_ns=1_050_000_000):
    (converted/'lidar').mkdir(parents=True)
    (converted/'lidar/42.bin').write_bytes(struct.pack('<4f',1,2,3,10))
    row={'timestamp_ns':'42','lidar_sensor_timestamp_ns':str(lidar_sensor_ns),
         'lidar_path':'lidar/42.bin'}
    for camera in ('arc_f','arc_r','arc_sl','arc_sr'):
        image=converted/f'{camera}.jpg'
        Image.new('RGB',(4,4),'red').save(image)
        row[f'{camera}_path']=image.name
        row[f'{camera}_delta_ns']='2'
    with (converted/'synchronization.csv').open('w', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(row))
        writer.writeheader();writer.writerow(row)
    return row


def test_export_with_protobuf_odometry(tmp_path):
    converted=tmp_path/'converted'
    converted.mkdir()
    create_sync(converted)
    fused=tmp_path/'bag/FusedOdometry'
    create_odometry(fused,[make_fused(1000,0,0), make_fused(1100,10,0)], [5,6])
    output=tmp_path/'dataset'
    assert export(converted,output,odometry_dir=fused)==1
    pcd=(output/'lidar_pcd/42.pcd').read_bytes()
    assert b'DATA binary\n' in pcd
    assert pcd.endswith(struct.pack('<4f',1,2,3,10))
    assert (output/'images/arc_f/42.jpg').is_file()
    assert np.load(output/'odometry_poses.npy')[0,0,3] == pytest.approx(5)
    assert (output/'odometry.csv').is_file()
    assert (output/'manifest.csv').is_file()
    with pytest.raises(FileExistsError):
        export(converted,output)


def test_export_rejects_unsafe_path(tmp_path):
    converted=tmp_path/'converted';converted.mkdir()
    row=create_sync(converted)
    row['arc_f_path']='../../outside.jpg'
    with (converted/'synchronization.csv').open('w', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(row))
        writer.writeheader();writer.writerow(row)
    with pytest.raises(ValueError,match='unsafe'):
        export(converted,tmp_path/'dataset')


def test_export_skip_policy_rejects_when_every_pose_is_missing(tmp_path):
    converted = tmp_path/'converted'
    converted.mkdir()
    create_sync(converted, lidar_sensor_ns=2_000_000_000)
    fused = tmp_path/'bag/FusedOdometry'
    create_odometry(fused, [make_fused(1000, 0, 0),
                            make_fused(1100, 10, 0)], [5, 6])
    output = tmp_path/'dataset'
    with pytest.raises(ValueError, match='no exportable'):
        export(converted, output, odometry_dir=fused,
               missing_odometry_policy='skip')


def test_export_missing_odometry_is_strict_by_default(tmp_path):
    converted = tmp_path/'converted'
    converted.mkdir()
    create_sync(converted, lidar_sensor_ns=2_000_000_000)
    fused = tmp_path/'bag/FusedOdometry'
    create_odometry(fused, [make_fused(1000, 0, 0),
                            make_fused(1100, 10, 0)], [5, 6])
    with pytest.raises(ValueError, match='outside'):
        export(converted, tmp_path/'dataset', odometry_dir=fused)
