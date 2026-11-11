"""Isolated dataset-builder smoke test with minimal substitutes for external mmengine/ocamcamera.

This tests our importer/orchestration without claiming that OCam conversion itself
works: it must still be validated in the real mmdetection3d environment.
"""
import subprocess
import sys
from pathlib import Path


def test_create_chronological_dataset_in_isolated_env(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / 'integration_scripts'
    source = r'''
import sys,types,os,json,csv,pickle
from pathlib import Path
import numpy as np

class OcamCamera:
    def __init__(self, filename, fov):
        self.width=800
        self.height=800
        self.fov=float(fov)

fake_ocam=types.ModuleType('ocamcamera');fake_ocam.OcamCamera=OcamCamera
sys.modules['ocamcamera']=fake_ocam
mmengine=types.ModuleType('mmengine')
def load(path):
    with open(path,'rb') as f:return pickle.load(f)
def dump(obj,path,file_format=None):
    with open(path,'wb') as f:pickle.dump(obj,f)
mmengine.load=load;mmengine.dump=dump
sys.modules['mmengine']=mmengine
sys.path.insert(0,sys.argv[1])
from build_combined_dataset import main
root=Path(sys.argv[2]);dataset=root/'sequence';dataset.mkdir(parents=True)
(dataset/'detzero_output').mkdir()
(directory:=dataset/'fisheye3dod_inference_2hz').mkdir()
(directory/'preparation.json').write_text(json.dumps({'image_mode':'canonical-ocam'}))
(dataset/'calibration.json').write_text(json.dumps({'extrinsics':{'CLIDAR':{'from':'CLIDAR','to':'BASELINK','matrix':np.eye(4).tolist()}}}))
np.save(dataset/'odometry_poses.npy',np.repeat(np.eye(4)[None],8,axis=0))
calibration=root/'calib_results.txt';calibration.write_text('stub 800x800 OCam calibration')
rows=[];frames=[];templates=[]
for i in range(8):
    token=1700000000000000000+i*500000000
    row={'timestamp_ns':str(token),'lidar_sensor_timestamp_ns':str(token),
         'camera_sync_target_ns':str(token),'camera_sync_clock':'sensor','odometry_index':str(i)}
    rows.append(row)
    frames.append({'frame_id':i,'timestamp_ns':token,'lidar_sensor_timestamp_ns':token,
                   'objects':[{'track_id':9,'name':'Car','box':[float(8+i),0,0,4,2,2,0],'score':.8}]})
    info={}
    for camera in ('fisheye_camera_front','fisheye_camera_left','fisheye_camera_right','fisheye_camera_rear'):
        p=f'images/{camera}/{token}.jpg';(directory/p).parent.mkdir(parents=True,exist_ok=True)
        (directory/p).write_bytes(b'fake image data')
        info[camera]={'image_size':(800,800),'cam_path':p,'cam2img':OcamCamera(str(calibration),220.)}
    templates.append({'token':str(token),'cam_info':{'cam_fisheye':info},'annos':{}})
with (dataset/'manifest.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(dataset/'detzero_output/improved_results.json').write_text(json.dumps(frames))
dump({'metainfo':{},'data_list':templates},directory/'fisheye3dod_infos_test.pkl')
output=root/'combined'
main(['--sequence','test',str(dataset),'--output',str(output),
      '--ocam-calibration',str(calibration),'--num-anchors','2',
      '--purge-seconds','0','--val-fraction','.25'])
summary=json.loads((output/'summary.json').read_text())
assert summary['frames']==8 and summary['train_frames']==6 and summary['val_frames']==2
assert summary['split']['clock']=='lidar_sensor_timestamp_ns'
assert summary['rosa_anchor']['shape']==[2,11]
assert (output/'rosa/fisheye3dod_infos_train.pkl').read_bytes()==(output/'fisheye3dod/fisheye3dod_infos_train.pkl').read_bytes()
assert '/.combined.building-' not in (output/'BENCHMARK.md').read_text()
print('SUCCESS',summary['frames'],summary['val_frames'])
'''
    result = subprocess.run([sys.executable, '-c', source, str(scripts), str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + '\n' + result.stdout
    assert 'SUCCESS 8 2' in result.stdout
