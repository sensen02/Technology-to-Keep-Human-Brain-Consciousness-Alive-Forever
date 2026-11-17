"""Independent saved112RGB identity/schema/occlusion audit; no Core or rendering."""
from pathlib import Path
import argparse,json,hashlib,os
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no arrays loaded');return
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});import numpy as np
 if (HERE/'VERIFY.json').exists():raise FileExistsError('independent audit alreadyexists')
 read=lambda p:json.loads(Path(p).read_text());m=read(HERE/'LIBRARY.json');pr=read(HERE/'PREREG.json');checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('frameaudit failed '+k)
 check('kind',m['kind']=='VISION_STATIC_HEAD_LIBRARY_V1');check('RGBpath',Path(m['rgb_path']).resolve()==HERE/'RGB.npz');check('RGBfileSHA',sha(m['rgb_path'])==m['rgb_sha256'])
 for p,h in m['source_hashes'].items():check('source:'+p,sha(p)==h)
 check('physical_identity_SHA',sha(m['physical_identity'])==m['physical_identity_sha256']);check('compiled_unchanged',m['compiled_unchanged'] and m['physics_steps']==0 and m['integration_attempts']==0)
 check('eye_order',m['eye_order']==['LEFT','RIGHT'])
 with np.load(m['rgb_path'],allow_pickle=False) as z:
  check('onlyrgb',z.files==['rgb']);rgb=z['rgb'].copy()
 check('112trueRGBpairs',rgb.shape==(112,2,32,32,3) and rgb.dtype==np.uint8);check('rows112',len(m['rows'])==112)
 expected=[(c['config_id'],c['split'],'pose'+str(pid),stim) for c in pr['configs'] for pid in c['pose_ids'] for stim in pr['stimuli']]
 observed=[];occlusion=[]
 for i,row in enumerate(m['rows']):
  check('index'+str(i),row['frame_id']==i);check('pixelSHA'+str(i),hashlib.sha256(rgb[i].tobytes()).hexdigest()==row['rgb_sha256']);observed.append((row['config_id'],row['split'],row['pose_id'],row['stimulus']))
 check('fixed_config_pose_stimulus_order',observed==expected)
 for cfg in pr['configs']:
  for pid in cfg['pose_ids']:
   selected=[r for r in m['rows'] if r['config_id']==cfg['config_id'] and r['pose_id']=='pose'+str(pid)];by={r['stimulus']:r for r in selected};off=rgb[by['off']['frame_id']];wall=rgb[by['occluder_only']['frame_id']]
   for stimulus in ('left','right','center'):
    row=by[stimulus];check(cfg['config_id']+str(pid)+stimulus+'_realvisible',np.any(rgb[row['frame_id']]!=off) and row['target_visible'] is True)
   for stimulus in ('occluded_left','occluded_right'):
    row=by[stimulus];equal=np.array_equal(rgb[row['frame_id']],wall);occlusion.append(equal);check(cfg['config_id']+str(pid)+stimulus+'_label_matches_measured_pixels',row['target_visible']==(not equal))
   check(cfg['config_id']+str(pid)+'_off_labels',by['off']['target_visible'] is False and by['occluder_only']['target_visible'] is False)
 allblocked=all(occlusion);check('occlusion_admission_truthful',m['target_fully_occluded_all']==allblocked and m['invisible_label_training_admitted']==allblocked)
 result=dict(integrity_all_pass=True,all_occlusion_exact=allblocked,invisible_label_training_admitted=allblocked,checks=checks,library_sha256=sha(HERE/'LIBRARY.json'),rgb_sha256=m['rgb_sha256'],source_hashes={str(HERE/'audit.py'):sha(HERE/'audit.py')},Core_constructions=0,physical_steps=0,render_calls=0,policy_model_loads=0,scope='Independent saved trueRGB/schema/hashes/opticalcontrast audit; no bodyvisualbehavior qualification')
 with (HERE/'VERIFY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),all_occlusion_exact=allblocked,physical_steps=0)))
if __name__=='__main__':main()
