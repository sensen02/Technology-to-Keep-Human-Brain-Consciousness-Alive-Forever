"""Independent newprocess56actualRGB/prefix42/new14closedbox audit; no renderer."""
from pathlib import Path
import argparse,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'landmark_far_v3/library'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-resource-released',action='store_true');a=ap.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no arrays');return
 import numpy as np
 m=read(HERE/'LIBRARY.json');checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('mirror library actual audit '+k)
 check('kind',m['kind']=='VISION_FAR_HEAD_LIBRARY_V4');check('RGBpath',Path(m['rgb_path']).resolve()==HERE/'RGB.npz');check('RGBsha',sha(HERE/'RGB.npz')==m['rgb_sha256']);check('old_library_sha',sha(OLD/'LIBRARY.json')==m['parent_library_sha256'])
 for p,h in m['source_hashes'].items():check('source:'+p,sha(p)==h)
 check('physical_identity',sha(m['physical_identity'])==m['physical_identity_sha256']);check('zero_steps',m['physics_steps']==m['integration_attempts']==m['model_forward_calls']==0 and m['compiled_unchanged'])
 with np.load(HERE/'RGB.npz',allow_pickle=False) as z:check('onlyRGB',z.files==['rgb']);rgb=z['rgb'].copy()
 with np.load(OLD/'RGB.npz',allow_pickle=False) as z:oldrgb=z['rgb'].copy()
 old=read(OLD/'LIBRARY.json');check('56shape',rgb.shape==(56,2,32,32,3) and rgb.dtype==np.uint8);check('prefix42exact',np.array_equal(rgb[:42],oldrgb));check('frameID56',len(m['rows'])==56 and [r['frame_id'] for r in m['rows']]==list(range(56)))
 for j,row in enumerate(m['rows']):
  check('rowsha'+str(j),hashlib.sha256(rgb[j].tobytes()).hexdigest()==row['rgb_sha256']);check('eyesoutside'+str(j),row['eye_outside_boxes'] and row['eye_box_min_clearance']>.005)
  if j<42:
   check('oldmetadata'+str(j),row['rgb_sha256']==old['rows'][j]['rgb_sha256'] and row['source_split']==old['rows'][j]['split'] and row['split']==('train' if row['config_id'].startswith('T') else 'development'))
 blocked=True
 for cfg in ('T0','T1','T2','T3','V4','V5','V6','V7'):
  rows=[r for r in m['rows'] if r['config_id']==cfg];check(cfg+'_sevenactual',len(rows)==7);ims={r['stimulus']:rgb[r['frame_id']] for r in rows}
  for scene in ('left','right','center'):check(cfg+scene+'_contrast',np.any(ims[scene]!=ims['off']))
  for r in rows:
   if r['stimulus'].startswith('occluded_'):
    equal=np.array_equal(ims[r['stimulus']],ims['occluder_only']);blocked=blocked and equal;check(cfg+r['stimulus']+'_truth',r['target_visible']==(not equal));check(cfg+r['stimulus']+'_samebothboxes',r['closed_box_centers_xyz']==rows[-1]['closed_box_centers_xyz'])
   if cfg in ('V6','V7'):check(cfg+r['stimulus']+'_newconfig',r['split']=='validation' and r['pose_id']==('pose9' if cfg=='V6' else 'pose10') and r['actual_saved_pose_index']==(8599 if cfg=='V6' else 11999))
 check('admissiontruth',m['target_fully_occluded_all']==blocked and m['invisible_label_training_admitted']==blocked);check('no_augmented_exposures',m['digital_augmented_exposures']==0)
 result=dict(integrity_all_pass=all(checks.values()),checks=checks,all_occlusion_exact=blocked,invisible_label_training_admitted=blocked,library_sha256=sha(HERE/'LIBRARY.json'),rgb_sha256=sha(HERE/'RGB.npz'),source_hashes={str(HERE/'audit.py'):sha(HERE/'audit.py')},physical_steps=0,Core_constructions=0,render_calls=0,old42pixelprefixexact=True,raw_actual_pairs=56,digital_augmented_exposures=0)
 with (HERE/'VERIFY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),old42exact=True,new14=True,all_occlusion_exact=blocked,physical_steps=0)))
if __name__=='__main__':main()
