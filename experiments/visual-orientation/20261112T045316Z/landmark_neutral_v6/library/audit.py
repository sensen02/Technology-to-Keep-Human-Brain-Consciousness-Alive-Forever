"""Independentnew672RGB/prefix616/new56actualposeaudit; no model/render/Core."""
from pathlib import Path
import argparse,json,hashlib
from layouts import new_configs
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE.parent;OLD=ROOT/'landmark_coverage_v5/library'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-resource-released',action='store_true');a=ap.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY noarrays');return
 import numpy as np
 m=read(HERE/'LIBRARY.json');old=read(OLD/'LIBRARY.json');pr=read(SCOPE/'PREREG.json');checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('neutralactualRGBAudit '+k)
 check('kind',m['kind']=='VISION_NEUTRAL_HEAD_LIBRARY_V6');check('RGBpath',Path(m['rgb_path']).resolve()==HERE/'RGB.npz');check('RGBSHA',sha(HERE/'RGB.npz')==m['rgb_sha256']);check('oldlibrarySHA',sha(OLD/'LIBRARY.json')==m['parent_library_sha256'])
 for p,h in m['source_hashes'].items():check('source:'+p,sha(p)==h)
 check('physicalidentity',sha(m['physical_identity'])==m['physical_identity_sha256']);check('zero_physics',m['physics_steps']==m['integration_attempts']==m['model_forward_calls']==0 and m['compiled_unchanged'])
 with np.load(HERE/'RGB.npz',allow_pickle=False) as z:check('onlyRGB',z.files==['rgb']);rgb=z['rgb'].copy()
 with np.load(OLD/'RGB.npz',allow_pickle=False) as z:oldrgb=z['rgb'].copy()
 check('672actualshape',rgb.shape==(672,2,32,32,3) and rgb.dtype==np.uint8);check('old616pixelprefixexact',np.array_equal(rgb[:616],oldrgb));check('ordered672rows',len(m['rows'])==672 and [r['frame_id'] for r in m['rows']]==list(range(672)))
 for i,r in enumerate(m['rows']):
  check('rawSHA'+str(i),hashlib.sha256(rgb[i].tobytes()).hexdigest()==r['rgb_sha256']);check('eyeoutside'+str(i),r['eye_outside_boxes'] and r['eye_box_min_clearance']>.005)
  if i<616:check('prefixmetadata'+str(i),r['rgb_sha256']==old['rows'][i]['rgb_sha256'] and r['source_split']==old['rows'][i]['split'] and r['split']==('train' if r['config_id'].startswith('T') else 'development'))
 blocked=old['target_fully_occluded_all'];configs=new_configs(pr)
 for j,cfg in enumerate(configs):
  rows=m['rows'][616+j*7:616+(j+1)*7];label=cfg['config_id']+'_'+cfg['pose_id'];ims={r['stimulus']:rgb[r['frame_id']] for r in rows};check(label+'_sceneorder',[r['stimulus'] for r in rows]==pr['library']['stimuli'])
  for r in rows:
   check(label+r['stimulus']+'_newconfig',r['config_id']==cfg['config_id'] and r['pose_id']==cfg['pose_id'] and r['actual_saved_pose_index']==cfg['saved_pose_index'] and r['split']=='validation')
   if r['stimulus'] in ('left','right','center'):check(label+r['stimulus']+'_contrast',np.any(ims[r['stimulus']]!=ims['off']) and r['target_visible'])
   if r['stimulus'].startswith('occluded_'):
    equal=np.array_equal(ims[r['stimulus']],ims['occluder_only']);blocked=blocked and equal;check(label+r['stimulus']+'_visibilitytruth',r['target_visible']==(not equal));check(label+r['stimulus']+'_sametwoboxes',r['closed_box_centers_xyz']==rows[-1]['closed_box_centers_xyz'])
 check('occlusionadmissiontruth',m['target_fully_occluded_all']==blocked and m['invisible_label_training_admitted']==blocked);check('nodigitalexposures',m['digital_augmented_exposures']==0)
 result=dict(integrity_all_pass=all(checks.values()),checks=checks,all_occlusion_exact=blocked,invisible_label_training_admitted=blocked,library_sha256=sha(HERE/'LIBRARY.json'),rgb_sha256=sha(HERE/'RGB.npz'),source_hashes={str(HERE/'audit.py'):sha(HERE/'audit.py'),str(HERE/'layouts.py'):sha(HERE/'layouts.py')},old616pixelprefixexact=True,new_actual_validation_pairs=56,Core_constructions=0,physical_steps=0,render_calls=0,policy_model_loads=0)
 with (HERE/'VERIFY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),old616exact=True,new56=True,all_occlusion_exact=blocked,physical_steps=0)))
if __name__=='__main__':main()
