"""Independentactual616RGB/posecoverage proof; trainingfielddiagnostic only,no model."""
from pathlib import Path
import argparse,json,hashlib
from layouts import library_configs
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-resource-released',action='store_true');a=ap.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no arrays');return
 import numpy as np
 from PIL import Image,ImageDraw
 m=read(HERE/'LIBRARY.json');pr=read(SCOPE/'PREREG.json');cfgs=library_configs(pr);checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('coverageactualRGBaudit '+k)
 check('kind',m['kind']=='VISION_COVERAGE_HEAD_LIBRARY_V5');check('path',Path(m['rgb_path']).resolve()==HERE/'RGB.npz');check('RGBSHA',sha(HERE/'RGB.npz')==m['rgb_sha256'])
 for p,h in m['source_hashes'].items():check('source:'+p,sha(p)==h)
 check('physicalidentity',sha(m['physical_identity'])==m['physical_identity_sha256']);check('zero_physics',m['physics_steps']==m['integration_attempts']==m['model_forward_calls']==0 and m['compiled_unchanged'])
 with np.load(HERE/'RGB.npz',allow_pickle=False) as z:check('onlyRGB',z.files==['rgb']);rgb=z['rgb'].copy()
 check('616actualshape',rgb.shape==(616,2,32,32,3) and rgb.dtype==np.uint8);check('orderedrows616',len(m['rows'])==616 and [r['frame_id'] for r in m['rows']]==list(range(616)))
 check('trainvalindexdisjoint',not set(pr['library']['train_pose_indices'])&set(pr['library']['validation_pose_indices']));blocked=True;trainstats=[];samples=[]
 for j,cfg in enumerate(cfgs):
  rows=m['rows'][j*7:(j+1)*7];label=cfg['config_id']+'_'+cfg['pose_id'];ims={r['stimulus']:rgb[r['frame_id']] for r in rows};check(label+'_stimulusorder',[r['stimulus'] for r in rows]==pr['library']['stimuli'])
  for r in rows:
   k=label+'_'+r['stimulus'];check(k+'_identity',r['config_id']==cfg['config_id'] and r['pose_id']==cfg['pose_id'] and r['split']==cfg['split'] and r['actual_saved_pose_index']==cfg['saved_pose_index']);check(k+'_RGBSHA',hashlib.sha256(rgb[r['frame_id']].tobytes()).hexdigest()==r['rgb_sha256']);check(k+'_outside',r['eye_outside_boxes'] and r['eye_box_min_clearance']>.005)
   if r['stimulus'] in ('left','right','center'):check(k+'_visiblecontrast',np.any(ims[r['stimulus']]!=ims['off']) and r['target_visible'])
   if r['stimulus'].startswith('occluded_'):
    equal=np.array_equal(ims[r['stimulus']],ims['occluder_only']);blocked=blocked and equal;check(k+'_visibilitytruth',r['target_visible']==(not equal));check(k+'_samefixedboxes',r['closed_box_centers_xyz']==rows[-1]['closed_box_centers_xyz'])
  if cfg['split']=='train':
   trainstats.append(dict(config_id=cfg['config_id'],pose_id=cfg['pose_id'],saved_pose_index=cfg['saved_pose_index'],raw_scene_hashes={r['stimulus']:r['rgb_sha256'] for r in rows},frame_std={s:float(ims[s].astype(float).std()) for s in ims},visible_contrast_MAE={s:float(np.abs(ims[s].astype(float)-ims['off']).mean()) for s in ('left','right','center')},eye_min_clearance=min(r['eye_box_min_clearance'] for r in rows)))
  # Only a predeclared subset of training poses displayed in compact panel; validation raw images remain saved.
  if cfg['split']=='train' and cfg['pose_id'] in ('trainpose0','trainpose5','trainpose10','trainpose15','trainpose19'):samples.append((label,np.concatenate([np.concatenate((ims[s][0],ims[s][1]),1) for s in pr['library']['stimuli']],1)))
 check('truthfulocclusionadmission',m['invisible_label_training_admitted']==blocked and m['target_fully_occluded_all']==blocked)
 diagnostic=dict(scope='TRAIN_RGB_FIELD_HASH_CONTRAST_ONLY_NO_POLICYFORWARD',training_actual_configs=80,training_actual_frames=560,training_saved_pose_count=20,validation_actual_frames=56,same_underlying_neutral_trajectory=True,not_independent_motion_trajectories=True,unique_training_raw_hashes=len({h for row in trainstats for h in row['raw_scene_hashes'].values()}),unique_training_background_hashes=len({row['raw_scene_hashes']['off'] for row in trainstats}),rows=trainstats,model_forward_calls=0,physical_steps=0)
 with (HERE/'TRAIN_COVERAGE_DIAGNOSTIC.json').open('x') as f:json.dump(diagnostic,f,indent=2);f.write('\n')
 panel=Image.new('RGB',(896,100*len(samples)),'white');draw=ImageDraw.Draw(panel)
 for j,(name,img) in enumerate(samples):draw.text((4,j*100),name+' TRAIN ONLY:off/center/left/right/occL/occR/box LEFT|RIGHT',fill='black');panel.paste(Image.fromarray(img).resize((896,72),Image.Resampling.NEAREST),(0,j*100+20))
 panel.save(HERE/'TRAIN_COVERAGE_PANEL.png')
 result=dict(integrity_all_pass=all(checks.values()),checks=checks,all_occlusion_exact=blocked,invisible_label_training_admitted=blocked,library_sha256=sha(HERE/'LIBRARY.json'),rgb_sha256=sha(HERE/'RGB.npz'),source_hashes={str(HERE/'audit.py'):sha(HERE/'audit.py'),str(HERE/'layouts.py'):sha(HERE/'layouts.py')},training_coverage_diagnostic=str(HERE/'TRAIN_COVERAGE_DIAGNOSTIC.json'),training_coverage_diagnostic_sha256=sha(HERE/'TRAIN_COVERAGE_DIAGNOSTIC.json'),physical_steps=0,Core_constructions=0,render_calls=0,policy_model_loads=0,training_policy_forward=0,validation_policy_forward=0)
 with (HERE/'VERIFY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),all_occlusion_exact=blocked,train_actual560=True,val_actual56=True,unique_train_hashes=diagnostic['unique_training_raw_hashes'],unique_train_backgrounds=diagnostic['unique_training_background_hashes'],physical_steps=0)))
if __name__=='__main__':main()
