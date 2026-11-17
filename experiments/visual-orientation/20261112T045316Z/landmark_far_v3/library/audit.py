"""Parent-only independent saved42RGB/source/closed-box optical audit; no Core."""
from pathlib import Path
import argparse,json,hashlib,os
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-resource-released',action='store_true');a=ap.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no arrays');return
 import numpy as np
 m=read(HERE/'LIBRARY.json');pr=read(SCOPE/'PREREG.json');checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('far library independent audit '+k)
 check('kind',m['kind']=='VISION_FAR_HEAD_LIBRARY_V3');check('RGBpath',Path(m['rgb_path']).resolve()==HERE/'RGB.npz');check('RGBsha',sha(HERE/'RGB.npz')==m['rgb_sha256'])
 for p,h in m['source_hashes'].items():check('source:'+p,sha(p)==h)
 check('physical_identity_sha',sha(m['physical_identity'])==m['physical_identity_sha256']);check('zero_physics',m['physics_steps']==m['integration_attempts']==m['model_forward_calls']==0 and m['compiled_unchanged'] is True)
 with np.load(HERE/'RGB.npz',allow_pickle=False) as z:
  check('onlyRGBkey',z.files==['rgb']);rgb=z['rgb'].copy()
 check('shape_dtype',rgb.shape==(42,2,32,32,3) and rgb.dtype==np.uint8);check('rows42',len(m['rows'])==42 and [r['frame_id'] for r in m['rows']]==list(range(42)));check('eye_order',m['eye_order']==['LEFT','RIGHT'])
 blocked=True
 for j,cfg in enumerate(pr['library']['configs']):
  rows=m['rows'][j*7:(j+1)*7];case={r['stimulus']:rgb[r['frame_id']] for r in rows}
  check(cfg['config_id']+'_sceneorder',[r['stimulus'] for r in rows]==pr['library']['stimuli'])
  for row in rows:
   label=cfg['config_id']+'_'+row['stimulus'];check(label+'_configuration',row['config_id']==cfg['config_id'] and row['split']==cfg['split'] and row['pose_id']==cfg['pose_id'] and row['actual_saved_pose_index']==cfg['saved_pose_index']);check(label+'_RGBrawsha',hashlib.sha256(rgb[row['frame_id']].tobytes()).hexdigest()==row['rgb_sha256']);check(label+'_headpose',np.asarray(row['actual_head_position']).shape==(3,) and np.asarray(row['actual_head_rotation']).shape==(3,3));check(label+'_eyeoutside',row['eye_outside_boxes'] is True and row['eye_box_min_clearance']>.005)
   if row['stimulus'] in ('left','right','center'):
    check(label+'_actualvisiblecontrast',np.any(case[row['stimulus']]!=case['off']) and row['target_visible'] is True)
   if row['stimulus'] in ('left','right'):check(label+'_bearingsign',(row['bearing_rad']>0)==(row['stimulus']=='left'))
   if row['stimulus'] in ('occluded_left','occluded_right'):
    actual=np.array_equal(case[row['stimulus']],case['occluder_only']);blocked=blocked and actual;check(label+'_visibility_truth',row['target_visible']==(not actual));check(label+'_bothfixedboxes',len(row['closed_box_centers_xyz'])==2 and row['closed_box_centers_xyz']==rows[-1]['closed_box_centers_xyz'])
  check(cfg['config_id']+'_box_labels',not rows[0]['target_visible'] and not rows[-1]['target_visible'])
 check('admission_truth',m['invisible_label_training_admitted']==blocked and m['target_fully_occluded_all']==blocked)
 result=dict(integrity_all_pass=all(checks.values()),checks=checks,all_occlusion_exact=blocked,invisible_label_training_admitted=blocked,library_sha256=sha(HERE/'LIBRARY.json'),rgb_sha256=sha(HERE/'RGB.npz'),source_hashes={str(HERE/'audit.py'):sha(HERE/'audit.py')},physical_steps=0,Core_constructions=0,render_calls=0,scope='Independent42faractualRGB and closedbox optical proof, not behavior/body qualification')
 with (HERE/'VERIFY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),all_occlusion_exact=blocked,physical_steps=0)))
if __name__=='__main__':main()
