"""Independent126savedRGB audit; proves exactold112prefix andnew14scene identities, no render."""
from pathlib import Path
import argparse,json,hashlib,os
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'frame_library_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--parent-resource-released',action='store_true');args=parser.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no arrayreads');return
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});import numpy as np
 if (HERE/'VERIFY.json').exists():raise FileExistsError('do not overwriteactualaudit')
 m=read(HERE/'LIBRARY.json');pr=read(HERE/'PREREG.json');old=read(OLD/'LIBRARY.json');oa=read(OLD/'VERIFY.json');checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('newRGBaudit '+k)
 check('newkind',m['kind']=='VISION_STATIC_HEAD_LIBRARY_V2');check('actualRGBpath',Path(m['rgb_path']).resolve()==HERE/'RGB.npz');check('actualRGBsha',sha(m['rgb_path'])==m['rgb_sha256']);check('oldindependentproof',oa['integrity_all_pass'] is True and all(v is True for v in oa['checks'].values()) and oa['library_sha256']==sha(OLD/'LIBRARY.json') and oa['rgb_sha256']==sha(OLD/'RGB.npz'))
 for p,h in m['source_hashes'].items():check('source:'+p,sha(p)==h)
 check('parent_library_bound',m['parent_library_sha256']==sha(OLD/'LIBRARY.json') and m['parent_RGB_sha256']==sha(OLD/'RGB.npz'))
 check('physicalidentity_bound',sha(m['physical_identity'])==m['physical_identity_sha256']);check('compiledphysics_unchanged_nointegration',m['compiled_unchanged'] is True and m['physics_steps']==0 and m['integration_attempts']==0)
 with np.load(m['rgb_path'],allow_pickle=False) as z:check('onlyRGBkey',z.files==['rgb']);rgb=z['rgb'].copy()
 with np.load(OLD/'RGB.npz',allow_pickle=False) as z:prefix=z['rgb'].copy()
 check('126trueRGBpairs',rgb.shape==(126,2,32,32,3) and rgb.dtype==np.uint8 and len(m['rows'])==126);check('old112pixelprefixexact',np.array_equal(rgb[:112],prefix) and m['old112pixelprefixexact'] is True);check('fixedeye_order',m['eye_order']==['LEFT','RIGHT'])
 changed={'split','selected_for_policy_sequences','unused_by_policytraining'}
 for i,row in enumerate(m['rows']):
  check('row'+str(i)+'_idSHA',row['frame_id']==i and hashlib.sha256(rgb[i].tobytes()).hexdigest()==row['rgb_sha256'])
  if i<112:
   prior=old['rows'][i];check('oldrow'+str(i)+'_metadataretained',all(row[k]==v for k,v in prior.items() if k not in changed));check('oldrow'+str(i)+'_splitversioned',row['source_split']==prior['split'] and row['split']==('train' if prior['config_id'].startswith('T') else 'development'))
 expected=[(cfg['config_id'],cfg['pose_id'],stim) for cfg in pr['new_configs'] for stim in pr['stimuli']];check('new14exactorder',[(r['config_id'],r['pose_id'],r['stimulus']) for r in m['rows'][112:]]==expected)
 newblocked=[]
 for cfg in pr['new_configs']:
  rows=[row for row in m['rows'][112:] if row['config_id']==cfg['config_id']];by={r['stimulus']:r for r in rows}
  check(cfg['config_id']+'_heldoutpose',all(r['split']=='validation' and r['pose_id']==cfg['pose_id'] and r['actual_saved_pose_index']==cfg['saved_pose_index'] and np.asarray(r['actual_head_position']).shape==(3,) and np.asarray(r['actual_head_rotation']).shape==(3,3) for r in rows))
  for stimulus in ('left','right','center'):
   r=by[stimulus];check(cfg['config_id']+stimulus+'_realvisible',r['target_visible'] is True and np.any(rgb[r['frame_id']]!=rgb[by['off']['frame_id']]));sign=1 if stimulus=='left' else -1 if stimulus=='right' else 0;check(cfg['config_id']+stimulus+'_environmentlabel',r['bearing_rad']==float(np.arctan2(sign*cfg['side'],cfg['forward'])))
  for stimulus in ('occluded_left','occluded_right'):
   r=by[stimulus];equal=np.array_equal(rgb[r['frame_id']],rgb[by['occluder_only']['frame_id']]);newblocked.append(equal);check(cfg['config_id']+stimulus+'_actualpixelvisibilitylabel',r['target_visible']==(not equal))
  check(cfg['config_id']+'_nontargetlabels',by['off']['target_visible'] is False and by['occluder_only']['target_visible'] is False)
 blocked=old['target_fully_occluded_all'] and all(newblocked);check('invisible_training_admission_truthful',m['target_fully_occluded_all']==blocked and m['invisible_label_training_admitted']==blocked)
 result=dict(integrity_all_pass=True,checks=checks,library_sha256=sha(HERE/'LIBRARY.json'),rgb_sha256=m['rgb_sha256'],old112pixelprefixexact=True,new14actual_frames=True,all_occlusion_exact=blocked,invisible_label_training_admitted=blocked,source_hashes={str(HERE/'audit.py'):sha(HERE/'audit.py'),str(OLD/'VERIFY.json'):sha(OLD/'VERIFY.json')},Core_constructions=0,render_calls=0,physics_steps=0,policy_model_loads=0,scope='Independent saved126actualRGB pixel/metadata/versionedheldout proof; not bodyvision qualification')
 with (HERE/'VERIFY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),old112exact=True,new14=True,all_occlusion_exact=blocked,physical_steps=0)))
if __name__=='__main__':main()
