"""Source-hashed actual saved arrays only; no model/physics or new inference."""
from pathlib import Path
import json,hashlib,os
ROOT=Path(__file__).resolve().parent;SCOPE=ROOT.parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 prereg=json.loads((ROOT/'PREREG.json').read_text());paths=[ROOT/'PREREG.json',Path(__file__),SCOPE/'PARENT_BODY_REVIEW.json',SCOPE/'fusion_aggregation_repair/train.npz'];hashes={str(p):sha(p) for p in paths};result={};mask=list(range(42))+list(range(84,126))+list(range(168,195))
 with np.load(paths[-1],allow_pickle=False) as z:mean=z['mean'].copy();std=z['std'].copy();train=z['body111'].copy()
 low=train.min((0,1));high=train.max((0,1))
 def stats(x):return dict(RMS=float(np.sqrt(np.mean(x*x))),mean=float(x.mean()),max_abs=float(abs(x).max()))
 for case in prereg['cases']:
  folder=SCOPE/'sensory_body_repair/trials'/case;v=json.loads((folder/'VERIFY.json').read_text());r=json.loads((folder/'RESULT.json').read_text());assert v['integrity_all_pass'] and all(v['checks'].values()) and sha(folder/'RESULT.json')==v['result_sha256'];p=folder/'neural_decisions.npz';assert sha(p)==r['neural_decisions']['sha256']
  for p2 in (p,folder/'RESULT.json',folder/'VERIFY.json'):hashes[str(p2)]=sha(p2)
  with np.load(p,allow_pickle=False) as z:values={k:z[k].copy() for k in ('time','visual_raw_turn','odor_raw_turn','raw_turn','actor_receptors','AN_held','DN_held','X238','brain_hidden_after','odor_hidden_after')}
  t=values['time']-5.8;windows={}
  for name,(a,b) in prereg['windows_ANN_s'].items():
   use=(t>=a-1e-8)&(t<b-1e-8);assert use.any();windows[name]={k:stats(values[k][use]) for k in ('visual_raw_turn','odor_raw_turn','raw_turn','AN_held','DN_held','actor_receptors')};windows[name]['neuralturn_abs_gt_09_fraction']=float(np.mean(abs(values['raw_turn'][use])>.9))
  actual=values['X238'][1160:,mask];nz=(actual-mean)/std;outside=(actual<low)|(actual>high);peaks=abs(nz).max(0);top=np.argsort(peaks)[-5:][::-1];entry=dict(windows=windows,body111_max_abs_z=float(abs(nz).max()),body111_outside_train_minmax_fraction=float(outside.mean()),top5_raw_indices=[dict(raw_index=mask[int(i)],max_abs_z=float(peaks[i]),outside_fraction=float(outside[:,i].mean())) for i in top],complete_leg_cycles=r['autonomous_cycles']['complete_per_leg'])
  if case=='O_left_odorcut':entry['actual_actor_zero_after1_6']=bool(np.all(values['actor_receptors'][t>=1.6-1e-8]==0));assert entry['actual_actor_zero_after1_6']
  if case.startswith('F_'):
   vraw=values['visual_raw_turn'];oraw=values['odor_raw_turn'];target=np.clip((vraw*abs(vraw)+oraw*abs(oraw))/(abs(vraw)+abs(oraw)+1e-8),-.78,.78);target=np.r_[0.,target[:-1]];use=t>=-1e-8;entry['same_declared_joint_teacher_RMSE_actual_new_body']=float(np.sqrt(np.mean((values['raw_turn'][use]-target[use])**2)));entry['actual_joint_output_max_abs']=float(abs(values['raw_turn'][use]).max())
  result[case]=entry
 for p,h in hashes.items():assert sha(p)==h
 report=dict(status='ACTUAL_SAVED_ARRAY_REMAINING_FAILURE_DIAGNOSTIC',cases=result,source_hashes=hashes,source_unchanged=True,model_forward_calls=0,optimizer_steps=0,new_physics_steps=0,physical_counterfactual_claim=False,interpretation='Residual sensory amplitude, fusion transient, joint extrapolation andbodydomain statistics are descriptive; no unique rootcause proof or newsuccessgate.')
 with (ROOT/'RESULT.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({case:{'early_turn':entry['windows']['early']['raw_turn'],'cutwindow_odor':entry['windows']['cut_recovery']['odor_raw_turn'],'cutwindow_decoded':entry['windows']['cut_recovery']['raw_turn'],'joint_RMSE':entry.get('same_declared_joint_teacher_RMSE_actual_new_body'),'top_body_indices':entry['top5_raw_indices']} for case,entry in result.items()},indent=2))
if __name__=='__main__':main()
