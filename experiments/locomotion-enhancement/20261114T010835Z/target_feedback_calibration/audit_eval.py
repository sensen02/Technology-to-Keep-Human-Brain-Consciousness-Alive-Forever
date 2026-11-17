"""IndependentNumPy conditional/action/target metrics; no NN or physicalbackend."""
import argparse,math,sys
from sources import ROOT,read,save,sha,verify


def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-saved-audit',action='store_true');a=p.parse_args()
 if not a.execute_parent_saved_audit:p.error('Parentsavedarrayauditrelease required')
 import numpy as np
 e=read(ROOT/'EVALUATION.json');verify(e['source_hashes']);verify(e['data_hashes']);checks={'arrays_SHA':sha(e['arrays']['path'])==e['arrays']['sha256'],'actualfinal600':e['step']==600 and sha(e['checkpoint'])==e['checkpoint_sha256']}
 with np.load(e['arrays']['path'],allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 checks['allfinite']=all(np.isfinite(v).all() for v in d.values());checks['shape12']=d['actual_target'].shape==d['conditional_action'].shape==(1920,12,42)
 base=np.ones(1920);base[:20]=10.;weights=base[:,None]*d['valid'];weights/=weights.sum(0)[None,:]
 for i,row in enumerate(e['conditional_rows']):
  keep=d['valid'][:,i].astype(bool);sq=(d['conditional_action'][:,i]-d['actual_target'][:,i])**2;masked=float(sq[keep].mean());weighted=float((sq*weights[:,i,None]).sum()/42)
  checks[row['id']+'_conditional_action']=abs(masked-row['masked_action_MSE'])<=1e-12 and abs(weighted-row['weighted_action_MSE'])<=1e-12
  actualsq=(d['actualX_action'][:,i]-d['actual_target'][:,i])**2;checks[row['id']+'_actualX_action']=abs(float(actualsq[keep].mean())-e['actualX_rows'][i]['masked_action_MSE'])<=1e-12
  if d['target_selector'][i]:
   delta=d['conditional_post_targets'][:,i]-d['actual_post_targets'][:,i];nmse=float(np.mean((delta/d['halfspan'])**2))
   checks[row['id']+'_nine_realized_target']=abs(nmse-row['target_normalized_MSE'])<=1e-12
  else:checks[row['id']+'_no_mixedtarget_supervision']=row['target_normalized_MSE'] is None
 checks['nine_selector_six_v1']=int(d['target_selector'].sum())==9 and int(d['v1'].sum())==6
 checks['no_behavior_or_neckPASS']=e['behavior_pass'] is None and e['new_neck_deployed'] is False
 checks['all8DEV_unlabeled']=len(e['DEV'])==8 and all(not x['labels_available'] and x['teacher_MSE'] is None for x in e['DEV'])
 checks['noTorch_Coreimports']=not any(n in sys.modules for n in ('torch','mujoco','physics','flygym','static_teacher'))
 save(ROOT/'EVALUATION_VERIFY.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,evaluation_sha256=sha(ROOT/'EVALUATION.json'),checkpoint_sha256=e['checkpoint_sha256'],physical_steps=0,neural_forwards=0,conditional_notphysicalcounterfactual=True))
 if not all(checks.values()):raise ValueError('Independentsavedtargetfeedbackmetricfailed')


if __name__=='__main__':main()
