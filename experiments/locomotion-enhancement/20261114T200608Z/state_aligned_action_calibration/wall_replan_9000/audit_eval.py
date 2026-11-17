"""SavedNumPy independentTFsupervision/conditionaldistance/targetaudit, noNN."""
import argparse,sys
from sources import ROOT,read,save,sha,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-saved-audit',action='store_true');a=p.parse_args()
 if not a.execute_parent_saved_audit:p.error('Parentsavedarrayauditrequired')
 import numpy as np
 e=read(ROOT/'EVALUATION.json');verify(e['source_hashes']);verify(e['data_hashes']);checks={'arraysSHA':sha(e['arrays']['path'])==e['arrays']['sha256'],'final900':e['step']==900 and sha(e['checkpoint'])==e['checkpoint_sha256']}
 with np.load(e['arrays']['path'],allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 checks['allfinite']=all(np.isfinite(v).all() for v in d.values());checks['shapes1920_12_42']=d['actual_target'].shape==d['actualX_action'].shape==d['conditional_action'].shape==(1920,12,42)
 base=np.ones(1920);base[:20]=10.;weights=base[:,None]*d['valid'];weights/=weights.sum(0)[None,:]
 for i,tf in enumerate(e['actualX_rows']):
  keep=d['valid'][:,i].astype(bool);sq=(d['actualX_action'][:,i]-d['actual_target'][:,i])**2;weighted=float((sq*weights[:,i,None]).sum()/42)
  checks[tf['id']+'_TFprimaryaction']=abs(float(sq[keep].mean())-tf['masked_action_MSE'])<=1e-12 and abs(weighted-tf['weighted_action_MSE'])<=1e-12
  cond=e['conditional_rows'][i];csq=(d['conditional_action'][:,i]-d['actual_target'][:,i])**2;checks[tf['id']+'_counterfactualdistance']=abs(float(csq[keep].mean())-cond['masked_action_MSE'])<=1e-12 and abs(float((csq*weights[:,i,None]).sum()/42)-cond['weighted_action_MSE'])<=1e-12
  if d['target_selector'][i]:
   delta=d['conditional_post_targets'][:,i]-d['actual_post_targets'][:,i];nmse=float(np.mean((delta/d['halfspan'])**2));checks[tf['id']+'_realizedtarget']=abs(nmse-cond['target_normalized_MSE'])<=1e-12
  else:checks[tf['id']+'_mixedtarget_not_supervised']=cond['target_normalized_MSE'] is None
 checks['labels21773_selector9_prior6']=int(d['valid'].sum())==21773 and int(d['target_selector'].sum())==9 and int(d['v1'].sum())==6
 checks['TFprimary_cond_not_supervision']=e['primary_supervision_metric']=='actualX_actionfit' and e['conditional_action_label_distance']=='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION'
 checks['DEV8unlabeled']=len(e['DEV'])==8 and all(not x['labels_available'] and x['teacher_MSE'] is None for x in e['DEV']);checks['no_body_or_neckPASS']=e['behavior_pass'] is None and e['new_neck_deployed'] is False
 checks['noTorch_Core']=not any(n in sys.modules for n in ('torch','mujoco','physics','flygym','static_teacher'))
 save(ROOT/'EVALUATION_VERIFY.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,evaluation_sha256=sha(ROOT/'EVALUATION.json'),checkpoint_sha256=e['checkpoint_sha256'],physical_steps=0,neural_forwards=0))
 if not all(checks.values()):raise ValueError('Statealignedsavedmetricsfailed')
if __name__=='__main__':main()
