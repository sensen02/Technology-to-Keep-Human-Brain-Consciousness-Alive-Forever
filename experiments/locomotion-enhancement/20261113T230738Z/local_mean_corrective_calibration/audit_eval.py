"""IndependentNumPy final12 maskedmetrics/bound audit, no Torch/teacher/Core."""
import argparse
import math
import sys
from sources import ROOT,read,save,sha,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-saved-audit',action='store_true');a=p.parse_args()
    if not a.execute_parent_saved_audit:p.error('Parentsavedmetricrelease required')
    import numpy as np
    e=read(ROOT/'EVALUATION.json');verify(e['source_hashes']);verify(e['data_hashes']);checks={'actualfinal600_SHA':e['step']==600 and sha(e['checkpoint'])==e['checkpoint_sha256'],'savedarrays_SHA':sha(e['arrays']['path'])==e['arrays']['sha256']}
    with np.load(e['arrays']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    y=a['actual_target'];pred=a['new_action'];old=a['old_mean'];res=a['new_residual'];valid=a['valid']
    checks['full12shape']=y.shape==pred.shape==old.shape==res.shape==(1920,12,42) and valid.shape==(1920,12)
    checks['allfinite']=all(np.isfinite(v).all() for v in a.values())
    checks['valid_binary_nonempty']=bool(np.all((valid==0.)|(valid==1.)) and np.all(valid.sum(0)>0))
    checks['newaction_formula']=bool(np.allclose(np.tanh(old+res),pred,atol=1e-12,rtol=0))
    checks['residual6bound']=bool(np.max(np.abs(res))<=6.+1e-12)
    base=np.ones(1920);base[:20]=10.;weight=base[:,None]*valid;weight/=weight.sum(0)[None,:]
    lo=np.tanh(old-6.);hi=np.tanh(old+6.);floor=(y-np.minimum(np.maximum(y,lo),hi))**2
    rows=[]
    for i,r in enumerate(e['rows']):
        keep=valid[:,i].astype(bool);sq=(pred[:,i]-y[:,i])**2;masked=float(sq[keep].mean());weighted=float((sq*weight[:,i,None]).sum()/42);floorRMSE=math.sqrt(float(floor[:,i][keep].mean()))
        checks[r['id']+'_mask_MSE']=abs(masked-r['masked_action_MSE'])<=1e-12 and abs(math.sqrt(masked)-r['masked_action_RMSE'])<=1e-12
        checks[r['id']+'_weighted_MSE']=abs(weighted-r['weighted_action_MSE'])<=1e-12
        checks[r['id']+'_valid_count']=int(keep.sum())==r['valid_rows']
        checks[r['id']+'_validrangefloor_gate']=floorRMSE<=.01
        rows.append(dict(id=r['id'],valid_rows=int(keep.sum()),masked_action_MSE=masked,weighted_action_MSE=weighted,validtarget_action_RMSE_floor=floorRMSE,unmasked_metrics_not_acceptance=True))
    checks['no_DEV_teacherlabel']=all(not row['labels_available'] and row['teacher_MSE'] is None for row in e['DEV']) and len(e['DEV'])==8
    checks['no_body_or_neckPASS']=e['behavior_pass'] is None and e['new_neck_deployed'] is False
    checks['noTorch_Coreimports']=not any(k in sys.modules for k in ('torch','mujoco','physics','flygym','static_teacher'))
    save(ROOT/'EVALUATION_VERIFY.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,rows=rows,evaluation_sha256=sha(ROOT/'EVALUATION.json'),checkpoint_sha256=e['checkpoint_sha256'],physical_steps=0,neural_forwards=0,invalid_history_rows_preserved=True))
    if not all(checks.values()):raise ValueError('Independentmasked12metricsauditfailed')
    print('LOCALMEAN_MASKED12_INDEPENDENT_METRICS_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
