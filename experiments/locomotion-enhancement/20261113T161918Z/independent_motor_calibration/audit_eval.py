"""Parent savedarray-only independent recompute, no Torch/Core."""
import argparse
import math
import sys
from sources import ROOT,read,save,sha,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-saved-audit',action='store_true');a=p.parse_args()
    if not a.execute_parent_saved_audit:p.error('Parent savedarrayaudit flagrequired')
    import numpy as np
    e=read(ROOT/'EVALUATION.json');verify(e['source_hashes']);verify(e['data_hashes']);checks={}
    checks['arrays_sha']=sha(e['arrays']['path'])==e['arrays']['sha256']
    checks['final600_sha']=e['step']==600 and sha(e['checkpoint'])==e['checkpoint_sha256']
    with np.load(e['arrays']['path'],allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
    y=arrays['actual_target'];pred=arrays['new_action'];res=arrays['new_residual'];old=arrays['old_mean'];command=arrays['command']
    checks['full_shapes']=y.shape==pred.shape==res.shape==old.shape==(1920,8,42) and command.shape==(1920,8,2)
    checks['finite_all']=all(np.isfinite(v).all() for v in arrays.values())
    checks['action_tanh_mean']=bool(np.allclose(np.tanh(old+res),pred,atol=1e-12,rtol=0))
    checks['new_residual_bound6']=bool(np.max(np.abs(res))<=6.+1e-12)
    for i,row in enumerate(e['rows']):
        mse=float(np.mean((pred[:,i]-y[:,i])**2));rmse=math.sqrt(mse)
        checks[row['id']+'_independentMSE']=abs(mse-row['action_MSE'])<1e-12 and abs(rmse-row['action_RMSE'])<1e-12
        checks[row['id']+'_independentResidual']=abs(float(np.max(np.abs(res[:,i])))-row['residual_absmax'])<1e-12
    checks['no_behavior_or_neck_PASS']=e['behavior_pass'] is None and e['new_neck_deployed'] is False
    checks['no_torch_physicalimports']=not any(k in sys.modules for k in ('torch','mujoco','physics','flygym'))
    save(ROOT/'EVALUATION_VERIFY.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,evaluation_sha256=sha(ROOT/'EVALUATION.json'),checkpoint_sha256=e['checkpoint_sha256'],physical_steps=0,neural_forwards=0))
    if not all(checks.values()):raise ValueError('Independent savedmetric auditfailed')
    print('INDEPENDENT_FIXED_OBSERVATION_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
