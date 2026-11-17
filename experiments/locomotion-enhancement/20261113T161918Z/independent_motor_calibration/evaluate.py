"""Final600 fixed-observation ANN metrics only, no body qualification."""
import argparse
import sys
from sources import ROOT,read,save,sha,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-evaluation',action='store_true');a=p.parse_args()
    if not a.execute_parent_evaluation:p.error('Parent finalNN evaluation flagrequired')
    from learning import cpu1,metrics
    cpu1()
    import torch
    import common,model
    from data import prepare
    trained=read(ROOT/'TRAIN/RESULT.json');checkpoint=ROOT/'TRAIN/checkpoint_000600.pt'
    if trained['updates']!=600 or trained['checkpoint']!=str(checkpoint) or sha(checkpoint)!=trained['sha256']:raise ValueError('Only actualfinal600')
    for step in (0,300,600):
        proof=read(ROOT/'TRAIN'/f'VERIFY_{step:06d}.json')
        if not proof['integrity_all_pass'] or sha(proof['checkpoint'])!=proof['checkpoint_sha256']:raise ValueError('All3saved qualifications required')
    m,payload=common.load_checkpoint(checkpoint,trained['sha256'],final=True);data=prepare(m);scan=model.make_scan(m);rows=metrics(scan,data)
    with torch.no_grad():
        residual,h=scan(data['features']);prediction=(data['oldmean']+residual).tanh();target=data['y']
        # Descriptive feasibility only: bound6 may be compared with target-minus-oldmean.
        desired=target.clamp(-1.+1e-9,1.-1e-9).atanh()-data['oldmean']
        diagnostic=dict(required_mean_residual_absmax=float(desired.abs().max()),required_over_bound6_fraction=float((desired.abs()>6.).double().mean()),target_clamped_for_atanh_only=True,target_extreme_abs1_fraction=float((target.abs()>=1.).double().mean()),no_target_or_runtime_scale_change=True)
        artifact=ROOT/'EVALUATION_ARRAYS.npz'
        import numpy as np
        np.savez_compressed(artifact,actual_target=target.numpy().copy(),old_mean=data['oldmean'].numpy().copy(),new_residual=residual.numpy().copy(),new_action=prediction.numpy().copy(),command=data['u'].numpy().copy())
    save(ROOT/'EVALUATION.json',dict(kind='FINAL600_INDEPENDENT_MOTOR_FIXED_OBSERVATION_EVALUATION_V1',checkpoint=str(checkpoint),checkpoint_sha256=trained['sha256'],step=600,rows=rows,descriptive_residual=diagnostic,arrays=dict(path=str(artifact),sha256=sha(artifact)),source_hashes=payload['config']['source_hashes'],data_hashes=payload['config']['data_hashes'],physical_steps=0,behavior_pass=None,new_neck_deployed=False,scope='Fit metrics only, no heldoutgeneralization/bodymotion/stop/pivotPASS inferred. All8trajectories aretraininginputs.'))
    print('FINAL600_FIXED_OBSERVATION_METRICS_SAVED',flush=True)


if __name__=='__main__':main()
