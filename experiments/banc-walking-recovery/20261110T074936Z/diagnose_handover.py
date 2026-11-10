"""Read-only learned-policy diagnostics. No physics step or model changes."""
import json
import numpy as np
import torch
from models import ActorCritic
from identity import HERE,atomic_json,sha256


def main():
    torch.set_num_threads(1)
    dataset=HERE/'BC_DATASET_3EP/episode_000.npz'
    with np.load(dataset,allow_pickle=False) as z:x=z['X238'].copy();y=z['Y42'].copy()
    previous=x[:,195:237];boundary=np.arange(len(x))%160==0
    report=dict(dataset=str(dataset),dataset_sha256=sha256(dataset),
      boundary_definition='one teacher decision at each .8s wave phase boundary;12/1920frames',
      zero_action_MSE=float((y*y).mean()),copy_previous_action_MSE=float(((previous-y)**2).mean()),models={})
    for folder,evaldir,arch in [('BC_FF238_1000','EVAL_FF238_HANDOVER','ff238'),('BC_GRU238_300','EVAL_GRU238_HANDOVER','gru238')]:
        checkpoint=HERE/folder/'final.pt'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=ActorCritic(arch,critic_obs_dim=255,dtype=torch.float64)
        model.load_state_dict(payload['state']['model'])
        with torch.no_grad():
            tensor=torch.as_tensor(x,dtype=torch.float64)
            mean=model.actor(tensor) if arch=='ff238' else model.actor.sequence(tensor[:,None],model.actor.initial_state(1))[0][:,0]
            prediction=mean.tanh().numpy()
        mse=(prediction-y)**2
        summary=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
           all_MSE=float(mse.mean()),boundary_MSE=float(mse[boundary].mean()),other_MSE=float(mse[~boundary].mean()),
           boundary_fraction=float(boundary.mean()),boundary_share_total_squared_error=float(mse[boundary].sum()/mse.sum()))
        with np.load(HERE/evaldir/'decisions_5ms.npz',allow_pickle=False) as z:
            observed=z['X238'];actions=z['action']
            summary.update(teacher_prefix_observation_maxdiff=float(np.max(np.abs(observed[:960]-x[:960]))),
               teacher_prefix_action_maxdiff=float(np.max(np.abs(actions[:960]-y[:960]))),
               first_autonomous_observation_maxdiff=float(np.max(np.abs(observed[960]-x[960]))),
               first_autonomous_action_RMSE=float(np.sqrt(np.mean((actions[960]-y[960])**2))),
               next_observation_RMS_difference=float(np.sqrt(np.mean((observed[961]-x[961])**2))),
               next_physical_feature_RMS_difference=float(np.sqrt(np.mean((observed[961,:195]-x[961,:195])**2))))
        if arch=='ff238':
            tensor=torch.as_tensor(x[960:961],dtype=torch.float64).requires_grad_()
            jac=torch.autograd.functional.jacobian(lambda z:model.actor(z).tanh(),tensor).detach().numpy()[0,:,0,:]
            summary.update(local_prev_action_jacobian_spectral_norm=float(np.linalg.svd(jac[:,195:237],compute_uv=False)[0]),
                local_prev_action_jacobian_frobenius=float(np.linalg.norm(jac[:,195:237])),
                local_physical_input_jacobian_frobenius=float(np.linalg.norm(jac[:,:195])))
        report['models'][arch]=summary
    report['findings']=[
      'Warmup states and actions match teacher data; first action error precedes distribution drift, so initial state handover mismatch is ruled out.',
      'Rare phase-boundary frames dominate residual imitation error. Low global MSE is insufficient.',
      'Large local previous-action sensitivity is an autoregressive instability risk, not by itself proof of the whole failure mechanism.',
      'Next controlled model test: mask redundant previous-action actor channels in train and deployment, preserving full physical state/target memory and recurrent history; compare baseline before changing physics.',
      'If necessary independently smooth teacher contact-role/endpoint timing and add recovery-state data; never claim MSE or teacher walking is neural success.']
    atomic_json(HERE/'HANDOVER_DIAGNOSIS.json',report)
    print(json.dumps({k:{s:v[s] for s in ('all_MSE','boundary_MSE','boundary_share_total_squared_error','first_autonomous_action_RMSE','next_physical_feature_RMS_difference')} for k,v in report['models'].items()}),flush=True)

if __name__=='__main__':main()
