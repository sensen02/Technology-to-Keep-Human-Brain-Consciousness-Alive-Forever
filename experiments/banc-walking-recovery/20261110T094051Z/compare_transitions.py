"""Read-only comparison of demonstrated smoothness and learned takeover error."""
import json
import numpy as np
import torch
from pathlib import Path
from models import ActorCritic
from identity import HERE,atomic_json,sha256


def main():
    torch.set_num_threads(1)
    result=dict(experts={},students={})
    root=HERE.parent
    for name,directory in [('original',root/'20261110T074936Z/BC_DATASET_3EP'),('smooth',HERE/'SMOOTH_DATASET')]:
        with np.load(directory/'episode_000.npz') as z:
            y=z['Y42'];step=np.sqrt(np.mean(np.diff(y,axis=0)**2,axis=1))
            boundary=np.arange(1,len(y))%160==0
            result['experts'][name]=dict(boundary_action_change_RMS_max=float(step[boundary].max()),
              any_action_change_RMS_max=float(step.max()),copy_previous_action_MSE=float(np.mean(np.diff(y,axis=0)**2)),
              dataset=str(directory),dataset_sha256=sha256(directory/'DATASET.json'))
    with np.load(HERE/'SMOOTH_DATASET/episode_000.npz') as z:x=z['X238'].copy();y=z['Y42'].copy()
    for folder,arch in [('BC_FF_SMOOTH_1000','ff238'),('BC_GRU_SMOOTH_300','gru238')]:
        path=HERE/folder/'final.pt'
        if not path.exists():continue
        m=ActorCritic(arch,critic_obs_dim=255,dtype=torch.float64)
        p=torch.load(path,weights_only=False,map_location='cpu');m.load_state_dict(p['state']['model'])
        with torch.no_grad():
            xx=torch.as_tensor(x,dtype=torch.float64)
            pred=m.actor(xx) if arch=='ff238' else m.actor.sequence(xx[:,None],m.actor.initial_state(1))[0][:,0]
            prediction=pred.tanh().numpy()
        error=(prediction-y)**2;boundary=np.arange(len(x))%160==0
        row=dict(all_MSE=float(error.mean()),boundary_MSE=float(error[boundary].mean()),
            boundary_share=float(error[boundary].sum()/error.sum()),first_handover_action_RMSE=float(np.sqrt(error[960].mean())))
        if arch=='ff238':
            xx=torch.as_tensor(x[960:961],dtype=torch.float64).requires_grad_()
            J=torch.autograd.functional.jacobian(lambda a:m.actor(a).tanh(),xx).detach().numpy()[0,:,0,:]
            row['previous_action_jacobian_spectral_norm']=float(np.linalg.svd(J[:,195:237],compute_uv=False)[0])
        result['students'][arch]=row
    result['interpretation']='Smooth demonstrations remove rare boundary jumps but physical takeover still must pass; no MSE-only claim.'
    atomic_json(HERE/'TRANSITION_COMPARISON.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
