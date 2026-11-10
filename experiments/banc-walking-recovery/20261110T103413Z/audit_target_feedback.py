"""Frozen-body actuator-memory Jacobian diagnostic, NOT full physical stability proof."""
from pathlib import Path
import json
import torch
import numpy as np
from physics import Core
from models import ActorCritic
from identity import HERE,atomic_json,sha256


def main():
    torch.set_num_threads(1)
    core=Core(adhesion=False)
    limits=core.plant.jnt_range[core.actuator.joint_ids]
    H=np.maximum((limits[:,1]-limits[:,0])*.5,.1)
    dataset=HERE.parent/'20261110T094051Z/SMOOTH_DATASET/episode_000.npz'
    with np.load(dataset) as z:x=z['X238'].copy();expert=z['Y42'].copy()
    candidates=[('smoothFF',HERE.parent/'20261110T094051Z/BC_FF_SMOOTH_1000/final.pt'),
      ('recoveryFF',HERE.parent/'20261110T094846Z/BC_FF_RECOVERY/final.pt'),
      ('halfFF',HERE.parent/'20261110T102045Z/BC_FF_HALF_RECOVERY/final.pt')]
    rows=[]
    for name,path in candidates:
        payload=torch.load(path,map_location='cpu',weights_only=False)
        model=ActorCritic('ff238',critic_obs_dim=255,dtype=torch.float64)
        model.load_state_dict(payload['state']['model'])
        reports=[]
        for index in [200,400,700,960,1000,1200,1500,1800]:
            inp=torch.as_tensor(x[index:index+1],dtype=torch.float64).requires_grad_()
            jac=torch.autograd.functional.jacobian(lambda a:model.actor(a).tanh(),inp).detach().numpy()[0,:,0,:]
            KT=(jac[:,42:84]+jac[:,126:168])/H[None,:]
            KP=jac[:,195:237]
            matrix=np.block([[np.eye(42)+.01*KT,.01*KP],[KT,KP]])
            expertKT=-10*np.eye(42)
            Mteacher=np.block([[np.eye(42)+.01*expertKT,np.zeros((42,42))],[expertKT,np.zeros((42,42))]])
            reports.append(dict(index=index,time_after_teacher_start_s=index*.005,
              frozen_body_memory_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(matrix)))),
              previous_action_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(KP)))),
              target_feedback_diagonal_mean=float(np.trace(KT)/42),
              target_feedback_positive_diagonal_count=int(np.sum(np.diag(KT)>0)),
              target_feedback_derivative_MSE_vs_teacher=float(np.mean((KT-expertKT)**2)),
              memory_20step_operator_norm=float(np.linalg.norm(np.linalg.matrix_power(matrix,20),2)),
              reference_teacher_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(Mteacher)))),
              expert_near_rate_saturation=int(np.sum(np.abs(expert[index])>.95))))
        rows.append(dict(model=name,checkpoint=str(path),checkpoint_sha256=sha256(path),states=reports))
    result=dict(results=rows,H=H.tolist(),observation_contract='target perturbation modifies BOTH normalized(target-q)42:84 AND normalized(target-q0)126:168 bydelta/H; previous actualaction195:237',
       policy_time_s=.005,rate_scale_rad_s=2.,target_increment='.01*normalizedaction',
       expert_interior_derivative='a=qdotdes/2 +(Tdesired-T)/(.05*2), K_T=-10I, K_prev=0 only holding qdes/qdotdes/body/contact/time fixed, excluding rate/target clips',
       limitation='Frozen-body memory subsystem only; full physical q/qvel/contact/GRU dynamics not included. Spectral radius is a diagnostic, not proof of full closedloop stability/instability.',
       next_test='If bad learned target feedback confirmed, bounded counterfactual target perturbation pairs and neural derivative supervision before changing body. No fixed controller outside ANN.')
    atomic_json(HERE/'TARGET_FEEDBACK_AUDIT.json',result)
    print(json.dumps({r['model']:dict(memory_spectral_radius=[s['frozen_body_memory_spectral_radius'] for s in r['states']],
        target_positive_diagonal=[s['target_feedback_positive_diagonal_count'] for s in r['states']]) for r in rows}),flush=True)

if __name__=='__main__':main()
