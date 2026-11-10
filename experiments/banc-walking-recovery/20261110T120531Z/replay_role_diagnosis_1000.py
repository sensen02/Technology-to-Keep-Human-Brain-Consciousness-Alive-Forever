"""OFFLINE classifier diagnosis on saved actual neural observations.
The classifier is never evaluated in the physical actor/control rollout.
"""
import json
import numpy as np
import torch
from torch import nn
from student_wave import load_student
from physics import Core
from identity import HERE,atomic_json,sha256


def main():
    torch.set_num_threads(1)
    checkpoint=HERE/'BC_GRU_ROLE_1000/final.pt'
    model,payload=load_student(checkpoint,Core(adhesion=False))
    head=nn.Linear(128,6,dtype=torch.float64)
    head.load_state_dict(payload['extra']['aux_head_state_dict'])
    decisions=HERE/'EVAL_GRU_ROLE_1000/decisions_5ms.npz'
    with np.load(decisions,allow_pickle=False) as z:x=z['X238'].copy();actions=z['action'].copy();source=z['source'].copy();times=z['time'].copy()
    state=model.actor.initial_state(1);means=[];prediction=[];entropy=[]
    with torch.no_grad():
        for obs in x:
            mean,state=model.actor.step(torch.as_tensor(obs[None],dtype=torch.float64),state)
            probs=head(state).softmax(-1)
            means.append(mean.tanh()[0].numpy());prediction.append(int(probs.argmax(-1)))
            entropy.append(float(-(probs*probs.clamp_min(1e-300).log()).sum()))
    mean=np.asarray(means);prediction=np.asarray(prediction);entropy=np.asarray(entropy)
    auto=source==3;teacher=source==2
    diff=float(np.max(np.abs(mean[auto]-actions[auto])))
    assert diff<1e-12,diff
    expected=(np.arange(len(x))//160)%6
    periods=[]
    for start in range(0,len(x),160):
        end=min(start+160,len(x));part=prediction[start:end]
        periods.append(dict(first_time=float(times[start]),last_time=float(times[end-1]),source=int(source[start]),
            planned_teacher_role=int(expected[start]),predicted_counts=np.bincount(part,minlength=6).tolist(),
            accuracy_vs_reference_timing=float(np.mean(part==expected[start:end])),mean_entropy=float(entropy[start:end].mean())))
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),decisions=str(decisions),decisions_sha256=sha256(decisions),
       offline_action_replay_maxdiff=diff,teacher_prefix_role_accuracy=float(np.mean(prediction[teacher]==expected[teacher])),
       autonomous_role_accuracy_vs_reference_timing=float(np.mean(prediction[auto]==expected[auto])),windows=periods,
       interpretation='Offline only. Planned teacher phase is a diagnostic reference, not observed true role under student. Classification does not prove internal oscillator, coordinated walking or sensory dependency. No head output entered actual control.')
    atomic_json(HERE/'OFFLINE_ROLE_DIAGNOSIS.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('windows',)}),flush=True)
    print('autonomous_windows',[(w['planned_teacher_role'],w['predicted_counts']) for w in periods if w['source']==3],flush=True)

if __name__=='__main__':main()
