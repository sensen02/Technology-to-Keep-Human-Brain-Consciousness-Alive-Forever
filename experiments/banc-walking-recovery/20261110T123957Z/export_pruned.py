"""Declared neural synapse pruning, not retraining or an external controller."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from physics import Core
import student_wave as sw
from ppo import save_checkpoint,load_checkpoint
from identity import HERE,capture,atomic_json,verify,verify_runtime,sha256

PARENT=HERE.parent/'20261110T120531Z/BC_GRU_ROLE_1000/final.pt'
OUT=HERE/'PRUNED_GRU'


def array_sha(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    c=Core(adhesion=False)
    model,payload=sw.load_student(PARENT,c)
    parent_report=json.loads(PARENT.with_name('RESULT.json').read_text())
    assert sha256(PARENT)==parent_report['checkpoint_sha256']
    assert model.architecture=='gru238'
    original=deepcopy(model)
    trainer=sw.make_trainer(model,payload['config'])
    load_checkpoint(PARENT,trainer,payload['identity'],restore_rng=True)
    assert trainer.updates==0 and not trainer.actor_optimizer.state and not trainer.critic_optimizer.state
    before={k:v.clone() for k,v in model.state_dict().items()}
    encoder_before=array_sha(model.actor.encoder.weight)
    with torch.no_grad():model.actor.encoder.weight[:,189:195].zero_()
    assert torch.count_nonzero(model.actor.encoder.weight[:,189:195])==0
    changed=[]
    for key,value in model.state_dict().items():
        if not torch.equal(value,before[key]):changed.append(key)
        if key!='actor.encoder.weight':assert torch.equal(value,before[key])
    assert changed==['actor.encoder.weight']
    expected=before['actor.encoder.weight'].clone();expected[:,189:195]=0
    assert torch.equal(model.actor.encoder.weight,expected)
    dataset=Path(payload['extra']['dataset_path']).parent
    with np.load(dataset/'episode_002.npz',allow_pickle=False) as z:x=z['X238'].copy()
    raw=torch.as_tensor(x,dtype=torch.float64);masked=raw.clone();masked[:,189:195]=0
    with torch.no_grad():
        a,ha=model.actor.sequence(raw[:,None],model.actor.initial_state(1))
        b,hb=original.actor.sequence(masked[:,None],original.actor.initial_state(1))
        mean_diff=float((a-b).abs().max());hidden_diff=float((ha-hb).abs().max())
    assert mean_diff<1e-12 and hidden_diff<1e-12
    probe=raw[100:101].clone().requires_grad_()
    mean,state=model.actor.step(probe,model.actor.initial_state(1))
    grad=torch.autograd.grad(mean.square().sum()+state.square().sum(),probe)[0]
    assert torch.count_nonzero(grad[:,189:195])==0 and torch.count_nonzero(grad[:,:42])>0
    out=sw.exclusive_out(OUT)
    metadata=dict(operation='set actor.encoder.weight[:,189:195] tozero exactly',parent=str(PARENT),parent_sha256=sha256(PARENT),
       encoder_before_sha256=encoder_before,encoder_after_sha256=array_sha(model.actor.encoder.weight),
       changed_state_keys=changed,pruned_input_columns=list(range(189,195)),pruned_weight_count=128*6,
       no_other_model_changes=True,no_physics_steps=True,not_retraining=True,
       expected_static_mask_applies='both teacher-prefix hidden priming and autonomous stage; not original post-boundary cut',
       full_sequence_equivalence=dict(heldout_rows=len(x),mean_maxdiff=mean_diff,final_hidden_maxdiff=hidden_diff),
       raw_input_contract='238unchanged; disconnection represented inNNweights only',
       interpretation='engineering neural connection intervention, not original trained weights; all motion still ANN-only after teacherwithdrawal')
    ident=capture(c,dict(kind='declared_neural_synapse_pruning',pruning=metadata),extra_paths=sw.SOURCE_FILES+[Path(__file__).resolve(),PARENT]+[Path(p) for p in payload['identity']['files']])
    atomic_json(out/'IDENTITY.json',ident);verify(ident);verify_runtime(payload['identity'],c)
    inherited={key:value for key,value in payload['extra'].items() if key in ('dataset_metadata','dataset_path','dataset_sha256','collection_identities')}
    save_checkpoint(out/'final.pt',trainer,ident,extra=dict(kind='declared_neural_synapse_pruning',pruning=metadata,
         ppo_optimizers='fresh empty inheritedBC-PPO states; no BC optimizer continuation claimed',**inherited))
    loaded,_=sw.load_student(out/'final.pt',c)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    atomic_json(out/'EXPORT_CHECKS.json',dict(all_passed=True,pruning=metadata,checkpoint_reload_exact=True,
        physical_model_unchanged=True,source_hashes={str(HERE/n):sha256(HERE/n) for n in ['export_pruned.py','models.py','student_wave.py','physics.py']}))
    atomic_json(out/'RESULT.json',dict(checkpoint=str(out/'final.pt'),checkpoint_sha256=sha256(out/'final.pt'),
        checkpoint_reload_exact=True,changed_state_keys=changed,physical_steps=0,closed_loop='NOT_EVALUATED'))
    print(json.dumps(dict(checkpoint=str(out/'final.pt'),changed=changed,equivalent_mean_maxdiff=mean_diff,
          equivalent_hidden_maxdiff=hidden_diff,raw_input_contract_unchanged=True,physical_steps=0)),flush=True)

if __name__=='__main__':main()
