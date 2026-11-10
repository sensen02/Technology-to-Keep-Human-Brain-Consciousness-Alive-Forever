"""Fixed S2/S5 role-auxiliary BC continuation: exact parent2000 -> total3000.
Role head is training-only. No physical stepping or teacher actions on this path.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
import role_aux_bc as aux
import student_wave as sw
from identity import HERE, atomic_json, capture, sha256, verify, verify_runtime
from ppo import load_checkpoint, save_checkpoint

GRANDPARENT = HERE.parent/'20261110T114148Z/BC_GRU_ROLE_AUX_300/final.pt'
ANCESTOR_1000 = HERE.parent/'20261110T120531Z/BC_GRU_ROLE_1000/final.pt'
PARENT = HERE.parent/'20261110T125712Z/BC_GRU_ROLE_2000/final.pt'
PARENT_RESULT = PARENT.parent/'RESULT.json'
PARENT_IDENTITY = PARENT.parent/'IDENTITY.json'
OUT = HERE/'BC_GRU_ROLE_3000'
PARENT_STEPS = 2000
CHECKS = HERE/'ROLE_CONTINUATION_CHECKS.json'
SOURCE_FILES = sw.SOURCE_FILES + aux.OWN_SOURCES + [HERE/'ROLE_AUX_CHECKS.json', HERE/'continue_role_bc.py', HERE/'check_continue_role.py', HERE/'PREREG.json']
TOTAL_STEPS = 3000


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and a.dtype == b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys() == b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def rng_state():
    return dict(numpy=np.random.get_state(),python=random.getstate(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    np.random.set_state(state['numpy']); random.setstate(state['python']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def config_digest(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def state_head_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    return {str(p.resolve()):sha256(p) for p in SOURCE_FILES}


def preflight():
    aux.preflight()  # Historical absolute source paths remain authoritative and valid.
    report=json.loads(CHECKS.read_text())
    if not report['all_passed'] or report['source_hashes'] != source_hashes():
        raise ValueError('continuation checked source/config drift or failed checks')
    for key,path in (('parent_checkpoint',PARENT),('parent_result',PARENT_RESULT),('parent_identity',PARENT_IDENTITY)):
        if report[key] != str(path) or sha256(path) != report[key+'_sha256']:
            raise ValueError('checked parent artifact changed: '+key)
    if report['fixed_budget'] != dict(parent_steps=PARENT_STEPS,total_steps=TOTAL_STEPS,additional_steps=1000):
        raise ValueError('checked fixed continuation profile changed')
    return report


def validate_aux_payload(payload, expected_config, step):
    """Pure validation before any model/head/optimizer state mutation."""
    extra=payload.get('extra') or {}
    if extra.get('kind') != 'role_aux_bc':
        raise ValueError('parent must be role_aux_bc')
    cfg=extra['aux_bc_config']
    if cfg != expected_config or extra['aux_config_sha256'] != config_digest(cfg):
        raise ValueError('canonical auxiliary configuration mismatch')
    if state_head_digest(extra['aux_head_state_dict']) != extra['aux_head_sha256']:
        raise ValueError('auxiliary head digest mismatch')
    if set(extra['aux_head_state_dict']) != {'weight','bias'} or extra['aux_head_state_dict']['weight'].shape != (6,128) or extra['aux_head_state_dict']['bias'].shape != (6,):
        raise ValueError('auxiliary head shape mismatch')
    if not all(v.dtype==torch.float64 and torch.isfinite(v).all() for v in extra['aux_head_state_dict'].values()):
        raise ValueError('auxiliary head nonfinite/dtype mismatch')
    curve=extra['curve']
    if len(curve)!=step+1 or [r['step'] for r in curve]!=list(range(step+1)):
        raise ValueError('complete global curve required')
    if payload['state']['actor']['state'] or payload['state']['critic']['state'] or payload['state']['updates'] or payload['state']['accepted_steps']:
        raise ValueError('PPO states must remain fresh')
    opt=extra['aux_bc_optimizer']; groups=opt['param_groups']
    params=[v for k,v in payload['state']['model'].items() if k.startswith('actor.') and k not in ('actor.initial_halfspan','actor.physical_mask')]+list(extra['aux_head_state_dict'].values())
    ids=list(range(len(params)))
    if len(groups)!=1 or groups[0]['params']!=ids or set(opt['state'])!=set(ids):
        raise ValueError('actor plus head Adam coverage/order mismatch')
    expected_group=dict(lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None,params=ids)
    if groups[0] != expected_group:
        raise ValueError('Adam hyperparameters changed')
    for i,p in zip(ids,params):
        s=opt['state'][i]
        if set(s)!={'step','exp_avg','exp_avg_sq'} or float(s['step'])!=step:
            raise ValueError('all actor/head Adam counters must equal global step')
        for key in ('exp_avg','exp_avg_sq'):
            v=s[key]
            if v.shape!=p.shape or v.dtype!=p.dtype or not torch.isfinite(v).all():
                raise ValueError('Adam moment shape/dtype/nonfinite mismatch')
    return extra


def validate_original_ancestry(checkpoint):
    # Preserve the complete original 300->1000 validator and historical paths.
    PARENT = ANCESTOR_1000
    PARENT_RESULT = PARENT.parent/'RESULT.json'
    PARENT_IDENTITY = PARENT.parent/'IDENTITY.json'
    PARENT_STEPS = 1000
    checkpoint=Path(checkpoint).expanduser()
    if not checkpoint.is_absolute() or checkpoint.resolve()!=PARENT.resolve():
        raise ValueError('only the fixed 1000-step parent is accepted; descendants rejected')
    result=json.loads(PARENT_RESULT.read_text())
    if result['checkpoint']!=str(PARENT) or sha256(checkpoint)!=result['checkpoint_sha256']:
        raise ValueError('external parent checkpoint SHA256 mismatch')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    original=dict(mode='train',dataset=str(aux.DATASET),out=str(GRANDPARENT.parent),steps=300,lr=.001,seed=20261110,role_weight=.001,
        aux_seed_offset=6001,aux_seed=20267111,head='training-only Linear(128,6), outside ActorCritic',
        loss='bounded BC sum/(T*42) + role_weight * CE sum/T',full_contiguous_BPTT=True,train_episodes=[0,1],validation_episodes=[2])
    expected=dict(original,out=str(PARENT.parent),steps=PARENT_STEPS,parent_checkpoint=str(GRANDPARENT),total_steps=PARENT_STEPS)
    extra=validate_aux_payload(payload,expected,PARENT_STEPS)
    grand_result_path=GRANDPARENT.parent/'RESULT.json'
    grand_result=json.loads(grand_result_path.read_text())
    grand=torch.load(GRANDPARENT,map_location='cpu',weights_only=False)
    validate_aux_payload(grand,original,300)
    lineage=dict(checkpoint=str(GRANDPARENT),checkpoint_sha256=sha256(GRANDPARENT),result=str(grand_result_path),
        result_sha256=sha256(grand_result_path),identity_sha256=grand['identity']['identity_sha256'],step=300,total_steps=1000,additional_steps=700)
    if (grand_result['checkpoint']!=str(GRANDPARENT) or grand_result['checkpoint_sha256']!=lineage['checkpoint_sha256'] or
        json.dumps(grand_result['model_identity'],sort_keys=True)!=json.dumps(grand['model_identity'],sort_keys=True) or grand_result['identity_sha256']!=lineage['identity_sha256'] or
        grand_result['final']!=grand['extra']['curve'][-1] or
        json.loads((GRANDPARENT.parent/'IDENTITY.json').read_text())!=grand['identity']):
        raise ValueError('original 300-step ancestry proof mismatch')
    verify(grand['identity'])
    if (extra['original_aux_bc_config']!=original or extra['parent']!=lineage or result['parent']!=lineage or
        extra['global_step']!=PARENT_STEPS or extra['rng_restored'] is not True or
        extra['curve'][:301]!=grand['extra']['curve'] or result['initial']!=grand_result['final']):
        raise ValueError('exact 300-to-1000 continuation ancestry mismatch')
    if (result['steps']!=PARENT_STEPS or any(extra[k]!=result[k] for k in ('aux_config_sha256','aux_head_sha256')) or
        extra['curve'][-1]!=result['final'] or result['actor_and_head_optimizer_steps']!=[PARENT_STEPS]*13 or
        any(result[k] is not True for k in ('critic_unchanged','std_unchanged','checkpoint_reload_exact')) or result['physical_steps']!=0 or
        not exact(frozen_payload(payload),frozen_payload(grand))):
        raise ValueError('parent result/config/head/curve/frozen-state mismatch')
    expected_identity=dict(mode='role_bc_continuation',cli=dict(checkpoint=str(GRANDPARENT),out=str(PARENT.parent),total_steps=1000,checkpoint_every=250),
        parent=lineage,original_aux_bc_config=original)
    if payload['identity']['config'] != expected_identity:
        raise ValueError('parent continuation identity configuration mismatch')
    if json.loads(PARENT_IDENTITY.read_text())!=payload['identity']:
        raise ValueError('external parent identity mismatch')
    verify(payload['identity'])
    meta,provenance=aux.validate_role_dataset(aux.DATASET)
    if extra['dataset_metadata']!=meta or extra['teacher_reference_provenance']!=provenance:
        raise ValueError('original dataset or teacher provenance changed')
    if extra['dataset_path']!=provenance['dataset_path'] or extra['dataset_sha256']!=provenance['dataset_sha256']:
        raise ValueError('dataset path/hash changed')
    if extra['collection_identities']!=[json.loads(Path(e['identity_path']).read_text()) for e in meta['episodes']]:
        raise ValueError('collection identities changed')
    return payload,aux.load_episodes(meta)


def validate_parent(checkpoint):
    checkpoint=Path(checkpoint).expanduser()
    if not checkpoint.is_absolute() or checkpoint.resolve()!=PARENT.resolve():
        raise ValueError('only the fixed 2000-step parent is accepted; descendants rejected')
    result=json.loads(PARENT_RESULT.read_text())
    if result['checkpoint']!=str(PARENT) or sha256(checkpoint)!=result['checkpoint_sha256']:
        raise ValueError('external parent checkpoint SHA256 mismatch')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    ancestor,episodes=validate_original_ancestry(ANCESTOR_1000)
    previous=ancestor['extra']['aux_bc_config']
    original=ancestor['extra']['original_aux_bc_config']
    expected=dict(previous,out=str(PARENT.parent),steps=2000,parent_checkpoint=str(ANCESTOR_1000),total_steps=2000)
    extra=validate_aux_payload(payload,expected,2000)
    ancestor_result_path=ANCESTOR_1000.parent/'RESULT.json'
    ancestor_result=json.loads(ancestor_result_path.read_text())
    lineage=dict(checkpoint=str(ANCESTOR_1000),checkpoint_sha256=sha256(ANCESTOR_1000),result=str(ancestor_result_path),
        result_sha256=sha256(ancestor_result_path),identity_sha256=ancestor['identity']['identity_sha256'],step=1000,total_steps=2000,additional_steps=1000)
    if (extra['original_aux_bc_config']!=original or extra['parent_aux_bc_config']!=previous or
        extra['parent']!=lineage or result['parent']!=lineage or extra['global_step']!=2000 or extra['rng_restored'] is not True or
        extra['curve'][:1001]!=ancestor['extra']['curve'] or result['initial']!=ancestor_result['final']):
        raise ValueError('exact 1000-to-2000 continuation-of-continuation ancestry mismatch')
    if (result['steps']!=2000 or any(extra[k]!=result[k] for k in ('aux_config_sha256','aux_head_sha256')) or
        extra['curve'][-1]!=result['final'] or result['actor_and_head_optimizer_steps']!=[2000]*13 or
        any(result[k] is not True for k in ('critic_unchanged','std_unchanged','checkpoint_reload_exact')) or result['physical_steps']!=0 or
        not exact(frozen_payload(payload),frozen_payload(ancestor))):
        raise ValueError('parent2000 result/config/head/curve/frozen-state mismatch')
    expected_identity=dict(mode='role_bc_continuation',cli=dict(checkpoint=str(ANCESTOR_1000),out=str(PARENT.parent),total_steps=2000,checkpoint_every=250),
        parent=lineage,original_aux_bc_config=original,parent_aux_bc_config=previous)
    if payload['identity']['config']!=expected_identity or json.loads(PARENT_IDENTITY.read_text())!=payload['identity']:
        raise ValueError('parent2000 continuation identity configuration mismatch')
    verify(payload['identity'])
    for key in ('dataset_metadata','teacher_reference_provenance','dataset_path','dataset_sha256','collection_identities'):
        if extra[key]!=ancestor['extra'][key]:
            raise ValueError('parent2000 dataset/provenance changed: '+key)
    return payload,episodes


def optimizer_steps(optimizer, model, head):
    values=[int(float(s['step'])) for s in optimizer.state_dict()['state'].values()]
    if len(values)!=len(list(model.actor.parameters()))+len(list(head.parameters())) or len(set(values))!=1:
        raise ValueError('actor/head Adam counters disagree')
    return values


def build_resume(checkpoint, core=None):
    payload,episodes=validate_parent(checkpoint)  # all training-extra checks precede load_student
    return restore_validated(checkpoint,payload,episodes,core)


def restore_validated(checkpoint,payload,episodes,core=None):
    """Shared exact constructor/load path after complete payload validation."""
    core=sw.Core(adhesion=False) if core is None else core
    model,loaded=sw.load_student(checkpoint,core)
    if not exact(payload,loaded):
        raise ValueError('checkpoint changed between validation and loading')
    head=aux.new_aux_head(20261110)
    optimizer=aux.new_optimizer(model,head,.001)
    head.load_state_dict(payload['extra']['aux_head_state_dict'])
    optimizer.load_state_dict(deepcopy(payload['extra']['aux_bc_optimizer']))
    trainer=sw.make_trainer(model,payload['config'])
    # LAST load restores model + independent fresh PPO + RNG, after all constructors.
    load_checkpoint(checkpoint,trainer,payload['identity'],restore_rng=True)
    if not exact(head.state_dict(),payload['extra']['aux_head_state_dict']) or not exact(optimizer.state_dict(),payload['extra']['aux_bc_optimizer']):
        raise ValueError('head or BC Adam restoration not exact')
    if not exact(rng_state(),payload['rng']):
        raise ValueError('RNG restoration not exact')
    model.train()
    return dict(checkpoint=Path(checkpoint),core=core,model=model,head=head,optimizer=optimizer,trainer=trainer,
                payload=payload,episodes=episodes,step=payload['extra']['curve'][-1]['step'],curve=deepcopy(payload['extra']['curve']))


def update(model,head,optimizer,episodes):
    optimizer.zero_grad(set_to_none=True)
    loss,_,_=aux.objective(model,head,episodes,.001)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite BC role loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for g in optimizer.param_groups for p in g['params']):
        raise FloatingPointError('missing/nonfinite actor or head gradient')
    optimizer.step()


def measure(r,step):
    row=dict(step=step,train=aux.metrics(r['model'],r['head'],r['episodes']['train']),heldout=aux.metrics(r['model'],r['head'],r['episodes']['validation']))
    row['train_loss']=row['train']['MSE']+.001*row['train']['role_CE']
    return row


def frozen_payload(payload):
    return {k:v for k,v in payload['state']['model'].items() if not k.startswith('actor.')}


def frozen(model):
    return {k:v.clone() for k,v in model.state_dict().items() if not k.startswith('actor.')}


def validate_cli(args):
    if (args.total_steps!=TOTAL_STEPS or args.checkpoint_every!=250 or
        not Path(args.out).is_absolute() or Path(args.out).resolve()!=OUT.resolve()):
        raise ValueError('fixed parent2000 -> BC_GRU_ROLE_3000 total3000, interval250 required')


def continuation(args):
    validate_cli(args)
    torch.set_num_threads(1)
    preflight()
    r=build_resume(args.checkpoint)
    initial=measure(r,PARENT_STEPS)
    if initial!=r['curve'][-1]:
        raise ValueError('parent train/heldout metrics not bitexact')
    out=sw.exclusive_out(args.out)
    parent=r['checkpoint']; old=r['payload']['extra']; previous=deepcopy(old['aux_bc_config'])
    original=deepcopy(old['original_aux_bc_config'])
    lineage=dict(checkpoint=str(parent),checkpoint_sha256=sha256(parent),result=str(PARENT_RESULT),result_sha256=sha256(PARENT_RESULT),
        identity_sha256=r['payload']['identity']['identity_sha256'],step=PARENT_STEPS,total_steps=TOTAL_STEPS,additional_steps=1000)
    state=deepcopy(rng_state())
    paths=list(r['payload']['identity']['files'])+SOURCE_FILES+[CHECKS,parent,PARENT_RESULT,parent.parent/'IDENTITY.json']
    ident=capture(r['core'],dict(mode='role_bc_continuation',cli=vars(args),parent=lineage,original_aux_bc_config=original,parent_aux_bc_config=previous),extra_paths=paths)
    restore_rng(state)  # capture/Core construction cannot alter the resumed stream
    if not exact(rng_state(),r['payload']['rng']):
        raise ValueError('RNG changed before first update')
    atomic_json(out/'IDENTITY.json',ident)
    untouched=frozen(r['model']); saved=[]
    print(json.dumps(initial),flush=True)
    for step in range(PARENT_STEPS+1,TOTAL_STEPS+1):
        update(r['model'],r['head'],r['optimizer'],r['episodes']['train'])
        r['curve'].append(measure(r,step))
        atomic_json(out/'CURVE.json',dict(curve=r['curve'],parent=lineage,closed_loop='NOT_EVALUATED'))
        if step%50==0:
            print(json.dumps(r['curve'][-1]),flush=True)
        if step%args.checkpoint_every==0 or step==TOTAL_STEPS:
            if not exact(untouched,frozen(r['model'])) or any(s!=step for s in optimizer_steps(r['optimizer'],r['model'],r['head'])):
                raise ValueError('critic/std or Adam counter drift')
            verify_runtime(ident,r['core'])
            extra=deepcopy(old)
            config=dict(previous,out=str(out),steps=step,parent_checkpoint=str(parent),total_steps=TOTAL_STEPS)
            extra.update(aux_head_state_dict=r['head'].state_dict(),aux_head_sha256=aux.head_digest(r['head']),
                aux_bc_optimizer=r['optimizer'].state_dict(),aux_bc_config=config,aux_config_sha256=config_digest(config),
                original_aux_bc_config=original,parent_aux_bc_config=previous,parent=lineage,curve=r['curve'],global_step=step,rng_restored=True)
            path=out/f'checkpoint_{step:06d}.pt'
            save_checkpoint(path,sw.make_trainer(r['model'],r['payload']['config']),ident,extra=extra)
            saved.append(dict(path=str(path),sha256=sha256(path),step=step))
            if step==TOTAL_STEPS:
                save_checkpoint(out/'final.pt',sw.make_trainer(r['model'],r['payload']['config']),ident,extra=extra)
    final=out/'final.pt'; payload=torch.load(final,map_location='cpu',weights_only=False)
    validate_aux_payload(payload,config,TOTAL_STEPS)
    resumed=restore_validated(final,payload,r['episodes'],r['core'])
    if (not exact(resumed['model'].state_dict(),r['model'].state_dict()) or
        not exact(resumed['head'].state_dict(),r['head'].state_dict()) or
        not exact(resumed['optimizer'].state_dict(),r['optimizer'].state_dict()) or
        not exact(payload['rng'],rng_state()) or measure(resumed,TOTAL_STEPS)!=r['curve'][-1]):
        raise ValueError('independent final model/head/Adam/RNG/metrics reload mismatch')
    result=dict(checkpoint=str(final),checkpoint_sha256=sha256(final),steps=TOTAL_STEPS,parent=lineage,initial=initial,final=r['curve'][-1],
        aux_head_sha256=aux.head_digest(r['head']),aux_config_sha256=config_digest(config),actor_and_head_optimizer_steps=optimizer_steps(r['optimizer'],r['model'],r['head']),
        critic_unchanged=True,std_unchanged=True,checkpoint_reload_exact=True,physical_steps=0,closed_loop='NOT_EVALUATED',checkpoints=saved)
    atomic_json(out/'RESULT.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True)
    p.add_argument('--total-steps',type=int,required=True); p.add_argument('--checkpoint-every',type=int,default=250)
    return continuation(p.parse_args())


if __name__=='__main__':
    main()
