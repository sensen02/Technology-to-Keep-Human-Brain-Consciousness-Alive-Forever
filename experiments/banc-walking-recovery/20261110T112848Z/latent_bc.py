"""Loss-only FFv2 latent behavior cloning; no collection or physical stepping.

Latent MSE changes saturation weighting; it is NOT a constant rescale of bounded
MSE. Actor inputs, inhibitory architecture, physical halfspans and tanh deployment
are exactly student_wave.new_model('ff238'). Critic/std/PPO losses are unchanged.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import student_wave as sw
from identity import HERE, atomic_json, capture, verify, sha256
from ppo import save_checkpoint, load_checkpoint

SOURCE_FILES = sw.SOURCE_FILES + [HERE/'latent_bc.py', HERE/'check_latent_bc.py']
EPSILON = .001
LABEL_CONTRACT = dict(action_epsilon=EPSILON, clip_bounds=[-.999,.999],
    target='atanh(clip(raw Y, -0.999, 0.999))', objective='mean((actor(X)-z)**2)',
    deployment='tanh(actor(X)); original FFv2 physical raw inputs',
    weighting='changes saturation weighting, not pure bounded-loss rescaling',
    gaussian_equivalence='per-coordinate fixed sigma=.05 Gaussian NLL = latent MSE/(2*.05**2) + log(.05*sqrt(2*pi)); action Jacobian is label-only constant',
    raw_action_metrics='always against original unclipped Y')


def source_hashes():
    return {str(p.resolve()): sha256(p) for p in SOURCE_FILES}


def preflight():
    checks = json.loads((HERE/'LATENT_CHECKS.json').read_text())
    if not checks['all_passed'] or checks['source_hashes'] != source_hashes():
        raise ValueError('latent checks missing, failed, or source drift')
    return sw.teacher_preflight()


def latent_targets(y, epsilon=EPSILON):
    if epsilon != EPSILON:
        raise ValueError('preregistered action epsilon must be .001')
    if not torch.isfinite(y).all() or (y.abs() > 1+1e-12).any():
        raise ValueError('finite bounded raw labels required')
    return torch.atanh(y.clamp(-1+epsilon,1-epsilon))


def latent_loss(model, episodes, epsilon=EPSILON):
    total = sum(y.numel() for x,y in episodes)
    if not total:
        raise ValueError('empty split')
    return sum((model.actor(x)-latent_targets(y,epsilon)).square().sum() for x,y in episodes)/total


@torch.no_grad()
def metrics(model, episodes, epsilon=EPSILON):
    total = sum(y.numel() for x,y in episodes)
    sums = dict(latent_MSE=0., action_MSE=0., clipped_action_MSE=0., clipping_distortion_MSE=0.)
    for x,y in episodes:
        mean=model.actor(x); action=mean.tanh(); clipped=y.clamp(-1+epsilon,1-epsilon)
        for key,value in dict(latent_MSE=(mean-latent_targets(y,epsilon)).square(),
                action_MSE=(action-y).square(), clipped_action_MSE=(action-clipped).square(),
                clipping_distortion_MSE=(clipped-y).square()).items():
            sums[key] += float(value.sum())
    result={k:v/total for k,v in sums.items()}
    result['action_MSE_raw_minus_clipped']=result['action_MSE']-result['clipped_action_MSE']
    return result


def label_stats(episodes, epsilon=EPSILON):
    y=torch.cat([y.reshape(-1) for x,y in episodes]); clipped=y.clamp(-1+epsilon,1-epsilon)
    return dict(elements=y.numel(), raw_exact_plus_one_count=int((y==1).sum()),
        raw_exact_minus_one_count=int((y==-1).sum()), raw_exact_abs_one_fraction=float((y.abs()==1).double().mean()),
        clipped_count=int((y!=clipped).sum()), clipped_fraction=float((y!=clipped).double().mean()),
        clipping_distortion_MSE=float((clipped-y).square().mean()), raw_min=float(y.min()),raw_max=float(y.max()))


def train(args):
    if args.steps < 0 or not math.isfinite(args.lr) or args.lr <= 0 or args.action_epsilon != EPSILON:
        raise ValueError('nonnegative steps, positive finite lr, fixed epsilon .001 required')
    meta=sw.validate_dataset(args.dataset)
    if len(meta['episodes']) != 3 or [e['episode'] for e in meta['episodes']] != [0,1,2] or [e['split'] for e in meta['episodes']] != ['train','train','validation']:
        raise ValueError('requires independent base episodes 0/1 train, 2 validation')
    if not meta['all_episode_conditions_passed']:
        raise ValueError('all teacher episodes must pass')
    out=sw.exclusive_out(args.out)
    sw.seed_all(args.seed)
    core=sw.Core(adhesion=False)
    dataset_path=Path(args.dataset).resolve()/'DATASET.json'
    extras=SOURCE_FILES+[dataset_path]
    extras.extend(Path(p) for p in meta['teacher_reference_evidence']['files'])
    for e in meta['episodes']:
        extras.extend(Path(e[k]) for k in ('path','identity_path','report_path'))
        extras.extend(Path(p) for p in e.get('reused_original_files',{}))
        if e.get('trace'): extras.append(Path(e['trace']['path']))
    ident=capture(core,dict(mode='latent_bc',cli=vars(args),label_contract=LABEL_CONTRACT,
                           dataset_sha256=sha256(dataset_path)),extra_paths=extras)
    atomic_json(out/'IDENTITY.json',ident)
    episodes={'train':[],'validation':[]}
    for e in meta['episodes']:
        with np.load(e['path'],allow_pickle=False) as z:
            episodes[e['split']].append((torch.as_tensor(z['X238'].copy(),dtype=torch.float64),
                                        torch.as_tensor(z['Y42'].copy(),dtype=torch.float64)))
    model=sw.new_model('ff238')
    before_critic={k:v.clone() for k,v in model.critic.state_dict().items()}
    before_std=model.raw_logstd.detach().clone()
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=args.lr)
    stats={split:label_stats(items) for split,items in episodes.items()}
    curve=[]
    for step in range(args.steps+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss=latent_loss(model,episodes['train'],args.action_epsilon)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite latent loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.actor.parameters()):
                raise FloatingPointError('missing/nonfinite actor gradient')
            optimizer.step()
        row=dict(step=step,train=metrics(model,episodes['train']),validation=metrics(model,episodes['validation']))
        curve.append(row)
        atomic_json(out/'CURVE.json',dict(curve=curve,label_contract=LABEL_CONTRACT))
        if step==0 or step==args.steps or step%25==0: print(json.dumps(row),flush=True)
    if not torch.equal(before_std,model.raw_logstd) or not all(torch.equal(v,model.critic.state_dict()[k]) for k,v in before_critic.items()):
        raise ValueError('critic/std modified by BC')
    trainer=sw.make_trainer(model)
    verify(ident)
    save_checkpoint(out/'final.pt',trainer,ident,extra=dict(kind='latent_bc',
        bc_optimizer=optimizer.state_dict(),bc_config=vars(args),curve=curve,label_contract=LABEL_CONTRACT,
        label_stats=stats,dataset_metadata=meta,dataset_path=str(dataset_path),dataset_sha256=sha256(dataset_path),
        collection_identities=[json.loads(Path(e['identity_path']).read_text()) for e in meta['episodes']],
        actor_only_bc=True,std_initial=.05,ppo_optimizers='fresh',normalization='unchanged physical raw FFv2 input transform',
        interpretation='latent MSE changes saturation weighting; MSE does not establish closed-loop gait'))
    check_model=sw.new_model('ff238')
    load_checkpoint(out/'final.pt',sw.make_trainer(check_model),ident,restore_rng=False)
    if not all(torch.equal(v,check_model.state_dict()[k]) for k,v in model.state_dict().items()):
        raise ValueError('checkpoint reload mismatch')
    result=dict(checkpoint=str(out/'final.pt'),checkpoint_sha256=sha256(out/'final.pt'),steps=args.steps,
        final=curve[-1],label_stats=stats,label_contract=LABEL_CONTRACT,identity_sha256=ident['identity_sha256'],
        checkpoint_reload_exact=True,critic_unchanged=True,std_unchanged=True,closed_loop='NOT_EVALUATED')
    atomic_json(out/'RESULT.json',result)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--dataset',required=True);t.add_argument('--out',required=True)
    t.add_argument('--steps',type=int,default=1000);t.add_argument('--lr',type=float,default=.001)
    t.add_argument('--seed',type=int,default=20261110);t.add_argument('--action-epsilon',type=float,default=EPSILON)
    return p


if __name__=='__main__':
    args=parser().parse_args();preflight();train(args)
