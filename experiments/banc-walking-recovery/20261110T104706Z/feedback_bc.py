"""Offline corrective target pairs for unchanged raw FF238 ANN; never a controller.

Both target sensor blocks move coherently. Previous action, body and teacher history
stay fixed. Finite pairs are exact synthetic transforms of stored float64 features;
the original proprioception measurements were quantized to float32. These are not
physical trajectories and their fit does not establish closed-loop stability.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import student_wave as sw
from identity import HERE, atomic_json, capture, sha256, verify
from ppo import save_checkpoint, load_checkpoint

BASE = HERE.parent/'20261110T094051Z/SMOOTH_DATASET'
PARENT = HERE.parent/'20261110T102045Z/BC_FF_HALF_RECOVERY/final.pt'
SLOPE = 10.0  # 1/(teacher target_tau_s .05 * physical MAX_RATE 2)
COMMAND_DT = .005
BOUND_MARGIN = 1e-6  # conservative allowance for original float32 sensor quantization


def geometry(c):
    jids = c.actuator.joint_ids
    half = np.maximum(np.ptp(c.plant.jnt_range[jids], axis=1)/2, .1)
    q0 = np.asarray(c.shim.q0)[c.a_qpos].copy()
    return half, q0, c.actuator.target_bounds_rad.copy()


def pair_config(epsilon=.002, pair_weight=.1, seed=20261331, interior_max=.8):
    if not all(math.isfinite(v) for v in (epsilon, pair_weight, interior_max)):
        raise ValueError('finite pair configuration required')
    if epsilon <= 0 or pair_weight < 0 or not 0 < interior_max < 1-SLOPE*epsilon:
        raise ValueError('positive epsilon, nonnegative weight and strictly interior labels required')
    return dict(epsilon=float(epsilon), pair_weight=float(pair_weight), seed=int(seed),
                interior_max=float(interior_max), slope_per_rad=-SLOPE,
                command_dt_s=COMMAND_DT, max_rate_rad_s=2., bound_margin_rad=BOUND_MARGIN,
                directions='one fixed independent Rademacher 42-vector per eligible row',
                target_blocks=[[42,84],[126,168]], previous_action_perturbed=False,
                loss='base_MSE + pair_weight * mean(((a_plus-a_minus) - (y_plus-y_minus))**2)',
                pair_units='normalized action squared; no division by epsilon or dt',
                sensor_quantization='original float32 proprioception in float64 raw dataset; exact synthetic linear pair transform',
                evidence_scope='offline fixed-body/history finite pairs, not physical trajectory or stability proof')


def make_pairs(x, y, half, q0, bounds, config, episode):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or x.shape[1] != 238 or y.shape != (len(x),42):
        raise ValueError('raw 238/42 dimensions required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite source data')
    eps = config['epsilon']
    target = x[:,126:168]*half+q0
    # Require the whole row to be unambiguously free of rate and horizon clipping.
    low, high = bounds[:,0]+BOUND_MARGIN, bounds[:,1]-BOUND_MARGIN
    eligible = np.max(np.abs(y),axis=1) < config['interior_max']
    for sign in (-1,1):
        # Every possible Rademacher sign must be valid, independent of RNG draw.
        t = target+sign*eps
        label = y-sign*SLOPE*eps
        end = t+COMMAND_DT*2*label
        eligible &= np.all((t > low)&(t < high)&(end > low)&(end < high),axis=1)
    endpoint = target+COMMAND_DT*2*y
    eligible &= np.all((endpoint > low)&(endpoint < high),axis=1)
    indices = np.flatnonzero(eligible)
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'],int(episode)]))
    direction = rng.choice(np.array([-1.,1.]),size=(len(indices),42))
    delta = eps*direction
    plus, minus = x[indices].copy(), x[indices].copy()
    for start, end in ((42,84),(126,168)):
        plus[:,start:end] += delta/half
        minus[:,start:end] -= delta/half
    # Interior criterion and horizon checks above make these clipping operations inactive.
    yp = np.clip(y[indices]-SLOPE*delta,-1,1)
    ym = np.clip(y[indices]+SLOPE*delta,-1,1)
    return dict(plus=plus, minus=minus, y_plus=yp, y_minus=ym,
                expected_difference=yp-ym, direction=direction, indices=indices,
                episode=int(episode), total_rows=len(x), eligible_rows=len(indices))


def prepare(dataset, c, config):
    meta = sw.validate_dataset(dataset)
    if [(e['episode'],e['split']) for e in meta['episodes']] != [(0,'train'),(1,'train'),(2,'validation')]:
        raise ValueError('requires episodes 0/1 train, 2 heldout')
    if not meta['all_episode_conditions_passed']:
        raise ValueError('base teacher episode condition failed')
    half,q0,bounds=geometry(c)
    data={'train':[], 'validation':[]}; pairs={'train':[], 'validation':[]}; counts=[]
    for e in meta['episodes']:
        with np.load(e['path'],allow_pickle=False) as z:
            x,y=z['X238'].copy(),z['Y42'].copy()
        p=make_pairs(x,y,half,q0,bounds,config,e['episode'])
        data[e['split']].append((torch.as_tensor(x),torch.as_tensor(y)))
        pairs[e['split']].append({k:torch.as_tensor(p[k]) for k in ('plus','minus','expected_difference')})
        counts.append(dict(episode=e['episode'],split=e['split'],total_rows=len(x),
                           eligible_rows=p['eligible_rows'],indices=p['indices'].tolist()))
    if any(sum(len(p['plus']) for p in pairs[split]) == 0 for split in pairs):
        raise ValueError('no interior pairs for train or holdout')
    return meta,data,pairs,dict(H=half.tolist(),q0=q0.tolist(),target_bounds=bounds.tolist(),episodes=counts)


def pair_loss(model, pairs):
    n=sum(p['plus'].numel()//238*42 for p in pairs)
    if not n:
        raise ValueError('empty corrective pair batch')
    return sum(((torch.tanh(model.actor(p['plus']))-torch.tanh(model.actor(p['minus'])))
                -p['expected_difference']).square().sum() for p in pairs)/n


def losses(model, data, pairs, weight):
    base=sw.bounded_bc_loss(model,data)
    paired=pair_loss(model,pairs)
    return base+weight*paired,base,paired


def check_preflight():
    path=HERE/'FEEDBACK_CHECKS.json'
    report=json.loads(path.read_text())
    if not report['all_passed']:
        raise ValueError('feedback checks failed')
    for source,digest in report['source_hashes'].items():
        if not Path(source).is_absolute() or sha256(source)!=digest:
            raise ValueError('feedback check source drift: '+source)


def train(args):
    if args.steps < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('positive steps and lr required')
    check_preflight()
    config=pair_config(args.epsilon,args.pair_weight,args.seed,args.interior_max)
    c=sw.Core(adhesion=False)
    model,parent_payload=sw.load_student(args.checkpoint,c)
    if model.architecture!='ff238':
        raise ValueError('initial experiment supports unchanged raw FF238 only')
    meta,data,pairs,selection=prepare(args.dataset,c,config)
    # Parent load never restores RNG. Fresh seed and BC Adam: MODEL ONLY warmstart.
    sw.seed_all(args.seed)
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=args.lr)
    out=sw.exclusive_out(args.out)
    dataset_path=Path(args.dataset).resolve()/'DATASET.json'
    parent=Path(args.checkpoint).resolve()
    extras=sw.SOURCE_FILES+[Path(__file__).resolve(),HERE/'check_feedback.py',HERE/'FEEDBACK_CHECKS.json',dataset_path,parent]
    extras.extend(Path(p) for p in parent_payload['identity']['files'])
    extras.extend(Path(p) for p in meta['teacher_reference_evidence']['files'])
    for e in meta['episodes']:
        extras.extend(Path(e[k]) for k in ('path','identity_path','report_path'))
        extras.extend(Path(p) for p in e.get('reused_original_files',{}))
    ident=capture(c,dict(mode='offline_target_feedback_bc',cli=vars(args),pair_config=config,
                        pair_selection=selection,parent_sha256=sha256(parent),
                        dataset_sha256=sha256(dataset_path),warmstart='model only; fresh BC Adam and RNG'),extra_paths=extras)
    atomic_json(out/'IDENTITY.json',ident)
    curve=[]
    for step in range(args.steps+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            total,_,_=losses(model,data['train'],pairs['train'],args.pair_weight)
            if not torch.isfinite(total):
                raise FloatingPointError('nonfinite corrective BC loss')
            total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.actor.parameters()):
                raise FloatingPointError('nonfinite corrective BC gradient')
            optimizer.step()
        with torch.no_grad():
            row={'step':step}
            for split in ('train','validation'):
                total,base,paired=losses(model,data[split],pairs[split],args.pair_weight)
                row.update({split+'_total':float(total),split+'_MSE':float(base),split+'_pair_MSE':float(paired),
                            split+'_normalized_directional_error':float(paired)/(2*SLOPE*args.epsilon)**2})
        curve.append(row)
        atomic_json(out/'CURVE.json',dict(curve=curve,interpretation='offline synthetic pairs only; no gait claim'))
        if step==0 or step==args.steps or step%25==0:
            print(json.dumps(row),flush=True)
    verify(ident)
    trainer=sw.make_trainer(model)
    extra=dict(kind='offline_target_feedback_bc',bc_optimizer=optimizer.state_dict(),bc_config=vars(args),
               curve=curve,dataset_metadata=meta,dataset_path=str(dataset_path),dataset_sha256=sha256(dataset_path),
               collection_identities=[json.loads(Path(e['identity_path']).read_text()) for e in meta['episodes']],
               parent_checkpoint=str(parent),parent_sha256=sha256(parent),parent_identity=parent_payload['identity'],
               pair_config=config,pair_selection=selection,actor_only_bc=True,ppo_optimizers='fresh',
               warmstart='model only; fresh BC Adam and RNG',deployment_transform='none; raw FF238 unchanged',
               interpretation='learned corrective labels; offline derivatives do not prove physical stability')
    save_checkpoint(out/'final.pt',trainer,ident,extra=extra)
    loaded,payload=sw.load_student(out/'final.pt',c)
    exact=all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    if not exact:
        raise ValueError('checkpoint weight reload mismatch')
    result=dict(checkpoint=str(out/'final.pt'),checkpoint_sha256=sha256(out/'final.pt'),steps=args.steps,
                initial=curve[0],final=curve[-1],pair_selection=selection,checkpoint_reload_exact=exact,
                identity_sha256=ident['identity_sha256'],closed_loop='NOT_EVALUATED')
    atomic_json(out/'RESULT.json',result)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('train')
    t.add_argument('--checkpoint',default=str(PARENT));t.add_argument('--dataset',default=str(BASE))
    t.add_argument('--out',required=True);t.add_argument('--steps',type=int,default=1000)
    t.add_argument('--lr',type=float,default=.0003);t.add_argument('--epsilon',type=float,default=.002)
    t.add_argument('--pair-weight',type=float,default=.1);t.add_argument('--seed',type=int,default=20261331)
    t.add_argument('--interior-max',type=float,default=.8)
    return p

if __name__=='__main__':
    print(json.dumps(train(parser().parse_args())),flush=True)
