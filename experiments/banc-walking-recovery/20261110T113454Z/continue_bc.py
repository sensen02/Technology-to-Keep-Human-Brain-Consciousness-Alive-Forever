"""Fixed-budget S2/S5 FFv2 BC continuation; no environment stepping on any path."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import student_wave as sw
from identity import HERE, atomic_json, capture, sha256, verify, verify_runtime
from ppo import load_checkpoint, save_checkpoint

SOURCE_FILES = sw.SOURCE_FILES + [HERE/'continue_bc.py', HERE/'check_continue_bc.py', HERE/'PREREG.json']
CHECKS = HERE/'BC_CONTINUATION_CHECKS.json'
TOTAL_STEPS = 5000


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def rng_state():
    return dict(numpy=np.random.get_state(), python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def source_hashes():
    return {str(p.resolve()): sha256(p) for p in SOURCE_FILES}


def preflight():
    report = json.loads(CHECKS.read_text())
    if not report['all_passed'] or report['source_hashes'] != source_hashes():
        raise ValueError('continuation checks failed or checked source/PREREG changed')
    for path, digest in report['source_hashes'].items():
        if not Path(path).is_absolute() or sha256(path) != digest:
            raise ValueError('continuation checked source drift: '+path)
    if sha256(report['parent_checkpoint']) != report['parent_checkpoint_sha256']:
        raise ValueError('checked original parent checkpoint changed')
    return report


def optimizer_steps(optimizer, model):
    state = optimizer.state_dict()
    params = list(model.actor.parameters())
    ids = [i for g in state['param_groups'] for i in g['params']]
    if len(state['param_groups']) != 1 or ids != list(range(len(params))) or set(state['state']) != set(ids):
        raise ValueError('BC Adam parameter IDs/order/state coverage mismatch')
    steps = []
    for i, p in zip(ids, params):
        entry = state['state'][i]
        if not p.requires_grad or set(entry) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('BC Adam requires all trainable actor parameters and original Adam state')
        for key in ('exp_avg', 'exp_avg_sq'):
            v = entry[key]
            if v.shape != p.shape or v.dtype != p.dtype or not torch.isfinite(v).all():
                raise ValueError('invalid BC Adam moments: '+key)
        step = float(entry['step'])
        if not math.isfinite(step) or step != int(step) or step < 1:
            raise ValueError('invalid BC Adam step')
        steps.append(int(step))
    if len(set(steps)) != 1:
        raise ValueError('BC Adam bias counters disagree')
    return steps


def build_resume(checkpoint, core=None):
    """Validate the complete original closure, restore full model/PPO/RNG and BC Adam.

    Shared by the executable and the independent two-path resume regression.
    Loading constructs the physical Core only; it never calls reset/step/teacher.
    """
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_absolute():
        raise ValueError('checkpoint must be an absolute path')
    checkpoint = checkpoint.resolve()
    core = sw.Core(adhesion=False) if core is None else core
    model, payload = sw.load_student(checkpoint, core)
    extra = payload.get('extra') or {}
    if extra.get('kind') != 'wave_bc':
        raise ValueError('parent extra.kind must be wave_bc')
    if model.architecture != 'ff238' or model.actor.TRANSFORM != 'physical_mlp_plus_trainable_inhibitory_target_v2':
        raise ValueError('fixed FF target-position v2 task required')
    config = extra['bc_config']
    if config['architecture'] != model.architecture or not math.isfinite(config['lr']) or config['lr'] <= 0:
        raise ValueError('parent BC configuration mismatch')
    curve = deepcopy(extra['curve'])
    if not curve or [r['step'] for r in curve] != list(range(curve[-1]['step']+1)):
        raise ValueError('complete global BC curve required')
    step = curve[-1]['step']
    if config['steps'] != step:
        raise ValueError('parent BC config/curve step mismatch')
    dataset_path = Path(extra['dataset_path'])
    if not dataset_path.is_absolute() or dataset_path.name != 'DATASET.json' or sha256(dataset_path) != extra['dataset_sha256']:
        raise ValueError('original dataset metadata identity mismatch')
    meta = sw.validate_dataset(dataset_path.parent)
    if meta != extra['dataset_metadata']:
        raise ValueError('original dataset metadata differs from checkpoint')
    manifests = [json.loads(Path(e['identity_path']).read_text()) for e in meta['episodes']]
    if manifests != extra['collection_identities']:
        raise ValueError('original collection identities differ from checkpoint')
    for manifest in manifests:
        verify_runtime(manifest, core)
    if not meta['all_episode_conditions_passed'] and not config['allow_failed_teacher_episodes']:
        raise ValueError('original teacher condition policy mismatch')
    episodes = {'train': [], 'validation': []}
    for e in meta['episodes']:
        with np.load(e['path'], allow_pickle=False) as z:
            episodes[e['split']].append((torch.as_tensor(z['X238'].copy(), dtype=torch.float64),
                                        torch.as_tensor(z['Y42'].copy(), dtype=torch.float64)))
    trainer = sw.make_trainer(model, payload['config'])
    load_checkpoint(checkpoint, trainer, payload['identity'], restore_rng=True)
    if trainer.actor_optimizer.state or trainer.critic_optimizer.state or trainer.updates or trainer.accepted_steps:
        raise ValueError('BC checkpoint must retain fresh independent PPO states')
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=config['lr'])
    optimizer.load_state_dict(deepcopy(extra['bc_optimizer']))
    if not exact(optimizer.state_dict(), extra['bc_optimizer']):
        raise ValueError('BC Adam restoration changed parameter IDs, moments, counters or hyperparameters')
    if optimizer_steps(optimizer, model) != [step]*len(list(model.actor.parameters())):
        raise ValueError('parent BC Adam counters disagree with global curve')
    if optimizer.param_groups[0]['lr'] != config['lr']:
        raise ValueError('parent BC Adam learning rate mismatch')
    if not exact(rng_state(), payload['rng']):
        raise ValueError('parent RNG restoration mismatch')
    model.train()
    return dict(checkpoint=checkpoint, core=core, model=model, trainer=trainer, payload=payload,
                optimizer=optimizer, episodes=episodes, curve=curve, step=step)


def validate_task(resume, report):
    """Bind the CLI checkpoint (or its continuation chain) to the checked task."""
    payload, extra = resume['payload'], resume['payload']['extra']
    anchor = report['parent_checkpoint']
    if str(resume['checkpoint']) != anchor:
        closure = payload['identity']['files'].get(anchor, {})
        if closure.get('sha256') != report['parent_checkpoint_sha256']:
            raise ValueError('checkpoint is not descended from the checked parent')
    if (extra['dataset_path'] != report['dataset_path'] or extra['dataset_sha256'] != report['dataset_sha256'] or
            extra.get('original_bc_config', extra['bc_config']) != report['parent_bc_config'] or
            json.dumps(payload['model_identity'], sort_keys=True) != json.dumps(report['model_identity'], sort_keys=True)):
        raise ValueError('continuation would change the checked model, dataset or original BC task')
    return True


def bc_update(model, optimizer, episodes):
    optimizer.zero_grad(set_to_none=True)
    loss = sw.bounded_bc_loss(model, episodes)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite BC loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.actor.parameters()):
        raise FloatingPointError('missing or nonfinite BC actor gradient')
    optimizer.step()


def measure(model, episodes, step):
    with torch.no_grad():
        return dict(step=step, train_MSE=float(sw.bounded_bc_loss(model, episodes['train'])),
                    heldout_MSE=float(sw.bounded_bc_loss(model, episodes['validation'])))


def frozen_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if k == 'raw_logstd' or k.startswith('critic.')}


def write_curve(out, curve, parent):
    atomic_json(out/'CURVE.json', dict(parent_checkpoint=str(parent), curve=curve,
                interpretation='supervised bounded action MSE; no closed-loop walking claim'))
    with (out/'CURVE.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'train_MSE', 'heldout_MSE'])
        writer.writeheader()
        writer.writerows(curve)


def continuation(args):
    if args.total_steps != TOTAL_STEPS or args.checkpoint_every < 1:
        raise ValueError('this preregistered continuation requires total-steps 5000 and positive checkpoint interval')
    torch.set_num_threads(1)
    report = preflight()
    r = build_resume(args.checkpoint)
    validate_task(r, report)
    if r['step'] >= args.total_steps:
        raise ValueError('total steps must exceed parent BC step')
    out = sw.exclusive_out(args.out)
    parent = r['checkpoint']
    parent_identity = parent.parent/'IDENTITY.json'
    if json.loads(parent_identity.read_text()) != r['payload']['identity']:
        raise ValueError('parent identity file differs from checkpoint')
    # Preserve every original dependency, rather than inventing a replacement dataset closure.
    extra_paths = list(r['payload']['identity']['files']) + SOURCE_FILES + [CHECKS, parent, parent_identity]
    ident = capture(r['core'], dict(mode='bc_continuation', cli=vars(args),
                    parent_checkpoint=str(parent), parent_checkpoint_sha256=sha256(parent),
                    parent_identity_sha256=r['payload']['identity']['identity_sha256'],
                    parent_step=r['step'], original_bc_config=r['payload']['extra']['bc_config'],
                    dataset_path=r['payload']['extra']['dataset_path'],
                    dataset_sha256=r['payload']['extra']['dataset_sha256']), extra_paths=extra_paths)
    atomic_json(out/'IDENTITY.json', ident)
    verify_runtime(ident, r['core'])
    if not exact(rng_state(), r['payload']['rng']):
        raise ValueError('identity capture consumed RNG')
    model, optimizer, episodes = r['model'], r['optimizer'], r['episodes']
    frozen = frozen_state(model)
    initial = measure(model, episodes, r['step'])
    if initial != r['curve'][-1]:
        raise ValueError('parent MSE does not reproduce exactly')
    print(json.dumps(initial), flush=True)
    curve = r['curve']
    parent_record = dict(checkpoint=str(parent), checkpoint_sha256=sha256(parent), step=r['step'],
                         identity_sha256=r['payload']['identity']['identity_sha256'])
    saved = []
    for step in range(r['step']+1, args.total_steps+1):
        bc_update(model, optimizer, episodes['train'])
        curve.append(measure(model, episodes, step))
        write_curve(out, curve, parent)
        if step % 250 == 0 or step == args.total_steps:
            print(json.dumps(curve[-1]), flush=True)
        if step % args.checkpoint_every == 0 or step == args.total_steps:
            if not exact(frozen_state(model), frozen):
                raise ValueError('BC changed critic or exploration standard deviation')
            if optimizer_steps(optimizer, model) != [step]*len(list(model.actor.parameters())):
                raise ValueError('BC Adam counter drift')
            verify_runtime(ident, r['core'])
            if not exact(model.identity(), r['payload']['model_identity']):
                raise ValueError('model identity changed')
            extra = deepcopy(r['payload']['extra'])
            extra.update(kind='wave_bc', bc_optimizer=optimizer.state_dict(), curve=curve,
                         parent=parent_record, global_step=step, rng_restored=True,
                         original_bc_config=extra.get('original_bc_config', extra['bc_config']))
            extra['bc_config'] = dict(extra['bc_config'], out=str(out), steps=step,
                                      parent_checkpoint=str(parent), total_steps=args.total_steps)
            path = out/f'checkpoint_{step:06d}.pt'
            if path.exists():
                raise FileExistsError(path)
            # Independent unused PPO states; BC Adam remains in extra exactly as before.
            trainer = sw.make_trainer(model, r['payload']['config'])
            save_checkpoint(path, trainer, ident, extra=extra)
            saved.append(dict(path=str(path), sha256=sha256(path), step=step))
            if step == args.total_steps:
                save_checkpoint(out/'final.pt', trainer, ident, extra=extra)
    final_rng = deepcopy(rng_state())
    check = build_resume(out/'final.pt')  # independent fresh model/trainer/load, not in-memory reuse
    reload_exact = (exact(model.state_dict(), check['model'].state_dict()) and
                    exact(optimizer.state_dict(), check['optimizer'].state_dict()) and
                    exact(final_rng, rng_state()) and check['step'] == args.total_steps and
                    exact(measure(check['model'], check['episodes'], check['step']), curve[-1]))
    if not reload_exact or not exact(frozen_state(check['model']), frozen):
        raise ValueError('independent final reload mismatch')
    result = dict(checkpoint=str(out/'final.pt'), checkpoint_sha256=sha256(out/'final.pt'),
                  global_step=args.total_steps, parent_step=r['step'], parent=parent_record,
                  actor_optimizer_steps=optimizer_steps(optimizer, model),
                  critic_unchanged=True, std_unchanged=True, checkpoint_reload_exact=reload_exact,
                  initial=initial, final=curve[-1], rng_restored=True, final_rng_reload_exact=True,
                  ppo_actor_state_entries=len(check['trainer'].actor_optimizer.state),
                  ppo_critic_state_entries=len(check['trainer'].critic_optimizer.state),
                  identity_sha256=ident['identity_sha256'], checkpoints=saved,
                  physical_steps=0, closed_loop='NOT_EVALUATED')
    atomic_json(out/'RESULT.json', result)
    print(json.dumps({k: result[k] for k in ('checkpoint', 'global_step', 'parent_step', 'initial', 'final', 'checkpoint_reload_exact')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--total-steps', required=True, type=int)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    return continuation(parser.parse_args())


if __name__ == '__main__':
    main()
