"""Conditional S2/S5 privileged representation supervision; replacement ANN only.

Labels are teacher provenance loss targets, never observations or policy outputs.
No gait execution on import, construction, checks, or training. Deployment remains
student_wave.load_student / actor.step / tanh; this module is not a deployment API.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import student_wave as sw
from identity import HERE, atomic_json, capture, verify, sha256
from ppo import save_checkpoint, load_checkpoint

DATASET = HERE.parent/'20261110T094051Z/SMOOTH_DATASET'
AUX_SEED_OFFSET = 6001
OWN_SOURCES = [HERE/'role_aux_bc.py', HERE/'check_role_aux.py']


def source_hashes():
    return {str(p.resolve()): sha256(p) for p in sw.SOURCE_FILES + OWN_SOURCES}


def preflight():
    report = json.loads((HERE/'ROLE_AUX_CHECKS.json').read_text())
    if not report['all_passed']:
        raise ValueError('role auxiliary checks failed')
    for p, digest in report['source_hashes'].items():
        if not Path(p).is_absolute() or sha256(p) != digest:
            raise ValueError('role auxiliary source drift: '+p)
    return True


def role_labels(decision_indices):
    """Integer sample index used exclusively as supervised target provenance."""
    idx = torch.as_tensor(decision_indices)
    if idx.dtype not in (torch.int32, torch.int64) or idx.ndim != 1 or (idx < 0).any():
        raise ValueError('nonnegative integer decision indices required')
    return torch.div(idx, 160, rounding_mode='floor').remainder(6).long()


def sequence_with_hidden(actor, x):
    """Whole contiguous (T,238) episode, explicit zero state, no internal resets."""
    if x.ndim != 2 or x.shape[1] != 238 or len(x) == 0:
        raise ValueError('nonempty contiguous (T,238) observations required')
    state = actor.initial_state(1)
    means, states = [], []
    for obs in x:
        mean, state = actor.step(obs[None], state)
        means.append(mean)
        states.append(state)
    return torch.stack(means)[:, 0], torch.stack(states)[:, 0]


def new_aux_head(seed):
    # Called AFTER sw.new_model: restores the baseline global RNG exactly.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + AUX_SEED_OFFSET)
        return nn.Linear(128, 6, dtype=torch.float64)


def new_optimizer(model, head, lr):
    return torch.optim.Adam(list(model.actor.parameters()) + list(head.parameters()), lr=lr)


def objective(model, head, episodes, role_weight):
    """Episode sample weighted BC and CE, full continuous BPTT, train only."""
    if not math.isfinite(role_weight) or role_weight < 0:
        raise ValueError('finite nonnegative role weight required')
    if role_weight == 0:
        # Exact original graph/accumulation; head gradients remain None.
        mse = sw.bounded_bc_loss(model, [(x, y) for x, y, labels in episodes])
        return mse, mse, mse.new_zeros(())
    samples = sum(len(x) for x, y, labels in episodes)
    mse_terms, role_terms = [], []
    for x, y, labels in episodes:
        mean, states = sequence_with_hidden(model.actor, x)
        mse_terms.append((mean.tanh()-y).square().sum())
        role_terms.append(F.cross_entropy(head(states), labels, reduction='sum'))
    mse = sum(mse_terms)/(samples*42)
    ce = sum(role_terms)/samples
    return mse + role_weight*ce, mse, ce


@torch.no_grad()
def metrics(model, head, episodes):
    n = sum(len(x) for x, y, labels in episodes)
    squared, ce, correct = 0., 0., 0
    for x, y, labels in episodes:
        mean, states = sequence_with_hidden(model.actor, x)
        logits = head(states)
        squared += float((mean.tanh()-y).square().sum())
        ce += float(F.cross_entropy(logits, labels, reduction='sum'))
        correct += int((logits.argmax(-1) == labels).sum())
    return dict(MSE=squared/(n*42), role_CE=ce/n, role_accuracy=correct/n)


def validate_role_dataset(directory):
    directory = Path(directory)
    if not directory.is_absolute() or directory.resolve() != DATASET.resolve():
        raise ValueError('requires the preregistered absolute SMOOTH_DATASET path')
    meta = sw.validate_dataset(directory)
    if meta['teacher_reference_evidence']['condition'] != sw.teacher_config(2):
        raise ValueError('exact .8/.36/.12/.01 teacher condition required')
    entries = meta['episodes']
    if [(e['episode'], e['split'], e['samples']) for e in entries] != [(0,'train',1920),(1,'train',1920),(2,'validation',1920)]:
        raise ValueError('requires independent train0/1, validation2, 2*960 samples')
    if not meta['all_episode_conditions_passed'] or meta['supervision'] != dict(gait_duration_s=9.6, physical_landing_tail_s=.4, landing_tail_excluded=True, samples_per_episode=1920):
        raise ValueError('teacher conditions or diagnostic-tail exclusion mismatch')
    core = sw.Core(adhesion=False)
    before_time = float(core.d.time)
    teacher = sw.StaticTeacher(core, sw.teacher_config(2))
    ref = teacher.reference.metadata
    order = list(ref['cfg']['order'])
    if order != ['RM','LH','RF','LM','RH','LF'] or order != ref['default_order'] or order != list(teacher.reference.order):
        raise ValueError('actual passed teacher order mismatch')
    if float(core.d.time) != before_time:
        raise ValueError('reference construction advanced physics')
    files = dict(meta['teacher_reference_evidence']['files'])
    files[str(Path(ref['source_path']).resolve())] = ref['source_sha256']
    for e in entries:
        original = e['reused_original_files']
        manifest_path = next(p for p in original if Path(p).name == 'IDENTITY.json')
        manifest = json.loads(Path(manifest_path).read_text())
        cfg = {**sw.StaticTeacher(core).config, **manifest['config']['config']}
        if any(cfg[k] != v for k,v in sw.teacher_config(2).items()):
            raise ValueError('original passed teacher condition mismatch')
        for name in ('static_teacher.py','wave_reference.py'):
            records = [v['sha256'] for p,v in manifest['files'].items() if Path(p).name == name]
            if sha256(HERE/name) not in records:
                raise ValueError('original teacher executable mapping drift: '+name)
        sample_path = next(p for p in original if Path(p).name == 'BC_SAMPLES.npz')
        with np.load(sample_path, allow_pickle=False) as z, np.load(e['path'], allow_pickle=False) as data:
            phase = z['source_phase']
            if phase.shape != (2000,) or not np.allclose(phase, np.arange(2000)*.005, rtol=0, atol=1e-10):
                raise ValueError('original teacher integer decision continuity mismatch')
            for original_key, data_key in [('obs','X238'),('action','Y42'),('critic_obs','V255')]:
                if not np.array_equal(z[original_key][:1920], data[data_key]):
                    raise ValueError('original contiguous gait slice mismatch')
        files.update(original)
    provenance = dict(kind='privileged teacher planned wave-role loss target ONLY',
        label_formula='floor(integer_decision_index/160)%6', decision_ms=5,
        role_period_ms=800, samples_per_cycle=960, classes=list(range(6)),
        class_to_teacher_leg=order, actual_passed_order_path="StaticTeacher.reference.metadata['cfg']['order']",
        reference_metadata=ref, teacher_config=sw.teacher_config(2),
        constructor_only_physics_steps=0, dataset_path=str((directory/'DATASET.json').resolve()),
        dataset_sha256=sha256(directory/'DATASET.json'), source_and_original_teacher_files=files,
        actor_input=False, initial_hidden=False, hidden_reset=False, action_gating=False,
        critic_input=False, deployment_phase=False, heldout_labels='report accuracy only; never optimize')
    return meta, provenance


def load_episodes(meta):
    episodes = dict(train=[], validation=[])
    for e in meta['episodes']:
        with np.load(e['path'], allow_pickle=False) as z:
            x, y = (torch.as_tensor(z[k].copy(), dtype=torch.float64) for k in ('X238','Y42'))
        episodes[e['split']].append((x, y, role_labels(torch.arange(len(x)))))
    return episodes


def identity_paths(meta, provenance):
    paths = sw.SOURCE_FILES + OWN_SOURCES + [HERE/'ROLE_AUX_CHECKS.json', Path(provenance['dataset_path'])]
    paths += [Path(p) for p in provenance['source_and_original_teacher_files']]
    paths += [Path(p) for p in meta['source_hashes']]
    for e in meta['episodes']:
        paths += [Path(e[k]) for k in ('path','identity_path','report_path')]
    return paths


def head_digest(head):
    h = hashlib.sha256()
    for k,v in sorted(head.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def checkpoint_extra(model, head, optimizer, args, meta, provenance, curve):
    config = dict(vars(args), aux_seed_offset=AUX_SEED_OFFSET,
                  aux_seed=args.seed+AUX_SEED_OFFSET, head='training-only Linear(128,6), outside ActorCritic',
                  loss='bounded BC sum/(T*42) + role_weight * CE sum/T',
                  full_contiguous_BPTT=True, train_episodes=[0,1], validation_episodes=[2])
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    return dict(kind='role_aux_bc', aux_head_state_dict=head.state_dict(),
        aux_head_sha256=head_digest(head), aux_config_sha256=digest,
        aux_bc_optimizer=optimizer.state_dict(), aux_bc_config=config, curve=curve,
        dataset_metadata=meta, dataset_path=provenance['dataset_path'], dataset_sha256=provenance['dataset_sha256'],
        collection_identities=[json.loads(Path(e['identity_path']).read_text()) for e in meta['episodes']],
        teacher_reference_provenance=provenance, actor_only_bc=True, std_initial=.05,
        ppo_optimizers='fresh', deployment_aux_head=False,
        integrity='source/config identity and externally recorded full checkpoint SHA256; load_student ignores auxiliary extra',
        interpretation='privileged representation supervision for replacement ANN; no original-brain or autonomous claim')


def train(args):
    if args.steps != 300 or args.lr != .001 or args.seed != 20261110:
        raise ValueError('fixed scratch comparison requires steps300 lr.001 seed20261110')
    if not math.isfinite(args.role_weight) or args.role_weight < 0:
        raise ValueError('finite nonnegative role weight required (zero control permitted)')
    preflight()
    meta, provenance = validate_role_dataset(args.dataset)
    out = sw.exclusive_out(args.out)
    sw.seed_all(args.seed)
    core = sw.Core(adhesion=False)
    ident = capture(core, dict(mode='role_aux_bc', cli=vars(args), teacher_reference_provenance=provenance,
        dataset_sha256=provenance['dataset_sha256'], aux_seed_offset=AUX_SEED_OFFSET),
        extra_paths=identity_paths(meta, provenance))
    atomic_json(out/'IDENTITY.json', ident)
    episodes = load_episodes(meta)
    model = sw.new_model('gru238')
    head = new_aux_head(args.seed)
    optimizer = new_optimizer(model, head, args.lr)
    untouched = {k:v.clone() for k,v in model.state_dict().items() if not k.startswith('actor.')}
    curve=[]
    for step in range(args.steps+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss, mse, ce = objective(model, head, episodes['train'], args.role_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite auxiliary BC loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params']):
                raise FloatingPointError('nonfinite auxiliary BC gradient')
            optimizer.step()
        row = dict(step=step, train=metrics(model, head, episodes['train']), heldout=metrics(model, head, episodes['validation']))
        row['train_loss'] = row['train']['MSE']+args.role_weight*row['train']['role_CE']
        curve.append(row)
        atomic_json(out/'CURVE.json', dict(curve=curve, closed_loop='NOT_EVALUATED'))
        if step == 0 or step == args.steps or step%25 == 0:
            print(json.dumps(row), flush=True)
    if not all(torch.equal(v, model.state_dict()[k]) for k,v in untouched.items()):
        raise ValueError('critic/std changed during actor-only supervision')
    verify(ident)
    save_checkpoint(out/'final.pt', sw.make_trainer(model), ident,
                    extra=checkpoint_extra(model, head, optimizer, args, meta, provenance, curve))
    loaded, payload = sw.load_student(out/'final.pt', core)
    if not all(torch.equal(v, loaded.state_dict()[k]) for k,v in model.state_dict().items()):
        raise ValueError('unchanged deployment loader reload mismatch')
    result = dict(checkpoint=str(out/'final.pt'), checkpoint_sha256=sha256(out/'final.pt'),
        steps=args.steps, final=curve[-1], checkpoint_reload_exact=True,
        model_identity=payload['model_identity'], identity_sha256=ident['identity_sha256'],
        aux_config_sha256=payload['extra']['aux_config_sha256'], aux_head_sha256=head_digest(head),
        closed_loop='NOT_EVALUATED', conditional_training='parent decision; baseline holdout never used in optimization')
    atomic_json(out/'RESULT.json', result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    t = sub.add_parser('train')
    t.add_argument('--dataset', required=True); t.add_argument('--out', required=True)
    t.add_argument('--steps', type=int, default=300); t.add_argument('--lr', type=float, default=.001)
    t.add_argument('--seed', type=int, default=20261110); t.add_argument('--role-weight', type=float, default=.001)
    return p


if __name__ == '__main__':
    train(parser().parse_args())
