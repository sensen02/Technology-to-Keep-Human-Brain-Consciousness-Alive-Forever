"""Meaningful exact-resume regression and read-only real-parent preflight; no gait."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import mujoco
import numpy as np
import torch

import continue_bc as cb
import student_wave as sw
from identity import HERE, atomic_json, sha256, verify
from ppo import save_checkpoint

PARENT = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T111240Z/BC_FF_TARGET_1000/final.pt')


def main():
    torch.set_num_threads(1)
    checks = []

    def check(name, condition):
        condition = bool(condition)
        checks.append(dict(name=name, passed=condition))
        if not condition:
            raise AssertionError(name)
        print('PASS '+name, flush=True)

    def reject(name, call):
        try:
            call()
        except (ValueError, RuntimeError, KeyError, FileNotFoundError):
            check(name, True)
        else:
            check(name, False)

    def forbidden(*args, **kwargs):
        raise AssertionError('physical stepping prohibited in BC continuation checks')

    hashes = cb.source_hashes()  # captured before any checking/training
    prereg = json.loads((HERE/'PREREG.json').read_text())
    check('PREREG fixed parent exists before source capture', prereg['parent_checkpoint'] == str(PARENT))
    parent_hash = sha256(PARENT)
    with patch.object(mujoco, 'mj_step', forbidden), patch.object(sw.Core, 'tick_policy', forbidden), patch.object(sw.Core, 'tick_rate', forbidden):
        with patch.object(torch.optim.Adam, 'step', forbidden):
            real = cb.build_resume(PARENT)
        check('real parent read-only preflight global step 1000', real['step'] == 1000)
        check('real parent BC Adam all bias counters 1000', cb.optimizer_steps(real['optimizer'], real['model']) == [1000]*7)
        check('real parent all Adam IDs/moments/hyperparameters exactly restored',
              cb.exact(real['optimizer'].state_dict(), real['payload']['extra']['bc_optimizer']))
        check('real parent python/numpy/torch RNG exactly restored', cb.exact(cb.rng_state(), real['payload']['rng']))
        check('real parent critic/std exactly loaded without update', cb.exact(real['model'].state_dict(), real['payload']['state']['model']))
        check('real parent physical H and transform unchanged', cb.exact(real['model'].identity(), real['payload']['model_identity']))
        check('real parent independent empty PPO states', not real['trainer'].actor_optimizer.state and not real['trainer'].critic_optimizer.state and
              real['trainer'].actor_optimizer is not real['optimizer'] and real['trainer'].critic_optimizer is not real['optimizer'])
        initial = cb.measure(real['model'], real['episodes'], 1000)
        check('real parent exact initial fullbatch train/heldout MSE', initial == real['curve'][-1])
        check('real parent measured loss consumes no RNG', cb.exact(cb.rng_state(), real['payload']['rng']))
        task = dict(parent_checkpoint=str(PARENT), parent_checkpoint_sha256=parent_hash,
                    parent_bc_config=real['payload']['extra']['bc_config'],
                    dataset_path=real['payload']['extra']['dataset_path'],
                    dataset_sha256=real['payload']['extra']['dataset_sha256'],
                    model_identity=real['payload']['model_identity'])
        check('CLI scope accepts checked original task', cb.validate_task(real, task))
        wrong_task = deepcopy(task)
        wrong_task['dataset_path'] = '/tmp/unrelated/DATASET.json'
        reject('CLI scope rejects different frozen dataset', lambda: cb.validate_task(real, wrong_task))
        wrong_anchor = dict(real, checkpoint=Path('/tmp/unrelated/final.pt'))
        reject('CLI scope rejects unrelated parent ancestry', lambda: cb.validate_task(wrong_anchor, task))
        h = real['model'].actor.initial_halfspan.detach().numpy().copy()

        def seeded_path():
            sw.seed_all(73123)
            model = sw.ActorCritic('ff238', critic_obs_dim=255, dtype=torch.float64, actor_halfspan=h)
            trainer = sw.make_trainer(model, real['payload']['config'])
            optimizer = torch.optim.Adam(model.actor.parameters(), lr=.001)
            # Genuine bounded BC loss; synthetic raw 238 observations and 42 targets.
            x = torch.randn(13, 238, dtype=torch.float64)*.1
            y = torch.randn(13, 42, dtype=torch.float64).tanh()*.2
            episodes = [(x, y)]
            frozen = cb.frozen_state(model)
            gain = model.actor.raw_gain.detach().clone()
            curve = [dict(step=0, train_MSE=float(sw.bounded_bc_loss(model, episodes).detach()), heldout_MSE=0.)]
            for step in (1, 2):
                cb.bc_update(model, optimizer, episodes)
                curve.append(dict(step=step, train_MSE=float(sw.bounded_bc_loss(model, episodes).detach()), heldout_MSE=0.))
            return model, trainer, optimizer, episodes, frozen, gain, curve

        direct, trainer, optimizer, episodes, frozen, initial_gain, curve = seeded_path()
        snapshot = deepcopy(direct.state_dict())
        opt_snapshot = deepcopy(optimizer.state_dict())
        rng_snapshot = deepcopy(cb.rng_state())
        repeat, repeat_trainer, repeat_opt, repeat_episodes, _, _, _ = seeded_path()
        check('independent identical-seed two-step actor path exactly reproducible', cb.exact(snapshot, repeat.state_dict()))
        check('independent identical-seed Adam moments/counters reproducible', cb.exact(opt_snapshot, repeat_opt.state_dict()))
        check('independent identical-seed RNG reproducible', cb.exact(rng_snapshot, cb.rng_state()))
        check('all actor parameters including inhibitory gain trained', not torch.equal(initial_gain, direct.actor.raw_gain) and cb.optimizer_steps(optimizer, direct) == [2]*7)
        check('two synthetic updates leave critic/std bitexact', cb.exact(frozen, cb.frozen_state(direct)))

        with tempfile.TemporaryDirectory(prefix='bc-continuation-check-') as temporary:
            root = Path(temporary)
            extra = deepcopy(real['payload']['extra'])
            extra.update(bc_optimizer=optimizer.state_dict(), curve=curve)
            extra['bc_config'] = dict(extra['bc_config'], steps=2)
            checkpoint = root/'two_step.pt'
            save_checkpoint(checkpoint, trainer, real['payload']['identity'], extra=extra)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            check('full checkpoint captures current python/numpy/torch RNG', cb.exact(saved['rng'], rng_snapshot))
            check('full checkpoint fresh PPO states independent of BC Adam', not saved['state']['actor']['state'] and not saved['state']['critic']['state'] and
                  bool(saved['extra']['bc_optimizer']['state']))

            def draws():
                return random.random(), np.random.random(4), torch.rand(4, dtype=torch.float64)

            direct_draws = draws()
            cb.bc_update(direct, optimizer, episodes)
            direct_rng = deepcopy(cb.rng_state())
            direct_opt = deepcopy(optimizer.state_dict())
            direct_state = deepcopy(direct.state_dict())
            # Deliberately perturb all RNG streams; build_resume must overwrite each.
            random.seed(17)
            np.random.seed(18)
            torch.manual_seed(19)
            resumed = cb.build_resume(checkpoint, core=real['core'])
            check('shared production build_resume restores step-two Adam without reset',
                  cb.exact(resumed['optimizer'].state_dict(), opt_snapshot) and cb.optimizer_steps(resumed['optimizer'], resumed['model']) == [2]*7)
            check('shared production build_resume restores every RNG after perturbation', cb.exact(cb.rng_state(), rng_snapshot))
            check('next independent random draws match direct path', cb.exact(draws(), direct_draws))
            cb.bc_update(resumed['model'], resumed['optimizer'], episodes)
            check('direct-vs-reloaded next update actor weights bitexact', cb.exact(direct.actor.state_dict(), resumed['model'].actor.state_dict()))
            check('direct-vs-reloaded entire model bitexact', cb.exact(direct_state, resumed['model'].state_dict()))
            check('direct-vs-reloaded Adam moments/counters/IDs bitexact', cb.exact(direct_opt, resumed['optimizer'].state_dict()))
            check('direct-vs-reloaded Adam bias counters advance 2 to 3', cb.optimizer_steps(resumed['optimizer'], resumed['model']) == [3]*7)
            check('direct-vs-reloaded post-update RNG bitexact', cb.exact(direct_rng, cb.rng_state()))
            check('continued critic/std unchanged', cb.exact(frozen, cb.frozen_state(resumed['model'])))
            check('continued PPO remains fresh and independent', not resumed['trainer'].actor_optimizer.state and not resumed['trainer'].critic_optimizer.state)

            # Independent second save/load uses the exact production continuation schema.
            chain_curve = curve + [dict(step=3, train_MSE=float(sw.bounded_bc_loss(resumed['model'], episodes).detach()), heldout_MSE=0.)]
            chain_extra = deepcopy(extra)
            chain_extra.update(bc_optimizer=resumed['optimizer'].state_dict(), curve=chain_curve, global_step=3,
                               parent=dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), step=2))
            chain_extra['bc_config'] = dict(chain_extra['bc_config'], steps=3)
            chain = root/'chain_step3.pt'
            save_checkpoint(chain, sw.make_trainer(resumed['model'], saved['config']), saved['identity'], extra=chain_extra)
            fresh = cb.build_resume(chain, core=real['core'])
            check('wave_bc continuation checkpoint supports another exact resume',
                  fresh['step'] == 3 and cb.exact(fresh['model'].state_dict(), direct_state) and cb.exact(fresh['optimizer'].state_dict(), direct_opt))

            def invalid(name, mutate):
                altered = deepcopy(extra)
                mutate(altered)
                path = root/(name+'.pt')
                save_checkpoint(path, trainer, saved['identity'], extra=altered)
                reject(name, lambda: cb.build_resume(path, core=real['core']))

            invalid('reject_non_wave_bc_parent', lambda e: e.update(kind='ppo'))
            invalid('reject_BC_Adam_counter_curve_mismatch', lambda e: e['bc_optimizer']['state'][0].update(step=torch.tensor(1.)))
            invalid('reject_missing_BC_Adam_moments', lambda e: e['bc_optimizer']['state'][0].pop('exp_avg'))
            invalid('reject_BC_Adam_parameter_ID_change', lambda e: e['bc_optimizer']['param_groups'][0].update(params=list(reversed(e['bc_optimizer']['param_groups'][0]['params']))))
            invalid('reject_dataset_hash_tamper', lambda e: e.update(dataset_sha256='0'*64))
            invalid('reject_embedded_dataset_metadata_tamper', lambda e: e['dataset_metadata'].update(complete=False))
            invalid('reject_BC_lr_override', lambda e: e['bc_config'].update(lr=.002))
            invalid('reject_architecture_override', lambda e: e['bc_config'].update(architecture='gru238'))
            invalid('reject_missing_global_curve_entries', lambda e: e.update(curve=e['curve'][1:]))
            # A newly hashed metadata file still cannot relabel changed episode bytes.
            bad_meta = deepcopy(extra['dataset_metadata'])
            bad_meta['episodes'][0]['sha256'] = '0'*64
            bad_dataset_dir = root/'bad_dataset'
            bad_dataset_dir.mkdir()
            atomic_json(bad_dataset_dir/'DATASET.json', bad_meta)
            def bad_episode(e):
                e.update(dataset_path=str(bad_dataset_dir/'DATASET.json'),
                         dataset_sha256=sha256(bad_dataset_dir/'DATASET.json'), dataset_metadata=bad_meta)
            invalid('reject_episode_data_digest_tamper_even_with_new_metadata_hash', bad_episode)
            manifest = deepcopy(saved['identity'])
            manifest['files'][str(HERE/'continue_bc.py') if str(HERE/'continue_bc.py') in manifest['files'] else next(iter(manifest['files']))]['sha256'] = '0'*64
            reject('reject_manifest_digest_tamper', lambda: verify(manifest))
            # Valid canonical manifest pointing to changed source: catches bytes, not only manifest corruption.
            import hashlib
            test_source = root/'source.py'
            test_source.write_text('original\n')
            source_manifest = deepcopy(saved['identity'])
            source_manifest['files'][str(test_source)] = dict(sha256=sha256(test_source), size=test_source.stat().st_size)
            canonical = {k: v for k, v in source_manifest.items() if k != 'identity_sha256'}
            source_manifest['identity_sha256'] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            check('valid source closure accepted before tamper', verify(source_manifest))
            test_source.write_text('tampered\n')
            reject('reject_changed_source_bytes_under_valid_manifest', lambda: verify(source_manifest))
            # Mutate a checkpoint byte and verify the recorded file closure rejects it.
            checkpoint_manifest = deepcopy(source_manifest)
            checkpoint_manifest['files'].pop(str(test_source))
            checkpoint_manifest['files'][str(checkpoint)] = dict(sha256=sha256(checkpoint), size=checkpoint.stat().st_size)
            canonical = {k: v for k, v in checkpoint_manifest.items() if k != 'identity_sha256'}
            checkpoint_manifest['identity_sha256'] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            check('valid parent checkpoint closure accepted before tamper', verify(checkpoint_manifest))
            with checkpoint.open('r+b') as stream:
                stream.seek(-1, 2)
                old = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([old[0] ^ 1]))
            reject('reject_changed_parent_checkpoint_bytes', lambda: verify(checkpoint_manifest))

        check('real parent checkpoint remains untouched', sha256(PARENT) == parent_hash)
        check('checked sources/PREREG remain fixed throughout checks', cb.source_hashes() == hashes)
        report = dict(schema_version=1, all_passed=all(c['passed'] for c in checks), checks=checks,
                      source_hashes=hashes, parent_checkpoint=str(PARENT), parent_checkpoint_sha256=parent_hash,
                      parent_bc_config=task['parent_bc_config'], dataset_path=task['dataset_path'],
                      dataset_sha256=task['dataset_sha256'], model_identity=task['model_identity'],
                      parent_step=real['step'], initial=initial, real_parent_optimization_updates=0,
                      synthetic_steps_before_save=2, synthetic_next_step=3, physical_steps=0,
                      main_5000_training_executed=False, threads=torch.get_num_threads(),
                      interpretation='exact resume/identity regression only; does not establish walking')
        atomic_json(cb.CHECKS, report)
        print(json.dumps(dict(all_passed=report['all_passed'], checks=len(checks), report=str(cb.CHECKS))), flush=True)
        return report


if __name__ == '__main__':
    main()
