"""S2/S4: two precomputed local-ray checks, research only (s97).

Derivative of run_native_rk4_gate_training.py (s96), not an optimizer.
Only parent-supplied ray_010 and ray_001; no direction, proposal or update.
All s92/s93/s94/s96 trajectories are read-only exact-byte dedup evidence.
No model compilation, neural or body calls on import or --help.
"""
from pathlib import Path
import argparse
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import traceback

# Reuse only side-effect-free evidence helpers; the full evaluation is below.
from run_native_rk4_gate_training import (
    sha, sync_dir, atomic_json, atomic_npz, numeric_model_snapshot,
    callback_snapshot, require_no_callbacks, raw_observation_record,
    support_spans, acceptance,
)

ROOT = Path(__file__).resolve().parent
MODEL = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
INITIAL = Path('/run/media/sensen/Data2/engine修复验收/s7-posture-ik/20261023T021141Z/DERIVED_INITIAL.npz')
ENDPOINTS = Path('/run/media/sensen/Data2/engine修复验收/s4-endpoints/20261023T011559Z')
BASE_RUN = Path('/home/sensen/Desktop/engine修复验收/s92-rk4-full-baseline/20261023T210730Z/run')
BASELINE = BASE_RUN/'baseline400ms/TRAJECTORY.npz'
DERIVED_FROM = ROOT/'run_native_rk4_gate_training.py'
S94_FROM = ROOT/'run_native_rk4_spsa_second.py'
S93_FROM = ROOT/'run_native_rk4_spsa_round.py'
BASELINE_FROM = ROOT/'run_native_rk4_full_baseline.py'
S96_RUN = Path('/home/sensen/Desktop/engine修复验收/s96-six-gate-training/20261023T214545Z/run_validated')
CACHE_RUNS = (
    Path('/home/sensen/Desktop/engine修复验收/s93-rk4-spsa/20261023T211511Z/run'),
    Path('/home/sensen/Desktop/engine修复验收/s94-rk4-spsa-small/20261023T212515Z/run'),
    S96_RUN)
CACHE_NAMES = ('round00_plus', 'round00_minus', 'round00_updated')
EVAL_NAMES = ('ray_010', 'ray_001')
FACTORS = dict(ray_010=.1, ray_001=.01)
TRAINING_FROM = ROOT/'run_formal_translation_training.py'
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s97-local-ray/20261023T215729Z/run')
OBJECTIVE_MODULE = OUTPUT.parent/'objective.py'
OBJECTIVE_PREREG = OUTPUT.parent/'OBJECTIVE_PREREG.json'
CANDIDATES_NPZ = OUTPUT.parent/'CANDIDATES.npz'
CANDIDATES_JSON = OUTPUT.parent/'CANDIDATES.json'
GATE_KEYS = ('height', 'angle', 'legs', 'load', 'nonfoot', 'limit')
BODY_DT, NEURAL_DT, COMMAND_DT = .0001, .0005, .005
BLOCKS, PHYSICS_PER_BLOCK, SAMPLES = 80, 50, 4001


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT, help='Fixed fresh output: '+str(OUTPUT))
    return p


def run(args):
    import numpy as np
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.embodied.formal_training import score_formal
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork
    from run_neural_muscle_closure import observe_quantities

    out = args.output
    if out != OUTPUT or not out.is_absolute() or out.exists():
        raise ValueError('fixed fresh output required; no automatic resume')
    required_parent = (OUTPUT.parent/'PREREG.json', OBJECTIVE_MODULE, OBJECTIVE_PREREG,
                       CANDIDATES_NPZ, CANDIDATES_JSON)
    for required in required_parent:
        if not required.is_file():
            raise ValueError('required parent input missing: '+str(required))
    # Objective source and immutable scales must be IDENTICAL to s96, not retuned.
    for current, prior in ((OBJECTIVE_MODULE, S96_RUN.parent/'objective.py'),
                           (OBJECTIVE_PREREG, S96_RUN.parent/'OBJECTIVE_PREREG.json')):
        if current.read_bytes() != prior.read_bytes():
            raise ValueError('objective/scales must be byte-identical to s96: '+str(current))
    objective_hashes = {str(p):sha(p) for p in (OBJECTIVE_MODULE, OBJECTIVE_PREREG)}
    scales = json.loads(OBJECTIVE_PREREG.read_text())['scales']
    if (not isinstance(scales, dict) or set(scales) != set(GATE_KEYS) or
            any(isinstance(scales[k], bool) or not isinstance(scales[k], (int, float)) or
                not np.isfinite(scales[k]) or scales[k] <= 0 for k in GATE_KEYS)):
        raise ValueError('six finite positive preregistered scales required')
    spec = importlib.util.spec_from_file_location('s97_offline_objective', str(OBJECTIVE_MODULE))
    if spec is None or spec.loader is None:
        raise ImportError('cannot load preregistered objective module')
    objective = importlib.util.module_from_spec(spec)
    objective_source = OBJECTIVE_MODULE.read_bytes()
    if hashlib.sha256(objective_source).hexdigest() != objective_hashes[str(OBJECTIVE_MODULE)]:
        raise RuntimeError('objective changed while loading')
    exec(compile(objective_source, str(OBJECTIVE_MODULE), 'exec'), objective.__dict__)
    if not callable(getattr(objective, 'proxy', None)) or not callable(getattr(objective, 'violations', None)):
        raise TypeError('objective module must expose proxy and violations')
    if objective_hashes != {str(p):sha(p) for p in (OBJECTIVE_MODULE, OBJECTIVE_PREREG)}:
        raise RuntimeError('objective/scales changed while importing')
    out.mkdir(parents=True, exist_ok=False)
    sync_dir(out.parent)
    lock = (out/'WRITER.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    caps = dict(neural_ticks=1600, physics_steps=8000, initial_forward=2, diagnostic_forward=8000)
    reserved, completed = dict.fromkeys(caps, 0), dict.fromkeys(caps, 0)
    ledger = dict(caps=caps, reserved=reserved, completed=completed, active=None,
        total_forward_cap=8002, recovery='NO_AUTOMATIC_RESUME; uncommitted reservations stay charged',
        scope='public calls only; RK4 native internal stages add work, no neural calls in stages')
    timing = dict(blocks=80, duration_s=.4, body_dt_s=BODY_DT, neural_dt_s=NEURAL_DT,
        command_dt_s=COMMAND_DT, samples=4001, neural_ticks_per_block=10, physics_steps_per_block=50,
        schedule='observe encode -> 10 neural ticks -> old pending 50 physics steps -> consume -> next pending',
        phase='continuous first200ms preparation, last200ms formal; labels only, no reset',
        scoring_windows='preparation samples0:2001; formal samples2000:4001; shared boundary sample',
        FIR='DT=.0005 bins ((0,5),(5,10),(10,20),(20,40),(40,80)); unchanged sensory encoding')
    atomic_json(out/'LEDGER.json', ledger)
    atomic_json(out/'PREREG.json', dict(evaluations_max=2, candidates=list(EVAL_NAMES),
        derived_from=str(DERIVED_FROM), training_from=str(TRAINING_FROM), timing=timing,
        baseline_path=str(BASELINE), baseline_usage='saved s92 read-only; zero rerun calls; original frozen starting weights only',
        candidates_npz=str(CANDIDATES_NPZ), candidates_json=str(CANDIDATES_JSON), factors=FACTORS,
        candidate_formula='clip(original + factor*s96UPDATE.effective_delta, np.finfo(float).eps, 4.); preserve original zero mask; exact bytes',
        optimizer='NONE; only two parent-precomputed candidates evaluated sequentially; no direction/proposal/update',
        objective='hashbound objective.py proxy(arrays,scales): total -> optimization_loss, six parts -> proxy_parts; formal window as preregistered in OBJECTIVE_PREREG.json',
        old_loss='loss remains unchanged formal_training.score_formal; recorded and rescored, no optimizer',
        new_loss='optimization_loss is six-gate proxy total; proxy_parts are six gate contributions; neither loss used for acceptance',
        proxy_series='each arm PROXY_SERIES.npz: time, phase, gate_keys, scales, violations [4001,6], contributions=(violations/scales)^2',
        objective_module=str(OBJECTIVE_MODULE), objective_prereg=str(OBJECTIVE_PREREG),
        objective_hashes=objective_hashes, scales=scales, gate_keys=list(GATE_KEYS),
        acceptance='each arm separately: both-window five safety extrema nonworsening; formal cumulative AND longest nonworsening, at least one strictly improved; tolerance1e-12; loss not acceptance',
        longest='contiguous good samples last-first; isolated good sample0ms',
        duplicates='bitwise identical weights reuse s92 baseline, ALL nine same-config s93/s94/s96 records, or current arms; hashbound results/trajectories/weights; zero calls; disclose; no spare-budget evaluation',
        cache_runs=[str(p) for p in CACHE_RUNS], cache_arms=list(CACHE_NAMES), historical_weights_usage='read-only dedup cache ONLY; never starting weights',
        cold_state='fresh network/FIR/body; initial q only; zero v/act/ctrl/pending; no phase reset',
        sole_model_change='same XML compile once; Euler -> RK4 only; dt .0001; exact s92 before/after',
        prohibited='no new direction/proposal/update, selection, best arm, deployment, hidden controls, pose or act overwrite, callbacks/delay/history/plugin',
        stopping='warning/nonfinite immediately save partial evidence and stop all arms; no autoresume',
        caps=caps, total_forward_cap=8002, behavior_claim='NONE', physical_units='NOT_ESTABLISHED'))
    shared_paths = [MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
        ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
        ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
        ROOT/'run_neural_muscle_closure.py', ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    cache_paths = [cache_run/f for cache_run in CACHE_RUNS for f in
        ('INPUT_HASHES.json', 'SETUP.json', 'MODEL_PROVENANCE.json', 'PREREG.json',
         'DIRECTION.npz', 'UPDATE.npz', 'EVALUATIONS.json', 'LEDGER.json', 'SUMMARY.json')]
    cache_paths += [cache_run/name/f for cache_run in CACHE_RUNS for name in CACHE_NAMES
                    for f in ('RESULT.json', 'TRAJECTORY.npz')]
    paths = shared_paths + [Path(__file__).resolve(), DERIVED_FROM, S94_FROM, S93_FROM, BASELINE_FROM, TRAINING_FROM,
        ROOT/'engine/embodied/formal_training.py', BASE_RUN/'INPUT_HASHES.json', BASE_RUN/'SETUP.json',
        BASE_RUN/'MODEL_PROVENANCE.json', BASE_RUN/'baseline400ms/RESULT.json', BASELINE,
        *required_parent, S96_RUN/'UPDATE.json', S96_RUN.parent/'objective.py',
        S96_RUN.parent/'OBJECTIVE_PREREG.json'] + cache_paths

    def reserve(name, block, delta):
        if ledger['active'] is not None:
            raise RuntimeError('uncommitted reservation; refusing more calls')
        if any(reserved[k]+v > caps[k] for k, v in delta.items()):
            raise RuntimeError('hard call cap exceeded')
        for key, value in delta.items():
            reserved[key] += value
        ledger['active'] = dict(evaluation=name, block=block, delta=delta,
            start_s=max(block, 0)*COMMAND_DT, end_s=(block+1)*COMMAND_DT if block >= 0 else 0.)
        atomic_json(out/'LEDGER.json', ledger)

    def commit():
        for key, value in ledger['active']['delta'].items():
            completed[key] += value
        ledger['active'] = None
        atomic_json(out/'LEDGER.json', ledger)

    try:
        callbacks_before = callback_snapshot(mj)
        require_no_callbacks(callbacks_before)
        atomic_json(out/'CALLBACKS_BEFORE.json', callbacks_before)
        if mj.__version__ != '3.9.0' or mj.mj_versionString() != '3.9.0':
            raise RuntimeError('requires installed/native MuJoCo3.9.0 as s92')
        identities = {str(p): sha(p) for p in paths}
        if any(identities[p] != h for p, h in objective_hashes.items()):
            raise RuntimeError('objective/scales changed before input binding')
        historical = json.loads((BASE_RUN/'INPUT_HASHES.json').read_text())
        actual = {str(p): identities[str(p)] for p in shared_paths}
        if len(actual) != 14 or any(historical[p] != h for p, h in actual.items()):
            raise ValueError('shared14 input/source hashes differ from s92')
        if historical[str(BASELINE_FROM)] != identities[str(BASELINE_FROM)]:
            raise ValueError('s92 baseline source hash differs')
        for cache_run in CACHE_RUNS:
            cache_inputs = json.loads((cache_run/'INPUT_HASHES.json').read_text())
            check_paths = shared_paths + [S93_FROM, BASELINE_FROM, TRAINING_FROM,
                ROOT/'engine/embodied/formal_training.py', BASE_RUN/'INPUT_HASHES.json',
                BASE_RUN/'SETUP.json', BASE_RUN/'MODEL_PROVENANCE.json', BASE_RUN/'baseline400ms/RESULT.json', BASELINE]
            if cache_run in (CACHE_RUNS[1], S96_RUN):
                check_paths.append(S94_FROM)
            if cache_run == S96_RUN:
                check_paths.append(DERIVED_FROM)
            for p in check_paths:
                if cache_inputs[str(p)] != identities[str(p)]:
                    raise ValueError('historical same-config/source identity differs: '+str(cache_run)+' '+str(p))
        atomic_json(out/'INPUT_HASHES.json', identities)
        atomic_json(out/'SHARED_SOURCE_CHECK.json', dict(passed=True, shared=actual,
            baseline_run=str(BASE_RUN), derivative_sha256=identities[str(DERIVED_FROM)],
            baseline_source_sha256=identities[str(BASELINE_FROM)], cache_runs=[str(p) for p in CACHE_RUNS],
            s93_source_equal=True, s94_source_equal=True, s96_source_equal=True, objective_hashes=objective_hashes))
        manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
        original = np.array(json.loads((ENDPOINTS/'FROZEN_WEIGHTS.json').read_text())['weights'], float)
        if (original.shape != (66,5) or not np.isfinite(original).all() or
                np.count_nonzero(original) != 330 or np.any((original <= 0) | (original > 4))):
            raise ValueError('expected original330 positive bounded weights')
        active = original > 0
        # Bind parent-provided candidates to exact saved s96 effective delta.
        candidate_meta = json.loads(CANDIDATES_JSON.read_text())
        update_meta = json.loads((S96_RUN/'UPDATE.json').read_text())
        with np.load(S96_RUN/'UPDATE.npz', allow_pickle=False) as saved_update:
            update_original = saved_update['original'].copy()
            effective_delta = saved_update['effective_delta'].copy()
            update_new = saved_update['new'].copy()
        def exact(a, b):
            return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        if (not exact(update_original, original) or effective_delta.dtype != original.dtype or
                effective_delta.shape != original.shape or not np.isfinite(effective_delta).all() or
                not exact(effective_delta, update_new-original) or
                not exact(np.asarray(update_meta['original'], float), original) or
                not exact(np.asarray(update_meta['effective_delta'], float), effective_delta) or
                not exact(np.asarray(update_meta['new'], float), update_new)):
            raise ValueError('saved s96 UPDATE json/npz original/effective_delta identity mismatch')
        with np.load(CANDIDATES_NPZ, allow_pickle=False) as saved_candidates:
            if set(saved_candidates.files) != {'original', 'effective_delta', *EVAL_NAMES}:
                raise ValueError('CANDIDATES.npz requires exactly original,effective_delta,ray_010,ray_001')
            candidates = {k:saved_candidates[k].copy() for k in saved_candidates.files}
        if not exact(candidates['original'], original) or not exact(candidates['effective_delta'], effective_delta):
            raise ValueError('parent candidates original/effective_delta differ from saved s96 UPDATE')
        candidate_audit = {}
        for name in EVAL_NAMES:
            expected = np.clip(original+FACTORS[name]*effective_delta, np.finfo(float).eps, 4.)
            expected[~active] = 0
            if not exact(candidates[name], expected):
                raise ValueError('parent candidate exact floating formula mismatch: '+name)
            candidate_audit[name] = dict(factor=FACTORS[name], formula_bitwise_verified=True,
                dtype=str(expected.dtype), shape=list(expected.shape),
                weights_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                max_abs_delta=float(np.max(np.abs(expected-original))), zero_mask_preserved=True)
        if not isinstance(candidate_meta, dict):
            raise ValueError('CANDIDATES.json must be an object')
        # JSON is fully hashbound, with any array copies checked when supplied.
        for key, value in candidates.items():
            if key in candidate_meta and isinstance(candidate_meta[key], list):
                if not exact(np.asarray(candidate_meta[key], float), value):
                    raise ValueError('CANDIDATES.json array differs from npz: '+key)
        atomic_json(out/'CANDIDATE_AUDIT.json', dict(passed=True, candidates=candidate_audit,
            source_npz=str(CANDIDATES_NPZ), source_json=str(CANDIDATES_JSON),
            source_npz_sha256=identities[str(CANDIDATES_NPZ)], source_json_sha256=identities[str(CANDIDATES_JSON)],
            s96_update_json_sha256=identities[str(S96_RUN/'UPDATE.json')],
            s96_update_npz_sha256=identities[str(S96_RUN/'UPDATE.npz')],
            original_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
            effective_delta_sha256=hashlib.sha256(effective_delta.tobytes()).hexdigest(),
            parent_metadata=candidate_meta, optimizer_calls=0, new_direction=False, deployment=False))
        with np.load(INITIAL, allow_pickle=False) as initial:
            q, sensory_ref = initial['new_qpos'].copy(), initial['sensory_reference_qpos'].copy()
        if q.shape != (49,) or sensory_ref.shape != (49,) or not np.isfinite([q,sensory_ref]).all():
            raise ValueError('invalid fixed initial/reference')
        baseline_result = json.loads((BASE_RUN/'baseline400ms/RESULT.json').read_text())
        if baseline_result['trajectory_sha256'] != identities[str(BASELINE)]:
            raise ValueError('s92 saved trajectory hash differs from RESULT')
        with np.load(BASELINE, allow_pickle=False) as saved:
            baseline_arrays = {k: saved[k].copy() for k in saved.files}
        if (baseline_arrays['qpos'].shape != (4001,49) or
                baseline_arrays['weights'].tobytes() != original.tobytes() or
                not np.array_equal(baseline_arrays['qpos'][0], q) or
                any(np.any(baseline_arrays[k][0]) for k in ('qvel','act','ctrl')) or
                np.any(baseline_arrays['warnings']) or
                not np.allclose(baseline_arrays['time'], np.arange(SAMPLES)*BODY_DT, rtol=0, atol=1e-12) or
                any(not np.isfinite(a).all() for a in baseline_arrays.values())):
            raise ValueError('s92 baseline weights/cold state/clock/finite identity mismatch')
        provenance = json.loads((BASE_RUN/'MODEL_PROVENANCE.json').read_text())
        m = mj.MjModel.from_xml_path(str(MODEL))
        if ((m.nu,m.nq,m.nv) != (90,49,48) or m.opt.timestep != BODY_DT or
                m.opt.integrator != mj.mjtIntegrator.mjINT_EULER or
                np.count_nonzero(m.geom_type == mj.mjtGeom.mjGEOM_MESH) != 71):
            raise ValueError('unexpected original full body/dt/Euler')
        if np.any(m.actuator_delay) or np.any(m.actuator_history) or m.nplugin != 0 or np.any(m.actuator_plugin != -1):
            raise RuntimeError('delay/history/plugin forbidden')
        atomic_json(out/'NATIVE_INPUT_PATH_CHECK.json', dict(actuator_delay=m.actuator_delay.tolist(),
            actuator_history=m.actuator_history.tolist(), actuator_historyadr=m.actuator_historyadr.tolist(),
            nplugin=int(m.nplugin), actuator_plugin=m.actuator_plugin.tolist(), no_delay_history_plugins=True))
        before = numeric_model_snapshot(m)
        if before != provenance['before'] or provenance['xml_sha256'] != identities[str(MODEL)]:
            raise RuntimeError('compiled before snapshot differs from s92')
        m.opt.integrator = mj.mjtIntegrator.mjINT_RK4  # ONLY model-field assignment.
        after = numeric_model_snapshot(m)
        changed = [k for k in before if before[k] != after[k]]
        if before.keys() != after.keys() or changed != ['opt.integrator'] or after != provenance['after']:
            raise RuntimeError('sole integrator change / exact s92 after snapshot failed')
        atomic_json(out/'MODEL_PROVENANCE.json', dict(xml_path=str(MODEL), xml_sha256=identities[str(MODEL)],
            before=before, after=after, changed_numeric_fields=changed, s92_before_equal=True, s92_after_equal=True,
            original_timestep_s=BODY_DT, runtime_timestep_s=float(m.opt.timestep),
            original_integrator=int(before['opt.integrator']), runtime_integrator=int(m.opt.integrator),
            no_other_model_field_changes=True, baseline_provenance_sha256=identities[str(BASE_RUN/'MODEL_PROVENANCE.json')]))
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        old_setup = json.loads((BASE_RUN/'SETUP.json').read_text())
        if weight_native != old_setup['weight_native'] or old_setup['leg_order'] != list(LEGS):
            raise ValueError('native weight/leg order differs from s92')

        def score_arrays(arrays):
            # Original score_formal stays the OLD score: loss and acceptance unchanged.
            score = score_formal(arrays, weight_native)
            spans = support_spans(arrays, weight_native)
            for phase in spans:
                score[phase]['longest_support_ms'] = spans[phase]
            if not np.isfinite(float(score['loss'])):
                raise ValueError('nonfinite formal score')
            return score

        def proxy_fields(arrays):
            result = objective.proxy(arrays, scales)
            if not isinstance(result, dict) or set(result) != {'total', 'parts'}:
                raise ValueError('proxy must return total and parts')
            parts = result['parts']
            if not isinstance(parts, dict) or set(parts) != set(GATE_KEYS):
                raise ValueError('proxy must return six named parts')
            total = float(result['total'])
            parts = {k:float(parts[k]) for k in GATE_KEYS}
            if not np.isfinite([total, *parts.values()]).all() or total < 0 or any(v < 0 for v in parts.values()):
                raise ValueError('nonfinite/negative proxy objective')
            if not np.isclose(total, sum(parts.values()), rtol=1e-12, atol=1e-12):
                raise ValueError('proxy total differs from sum of six parts')
            return dict(optimization_loss=total, proxy_parts=parts)

        def save_proxy_series(dest, arrays):
            violation = np.asarray(objective.violations(arrays), float)
            scale_array = np.asarray([scales[k] for k in GATE_KEYS], float)
            if violation.shape != (SAMPLES,6) or not np.isfinite(violation).all() or np.any(violation < 0):
                raise ValueError('violations must be finite nonnegative [4001,6]')
            contributions = (violation/scale_array)**2
            if not np.isfinite(contributions).all():
                raise ValueError('nonfinite proxy contributions')
            path = dest/'PROXY_SERIES.npz'
            atomic_npz(path, time=arrays['time'], phase=arrays['phase'], gate_keys=np.asarray(GATE_KEYS),
                scales=scale_array, violations=violation, contributions=contributions)
            return dict(proxy_series_path=str(path), proxy_series_sha256=sha(path))

        baseline_score = score_arrays(baseline_arrays)
        # s92 RESULT is trajectory provenance, not a score object. Recompute only.
        baseline_record = dict(name='saved_s92_baseline', **baseline_score, **proxy_fields(baseline_arrays),
            trajectory_path=str(BASELINE), trajectory_sha256=identities[str(BASELINE)], simulation_calls=0,
            source_result_path=str(BASE_RUN/'baseline400ms/RESULT.json'),
            source_result_sha256=identities[str(BASE_RUN/'baseline400ms/RESULT.json')],
            weights_sha256=hashlib.sha256(original.tobytes()).hexdigest(), weights_dtype=str(original.dtype),
            weights_shape=list(original.shape))
        atomic_json(out/'BASELINE_RESCORE.json', baseline_record)
        cache = [(original.copy(), baseline_record)]
        for cache_run, cache_label in zip(CACHE_RUNS, ('s93','s94','s96')):
            cache_setup = json.loads((cache_run/'SETUP.json').read_text())
            cache_prereg = json.loads((cache_run/'PREREG.json').read_text())
            cache_provenance = json.loads((cache_run/'MODEL_PROVENANCE.json').read_text())
            cache_ledger = json.loads((cache_run/'LEDGER.json').read_text())
            cache_summary = json.loads((cache_run/'SUMMARY.json').read_text())
            if (cache_prereg['timing'] != timing or cache_setup['timing'] != timing or
                    cache_setup['weight_native'] != weight_native or cache_setup['leg_order'] != list(LEGS) or
                    cache_prereg['baseline_path'] != str(BASELINE) or
                    cache_prereg['cold_state'] != 'fresh network/FIR/body; initial q only; zero v/act/ctrl/pending; no phase reset' or
                    cache_provenance['before'] != before or cache_provenance['after'] != after or
                    cache_provenance['xml_sha256'] != identities[str(MODEL)] or
                    cache_ledger['active'] is not None or cache_ledger['reserved'] != cache_ledger['completed'] or
                    cache_summary['status'] != 'BOUNDED_EXPERIMENT_COMPLETE'):
                raise ValueError(cache_label+' cache same-config/model/completion mismatch')
            cache_evaluations = json.loads((cache_run/'EVALUATIONS.json').read_text())
            if [r['name'] for r in cache_evaluations] != list(CACHE_NAMES):
                raise ValueError(cache_label+' cache must include ALL three records')
            with np.load(cache_run/'DIRECTION.npz', allow_pickle=False) as saved_direction, np.load(cache_run/'UPDATE.npz', allow_pickle=False) as saved_update:
                if (saved_direction['original'].dtype != original.dtype or
                        saved_direction['original'].tobytes() != original.tobytes() or
                        saved_update['original'].dtype != original.dtype or
                        saved_update['original'].tobytes() != original.tobytes()):
                    raise ValueError(cache_label+' cache original weights differ from s92')
                cache_expected_weights = [saved_direction['wp'].copy(), saved_direction['wm'].copy(), saved_update['new'].copy()]
            for name, expected_weights, listed in zip(CACHE_NAMES, cache_expected_weights, cache_evaluations):
                result_path, trajectory = cache_run/name/'RESULT.json', cache_run/name/'TRAJECTORY.npz'
                prior = json.loads(result_path.read_text())
                if (prior != listed or prior['name'] != name or prior['trajectory_path'] != str(trajectory) or
                        prior['trajectory_sha256'] != identities[str(trajectory)]):
                    raise ValueError(cache_label+' result/trajectory identity mismatch: '+name)
                with np.load(trajectory, allow_pickle=False) as saved:
                    arrays = {k:saved[k].copy() for k in saved.files}
                prior_weights = arrays['weights']
                if (set(arrays) != set(baseline_arrays) or
                        any(arrays[k].shape != baseline_arrays[k].shape for k in arrays) or
                        prior_weights.dtype != original.dtype or prior_weights.shape != original.shape or
                        prior_weights.tobytes() != expected_weights.tobytes() or not np.isfinite(prior_weights).all() or
                        np.any(prior_weights[~active] != 0) or np.any(prior_weights[active] < np.finfo(float).eps) or np.any(prior_weights > 4) or
                        not np.array_equal(arrays['qpos'][0], q) or
                        any(np.any(arrays[k][0]) for k in ('qvel','act','ctrl','applied')) or
                        np.any(arrays['warnings']) or any(not np.isfinite(a).all() for a in arrays.values()) or
                        not np.array_equal(arrays['phase'], baseline_arrays['phase']) or
                        not np.allclose(arrays['time'],np.arange(SAMPLES)*BODY_DT,rtol=0,atol=1e-12) or
                        not np.array_equal(arrays['observation_time_s'],np.arange(BLOCKS)*COMMAND_DT) or
                        not np.array_equal(arrays['pending_before'],arrays['applied']) or
                        not np.array_equal(arrays['pending_after'],arrays['decoded']) or
                        not np.array_equal(arrays['applied'][1:],arrays['decoded'][:-1])):
                    raise ValueError(cache_label+' trajectory/weights/cold/clock identity mismatch: '+name)
                rescored = score_arrays(arrays)
                if any(prior[k] != value for k, value in rescored.items()):
                    raise ValueError(cache_label+' saved RESULT differs from old-score rescore: '+name)
                # New fields computed from this cached arm's OWN saved arrays only.
                record = dict(prior, **proxy_fields(arrays), name='saved_'+cache_label+'_'+name,
                    source_result_path=str(result_path), source_result_sha256=identities[str(result_path)],
                    weights_sha256=hashlib.sha256(prior_weights.tobytes()).hexdigest(),
                    weights_dtype=str(prior_weights.dtype), weights_shape=list(prior_weights.shape))
                cache.append((prior_weights.copy(), record))
                del arrays
        atomic_json(out/'DEDUP_CACHE.json', dict(read_only=True, starting_weights='original s92 FROZEN_WEIGHTS only',
            entries=[record for _,record in cache], input_hashes={str(p):identities[str(p)] for p in cache_paths},
            equality='dtype + shape + exact weights bytes; trajectory SHA256 + RESULT SHA256 + weights SHA256',
            all_s93_arms_included=True, all_s94_arms_included=True, all_s96_arms_included=True, no_spare_budget=True))
        built = build_neural_tier()
        a = built['data']
        tier = retain_motor_endpoints(a, downstream_union(a,built['global_index'],
            ['720575941433155799','720575941566493282'],hops=2,max_neurons=20000),
            ['720575941391252886','720575941511883985','720575941497037218','720575941559306355'])
        roots = a['root_ids'][tier['global_index']]
        if len(roots) != 17099:
            raise ValueError('unexpected neural tier')
        edges = build_edges(a,tier,built['cut'])
        groups,_ = channel_index_map(len(roots),roots)
        atomic_npz(out/'TIER_AND_CONNECTIVITY.npz',roots=roots,
            **{prefix+'_'+key:value for prefix,mat in [('We',edges['We']),('Wi',edges['Wi'])]
               for key,value in [('data',mat.tocsr().data),('indices',mat.tocsr().indices),
                                ('indptr',mat.tocsr().indptr),('shape',np.array(mat.shape))]})
        joints, geom_leg = {}, {}
        for leg in LEGS:
            js = [j for j in range(m.njnt) if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j) or '').lower().startswith('joint_'+leg)]
            tib = next(j for j in js if 'tibia' in mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j).lower())
            joints[leg] = dict(q=[int(m.jnt_qposadr[j]) for j in js], tib_q=int(m.jnt_qposadr[tib]), tib_v=int(m.jnt_dofadr[tib]))
        for g in range(m.ngeom):
            nn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) or '').lower()
            bd = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower()
            for leg in LEGS:
                if 'tars' in nn+bd and (nn.startswith(leg) or bd.startswith(leg)):
                    geom_leg[g] = leg
        floors = np.flatnonzero(m.geom_type == mj.mjtGeom.mjGEOM_PLANE)
        if len(floors) != 1:
            raise ValueError('expected one floor')
        floor = int(floors[0])
        limited = np.flatnonzero(m.jnt_limited)
        qa = m.jnt_qposadr[limited]
        spec = mj.mjtState.mjSTATE_INTEGRATION
        atomic_json(out/'SETUP.json', dict(neurons=len(roots), leg_order=list(LEGS), weight_native=weight_native,
            physical_units='NOT_ESTABLISHED', timing=timing, source_hashes=identities,
            mapping_digest=FiniteMotorTranslator(manifest,original).mapping_digest,
            neural_parameters=dataclasses.asdict(ConductanceNetwork(len(roots),dt_ms=.5).p),
            dependencies=dict(python=sys.version,numpy=np.__version__,mujoco=mj.__version__,native=mj.mj_versionString())))

        def check_invariants():
            require_no_callbacks(callback_snapshot(mj))
            if numeric_model_snapshot(m) != after or m.opt.timestep != BODY_DT:
                raise RuntimeError('runtime model changed')

        def sample(d, phase):
            contacts = np.zeros(6,dtype=np.int32)
            force = np.zeros((6,3))
            depth = 0.
            for ci in range(d.ncon):
                c = d.contact[ci]
                g1,g2 = int(c.geom1),int(c.geom2)
                if floor not in (g1,g2):
                    continue
                foot = g2 if g1 == floor else g1
                if foot in geom_leg:
                    leg = LEGS.index(geom_leg[foot])
                    if c.dist < 0:
                        contacts[leg] += 1
                    f = np.zeros(6)
                    mj.mj_contactForce(m,d,ci,f)
                    force[leg] += (1 if g1 == floor else -1)*c.frame.reshape(3,3).T@f[:3]
                else:
                    depth = max(depth,-float(c.dist))
            excess = np.maximum(np.maximum(m.jnt_range[limited,0]-d.qpos[qa],d.qpos[qa]-m.jnt_range[limited,1]),0)
            return dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),
                contacts=contacts,foot_force=force,nonfoot_depth=depth,limit_excess=float(excess.max(initial=0)),
                warnings=np.array([w.number for w in d.warning]),time=float(d.time),phase=phase,
                actuator_force=d.actuator_force.copy(),actuator_length=d.actuator_length.copy(),
                actuator_velocity=d.actuator_velocity.copy())

        finite_keys = ('qpos','qvel','act','ctrl','foot_force','nonfoot_depth','limit_excess','time',
                       'actuator_force','actuator_length','actuator_velocity')
        results = []
        fresh_evaluations = 0

        def evaluate(weights, name):
            nonlocal fresh_evaluations
            if name not in EVAL_NAMES or any(r['name'] == name for r in results):
                raise RuntimeError('only two fixed named evaluations allowed; no spare-budget arms')
            if identities != {str(p):sha(p) for p in paths}:
                raise RuntimeError('source/input changed before evaluation')
            weights = np.asarray(weights, float)
            if (weights.shape != original.shape or not np.isfinite(weights).all() or
                    np.any(weights[~active] != 0) or np.any(weights[active] < np.finfo(float).eps) or np.any(weights > 4)):
                raise ValueError('invalid candidate weights/mask/bounds')
            dest = out/name
            dest.mkdir(exist_ok=False)
            sync_dir(out)
            for prior_weights, prior in cache:
                if weights.dtype == prior_weights.dtype and weights.shape == prior_weights.shape and weights.tobytes() == prior_weights.tobytes():
                    with np.load(prior['trajectory_path'], allow_pickle=False) as saved:
                        reused_arrays = {k:saved[k].copy() for k in saved.files}
                    record = dict(prior, name=name, reused_from=prior['name'], fresh_simulation=False,
                        simulation_calls=0, weights_bitwise_identical=True)
                    record.update(proxy_fields(reused_arrays))
                    record.update(save_proxy_series(dest, reused_arrays))
                    atomic_npz(dest/'WEIGHTS.npz', weights=weights)
                    atomic_json(dest/'REUSE.json', record)
                    atomic_json(dest/'RESULT.json', record)
                    results.append(record)
                    atomic_json(out/'EVALUATIONS.json', results)
                    return record
            if fresh_evaluations >= 2:
                raise RuntimeError('third evaluation forbidden')
            fresh_evaluations += 1
            net = ConductanceNetwork(len(roots),dt_ms=.5)
            net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest,weights)
            if dec.DT != NEURAL_DT or dec.BINS != ((0,5),(5,10),(10,20),(20,40),(40,80)):
                raise RuntimeError('FIR timing changed')
            enc = ChannelSensoryEncoder(groups,len(roots))
            check_invariants()
            d = mj.MjData(m)
            d.qpos[:] = q  # Sole pose write; no act/velocity overwrite, phase reset or auxiliary forces.
            pending = np.zeros(90)
            if d.qvel.any() or d.act.any() or d.ctrl.any() or d.time != 0 or d.xfrc_applied.any() or d.qfrc_applied.any() or dec.history.any():
                raise RuntimeError('noncold initial state')
            reserve(name,-1,dict(initial_forward=1))
            mj.mj_forward(m,d)
            initial_row = sample(d,0)
            atomic_npz(dest/'INITIAL.npz',**initial_row,weights=weights)
            if not all(np.isfinite(initial_row[k]).all() for k in finite_keys) or np.any(initial_row['warnings']):
                atomic_json(dest/'INVALID.json',dict(block=-1,reason='invalid_initial_state'))
                raise RuntimeError('invalid initial state saved; reservation charged')
            check_invariants()
            commit()
            rows, block_data, shard_index = [initial_row], [], []
            raw_schema = None
            for block in range(BLOCKS):
                check_invariants()
                reserve(name,block,dict(neural_ticks=10,physics_steps=50,diagnostic_forward=50))
                observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                rawobs, raw_arrays = raw_observation_record(observed)
                schema = {k:v.shape for k,v in raw_arrays.items()}
                if len(schema) != 30 or (raw_schema is not None and schema != raw_schema):
                    raise RuntimeError('expected stable30-field raw observation schema')
                raw_schema = schema
                atomic_json(dest/f'rawobs_{block:03d}.json',dict(block=block,observation_time_s=float(d.time),
                    quantities=rawobs,source='actual observe_quantities before encode',flattened_names=sorted(raw_arrays)))
                current,_ = enc.encode(observed)
                if current.shape != (17099,) or not np.isfinite(current).all():
                    atomic_npz(dest/'INVALID_CURRENT.npz',current=current,**raw_arrays)
                    raise RuntimeError('invalid current saved; stop all arms')
                spike_rows = []
                for tick in range(10):
                    spike = net.step(current_nA=current).copy()
                    neural_state = {k:v for k,v in vars(net).items() if isinstance(v,np.ndarray)}
                    if (not np.isfinite(spike).all() or any(not np.isfinite(v).all() for v in neural_state.values())):
                        atomic_npz(dest/'INVALID_NEURAL.npz',current=current,spike=spike,**neural_state)
                        atomic_json(dest/'INVALID.json',dict(block=block,tick=tick,reason='nonfinite_neural'))
                        raise RuntimeError('nonfinite neural state saved; stop all arms')
                    spike_rows.append(spike)
                spikes = np.array(spike_rows,dtype=bool)
                applied = pending.copy()
                d.ctrl[:] = applied
                new_rows, invalid = [], None
                for sub in range(PHYSICS_PER_BLOCK):
                    require_no_callbacks(callback_snapshot(mj))
                    if d.xfrc_applied.any() or d.qfrc_applied.any() or not np.array_equal(d.ctrl,applied):
                        raise RuntimeError('unexpected force or hidden command')
                    mj.mj_step(m,d)
                    if np.any([w.number for w in d.warning]) or any(not np.isfinite(x).all() for x in (d.qpos,d.qvel,d.act,d.ctrl)):
                        invalid = dict(block=block,substep=sub,reason='warning_or_nonfinite_after_step')
                        new_rows.append(sample(d,int(block >= 40)))
                        break
                    require_no_callbacks(callback_snapshot(mj))
                    mj.mj_forward(m,d)
                    require_no_callbacks(callback_snapshot(mj))
                    row = sample(d,int(block >= 40))
                    new_rows.append(row)
                    if (not all(np.isfinite(row[k]).all() for k in finite_keys) or np.any(row['warnings']) or
                            not np.array_equal(row['ctrl'],applied) or d.xfrc_applied.any() or d.qfrc_applied.any()):
                        invalid = dict(block=block,substep=sub,reason='nonfinite_warning_hidden_command_or_force')
                        break
                if invalid is not None:
                    data = {key:np.array([r[key] for r in new_rows]) for key in initial_row}
                    data.update(current=current,spikes=spikes,applied=applied,pending_before=applied,
                        pending_unconsumed=pending.copy(),observation_time_s=np.asarray(block*COMMAND_DT),**raw_arrays)
                    atomic_npz(dest/f'block_{block:03d}.npz',**data)
                    atomic_json(dest/'INVALID.json',dict(**invalid,decoded_and_features_valid=False,
                        note='consume not called; decoded/features/pending_after absent, not fabricated'))
                    raise RuntimeError('partial invalid shard saved; reservation charged; stop all arms')
                pending, info = dec.consume(spikes,spike_dt_s=.0005,start_s=block*.005,end_s=(block+1)*.005)
                data = {key:np.array([r[key] for r in new_rows]) for key in initial_row}
                data.update(current=current,spikes=spikes,applied=applied,decoded=pending,
                    pending_before=applied.copy(),pending_after=pending.copy(),
                    observation_time_s=np.asarray(block*COMMAND_DT),features=np.asarray(info['pool_features']),**raw_arrays)
                shard = dest/f'block_{block:03d}.npz'
                atomic_npz(shard,**data)
                shard_index.append(dict(block=block,path=str(shard),sha256=sha(shard),start_s=block*.005,end_s=(block+1)*.005,
                    rawobs_path=str(dest/f'rawobs_{block:03d}.json'),rawobs_sha256=sha(dest/f'rawobs_{block:03d}.json')))
                atomic_json(dest/'SHARDS.json',shard_index)
                if not np.isfinite(pending).all() or not np.isfinite(info['pool_features']).all():
                    atomic_json(dest/'INVALID.json',dict(block=block,reason='nonfinite_decoded_or_features'))
                    raise RuntimeError('invalid decoded shard saved; reservation charged; stop all arms')
                check_invariants()
                body = np.empty(mj.mj_stateSize(m,spec))
                mj.mj_getState(m,d,body,spec)
                snapshot = {key:value for key,value in vars(net).items() if isinstance(value,np.ndarray)}
                atomic_npz(dest/'LATEST_STATE.npz',body_integration=body,FIR_history=dec.history,
                    pending=pending,pending_before=applied,pending_after=pending,weights=weights,**snapshot)
                scalars = {key:value for key,value in vars(net).items() if isinstance(value,(int,float,str))}
                atomic_json(dest/'LATEST_STATE_META.json',dict(block=block,neural_scalars=scalars,start_s=block*.005,end_s=(block+1)*.005,
                    FIR_elapsed_s=dec.elapsed_s,FIR_interval=dec.interval,state_spec='mjSTATE_INTEGRATION',state_size=int(body.size),
                    pending_aliases=dict(applied='pending_before',decoded='pending_after',final_pending_applied=False if block == 79 else None),
                    snapshot_sha256=sha(dest/'LATEST_STATE.npz'),shard=str(shard),shard_sha256=sha(shard),input_hashes=identities,
                    encoder='stateless unchanged config/reference',model_snapshot_equal=True,callbacks_none=True,complete_block=True,automatic_resume=False))
                rows.extend(new_rows)
                block_data.append(data)
                commit()
            arrays = {key:np.array([r[key] for r in rows]) for key in initial_row}
            for key in ('spikes','current','applied','decoded','features','pending_before','pending_after','observation_time_s',*raw_arrays):
                arrays[key] = np.array([b[key] for b in block_data])
            arrays['weights'] = weights
            shapes = dict(qpos=(4001,49),qvel=(4001,48),act=(4001,90),ctrl=(4001,90),contacts=(4001,6),
                foot_force=(4001,6,3),nonfoot_depth=(4001,),limit_excess=(4001,),time=(4001,),phase=(4001,),
                warnings=(4001,len(d.warning)),weights=(66,5),actuator_force=(4001,90),actuator_length=(4001,90),
                actuator_velocity=(4001,90),current=(80,17099),spikes=(80,10,17099),features=(80,66,5),
                applied=(80,90),decoded=(80,90),pending_before=(80,90),pending_after=(80,90),observation_time_s=(80,))
            shapes.update({key:(BLOCKS,)+shape for key,shape in raw_schema.items()})
            if set(arrays) != set(shapes) or len(shard_index) != BLOCKS or any(arrays[k].shape != s for k,s in shapes.items()):
                raise RuntimeError('incomplete trajectory/raw/neural/actual actuator records')
            if (not np.allclose(arrays['time'],np.arange(SAMPLES)*BODY_DT,rtol=0,atol=1e-12) or
                    not np.array_equal(arrays['phase'],np.r_[np.zeros(2001,dtype=int),np.ones(2000,dtype=int)]) or
                    not np.array_equal(arrays['observation_time_s'],np.arange(BLOCKS)*COMMAND_DT) or
                    np.any(arrays['warnings']) or any(not np.isfinite(v).all() for v in arrays.values())):
                raise RuntimeError('continuous clock/phase/finite mismatch')
            if (not np.array_equal(arrays['pending_before'],arrays['applied']) or
                    not np.array_equal(arrays['pending_after'],arrays['decoded']) or np.any(arrays['applied'][0]) or
                    not np.array_equal(arrays['applied'][1:],arrays['decoded'][:-1])):
                raise RuntimeError('old pending ordering mismatch')
            for block in range(BLOCKS):
                if not np.array_equal(arrays['ctrl'][block*50+1:(block+1)*50+1],np.broadcast_to(arrays['applied'][block],(50,90))):
                    raise RuntimeError('physical held command mismatch')
            check_invariants()
            atomic_npz(dest/'TRAJECTORY.npz',**arrays)
            score = score_arrays(arrays)
            record = dict(name=name,**score,**proxy_fields(arrays),**save_proxy_series(dest, arrays),
                trajectory_path=str(dest/'TRAJECTORY.npz'),trajectory_sha256=sha(dest/'TRAJECTORY.npz'),
                fresh_simulation=True,reused_from=None,simulation_calls=dict(neural_ticks=800,physics_steps=4000,explicit_forward=4001))
            atomic_json(dest/'RESULT.json',record)
            results.append(record)
            cache.append((weights.copy(),record))
            atomic_json(out/'EVALUATIONS.json',results)
            return record

        # Evaluate only these two existing parent arrays, in fixed order.
        decisions = {}
        for name in EVAL_NAMES:
            record = evaluate(candidates[name], name)
            decision = dict(name=name, baseline=baseline_record, candidate=record,
                **acceptance(baseline_record, record), behavior_claim='NONE', deployment=False)
            atomic_json(out/name/'ACCEPTANCE.json', decision)
            decisions[name] = decision
        atomic_json(out/'ACCEPTANCE.json', dict(comparisons=decisions,
            baseline=baseline_record, all_arms=results, deployment=False,
            note='Independent per-arm decisions only; no selection or best arm'))
        if [r['name'] for r in results] != list(EVAL_NAMES):
            raise RuntimeError('exact two-candidate sequence required')
        expected_calls = dict(neural_ticks=800*fresh_evaluations,physics_steps=4000*fresh_evaluations,
            initial_forward=fresh_evaluations,diagnostic_forward=4000*fresh_evaluations)
        if reserved != expected_calls or completed != expected_calls or ledger['active'] is not None:
            raise RuntimeError('exact fresh-arm call ledger mismatch')
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input changed during experiment')
        check_invariants()
        callbacks_end = callback_snapshot(mj)
        if callbacks_end != callbacks_before:
            raise RuntimeError('callbacks changed')
        atomic_json(out/'CALLBACKS_END.json',callbacks_end)
        atomic_json(out/'MODEL_END_CHECK.json',dict(passed=True,snapshot=numeric_model_snapshot(m),matches_after=True,s92_after_equal=True,callbacks_none=True))
        atomic_json(out/'SUMMARY.json',dict(status='BOUNDED_EXPERIMENT_COMPLETE',baseline_rescore=baseline_record,
            evaluations=results, names=list(EVAL_NAMES),
            statuses={name:dict(status='COMPLETE', accepted=decisions[name]['accepted'],
                decision=decisions[name]['decision'], acceptance_path=str(out/name/'ACCEPTANCE.json')) for name in EVAL_NAMES},
            fresh_evaluations=fresh_evaluations,ledger=ledger,weight_native=weight_native,
            objective_hashes=objective_hashes,scales=scales,candidate_audit_path=str(out/'CANDIDATE_AUDIT.json'),
            old_loss='unchanged formal_training.score_formal loss',new_loss='optimization_loss: six-gate proxy total',
            behavior_claim='NONE',physical_units='NOT_ESTABLISHED',deployment=False,automatic_resume=False,
            note='Both precomputed arms reported separately; no selection/best/deploy; neither loss used for acceptance; no new direction/proposal/update/run.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
