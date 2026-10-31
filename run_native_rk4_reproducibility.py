"""S2/S4 s98: exactly two explicitly authorized fresh repeatability evaluations.

Derivative of run_native_rk4_ray_check.py; historical weights are copied as bits,
never proposed, updated or deduplicated. Import and --help perform no simulation.
Parent supplies byte-identical s96 objective.py and OBJECTIVE_PREREG.json.
"""
from pathlib import Path
import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import random
import sys
import traceback

# Only offline helper definitions imported; neither historical runner is imported.
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
RAY_RUN = Path('/home/sensen/Desktop/engine修复验收/s97-local-ray/20261023T215729Z/run')
S96_RUN = Path('/home/sensen/Desktop/engine修复验收/s96-six-gate-training/20261023T214545Z/run_validated')
DERIVED_FROM = ROOT/'run_native_rk4_ray_check.py'
BASELINE_FROM = ROOT/'run_native_rk4_full_baseline.py'
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s98-evaluation-repeatability/20261023T220728Z/run')
OBJECTIVE_MODULE = OUTPUT.parent/'objective.py'
OBJECTIVE_PREREG = OUTPUT.parent/'OBJECTIVE_PREREG.json'
EVAL_NAMES = ('repeat_s92', 'repeat_ray_001')
HISTORIES = {'repeat_s92': (BASE_RUN, 'baseline400ms'), 'repeat_ray_001': (RAY_RUN, 'ray_001')}
GATE_KEYS = ('height', 'angle', 'legs', 'load', 'nonfoot', 'limit')
BODY_DT, NEURAL_DT, COMMAND_DT = .0001, .0005, .005
BLOCKS, PHYSICS_PER_BLOCK, SAMPLES = 80, 50, 4001


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT, help='Fixed fresh output: '+str(OUTPUT))
    return p


def json_value(value):
    """Lossless JSON conversion for recorded RNG states/config, not repr guesses."""
    import numpy as np
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), values=value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError('unrecordable configuration/state: '+str(type(value)))


def rng_snapshot(np):
    return dict(python_random=json_value(random.getstate()), numpy_global=json_value(np.random.get_state()),
        neural_rng='NOT_APPLICABLE: ConductanceNetwork owns/uses no RNG in constructor or step; no seed introduced',
        historical_rng='NOT_RECORDED in historical evidence; no historical seed inferred')


def run(args):
    import numpy as np
    import scipy
    from scipy import sparse
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.embodied.formal_training import score_formal
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    from run_neural_muscle_closure import observe_quantities

    out = args.output
    if out != OUTPUT or not out.is_absolute() or out.exists():
        raise ValueError('fixed fresh output required; no automatic resume')
    required_parent = (OUTPUT.parent/'PREREG.json', OBJECTIVE_MODULE, OBJECTIVE_PREREG)
    for p in required_parent:
        if not p.is_file():
            raise ValueError('required parent input missing: '+str(p))
    for current, prior in ((OBJECTIVE_MODULE, S96_RUN.parent/'objective.py'),
                           (OBJECTIVE_PREREG, S96_RUN.parent/'OBJECTIVE_PREREG.json')):
        if current.read_bytes() != prior.read_bytes():
            raise ValueError('objective/scales must be byte-identical to s96: '+str(current))
    objective_hashes = {str(p): sha(p) for p in (OBJECTIVE_MODULE, OBJECTIVE_PREREG)}
    scales = json.loads(OBJECTIVE_PREREG.read_text())['scales']
    if (not isinstance(scales, dict) or set(scales) != set(GATE_KEYS) or
            any(isinstance(scales[k], bool) or not isinstance(scales[k], (int, float)) or
                not np.isfinite(scales[k]) or scales[k] <= 0 for k in GATE_KEYS)):
        raise ValueError('six finite positive preregistered scales required')
    spec_obj = importlib.util.spec_from_file_location('s98_offline_objective', str(OBJECTIVE_MODULE))
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError('cannot load preregistered objective')
    objective = importlib.util.module_from_spec(spec_obj)
    objective_source = OBJECTIVE_MODULE.read_bytes()
    if hashlib.sha256(objective_source).hexdigest() != objective_hashes[str(OBJECTIVE_MODULE)]:
        raise RuntimeError('objective changed before loading')
    exec(compile(objective_source, str(OBJECTIVE_MODULE), 'exec'), objective.__dict__)
    if not all(callable(getattr(objective, k, None)) for k in ('proxy', 'violations')):
        raise TypeError('objective requires proxy and violations')
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
    atomic_json(out/'PREREG.json', dict(evaluations_exact=2, names=list(EVAL_NAMES), timing=timing,
        derived_from=str(DERIVED_FROM), baseline_source=str(BASELINE_FROM),
        weights='dtype + shape + exact bytes copied ONLY from each historical TRAJECTORY.npz weights',
        duplicate_cache='DISABLED: explicit authorization to repeat evaluations, not search; every named arm fresh',
        optimizer='NONE; no arithmetic/proposal/update/direction/selection/cache reuse',
        historical={k:dict(run=str(r), arm=n, trajectory=str(r/n/'TRAJECTORY.npz'), result=str(r/n/'RESULT.json'))
                    for k,(r,n) in HISTORIES.items()},
        objective_hashes=objective_hashes, scales=scales, gate_keys=list(GATE_KEYS),
        old_loss='unchanged formal_training.score_formal; report only',
        comparisons='descriptive historical same-arm and physical old-acceptance comparisons ONLY; no acceptance updates or selection',
        cold_state='fresh network/FIR/body; initial q only; zero v/act/ctrl/pending; no phase reset',
        sole_model_change='same XML compile once; Euler -> RK4 only; dt .0001; exact s92 before/after',
        neural_initialization='existing deterministic ConductanceNetwork default constructor; no new seed',
        initial_evidence='accessible arrays, sparse connectivity, all scalar/config fields, RNG states saved before any forward or neural/body step',
        historical_missing='historical full neural initial arrays and Python/NumPy RNG snapshots NOT_RECORDED; no fabricated historical seed',
        prohibited='no new direction/proposal/update, selection, best arm, deployment, hidden controls, pose or act overwrite, callbacks/delay/history/plugin',
        stopping='warning/nonfinite immediately save partial evidence and stop all arms; no autoresume',
        caps=caps, total_forward_cap=8002, acceptance_updates=False, deployment=False,
        behavior_claim='NONE', physical_units='NOT_ESTABLISHED'))
    shared_paths = [MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
        ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
        ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
        ROOT/'run_neural_muscle_closure.py', ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    paths = list(dict.fromkeys(shared_paths + [Path(__file__).resolve(), DERIVED_FROM, BASELINE_FROM,
        ROOT/'run_native_rk4_gate_training.py', ROOT/'engine/embodied/formal_training.py',
        *required_parent, S96_RUN.parent/'objective.py', S96_RUN.parent/'OBJECTIVE_PREREG.json'] +
        [r/f for r,_ in HISTORIES.values() for f in ('INPUT_HASHES.json','SETUP.json','MODEL_PROVENANCE.json','PREREG.json','LEDGER.json')] +
        [r/n/f for r,n in HISTORIES.values() for f in ('TRAJECTORY.npz','RESULT.json')]))

    def reserve(name, block, delta):
        if ledger['active'] is not None:
            raise RuntimeError('uncommitted reservation; refusing more calls')
        if any(reserved[k]+v > caps[k] for k,v in delta.items()):
            raise RuntimeError('hard call cap exceeded')
        for k,v in delta.items():
            reserved[k] += v
        ledger['active'] = dict(evaluation=name, block=block, delta=delta,
            start_s=max(block,0)*COMMAND_DT, end_s=(block+1)*COMMAND_DT if block >= 0 else 0.)
        atomic_json(out/'LEDGER.json', ledger)

    def commit():
        for k,v in ledger['active']['delta'].items():
            completed[k] += v
        ledger['active'] = None
        atomic_json(out/'LEDGER.json', ledger)

    try:
        callbacks_before = callback_snapshot(mj)
        require_no_callbacks(callbacks_before)
        atomic_json(out/'CALLBACKS_BEFORE.json', callbacks_before)
        if mj.__version__ != '3.9.0' or mj.mj_versionString() != '3.9.0':
            raise RuntimeError('requires installed/native MuJoCo3.9.0 as s92')
        identities = {str(p): sha(p) for p in paths}
        if any(identities[p] != h for p,h in objective_hashes.items()):
            raise RuntimeError('objective/scales changed before binding')
        actual = {str(p): identities[str(p)] for p in shared_paths}
        for prior_run, source in ((BASE_RUN, BASELINE_FROM), (RAY_RUN, DERIVED_FROM)):
            prior_hashes = json.loads((prior_run/'INPUT_HASHES.json').read_text())
            if len(actual) != 14 or any(prior_hashes[p] != h for p,h in actual.items()):
                raise ValueError('shared14 differs from historical run: '+str(prior_run))
            if prior_hashes[str(source)] != identities[str(source)]:
                raise ValueError('historical source identity differs: '+str(source))
            old_ledger = json.loads((prior_run/'LEDGER.json').read_text())
            if old_ledger['active'] is not None or old_ledger['reserved'] != old_ledger['completed']:
                raise ValueError('historical run not fully committed')
        atomic_json(out/'INPUT_HASHES.json', identities)
        atomic_json(out/'SHARED_SOURCE_CHECK.json', dict(passed=True, shared=actual,
            historical_runs=[str(BASE_RUN),str(RAY_RUN)], baseline_source_sha256=identities[str(BASELINE_FROM)],
            derivative_sha256=identities[str(DERIVED_FROM)], objective_hashes=objective_hashes))
        config_stream = io.StringIO()
        with contextlib.redirect_stdout(config_stream):
            np.show_config()
        try:
            from threadpoolctl import threadpool_info
            threadpools = threadpool_info()
        except ImportError:
            threadpools = 'threadpoolctl NOT_INSTALLED'
        atomic_json(out/'ENVIRONMENT.json', dict(python=sys.version, executable=sys.executable,
            platform=platform.platform(), uname=list(platform.uname()), machine=platform.machine(),
            processor=platform.processor(), byteorder=sys.byteorder, cpu_count=os.cpu_count(),
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            versions=dict(numpy=np.__version__,scipy=scipy.__version__,mujoco=mj.__version__,native=mj.mj_versionString()),
            libraries=sorted([dict(name=d.metadata.get('Name'), version=d.version) for d in importlib.metadata.distributions()], key=lambda x: str(x['name'])),
            environment=dict(os.environ), numpy_configuration=config_stream.getvalue(), threadpools=threadpools,
            initial_rng=rng_snapshot(np), historical_environment='Full environment/thread/RNG snapshots NOT_RECORDED; only historical SETUP dependencies available'))
        manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
        with np.load(INITIAL, allow_pickle=False) as z:
            q, sensory_ref = z['new_qpos'].copy(), z['sensory_reference_qpos'].copy()
        if q.shape != (49,) or sensory_ref.shape != (49,) or not np.isfinite([q,sensory_ref]).all():
            raise ValueError('invalid fixed initial/reference')
        historical_arrays, historical_results, repeat_weights = {}, {}, {}
        for name,(r,n) in HISTORIES.items():
            trajectory, result_path = r/n/'TRAJECTORY.npz', r/n/'RESULT.json'
            result = json.loads(result_path.read_text())
            if result['trajectory_sha256'] != identities[str(trajectory)]:
                raise ValueError('historical RESULT trajectory hash mismatch: '+name)
            with np.load(trajectory, allow_pickle=False) as z:
                arrays = {k:z[k].copy() for k in z.files}
            w = arrays['weights'].copy()
            if (w.dtype != np.dtype(float) or w.shape != (66,5) or not np.isfinite(w).all() or
                    np.count_nonzero(w) != 330 or np.any(w <= 0) or np.any(w > 4) or
                    arrays['qpos'].shape != (4001,49) or not np.array_equal(arrays['qpos'][0],q) or
                    any(np.any(arrays[k][0]) for k in ('qvel','act','ctrl','applied')) or
                    np.any(arrays['warnings']) or any(not np.isfinite(a).all() for a in arrays.values()) or
                    not np.allclose(arrays['time'],np.arange(SAMPLES)*BODY_DT,rtol=0,atol=1e-12)):
                raise ValueError('historical weights/cold state/clock/finite mismatch: '+name)
            historical_arrays[name], historical_results[name], repeat_weights[name] = arrays, result, w
        original = repeat_weights['repeat_s92']
        frozen = np.array(json.loads((ENDPOINTS/'FROZEN_WEIGHTS.json').read_text())['weights'],float)
        if original.dtype != frozen.dtype or original.shape != frozen.shape or original.tobytes() != frozen.tobytes():
            raise ValueError('historical s92 weights differ from original frozen weights')
        active = original > 0
        atomic_json(out/'WEIGHT_SOURCE_AUDIT.json', dict(duplicate_cache='DISABLED', explicit_repeat_authorized=True,
            arms={name:dict(trajectory=str(HISTORIES[name][0]/HISTORIES[name][1]/'TRAJECTORY.npz'),
                trajectory_sha256=identities[str(HISTORIES[name][0]/HISTORIES[name][1]/'TRAJECTORY.npz')],
                weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(), dtype=str(w.dtype), shape=list(w.shape),
                exact_copy=True, arithmetic=False) for name,w in repeat_weights.items()},
            new_direction=False, proposal=False, update=False, selection=False, deployment=False))
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
            raise RuntimeError('sole integrator change / exact s92 after failed')
        for r,_ in HISTORIES.values():
            prior = json.loads((r/'MODEL_PROVENANCE.json').read_text())
            setup = json.loads((r/'SETUP.json').read_text())
            old_timing = setup['timing']
            numeric_timing = ('blocks','duration_s','body_dt_s','neural_dt_s','command_dt_s',
                              'samples','neural_ticks_per_block','physics_steps_per_block')
            if (prior['before'] != before or prior['after'] != after or
                    any(old_timing[k] != timing[k] for k in numeric_timing) or
                    (r == RAY_RUN and old_timing != timing)):
                raise ValueError('historical model/runtime timing differs: '+str(r))
        atomic_json(out/'MODEL_PROVENANCE.json', dict(xml_path=str(MODEL),xml_sha256=identities[str(MODEL)],
            before=before,after=after,changed_numeric_fields=changed,s92_before_equal=True,s92_after_equal=True,
            original_timestep_s=BODY_DT,runtime_timestep_s=float(m.opt.timestep),
            original_integrator=int(before['opt.integrator']),runtime_integrator=int(m.opt.integrator),
            no_other_model_field_changes=True,baseline_provenance_sha256=identities[str(BASE_RUN/'MODEL_PROVENANCE.json')]))
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        for r,_ in HISTORIES.values():
            setup = json.loads((r/'SETUP.json').read_text())
            if setup['weight_native'] != weight_native or setup['leg_order'] != list(LEGS):
                raise ValueError('historical native weight/leg order differs')

        def score_arrays(arrays):
            score = score_formal(arrays, weight_native)
            for phase,span in support_spans(arrays,weight_native).items():
                score[phase]['longest_support_ms'] = span
            if not np.isfinite(float(score['loss'])):
                raise ValueError('nonfinite formal score')
            return score

        def proxy_fields(arrays):
            result = objective.proxy(arrays,scales)
            if not isinstance(result,dict) or set(result) != {'total','parts'} or set(result['parts']) != set(GATE_KEYS):
                raise ValueError('invalid proxy structure')
            total = float(result['total'])
            parts = {k:float(result['parts'][k]) for k in GATE_KEYS}
            if not np.isfinite([total,*parts.values()]).all() or min([total,*parts.values()]) < 0 or not np.isclose(total,sum(parts.values()),rtol=1e-12,atol=1e-12):
                raise ValueError('invalid proxy values')
            return dict(optimization_loss=total,proxy_parts=parts)

        def save_proxy_series(dest,arrays):
            violation = np.asarray(objective.violations(arrays),float)
            scale_array = np.asarray([scales[k] for k in GATE_KEYS],float)
            if violation.shape != (SAMPLES,6) or not np.isfinite(violation).all() or np.any(violation < 0):
                raise ValueError('invalid violations')
            contributions = (violation/scale_array)**2
            if not np.isfinite(contributions).all():
                raise ValueError('invalid proxy contributions')
            p = dest/'PROXY_SERIES.npz'
            atomic_npz(p,time=arrays['time'],phase=arrays['phase'],gate_keys=np.asarray(GATE_KEYS),
                scales=scale_array,violations=violation,contributions=contributions)
            return dict(proxy_series_path=str(p),proxy_series_sha256=sha(p))

        historical_records = {}
        for name,arrays in historical_arrays.items():
            r,n = HISTORIES[name]
            oldscore = score_arrays(arrays)
            # s92 RESULT only binds provenance; s97 contains score fields as well.
            if name == 'repeat_ray_001' and any(historical_results[name][k] != v for k,v in oldscore.items()):
                raise ValueError('historical s97 old-loss report differs from rescore')
            historical_records[name] = dict(name='historical_'+name,**oldscore,**proxy_fields(arrays),
                trajectory_path=str(r/n/'TRAJECTORY.npz'),trajectory_sha256=identities[str(r/n/'TRAJECTORY.npz')],
                source_result_path=str(r/n/'RESULT.json'),source_result_sha256=identities[str(r/n/'RESULT.json')],simulation_calls=0)
        atomic_json(out/'HISTORICAL_RESCORES.json',historical_records)
        built = build_neural_tier()
        a = built['data']
        tier = retain_motor_endpoints(a,downstream_union(a,built['global_index'],
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
        joints,geom_leg = {},{}
        for leg in LEGS:
            js = [j for j in range(m.njnt) if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j) or '').lower().startswith('joint_'+leg)]
            tib = next(j for j in js if 'tibia' in mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j).lower())
            joints[leg] = dict(q=[int(m.jnt_qposadr[j]) for j in js],tib_q=int(m.jnt_qposadr[tib]),tib_v=int(m.jnt_dofadr[tib]))
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
        state_spec = mj.mjtState.mjSTATE_INTEGRATION
        atomic_json(out/'SETUP.json',dict(neurons=len(roots),leg_order=list(LEGS),weight_native=weight_native,
            physical_units='NOT_ESTABLISHED',timing=timing,source_hashes=identities,
            mapping_digest=FiniteMotorTranslator(manifest,original).mapping_digest,
            neural_parameters=dataclasses.asdict(ConductanceParams()),
            dependencies=dict(python=sys.version,numpy=np.__version__,mujoco=mj.__version__,native=mj.mj_versionString())))

        def check_invariants():
            require_no_callbacks(callback_snapshot(mj))
            if numeric_model_snapshot(m) != after or m.opt.timestep != BODY_DT:
                raise RuntimeError('runtime model changed')

        def sample(d,phase):
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
                actuator_force=d.actuator_force.copy(),actuator_length=d.actuator_length.copy(),actuator_velocity=d.actuator_velocity.copy())

        finite_keys = ('qpos','qvel','act','ctrl','foot_force','nonfoot_depth','limit_excess','time',
                       'actuator_force','actuator_length','actuator_velocity')
        results = []
        fresh_evaluations = 0
        initial_identities = {}

        def save_initial(dest,net,dec,enc,d,pending,name,rng_before_constructor):
            arrays,scalars,configs,rngs = {},{},{},{}
            for key,value in vars(net).items():
                if isinstance(value,np.ndarray):
                    arrays[key] = value.copy()
                elif sparse.issparse(value):
                    mat = value.tocsr()
                    arrays.update({key+'_data':mat.data.copy(),key+'_indices':mat.indices.copy(),
                        key+'_indptr':mat.indptr.copy(),key+'_shape':np.asarray(mat.shape)})
                elif dataclasses.is_dataclass(value):
                    configs[key] = dataclasses.asdict(value)
                elif isinstance(value,(bool,int,float,str,np.generic)) or value is None:
                    scalars[key] = json_value(value)
                elif isinstance(value,np.random.Generator):
                    rngs[key] = json_value(value.bit_generator.state)
                elif isinstance(value,np.random.RandomState):
                    rngs[key] = json_value(value.get_state())
                elif isinstance(value,random.Random):
                    rngs[key] = json_value(value.getstate())
                else:
                    raise TypeError('unrecorded neural initial field: '+key)
            if (net.steps != 0 or net.t_ms != 0 or net._ring_i != 0 or
                    not np.all(net.v == net.p.rest_mV) or
                    any(np.any(getattr(net,k)) for k in ('_ring','ge','gi','adaptation','refractory_until_ms','spike_count'))):
                raise RuntimeError('original deterministic neural initialization changed')
            atomic_npz(dest/'NEURAL_INITIAL.npz',**arrays)
            body = np.empty(mj.mj_stateSize(m,state_spec))
            mj.mj_getState(m,d,body,state_spec)
            atomic_npz(dest/'COLD_INITIAL.npz',body_integration=body,qpos=d.qpos.copy(),qvel=d.qvel.copy(),
                act=d.act.copy(),ctrl=d.ctrl.copy(),pending=pending.copy(),FIR_history=dec.history.copy(),sensory_reference=sensory_ref)
            initial_meta = dict(name=name,neural_scalars=scalars,neural_configs=configs,neural_rng_states=rngs,
                rng_before_constructor=rng_before_constructor,rng_before_calls=rng_snapshot(np),
                neural_rng_policy='No RNG in existing ConductanceNetwork initialization/step; no new seed',
                network_all_fields=sorted(vars(net)),array_fields=sorted(arrays),
                array_identity={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()},
                neural_initial_sha256=sha(dest/'NEURAL_INITIAL.npz'),cold_initial_sha256=sha(dest/'COLD_INITIAL.npz'),
                model_snapshot=after,source_hashes=identities,FIR_elapsed_s=dec.elapsed_s,FIR_interval=dec.interval,
                encoder_config={k:json_value(v) for k,v in vars(enc).items()},
                historical_initial_arrays='NOT_RECORDED: no historical full initial network snapshot',
                historical_rng='NOT_RECORDED: no fabricated historical seed; deterministic initialization inferred only from bound source',
                before_initial_forward=True,before_any_neural_or_body_step=True,automatic_resume=False)
            atomic_json(dest/'INITIAL_STATE_META.json',initial_meta)
            initial_identities[name] = {str(dest/f):sha(dest/f) for f in ('NEURAL_INITIAL.npz','COLD_INITIAL.npz','INITIAL_STATE_META.json')}
            atomic_json(out/'INITIAL_STATE_HASHES.json',initial_identities)

        def evaluate(weights,name):
            nonlocal fresh_evaluations
            if name not in EVAL_NAMES or any(r['name'] == name for r in results) or fresh_evaluations >= 2:
                raise RuntimeError('only exactly two fixed named fresh evaluations authorized')
            if identities != {str(p):sha(p) for p in paths}:
                raise RuntimeError('source/input changed before evaluation')
            historical_w = historical_arrays[name]['weights']
            if (weights.dtype != historical_w.dtype or weights.shape != historical_w.shape or
                    weights.tobytes() != historical_w.tobytes()):
                raise ValueError('repeat weights differ bitwise from historical trajectory')
            dest = out/name
            dest.mkdir(exist_ok=False)
            sync_dir(out)
            atomic_npz(dest/'WEIGHTS.npz',weights=weights)
            fresh_evaluations += 1
            rng_before_constructor = rng_snapshot(np)
            net = ConductanceNetwork(len(roots),dt_ms=.5)
            net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest,weights)
            if dec.DT != NEURAL_DT or dec.BINS != ((0,5),(5,10),(10,20),(20,40),(40,80)):
                raise RuntimeError('FIR timing changed')
            enc = ChannelSensoryEncoder(groups,len(roots))
            check_invariants()
            d = mj.MjData(m)
            d.qpos[:] = q  # Sole pose write; no act/velocity overwrite or auxiliary force.
            pending = np.zeros(90)
            if d.qvel.any() or d.act.any() or d.ctrl.any() or d.time != 0 or d.xfrc_applied.any() or d.qfrc_applied.any() or dec.history.any():
                raise RuntimeError('noncold initial state')
            save_initial(dest,net,dec,enc,d,pending,name,rng_before_constructor)
            reserve(name,-1,dict(initial_forward=1))
            mj.mj_forward(m,d)
            initial_row = sample(d,0)
            atomic_npz(dest/'INITIAL.npz',**initial_row,weights=weights)
            if not all(np.isfinite(initial_row[k]).all() for k in finite_keys) or np.any(initial_row['warnings']):
                atomic_json(dest/'INVALID.json',dict(block=-1,reason='invalid_initial_state'))
                raise RuntimeError('invalid initial state saved; reservation charged')
            check_invariants()
            commit()
            rows,block_data,shard_index = [initial_row],[],[]
            raw_schema = None
            for block in range(BLOCKS):
                check_invariants()
                reserve(name,block,dict(neural_ticks=10,physics_steps=50,diagnostic_forward=50))
                observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                rawobs,raw_arrays = raw_observation_record(observed)
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
                    if not np.isfinite(spike).all() or any(not np.isfinite(v).all() for v in neural_state.values()):
                        atomic_npz(dest/'INVALID_NEURAL.npz',current=current,spike=spike,**neural_state)
                        atomic_json(dest/'INVALID.json',dict(block=block,tick=tick,reason='nonfinite_neural'))
                        raise RuntimeError('nonfinite neural state saved; stop all arms')
                    spike_rows.append(spike)
                spikes = np.array(spike_rows,dtype=bool)
                applied = pending.copy()
                d.ctrl[:] = applied
                new_rows,invalid = [],None
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
                pending,info = dec.consume(spikes,spike_dt_s=.0005,start_s=block*.005,end_s=(block+1)*.005)
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
                body = np.empty(mj.mj_stateSize(m,state_spec))
                mj.mj_getState(m,d,body,state_spec)
                snapshot = {key:value for key,value in vars(net).items() if isinstance(value,np.ndarray)}
                atomic_npz(dest/'LATEST_STATE.npz',body_integration=body,FIR_history=dec.history,
                    pending=pending,pending_before=applied,pending_after=pending,weights=weights,**snapshot)
                scalars = {key:value for key,value in vars(net).items() if isinstance(value,(int,float,str))}
                atomic_json(dest/'LATEST_STATE_META.json',dict(block=block,neural_scalars=scalars,start_s=block*.005,end_s=(block+1)*.005,
                    FIR_elapsed_s=dec.elapsed_s,FIR_interval=dec.interval,state_spec='mjSTATE_INTEGRATION',state_size=int(body.size),
                    pending_aliases=dict(applied='pending_before',decoded='pending_after',final_pending_applied=False if block == 79 else None),
                    snapshot_sha256=sha(dest/'LATEST_STATE.npz'),shard=str(shard),shard_sha256=sha(shard),input_hashes=identities,
                    initial_state_hashes=initial_identities[name],initial_rng_path=str(dest/'INITIAL_STATE_META.json'),
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
            record = dict(name=name,**score,**proxy_fields(arrays),**save_proxy_series(dest,arrays),
                trajectory_path=str(dest/'TRAJECTORY.npz'),trajectory_sha256=sha(dest/'TRAJECTORY.npz'),
                fresh_simulation=True,reused_from=None,duplicate_cache='DISABLED',
                weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),initial_state_hashes=initial_identities[name],
                simulation_calls=dict(neural_ticks=800,physics_steps=4000,explicit_forward=4001),
                acceptance_updates=False,deployment=False)
            atomic_json(dest/'RESULT.json',record)
            old = historical_arrays[name]
            comparisons = {}
            if set(arrays) != set(old):
                raise ValueError('historical/current trajectory schemas differ')
            for key,x in arrays.items():
                y = old[key]
                same_layout = x.shape == y.shape and x.dtype == y.dtype
                comparisons[key] = dict(shape=list(x.shape),historical_shape=list(y.shape),dtype=str(x.dtype),historical_dtype=str(y.dtype),
                    bitwise_equal=bool(same_layout and x.tobytes() == y.tobytes()),
                    max_abs_difference=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.shape == y.shape and x.size else None)
            atomic_json(dest/'REPEATABILITY.json',dict(name=name,historical=historical_records[name],current=record,
                arrays=comparisons,all_arrays_bitwise_equal=all(c['bitwise_equal'] for c in comparisons.values()),
                descriptive_only=True,selection=False,acceptance_updates=False,deployment=False))
            results.append(record)
            atomic_json(out/'EVALUATIONS.json',results)
            return record

        descriptive = {}
        for name in EVAL_NAMES:
            record = evaluate(repeat_weights[name],name)
            comparison = dict(name=name,historical_same_arm=historical_records[name],current=record,
                old_physical_comparison=acceptance(historical_records[name],record),
                versus_historical_s92=acceptance(historical_records['repeat_s92'],record),
                descriptive_only=True,acceptance_updates=False,selection=False,deployment=False)
            atomic_json(out/name/'PHYSICAL_DESCRIPTIVE_COMPARISON.json',comparison)
            descriptive[name] = comparison
        if fresh_evaluations != 2 or [r['name'] for r in results] != list(EVAL_NAMES):
            raise RuntimeError('exact two fresh repeat sequence required')
        if reserved != caps or completed != caps or ledger['active'] is not None:
            raise RuntimeError('exact two fresh-arm call ledger mismatch')
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input changed during experiment')
        for arm_hashes in initial_identities.values():
            if any(sha(p) != h for p,h in arm_hashes.items()):
                raise RuntimeError('initial-state evidence changed')
        check_invariants()
        callbacks_end = callback_snapshot(mj)
        if callbacks_end != callbacks_before:
            raise RuntimeError('callbacks changed')
        atomic_json(out/'CALLBACKS_END.json',callbacks_end)
        atomic_json(out/'MODEL_END_CHECK.json',dict(passed=True,snapshot=numeric_model_snapshot(m),matches_after=True,
            s92_after_equal=True,callbacks_none=True,source_hashes_unchanged=True,initial_state_hashes_unchanged=True))
        atomic_json(out/'SUMMARY.json',dict(status='BOUNDED_REPEATABILITY_COMPLETE',historical_rescores=historical_records,
            evaluations=results,names=list(EVAL_NAMES),fresh_evaluations=fresh_evaluations,ledger=ledger,
            weight_native=weight_native,objective_hashes=objective_hashes,scales=scales,
            descriptive_physical_comparisons=descriptive,duplicate_cache='DISABLED: authorized repeat, not search',
            initial_state_hashes=initial_identities,old_loss='unchanged formal_training.score_formal loss',
            new_loss='optimization_loss: identical s96 six-gate proxy total, report only',
            acceptance_updates=False,selection=False,proposal=False,update=False,deployment=False,
            behavior_claim='NONE',physical_units='NOT_ESTABLISHED',automatic_resume=False,
            note='Exactly two fresh historical-weight repeats, no historical runner imported/rerun and no cache reuse.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
