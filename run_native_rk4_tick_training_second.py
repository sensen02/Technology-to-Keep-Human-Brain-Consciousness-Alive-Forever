"""S2/S4 s102: one bounded second tick-consumption FIR training round.

Direct baseline is the EXACT saved S101 updated trajectory/weights, not a
regenerated update. S100 is the fixed physical reference. Help/import performs
no simulation. No deployment, next round, backtrack, resume or retry.
"""
from pathlib import Path
import argparse
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import sys
import traceback

from run_native_rk4_gate_training import (
    sha, sync_dir, atomic_json, atomic_npz, atomic_csv, numeric_model_snapshot,
    callback_snapshot, require_no_callbacks, raw_observation_record,
    support_spans, acceptance,
)
from run_native_rk4_reproducibility import json_value, rng_snapshot

ROOT = Path(__file__).resolve().parent
MODEL = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
INITIAL = Path('/run/media/sensen/Data2/engine修复验收/s7-posture-ik/20261023T021141Z/DERIVED_INITIAL.npz')
ENDPOINTS = Path('/run/media/sensen/Data2/engine修复验收/s4-endpoints/20261023T011559Z')
S92_RUN = Path('/home/sensen/Desktop/engine修复验收/s92-rk4-full-baseline/20261023T210730Z/run')
BASE_RUN = Path('/home/sensen/Desktop/engine修复验收/s100-tick-consumption/20261023T222456Z/run')
S100_REFERENCE = BASE_RUN/'tick400ms/TRAJECTORY.npz'
DIRECT_RUN = Path('/home/sensen/Desktop/engine修复验收/s101-tick-training/20261023T223403Z/run')
BASELINE = DIRECT_RUN/'round00_updated/TRAJECTORY.npz'
DIRECT_RESULT = DIRECT_RUN/'round00_updated/RESULT.json'
S100_SOURCE = ROOT/'run_native_rk4_tick_consumption.py'
S101_SOURCE = ROOT/'run_native_rk4_tick_training.py'
S96_PARENT = Path('/home/sensen/Desktop/engine修复验收/s96-six-gate-training/20261023T214545Z')
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s102-tick-training-second/20261023T224501Z/run')
GATE_KEYS = ('height','angle','legs','load','nonfoot','limit')
BODY_DT, NEURAL_DT, COMMAND_DT = .0001, .0005, .005
BLOCKS, TICKS, SAMPLES = 80, 800, 4001
ARMS = ('round00_plus','round00_minus','round00_updated')
SEED, DELTA, MAX_STEP = 10201, 1e-6, 1e-7


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    return p


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
    from engine.embodied.grounded_training import propose_weights
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    from run_neural_muscle_closure import observe_quantities

    out = args.output
    if out != OUTPUT or not out.is_absolute() or out.exists():
        raise ValueError('fixed fresh output only; no automatic resume')
    required = [out.parent/f for f in ('PREREG.json','objective.py','OBJECTIVE_PREREG.json','STATIC_CHECK.json','OFFLINE_FIR_CHECK.json','OFFLINE_FIR_CHECK.npz','DIRECTION.json','DIRECTION.npz')]
    if any(not p.is_file() for p in required):
        raise ValueError('parent preregistration/direction/static/offline checks required')
    for name in ('objective.py','OBJECTIVE_PREREG.json'):
        if (out.parent/name).read_bytes() != (S96_PARENT/name).read_bytes() or (out.parent/name).read_bytes() != (BASE_RUN.parent/name).read_bytes():
            raise ValueError('objective/scales must be original s96 and s100 bytecopy')
    static = json.loads((out.parent/'STATIC_CHECK.json').read_text())
    offline = json.loads((out.parent/'OFFLINE_FIR_CHECK.json').read_text())
    s100_hash = sha(S100_SOURCE)
    if not static.get('passed') or not offline.get('passed'):
        raise ValueError('failed pre-run static/offline checks')
    # S100 offline evidence used S92 spikes ONLY for FIR batching equivalence.
    # Old-pending S92 is NEVER an evaluation cache or a training baseline here.
    if (out.parent/'OFFLINE_FIR_CHECK.json').read_bytes() != (BASE_RUN.parent/'OFFLINE_FIR_CHECK.json').read_bytes() or (out.parent/'OFFLINE_FIR_CHECK.npz').read_bytes() != (BASE_RUN.parent/'OFFLINE_FIR_CHECK.npz').read_bytes():
        raise ValueError('require immutable s100 offline evidence bytecopy')
    if offline['runner_sha256'] != s100_hash or offline['arrays_sha256'] != sha(out.parent/'OFFLINE_FIR_CHECK.npz') or offline['translator_sha256'] != sha(ROOT/'engine/embodied/motor_translation.py'):
        raise ValueError('s100 offline evidence/source changed')
    if sha(Path(offline['source'])) != offline['source_sha256']:
        raise ValueError('immutable offline evidence source changed')
    if 'runner_sha256' in static and static['runner_sha256'] not in (sha(Path(__file__)),s100_hash):
        raise ValueError('stale static runner binding')
    import ast
    def tick_loop(path):
        tree = ast.parse(path.read_text())
        loops = [n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='tick']
        if len(loops)!=1: raise ValueError('unique s100 tick loop required')
        return ast.dump(loops[0],include_attributes=False)
    if tick_loop(Path(__file__)) != tick_loop(S100_SOURCE) or tick_loop(S101_SOURCE) != tick_loop(S100_SOURCE):
        raise ValueError('s100 inner tick-loop AST mismatch')
    objective_path = out.parent/'objective.py'
    spec = importlib.util.spec_from_file_location('s102_objective',str(objective_path))
    objective = importlib.util.module_from_spec(spec)
    exec(compile(objective_path.read_bytes(),str(objective_path),'exec'),objective.__dict__)
    scales = json.loads((out.parent/'OBJECTIVE_PREREG.json').read_text())['scales']
    if set(scales)!=set(GATE_KEYS) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in scales.values()):
        raise ValueError('invalid six scales')
    objective_hashes = {str(out.parent/f):sha(out.parent/f) for f in ('objective.py','OBJECTIVE_PREREG.json')}
    shared = [MODEL,INITIAL,ENDPOINTS/'MANIFEST.json',ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py',ROOT/'engine/embodied/motor_translation.py',ROOT/'engine/embodied/sensory_channels.py',
        ROOT/'engine/neural_cond.py',ROOT/'engine/embodied/adapters.py',ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py',ROOT/'engine/neck_cut_data.py',ROOT/'run_neural_muscle_closure.py',ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    archive_names = ('INPUT_HASHES.json','MODEL_PROVENANCE.json','LEDGER.json','SETUP.json','TIMING.json','PREREG.json','SUMMARY.json','MODEL_END_CHECK.json')
    archive_paths = [run/name for run in (BASE_RUN,DIRECT_RUN) for name in archive_names]
    cache_sources = [S100_REFERENCE]+[DIRECT_RUN/arm/'TRAJECTORY.npz' for arm in ARMS]
    cache_results = [p.parent/'RESULT.json' for p in cache_sources]
    config_files = ('TICK_SCHEDULE.json','NEURAL_INITIAL.npz','COLD_INITIAL.npz','INITIAL_STATE_META.json')
    archive_paths += [p.parent/name for p in cache_sources for name in config_files]
    archive_paths += [DIRECT_RUN/name for name in ('UPDATE.npz','UPDATE.json','EVALUATIONS.json','INITIAL_STATE_HASHES.json')]
    archive_paths += [DIRECT_RUN.parent/name for name in ('DIRECTION.npz','DIRECTION.json','objective.py','OBJECTIVE_PREREG.json')]
    paths = list(dict.fromkeys(shared+required+archive_paths+cache_sources+cache_results+[
        BASE_RUN.parent/'COMPARISON.json',BASE_RUN.parent/'OFFLINE_FIR_CHECK.json',BASE_RUN.parent/'OFFLINE_FIR_CHECK.npz',BASE_RUN.parent/'STATIC_CHECK.json',S100_SOURCE,S101_SOURCE,
        S92_RUN/'INPUT_HASHES.json',S92_RUN/'MODEL_PROVENANCE.json',ROOT/'run_native_rk4_full_baseline.py',ROOT/'run_native_rk4_gate_training.py',ROOT/'run_native_rk4_reproducibility.py',ROOT/'engine/embodied/formal_training.py',Path(__file__).resolve(),S96_PARENT/'objective.py',S96_PARENT/'OBJECTIVE_PREREG.json']))
    identities = {str(p):sha(p) for p in paths}
    prior_hashes = json.loads((BASE_RUN/'INPUT_HASHES.json').read_text())
    direct_hashes = json.loads((DIRECT_RUN/'INPUT_HASHES.json').read_text())
    s92_hashes = json.loads((S92_RUN/'INPUT_HASHES.json').read_text())
    if len(shared)!=14 or any(prior_hashes[str(p)]!=identities[str(p)] or direct_hashes[str(p)]!=identities[str(p)] or s92_hashes[str(p)]!=identities[str(p)] for p in shared+[ROOT/'run_native_rk4_full_baseline.py']):
        raise ValueError('s100/s101/s92 shared14/source binding failed')
    if prior_hashes[str(S100_SOURCE)] != s100_hash or direct_hashes[str(S100_SOURCE)] != s100_hash or direct_hashes[str(S101_SOURCE)] != sha(S101_SOURCE):
        raise ValueError('archived tick runner source differs')
    # Authenticate every recorded S101 input, including objective, direction,
    # static/offline evidence and S100 records, without modifying any source.
    if any(sha(Path(p))!=h for p,h in direct_hashes.items()):
        raise ValueError('s101 archived input authentication failed')
    comparison = json.loads((BASE_RUN.parent/'COMPARISON.json').read_text())
    if not comparison.get('verified') or not comparison.get('weights_original_bitwise') or not comparison.get('shared14_unchanged') or not comparison.get('model_same_s92'):
        raise ValueError('independent s100 comparison not verified')
    for source_run in (BASE_RUN,DIRECT_RUN):
        old_ledger = json.loads((source_run/'LEDGER.json').read_text())
        if old_ledger['active'] is not None or old_ledger['reserved'] != old_ledger['completed'] or old_ledger['completed'] != old_ledger['caps']:
            raise ValueError('uncommitted archived tick run')
        if not json.loads((source_run/'MODEL_END_CHECK.json').read_text()).get('passed'):
            raise ValueError('archived model end check failed')
    timing = json.loads((BASE_RUN/'TIMING.json').read_text())
    if timing != json.loads((DIRECT_RUN/'TIMING.json').read_text()):
        raise ValueError('s101 timing differs from fixed s100')
    for source_run in (BASE_RUN,DIRECT_RUN):
        setup = json.loads((source_run/'SETUP.json').read_text())
        if setup['timing']!=timing or setup['neurons']!=17099:
            raise ValueError('archived setup mismatch')
    if json.loads((BASE_RUN/'MODEL_PROVENANCE.json').read_text())['after'] != json.loads((DIRECT_RUN/'MODEL_PROVENANCE.json').read_text())['after']:
        raise ValueError('s101 model differs from s100')
    with np.load(BASELINE,allow_pickle=False) as z:
        baseline = {k:z[k].copy() for k in z.files}
    baseline_saved = json.loads(DIRECT_RESULT.read_text())
    if baseline_saved['trajectory_sha256'] != sha(BASELINE) or baseline_saved['trajectory_path']!=str(BASELINE):
        raise ValueError('direct s101 baseline result binding failed')
    original = baseline['weights'].copy()
    # This is a saved-W equality CHECK only. Never call propose_weights to
    # regenerate S101, and never compare this baseline to original frozen W.
    with np.load(DIRECT_RUN/'UPDATE.npz',allow_pickle=False) as z:
        saved_update = z['new'].copy()
    update_json = json.loads((DIRECT_RUN/'UPDATE.json').read_text())
    if original.shape!=(66,5) or original.dtype!=saved_update.dtype or original.tobytes()!=saved_update.tobytes() or np.asarray(update_json['new'],dtype=original.dtype).tobytes()!=original.tobytes():
        raise ValueError('direct baseline must equal exact SAVED s101 UPDATE.new')
    if baseline_saved['weights_sha256'] != hashlib.sha256(original.tobytes()).hexdigest():
        raise ValueError('direct baseline saved weight hash mismatch')
    active = original>0
    direction = np.random.default_rng(SEED).choice([-1.,1.],size=original.shape)
    wp = np.clip(original+DELTA*direction,np.finfo(float).eps,4.)
    wm = np.clip(original-DELTA*direction,np.finfo(float).eps,4.)
    wp[~active] = wm[~active] = 0
    expected_direction = dict(original=original,direction=direction,active=active,wp=wp,wm=wm)
    with np.load(out.parent/'DIRECTION.npz',allow_pickle=False) as z:
        for key,value in expected_direction.items():
            if z[key].shape!=value.shape or z[key].dtype!=value.dtype or z[key].tobytes()!=value.tobytes():
                raise ValueError('parent direction NPZ differs: '+key)
    direction_json = json.loads((out.parent/'DIRECTION.json').read_text())
    if direction_json['seed']!=SEED or direction_json['delta']!=DELTA:
        raise ValueError('direction seed/delta mismatch')
    for key,value in expected_direction.items():
        if np.asarray(direction_json[key],dtype=value.dtype).shape!=value.shape or np.asarray(direction_json[key],dtype=value.dtype).tobytes()!=value.tobytes():
            raise ValueError('parent direction JSON differs: '+key)
    with np.load(INITIAL,allow_pickle=False) as z:
        q,sensory_ref = z['new_qpos'].copy(),z['sensory_reference_qpos'].copy()
    if not np.array_equal(q,baseline['qpos'][0]) or any(np.any(baseline[k][0]) for k in ('qvel','act','ctrl','applied')):
        raise ValueError('direct baseline cold body differs')
    out.mkdir(parents=True,exist_ok=False); sync_dir(out.parent)
    lock = (out/'WRITER.lock').open('x'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    caps = dict(neural_ticks=2400,physics_steps=12000,initial_forward=3,diagnostic_forward=12000)
    reserved,completed = dict.fromkeys(caps,0),dict.fromkeys(caps,0)
    ledger = dict(caps=caps,reserved=reserved,completed=completed,active=None,total_forward_cap=12003,
        recovery='NO_RESUME; uncommitted full-block reservation remains charged; never retry invalid arm',
        scope='public calls only; native RK4 internal stages additional; no neural calls inside RK4')
    current_arm = None
    def reserve(block,delta):
        if ledger['active'] is not None or any(reserved[k]+v>caps[k] for k,v in delta.items()):
            raise RuntimeError('uncommitted reservation or hard cap exceeded')
        for k,v in delta.items(): reserved[k] += v
        ledger['active'] = dict(arm=current_arm,block=block,delta=delta)
        atomic_json(out/'LEDGER.json',ledger)
    def commit():
        for k,v in ledger['active']['delta'].items(): completed[k] += v
        ledger['active'] = None; atomic_json(out/'LEDGER.json',ledger)
    atomic_json(out/'LEDGER.json',ledger)
    atomic_json(out/'TIMING.json',timing)
    atomic_json(out/'INPUT_HASHES.json',identities)
    atomic_json(out/'PREREG.json',dict(arms=list(ARMS),max_fresh_evaluations=3,timing=timing,caps=caps,total_forward_cap=12003,
        seed=SEED,delta=DELTA,max_step=MAX_STEP,k=0,baseline=str(BASELINE),baseline_result=str(DIRECT_RESULT),s100_reference=str(S100_REFERENCE),
        parent_prereg=str(out.parent/'PREREG.json'),direction_files=[str(out.parent/'DIRECTION.npz'),str(out.parent/'DIRECTION.json')],
        optimizer='grounded_training.propose_weights; optimization_loss only; constant k=0, not iteration k=1; old loss report only',
        dedup='exact dtype/shape/bytes; authenticated/rescored S100 + all three S101 arms + current arms ONLY; old-pending S92 excluded; no replacement budget',
        cold='fresh body/network/FIR per fresh arm; s100 schedule unchanged; no pending, prefill or auxiliary controller',
        selection='round00_updated only; physical acceptance separately all three vs DIRECT S101 and fixed S100; strict proxy below DIRECT for training_valid',
        deployment=False,behavior_claim='NONE',physical_units='NOT_ESTABLISHED',automatic_resume=False))
    try:
        before_cb = callback_snapshot(mj); require_no_callbacks(before_cb)
        atomic_json(out/'CALLBACKS_BEFORE.json',before_cb)
        if mj.__version__!='3.9.0' or mj.mj_versionString()!='3.9.0': raise RuntimeError('MuJoCo3.9.0 required')
        atomic_json(out/'ENVIRONMENT.json',dict(executable=sys.executable,python=sys.version,platform=platform.platform(),
            versions=dict(numpy=np.__version__,scipy=scipy.__version__,mujoco=mj.__version__,native=mj.mj_versionString()),
            environment=dict(os.environ),initial_rng=rng_snapshot(np),neural_rng='existing deterministic constructor; no new seed'))
        provenance = json.loads((BASE_RUN/'MODEL_PROVENANCE.json').read_text())
        m = mj.MjModel.from_xml_path(str(MODEL))
        if (m.nu,m.nq,m.nv)!=(90,49,48) or m.opt.timestep!=BODY_DT or m.opt.integrator!=mj.mjtIntegrator.mjINT_EULER or np.count_nonzero(m.geom_type==mj.mjtGeom.mjGEOM_MESH)!=71:
            raise ValueError('original fullbody/dt/Euler mismatch')
        if np.any(m.actuator_delay) or np.any(m.actuator_history) or m.nplugin or np.any(m.actuator_plugin!=-1): raise RuntimeError('delay/history/plugin forbidden')
        before = numeric_model_snapshot(m)
        if before!=provenance['before'] or provenance['xml_sha256']!=sha(MODEL): raise RuntimeError('s100 before mismatch')
        m.opt.integrator = mj.mjtIntegrator.mjINT_RK4  # ONLY model-field assignment.
        after = numeric_model_snapshot(m)
        if after!=provenance['after'] or [k for k in before if before[k]!=after[k]]!=['opt.integrator']: raise RuntimeError('s100 RK4 after mismatch')
        tick_provenance = json.loads((DIRECT_RUN/'MODEL_PROVENANCE.json').read_text())
        if before!=tick_provenance['before'] or after!=tick_provenance['after']: raise RuntimeError('s101 model mismatch')
        atomic_json(out/'MODEL_PROVENANCE.json',dict(xml_path=str(MODEL),xml_sha256=sha(MODEL),before=before,after=after,changed_numeric_fields=['opt.integrator'],no_other_model_field_changes=True,s92_before_equal=True,s92_after_equal=True,s100_equal=True,s101_equal=True))
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
        built = build_neural_tier(); a = built['data']
        tier = retain_motor_endpoints(a,downstream_union(a,built['global_index'],['720575941433155799','720575941566493282'],hops=2,max_neurons=20000),['720575941391252886','720575941511883985','720575941497037218','720575941559306355'])
        roots = a['root_ids'][tier['global_index']]
        if len(roots)!=17099: raise ValueError('tier differs')
        edges = build_edges(a,tier,built['cut']); groups,_ = channel_index_map(len(roots),roots)
        joints,geom_leg = {},{}
        for leg in LEGS:
            js = [j for j in range(m.njnt) if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j) or '').lower().startswith('joint_'+leg)]
            tib = next(j for j in js if 'tibia' in mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j).lower())
            joints[leg] = dict(q=[int(m.jnt_qposadr[j]) for j in js],tib_q=int(m.jnt_qposadr[tib]),tib_v=int(m.jnt_dofadr[tib]))
        for g in range(m.ngeom):
            nn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) or '').lower()
            bd = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower()
            for leg in LEGS:
                if 'tars' in nn+bd and (nn.startswith(leg) or bd.startswith(leg)): geom_leg[g] = leg
        floors = np.flatnonzero(m.geom_type==mj.mjtGeom.mjGEOM_PLANE)
        if len(floors)!=1: raise ValueError('one floor required')
        floor = int(floors[0]); limited = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[limited]
        state_spec = mj.mjtState.mjSTATE_INTEGRATION
        def invariants():
            require_no_callbacks(callback_snapshot(mj))
            if numeric_model_snapshot(m)!=after: raise RuntimeError('model changed')
        def sample(d,phase):
            contacts = np.zeros(6,dtype=np.int32); force = np.zeros((6,3)); depth = 0.
            for ci in range(d.ncon):
                c = d.contact[ci]; g1,g2 = int(c.geom1),int(c.geom2)
                if floor not in (g1,g2): continue
                foot = g2 if g1==floor else g1
                if foot in geom_leg:
                    leg = LEGS.index(geom_leg[foot])
                    if c.dist<0: contacts[leg] += 1
                    f = np.zeros(6); mj.mj_contactForce(m,d,ci,f)
                    force[leg] += (1 if g1==floor else -1)*c.frame.reshape(3,3).T@f[:3]
                else: depth = max(depth,-float(c.dist))
            excess = np.maximum(np.maximum(m.jnt_range[limited,0]-d.qpos[qa],d.qpos[qa]-m.jnt_range[limited,1]),0)
            return dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),contacts=contacts,foot_force=force,nonfoot_depth=depth,
                limit_excess=float(excess.max(initial=0)),warnings=np.asarray([w.number for w in d.warning]),time=float(d.time),phase=phase,
                actuator_force=d.actuator_force.copy(),actuator_length=d.actuator_length.copy(),actuator_velocity=d.actuator_velocity.copy())
        def valid(row): return all(np.isfinite(v).all() for v in row.values()) and not np.any(row['warnings'])
        def scores(data):
            result = score_formal(data,weight_native)
            for phase,span in support_spans(data,weight_native).items(): result[phase]['longest_support_ms'] = span
            proxy = objective.proxy(data,scales)
            if set(proxy)!= {'total','parts'} or set(proxy['parts'])!=set(GATE_KEYS): raise ValueError('proxy schema')
            result.update(optimization_loss=float(proxy['total']),proxy_parts={k:float(proxy['parts'][k]) for k in GATE_KEYS})
            if not np.isfinite([result['loss'],result['optimization_loss'],*result['proxy_parts'].values()]).all(): raise ValueError('nonfinite score')
            return result
        results,cache,initial_hashes_all = [],[],{}
        cache_authentication = []
        canonical_schedule = json.loads((S100_REFERENCE.parent/'TICK_SCHEDULE.json').read_text())
        s100_setup = json.loads((BASE_RUN/'SETUP.json').read_text())
        s100_initial_meta = json.loads((S100_REFERENCE.parent/'INITIAL_STATE_META.json').read_text())
        for source in cache_sources:
            source_run = BASE_RUN if source==S100_REFERENCE else DIRECT_RUN
            saved = json.loads((source.parent/'RESULT.json').read_text())
            if saved['trajectory_path']!=str(source) or saved['trajectory_sha256']!=sha(source):
                raise ValueError('cache source result/trajectory binding failed')
            schedule = json.loads((source.parent/'TICK_SCHEDULE.json').read_text())
            # Commands legitimately differ across W; authenticate timing/order,
            # not equality of the weight-dependent actual_ctrl values.
            schedule_config = lambda rows: [{k:v for k,v in row.items() if k!='actual_ctrl'} for row in rows]
            if schedule_config(schedule)!=schedule_config(canonical_schedule):
                raise ValueError('cache tick schedule differs from s100')
            with np.load(source,allow_pickle=False) as schedule_arrays:
                if len(schedule)!=800 or any(not np.array_equal(np.asarray(row['actual_ctrl']),schedule_arrays['applied'][i]) for i,row in enumerate(schedule)):
                    raise ValueError('cache schedule command/trajectory mismatch')
            setup = json.loads((source_run/'SETUP.json').read_text())
            if any(setup[k]!=s100_setup[k] for k in ('neurons','leg_order','weight_native','timing','neural_parameters','mapping_digest')):
                raise ValueError('cache setup differs from s100')
            meta = json.loads((source.parent/'INITIAL_STATE_META.json').read_text())
            for key in ('neural_scalars','neural_configs','encoder_config','FIR_elapsed_s','FIR_interval','before_initial_forward','before_any_neural_or_body_step'):
                if meta[key]!=s100_initial_meta[key]: raise ValueError('cache cold configuration differs: '+key)
            for filename in ('NEURAL_INITIAL.npz','COLD_INITIAL.npz'):
                with np.load(source.parent/filename,allow_pickle=False) as z, np.load(S100_REFERENCE.parent/filename,allow_pickle=False) as ref:
                    if set(z.files)!=set(ref.files) or any(z[k].shape!=ref[k].shape or z[k].dtype!=ref[k].dtype or z[k].tobytes()!=ref[k].tobytes() for k in z.files):
                        raise ValueError('cache cold initial arrays differ from s100')
            with np.load(source,allow_pickle=False) as z:
                arrays = {k:z[k].copy() for k in z.files}
            weights = arrays['weights']
            if saved['weights_sha256']!=hashlib.sha256(weights.tobytes()).hexdigest(): raise ValueError('cache weight hash mismatch')
            record = scores(arrays)
            if any(record[k]!=saved[k] for k in ('loss','optimization_loss','proxy_parts','preparation','formal')):
                raise ValueError('cache source score/proxy verification failed')
            for k in record:
                if k in saved and record[k]!=saved[k]: raise ValueError('cache score field differs: '+k)
            name = 's100_tick400ms' if source==S100_REFERENCE else 's101_'+source.parent.name
            record.update(name=name,trajectory_path=str(source),trajectory_sha256=sha(source),weights_sha256=saved['weights_sha256'])
            cache.append((weights.copy(),record))
            cache_authentication.append(dict(name=name,source_result=str(source.parent/'RESULT.json'),result_sha256=sha(source.parent/'RESULT.json'),trajectory_path=str(source),trajectory_sha256=sha(source),weights_sha256=saved['weights_sha256'],same_tick_configuration=True,rescored=True,source_configuration_hashes={str(source.parent/f):sha(source.parent/f) for f in config_files}))
        s100refrescore = cache[0][1]
        baseline_record = cache[-1][1]
        if baseline_record['trajectory_path']!=str(BASELINE): raise ValueError('wrong direct cache baseline')
        atomic_json(out/'CACHE_AUTHENTICATION.json',cache_authentication)
        direct_evidence = dict(record=baseline_record,source_result=str(DIRECT_RESULT),source_result_sha256=sha(DIRECT_RESULT),same_tick_configuration=True,source_sha256=sha(S101_SOURCE),saved_update_new_bitwise_equal=True,baseline_regenerated=False)
        atomic_json(out/'DIRECT_BASELINE_RESCORE.json',direct_evidence)
        atomic_json(out/'BASELINE_VERIFIED.json',direct_evidence)
        atomic_json(out/'S100_REFERENCE_RESCORE.json',dict(record=s100refrescore,source_result=str(S100_REFERENCE.parent/'RESULT.json'),same_tick_configuration=True,comparison=str(BASE_RUN.parent/'COMPARISON.json'),source_sha256=s100_hash))
        fresh_evaluations = 0
        def evaluate(weights,ARM):
            nonlocal current_arm,fresh_evaluations
            if ARM!=ARMS[len(results)]: raise RuntimeError('fixed arm order only')
            current_arm = ARM
            dest = out/ARM; dest.mkdir(); sync_dir(out)
            atomic_json(dest/'TIMING.json',timing)
            for previous,record in cache:
                if weights.dtype==previous.dtype and weights.shape==previous.shape and weights.tobytes()==previous.tobytes():
                    source = Path(record['trajectory_path'])
                    if sha(source)!=record['trajectory_sha256']: raise ValueError('reuse trajectory changed')
                    with np.load(source,allow_pickle=False) as z:
                        reused_arrays = {k:z[k].copy() for k in z.files}
                    if reused_arrays['weights'].shape!=weights.shape or reused_arrays['weights'].dtype!=weights.dtype or reused_arrays['weights'].tobytes()!=weights.tobytes(): raise ValueError('reuse weight mismatch')
                    rescored = scores(reused_arrays)
                    if any(rescored[k]!=record[k] for k in ('loss','optimization_loss','proxy_parts','preparation','formal')): raise ValueError('reuse score/proxy mismatch')
                    reused = dict(record,name=ARM,fresh_simulation=False,reused_from=record['name'],reuse_source=str(source),same_tick_configuration=True,simulation_calls=dict(neural_ticks=0,physics_steps=0,explicit_forward=0))
                    atomic_json(dest/'RESULT.json',reused)
                    atomic_json(dest/'REUSE.json',dict(verified_hash=sha(source),verified_scores=True,exact_weight_bytes=True,same_tick_configuration=True,source_arm=record['name'],source_path=str(source),schedule_path=str(source.parent/'TICK_SCHEDULE.json'),initial_snapshot_paths=[str(source.parent/f) for f in ('NEURAL_INITIAL.npz','COLD_INITIAL.npz','INITIAL_STATE_META.json')],no_replacement_evaluation=True))
                    results.append(reused); atomic_json(out/'EVALUATIONS.json',results)
                    return reused
            if fresh_evaluations>=3: raise RuntimeError('three fresh evaluations maximum')
            fresh_evaluations += 1
            rng_before = rng_snapshot(np)
            net = ConductanceNetwork(len(roots),dt_ms=.5); net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest,weights); enc = ChannelSensoryEncoder(groups,len(roots))
            if dec.DT!=NEURAL_DT or dec.BINS!=((0,5),(5,10),(10,20),(20,40),(40,80)): raise RuntimeError('FIR changed')
            d = mj.MjData(m); d.qpos[:] = q  # Sole pose write per fresh arm.
            if d.qvel.any() or d.act.any() or d.ctrl.any() or d.time or d.xfrc_applied.any() or d.qfrc_applied.any() or dec.history.any(): raise RuntimeError('noncold')
            neural_arrays,neural_scalars,neural_configs = {},{},{}
            for key,value in vars(net).items():
                if isinstance(value,np.ndarray): neural_arrays[key] = value.copy()
                elif sparse.issparse(value):
                    mat = value.tocsr()
                    neural_arrays.update({key+'_data':mat.data.copy(),key+'_indices':mat.indices.copy(),key+'_indptr':mat.indptr.copy(),key+'_shape':np.asarray(mat.shape)})
                elif dataclasses.is_dataclass(value): neural_configs[key] = dataclasses.asdict(value)
                elif isinstance(value,(bool,int,float,str,np.generic)) or value is None: neural_scalars[key] = json_value(value)
                else: raise TypeError('unrecorded initial net field '+key)
            if net.steps or net.t_ms or net._ring_i or not np.all(net.v==net.p.rest_mV) or any(np.any(getattr(net,k)) for k in ('_ring','ge','gi','adaptation','refractory_until_ms','spike_count')): raise RuntimeError('neural cold mismatch')
            body = np.empty(mj.mj_stateSize(m,state_spec)); mj.mj_getState(m,d,body,state_spec)
            atomic_npz(dest/'NEURAL_INITIAL.npz',**neural_arrays)
            atomic_npz(dest/'COLD_INITIAL.npz',body_integration=body,qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),FIR_history=dec.history.copy(),sensory_reference=sensory_ref)
            atomic_json(dest/'INITIAL_STATE_META.json',dict(neural_scalars=neural_scalars,neural_configs=neural_configs,rng_before_constructor=rng_before,
                before_initial_forward=True,before_any_neural_or_body_step=True,encoder_config=json_value(vars(enc)),FIR_elapsed_s=dec.elapsed_s,FIR_interval=dec.interval,automatic_resume=False))
            initial_hashes = {str(dest/f):sha(dest/f) for f in ('NEURAL_INITIAL.npz','COLD_INITIAL.npz','INITIAL_STATE_META.json')}
            initial_hashes_all.update(initial_hashes); atomic_json(out/'INITIAL_STATE_HASHES.json',initial_hashes_all)
            atomic_json(out/'SETUP.json',dict(neurons=len(roots),leg_order=list(LEGS),weight_native=weight_native,timing=timing,neural_parameters=dataclasses.asdict(ConductanceParams()),mapping_digest=dec.mapping_digest,source_hashes=identities))
            atomic_npz(dest/'WEIGHTS.npz',weights=weights)
            reserve(-1,dict(initial_forward=1)); mj.mj_forward(m,d); first = sample(d,0)
            atomic_npz(dest/'INITIAL.npz',**first,weights=weights)
            if not valid(first): raise RuntimeError('invalid initial sample saved; reservation charged')
            invariants(); commit()
            rows,blocks,shards,schedule = [first],[],[],[]; raw_schema = None
            for block in range(BLOCKS):
                invariants(); reserve(block,dict(neural_ticks=10,physics_steps=50,diagnostic_forward=50))
                observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                rawobs,raw_arrays = raw_observation_record(observed)
                schema = {k:v.shape for k,v in raw_arrays.items()}
                if len(schema)!=30 or (raw_schema is not None and schema!=raw_schema): raise RuntimeError('raw30 schema differs')
                raw_schema = schema
                atomic_json(dest/f'rawobs_{block:03d}.json',dict(block=block,observation_time_s=float(d.time),quantities=rawobs,source='actual observe_quantities before encode',flattened_names=sorted(raw_arrays)))
                current,_ = enc.encode(observed)
                if current.shape!=(17099,) or not np.isfinite(current).all():
                    atomic_npz(dest/'INVALID_CURRENT.npz',current=current,**raw_arrays); raise RuntimeError('invalid current saved')
                new_rows, spikes, commands, features, tick_rows = [],[],[],[],[]
                try:
                    for tick in range(10):
                        k = block*10+tick; start,end = k*NEURAL_DT,(k+1)*NEURAL_DT
                        if not np.isclose(d.time,start,rtol=0,atol=1e-12): raise RuntimeError('physical clock differs')
                        spike = net.step(current_nA=current).copy(); spikes.append(spike)
                        if not np.isfinite(spike).all() or any(not np.isfinite(v).all() for v in vars(net).values() if isinstance(v,np.ndarray)): raise RuntimeError('nonfinite neural')
                        applied,info = dec.consume(spike[None,:],spike_dt_s=NEURAL_DT,start_s=start,end_s=end)
                        feature = np.asarray(info['pool_features']); commands.append(applied.copy()); features.append(feature)
                        if not np.isfinite(applied).all() or not np.isfinite(feature).all() or dec.interval!=k: raise RuntimeError('invalid consumption')
                        if not np.isclose(net.t_ms/1000,end,rtol=0,atol=1e-12): raise RuntimeError('neural model clock differs')
                        tick_row = dict(tick=k,block=block,sample_time_s=block*COMMAND_DT,neural_start_s=start,neural_end_s=end,
                            generation_model_tick_end_s=end,physical_apply_start_s=start,physical_apply_end_s=end,feature_end_s=info['source_end_s'],
                            decoder_model_valid_from_s=info['valid_from_s'],consume_id=info['source_interval'],actual_ctrl=applied.tolist(),logical_order='generated -> consumed/decoded -> body5steps')
                        tick_rows.append(tick_row)
                        d.ctrl[:] = applied
                        for sub in range(5):
                            require_no_callbacks(callback_snapshot(mj))
                            if d.xfrc_applied.any() or d.qfrc_applied.any() or not np.array_equal(d.ctrl,applied): raise RuntimeError('hidden command/force')
                            mj.mj_step(m,d)
                            if np.any([w.number for w in d.warning]) or any(not np.isfinite(v).all() for v in (d.qpos,d.qvel,d.act,d.ctrl)):
                                new_rows.append(sample(d,int(block>=40))); raise RuntimeError('warning/nonfinite after step')
                            require_no_callbacks(callback_snapshot(mj)); mj.mj_forward(m,d); require_no_callbacks(callback_snapshot(mj))
                            row = sample(d,int(block>=40)); new_rows.append(row)
                            if not valid(row) or not np.array_equal(row['ctrl'],applied) or d.xfrc_applied.any() or d.qfrc_applied.any(): raise RuntimeError('invalid sampled body/command/force')
                except BaseException:
                    partial = {key:np.asarray([r[key] for r in new_rows]) for key in first}
                    partial.update(current=current,spikes=np.asarray(spikes),applied=np.asarray(commands),features=np.asarray(features),**raw_arrays)
                    atomic_npz(dest/f'INVALID_BLOCK_{block:03d}.npz',**partial)
                    atomic_json(dest/'INVALID.json',dict(block=block,generated_ticks=len(spikes),consumed_ticks=len(commands),body_samples=len(new_rows),schedule=tick_rows,reservation_charged=True,automatic_resume=False))
                    raise
                data = {key:np.asarray([r[key] for r in new_rows]) for key in first}
                data.update(current=current,spikes=np.asarray(spikes,dtype=bool),applied=np.asarray(commands),decoded=np.asarray(commands),features=np.asarray(features),observation_time_s=np.asarray(block*COMMAND_DT),**raw_arrays)
                shard = dest/f'block_{block:03d}.npz'; atomic_npz(shard,**data)
                atomic_json(dest/f'ticks_{block:03d}.json',tick_rows)
                shards.append(dict(block=block,path=str(shard),sha256=sha(shard),schedule_path=str(dest/f'ticks_{block:03d}.json'),schedule_sha256=sha(dest/f'ticks_{block:03d}.json'),rawobs_path=str(dest/f'rawobs_{block:03d}.json'),rawobs_sha256=sha(dest/f'rawobs_{block:03d}.json')))
                atomic_json(dest/'SHARDS.json',shards)
                body = np.empty(mj.mj_stateSize(m,state_spec)); mj.mj_getState(m,d,body,state_spec)
                atomic_npz(dest/'LATEST_STATE.npz',body_integration=body,FIR_history=dec.history,actual_ctrl=d.ctrl.copy(),weights=weights,**{k:v for k,v in vars(net).items() if isinstance(v,np.ndarray)})
                atomic_json(dest/'LATEST_STATE_META.json',dict(block=block,neural_scalars={k:json_value(v) for k,v in vars(net).items() if isinstance(v,(bool,int,float,str,np.generic)) or v is None},FIR_elapsed_s=dec.elapsed_s,FIR_interval=dec.interval,
                    state_spec='mjSTATE_INTEGRATION',snapshot_sha256=sha(dest/'LATEST_STATE.npz'),shard_sha256=sha(shard),source_hashes=identities,initial_state_hashes=initial_hashes,complete_block=True,automatic_resume=False))
                invariants(); rows.extend(new_rows); blocks.append(data); schedule.extend(tick_rows); commit()
            arrays = {key:np.asarray([r[key] for r in rows]) for key in first}
            for key in ('spikes','current','observation_time_s',*raw_arrays): arrays[key] = np.asarray([b[key] for b in blocks])
            for key in ('applied','decoded','features'): arrays[key] = np.concatenate([b[key] for b in blocks],axis=0)
            arrays['weights'] = weights
            shapes = dict(qpos=(4001,49),qvel=(4001,48),act=(4001,90),ctrl=(4001,90),contacts=(4001,6),foot_force=(4001,6,3),
                nonfoot_depth=(4001,),limit_excess=(4001,),warnings=(4001,len(d.warning)),time=(4001,),phase=(4001,),actuator_force=(4001,90),
                actuator_length=(4001,90),actuator_velocity=(4001,90),spikes=(80,10,17099),current=(80,17099),observation_time_s=(80,),applied=(800,90),decoded=(800,90),features=(800,66,5),weights=(66,5))
            shapes.update({k:(80,)+v for k,v in raw_schema.items()})
            if set(arrays)!=set(shapes) or any(arrays[k].shape!=v for k,v in shapes.items()) or len(schedule)!=800: raise RuntimeError('incomplete trajectory')
            if not np.allclose(arrays['time'],np.arange(SAMPLES)*BODY_DT,rtol=0,atol=1e-12) or not np.array_equal(arrays['phase'],np.r_[np.zeros(2001,dtype=int),np.ones(2000,dtype=int)]) or np.any(arrays['warnings']) or any(not np.isfinite(v).all() for v in arrays.values()): raise RuntimeError('clock/phase/finite mismatch')
            if not np.array_equal(arrays['applied'],arrays['decoded']): raise RuntimeError('decode/apply mismatch')
            for k in range(TICKS):
                if not np.array_equal(arrays['ctrl'][k*5+1:(k+1)*5+1],np.broadcast_to(arrays['applied'][k],(5,90))): raise RuntimeError('tick command hold mismatch')
            if net.steps!=800 or dec.interval!=799 or not np.isclose(dec.elapsed_s,.4,rtol=0,atol=1e-12): raise RuntimeError('exact tick consumption mismatch')
            atomic_npz(dest/'TRAJECTORY.npz',**arrays); atomic_json(dest/'TICK_SCHEDULE.json',schedule)
            score = scores(arrays)
            violations = np.asarray(objective.violations(arrays),float); scale_array = np.asarray([scales[k] for k in GATE_KEYS])
            if violations.shape!=(4001,6) or not np.isfinite(violations).all() or np.any(violations<0): raise ValueError('invalid violations')
            atomic_npz(dest/'PROXY_SERIES.npz',time=arrays['time'],phase=arrays['phase'],gate_keys=np.asarray(GATE_KEYS),scales=scale_array,violations=violations,contributions=(violations/scale_array)**2)
            record = dict(name=ARM,**score,trajectory_path=str(dest/'TRAJECTORY.npz'),trajectory_sha256=sha(dest/'TRAJECTORY.npz'),fresh_simulation=True,
                weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),simulation_calls=dict(neural_ticks=800,physics_steps=4000,explicit_forward=4001),same_tick_configuration=True,acceptance_updates=False,deployment=False)
            atomic_json(dest/'RESULT.json',record)
            invariants()
            if any(sha(p)!=h for p,h in initial_hashes.items()): raise RuntimeError('initial evidence changed')
            cache.append((weights.copy(),record)); results.append(record); atomic_json(out/'EVALUATIONS.json',results)
            return record

        plus = evaluate(wp,ARMS[0]); minus = evaluate(wm,ARMS[1])
        lp,lm = float(plus['optimization_loss']),float(minus['optimization_loss'])
        new = np.asarray(propose_weights(original,lp,lm,wp,wm,k=0,max_step=MAX_STEP),float)
        separation = wp-wm
        secant = np.divide(lp-lm,separation,out=np.zeros_like(original),where=active)
        gradient = secant.copy()
        raw_step = -.05/(0+1)**.602 * secant
        scale = min(1.,MAX_STEP/max(float(np.max(np.abs(raw_step))),np.finfo(float).tiny))
        scaled_step = scale*raw_step
        expected_new = np.clip(original+scaled_step,np.finfo(float).eps,4.)
        expected_new[~active] = 0
        effective_delta = new-original
        if (new.shape!=original.shape or new.tobytes()!=expected_new.tobytes() or not np.isfinite(new).all() or
                np.any(new[~active]!=0) or np.any(new[active]<np.finfo(float).eps) or np.any(new>4) or
                np.max(np.abs(effective_delta))>MAX_STEP+1e-12):
            raise ValueError('optimizer exact formula/bounds/mask/maxstep mismatch')
        audit = dict(original=original,direction=direction,active=active,wp=wp,wm=wm,separation=separation,
            raw_secant=secant,gradient=gradient,raw_step=raw_step,scaled_step=scaled_step,new=new,effective_delta=effective_delta)
        # Single centered directional secant projection, NOT 330 independent gradients.
        center = (wp+wm)/2
        halfspan = (wp-wm)/2
        projection = float(np.sum((new-center)*halfspan)/np.sum(halfspan*halfspan))
        predicted_change = float((lp-lm)/2*projection)
        atomic_npz(out/'UPDATE.npz',**audit,lp=np.asarray(lp),lm=np.asarray(lm),scale=np.asarray(scale),k=np.asarray(0),max_step=np.asarray(MAX_STEP),old_lp=np.asarray(plus['loss']),old_lm=np.asarray(minus['loss']),optimization_lp=np.asarray(lp),optimization_lm=np.asarray(lm),center=center,halfspan=halfspan,centered_projection=np.asarray(projection),predicted_change=np.asarray(predicted_change))
        atomic_json(out/'UPDATE.json',dict(**{k:v.tolist() for k,v in audit.items()},weights=original.tolist(),rawsecant=secant.tolist(),rawstep=raw_step.tolist(),delta=effective_delta.tolist(),
            lp=lp,lm=lm,optimization_lp=lp,optimization_lm=lm,old_lp=plus['loss'],old_lm=minus['loss'],objective_hashes=objective_hashes,scales=scales,k=0,max_step=MAX_STEP,scale=scale,
            objective='optimization_loss only; old loss report only',formula=dict(raw_secant='divide(lp-lm,wp-wm,out=zeros_like(w),where=w>0)',raw_step='-.05/(k+1)**.602 * raw_secant',scale='min(1.,max_step/max(float(max(abs(raw_step))),finfo(float).tiny))',new='clip(w+scale*raw_step,finfo(float).eps,4.); new[w<=0]=0',effective_delta='new-w'),
            formula_bitwise_verified=True,center=center.tolist(),halfspan=halfspan.tolist(),centered_projection=projection,predicted_change=predicted_change,prediction_formula='(lp-lm)/2 * dot(new-center,halfspan)/dot(halfspan,halfspan); scalar directional projection, NOT sum330',gradient_claim=False))
        fields = ['flat_index','pool','bin']+list(audit)+['lp','lm','scale','k','max_step']
        atomic_csv(out/'UPDATE330.csv',fields,[dict(flat_index=i,pool=i//5,bin=i%5,**{key:value.ravel()[i].item() for key,value in audit.items()},lp=lp,lm=lm,scale=scale,k=0,max_step=MAX_STEP) for i in range(330)])
        # Explicit durable candidate and SHA evidence BEFORE evaluate is called.
        atomic_npz(out/'UPDATED_CANDIDATE.npz',weights=new)
        candidate_sha = hashlib.sha256(new.tobytes()).hexdigest()
        candidate_file_sha = sha(out/'UPDATED_CANDIDATE.npz')
        atomic_json(out/'UPDATED_CANDIDATE.json',dict(weights=new.tolist(),shape=list(new.shape),dtype=str(new.dtype),weights_sha256=candidate_sha,npz_path=str(out/'UPDATED_CANDIDATE.npz'),npz_sha256=candidate_file_sha,before_candidate_evaluation=True,update_path=str(out/'UPDATE.npz'),update_sha256=sha(out/'UPDATE.npz'),audit330_sha256=sha(out/'UPDATE330.csv'),k=0,max_step=MAX_STEP,direct_baseline=str(BASELINE)))
        updated = evaluate(new,ARMS[2])
        if sha(out/'UPDATED_CANDIDATE.npz')!=candidate_file_sha or updated['weights_sha256']!=candidate_sha:
            raise ValueError('candidate changed during evaluation')
        comparisons = {record['name']:acceptance(baseline_record,record,tolerance=1e-12) for record in results}
        comparisons_s100 = {record['name']:acceptance(s100refrescore,record,tolerance=1e-12) for record in results}
        strictly_lower = bool(updated['optimization_loss']<baseline_record['optimization_loss'])
        cumulative_s100_proxy_change = float(updated['optimization_loss']-s100refrescore['optimization_loss'])
        decision = dict(selected_arm=ARMS[2],selection_rule='updated only eligible for later validation; never best-arm selection; no deployment',baseline=baseline_record,direct_baseline=baseline_record,s100refrescore=s100refrescore,all_arms=results,comparisons=comparisons,comparisons_s100=comparisons_s100,
            **acceptance(baseline_record,updated,tolerance=1e-12),s100_acceptance=comparisons_s100[ARMS[2]],proxy_strictly_lower_baseline=strictly_lower,proxy_strictly_lower_direct_baseline=strictly_lower,cumulative_s100_proxy_change=cumulative_s100_proxy_change,training_valid=strictly_lower,training_valid_scope='strict proxy improvement vs DIRECT S101 only, NOT physical acceptance/support/walking',deployment=False,behavior_claim='NONE')
        atomic_json(out/'ACCEPTANCE.json',decision)
        expected_calls = dict(neural_ticks=800*fresh_evaluations,physics_steps=4000*fresh_evaluations,initial_forward=fresh_evaluations,diagnostic_forward=4000*fresh_evaluations)
        if reserved!=expected_calls or completed!=expected_calls or ledger['active'] is not None: raise RuntimeError('exact fresh-arm call ledger mismatch')
        if identities!={str(p):sha(p) for p in paths} or any(sha(p)!=h for p,h in initial_hashes_all.items()): raise RuntimeError('source/input/initial evidence changed')
        invariants(); end_cb = callback_snapshot(mj)
        if end_cb!=before_cb: raise RuntimeError('callbacks changed')
        atomic_json(out/'CALLBACKS_END.json',end_cb)
        atomic_json(out/'MODEL_END_CHECK.json',dict(passed=True,snapshot=numeric_model_snapshot(m),matches_after=True,s92_after_equal=True,s100_equal=True,s101_equal=True,callbacks_none=True,source_hashes_unchanged=True))
        atomic_json(out/'SUMMARY.json',dict(status='BOUNDED_EXPERIMENT_COMPLETE',baseline_rescore=baseline_record,direct_baseline_rescore=baseline_record,s100refrescore=s100refrescore,evaluations=results,comparisons=comparisons,comparisons_s100=comparisons_s100,accepted=decision['accepted'],s100_accepted=comparisons_s100[ARMS[2]]['accepted'],training_valid=strictly_lower,proxy_strictly_lower_baseline=strictly_lower,cumulative_s100_proxy_change=cumulative_s100_proxy_change,fresh_evaluations=fresh_evaluations,ledger=ledger,timing=timing,weight_native=weight_native,initial_state_hashes=initial_hashes_all,objective_hashes=objective_hashes,scales=scales,selection_eligible_arm=ARMS[2],selection=False,deployment=False,behavior_claim='NONE',physical_units='NOT_ESTABLISHED',automatic_resume=False,note='No extra direction/backtrack/update; DIRECT S101 saved updated baseline, fixed S100 physical reference; old loss report only; all three physical comparisons separate against BOTH'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
