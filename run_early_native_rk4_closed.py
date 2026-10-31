"""S2/S4 preparation-only native RK4 closed-loop 20ms diagnostic.

Explicit derivative of run_early_timestep_refinement.py; model/callback checks
from run_early_native_rk4_replay.py. Original weights, cold q/zero v/act/ctrl,
fresh neural network/FIR, original observe->10 neural ticks->old pending hold
50 native 100us steps->consume ordering. Four blocks only; NO_BEHAVIOR_CLAIM.
Only compiled-model mutation: opt.integrator Euler -> mjINT_RK4; dt unchanged.
40 neural ticks, 200 public mj_step, 201 explicit mj_forward; RK4 internal
stages are additional native workload, never additional neural calls.
No optimization, scalar scoring, 400ms run, automatic resume or calls on
import/--help. Historical trajectories are hash references, not input commands.
"""
from pathlib import Path
import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parent
MODEL = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
INITIAL = Path('/run/media/sensen/Data2/engine修复验收/s7-posture-ik/20261023T021141Z/DERIVED_INITIAL.npz')
ENDPOINTS = Path('/run/media/sensen/Data2/engine修复验收/s4-endpoints/20261023T011559Z')
BASE_RUN = Path('/home/sensen/Desktop/engine修复验收/s79-grounded-translation/20261023T185506Z/run')
BASELINE = BASE_RUN/'baseline/TRAJECTORY.npz'
S90 = Path('/home/sensen/Desktop/engine修复验收/s90-native-rk4/20261023T205300Z/run')
DERIVED_FROM = ROOT/'run_early_timestep_refinement.py'
CHECKS_FROM = ROOT/'run_early_native_rk4_replay.py'
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s91-rk4-closed/20261023T205921Z/run')
SOURCE_TAG = '3.9.0'
BODY_DT = .0001
NEURAL_DT = .0005
COMMAND_DT = .005
BLOCKS = 4
PHYSICS_PER_BLOCK = 50


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def numeric_model_snapshot(model):
    # Same exposed numeric snapshot algorithm as s86/s90; no second compilation.
    import numpy as np
    fields = {}
    for prefix, obj in (('model',model),('opt',model.opt)):
        for key in dir(obj):
            if key.startswith('_'):
                continue
            value = getattr(obj,key)
            if isinstance(value,np.ndarray):
                fields[prefix+'.'+key] = dict(shape=list(value.shape),dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
            elif isinstance(value,(bool,int,float,np.integer,np.floating)):
                fields[prefix+'.'+key] = value.item() if isinstance(value,np.generic) else value
    return fields


def callback_snapshot(mj):
    """Get identities only; never install/clear callbacks or invoke their bodies."""
    result = {}
    names = ('control', 'act_dyn', 'act_gain', 'act_bias', 'passive')
    for name in names:
        getter_name = 'get_mjcb_'+name
        getter = getattr(mj, getter_name, None)
        if getter is None:
            if name == 'control':
                raise RuntimeError('required get_mjcb_control getter not exposed')
            result[name] = dict(exposed=False, identity=None)
            continue
        callback = getter()
        result[name] = dict(exposed=True, identity=None if callback is None else
            dict(id=id(callback),module=getattr(callback,'__module__',None),
                 qualname=getattr(callback,'__qualname__',None),repr=repr(callback)))
    return result


def require_no_callbacks(snapshot):
    if not snapshot['control']['exposed'] or any(v['identity'] is not None for v in snapshot.values()):
        raise RuntimeError('callbacks forbidden; no hidden controls/dynamics/gain/bias/passive forces')


def raw_observation_record(observed):
    """Serialize actual observe_quantities dict, not a reconstructed observation."""
    import numpy as np
    flattened = {}

    def convert(value, path):
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        if isinstance(value, dict):
            return {str(k):convert(v,path+[str(k)]) for k,v in value.items()}
        if isinstance(value, (list,tuple,np.ndarray)):
            arr = np.asarray(value)
            if arr.dtype.kind not in 'biuf' or not np.isfinite(arr).all():
                raise ValueError('raw sensory observation must be finite numeric data')
            flattened['rawobs_'+'_'.join(path)] = arr.copy()
            return arr.tolist()
        if isinstance(value, (bool,int,float,np.integer,np.floating,np.bool_)):
            if not np.isfinite(value):
                raise ValueError('nonfinite raw sensory observation')
            flattened['rawobs_'+'_'.join(path)] = np.asarray(value).copy()
            return value.item() if isinstance(value,np.generic) else value
        raise TypeError('unsupported raw sensory quantity '+repr(type(value)))

    record = convert(observed, [])
    return record, flattened


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT,
                   help='Fixed fresh output path: '+str(OUTPUT))
    return p


def run(args):
    import numpy as np
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork
    from run_neural_muscle_closure import observe_quantities

    out = args.output
    if out != OUTPUT or not out.is_absolute() or out.exists():
        raise ValueError('fixed fresh output required; preregistered parent may exist; no resume')
    out.mkdir(parents=True, exist_ok=False)
    sync_dir(out.parent)
    sync_dir(out.parent.parent)
    lock = (out/'WRITER.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    caps = dict(neural_ticks=40, physics_steps=200, initial_forward=1, diagnostic_forward=200)
    reserved = dict.fromkeys(caps, 0)
    completed = dict.fromkeys(caps, 0)
    ledger = dict(caps=caps, reserved=reserved, completed=completed, active=None,
        total_forward_cap=201,
        scope='public calls only; RK4 internal native stages exceed Euler workload; no staged neural calls',
        recovery='NO_AUTOMATIC_RESUME; uncommitted reservations remain charged')
    timing = dict(blocks=BLOCKS, duration_s=.020, body_dt_s=BODY_DT,
        original_body_dt_s=.0001, neural_dt_s=NEURAL_DT, command_dt_s=COMMAND_DT,
        samples=201, neural_ticks_per_block=10, physics_steps_per_block=PHYSICS_PER_BLOCK,
        schedule='observe encode -> 10 neural ticks .0005 -> old pending 50 physics steps .0001 -> consume -> next pending',
        intervals_s=[[b*COMMAND_DT,(b+1)*COMMAND_DT] for b in range(BLOCKS)],
        phase='ALL_PREPARATION_ONLY; no formal phase; no 400ms run',
        sensory_encoding='same stateless original encoder/config; no dt rescaling',
        FIR='unchanged DT=.0005 and bins ((0,5),(5,10),(10,20),(20,40),(40,80))')
    atomic_json(out/'LEDGER.json', ledger)
    atomic_json(out/'PREREG.json', dict(evaluations=1, candidates=['closed20ms'],
        status='NO_BEHAVIOR_CLAIM', derived_from=str(DERIVED_FROM), checks_from=str(CHECKS_FROM),timing=timing,
        baseline_path=str(BASELINE), baseline_usage='read-only identity/reference; zero rerun calls or replay commands',
        s90_usage='model before/after identity and historical trajectory hash reference only; zero historical simulation',
        objective='NONE; no optimization, scalar score, score_formal or acceptance decision',
        initial_weights='original frozen 66x5 FIR; all 330 coefficients unchanged',
        cold_state='s79 q; zero qvel/act/ctrl/pending, fresh network and FIR history',
        controller='binary MN spikes -> fixed FIR -> muscle; sole initial q write; no auxiliary force',
        sole_model_change='compile same XML once; assert dt .0001 and Euler; set opt.integrator=mjINT_RK4 only',
        actuator_records='actual actuator_force/actuator_length/actuator_velocity 201x90',
        raw_observations='actual observe_quantities output before encoding each block; JSON plus flattened rawobs arrays',
        pending_aliases=dict(applied='pending_before',decoded='pending_after',
            final_pending='decoded at 20ms saved but never applied after window'),
        stopping='any nonfinite or MuJoCo warning: save partial shard and stop',
        caps=caps, total_forward_cap=201, automatic_resume=False))
    shared_paths = [MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
        ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
        ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
        ROOT/'run_neural_muscle_closure.py', ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    paths = shared_paths + [Path(__file__).resolve(), DERIVED_FROM, CHECKS_FROM,BASE_RUN/'INPUT_HASHES.json',
        BASE_RUN/'SETUP.json',BASE_RUN/'baseline/RESULT.json',BASELINE,
        S90/'INPUT_HASHES.json',S90/'MODEL_PROVENANCE.json',S90/'replay20ms/RESULT.json',
        S90/'replay20ms/TRAJECTORY.npz']

    def reserve(name, block, delta):
        if ledger['active'] is not None:
            raise RuntimeError('uncommitted reservation; refusing another call')
        for key, value in delta.items():
            if reserved[key]+value > caps[key]:
                raise RuntimeError('hard call budget exceeded')
        for key, value in delta.items():
            reserved[key] += value
        ledger['active'] = dict(evaluation=name, block=block, delta=delta,
            start_s=max(block,0)*COMMAND_DT,
            end_s=(block+1)*COMMAND_DT if block >= 0 else 0.)
        atomic_json(out/'LEDGER.json', ledger)

    def commit():
        for key, value in ledger['active']['delta'].items():
            completed[key] += value
        ledger['active'] = None
        atomic_json(out/'LEDGER.json', ledger)

    try:
        callbacks_before = callback_snapshot(mj)
        atomic_json(out/'CALLBACKS_BEFORE.json',callbacks_before)
        require_no_callbacks(callbacks_before)
        if mj.__version__ != SOURCE_TAG or mj.mj_versionString() != SOURCE_TAG:
            raise RuntimeError('s90 identity requires installed/native MuJoCo3.9.0')
        identities = {str(p):sha(p) for p in paths}
        historical = json.loads((BASE_RUN/'INPUT_HASHES.json').read_text())
        old_runner = str(ROOT/'run_grounded_translation_training.py')
        expected = {p:h for p,h in historical.items() if p != old_runner}
        actual = {str(p):identities[str(p)] for p in shared_paths}
        s90_hashes = json.loads((S90/'INPUT_HASHES.json').read_text())
        if len(actual) != 14 or actual != expected or any(s90_hashes[p] != h for p,h in actual.items()):
            raise ValueError('shared14 s79/s90 source/input hashes differ; refusing all calls')
        if (s90_hashes[str(CHECKS_FROM)] != identities[str(CHECKS_FROM)]
                or s90_hashes[str(DERIVED_FROM)] != identities[str(DERIVED_FROM)]):
            raise ValueError('derivative/check source identities differ from s90')
        atomic_json(out/'INPUT_HASHES.json', identities)
        atomic_json(out/'SHARED_SOURCE_CHECK.json', dict(passed=True, shared=actual,
            excluded_historical_runner=old_runner, derived_from=str(DERIVED_FROM),
            checks_from=str(CHECKS_FROM),derivative_sha256=identities[str(Path(__file__).resolve())]))
        manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
        original = np.array(json.loads((ENDPOINTS/'FROZEN_WEIGHTS.json').read_text())['weights'],float)
        if (original.shape != (66,5) or not np.isfinite(original).all()
                or np.count_nonzero(original) != 330 or np.any((original <= 0)|(original > 4))):
            raise ValueError('expected unchanged original 330 positive bounded FIR coefficients')
        with np.load(INITIAL,allow_pickle=False) as initial:
            q = initial['new_qpos'].copy()
            sensory_ref = initial['sensory_reference_qpos'].copy()
        if (q.shape != (49,) or sensory_ref.shape != (49,) or
                not np.isfinite(q).all() or not np.isfinite(sensory_ref).all()):
            raise ValueError('invalid fixed initial or sensory reference')
        baseline_result = json.loads((BASE_RUN/'baseline/RESULT.json').read_text())
        s90_result = json.loads((S90/'replay20ms/RESULT.json').read_text())
        if (baseline_result['trajectory_sha256'] != identities[str(BASELINE)]
                or s90_result['trajectory_sha256'] != identities[str(S90/'replay20ms/TRAJECTORY.npz')]):
            raise ValueError('saved reference trajectory hash differs from historical RESULT')
        with np.load(BASELINE,allow_pickle=False) as saved:
            if (saved['qpos'].shape != (4001,49) or not np.array_equal(saved['weights'],original)
                    or not np.array_equal(saved['qpos'][0],q)
                    or any(np.any(saved[k][0]) for k in ('qvel','act','ctrl'))
                    or not np.allclose(saved['time'][:201],np.arange(201)*BODY_DT,rtol=0,atol=1e-12)):
                raise ValueError('saved baseline cold initial state/clock/weights mismatch')
        atomic_json(out/'REFERENCE_IDENTITIES.json',dict(
            baseline=dict(path=str(BASELINE),sha256=identities[str(BASELINE)],
                usage='cold initial state/clock/weights identity; no commands used',
                historical_actuator_records='MISSING; never reconstructed'),
            s90=dict(provenance_path=str(S90/'MODEL_PROVENANCE.json'),
                provenance_sha256=identities[str(S90/'MODEL_PROVENANCE.json')],
                trajectory_path=str(S90/'replay20ms/TRAJECTORY.npz'),
                trajectory_sha256=identities[str(S90/'replay20ms/TRAJECTORY.npz')],
                trajectory_usage='HASH_REFERENCE_ONLY; not loaded or replayed'),
            read_only=True,historical_compile_calls=0,historical_simulation_calls=0))
        m = mj.MjModel.from_xml_path(str(MODEL))
        if ((m.nu,m.nq,m.nv) != (90,49,48) or m.opt.timestep != BODY_DT
                or m.opt.integrator != mj.mjtIntegrator.mjINT_EULER
                or np.count_nonzero(m.geom_type == mj.mjtGeom.mjGEOM_MESH) != 71):
            raise ValueError('unexpected original full physical model/dt/Euler integrator')
        if (np.any(m.actuator_delay) or np.any(m.actuator_history) or m.nplugin != 0
                or np.any(m.actuator_plugin != -1)):
            raise RuntimeError('delayed/history/plugin actuator paths forbidden')
        atomic_json(out/'NATIVE_INPUT_PATH_CHECK.json',dict(
            actuator_delay=m.actuator_delay.tolist(),actuator_history=m.actuator_history.tolist(),
            actuator_historyadr=m.actuator_historyadr.tolist(),nplugin=int(m.nplugin),
            actuator_plugin=m.actuator_plugin.tolist(),no_delay_history_plugins=True))
        s90_provenance = json.loads((S90/'MODEL_PROVENANCE.json').read_text())
        before = numeric_model_snapshot(m)
        if before != s90_provenance['before'] or s90_provenance['xml_sha256'] != identities[str(MODEL)]:
            raise RuntimeError('compiled before snapshot differs from s90 original model')
        old_dt = float(m.opt.timestep)
        old_integrator = int(m.opt.integrator)
        m.opt.integrator = mj.mjtIntegrator.mjINT_RK4  # The ONLY model-field assignment.
        after = numeric_model_snapshot(m)
        changed = [k for k in before if before[k] != after[k]]
        if (before.keys() != after.keys() or changed != ['opt.integrator'] or m.opt.timestep != BODY_DT
                or m.opt.integrator != mj.mjtIntegrator.mjINT_RK4 or after != s90_provenance['after']):
            raise RuntimeError('integrator-only change / exact s90 RK4 after snapshot check failed')
        atomic_json(out/'MODEL_PROVENANCE.json',dict(xml_path=str(MODEL),xml_sha256=identities[str(MODEL)],
            original_timestep_s=old_dt,runtime_timestep_s=float(m.opt.timestep),
            original_integrator=old_integrator,runtime_integrator=int(m.opt.integrator),
            changed_numeric_fields=changed,before=before,after=after,
            no_other_model_field_changes=True,s90_before_equal=True,s90_after_equal=True,
            s90_provenance_sha256=identities[str(S90/'MODEL_PROVENANCE.json')],
            compile='same XML compiled once; no mj_setConst or dt/muscle/contact/qact edits'))
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        old_setup = json.loads((BASE_RUN/'SETUP.json').read_text())
        if weight_native != old_setup['weight_native'] or old_setup['leg_order'] != list(LEGS):
            raise ValueError('native force reference/leg order differs from s79')
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
            joints[leg] = dict(q=[int(m.jnt_qposadr[j]) for j in js],
                              tib_q=int(m.jnt_qposadr[tib]),tib_v=int(m.jnt_dofadr[tib]))
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
        atomic_json(out/'SETUP.json',dict(neurons=len(roots),leg_order=list(LEGS),
            weight_native=weight_native,physical_units='NOT_ESTABLISHED',timing=timing,
            original_timestep_s=old_dt,runtime_timestep_s=float(m.opt.timestep),
            original_integrator=old_integrator,runtime_integrator=int(m.opt.integrator),
            model_provenance_path=str(out/'MODEL_PROVENANCE.json'),
            model_provenance_sha256=sha(out/'MODEL_PROVENANCE.json'),
            no_other_model_field_changes=True,callbacks_before=callbacks_before,
            mapping_digest=FiniteMotorTranslator(manifest,original).mapping_digest,
            neural_parameters=dataclasses.asdict(ConductanceNetwork(len(roots),dt_ms=.5).p),
            dependencies=dict(python=sys.version,numpy=np.__version__,mujoco=mj.__version__,native=mj.mj_versionString()),
            source_hashes=identities))

        def check_invariants():
            require_no_callbacks(callback_snapshot(mj))
            if numeric_model_snapshot(m) != after or m.opt.timestep != BODY_DT:
                raise RuntimeError('runtime model changed after integrator-only setup')

        def sample(d):
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
            excess = np.maximum(np.maximum(m.jnt_range[limited,0]-d.qpos[qa],
                                            d.qpos[qa]-m.jnt_range[limited,1]),0)
            return dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),
                contacts=contacts,foot_force=force,nonfoot_depth=depth,
                limit_excess=float(excess.max(initial=0)),warnings=np.array([w.number for w in d.warning]),
                time=float(d.time),phase=0,actuator_force=d.actuator_force.copy(),
                actuator_length=d.actuator_length.copy(),actuator_velocity=d.actuator_velocity.copy())

        finite_keys = ('qpos','qvel','act','ctrl','foot_force','nonfoot_depth','limit_excess',
                       'time','actuator_force','actuator_length','actuator_velocity')

        def evaluate(weights,name):
            if identities != {str(p):sha(p) for p in paths}:
                raise RuntimeError('source/input identities changed before evaluation')
            if not np.array_equal(weights,original) or name != 'closed20ms':
                raise ValueError('only original frozen weights / closed20ms permitted')
            dest = out/name
            dest.mkdir(exist_ok=False)
            sync_dir(out)
            net = ConductanceNetwork(len(roots),dt_ms=.5)
            net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest,weights)
            if dec.DT != NEURAL_DT or dec.BINS != ((0,5),(5,10),(10,20),(20,40),(40,80)):
                raise RuntimeError('translator timing changed')
            enc = ChannelSensoryEncoder(groups,len(roots))
            check_invariants()
            d = mj.MjData(m)
            d.qpos[:] = q  # Sole pose write; no phase-boundary writes, act writes or resets.
            pending = np.zeros(90)
            if (d.qvel.any() or d.act.any() or d.ctrl.any() or d.time != 0
                    or d.xfrc_applied.any() or d.qfrc_applied.any() or dec.history.any()):
                raise RuntimeError('noncold initial state or unexpected force')
            reserve(name,-1,dict(initial_forward=1))
            mj.mj_forward(m,d)
            check_invariants()
            initial_row = sample(d)
            atomic_npz(dest/'INITIAL.npz',**initial_row,weights=weights)
            if not all(np.isfinite(initial_row[k]).all() for k in finite_keys) or np.any(initial_row['warnings']):
                atomic_json(dest/'INVALID.json',dict(block=-1,reason='invalid_initial_state'))
                raise RuntimeError('invalid initial state saved; reservation remains charged')
            commit()
            rows,block_data,shard_index = [initial_row],[],[]
            for block in range(BLOCKS):
                check_invariants()
                reserve(name,block,dict(neural_ticks=10,physics_steps=50,diagnostic_forward=50))
                observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                rawobs,raw_arrays = raw_observation_record(observed)
                atomic_json(dest/f'rawobs_{block:03d}.json',dict(block=block,
                    observation_time_s=float(d.time),quantities=rawobs,
                    source='actual observe_quantities output before encode',
                    flattened_names=sorted(raw_arrays)))
                current,_ = enc.encode(observed)
                if not np.isfinite(current).all():
                    raise RuntimeError('nonfinite sensory current')
                # Exactly ten original outer-clock neural calls; none in native RK4 stages.
                spikes = np.array([net.step(current_nA=current).copy() for _ in range(10)],dtype=bool)
                applied = pending.copy()
                d.ctrl[:] = applied
                new_rows,invalid = [],None
                for sub in range(PHYSICS_PER_BLOCK):
                    require_no_callbacks(callback_snapshot(mj))
                    if (d.xfrc_applied.any() or d.qfrc_applied.any()
                            or not np.array_equal(d.ctrl,applied)):
                        raise RuntimeError('unexpected external force or hidden command')
                    mj.mj_step(m,d)
                    require_no_callbacks(callback_snapshot(mj))
                    mj.mj_forward(m,d)
                    require_no_callbacks(callback_snapshot(mj))
                    row = sample(d)
                    new_rows.append(row)
                    if (not all(np.isfinite(row[k]).all() for k in finite_keys)
                            or np.any(row['warnings']) or not np.array_equal(row['ctrl'],applied)
                            or d.xfrc_applied.any() or d.qfrc_applied.any()):
                        invalid = dict(block=block,substep=sub,reason='nonfinite_warning_hidden_command_or_force')
                        break
                pending,info = dec.consume(spikes,spike_dt_s=.0005,
                    start_s=block*.005,end_s=(block+1)*.005)
                if not np.isfinite(pending).all() or not np.isfinite(info['pool_features']).all():
                    invalid = dict(block=block,reason='nonfinite_decoded_or_features')
                data = {key:np.array([r[key] for r in new_rows]) for key in initial_row}
                data.update(current=current,spikes=spikes,applied=applied,decoded=pending,
                    pending_before=applied.copy(),pending_after=pending.copy(),
                    observation_time_s=np.asarray(block*COMMAND_DT),
                    features=np.asarray(info['pool_features']),**raw_arrays)
                shard = dest/f'block_{block:03d}.npz'
                atomic_npz(shard,**data)
                shard_index.append(dict(block=block,path=str(shard),sha256=sha(shard),
                    start_s=block*.005,end_s=(block+1)*.005,
                    rawobs_path=str(dest/f'rawobs_{block:03d}.json'),
                    rawobs_sha256=sha(dest/f'rawobs_{block:03d}.json')))
                atomic_json(dest/'SHARDS.json',shard_index)
                check_invariants()
                body = np.empty(mj.mj_stateSize(m,spec))
                mj.mj_getState(m,d,body,spec)
                snapshot = {key:value for key,value in vars(net).items() if isinstance(value,np.ndarray)}
                atomic_npz(dest/'LATEST_STATE.npz',body_integration=body,FIR_history=dec.history,
                    pending=pending,pending_before=applied,pending_after=pending,weights=weights,**snapshot)
                scalars = {key:value for key,value in vars(net).items() if isinstance(value,(int,float,str))}
                atomic_json(dest/'LATEST_STATE_META.json',dict(block=block,neural_scalars=scalars,
                    start_s=block*.005,end_s=(block+1)*.005,
                    FIR_elapsed_s=dec.elapsed_s,FIR_interval=dec.interval,
                    state_spec='mjSTATE_INTEGRATION',state_size=int(body.size),
                    pending_aliases=dict(applied='pending_before; held during this physical block',
                        decoded='pending_after; eligible only for next physical block',
                        final_pending_applied=False if block == BLOCKS-1 else None),
                    snapshot_sha256=sha(dest/'LATEST_STATE.npz'),shard=str(shard),shard_sha256=sha(shard),
                    input_hashes=identities,encoder='stateless physical encode; unchanged config/reference',
                    model_snapshot_equal=True,callbacks_none=True,
                    complete_block=invalid is None,automatic_resume=False))
                rows.extend(new_rows)
                block_data.append(data)
                if invalid is not None:
                    atomic_json(dest/'INVALID.json',invalid)
                    raise RuntimeError('invalid trajectory: partial shard saved, reservation conservatively charged')
                commit()
            arrays = {key:np.array([r[key] for r in rows]) for key in initial_row}
            for key in ('spikes','current','applied','decoded','features','pending_before',
                        'pending_after','observation_time_s',*raw_arrays):
                arrays[key] = np.array([b[key] for b in block_data])
            arrays['weights'] = weights
            shapes = dict(qpos=(201,49),qvel=(201,48),act=(201,90),ctrl=(201,90),
                contacts=(201,6),foot_force=(201,6,3),nonfoot_depth=(201,),limit_excess=(201,),
                actuator_force=(201,90),actuator_length=(201,90),actuator_velocity=(201,90),
                current=(4,17099),spikes=(4,10,17099),features=(4,66,5),applied=(4,90),decoded=(4,90),
                pending_before=(4,90),pending_after=(4,90),observation_time_s=(4,))
            if any(arrays[k].shape != shape for k,shape in shapes.items()):
                raise RuntimeError('incomplete trajectory/neural/actual actuator records')
            if (arrays['time'].shape != (201,) or
                    not np.allclose(arrays['time'],np.arange(201)*BODY_DT,rtol=0,atol=1e-12)
                    or np.any(arrays['phase']) or np.any(arrays['warnings'])
                    or not np.array_equal(arrays['weights'],original)):
                raise RuntimeError('20ms clock/preparation/original weights mismatch')
            if (not np.array_equal(arrays['pending_before'],arrays['applied'])
                    or not np.array_equal(arrays['pending_after'],arrays['decoded'])
                    or np.any(arrays['applied'][0])
                    or not np.array_equal(arrays['applied'][1:],arrays['decoded'][:-1])):
                raise RuntimeError('old pending / decoded-next ordering mismatch')
            for block in range(BLOCKS):
                if not np.array_equal(arrays['ctrl'][block*50+1:(block+1)*50+1],
                                      np.broadcast_to(arrays['applied'][block],(50,90))):
                    raise RuntimeError('physical held command mismatch')
            check_invariants()
            callbacks_end = callback_snapshot(mj)
            atomic_json(out/'CALLBACKS_END.json',callbacks_end)
            require_no_callbacks(callbacks_end)
            if callbacks_before != callbacks_end:
                raise RuntimeError('callback identities changed during evaluation')
            atomic_json(out/'MODEL_END_CHECK.json',dict(passed=True,snapshot=numeric_model_snapshot(m),
                matches_after=True,s90_after_equal=True,callbacks_none=True))
            atomic_npz(dest/'TRAJECTORY.npz',**arrays)
            record = dict(name=name,status='NO_BEHAVIOR_CLAIM',
                trajectory_path=str(dest/'TRAJECTORY.npz'),trajectory_sha256=sha(dest/'TRAJECTORY.npz'),
                timing=timing,original_timestep_s=old_dt,runtime_timestep_s=float(m.opt.timestep),
                original_integrator=old_integrator,runtime_integrator=int(m.opt.integrator),
                model_provenance_path=str(out/'MODEL_PROVENANCE.json'),
                model_provenance_sha256=sha(out/'MODEL_PROVENANCE.json'),
                no_other_model_field_changes=True,phase='ALL_PREPARATION_ONLY',
                weights='ORIGINAL_UNCHANGED',callbacks_none=True,
                raw_observation_keys=sorted(raw_arrays),pending_aliases=dict(applied='pending_before',decoded='pending_after'),
                internal_work='RK4 internal native stages exceed Euler workload; public-call caps only',
                optimization=False,scalar_scoring=False,behavior_claim='NONE',automatic_resume=False)
            atomic_json(dest/'RESULT.json',record)
            atomic_json(out/'EVALUATIONS.json',[record])
            return record

        result = evaluate(original,'closed20ms')
        if reserved != caps or completed != caps or ledger['active'] is not None:
            raise RuntimeError('final exact call ledger mismatch')
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input identities changed during diagnostic')
        atomic_json(out/'SUMMARY.json',dict(status='NO_BEHAVIOR_CLAIM',
            evaluation=result,ledger=ledger,weight_native=weight_native,
            note='One preparation-only native100us RK4 closed-loop 20ms diagnostic; no score, optimizer, acceptance or deployment.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
