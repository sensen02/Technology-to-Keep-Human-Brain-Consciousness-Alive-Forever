"""S2/S4 native RK4 fixed-command diagnostic; NO_BEHAVIOR_CLAIM.

Explicit derivative of run_early_fixed_command_replay.py (s87).
One fresh replay20ms: s79 actual applied[:4], 4 x 50 native 100us steps.
Only compiled-model mutation: opt.integrator Euler -> mjINT_RK4.
Zero neural calls; 200 public mj_step and 201 explicit mj_forward calls.
RK4 internal stage workload exceeds Euler; public call caps are not work parity.
No custom activation writes, changed muscle/contact parameters, or hidden commands.
Import and --help do not import MuJoCo, inspect callbacks, compile, or simulate.
"""
from pathlib import Path
import argparse
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
S86 = Path('/home/sensen/Desktop/engine修复验收/s86-early-timestep/20261023T202859Z/run')
S87 = Path('/home/sensen/Desktop/engine修复验收/s87-fixed-command-replay/20261023T203521Z/run')
S88 = Path('/home/sensen/Desktop/engine修复验收/s88-three-step-replay/20261023T204113Z/run')
S89 = Path('/home/sensen/Desktop/engine修复验收/s89-activation-integrator/20261023T204717Z')
DERIVED_FROM = ROOT/'run_early_fixed_command_replay.py'
SNAPSHOT_ANCESTOR = ROOT/'run_early_timestep_refinement.py'
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s90-native-rk4/20261023T205300Z/run')
SOURCE_TAG = '3.9.0'
RK4_SOURCE_URL = 'https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_forward.c'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
BODY_DT = .0001
COMMAND_DT = .005
BLOCKS = 4
PHYSICS_PER_BLOCK = 50
PHYSICS_STEPS = 200
SAMPLES = 201
SOURCE_REVIEW = OUTPUT.parent/'SOURCE_REVIEW.json'


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
    # Literal s87/s86 helper body; numpy remains a local import.
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT,
                   help='Fixed fresh output path: '+str(OUTPUT))
    return p


def run(args):
    import numpy as np
    import mujoco as mj

    out = args.output
    if out != OUTPUT or not out.is_absolute() or out.exists():
        raise ValueError('fixed fresh run folder required; parent may exist; no resume')
    out.mkdir(parents=True, exist_ok=False)
    sync_dir(out.parent)
    sync_dir(out.parent.parent)
    lock = (out/'WRITER.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    caps = dict(neural_ticks=0, physics_steps=200, initial_forward=1, diagnostic_forward=200)
    reserved = dict.fromkeys(caps, 0)
    completed = dict.fromkeys(caps, 0)
    ledger = dict(caps=caps, reserved=reserved, completed=completed, active=None,
        total_forward_cap=201,
        scope='public calls only; RK4 internal stage workload exceeds Euler',
        recovery='NO_AUTOMATIC_RESUME; uncommitted reservations remain charged')
    timing = dict(blocks=4, duration_s=.020, body_dt_s=BODY_DT,
        original_body_dt_s=.0001, command_dt_s=COMMAND_DT, samples=SAMPLES,
        neural_ticks_per_block=0, physics_steps_per_block=50,
        schedule='saved s79 actual applied[i] -> hold 50 steps: mj_step then explicit mj_forward',
        intervals_s=[[b*COMMAND_DT,(b+1)*COMMAND_DT] for b in range(BLOCKS)],
        phase='FIXED_COMMAND_REPLAY_ONLY; NOT_AUTONOMOUS; NO_BEHAVIOR_CLAIM')
    atomic_json(out/'LEDGER.json',ledger)
    atomic_json(out/'PREREG.json',dict(evaluations=1,candidates=['replay20ms'],
        status='NO_BEHAVIOR_CLAIM',controller='FIXED_COMMAND_REPLAY_NOT_AUTONOMOUS',
        derived_from=str(DERIVED_FROM),timing=timing,caps=caps,total_forward_cap=201,
        command_source=str(BASELINE),command_key='applied',command_slice=[0,4],
        decoded_used=False,neural_execution=False,automatic_resume=False,
        cold_state='s79 initial q; fresh MjData zero qvel/act/ctrl; sole initial q write; no external force',
        sole_model_change='opt.integrator Euler -> mjINT_RK4; original .0001 timestep unchanged',
        source_tag=SOURCE_TAG,rk4_source_url=RK4_SOURCE_URL,
        internal_work='RK4 integrates native body/activation together and evaluates internal stages; greater workload than Euler',
        reference_s89=str(S89/'ACTIVATIONS.npz'),reference_s88=str(S88/'replay20ms/TRAJECTORY.npz'),
        historical_simulation_calls=0,
        stopping='any nonfinite or MuJoCo warning: save partial shard and stop',
        objective='NONE; no scoring, optimization, acceptance, or autonomous claim'))
    shared_paths = [MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
        ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
        ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
        ROOT/'run_neural_muscle_closure.py', ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    # Neural source identities only: none of these source files is imported/executed.
    paths = shared_paths + [Path(__file__).resolve(),DERIVED_FROM,SNAPSHOT_ANCESTOR,SOURCE_REVIEW,
        BASE_RUN/'INPUT_HASHES.json',BASE_RUN/'SETUP.json',BASE_RUN/'baseline/RESULT.json',BASELINE,
        S86/'MODEL_PROVENANCE.json',S86/'SETUP.json',S86/'fine20ms/TRAJECTORY.npz',
        S86/'INPUT_HASHES.json',S86/'fine20ms/RESULT.json',
        S87/'INPUT_HASHES.json',S87/'MODEL_PROVENANCE.json',
        S88/'INPUT_HASHES.json',S88/'MODEL_PROVENANCE.json',S88/'SETUP.json',
        S88/'replay20ms/TRAJECTORY.npz',S88/'replay20ms/RESULT.json',
        S89/'ACTIVATIONS.npz',S89/'PREREG.json',S89/'SUMMARY.json',
        S89/'engine_forward.c',S89/'engine_support.c',S89/'engine_util_misc.c']

    def reserve(block,delta):
        if ledger['active'] is not None:
            raise RuntimeError('uncommitted reservation; refusing another call')
        for key,value in delta.items():
            if reserved[key]+value > caps[key]:
                raise RuntimeError('hard call budget exceeded')
        for key,value in delta.items():
            reserved[key] += value
        ledger['active'] = dict(evaluation='replay20ms',block=block,delta=delta,
            start_s=max(block,0)*COMMAND_DT,
            end_s=(block+1)*COMMAND_DT if block >= 0 else 0.)
        atomic_json(out/'LEDGER.json',ledger)

    def commit():
        for key,value in ledger['active']['delta'].items():
            completed[key] += value
        ledger['active'] = None
        atomic_json(out/'LEDGER.json',ledger)

    try:
        callbacks_before = callback_snapshot(mj)
        atomic_json(out/'CALLBACKS_BEFORE.json',callbacks_before)
        require_no_callbacks(callbacks_before)
        if mj.__version__ != SOURCE_TAG or mj.mj_versionString() != SOURCE_TAG:
            raise RuntimeError('tag3.9.0 source requires installed/native MuJoCo3.9.0')
        identities = {str(p):sha(p) for p in paths}
        historical = json.loads((BASE_RUN/'INPUT_HASHES.json').read_text())
        old_runner = str(ROOT/'run_grounded_translation_training.py')
        expected = {p:h for p,h in historical.items() if p != old_runner}
        actual = {str(p):identities[str(p)] for p in shared_paths}
        prior_hashes = {name:json.loads((folder/'INPUT_HASHES.json').read_text())
            for name,folder in (('s86',S86),('s87',S87),('s88',S88))}
        if len(actual) != 14 or actual != expected or any(
                saved[p] != h for saved in prior_hashes.values() for p,h in actual.items()):
            raise ValueError('shared14 source/input hashes differ from s79/s86/s87/s88')
        if (prior_hashes['s86'][str(SNAPSHOT_ANCESTOR)] != identities[str(SNAPSHOT_ANCESTOR)]
                or prior_hashes['s87'][str(DERIVED_FROM)] != identities[str(DERIVED_FROM)]
                or prior_hashes['s88'][str(DERIVED_FROM)] != identities[str(DERIVED_FROM)]):
            raise ValueError('fixed derivative source identity differs from historical run')
        atomic_json(out/'INPUT_HASHES.json',identities)
        atomic_json(out/'SHARED_SOURCE_CHECK.json',dict(passed=True,shared=actual,
            excluded_historical_runner=old_runner,hash_only_no_neural_imports=True,
            derived_from=str(DERIVED_FROM),derived_from_sha256=identities[str(DERIVED_FROM)],
            derivative_sha256=identities[str(Path(__file__).resolve())]))
        baseline_result = json.loads((BASE_RUN/'baseline/RESULT.json').read_text())
        fine_result = json.loads((S86/'fine20ms/RESULT.json').read_text())
        quarter_result = json.loads((S88/'replay20ms/RESULT.json').read_text())
        if any(record['trajectory_sha256'] != identities[str(path)] for record,path in (
                (baseline_result,BASELINE),(fine_result,S86/'fine20ms/TRAJECTORY.npz'),
                (quarter_result,S88/'replay20ms/TRAJECTORY.npz'))):
            raise ValueError('historical RESULT/TRAJECTORY identity mismatch')
        old_setup = json.loads((BASE_RUN/'SETUP.json').read_text())
        fine_setup = json.loads((S86/'SETUP.json').read_text())
        quarter_setup = json.loads((S88/'SETUP.json').read_text())
        fine_provenance = json.loads((S86/'MODEL_PROVENANCE.json').read_text())
        replay_provenance = json.loads((S87/'MODEL_PROVENANCE.json').read_text())
        quarter_provenance = json.loads((S88/'MODEL_PROVENANCE.json').read_text())
        for setup,folder in ((fine_setup,S86),(quarter_setup,S88)):
            if (setup['leg_order'] != list(LEGS) or setup['model_provenance_sha256'] !=
                    identities[str(folder/'MODEL_PROVENANCE.json')]):
                raise ValueError('historical leg order/provenance identity mismatch')
        if old_setup['leg_order'] != list(LEGS):
            raise ValueError('s79 leg order differs')
        with np.load(INITIAL,allow_pickle=False) as initial:
            q = initial['new_qpos'].copy()
        with np.load(BASELINE,allow_pickle=False) as saved:
            commands = saved['applied'][:BLOCKS].copy()  # Actual held commands, NEVER decoded.
            weights = saved['weights'].copy()
            if (saved['qpos'].shape != (4001,49) or q.shape != (49,)
                    or not np.array_equal(saved['qpos'][0],q)
                    or any(np.any(saved[k][0]) for k in ('qvel','act','ctrl'))
                    or not np.allclose(saved['time'][:201],np.arange(201)*BODY_DT,rtol=0,atol=1e-12)):
                raise ValueError('s79 cold initial state/clock mismatch')
            for block in range(BLOCKS):
                if not np.array_equal(saved['ctrl'][block*50+1:(block+1)*50+1],
                                      np.broadcast_to(commands[block],(50,90))):
                    raise ValueError('s79 actual applied does not match saved held ctrl')
        if (commands.shape != (4,90) or not np.isfinite(commands).all()
                or np.any((commands < 0)|(commands > 1)) or not np.isfinite(q).all()):
            raise ValueError('invalid actual replay commands/initial q')
        for path in (S86/'fine20ms/TRAJECTORY.npz',S88/'replay20ms/TRAJECTORY.npz'):
            with np.load(path,allow_pickle=False) as saved:
                if (not np.array_equal(saved['qpos'][0],q)
                        or any(np.any(saved[k][0]) for k in ('qvel','act','ctrl'))):
                    raise ValueError('historical cold initial state differs')
                if path.parent.parent == S88 and (not np.array_equal(saved['applied'],commands)
                        or not np.array_equal(saved['commands'],commands)):
                    raise ValueError('s88 fixed actual commands differ')
        atomic_json(out/'REFERENCE_IDENTITIES.json',dict(
            references={str(p):identities[str(p)] for p in paths if p.is_relative_to(S88) or p.is_relative_to(S89)},
            source_tag=SOURCE_TAG,rk4_source_url=RK4_SOURCE_URL,
            historical_body_compile_calls=0,historical_simulation_calls=0,read_only=True))
        atomic_json(out/'FIXED_COMMANDS.json',dict(source=str(BASELINE),sha256=identities[str(BASELINE)],
            key='applied',slice=[0,4],shape=list(commands.shape),commands=commands.tolist(),
            bytes_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),decoded_used=False))
        m = mj.MjModel.from_xml_path(str(MODEL))
        if ((m.nu,m.nq,m.nv) != (90,49,48) or m.opt.timestep != BODY_DT
                or m.opt.integrator != mj.mjtIntegrator.mjINT_EULER
                or np.count_nonzero(m.geom_type == mj.mjtGeom.mjGEOM_MESH) != 71):
            raise ValueError('unexpected original full physical model/dt/Euler integrator')
        if (np.any(m.actuator_delay) or np.any(m.actuator_history) or m.nplugin != 0
                or np.any(m.actuator_plugin != -1)):
            raise RuntimeError('delayed/history/plugin actuator paths forbidden; fixed ctrl must remain native held input')
        atomic_json(out/'NATIVE_INPUT_PATH_CHECK.json',dict(
            actuator_delay=m.actuator_delay.tolist(),actuator_history=m.actuator_history.tolist(),
            actuator_historyadr=m.actuator_historyadr.tolist(),nplugin=int(m.nplugin),
            actuator_plugin=m.actuator_plugin.tolist(),no_delay_history_plugins=True,
            independent_source_review=str(SOURCE_REVIEW),source_review_sha256=identities[str(SOURCE_REVIEW)]))
        before = numeric_model_snapshot(m)
        for provenance in (fine_provenance,replay_provenance,quarter_provenance):
            if before != provenance['before'] or provenance['xml_sha256'] != identities[str(MODEL)]:
                raise RuntimeError('compiled before snapshot differs from original s86/s87/s88 (s79 XML identity)')
        old_dt = float(m.opt.timestep)
        old_integrator = int(m.opt.integrator)
        m.opt.integrator = mj.mjtIntegrator.mjINT_RK4  # ONLY compiled model mutation.
        after = numeric_model_snapshot(m)
        changed = [k for k in before if before[k] != after[k]]
        if (before.keys() != after.keys() or changed != ['opt.integrator']
                or m.opt.timestep != .0001 or m.opt.integrator != mj.mjtIntegrator.mjINT_RK4):
            raise RuntimeError('integrator-only mutation / original native dt check failed')
        atomic_json(out/'MODEL_PROVENANCE.json',dict(xml_path=str(MODEL),xml_sha256=identities[str(MODEL)],
            original_timestep_s=old_dt,runtime_timestep_s=float(m.opt.timestep),
            original_integrator=old_integrator,runtime_integrator=int(m.opt.integrator),
            source_tag=SOURCE_TAG,rk4_source_url=RK4_SOURCE_URL,
            changed_numeric_fields=changed,before=before,after=after,no_other_model_field_changes=True,
            s79_xml_identity_equal=True,s86_before_equal=True,s87_before_equal=True,s88_before_equal=True,
            s86_provenance_sha256=identities[str(S86/'MODEL_PROVENANCE.json')],
            snapshot_helper_source=str(SNAPSHOT_ANCESTOR),snapshot_helper_source_sha256=identities[str(SNAPSHOT_ANCESTOR)],
            compile='same XML compiled once; no mj_setConst, dt/muscle/contact/qact edits'))
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        if any(weight_native != setup['weight_native'] for setup in (old_setup,fine_setup,quarter_setup)):
            raise ValueError('native force reference differs')
        geom_leg = {}
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
        atomic_json(out/'SETUP.json',dict(neurons_executed=0,leg_order=list(LEGS),
            controller='FIXED_COMMAND_REPLAY_NOT_AUTONOMOUS',weight_native=weight_native,
            physical_units='NOT_ESTABLISHED',timing=timing,original_timestep_s=old_dt,
            runtime_timestep_s=float(m.opt.timestep),runtime_integrator=int(m.opt.integrator),
            no_other_model_field_changes=True,callbacks_before=callbacks_before,
            model_provenance_path=str(out/'MODEL_PROVENANCE.json'),
            model_provenance_sha256=sha(out/'MODEL_PROVENANCE.json'),
            dependencies=dict(python=sys.version,numpy=np.__version__,mujoco=mj.__version__,native=mj.mj_versionString()),
            source_hashes=identities))

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
        dest = out/'replay20ms'
        dest.mkdir(exist_ok=False)
        sync_dir(out)
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('input identities changed before evaluation')
        require_no_callbacks(callback_snapshot(mj))
        d = mj.MjData(m)
        d.qpos[:] = q  # Sole pose write; never reset or write act/qvel after initialization.
        if (d.qvel.any() or d.act.any() or d.ctrl.any() or d.time != 0
                or d.xfrc_applied.any() or d.qfrc_applied.any()):
            raise RuntimeError('noncold state or unexpected external force')
        reserve(-1,dict(initial_forward=1))
        mj.mj_forward(m,d)
        initial_row = sample(d)
        atomic_npz(dest/'INITIAL.npz',**initial_row,weights=weights)
        if not all(np.isfinite(initial_row[k]).all() for k in finite_keys) or np.any(initial_row['warnings']):
            atomic_json(dest/'INVALID.json',dict(block=-1,reason='invalid_initial_state'))
            raise RuntimeError('invalid initial state saved; reservation remains charged')
        commit()
        rows,shard_index = [initial_row],[]
        for block in range(BLOCKS):
            reserve(block,dict(physics_steps=50,diagnostic_forward=50))
            applied = commands[block].copy()
            d.ctrl[:] = applied
            new_rows,invalid = [],None
            for sub in range(PHYSICS_PER_BLOCK):
                require_no_callbacks(callback_snapshot(mj))
                if (d.xfrc_applied.any() or d.qfrc_applied.any()
                        or not np.array_equal(d.ctrl,applied)):
                    raise RuntimeError('unexpected external force or hidden command')
                mj.mj_step(m,d)
                mj.mj_forward(m,d)
                row = sample(d)
                new_rows.append(row)
                if (not all(np.isfinite(row[k]).all() for k in finite_keys)
                        or np.any(row['warnings']) or not np.array_equal(row['ctrl'],applied)
                        or d.xfrc_applied.any() or d.qfrc_applied.any()):
                    invalid = dict(block=block,substep=sub,reason='nonfinite_warning_hidden_command_or_force')
                    break
            data = {key:np.array([r[key] for r in new_rows]) for key in initial_row}
            data.update(applied=applied)
            shard = dest/f'block_{block:03d}.npz'
            atomic_npz(shard,**data)
            shard_index.append(dict(block=block,path=str(shard),sha256=sha(shard),
                start_s=block*.005,end_s=(block+1)*.005))
            atomic_json(dest/'SHARDS.json',shard_index)
            body = np.empty(mj.mj_stateSize(m,spec))
            mj.mj_getState(m,d,body,spec)
            atomic_npz(dest/'LATEST_STATE.npz',body_integration=body,commands=commands,
                applied=applied,next_command_index=np.array(block+1),weights=weights)
            atomic_json(dest/'LATEST_STATE_META.json',dict(block=block,
                start_s=block*.005,end_s=(block+1)*.005,
                state_spec='mjSTATE_INTEGRATION',state_size=int(body.size),
                snapshot_sha256=sha(dest/'LATEST_STATE.npz'),shard=str(shard),shard_sha256=sha(shard),
                input_hashes=identities,controller='FIXED_COMMAND_REPLAY_NOT_AUTONOMOUS',
                complete_block=invalid is None,automatic_resume=False))
            rows.extend(new_rows)
            if invalid is not None:
                atomic_json(dest/'INVALID.json',invalid)
                raise RuntimeError('invalid trajectory: partial shard saved; reservation remains charged')
            commit()
        arrays = {key:np.array([r[key] for r in rows]) for key in initial_row}
        arrays.update(applied=commands,commands=commands,weights=weights)
        shapes = dict(qpos=(201,49),qvel=(201,48),act=(201,90),ctrl=(201,90),
            contacts=(201,6),foot_force=(201,6,3),nonfoot_depth=(201,),limit_excess=(201,),
            actuator_force=(201,90),actuator_length=(201,90),actuator_velocity=(201,90),
            applied=(4,90),commands=(4,90))
        if any(arrays[k].shape != shape for k,shape in shapes.items()):
            raise RuntimeError('incomplete physical trajectory/actual actuator records')
        if (arrays['time'].shape != (201,) or
                not np.allclose(arrays['time'],np.arange(201)*BODY_DT,rtol=0,atol=1e-12)
                or np.any(arrays['phase']) or np.any(arrays['warnings'])):
            raise RuntimeError('20ms clock/replay phase/warnings mismatch')
        for block in range(BLOCKS):
            if not np.array_equal(arrays['ctrl'][block*50+1:(block+1)*50+1],
                                  np.broadcast_to(commands[block],(50,90))):
                raise RuntimeError('replay command hold mismatch')
        callbacks_end = callback_snapshot(mj)
        atomic_json(out/'CALLBACKS_END.json',callbacks_end)
        require_no_callbacks(callbacks_end)
        if callbacks_before != callbacks_end:
            raise RuntimeError('callback identities changed during replay')
        if numeric_model_snapshot(m) != after or m.opt.timestep != .0001:
            raise RuntimeError('runtime model changed after integrator-only setup')
        atomic_npz(dest/'TRAJECTORY.npz',**arrays)
        record = dict(name='replay20ms',status='NO_BEHAVIOR_CLAIM',
            controller='FIXED_COMMAND_REPLAY_NOT_AUTONOMOUS',
            trajectory_path=str(dest/'TRAJECTORY.npz'),trajectory_sha256=sha(dest/'TRAJECTORY.npz'),
            timing=timing,original_timestep_s=old_dt,runtime_timestep_s=float(m.opt.timestep),
            original_integrator=old_integrator,runtime_integrator=int(m.opt.integrator),
            model_provenance_path=str(out/'MODEL_PROVENANCE.json'),
            model_provenance_sha256=sha(out/'MODEL_PROVENANCE.json'),
            no_other_model_field_changes=True,command_source=str(BASELINE),
            command_key='applied',command_slice=[0,4],weights='BASELINE_METADATA_ONLY',
            callbacks_none=True,callbacks_before=callbacks_before,callbacks_end=callbacks_end,
            source_tag=SOURCE_TAG,internal_work='RK4 internal stage workload exceeds Euler; public-call budget only',
            neural_execution=False,optimization=False,behavior_claim='NONE',automatic_resume=False)
        atomic_json(dest/'RESULT.json',record)
        atomic_json(out/'EVALUATIONS.json',[record])
        if reserved != caps or completed != caps or ledger['active'] is not None:
            raise RuntimeError('final exact call ledger mismatch')
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input identities changed during replay')
        atomic_json(out/'SUMMARY.json',dict(status='NO_BEHAVIOR_CLAIM',evaluation=record,
            ledger=ledger,weight_native=weight_native,
            note='One native100us RK4 20ms fixed-command diagnostic; not autonomous, no neural execution or behavior claim.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
