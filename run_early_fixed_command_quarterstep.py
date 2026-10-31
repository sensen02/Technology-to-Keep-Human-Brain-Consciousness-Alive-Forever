"""S2/S4 quarter-step fixed-command body replay diagnostic; NO_BEHAVIOR_CLAIM.

Explicit derivative of run_early_fixed_command_replay.py, NOT autonomous evaluation.
Read original s79 actual applied[:4], hold each for 200 physical 25us steps.
Exactly one 20ms evaluation: 800 steps, 801 forwards, zero neural calls.
Saved s87 50us trajectory is an identity-bound read-only input, never rerun.
No neural network, encoder, translator, decode, optimization, or automatic resume.
Import and --help do not compile a model or execute any simulation calls.
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
S87_TRAJECTORY = S87/'replay20ms/TRAJECTORY.npz'
DERIVED_FROM = ROOT/'run_early_fixed_command_replay.py'
SNAPSHOT_ANCESTOR = ROOT/'run_early_timestep_refinement.py'
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s88-three-step-replay/20261023T204113Z/run')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
BODY_DT = .000025
COMMAND_DT = .005
BLOCKS = 4
PHYSICS_PER_BLOCK = 200
PHYSICS_STEPS = BLOCKS * PHYSICS_PER_BLOCK
SAMPLES = PHYSICS_STEPS + 1


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
    # Literal s87/s86 helper body; no import-time runtime work.
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
    caps = dict(neural_ticks=0, physics_steps=PHYSICS_STEPS,
                initial_forward=1, diagnostic_forward=PHYSICS_STEPS)
    reserved = dict.fromkeys(caps, 0)
    completed = dict.fromkeys(caps, 0)
    ledger = dict(caps=caps, reserved=reserved, completed=completed, active=None,
        total_forward_cap=SAMPLES,
        recovery='NO_AUTOMATIC_RESUME; uncommitted reservations remain charged')
    timing = dict(blocks=BLOCKS, duration_s=.020, body_dt_s=BODY_DT,
        original_body_dt_s=.0001, command_dt_s=COMMAND_DT,
        neural_ticks_per_block=0, physics_steps_per_block=PHYSICS_PER_BLOCK,
        samples=SAMPLES,
        schedule='saved s79 actual applied[i] -> hold 200 steps: mj_step then mj_forward',
        intervals_s=[[b*COMMAND_DT,(b+1)*COMMAND_DT] for b in range(BLOCKS)],
        phase='FIXED_COMMAND_REPLAY_ONLY; NOT_AUTONOMOUS; NO_BEHAVIOR_CLAIM')
    atomic_json(out/'LEDGER.json',ledger)
    atomic_json(out/'PREREG.json',dict(evaluations=1,candidates=['replay20ms'],
        status='NO_BEHAVIOR_CLAIM',controller='FIXED_COMMAND_REPLAY_NOT_AUTONOMOUS',
        derived_from=str(DERIVED_FROM),timing=timing,caps=caps,total_forward_cap=SAMPLES,
        command_source=str(BASELINE),command_key='applied',command_slice=[0,4],
        saved_s87_trajectory=str(S87_TRAJECTORY),historical_simulation_calls=0,
        decoded_used=False,neural_execution=False,automatic_resume=False,
        cold_state='s79 initial q; fresh MjData zero qvel/act/ctrl; sole initial q write',
        sole_model_change='runtime opt.timestep .0001 -> .000025; before equals s87/s86; after equals s87 except opt.timestep',
        stopping='any nonfinite or MuJoCo warning: save partial shard and stop',
        objective='NONE; no scoring, optimization, acceptance, or autonomous claim'))
    shared_paths = [MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
        ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
        ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
        ROOT/'run_neural_muscle_closure.py', ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    # These source files are hashed only, never imported/executed.
    paths = shared_paths + [Path(__file__).resolve(),DERIVED_FROM,SNAPSHOT_ANCESTOR,
        BASE_RUN/'INPUT_HASHES.json',BASE_RUN/'SETUP.json',BASE_RUN/'baseline/RESULT.json',BASELINE,
        S86/'MODEL_PROVENANCE.json',S86/'SETUP.json',S86/'fine20ms/TRAJECTORY.npz',
        S86/'INPUT_HASHES.json',S86/'fine20ms/RESULT.json',
        S87/'MODEL_PROVENANCE.json',S87/'SETUP.json',S87/'INPUT_HASHES.json',
        S87/'replay20ms/RESULT.json',S87_TRAJECTORY]

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
        if (PHYSICS_STEPS != 800 or SAMPLES != 801 or
                PHYSICS_PER_BLOCK * BODY_DT != COMMAND_DT or BLOCKS * COMMAND_DT != .020):
            raise RuntimeError('fixed quarter-step schedule mismatch')
        identities = {str(p):sha(p) for p in paths}
        historical = json.loads((BASE_RUN/'INPUT_HASHES.json').read_text())
        old_runner = str(ROOT/'run_grounded_translation_training.py')
        expected = {p:h for p,h in historical.items() if p != old_runner}
        actual = {str(p):identities[str(p)] for p in shared_paths}
        s86_hashes = json.loads((S86/'INPUT_HASHES.json').read_text())
        s87_hashes = json.loads((S87/'INPUT_HASHES.json').read_text())
        if (len(actual) != 14 or actual != expected or
                any(s86_hashes[p] != h or s87_hashes[p] != h for p,h in actual.items())):
            raise ValueError('shared14 source/input hashes differ from s79/s86/s87')
        if (s86_hashes[str(SNAPSHOT_ANCESTOR)] != identities[str(SNAPSHOT_ANCESTOR)] or
                s87_hashes[str(SNAPSHOT_ANCESTOR)] != identities[str(SNAPSHOT_ANCESTOR)] or
                s87_hashes[str(DERIVED_FROM)] != identities[str(DERIVED_FROM)]):
            raise ValueError('s86/s87 derivative source identity differs')
        if any(identities[p] != h for p,h in s87_hashes.items() if p in identities):
            raise ValueError('s87 shared historical input identity differs')
        atomic_json(out/'INPUT_HASHES.json',identities)
        atomic_json(out/'SHARED_SOURCE_CHECK.json',dict(passed=True,shared=actual,
            excluded_historical_runner=old_runner,hash_only_no_neural_imports=True,
            derived_from=str(DERIVED_FROM),derived_from_sha256=identities[str(DERIVED_FROM)],
            derivative_sha256=identities[str(Path(__file__).resolve())]))
        baseline_result = json.loads((BASE_RUN/'baseline/RESULT.json').read_text())
        if baseline_result['trajectory_sha256'] != identities[str(BASELINE)]:
            raise ValueError('baseline RESULT/TRAJECTORY hash mismatch')
        old_setup = json.loads((BASE_RUN/'SETUP.json').read_text())
        fine_setup = json.loads((S86/'SETUP.json').read_text())
        fine_provenance = json.loads((S86/'MODEL_PROVENANCE.json').read_text())
        fine_result = json.loads((S86/'fine20ms/RESULT.json').read_text())
        replay_setup = json.loads((S87/'SETUP.json').read_text())
        replay_provenance = json.loads((S87/'MODEL_PROVENANCE.json').read_text())
        replay_result = json.loads((S87/'replay20ms/RESULT.json').read_text())
        if (old_setup['leg_order'] != list(LEGS) or fine_setup['leg_order'] != list(LEGS)
                or replay_setup['leg_order'] != list(LEGS)
                or fine_setup['model_provenance_sha256'] != identities[str(S86/'MODEL_PROVENANCE.json')]
                or fine_result['trajectory_sha256'] != identities[str(S86/'fine20ms/TRAJECTORY.npz')]
                or replay_setup['model_provenance_sha256'] != identities[str(S87/'MODEL_PROVENANCE.json')]
                or replay_result['model_provenance_sha256'] != identities[str(S87/'MODEL_PROVENANCE.json')]
                or replay_result['trajectory_sha256'] != identities[str(S87_TRAJECTORY)]):
            raise ValueError('s79/s86/s87 leg order or provenance/trajectory identity mismatch')
        with np.load(INITIAL,allow_pickle=False) as initial:
            q = initial['new_qpos'].copy()
        with np.load(BASELINE,allow_pickle=False) as saved:
            commands = saved['applied'][:BLOCKS].copy()  # Actual held commands, NEVER decoded.
            weights = saved['weights'].copy()
            if (saved['qpos'].shape != (4001,49) or q.shape != (49,)
                    or not np.array_equal(saved['qpos'][0],q)
                    or any(np.any(saved[k][0]) for k in ('qvel','act','ctrl'))
                    or not np.allclose(saved['time'][:201],np.arange(201)*.0001,rtol=0,atol=1e-12)):
                raise ValueError('s79 cold initial state/clock mismatch')
            for block in range(BLOCKS):
                if not np.array_equal(saved['ctrl'][block*50+1:(block+1)*50+1],
                                      np.broadcast_to(commands[block],(50,90))):
                    raise ValueError('s79 saved applied does not match actual held ctrl')
        if (commands.shape != (4,90) or not np.isfinite(commands).all()
                or np.any((commands < 0)|(commands > 1)) or not np.isfinite(q).all()):
            raise ValueError('invalid actual replay commands/initial q')
        with np.load(S86/'fine20ms/TRAJECTORY.npz',allow_pickle=False) as fine:
            if (not np.array_equal(fine['qpos'][0],q)
                    or any(np.any(fine[k][0]) for k in ('qvel','act','ctrl'))):
                raise ValueError('s86 cold initial state differs')
        with np.load(S87_TRAJECTORY,allow_pickle=False) as replay:
            if (replay['qpos'].shape != (401,49) or replay['time'].shape != (401,)
                    or not np.array_equal(replay['qpos'][0],q)
                    or any(np.any(replay[k][0]) for k in ('qvel','act','ctrl'))
                    or not np.array_equal(replay['applied'],commands)
                    or not np.array_equal(replay['weights'],weights)
                    or not np.allclose(replay['time'],np.arange(401)*.00005,rtol=0,atol=1e-12)):
                raise ValueError('saved s87 cold state/clock/actual commands/weights differ')
            for block in range(BLOCKS):
                if not np.array_equal(replay['ctrl'][block*100+1:(block+1)*100+1],
                                      np.broadcast_to(commands[block],(100,90))):
                    raise ValueError('saved s87 actual held ctrl differs from s79 commands')
            if any(replay[k].shape != (401,90) for k in
                   ('actuator_force','actuator_length','actuator_velocity')):
                raise ValueError('saved s87 actual actuator records missing/incomplete')
        atomic_json(out/'BASELINE_IDENTITY.json',dict(path=str(BASELINE),sha256=identities[str(BASELINE)],
            command_key='applied',command_slice=[0,4],read_only=True,simulation_calls=0,
            historical_actuator_records='MISSING; not fabricated or reconstructed'))
        atomic_json(out/'S87_IDENTITY.json',dict(path=str(S87_TRAJECTORY),
            sha256=identities[str(S87_TRAJECTORY)],read_only=True,simulation_calls=0,
            body_dt_s=.00005,samples=401,
            provenance_path=str(S87/'MODEL_PROVENANCE.json'),
            provenance_sha256=identities[str(S87/'MODEL_PROVENANCE.json')],
            input_hashes_path=str(S87/'INPUT_HASHES.json'),
            input_hashes_sha256=identities[str(S87/'INPUT_HASHES.json')]))
        m = mj.MjModel.from_xml_path(str(MODEL))
        if ((m.nu,m.nq,m.nv) != (90,49,48) or m.opt.timestep != .0001
                or np.count_nonzero(m.geom_type == mj.mjtGeom.mjGEOM_MESH) != 71):
            raise ValueError('unexpected original complete physical model/timestep')
        before = numeric_model_snapshot(m)
        old_dt = float(m.opt.timestep)
        m.opt.timestep = BODY_DT  # ONLY model-field assignment.
        after = numeric_model_snapshot(m)
        changed = [k for k in before if before[k] != after[k]]
        s87_after = replay_provenance['after']
        s87_after_differences = [k for k in after if after[k] != s87_after.get(k)]
        if (before.keys() != after.keys() or changed != ['opt.timestep']
                or m.opt.timestep != BODY_DT or before != fine_provenance['before']
                or before != replay_provenance['before']
                or after.keys() != s87_after.keys()
                or s87_after_differences != ['opt.timestep']
                or s87_after['opt.timestep'] != .00005
                or fine_provenance['xml_sha256'] != identities[str(MODEL)]
                or replay_provenance['xml_sha256'] != identities[str(MODEL)]):
            raise RuntimeError('timestep-only mutation / s86/s87 compiled numeric model check failed')
        atomic_json(out/'MODEL_PROVENANCE.json',dict(xml_path=str(MODEL),xml_sha256=identities[str(MODEL)],
            original_timestep_s=old_dt,runtime_timestep_s=float(m.opt.timestep),
            changed_numeric_fields=changed,before=before,after=after,no_other_model_field_changes=True,
            s86_before_equal=True,s87_before_equal=True,s87_after_equal_except_timestep=True,
            s87_after_differing_numeric_fields=s87_after_differences,
            s86_provenance_sha256=identities[str(S86/'MODEL_PROVENANCE.json')],
            s87_provenance_sha256=identities[str(S87/'MODEL_PROVENANCE.json')],
            snapshot_helper_source=str(DERIVED_FROM),snapshot_helper_source_sha256=identities[str(DERIVED_FROM)],
            compile='same XML compiled once; no mj_setConst or other model edits'))
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        if (weight_native != old_setup['weight_native'] or weight_native != fine_setup['weight_native']
                or weight_native != replay_setup['weight_native']):
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
            runtime_timestep_s=float(m.opt.timestep),no_other_model_field_changes=True,
            model_provenance_path=str(out/'MODEL_PROVENANCE.json'),
            model_provenance_sha256=sha(out/'MODEL_PROVENANCE.json'),
            dependencies=dict(python=sys.version,numpy=np.__version__,mujoco=mj.__version__),source_hashes=identities))

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
        d = mj.MjData(m)
        d.qpos[:] = q  # Sole pose write; no phase-boundary reset, act fill, or overwrite.
        if (d.qvel.any() or d.act.any() or d.ctrl.any() or d.time != 0
                or d.xfrc_applied.any() or d.qfrc_applied.any()):
            raise RuntimeError('noncold state or unexpected force')
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
            reserve(block,dict(physics_steps=PHYSICS_PER_BLOCK,diagnostic_forward=PHYSICS_PER_BLOCK))
            applied = commands[block].copy()
            d.ctrl[:] = applied
            new_rows,invalid = [],None
            for sub in range(PHYSICS_PER_BLOCK):
                if d.xfrc_applied.any() or d.qfrc_applied.any():
                    raise RuntimeError('unexpected external force')
                mj.mj_step(m,d)
                mj.mj_forward(m,d)
                row = sample(d)
                new_rows.append(row)
                if not all(np.isfinite(row[k]).all() for k in finite_keys) or np.any(row['warnings']):
                    invalid = dict(block=block,substep=sub,reason='nonfinite_or_warning')
                    break
            data = {key:np.array([r[key] for r in new_rows]) for key in initial_row}
            data.update(applied=applied)
            shard = dest/f'block_{block:03d}.npz'
            atomic_npz(shard,**data)
            shard_index.append(dict(block=block,path=str(shard),sha256=sha(shard),
                start_s=block*COMMAND_DT,end_s=(block+1)*COMMAND_DT))
            atomic_json(dest/'SHARDS.json',shard_index)
            body = np.empty(mj.mj_stateSize(m,spec))
            mj.mj_getState(m,d,body,spec)
            atomic_npz(dest/'LATEST_STATE.npz',body_integration=body,commands=commands,
                applied=applied,next_command_index=np.array(block+1),weights=weights)
            atomic_json(dest/'LATEST_STATE_META.json',dict(block=block,
                start_s=block*COMMAND_DT,end_s=(block+1)*COMMAND_DT,
                state_spec='mjSTATE_INTEGRATION',state_size=int(body.size),
                snapshot_sha256=sha(dest/'LATEST_STATE.npz'),shard=str(shard),shard_sha256=sha(shard),
                input_hashes=identities,controller='FIXED_COMMAND_REPLAY_NOT_AUTONOMOUS',
                complete_block=invalid is None,automatic_resume=False))
            rows.extend(new_rows)
            if invalid is not None:
                atomic_json(dest/'INVALID.json',invalid)
                raise RuntimeError('invalid trajectory: partial shard saved, reservation remains charged')
            commit()
        arrays = {key:np.array([r[key] for r in rows]) for key in initial_row}
        arrays.update(applied=commands,commands=commands,weights=weights)
        shapes = dict(qpos=(SAMPLES,49),qvel=(SAMPLES,48),act=(SAMPLES,90),ctrl=(SAMPLES,90),
            contacts=(SAMPLES,6),foot_force=(SAMPLES,6,3),nonfoot_depth=(SAMPLES,),limit_excess=(SAMPLES,),
            actuator_force=(SAMPLES,90),actuator_length=(SAMPLES,90),actuator_velocity=(SAMPLES,90),
            applied=(4,90),commands=(4,90))
        if any(arrays[k].shape != shape for k,shape in shapes.items()):
            raise RuntimeError('incomplete physical trajectory/actual actuator records')
        if (arrays['time'].shape != (SAMPLES,) or
                not np.allclose(arrays['time'],np.arange(SAMPLES)*BODY_DT,rtol=0,atol=1e-12)
                or np.any(arrays['phase']) or np.any(arrays['warnings'])):
            raise RuntimeError('20ms clock/replay phase/warnings mismatch')
        for block in range(BLOCKS):
            if not np.array_equal(arrays['ctrl'][block*PHYSICS_PER_BLOCK+1:(block+1)*PHYSICS_PER_BLOCK+1],
                                  np.broadcast_to(commands[block],(PHYSICS_PER_BLOCK,90))):
                raise RuntimeError('replay command hold mismatch')
        if numeric_model_snapshot(m) != after:
            raise RuntimeError('runtime model changed after timestep-only setup')
        atomic_npz(dest/'TRAJECTORY.npz',**arrays)
        record = dict(name='replay20ms',status='NO_BEHAVIOR_CLAIM',
            controller='FIXED_COMMAND_REPLAY_NOT_AUTONOMOUS',
            trajectory_path=str(dest/'TRAJECTORY.npz'),trajectory_sha256=sha(dest/'TRAJECTORY.npz'),
            timing=timing,original_timestep_s=old_dt,runtime_timestep_s=float(m.opt.timestep),
            model_provenance_path=str(out/'MODEL_PROVENANCE.json'),
            model_provenance_sha256=sha(out/'MODEL_PROVENANCE.json'),
            no_other_model_field_changes=True,command_source=str(BASELINE),
            command_key='applied',command_slice=[0,4],weights='BASELINE_METADATA_ONLY',
            saved_s87_trajectory=str(S87_TRAJECTORY),saved_s87_trajectory_sha256=identities[str(S87_TRAJECTORY)],
            historical_simulation_calls=0,
            neural_execution=False,optimization=False,behavior_claim='NONE',automatic_resume=False)
        atomic_json(dest/'RESULT.json',record)
        atomic_json(out/'EVALUATIONS.json',[record])
        if reserved != caps or completed != caps or ledger['active'] is not None:
            raise RuntimeError('final exact call ledger mismatch')
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input identities changed during replay')
        atomic_json(out/'SUMMARY.json',dict(status='NO_BEHAVIOR_CLAIM',evaluation=record,
            ledger=ledger,weight_native=weight_native,
            note='One 20ms fixed-command body diagnostic; not autonomous, no neural execution or behavior claim.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
