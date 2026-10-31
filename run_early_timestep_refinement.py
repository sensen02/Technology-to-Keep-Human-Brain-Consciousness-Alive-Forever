"""S2/S4 preparation-only 20ms timestep diagnostic; NO_BEHAVIOR_CLAIM.

Explicit derivative of run_formal_translation_training.py: one original-weight
cold s79 evaluation, four 5ms command blocks. Only the compiled model's scalar
opt.timestep changes (100us -> 50us). No optimization, formal-window scoring,
extra evaluations, automatic resume, or calls on import/--help.
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
DERIVED_FROM = ROOT/'run_formal_translation_training.py'
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s86-early-timestep/20261023T202859Z/run')
BODY_DT = .00005
NEURAL_DT = .0005
COMMAND_DT = .005
BLOCKS = 4
PHYSICS_PER_BLOCK = 100


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
    caps = dict(neural_ticks=40, physics_steps=400, initial_forward=1, diagnostic_forward=400)
    reserved = dict.fromkeys(caps, 0)
    completed = dict.fromkeys(caps, 0)
    ledger = dict(caps=caps, reserved=reserved, completed=completed, active=None,
                  total_forward_cap=401,
                  recovery='NO_AUTOMATIC_RESUME; uncommitted reservations remain charged')
    timing = dict(blocks=BLOCKS, duration_s=.020, body_dt_s=BODY_DT,
        original_body_dt_s=.0001, neural_dt_s=NEURAL_DT, command_dt_s=COMMAND_DT,
        neural_ticks_per_block=10, physics_steps_per_block=PHYSICS_PER_BLOCK,
        schedule='observe encode -> 10 neural ticks .0005 -> old pending 100 physics steps .00005 -> consume -> next pending',
        intervals_s=[[b*COMMAND_DT,(b+1)*COMMAND_DT] for b in range(BLOCKS)],
        phase='ALL_PREPARATION_ONLY; no formal phase; no 400ms run',
        sensory_encoding='same stateless original encoder/config; no body-dt rescaling',
        FIR='unchanged DT=.0005 and bins ((0,5),(5,10),(10,20),(20,40),(40,80))')
    atomic_json(out/'LEDGER.json', ledger)
    atomic_json(out/'PREREG.json', dict(evaluations=1, candidates=['fine20ms'],
        status='NO_BEHAVIOR_CLAIM', derived_from=str(DERIVED_FROM), timing=timing,
        baseline_path=str(BASELINE), baseline_usage='read-only first 201 samples [0,20]ms; comparison later; zero rerun calls',
        objective='NONE; no optimization or score_formal; no acceptance decision',
        initial_weights='original frozen 66x5 FIR; all 330 coefficients unchanged',
        cold_state='s79 q; zero qvel/act/ctrl/pending, fresh network and FIR history',
        controller='binary MN spikes -> fixed FIR -> muscle; sole initial q write',
        sole_model_change='compile same XML, assert old .0001, set opt.timestep=.00005, assert all other numeric model/option fields unchanged',
        actuator_records='actual actuator_force/actuator_length/actuator_velocity 401x90',
        stopping='any nonfinite or MuJoCo warning: save partial shard and stop',
        caps=caps, total_forward_cap=401, automatic_resume=False))
    shared_paths = [MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
        ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
        ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
        ROOT/'run_neural_muscle_closure.py', ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    paths = shared_paths + [Path(__file__).resolve(), DERIVED_FROM, BASE_RUN/'INPUT_HASHES.json',
                            BASE_RUN/'SETUP.json', BASE_RUN/'baseline/RESULT.json', BASELINE]

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

    def numeric_model_snapshot(model):
        # Read every exposed numeric model/option field; no second compilation.
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

    try:
        identities = {str(p):sha(p) for p in paths}
        historical = json.loads((BASE_RUN/'INPUT_HASHES.json').read_text())
        old_runner = str(ROOT/'run_grounded_translation_training.py')
        expected = {p:h for p,h in historical.items() if p != old_runner}
        actual = {str(p):identities[str(p)] for p in shared_paths}
        if len(actual) != 14 or actual != expected:
            raise ValueError('shared14 s79 source/input hashes differ; refusing all calls')
        atomic_json(out/'INPUT_HASHES.json', identities)
        atomic_json(out/'SHARED_SOURCE_CHECK.json', dict(passed=True, shared=actual,
            excluded_historical_runner=old_runner, derived_from=str(DERIVED_FROM),
            derivative_sha256=identities[str(Path(__file__).resolve())]))
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
        if baseline_result['trajectory_sha256'] != identities[str(BASELINE)]:
            raise ValueError('saved baseline trajectory hash differs from historical RESULT')
        with np.load(BASELINE,allow_pickle=False) as saved:
            if (saved['qpos'].shape != (4001,49) or not np.array_equal(saved['weights'],original)
                    or not np.array_equal(saved['qpos'][0],q)
                    or any(np.any(saved[k][0]) for k in ('qvel','act','ctrl'))
                    or not np.allclose(saved['time'][:201],np.arange(201)*.0001,rtol=0,atol=1e-12)):
                raise ValueError('saved baseline cold initial state/clock/weights mismatch')
        atomic_json(out/'BASELINE_IDENTITY.json', dict(path=str(BASELINE),
            sha256=identities[str(BASELINE)], comparison_sample_slice=[0,201],
            comparison_time_s=[0.,.020], read_only=True, simulation_calls=0,
            score='NOT_SCORED; later independent first201-sample comparison',
            historical_actuator_records='MISSING; never fabricated or reconstructed'))
        m = mj.MjModel.from_xml_path(str(MODEL))
        if (m.nu,m.nq,m.nv) != (90,49,48) or m.opt.timestep != .0001:
            raise ValueError('unexpected original physical model or timestep')
        if np.count_nonzero(m.geom_type == mj.mjtGeom.mjGEOM_MESH) != 71:
            raise ValueError('complete body required')
        before = numeric_model_snapshot(m)
        old_dt = float(m.opt.timestep)
        m.opt.timestep = BODY_DT  # The ONLY model-field assignment.
        after = numeric_model_snapshot(m)
        changed = [k for k in before if before[k] != after[k]]
        if before.keys() != after.keys() or changed != ['opt.timestep'] or m.opt.timestep != BODY_DT:
            raise RuntimeError('model timestep-only change check failed')
        model_provenance = dict(xml_path=str(MODEL),xml_sha256=identities[str(MODEL)],
            original_timestep_s=old_dt,runtime_timestep_s=float(m.opt.timestep),
            changed_numeric_fields=changed, before=before, after=after,
            no_other_model_field_changes=True, compile='same XML compiled once; no mj_setConst or model edits')
        atomic_json(out/'MODEL_PROVENANCE.json',model_provenance)
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        old_setup = json.loads((BASE_RUN/'SETUP.json').read_text())
        if weight_native != old_setup['weight_native']:
            raise ValueError('native force reference differs from s79')
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
            model_provenance_path=str(out/'MODEL_PROVENANCE.json'),
            model_provenance_sha256=sha(out/'MODEL_PROVENANCE.json'),
            no_other_model_field_changes=True,
            mapping_digest=FiniteMotorTranslator(manifest,original).mapping_digest,
            neural_parameters=dataclasses.asdict(ConductanceNetwork(len(roots),dt_ms=.5).p),
            dependencies=dict(python=sys.version,numpy=np.__version__,mujoco=mj.__version__),
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

        finite_keys = ('qpos','qvel','act','ctrl','foot_force',
                       'actuator_force','actuator_length','actuator_velocity')

        def evaluate(weights,name):
            if identities != {str(p):sha(p) for p in paths}:
                raise RuntimeError('source/input identities changed before evaluation')
            if not np.array_equal(weights,original) or name != 'fine20ms':
                raise ValueError('only original frozen weights / fine20ms permitted')
            dest = out/name
            dest.mkdir(exist_ok=False)
            sync_dir(out)
            net = ConductanceNetwork(len(roots),dt_ms=.5)
            net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest,weights)
            if dec.DT != NEURAL_DT or dec.BINS != ((0,5),(5,10),(10,20),(20,40),(40,80)):
                raise RuntimeError('translator timing changed')
            enc = ChannelSensoryEncoder(groups,len(roots))
            d = mj.MjData(m)
            d.qpos[:] = q  # Sole pose write; no phase-boundary writes or resets.
            pending = np.zeros(90)
            if (d.qvel.any() or d.act.any() or d.ctrl.any() or
                    d.xfrc_applied.any() or d.qfrc_applied.any() or dec.history.any()):
                raise RuntimeError('noncold initial state or unexpected force')
            reserve(name,-1,dict(initial_forward=1))
            mj.mj_forward(m,d)
            initial_row = sample(d)
            atomic_npz(dest/'INITIAL.npz',**initial_row,weights=weights)
            if not all(np.isfinite(initial_row[k]).all() for k in finite_keys) or np.any(initial_row['warnings']):
                atomic_json(dest/'INVALID.json',dict(block=-1,reason='invalid_initial_state'))
                raise RuntimeError('invalid initial state saved; reservation remains charged')
            commit()
            rows,block_data,shard_index = [initial_row],[],[]
            for block in range(BLOCKS):
                reserve(name,block,dict(neural_ticks=10,physics_steps=100,diagnostic_forward=100))
                current,_ = enc.encode(observe_quantities(m,d,sensory_ref,joints,geom_leg))
                spikes = np.array([net.step(current_nA=current).copy() for _ in range(10)],dtype=bool)
                applied = pending.copy()
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
                pending,info = dec.consume(spikes,spike_dt_s=.0005,
                    start_s=block*.005,end_s=(block+1)*.005)
                data = {key:np.array([r[key] for r in new_rows]) for key in initial_row}
                data.update(current=current,spikes=spikes,applied=applied,decoded=pending,
                            features=np.asarray(info['pool_features']))
                shard = dest/f'block_{block:03d}.npz'
                atomic_npz(shard,**data)
                shard_index.append(dict(block=block,path=str(shard),sha256=sha(shard),
                    start_s=block*.005,end_s=(block+1)*.005))
                atomic_json(dest/'SHARDS.json',shard_index)
                body = np.empty(mj.mj_stateSize(m,spec))
                mj.mj_getState(m,d,body,spec)
                snapshot = {key:value for key,value in vars(net).items() if isinstance(value,np.ndarray)}
                atomic_npz(dest/'LATEST_STATE.npz',body_integration=body,FIR_history=dec.history,
                           pending=pending,weights=weights,**snapshot)
                scalars = {key:value for key,value in vars(net).items() if isinstance(value,(int,float,str))}
                atomic_json(dest/'LATEST_STATE_META.json',dict(block=block,neural_scalars=scalars,
                    start_s=block*.005,end_s=(block+1)*.005,
                    FIR_elapsed_s=dec.elapsed_s,FIR_interval=dec.interval,
                    snapshot_sha256=sha(dest/'LATEST_STATE.npz'),shard=str(shard),shard_sha256=sha(shard),
                    input_hashes=identities,encoder='stateless physical encode; unchanged config/reference',
                    complete_block=invalid is None,automatic_resume=False))
                rows.extend(new_rows)
                block_data.append(data)
                if invalid is not None:
                    atomic_json(dest/'INVALID.json',invalid)
                    raise RuntimeError('invalid trajectory: partial shard saved, reservation conservatively charged')
                commit()
            arrays = {key:np.array([r[key] for r in rows]) for key in initial_row}
            arrays.update(spikes=np.array([b['spikes'] for b in block_data],dtype=bool),
                current=np.array([b['current'] for b in block_data]),
                applied=np.array([b['applied'] for b in block_data]),
                decoded=np.array([b['decoded'] for b in block_data]),
                features=np.array([b['features'] for b in block_data]),weights=weights)
            shapes = dict(qpos=(401,49),qvel=(401,48),act=(401,90),ctrl=(401,90),
                actuator_force=(401,90),actuator_length=(401,90),actuator_velocity=(401,90),
                current=(4,17099),spikes=(4,10,17099),features=(4,66,5),applied=(4,90),decoded=(4,90))
            if any(arrays[k].shape != shape for k,shape in shapes.items()):
                raise RuntimeError('incomplete trajectory/neural/actual actuator records')
            if (arrays['time'].shape != (401,) or
                    not np.allclose(arrays['time'],np.arange(401)*.00005,rtol=0,atol=1e-12)
                    or np.any(arrays['phase']) or not np.array_equal(arrays['weights'],original)):
                raise RuntimeError('20ms clock/preparation/original weights mismatch')
            if numeric_model_snapshot(m) != after:
                raise RuntimeError('runtime model changed after timestep-only setup')
            atomic_npz(dest/'TRAJECTORY.npz',**arrays)
            record = dict(name=name,status='NO_BEHAVIOR_CLAIM',
                trajectory_path=str(dest/'TRAJECTORY.npz'),trajectory_sha256=sha(dest/'TRAJECTORY.npz'),
                timing=timing,original_timestep_s=old_dt,runtime_timestep_s=float(m.opt.timestep),
                model_provenance_path=str(out/'MODEL_PROVENANCE.json'),
                model_provenance_sha256=sha(out/'MODEL_PROVENANCE.json'),
                no_other_model_field_changes=True,phase='ALL_PREPARATION_ONLY',
                weights='ORIGINAL_UNCHANGED',comparison='PENDING_READ_ONLY_FIRST201_BASELINE_COMPARISON',
                optimization=False,behavior_claim='NONE',automatic_resume=False)
            atomic_json(dest/'RESULT.json',record)
            atomic_json(out/'EVALUATIONS.json',[record])
            return record

        result = evaluate(original,'fine20ms')
        if reserved != caps or completed != caps or ledger['active'] is not None:
            raise RuntimeError('final exact call ledger mismatch')
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input identities changed during diagnostic')
        atomic_json(out/'SUMMARY.json',dict(status='NO_BEHAVIOR_CLAIM',
            evaluation=result,ledger=ledger,weight_native=weight_native,
            note='One preparation-only 20ms timestep diagnostic; no formal score, optimizer, acceptance, or deployment.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
