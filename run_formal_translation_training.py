"""Bounded S2/S4 formal-window-only FIR training; research, not walking.

Re-score the saved s79 baseline, never rerun it. Exactly three fresh continuous
400ms cold candidates (plus, minus, updated), one direction seed 8201. No neural
or physics calls on import or --help. Interrupted reservations stay charged.
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
    with path.with_suffix(path.suffix+'.tmp').open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path.with_suffix(path.suffix+'.tmp'), path)
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
    p.add_argument('--output', type=Path, required=True,
                   help='Fresh absolute output directory; never reused')
    return p


def run(args):
    import numpy as np
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.embodied.grounded_training import propose_weights
    from engine.embodied.formal_training import score_formal
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork
    from run_neural_muscle_closure import observe_quantities

    out = args.output
    if not out.is_absolute() or out.exists():
        raise ValueError('fresh absolute --output required')
    # Never nest a new writer inside any historical input directory.
    protected = (BASE_RUN.parent, MODEL.parent, INITIAL.parent, ENDPOINTS)
    if any(out.resolve().is_relative_to(p.resolve()) for p in protected):
        raise ValueError('output must not be inside a historical source directory')
    out.mkdir(parents=True, exist_ok=False)
    sync_dir(out.parent)
    lock = (out/'WRITER.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    caps = dict(neural_ticks=2400, physics_steps=12000,
                initial_forward=3, diagnostic_forward=12000)
    reserved = dict.fromkeys(caps, 0)
    completed = dict.fromkeys(caps, 0)
    ledger = dict(caps=caps, reserved=reserved, completed=completed, active=None,
                  recovery='NO_AUTOMATIC_RESUME; uncommitted reservations remain charged')
    atomic_json(out/'LEDGER.json', ledger)
    atomic_json(out/'PREREG.json', dict(evaluations=3,
        candidates=['plus', 'minus', 'updated'], baseline='saved s79 only; zero calls',
        baseline_path=str(BASELINE), prepare_s=.2, formal_s=.2,
        body_dt_s=.0001, neural_dt_s=.0005, command_dt_s=.005,
        schedule='observe -> 10 neural ticks -> old pending 50 physics steps -> consume; new pending next block',
        seed=8201, historical_seed=750401, perturbation_weight=.02, max_update_weight=.02,
        optimizer='propose_weights(original, lp, lm, wp, wm, 0, max_step=.02)',
        objective='score_formal(full_saved_arrays, weight_native): formal [200,400]ms loss ONLY',
        preparation='[0,200]ms diagnostics only, not included in optimizer loss',
        initial_weights='original frozen 66x5 FIR; all 330 originally nonzero; not scaled',
        zero_mask='preserved', phase_boundary='label only; never reset',
        cold_state='s79 q; zero qvel/act/ctrl/pending, fresh network and FIR history',
        controller='binary MN spikes -> fixed FIR -> muscle; sole initial q write',
        safety=dict(windows_ms=[[0,200],[200,400]], tolerance=1e-12,
                    each_window='min height nondecreasing; max descent, angle, nonfoot depth, limit excess nonincreasing versus saved baseline'),
        acceptance='updated only: safety in BOTH 200ms windows AND strictly greater formal combined-support integral than baseline; lower loss alone never accepts',
        initial_failures='retain and report; do not erase or reinterpret early failures',
        comparison='PENDING_INDEPENDENT_VERIFICATION; parent independently applies preregistered acceptance',
        actuator_records='actual actuator_force/actuator_length/actuator_velocity, 4001x90 each; baseline missing fields never fabricated',
        stopping='any nonfinite or MuJoCo warning: save partial shard and stop all candidates',
        claim='RESEARCH_ONLY; no autonomous standing/walking established', caps=caps))

    # This is the explicit s79 common source/input set, not an engine glob.
    # Neither runner nor the new objective belongs in the shared comparison.
    shared_paths = [MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
        ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
        ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
        ROOT/'run_neural_muscle_closure.py',
        ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    paths = shared_paths + [Path(__file__), ROOT/'engine/embodied/formal_training.py',
        BASE_RUN/'INPUT_HASHES.json', BASE_RUN/'SETUP.json', BASE_RUN/'baseline/RESULT.json', BASELINE]

    def reserve(name, block, delta):
        if ledger['active'] is not None:
            raise RuntimeError('uncommitted reservation; refusing another call')
        for key, value in delta.items():
            if reserved[key]+value > caps[key]:
                raise RuntimeError('hard call budget exceeded')
        for key, value in delta.items():
            reserved[key] += value
        ledger['active'] = dict(evaluation=name, block=block, delta=delta)
        atomic_json(out/'LEDGER.json', ledger)

    def commit():
        for key, value in ledger['active']['delta'].items():
            completed[key] += value
        ledger['active'] = None
        atomic_json(out/'LEDGER.json', ledger)

    try:
        identities = {str(p): sha(p) for p in paths}
        historical = json.loads((BASE_RUN/'INPUT_HASHES.json').read_text())
        old_runner = str(ROOT/'run_grounded_translation_training.py')
        expected = {p: h for p, h in historical.items() if p != old_runner}
        actual = {str(p): identities[str(p)] for p in shared_paths}
        if actual != expected:
            raise ValueError('shared s79 source/input hashes differ; refusing all calls')
        atomic_json(out/'INPUT_HASHES.json', identities)
        atomic_json(out/'SHARED_SOURCE_CHECK.json', dict(passed=True, shared=actual,
            excluded_historical_runner=old_runner,
            objective_identity=identities[str(ROOT/'engine/embodied/formal_training.py')]))
        manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
        original = np.array(json.loads((ENDPOINTS/'FROZEN_WEIGHTS.json').read_text())['weights'], float)
        if (original.shape != (66,5) or not np.isfinite(original).all()
                or np.count_nonzero(original) != 330 or np.any((original <= 0) | (original > 4))):
            raise ValueError('expected all 330 original nonzero bounded frozen FIR weights')
        frozen_zero = original == 0
        with np.load(INITIAL, allow_pickle=False) as initial:
            q = initial['new_qpos'].copy()
            sensory_ref = initial['sensory_reference_qpos'].copy()
        if (q.shape != (49,) or sensory_ref.shape != (49,) or
                not np.isfinite(q).all() or not np.isfinite(sensory_ref).all()):
            raise ValueError('invalid fixed initial or sensory reference')
        with np.load(BASELINE, allow_pickle=False) as saved:
            baseline_arrays = {key: saved[key].copy() for key in saved.files}
        baseline_result = json.loads((BASE_RUN/'baseline/RESULT.json').read_text())
        if baseline_result['trajectory_sha256'] != identities[str(BASELINE)]:
            raise ValueError('saved baseline trajectory hash differs from historical RESULT')
        if (baseline_arrays['qpos'].shape != (4001,49) or
                not np.array_equal(baseline_arrays['weights'], original) or
                not np.array_equal(baseline_arrays['qpos'][0], q) or
                any(np.any(baseline_arrays[k][0]) for k in ('qvel','act','ctrl')) or
                not np.allclose(baseline_arrays['time'],np.arange(4001)*.0001,rtol=0,atol=1e-12)):
            raise ValueError('saved baseline cold initial state/clock/weights mismatch')
        m = mj.MjModel.from_xml_path(str(MODEL))
        if (m.nu,m.nq,m.nv) != (90,49,48) or m.opt.timestep != .0001:
            raise ValueError('unexpected physical model or clock')
        if np.count_nonzero(m.geom_type == mj.mjtGeom.mjGEOM_MESH) != 71:
            raise ValueError('complete body required')
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        old_setup = json.loads((BASE_RUN/'SETUP.json').read_text())
        if weight_native != old_setup['weight_native']:
            raise ValueError('native force reference differs from s79')
        baseline_score = score_formal(baseline_arrays, weight_native)
        if not np.isfinite(float(baseline_score['loss'])):
            raise ValueError('nonfinite baseline formal score')
        atomic_json(out/'BASELINE_RESCORE.json', dict(name='saved_s79_baseline',
            **baseline_score, trajectory_path=str(BASELINE), trajectory_sha256=sha(BASELINE),
            simulation_calls=0, actuator_records='NOT_SAVED_IN_HISTORICAL_BASELINE; no reconstruction calls'))
        # Explicit original setup; no exec, historical imports, or warmed-state loader.
        built = build_neural_tier()
        a = built['data']
        tier = retain_motor_endpoints(a, downstream_union(a,built['global_index'],
            ['720575941433155799','720575941566493282'], hops=2,max_neurons=20000),
            ['720575941391252886','720575941511883985','720575941497037218','720575941559306355'])
        roots = a['root_ids'][tier['global_index']]
        if len(roots) != 17099:
            raise ValueError('unexpected neural tier')
        edges = build_edges(a,tier,built['cut'])
        groups,_ = channel_index_map(len(roots),roots)
        atomic_npz(out/'TIER_AND_CONNECTIVITY.npz', roots=roots,
            **{prefix+'_'+key: value for prefix, mat in [('We',edges['We']),('Wi',edges['Wi'])]
               for key,value in [('data',mat.tocsr().data),('indices',mat.tocsr().indices),
                                ('indptr',mat.tocsr().indptr),('shape',np.array(mat.shape))]})
        joints, geom_leg = {}, {}
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
        atomic_json(out/'SETUP.json', dict(neurons=len(roots), leg_order=list(LEGS),
            weight_native=weight_native, physical_units='NOT_ESTABLISHED',
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
                limit_excess=float(excess.max(initial=0)),
                warnings=np.array([w.number for w in d.warning]),time=float(d.time),
                phase=0 if d.time < .2-1e-12 else 1,
                actuator_force=d.actuator_force.copy(),actuator_length=d.actuator_length.copy(),
                actuator_velocity=d.actuator_velocity.copy())

        finite_keys = ('qpos','qvel','act','ctrl','foot_force',
                       'actuator_force','actuator_length','actuator_velocity')
        results = []

        def evaluate(weights,name):
            if identities != {str(p):sha(p) for p in paths}:
                raise RuntimeError('source/input identities changed before candidate')
            dest = out/name
            dest.mkdir(exist_ok=False)
            sync_dir(out)
            weights = np.asarray(weights,float)
            if (weights.shape != original.shape or not np.isfinite(weights).all() or
                    np.any((weights < 0) | (weights > 4)) or np.any(weights[frozen_zero] != 0)):
                raise ValueError('invalid bounded weights or zero mask')
            net = ConductanceNetwork(len(roots),dt_ms=.5)
            net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest,weights)
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
            if (not all(np.isfinite(initial_row[k]).all() for k in finite_keys)
                    or np.any(initial_row['warnings'])):
                atomic_json(dest/'INVALID.json',dict(block=-1,reason='invalid_initial_state'))
                raise RuntimeError('invalid initial state saved; reservation remains charged')
            commit()
            rows = [initial_row]
            block_data = []
            shard_index = []
            for block in range(80):
                reserve(name,block,dict(neural_ticks=10,physics_steps=50,diagnostic_forward=50))
                current,_ = enc.encode(observe_quantities(m,d,sensory_ref,joints,geom_leg))
                spikes = np.array([net.step(current_nA=current).copy() for _ in range(10)],dtype=bool)
                applied = pending.copy()
                d.ctrl[:] = applied
                new_rows = []
                invalid = None
                for sub in range(50):
                    if d.xfrc_applied.any() or d.qfrc_applied.any():
                        raise RuntimeError('unexpected external force')
                    mj.mj_step(m,d)
                    mj.mj_forward(m,d)
                    row = sample(d)
                    new_rows.append(row)
                    if (not all(np.isfinite(row[k]).all() for k in finite_keys)
                            or np.any(row['warnings'])):
                        invalid = dict(block=block,substep=sub,reason='nonfinite_or_warning')
                        break
                pending,info = dec.consume(spikes,spike_dt_s=.0005,
                    start_s=block*.005,end_s=(block+1)*.005)
                data = {key:np.array([r[key] for r in new_rows]) for key in initial_row}
                data.update(current=current,spikes=spikes,applied=applied,decoded=pending,
                            features=np.asarray(info['pool_features']))
                shard = dest/f'block_{block:03d}.npz'
                atomic_npz(shard,**data)
                shard_index.append(dict(block=block,path=str(shard),sha256=sha(shard)))
                atomic_json(dest/'SHARDS.json',shard_index)
                body = np.empty(mj.mj_stateSize(m,spec))
                mj.mj_getState(m,d,body,spec)
                snapshot = {key:value for key,value in vars(net).items() if isinstance(value,np.ndarray)}
                atomic_npz(dest/'LATEST_STATE.npz',body_integration=body,FIR_history=dec.history,
                           pending=pending,weights=weights,**snapshot)
                scalars = {key:value for key,value in vars(net).items() if isinstance(value,(int,float,str))}
                atomic_json(dest/'LATEST_STATE_META.json',dict(block=block,neural_scalars=scalars,
                    FIR_elapsed_s=dec.elapsed_s,FIR_interval=dec.interval,
                    snapshot_sha256=sha(dest/'LATEST_STATE.npz'),
                    shard=str(shard),shard_sha256=sha(shard),input_hashes=identities,
                    encoder='stateless physical encode; fixed config and references',
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
            if any(arrays[k].shape != (4001,90) for k in
                   ('actuator_force','actuator_length','actuator_velocity')):
                raise RuntimeError('incomplete actual actuator records')
            atomic_npz(dest/'TRAJECTORY.npz',**arrays)
            score = score_formal(arrays, weight_native)
            if not np.isfinite(float(score['loss'])):
                raise ValueError('nonfinite formal objective')
            record = dict(name=name,**score,trajectory_path=str(dest/'TRAJECTORY.npz'),
                          trajectory_sha256=sha(dest/'TRAJECTORY.npz'))
            atomic_json(dest/'RESULT.json',record)
            results.append(record)
            atomic_json(out/'EVALUATIONS.json',results)
            return float(score['loss'])

        direction = np.random.default_rng(8201).choice([-1.,1.],size=original.shape)
        wp = np.clip(original+.02*direction,0,4)
        wm = np.clip(original-.02*direction,0,4)
        wp[frozen_zero] = wm[frozen_zero] = 0
        atomic_json(out/'DIRECTION.json',dict(seed=8201,original=original.tolist(),
            direction=direction.tolist(),wp=wp.tolist(),wm=wm.tolist()))
        lp = evaluate(wp,'round00_plus')
        lm = evaluate(wm,'round00_minus')
        new = np.asarray(propose_weights(original,lp,lm,wp,wm,0,max_step=.02),float)
        if (new.shape != original.shape or not np.isfinite(new).all() or
                np.any((new<0)|(new>4)) or np.any(new[frozen_zero] != 0) or
                np.max(np.abs(new-original)) > .02+1e-12):
            raise ValueError('optimizer violated full-FIR weight update contract')
        atomic_json(out/'UPDATE.json',dict(weights=original.tolist(),direction=direction.tolist(),
            wp=wp.tolist(),wm=wm.tolist(),lp=lp,lm=lm,new=new.tolist(),k=0,max_step=.02))
        evaluate(new,'round00_updated')
        if reserved != caps or completed != caps or ledger['active'] is not None:
            raise RuntimeError('final exact call ledger mismatch')
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input identities changed during experiment')
        atomic_json(out/'SUMMARY.json',dict(status='BOUNDED_EXPERIMENT_COMPLETE',
            comparison='PENDING_INDEPENDENT_VERIFICATION',accepted=None,
            behavior_claim='NONE',baseline_rescore=baseline_score,evaluations=results,
            ledger=ledger,weight_native=weight_native,
            note='Formal loss only optimized; acceptance requires both-window safety and strictly larger formal support integral. All initial failures remain in full saved scores. No deployment.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
