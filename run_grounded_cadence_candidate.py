"""Opt-in bounded S2/S4 single cold cadence candidate; no training or baseline rerun.

Explicit standalone derivative of run_grounded_translation_training.py. No
simulation runs on import or --help. One unforced continuous 400ms rollout uses
current physical observation -> one neural tick -> consume -> five physics steps.
Neural end time leads physical apply start by 0.5ms: an engineering schedule,
not a biological zero-delay claim. Interrupted reservations are never replayed.
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
BASELINE = Path('/home/sensen/Desktop/engine修复验收/s79-grounded-translation/20261023T185506Z/run/baseline/TRAJECTORY.npz')
ORIGINAL_RUNNER = ROOT/'run_grounded_translation_training.py'


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
        np.savez(f, **arrays)  # Deliberately uncompressed: bounded synchronous I/O.
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True,
                   help='New absolute output directory; never reused; exactly one candidate')
    return p


def run(args):
    import numpy as np
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.embodied.grounded_training import score_trajectory
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork
    from run_neural_muscle_closure import observe_quantities

    out = args.output
    if not out.is_absolute():
        raise ValueError('absolute --output required')
    out.mkdir(parents=True, exist_ok=False)
    sync_dir(out.parent)
    lock = (out/'WRITER.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    caps = dict(neural_ticks=800, physics_steps=4000,
                initial_forward=1, diagnostic_forward=4000)
    reserved = dict.fromkeys(caps, 0)
    completed = dict.fromkeys(caps, 0)
    ledger = dict(caps=caps, reserved=reserved, completed=completed, active=None,
                  recovery='NO_AUTOMATIC_RESUME; uncommitted reservations remain charged')
    atomic_json(out/'PREREG.json', dict(evaluations=1, training=False, baseline_rerun=False,
        prepare_s=.2, evaluate_s=.2, body_dt_s=.0001, neural_dt_s=.0005,
        command_dt_s=.0005, groups=80, ticks_per_group=10, physics_steps_per_tick=5,
        initial_weights='original frozen weights, not 0.45 scaled',
        zero_mask='preserved', FIR_history='cold zero history; continuous thereafter',
        phase_boundary='label only; no reset',
        controller='current physical observation -> 1 tick -> consume -> immediately apply 5 physics steps',
        timing='neural [tick*.0005,(tick+1)*.0005]; physical applies same nominal interval; neural end leads physical start .0005s',
        pose='sole initial qpos write; no qvel/act/time resets or added forces',
        stopping='any nonfinite or MuJoCo warning: persist partial, stop candidate',
        criteria=dict(scorer='engine.embodied.grounded_training.score_trajectory unchanged',
            engineering='height>=.85 reference; angle<=.35rad; >=3 contacting legs; vertical foot force>=.8 weight; nonfoot depth<=1e-12; limit excess<=1e-6',
            preparation='all samples in terminal 20ms pass original gates',
            evaluation='all samples in full final 200ms pass original gates',
            comparison='actual identical-gate support integral nonworse in both segments and strictly better in at least one; all full400ms height/descent/attitude/nonfoot/limit safety metrics nonworse at 1e-12; loss alone insufficient',
            behavior='RESEARCH_ONLY; no standing/walking claim from loss improvement'), caps=caps))
    atomic_json(out/'LEDGER.json', ledger)

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
        # Preserve source identities, not just the runner's name. The only excluded
        # baseline key is its naturally different original runner path.
        paths = [Path(__file__), MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
            ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
            ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
            ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
            ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
            ROOT/'run_neural_muscle_closure.py',
            ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
        identities = {str(p): sha(p) for p in paths}
        baseline_inputs = BASELINE.parent.parent/'INPUT_HASHES.json'
        preserved = {str(p):sha(p) for p in (BASELINE, baseline_inputs, ORIGINAL_RUNNER)}
        saved = json.loads(baseline_inputs.read_text())
        shared = {k:v for k,v in identities.items() if k != str(Path(__file__))}
        saved_shared = {k:v for k,v in saved.items() if k != str(ORIGINAL_RUNNER)}
        if shared != saved_shared:
            raise RuntimeError('saved s79 source/input identities differ; refusing physical setup')
        atomic_json(out/'INPUT_HASHES.json', identities)
        atomic_json(out/'BASELINE_IDENTITY.json', dict(path=str(BASELINE), sha256=sha(BASELINE),
            input_hashes_path=str(baseline_inputs), input_hashes_sha256=sha(baseline_inputs),
            shared_source_inputs_match=True, excluded_runner_path=str(ORIGINAL_RUNNER),
            preserved_files=preserved, baseline_rerun=False))
        manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
        original = np.array(json.loads((ENDPOINTS/'FROZEN_WEIGHTS.json').read_text())['weights'], float)
        if original.shape != (66,5):
            raise ValueError('expected original 66 pools x 5 bins')
        frozen_zero = original == 0
        m = mj.MjModel.from_xml_path(str(MODEL))
        if (m.nu,m.nq,m.nv) != (90,49,48) or m.opt.timestep != .0001:
            raise ValueError('unexpected physical model or clock')
        if np.count_nonzero(m.geom_type == mj.mjtGeom.mjGEOM_MESH) != 71:
            raise ValueError('complete body required')
        with np.load(INITIAL, allow_pickle=False) as initial:
            q = initial['new_qpos'].copy()
            sensory_ref = initial['sensory_reference_qpos'].copy()
        with np.load(BASELINE, allow_pickle=False) as baseline:
            if not np.array_equal(baseline['weights'],original):
                raise ValueError('saved baseline does not use original frozen weights')
            baseline_arrays = {k:baseline[k].copy() for k in (
                'qpos','qvel','contacts','foot_force','nonfoot_depth','limit_excess','time')}
        if (baseline_arrays['qpos'].shape != (4001,49) or
                not np.allclose(baseline_arrays['time'],np.arange(4001)*.0001,rtol=0,atol=1e-12)):
            raise ValueError('saved baseline requires continuous complete 400ms trajectory')
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
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
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
                actuator_force=d.actuator_force.copy(),actuator_length=d.actuator_length.copy(),
                actuator_velocity=d.actuator_velocity.copy(),
                contacts=contacts,foot_force=force,nonfoot_depth=depth,
                limit_excess=float(excess.max(initial=0)),
                warnings=np.array([w.number for w in d.warning]),time=float(d.time),
                phase=0 if d.time < .2-1e-12 else 1)

        def score(arrays):
            return score_trajectory(qpos=arrays['qpos'],qvel=arrays['qvel'],contacts=arrays['contacts'],
                foot_force=arrays['foot_force'],nonfoot_depth=arrays['nonfoot_depth'],
                limit_excess=arrays['limit_excess'],reference_qpos=q,weight_native=weight_native,
                dt_s=.0001,prepare_s=.2)

        baseline_score = score(baseline_arrays)
        atomic_json(out/'SAVED_BASELINE_SCORE.json',dict(path=str(BASELINE),
            trajectory_sha256=preserved[str(BASELINE)],**baseline_score))
        name = 'candidate'
        dest = out/name
        dest.mkdir(exist_ok=False)
        sync_dir(out)
        weights = original.copy()
        net = ConductanceNetwork(len(roots),dt_ms=.5)
        net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
        dec = FiniteMotorTranslator(manifest,weights)
        enc = ChannelSensoryEncoder(groups,len(roots))
        d = mj.MjData(m)
        d.qpos[:] = q  # Sole pose write: initial physical placement, never repeated.
        command = np.zeros(90)
        reserve(name,-1,dict(initial_forward=1))
        mj.mj_forward(m,d)
        initial_row = sample(d)
        atomic_npz(dest/'INITIAL.npz',**initial_row,weights=weights)
        commit()
        rows = [initial_row]
        block_data = []
        shard_index = []
        event_keys = ('tick','neural_start_s','neural_end_s',
                      'physical_apply_start_s','physical_apply_end_s')
        for block in range(80):
            reserve(name,block,dict(neural_ticks=10,physics_steps=50,diagnostic_forward=50))
            new_rows, tick_data = [], []
            invalid = None
            for offset in range(10):
                tick = block*10+offset
                neural_start_s = tick*.0005
                neural_end_s = (tick+1)*.0005
                physical_apply_start_s = float(d.time)
                current,_ = enc.encode(observe_quantities(m,d,sensory_ref,joints,geom_leg))
                spikes = np.asarray(net.step(current_nA=current),dtype=bool).copy()[None,:]
                command,info = dec.consume(spikes,spike_dt_s=.0005,
                    start_s=neural_start_s,end_s=neural_end_s)
                applied = command.copy()
                d.ctrl[:] = applied
                event = dict(tick=tick,neural_start_s=neural_start_s,neural_end_s=neural_end_s,
                    physical_apply_start_s=physical_apply_start_s,physical_apply_end_s=float(d.time),
                    current=current.copy(),spikes=spikes,applied=applied,decoded=command.copy(),
                    features=np.asarray(info['pool_features']))
                for sub in range(5):
                    if d.xfrc_applied.any() or d.qfrc_applied.any():
                        raise RuntimeError('unexpected external force')
                    mj.mj_step(m,d)
                    mj.mj_forward(m,d)
                    row = sample(d)
                    new_rows.append(row)
                    if (not all(np.isfinite(row[k]).all() for k in (
                            'qpos','qvel','act','ctrl','foot_force','actuator_force',
                            'actuator_length','actuator_velocity')) or np.any(row['warnings'])):
                        invalid = dict(block=block,tick=tick,substep=sub,reason='nonfinite_or_warning')
                        break
                event['physical_apply_end_s'] = float(d.time)
                tick_data.append(event)
                if invalid is not None:
                    break
            data = {key:np.array([r[key] for r in new_rows]) for key in initial_row}
            data.update({key:np.array([t[key] for t in tick_data]) for key in tick_data[0]})
            shard = dest/f'block_{block:03d}.npz'
            atomic_npz(shard,**data)
            shard_index.append(dict(block=block,path=str(shard),sha256=sha(shard)))
            atomic_json(dest/'SHARDS.json',shard_index)
            body = np.empty(mj.mj_stateSize(m,spec))
            mj.mj_getState(m,d,body,spec)
            snapshot = {key:value for key,value in vars(net).items() if isinstance(value,np.ndarray)}
            atomic_npz(dest/'LATEST_STATE.npz',body_integration=body,FIR_history=dec.history,
                       command=command,weights=weights,**snapshot)
            scalars = {key:value for key,value in vars(net).items() if isinstance(value,(int,float,str))}
            atomic_json(dest/'LATEST_STATE_META.json',dict(block=block,last_tick=tick,
                neural_scalars=scalars,FIR_elapsed_s=dec.elapsed_s,FIR_interval=dec.interval,
                physical_time_s=float(d.time),body_state_spec=int(spec),
                snapshot_sha256=sha(dest/'LATEST_STATE.npz'),
                shard=str(shard),shard_sha256=sha(shard),input_hashes=identities,
                encoder='stateless physical encode; fixed config and references',
                complete_block=invalid is None,automatic_resume=False,
                ledger_active=ledger['active'],
                checkpoint_atomicity='state and metadata separately atomic; no automatic recovery'))
            rows.extend(new_rows)
            block_data.append(data)
            if invalid is not None:
                atomic_json(dest/'INVALID.json',invalid)
                raise RuntimeError('invalid trajectory: partial shard saved, reservation conservatively charged')
            commit()
        arrays = {key:np.array([r[key] for r in rows]) for key in initial_row}
        arrays.update({key:np.concatenate([b[key] for b in block_data],axis=0)
                       for key in ('spikes','current','applied','decoded','features')+event_keys})
        arrays['weights'] = weights
        expected = dict(spikes=(800,1,len(roots)),current=(800,len(roots)),
                        applied=(800,90),decoded=(800,90),features=(800,66,5),
                        actuator_force=(4001,90),actuator_length=(4001,90),
                        actuator_velocity=(4001,90),qpos=(4001,49))
        if any(arrays[key].shape != shape for key,shape in expected.items()):
            raise RuntimeError('final trajectory shape mismatch')
        if (not np.array_equal(weights,original) or np.any(weights[frozen_zero] != 0) or
                not np.array_equal(arrays['applied'],arrays['decoded'])):
            raise RuntimeError('frozen weight or immediate command contract violated')
        if not (np.allclose(arrays['neural_end_s']-arrays['physical_apply_start_s'],.0005,
                            rtol=0,atol=1e-12) and
                np.allclose(arrays['physical_apply_end_s'],arrays['neural_end_s'],rtol=0,atol=1e-12)):
            raise RuntimeError('event timing contract violated')
        atomic_npz(dest/'TRAJECTORY.npz',**arrays)
        candidate_score = score(arrays)
        record = dict(name=name,**candidate_score,trajectory_sha256=sha(dest/'TRAJECTORY.npz'))
        atomic_json(dest/'RESULT.json',record)
        atomic_json(out/'EVALUATIONS.json',[record])
        if reserved != caps or completed != caps:
            raise RuntimeError('final exact call ledger mismatch')
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input identities changed during experiment')
        if preserved != {p:sha(p) for p in preserved}:
            raise RuntimeError('saved baseline or original runner changed during experiment')
        def comparison_metrics(a):
            height = a['qpos'][:,2]/q[2]
            quat = a['qpos'][:,3:7]
            angle = 2*np.arccos(np.clip(np.abs(quat@q[3:7])/
                (np.linalg.norm(quat,axis=1)*np.linalg.norm(q[3:7])),0,1))
            good = ((height >= .85) & (angle <= .35) & ((a['contacts']>0).sum(axis=1) >= 3)
                & (a['foot_force'][:,:,2].sum(axis=1)/weight_native >= .8)
                & (a['nonfoot_depth'] <= 1e-12) & (a['limit_excess'] <= 1e-6))
            return dict(min_height_ratio=float(height.min()),
                max_downward_velocity=float(np.maximum(-a['qvel'][:,2],0).max()),
                max_angle_rad=float(angle.max()),max_nonfoot_depth=float(a['nonfoot_depth'].max()),
                max_joint_limit_excess=float(a['limit_excess'].max()),
                preparation_support_s=float(np.trapezoid(good[:2001].astype(float),dx=.0001)),
                evaluation_support_s=float(np.trapezoid(good[2000:].astype(float),dx=.0001)))
        bm, cm = comparison_metrics(baseline_arrays), comparison_metrics(arrays)
        tolerance = 1e-12
        safety = {'min_height_ratio':cm['min_height_ratio'] >= bm['min_height_ratio']-tolerance}
        safety.update({key:cm[key] <= bm[key]+tolerance for key in (
            'max_downward_velocity','max_angle_rad','max_nonfoot_depth','max_joint_limit_excess')})
        support_keys = ('preparation_support_s','evaluation_support_s')
        support_nonworse = all(cm[k] >= bm[k]-tolerance for k in support_keys)
        support_improved = any(cm[k] > bm[k]+tolerance for k in support_keys)
        accepted = bool(all(safety.values()) and support_nonworse and support_improved)
        comparison = dict(baseline=bm,candidate=cm,tolerance=tolerance,safety_nonworsening=safety,
            support_nonworse_both_segments=support_nonworse,support_strictly_improved=support_improved,
            eligible_for_later_training_consideration=accepted,
            success=bool(candidate_score['eligible']),loss_alone_insufficient=True)
        atomic_json(out/'COMPARISON.json',comparison)
        atomic_json(out/'SUMMARY.json',dict(status='SINGLE_CADENCE_CANDIDATE_COMPLETE',
            behavior_claim='NONE',candidate=record,saved_baseline=baseline_score,
            baseline_path=str(BASELINE),baseline_sha256=preserved[str(BASELINE)],baseline_rerun=False,
            loss_delta=float(candidate_score['loss']-baseline_score['loss']),
            comparison=comparison,
            decision='SUPPORT_IMPROVED_SAFETY_NONWORSE_CONSIDER_LATER_TRAINING' if accepted else 'REJECT_SINGLE_CADENCE_CANDIDATE',
            ledger=ledger,weight_native=weight_native,source_hashes_unchanged=True,
            note='Unforced cold 400ms continuous physical state. Neural end leads physical start .5ms; not zero biological latency.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
