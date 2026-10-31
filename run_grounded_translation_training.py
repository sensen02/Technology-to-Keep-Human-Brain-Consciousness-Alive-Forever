"""Opt-in bounded S2/S4 experiment: cold physical preparation, never pose forcing.

No simulation runs on import or --help. Each evaluation is a fresh 400ms physical
neural rollout (200ms preparation + 200ms evaluation). This is research, not a
walking claim. An interrupted reservation is conservatively charged, not replayed.
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
    p.add_argument('--output', type=Path, required=True, help='New absolute output directory; never reused')
    p.add_argument('--rounds', type=int, default=1, help='SPSA rounds, default 1, range 1..6; 1+3*rounds evaluations')
    return p


def run(args):
    import numpy as np
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.embodied.grounded_training import score_trajectory, propose_weights
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork
    from run_neural_muscle_closure import observe_quantities

    out = args.output
    if not out.is_absolute() or not 1 <= args.rounds <= 6:
        raise ValueError('absolute --output and --rounds within 1..6 required')
    out.mkdir(parents=True, exist_ok=False)
    sync_dir(out.parent)
    lock = (out/'WRITER.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    eval_count = 1+3*args.rounds
    caps = dict(neural_ticks=800*eval_count, physics_steps=4000*eval_count,
                initial_forward=eval_count, diagnostic_forward=4000*eval_count)
    reserved = dict.fromkeys(caps, 0)
    completed = dict.fromkeys(caps, 0)
    ledger = dict(caps=caps, reserved=reserved, completed=completed, active=None,
                  recovery='NO_AUTOMATIC_RESUME; uncommitted reservations remain charged')
    atomic_json(out/'PREREG.json', dict(rounds=args.rounds, evaluations=eval_count,
        prepare_s=.2, evaluate_s=.2, body_dt_s=.0001, neural_dt_s=.0005,
        command_dt_s=.005, seed=750401, perturbation_weight=.02, max_update_weight=.02,
        initial_weights='original frozen weights, not 0.45 scaled',
        zero_mask='preserved', phase_boundary='label only; no reset',
        controller='binary MN spikes -> fixed FIR -> muscle; no pose writes after initial q',
        stopping='any nonfinite or MuJoCo warning: persist partial, stop all evaluations',
        claim='RESEARCH_ONLY; no autonomous standing/walking established', caps=caps))
    paths = [Path(__file__), MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
        ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
        ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
        ROOT/'run_neural_muscle_closure.py',
        ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    identities = {str(p): sha(p) for p in paths}
    atomic_json(out/'INPUT_HASHES.json', identities)
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
                contacts=contacts,foot_force=force,nonfoot_depth=depth,
                limit_excess=float(excess.max(initial=0)),
                warnings=np.array([w.number for w in d.warning]),time=float(d.time),
                phase=0 if d.time < .2-1e-12 else 1)

        results = []
        def evaluate(weights,name):
            dest = out/name
            dest.mkdir(exist_ok=False)
            sync_dir(out)
            weights = np.asarray(weights,float)
            if weights.shape != original.shape or np.any(weights[frozen_zero] != 0):
                raise ValueError('weight shape or zero mask mismatch')
            net = ConductanceNetwork(len(roots),dt_ms=.5)
            net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest,weights)
            enc = ChannelSensoryEncoder(groups,len(roots))
            d = mj.MjData(m)
            d.qpos[:] = q  # Sole pose write: initial physical placement, never repeated.
            pending = np.zeros(90)
            reserve(name,-1,dict(initial_forward=1))
            mj.mj_forward(m,d)
            initial_row = sample(d)
            atomic_npz(dest/'INITIAL.npz',**initial_row,weights=weights)
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
                    if (not all(np.isfinite(row[k]).all() for k in ('qpos','qvel','act','ctrl','foot_force'))
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
            atomic_npz(dest/'TRAJECTORY.npz',**arrays)
            score = score_trajectory(qpos=arrays['qpos'],qvel=arrays['qvel'],contacts=arrays['contacts'],
                foot_force=arrays['foot_force'],nonfoot_depth=arrays['nonfoot_depth'],
                limit_excess=arrays['limit_excess'],reference_qpos=q,weight_native=weight_native,
                dt_s=.0001,prepare_s=.2)
            record = dict(name=name,**score,trajectory_sha256=sha(dest/'TRAJECTORY.npz'))
            atomic_json(dest/'RESULT.json',record)
            results.append(record)
            atomic_json(out/'EVALUATIONS.json',results)
            return float(score['loss'])

        weights = original.copy()
        evaluate(weights,'baseline')
        rng = np.random.default_rng(750401)
        for k in range(args.rounds):
            direction = rng.choice([-1.,1.],size=weights.shape)
            wp = np.clip(weights+.02*direction,0,4)
            wm = np.clip(weights-.02*direction,0,4)
            wp[frozen_zero] = wm[frozen_zero] = 0
            lp = evaluate(wp,f'round{k:02d}_plus')
            lm = evaluate(wm,f'round{k:02d}_minus')
            new = np.asarray(propose_weights(weights,lp,lm,wp,wm,k,max_step=.02),float)
            new[frozen_zero] = 0
            if (new.shape != weights.shape or not np.isfinite(new).all() or np.any((new<0)|(new>4))
                    or np.max(np.abs(new-weights)) > .02+1e-12):
                raise ValueError('optimizer violated full-FIR weight update contract')
            atomic_json(out/f'round{k:02d}_UPDATE.json',dict(weights=weights.tolist(),
                direction=direction.tolist(),wp=wp.tolist(),wm=wm.tolist(),lp=lp,lm=lm,new=new.tolist()))
            evaluate(new,f'round{k:02d}_updated')
            weights = new
        if reserved != caps or completed != caps:
            raise RuntimeError('final exact call ledger mismatch')
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input identities changed during experiment')
        atomic_json(out/'SUMMARY.json',dict(status='BOUNDED_EXPERIMENT_COMPLETE',
            behavior_claim='NONE',evaluations=results,ledger=ledger,weight_native=weight_native,
            note='Preparation and free evaluation both unforced. Full 400ms physical state continuous.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
