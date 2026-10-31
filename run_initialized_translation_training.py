"""S2/S4: one matched 40ms neural-initialized, four-arm 400ms FIR training round.

No runtime controller, external forces, activation prefill or post-initial pose writes.
Import and --help do not compile a body or advance a network. No automatic resume.
"""
from pathlib import Path
import argparse
import dataclasses
import importlib.util
import json
import shutil
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from run_native_rk4_gate_training import (sha, atomic_json, atomic_npz,
    callback_snapshot, require_no_callbacks, raw_observation_record, numeric_model_snapshot)
from run_native_rk4_tick_consumption import MODEL, INITIAL, ENDPOINTS, ROOT, S96_PARENT
from run_neural_initialized_startup import WEIGHTS

SEED, EPSILON, MAX_STEP, K = 10901, 1e-6, 1e-7, 0
PRE, TICKS, SUBSTEPS = 80, 800, 5


def neural_meta(net):
    """Lossless JSON-compatible scalar/config half of the network snapshot."""
    scalars, configs = {}, {}
    from scipy import sparse
    for key, value in vars(net).items():
        if isinstance(value, np.ndarray) or sparse.issparse(value):
            continue
        if dataclasses.is_dataclass(value):
            configs[key] = dataclasses.asdict(value)
        elif isinstance(value, np.generic):
            scalars[key] = value.item()
        elif isinstance(value, (bool, int, float, str)) or value is None:
            scalars[key] = value
        else:
            raise TypeError('unrecorded network field ' + key)
    return dict(neural_scalars=scalars, neural_configs=configs)


def mesh_dependencies():
    """Resolve actual XML assets before the first model compile."""
    tree = ET.parse(MODEL)
    if tree.findall('.//include'):
        raise ValueError('unexpected includes require explicit dependency resolution')
    compiler = tree.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''
    paths = []
    for mesh in tree.findall('.//asset/mesh'):
        source = Path(mesh.attrib['file'])
        path = source if source.is_absolute() else MODEL.parent / meshdir / source
        paths.append(path.resolve(strict=True))
    if len(paths) != 71:
        raise ValueError('expected 71 mesh dependencies')
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output
    if not out.is_absolute() or out.exists():
        raise ValueError('absolute fresh --output directory required; no resume')
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.embodied.grounded_training import propose_weights
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork
    from run_neural_muscle_closure import observe_quantities
    import initialized_training_objective

    dependencies = mesh_dependencies()
    # All project Python dependencies and actual data/assets are bound before runtime.
    sources = list(dict.fromkeys([MODEL, INITIAL, WEIGHTS, ENDPOINTS/'MANIFEST.json',
        ENDPOINTS/'FROZEN_WEIGHTS.json', ROOT/'data/flywire/banc_connectome.npz',
        ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json',
        S96_PARENT/'objective.py', S96_PARENT/'OBJECTIVE_PREREG.json',
        Path(__file__).resolve(), ROOT/'initialized_training_objective.py',
        ROOT/'run_neural_initialized_startup.py', ROOT/'run_neural_muscle_closure.py',
        ROOT/'run_native_rk4_gate_training.py', ROOT/'run_native_rk4_tick_consumption.py',
        ROOT/'run_native_rk4_reproducibility.py', *ROOT.glob('engine/**/*.py'), *dependencies]))
    identities = {str(s): sha(s) for s in sources}
    out.mkdir(parents=True, exist_ok=False)
    for name in ('objective.py', 'OBJECTIVE_PREREG.json'):
        shutil.copyfile(S96_PARENT/name, out/name)
        if (out/name).read_bytes() != (S96_PARENT/name).read_bytes():
            raise ValueError('objective bytecopy failed')
    spec = importlib.util.spec_from_file_location('initialized_s96_objective', str(out/'objective.py'))
    objective = importlib.util.module_from_spec(spec)
    exec(compile((out/'objective.py').read_bytes(), str(out/'objective.py'), 'exec'), objective.__dict__)
    scales = json.loads((out/'OBJECTIVE_PREREG.json').read_text())['scales']
    atomic_json(out/'INPUT_HASHES.json', identities)
    atomic_json(out/'MODEL_DEPENDENCIES.json', {str(s): identities[str(s)] for s in dependencies})
    atomic_json(out/'PREREG.json', dict(seed=SEED, epsilon=EPSILON, max_step=MAX_STEP, k=K,
        scales=scales, early_load_scale=.8, early_limit_scale=.001, early_limit_threshold=1e-6,
        early_window_s=[0,.01], early_aggregation='trapezoid dx=.0001 divided by .01',
        arms=['baseline','plus','minus','updated'], initialization_ticks=PRE,
        initialization_ms=40, physical_duration_ms=400, dt_ms=.1, neural_dt_ms=.5,
        sensory_hold_ms=5, source_offset_s=.04, same_weights=str(WEIGHTS),
        source_hashes=identities, objective_sha256=sha(out/'objective.py'),
        objective_prereg_sha256=sha(out/'OBJECTIVE_PREREG.json'),
        expected_calls=dict(model_load=4,forward=16004,step=16000,neural_ticks=3520),
        initialization='held actual initial observation; genuine net.step spikes; computational only; no body integration',
        physical_initial='sole initial qpos placement; zero qvel/act/ctrl/external loads; no later pose writes',
        clocks='continuous neural/FIR offset .04s; generated source END before physical interval START operator split',
        training='only existing FIR W; seed choice +/-1; preserve zero mask; probes clip [machine eps,4]; propose_weights on composite totals k0 maxstep1e-7',
        selection=False, deployment=False, checkpoint_decision='parent after independent verification',
        physical_units='NOT_ESTABLISHED', behavior_claim='NONE', automatic_resume=False))
    ledger = dict(model_load=0,forward=0,step=0,neural_ticks=0,active=None,completed=[])
    def reserve(**delta):
        for key, value in delta.items(): ledger[key] += value
        atomic_json(out/'CALL_LEDGER.json', ledger)
    reserve()
    with np.load(WEIGHTS, allow_pickle=False) as z: weights = z['weights'].copy()
    with np.load(INITIAL, allow_pickle=False) as z:
        q = z['new_qpos'].copy(); sensory_ref = z['sensory_reference_qpos'].copy()
    if weights.shape != (66,5) or not np.isfinite(weights).all() or np.any((weights<0)|(weights>4)):
        raise ValueError('invalid original W')
    direction = np.random.default_rng(SEED).choice([-1.,1.], size=weights.shape)
    mask = weights > 0
    plus = np.where(mask, np.clip(weights+EPSILON*direction,np.finfo(float).eps,4),0)
    minus = np.where(mask, np.clip(weights-EPSILON*direction,np.finfo(float).eps,4),0)
    atomic_npz(out/'DIRECTION.npz', weights=weights, direction=direction, plus=plus, minus=minus)
    manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
    results = []
    try:
        built = build_neural_tier(); a = built['data']
        tier = retain_motor_endpoints(a,downstream_union(a,built['global_index'],['720575941433155799','720575941566493282'],hops=2,max_neurons=20000),['720575941391252886','720575941511883985','720575941497037218','720575941559306355'])
        roots = a['root_ids'][tier['global_index']]; assert len(roots)==17099
        edges = build_edges(a,tier,built['cut']); groups,_ = channel_index_map(len(roots),roots)
        reference_before = reference_after = None
        for label in ('baseline','plus','minus','updated'):
            if label == 'updated':
                arm_weights = propose_weights(weights,results[1]['total'],results[2]['total'],plus,minus,K,max_step=MAX_STEP)
                atomic_npz(out/'UPDATE.npz', updated=arm_weights, weights=weights, plus=plus, minus=minus,
                    lp=np.asarray(results[1]['total']),lm=np.asarray(results[2]['total']))
            else:
                arm_weights = {'baseline':weights,'plus':plus,'minus':minus}[label]
            dest = out/label; dest.mkdir(); ledger['active'] = label
            reserve(model_load=1); m = mj.MjModel.from_xml_path(str(MODEL))
            assert (m.nq,m.nv,m.nu,m.nbody,m.nmesh)==(49,48,90,73,71) and m.opt.timestep==.0001
            assert m.opt.integrator==mj.mjtIntegrator.mjINT_EULER
            before = numeric_model_snapshot(m)
            m.opt.integrator = mj.mjtIntegrator.mjINT_RK4
            after = numeric_model_snapshot(m)
            assert [key for key in before if before[key]!=after[key]]==['opt.integrator']
            if reference_before is None: reference_before,reference_after = before,after
            assert before==reference_before and after==reference_after
            atomic_json(dest/'MODEL_PROVENANCE.json', dict(before=before,after=after,changed_numeric_fields=['opt.integrator'],xml_sha256=sha(MODEL)))
            require_no_callbacks(callback_snapshot(mj))
            d = mj.MjData(m); d.qpos[:] = q  # Sole pose write.
            net = ConductanceNetwork(len(roots),dt_ms=.5)
            net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest,arm_weights); enc = ChannelSensoryEncoder(groups,len(roots))
            assert not dec.history.any() and net.steps==0 and net.t_ms==0
            atomic_npz(dest/'NEURAL_COLD.npz', **{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)})
            atomic_json(dest/'NEURAL_COLD_META.json', neural_meta(net))
            connectivity = {}
            for name in ('We','Wi'):
                matrix = getattr(net,name).tocsr()
                for field in ('data','indices','indptr'):
                    connectivity[name+'_'+field] = getattr(matrix,field)
                connectivity[name+'_shape'] = np.asarray(matrix.shape)
            atomic_npz(dest/'CONNECTIVITY.npz', **connectivity)
            joints,geom_leg = {},{}
            for leg in LEGS:
                js = [j for j in range(m.njnt) if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j) or '').lower().startswith('joint_'+leg)]
                tib = next(j for j in js if 'tibia' in mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j).lower())
                joints[leg] = dict(q=[int(m.jnt_qposadr[j]) for j in js],tib_q=int(m.jnt_qposadr[tib]),tib_v=int(m.jnt_dofadr[tib]))
            for g in range(m.ngeom):
                gn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) or '').lower()
                bn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower()
                for leg in LEGS:
                    if 'tars' in gn+bn and (gn.startswith(leg) or bn.startswith(leg)): geom_leg[g] = leg
            floor, = np.flatnonzero(m.geom_type==mj.mjtGeom.mjGEOM_PLANE)
            limited = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[limited]
            reserve(forward=1); mj.mj_forward(m,d)
            initial_observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
            initial_current,_ = enc.encode(initial_observed)
            atomic_json(dest/'INITIAL_OBSERVATION.json',raw_observation_record(initial_observed)[0])
            warmsp,warmf,warmu = [],[],[]
            for j in range(PRE):
                reserve(neural_ticks=1); sp = net.step(current_nA=initial_current).copy()
                u,info = dec.consume(sp[None,:],spike_dt_s=.0005,start_s=j*.0005,end_s=(j+1)*.0005)
                warmsp.append(sp); warmf.append(info['pool_features']); warmu.append(u)
            release_features = np.stack([dec.history[lo:hi].mean(0) for lo,hi in dec.BINS],axis=1)
            pooled = np.clip((release_features*arm_weights).sum(1),0,1)
            release_ctrl = np.zeros(m.nu); mapped = dec.actuator_pool>=0
            release_ctrl[mapped] = pooled[dec.actuator_pool[mapped]]
            atomic_npz(dest/'INITIALIZATION.npz',spikes=np.asarray(warmsp,dtype=bool),features=np.asarray(warmf),decoded=np.asarray(warmu),
                current=initial_current,history=dec.history,release_features=release_features,release_decoded=release_ctrl,
                weights=arm_weights,body_qpos=d.qpos,body_qvel=d.qvel,body_act=d.act,body_ctrl=d.ctrl,
                **{k:v for k,v in vars(net).items() if isinstance(v,np.ndarray)})
            atomic_json(dest/'NEURAL_RELEASE_META.json',neural_meta(net))
            assert d.time==0 and np.array_equal(d.qpos,q) and not any(x.any() for x in (d.qvel,d.act,d.ctrl,d.xfrc_applied,d.qfrc_applied))
            def sample():
                contacts = np.zeros(6,dtype=int); force = np.zeros((6,3)); depth = 0.
                for ci,c in enumerate(d.contact):
                    g1,g2 = int(c.geom1),int(c.geom2)
                    if floor not in (g1,g2): continue
                    foot = g2 if g1==floor else g1
                    if foot in geom_leg:
                        li = LEGS.index(geom_leg[foot]); contacts[li] += int(c.dist<0)
                        f = np.zeros(6); mj.mj_contactForce(m,d,ci,f)
                        force[li] += (1 if g1==floor else -1)*c.frame.reshape(3,3).T@f[:3]
                    else: depth = max(depth,-float(c.dist))
                excess = np.maximum(np.maximum(m.jnt_range[limited,0]-d.qpos[qa],d.qpos[qa]-m.jnt_range[limited,1]),0)
                return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),
                    contacts=contacts,foot_force=force,nonfoot_depth=depth,limit_excess=float(excess.max(initial=0)),
                    actuator_force=d.actuator_force.copy(),qfrc_applied=d.qfrc_applied.copy(),xfrc_applied=d.xfrc_applied.copy(),
                    warnings=np.asarray([w.number for w in d.warning]))
            rows = [sample()]; currents,spikes,features,commands,schedule = [],[],[],[],[]
            for tick in range(TICKS):
                if tick%10==0:
                    observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                    current,_ = enc.encode(observed)
                    atomic_json(dest/f'OBSERVATION_{tick:02d}.json',raw_observation_record(observed)[0])
                currents.append(current.copy()); reserve(neural_ticks=1)
                sp = net.step(current_nA=current).copy()
                start,end = (PRE+tick)*.0005,(PRE+tick+1)*.0005
                u,info = dec.consume(sp[None,:],spike_dt_s=.0005,start_s=start,end_s=end)
                assert abs(net.t_ms/1000-end)<1e-12 and dec.interval==PRE+tick
                assert abs(d.time-tick*.0005)<1e-12
                spikes.append(sp); features.append(info['pool_features']); commands.append(u.copy())
                schedule.append(dict(tick=tick,source_start_s=start,source_end_s=end,physical_start_s=tick*.0005,
                    physical_end_s=(tick+1)*.0005,source_offset_s=.04,consume_id=dec.interval))
                d.ctrl[:] = u
                for sub in range(SUBSTEPS):
                    require_no_callbacks(callback_snapshot(mj))
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    reserve(step=1,forward=1); mj.mj_step(m,d); mj.mj_forward(m,d)
                    row = sample(); rows.append(row)
                    assert all(np.isfinite(v).all() for v in row.values()) and not row['warnings'].any()
                    assert not row['qfrc_applied'].any() and not row['xfrc_applied'].any()
            arrays = {key:np.asarray([r[key] for r in rows]) for key in rows[0]}
            arrays.update(spikes=np.asarray(spikes),features=np.asarray(features),applied=np.asarray(commands),
                decoded=np.asarray(commands),current=np.asarray(currents),weights=arm_weights,
                phase=np.r_[np.zeros(2001,dtype=int),np.ones(2000,dtype=int)])
            assert arrays['qpos'].shape==(4001,49) and arrays['spikes'].shape==(800,17099)
            assert np.allclose(arrays['time'],np.arange(4001)*.0001,rtol=0,atol=1e-12)
            assert net.steps==880 and dec.interval==879 and abs(dec.elapsed_s-.44)<1e-12
            assert numeric_model_snapshot(m)==after
            atomic_npz(dest/'TRAJECTORY.npz',**arrays); atomic_json(dest/'SCHEDULE.json',schedule)
            weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
            scores = initialized_training_objective.score(arrays,weight_native,scales,objective)
            assert set(scores)=={'total','formal_proxy','early_load','early_limit'} and np.isfinite(list(scores.values())).all()
            evidence = list(dest.glob('*'))
            hashes = {str(f):sha(f) for f in evidence if f.is_file()}
            record = dict(label=label,initialization_ticks=PRE,source_offset_s=.04,weight_native=weight_native,
                **{key:float(value) for key,value in scores.items()},hashes=hashes,
                trajectory_sha256=sha(dest/'TRAJECTORY.npz'),initialization_sha256=sha(dest/'INITIALIZATION.npz'),
                neural_cold_sha256=sha(dest/'NEURAL_COLD.npz'),neural_cold_meta_sha256=sha(dest/'NEURAL_COLD_META.json'),
                connectivity_sha256=sha(dest/'CONNECTIVITY.npz'),schedule_sha256=sha(dest/'SCHEDULE.json'),
                direction_sha256=sha(out/'DIRECTION.npz'),update_sha256=sha(out/'UPDATE.npz') if label=='updated' else None,
                neural_final_s=net.t_ms/1000,FIR_final_s=dec.elapsed_s,physical_final_s=float(d.time),behavior_claim='NONE')
            results.append(record); atomic_json(dest/'RESULT.json',record); atomic_json(out/'RESULTS.json',results)
            ledger['completed'].append(label); ledger['active']=None; reserve()
        assert {k:ledger[k] for k in ('model_load','forward','step','neural_ticks')}==dict(model_load=4,forward=16004,step=16000,neural_ticks=3520)
        assert identities=={str(s):sha(s) for s in sources}
        for name in ('objective.py','OBJECTIVE_PREREG.json'):
            assert (out/name).read_bytes()==(S96_PARENT/name).read_bytes()
        atomic_json(out/'COMPLETE.json',dict(status='COMPLETE',physical_units='NOT_ESTABLISHED',deployment=False,
            checkpoint_decision='parent after independent verification',ledger=ledger))
        print(json.dumps(results))
    except BaseException:
        atomic_json(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger)); raise


if __name__=='__main__':
    main()
