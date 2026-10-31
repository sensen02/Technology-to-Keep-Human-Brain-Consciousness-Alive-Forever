"""Bounded S2/S4 single-pool attenuation; explicit derivative of run_quarter_translation_backtrack.py.

Independently select one pool from fixed s84 200ms mechanics, then attenuate only
that original s79 FIR row by .9. One fresh continuous cold 200+200ms evaluation.
Import and --help perform no neural/physics work. Acceptance remains pending
independent both-window safety and strict formal cumulative-support verification.
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
SELECTION_DIR = Path('/home/sensen/Desktop/engine修复验收/s85-single-pool-attenuation/20261023T202215Z')
SELECTION = SELECTION_DIR/'SELECTION.json'
SAVED_WEIGHTS = SELECTION_DIR/'CANDIDATE_WEIGHTS.npz'
MECHANICS = Path('/home/sensen/Desktop/engine修复验收/s84-mechanical-mapping/20261023T201631Z/formal_start_MECHANICS.npz')
OUTPUT = SELECTION_DIR/'run'
FACTOR = .9


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
    p.add_argument('--output', type=Path, default=OUTPUT,
                   help='Fresh absolute output directory; never reused (default: %(default)s)')
    return p


def recompute_selection(manifest, original):
    """Pure saved-array selection; no network, model, step or forward calls."""
    import numpy as np
    from engine.embodied.motor_translation import FiniteMotorTranslator
    dec = FiniteMotorTranslator(manifest, original)
    if len(dec.pools) != 66 or dec.actuator_pool.shape != (90,):
        raise ValueError('expected 66 sorted shared pools and 90 actuator routes')
    with np.load(MECHANICS, allow_pickle=False) as saved:
        moment = saved['moment'].copy()
        active_force = saved['active_force'].copy()
    if (moment.shape != (90,48) or active_force.shape != (90,) or
            not np.isfinite(moment).all() or not np.isfinite(active_force).all()):
        raise ValueError('invalid fixed s84 mechanical arrays')
    contributions = moment[:,6:]*active_force[:,None]
    tau = np.array([contributions[dec.actuator_pool == p].sum(0) for p in range(66)])
    def cancellation(t):
        return float((np.abs(t).sum(0)-np.abs(t.sum(0))).sum())
    before = cancellation(tau)
    trials = []
    for pool in range(66):
        trial = tau.copy()
        trial[pool] *= FACTOR
        after = cancellation(trial)
        trials.append(dict(pool=pool, C_before=before, C_after=after,
            reduction=before-after,
            actuators=[int(x) for x in np.flatnonzero(dec.actuator_pool == pool)]))
    selected = sorted(trials, key=lambda r: (-r['reduction'], r['pool']))[0]
    return dict(valid=bool(selected['reduction'] > 0), selected=selected,
                all_trials=trials, mapping_digest=dec.mapping_digest), tau, contributions, dec.actuator_pool


def run(args):
    import numpy as np
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.embodied.formal_training import score_formal
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork
    from run_neural_muscle_closure import observe_quantities

    out = args.output
    if not out.is_absolute() or out.exists():
        raise ValueError('fresh absolute --output required')
    protected = (BASE_RUN.parent, MECHANICS.parent, MODEL.parent, INITIAL.parent, ENDPOINTS)
    if (out.resolve() == SELECTION_DIR.resolve() or
            any(out.resolve().is_relative_to(p.resolve()) for p in protected)):
        raise ValueError('output must not overwrite historical source directories')
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
    atomic_json(out/'LEDGER.json', ledger)
    atomic_json(out/'PREREG.json', dict(evaluations=1, candidates=['attenuated'], factor=FACTOR,
        formula='original s79 weights; selected pool row *= .9; all other rows bitwise unchanged',
        selection='fixed s84 200ms moment[:,6:]*active_force; sum by 66 sorted shared pools; C=sum_j(sum_p abs(tau_pj)-abs(sum_p tau_pj)); trial each pool*.9; largest positive reduction, exact ties smaller pool; no positive reduction stops before all calls',
        selection_path=str(SELECTION), mechanical_path=str(MECHANICS), saved_weights_path=str(SAVED_WEIGHTS),
        baseline='saved s79 only; zero calls', baseline_path=str(BASELINE),
        training='NONE; no random direction, optimizer, or additional evaluations',
        scope='fixed-state active-contribution heuristic, not FIR dynamics or support prediction; no pose change',
        prepare_s=.2, formal_s=.2, body_dt_s=.0001, neural_dt_s=.0005, command_dt_s=.005,
        schedule='observe -> 10 neural ticks -> old pending 50 physics steps -> consume; new pending next block',
        objective='unchanged score_formal(saved trajectory arrays, weight_native): formal [200,400]ms loss ONLY',
        preparation='[0,200]ms diagnostics only',
        initial_weights='original frozen 66x5 FIR; original zero mask and mapping preserved',
        zero_mask='preserved', phase_boundary='label only; never reset',
        cold_state='s79 q; zero qvel/act/ctrl/pending, fresh network and FIR history; no prefill',
        controller='binary MN spikes -> fixed FIR -> muscle; sole initial q write',
        safety=dict(windows_ms=[[0,200],[200,400]], tolerance=1e-12,
                    each_window='min height nondecreasing; max descent, angle, nonfoot depth, limit excess nonincreasing versus saved baseline'),
        acceptance='attenuated only: safety in BOTH 200ms windows AND strictly greater formal cumulative combined-support integral than baseline; lower loss alone never accepts',
        initial_failures='retain and report; do not erase or reinterpret early failures',
        comparison='PENDING_INDEPENDENT_VERIFICATION; parent independently applies preregistered acceptance',
        actuator_records='actual actuator_force/actuator_length/actuator_velocity, 4001x90 each; baseline missing fields never fabricated',
        stopping='any nonfinite or MuJoCo warning: save partial shard and stop',
        claim='RESEARCH_ONLY; no autonomous standing/walking established', caps=caps))

    # Identical explicit 14-item common source/input set from s79.
    shared_paths = [MODEL, INITIAL, ENDPOINTS/'MANIFEST.json', ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py', ROOT/'engine/embodied/motor_translation.py',
        ROOT/'engine/embodied/sensory_channels.py', ROOT/'engine/neural_cond.py',
        ROOT/'engine/embodied/adapters.py', ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py', ROOT/'engine/neck_cut_data.py',
        ROOT/'run_neural_muscle_closure.py',
        ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    paths = shared_paths + [Path(__file__), ROOT/'run_quarter_translation_backtrack.py',
        ROOT/'engine/embodied/formal_training.py',
        BASE_RUN/'INPUT_HASHES.json', BASE_RUN/'SETUP.json', BASE_RUN/'baseline/RESULT.json', BASELINE,
        SELECTION, SAVED_WEIGHTS, MECHANICS]

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
        if len(shared_paths) != 14 or actual != expected:
            raise ValueError('shared s79 source/input hashes differ; refusing all calls')
        saved_selection = json.loads(SELECTION.read_text())
        expected_selection_sources = {str(p): identities[str(p)] for p in
                                      (MECHANICS, BASELINE, ENDPOINTS/'MANIFEST.json')}
        if saved_selection['source_hashes'] != expected_selection_sources:
            raise ValueError('saved selection source hashes differ; refusing all calls')
        atomic_json(out/'INPUT_HASHES.json', identities)
        atomic_json(out/'SHARED_SOURCE_CHECK.json', dict(passed=True, shared=actual,
            excluded_historical_runner=old_runner, selection_sources=expected_selection_sources,
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
        selection, pool_tau, contributions, actuator_pool = recompute_selection(manifest, original)
        atomic_json(out/'RECOMPUTED_SELECTION.json', selection)
        if not selection['valid']:
            atomic_json(out/'END_INPUT_HASHES.json', {str(p):sha(p) for p in paths})
            if identities != {str(p):sha(p) for p in paths}:
                raise RuntimeError('source/input identities changed during selection')
            atomic_json(out/'SUMMARY.json', dict(status='STOP_NO_POSITIVE_REDUCTION',
                comparison='PENDING_INDEPENDENT_VERIFICATION', accepted=None, ledger=ledger,
                simulation_calls=0, selection=selection))
            return
        if any(selection[k] != saved_selection[k] for k in
               ('valid','selected','all_trials','mapping_digest')):
            raise ValueError('independent exact selection differs from saved record')
        pool = selection['selected']['pool']
        attenuated = original.copy()
        attenuated[pool] *= FACTOR
        with np.load(SAVED_WEIGHTS, allow_pickle=False) as saved:
            for key, value in dict(baseline=original, new=attenuated, pool_tau=pool_tau,
                                  actuator_active_contributions=contributions,
                                  actuator_pool=actuator_pool).items():
                if not np.array_equal(saved[key], value):
                    raise ValueError('saved candidate '+key+' differs from independent recomputation')
        original_digest = FiniteMotorTranslator(manifest,original).mapping_digest
        candidate_digest = FiniteMotorTranslator(manifest,attenuated).mapping_digest
        old_setup = json.loads((BASE_RUN/'SETUP.json').read_text())
        other = np.arange(66) != pool
        if (not np.array_equal(attenuated[other], original[other]) or
                not np.array_equal(attenuated[pool], original[pool]*FACTOR) or
                not np.array_equal(attenuated == 0, frozen_zero) or
                not np.isfinite(attenuated).all() or np.any((attenuated < 0) | (attenuated > 4)) or
                not (candidate_digest == original_digest == old_setup['mapping_digest'] == selection['mapping_digest'])):
            raise ValueError('single-row attenuation, original mask, or mapping mismatch')
        atomic_npz(out/'CANDIDATE_WEIGHTS.npz', baseline=original, new=attenuated,
                   pool_tau=pool_tau, actuator_active_contributions=contributions,
                   actuator_pool=actuator_pool, selected_pool=np.array(pool), factor=np.array(FACTOR))
        atomic_json(out/'PARAMETER_AUDIT.json', dict(factor=FACTOR, selected=selection['selected'],
            formula='original s79 weights; selected row *= .9; others bitwise unchanged',
            original_path=str(ENDPOINTS/'FROZEN_WEIGHTS.json'), baseline_path=str(BASELINE),
            mechanical_path=str(MECHANICS), selection_path=str(SELECTION),
            saved_candidate_path=str(SAVED_WEIGHTS), candidate_path=str(out/'CANDIDATE_WEIGHTS.npz'),
            candidate_sha256=sha(out/'CANDIDATE_WEIGHTS.npz'), shape=list(attenuated.shape),
            original_zero_mask_preserved=True, other_rows_bitwise_unchanged=True,
            mapping_digest=candidate_digest, mapping_preserved=True,
            max_abs_step=float(np.max(np.abs(attenuated-original))), source_hashes=identities))
        # All identities and candidate parameters validated before model/neural setup.
        if identities != {str(p):sha(p) for p in paths}:
            raise RuntimeError('source/input identities changed before setup')
        m = mj.MjModel.from_xml_path(str(MODEL))
        if (m.nu,m.nq,m.nv) != (90,49,48) or m.opt.timestep != .0001:
            raise ValueError('unexpected physical model or clock')
        if np.count_nonzero(m.geom_type == mj.mjtGeom.mjGEOM_MESH) != 71:
            raise ValueError('complete body required')
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
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

        evaluate(attenuated,'attenuated')
        if reserved != caps or completed != caps or ledger['active'] is not None:
            raise RuntimeError('final exact call ledger mismatch')
        end_identities = {str(p):sha(p) for p in paths}
        atomic_json(out/'END_INPUT_HASHES.json',end_identities)
        if identities != end_identities:
            raise RuntimeError('source/input identities changed during experiment')
        atomic_json(out/'SUMMARY.json',dict(status='BOUNDED_EXPERIMENT_COMPLETE',
            comparison='PENDING_INDEPENDENT_VERIFICATION',accepted=None,
            behavior_claim='NONE',baseline_rescore=baseline_score,evaluations=results,
            ledger=ledger,weight_native=weight_native,factor=FACTOR,selected_pool=pool,
            note='One fixed-state selected single-row attenuation only. Acceptance requires both-window safety and strictly larger formal cumulative support integral. All initial failures remain in saved scores. No deployment.'))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(parser().parse_args())
