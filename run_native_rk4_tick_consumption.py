"""S2/S4 s100: ONE fresh 400ms arm, tickwise causal FIR consumption.

Import/--help/--offline-check never build a body or advance a neural network.
The only runtime difference from s92 is operator-split tick consumption/application.
Parent supplies byte-identical s96 objective.py and OBJECTIVE_PREREG.json.
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
    sha, sync_dir, atomic_json, atomic_npz, numeric_model_snapshot,
    callback_snapshot, require_no_callbacks, raw_observation_record,
    support_spans, acceptance,
)
from run_native_rk4_reproducibility import json_value, rng_snapshot

ROOT = Path(__file__).resolve().parent
MODEL = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
INITIAL = Path('/run/media/sensen/Data2/engine修复验收/s7-posture-ik/20261023T021141Z/DERIVED_INITIAL.npz')
ENDPOINTS = Path('/run/media/sensen/Data2/engine修复验收/s4-endpoints/20261023T011559Z')
BASE_RUN = Path('/home/sensen/Desktop/engine修复验收/s92-rk4-full-baseline/20261023T210730Z/run')
BASELINE = BASE_RUN/'baseline400ms/TRAJECTORY.npz'
S96_PARENT = Path('/home/sensen/Desktop/engine修复验收/s96-six-gate-training/20261023T214545Z')
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s100-tick-consumption/20261023T222456Z/run')
GATE_KEYS = ('height','angle','legs','load','nonfoot','limit')
BODY_DT, NEURAL_DT, COMMAND_DT = .0001, .0005, .005
BLOCKS, TICKS, SAMPLES = 80, 800, 4001
ARM = 'tick400ms'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--offline-check',action='store_true',help='Saved spikes only; no body compile/neural execution; writes parent OFFLINE_FIR_CHECK JSON/NPZ')
    return p


def offline_check():
    import numpy as np
    from engine.embodied.motor_translation import FiniteMotorTranslator
    manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
    source_hash = sha(BASELINE)
    with np.load(BASELINE,allow_pickle=False) as z:
        spikes, weights = z['spikes'].copy(), z['weights'].copy()
    if spikes.shape != (80,10,17099):
        raise ValueError('saved spike layout differs')
    batched, single = FiniteMotorTranslator(manifest,weights), FiniteMotorTranslator(manifest,weights)
    batch_f, tick_f, batch_u, tick_u = [],[],[],[]
    history_exact = []
    for block in range(80):
        u, info = batched.consume(spikes[block],spike_dt_s=NEURAL_DT,start_s=block*COMMAND_DT,end_s=(block+1)*COMMAND_DT)
        batch_u.append(u); batch_f.append(np.asarray(info['pool_features']))
        for j in range(10):
            k = block*10+j
            v, details = single.consume(spikes[block,j:j+1],spike_dt_s=NEURAL_DT,start_s=k*NEURAL_DT,end_s=(k+1)*NEURAL_DT)
            tick_u.append(v); tick_f.append(np.asarray(details['pool_features']))
        history_exact.append(np.array_equal(batched.history,single.history))
    bf, tf, bu, tu = map(np.asarray,(batch_f,tick_f,batch_u,tick_u))
    passed = bool(np.array_equal(bf,tf[9::10]) and np.array_equal(bu,tu[9::10]) and all(history_exact))
    if sha(BASELINE) != source_hash:
        raise RuntimeError('saved source changed')
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    atomic_npz(OUTPUT.parent/'OFFLINE_FIR_CHECK.npz',batch_features=bf,tick_features=tf,batch_decoded=bu,tick_decoded=tu,history_exact=np.asarray(history_exact))
    record = dict(passed=passed,features_exact=bool(np.array_equal(bf,tf[9::10])),decoded_exact=bool(np.array_equal(bu,tu[9::10])),
        history_exact=all(history_exact),source=str(BASELINE),source_sha256=source_hash,manifest_sha256=sha(ENDPOINTS/'MANIFEST.json'),
        translator_sha256=sha(ROOT/'engine/embodied/motor_translation.py'),runner_sha256=sha(Path(__file__)),
        comparison='80x10 versus 800x1 saved spikes; features/decode/history EXACT at every tenth tick',
        neural_ticks=0,body_compiles=0,physics_steps=0,explicit_forward=0,
        arrays_sha256=sha(OUTPUT.parent/'OFFLINE_FIR_CHECK.npz'))
    atomic_json(OUTPUT.parent/'OFFLINE_FIR_CHECK.json',record)
    if not passed:
        raise RuntimeError('offline FIR equivalence failed')
    print(json.dumps(record,ensure_ascii=False))


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
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    from run_neural_muscle_closure import observe_quantities

    out = args.output
    if out != OUTPUT or not out.is_absolute() or out.exists():
        raise ValueError('fixed fresh output only; no automatic resume')
    required = [out.parent/f for f in ('PREREG.json','objective.py','OBJECTIVE_PREREG.json','STATIC_CHECK.json','OFFLINE_FIR_CHECK.json','OFFLINE_FIR_CHECK.npz')]
    if any(not p.is_file() for p in required):
        raise ValueError('parent preregistration/static/offline checks required')
    for name in ('objective.py','OBJECTIVE_PREREG.json'):
        if (out.parent/name).read_bytes() != (S96_PARENT/name).read_bytes():
            raise ValueError('objective/scales must be original s96 bytecopy')
    offline = json.loads((out.parent/'OFFLINE_FIR_CHECK.json').read_text())
    static = json.loads((out.parent/'STATIC_CHECK.json').read_text())
    if not offline.get('passed') or not static.get('passed') or offline['runner_sha256'] != sha(Path(__file__)):
        raise ValueError('failed or stale pre-run static/offline checks')
    if offline['source_sha256'] != sha(BASELINE) or offline['arrays_sha256'] != sha(out.parent/'OFFLINE_FIR_CHECK.npz') or offline['translator_sha256'] != sha(ROOT/'engine/embodied/motor_translation.py'):
        raise ValueError('offline inputs changed')
    objective_path = out.parent/'objective.py'
    objective_source = objective_path.read_bytes()
    spec = importlib.util.spec_from_file_location('s100_objective',str(objective_path))
    objective = importlib.util.module_from_spec(spec)
    exec(compile(objective_source,str(objective_path),'exec'),objective.__dict__)
    scales = json.loads((out.parent/'OBJECTIVE_PREREG.json').read_text())['scales']
    if set(scales) != set(GATE_KEYS) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError('invalid six scales')
    shared = [MODEL,INITIAL,ENDPOINTS/'MANIFEST.json',ENDPOINTS/'FROZEN_WEIGHTS.json',
        ROOT/'engine/embodied/grounded_training.py',ROOT/'engine/embodied/motor_translation.py',ROOT/'engine/embodied/sensory_channels.py',
        ROOT/'engine/neural_cond.py',ROOT/'engine/embodied/adapters.py',ROOT/'engine/embodied/command_tier.py',
        ROOT/'engine/embodied/motor_endpoints.py',ROOT/'engine/neck_cut_data.py',ROOT/'run_neural_muscle_closure.py',ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json']
    paths = list(dict.fromkeys(shared + required + [BASELINE,BASE_RUN/'baseline400ms/RESULT.json',BASE_RUN/'INPUT_HASHES.json',BASE_RUN/'MODEL_PROVENANCE.json',BASE_RUN/'LEDGER.json',BASE_RUN/'SETUP.json',
        ROOT/'run_native_rk4_full_baseline.py',ROOT/'run_native_rk4_gate_training.py',ROOT/'run_native_rk4_reproducibility.py',ROOT/'engine/embodied/formal_training.py',Path(__file__).resolve(),S96_PARENT/'objective.py',S96_PARENT/'OBJECTIVE_PREREG.json']))
    identities = {str(p):sha(p) for p in paths}
    prior_hashes = json.loads((BASE_RUN/'INPUT_HASHES.json').read_text())
    if len(shared) != 14 or any(prior_hashes[str(p)] != identities[str(p)] for p in shared+[ROOT/'run_native_rk4_full_baseline.py']):
        raise ValueError('s92 shared14/source binding failed')
    old_ledger = json.loads((BASE_RUN/'LEDGER.json').read_text())
    if old_ledger['active'] is not None or old_ledger['reserved'] != old_ledger['completed']:
        raise ValueError('uncommitted baseline')
    with np.load(BASELINE,allow_pickle=False) as z:
        baseline = {k:z[k].copy() for k in z.files}
    if json.loads((BASE_RUN/'baseline400ms/RESULT.json').read_text())['trajectory_sha256'] != sha(BASELINE):
        raise ValueError('baseline result binding failed')
    weights = baseline['weights'].copy()
    frozen = np.asarray(json.loads((ENDPOINTS/'FROZEN_WEIGHTS.json').read_text())['weights'],float)
    if weights.shape != (66,5) or weights.dtype != frozen.dtype or weights.tobytes() != frozen.tobytes():
        raise ValueError('original s92 weights bytecopy required')
    with np.load(INITIAL,allow_pickle=False) as z:
        q, sensory_ref = z['new_qpos'].copy(),z['sensory_reference_qpos'].copy()
    if not np.array_equal(q,baseline['qpos'][0]) or any(np.any(baseline[k][0]) for k in ('qvel','act','ctrl','applied')):
        raise ValueError('baseline cold body differs')
    out.mkdir(parents=True,exist_ok=False); sync_dir(out.parent)
    lock = (out/'WRITER.lock').open('x'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    caps = dict(neural_ticks=800,physics_steps=4000,initial_forward=1,diagnostic_forward=4000)
    reserved, completed = dict.fromkeys(caps,0),dict.fromkeys(caps,0)
    ledger = dict(caps=caps,reserved=reserved,completed=completed,active=None,total_forward_cap=4001,
        recovery='NO_RESUME; uncommitted full-block reservation remains charged; never retry invalid arm',
        scope='public calls only; native RK4 internal stages additional; no neural calls inside RK4')
    timing = dict(blocks=80,duration_s=.4,body_dt_s=BODY_DT,neural_dt_s=NEURAL_DT,command_dt_s=COMMAND_DT,samples=4001,
        neural_ticks_per_block=10,physics_steps_per_block=50,commands_per_tick=1,physics_steps_per_tick=5,
        logical_order='At physical t: observe/encode only if block boundary; advance actual neural model once; consume that generated spike once; decode immediately; apply ctrl; advance five body steps to t+.0005.',
        split_clocks='Lie/operator split: generated model interval [t,t+.0005] ends before body interval [t,t+.0005] is advanced. Decoder valid_from/feature_end are NEURAL model clock, NOT physical apply start. Both clocks explicitly saved.',
        no_future_data='No extrapolated spikes or future body observations. Neural step is actually computed before consume and body. No .5ms physical shift, extra initial tick, delay, discarded final tick, old pending, or prefill.',
        first_tick='sample 0; neural model start0/end.0005; physical apply start0/end.0005; first five physics steps use this generated command',
        last_tick='neural end .4 and physical end .4; final decoded command applied to final five body steps',
        FIR='UNCHANGED .5ms ticks, bins ((0,5),(5,10),(10,20),(20,40),(40,80)); full80tick/40ms zero-cold history',
        phase='continuous first200ms preparation, last200ms formal; labels only, no reset',scoring_windows='prep0:2001; formal2000:4001')
    def reserve(block,delta):
        if ledger['active'] is not None or any(reserved[k]+v>caps[k] for k,v in delta.items()):
            raise RuntimeError('uncommitted reservation or hard cap exceeded')
        for k,v in delta.items(): reserved[k] += v
        ledger['active'] = dict(arm=ARM,block=block,delta=delta)
        atomic_json(out/'LEDGER.json',ledger)
    def commit():
        for k,v in ledger['active']['delta'].items(): completed[k] += v
        ledger['active'] = None; atomic_json(out/'LEDGER.json',ledger)
    atomic_json(out/'LEDGER.json',ledger)
    atomic_json(out/'TIMING.json',timing)
    atomic_json(out/'INPUT_HASHES.json',identities)
    atomic_json(out/'PREREG.json',dict(arms=[ARM],fresh_evaluations=1,timing=timing,caps=caps,total_forward_cap=4001,
        only_change='schedule: consume/decode actual generated spike once per .5ms, immediate physical split application',
        weights='s92 original dtype/shape/bytes copied; no arithmetic/update/selection/deployment',
        cold='fresh body/network/FIR; sole qpos initial placement; zero velocity/act/ctrl; no prefill/controller/old pending',
        objective='original s96 bytecopy, report only; score_formal unchanged; original physical acceptance descriptive',
        acceptance_updates=False,deployment=False,behavior_claim='NONE',physical_units='NOT_ESTABLISHED',automatic_resume=False))
    try:
        before_cb = callback_snapshot(mj); require_no_callbacks(before_cb)
        atomic_json(out/'CALLBACKS_BEFORE.json',before_cb)
        if mj.__version__ != '3.9.0' or mj.mj_versionString() != '3.9.0': raise RuntimeError('MuJoCo3.9.0 required')
        atomic_json(out/'ENVIRONMENT.json',dict(executable=sys.executable,python=sys.version,platform=platform.platform(),
            versions=dict(numpy=np.__version__,scipy=scipy.__version__,mujoco=mj.__version__,native=mj.mj_versionString()),
            environment=dict(os.environ),initial_rng=rng_snapshot(np),neural_rng='existing deterministic constructor; no new seed'))
        provenance = json.loads((BASE_RUN/'MODEL_PROVENANCE.json').read_text())
        m = mj.MjModel.from_xml_path(str(MODEL))
        if (m.nu,m.nq,m.nv)!=(90,49,48) or m.opt.timestep!=BODY_DT or m.opt.integrator!=mj.mjtIntegrator.mjINT_EULER or np.count_nonzero(m.geom_type==mj.mjtGeom.mjGEOM_MESH)!=71:
            raise ValueError('original fullbody/dt/Euler mismatch')
        if np.any(m.actuator_delay) or np.any(m.actuator_history) or m.nplugin or np.any(m.actuator_plugin!=-1): raise RuntimeError('delay/history/plugin forbidden')
        before = numeric_model_snapshot(m)
        if before != provenance['before'] or provenance['xml_sha256'] != sha(MODEL): raise RuntimeError('s92 before mismatch')
        m.opt.integrator = mj.mjtIntegrator.mjINT_RK4  # ONLY model-field assignment.
        after = numeric_model_snapshot(m)
        if after != provenance['after'] or [k for k in before if before[k]!=after[k]]!=['opt.integrator']: raise RuntimeError('s92 RK4 after mismatch')
        atomic_json(out/'MODEL_PROVENANCE.json',dict(xml_path=str(MODEL),xml_sha256=sha(MODEL),before=before,after=after,changed_numeric_fields=['opt.integrator'],no_other_model_field_changes=True,s92_before_equal=True,s92_after_equal=True))
        weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
        built = build_neural_tier(); a = built['data']
        tier = retain_motor_endpoints(a,downstream_union(a,built['global_index'],['720575941433155799','720575941566493282'],hops=2,max_neurons=20000),['720575941391252886','720575941511883985','720575941497037218','720575941559306355'])
        roots = a['root_ids'][tier['global_index']]
        if len(roots)!=17099: raise ValueError('tier differs')
        edges = build_edges(a,tier,built['cut']); groups,_ = channel_index_map(len(roots),roots)
        joints, geom_leg = {},{}
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
        dest = out/ARM; dest.mkdir(); sync_dir(out)
        rng_before = rng_snapshot(np)
        net = ConductanceNetwork(len(roots),dt_ms=.5); net.set_connectivity(edges['We']*.2,edges['Wi']*.2)
        dec = FiniteMotorTranslator(manifest,weights); enc = ChannelSensoryEncoder(groups,len(roots))
        if dec.DT!=NEURAL_DT or dec.BINS!=((0,5),(5,10),(10,20),(20,40),(40,80)): raise RuntimeError('FIR changed')
        d = mj.MjData(m); d.qpos[:] = q  # Sole pose write.
        if d.qvel.any() or d.act.any() or d.ctrl.any() or d.time or d.xfrc_applied.any() or d.qfrc_applied.any() or dec.history.any(): raise RuntimeError('noncold')
        neural_arrays, neural_scalars, neural_configs = {},{},{}
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
        atomic_json(out/'INITIAL_STATE_HASHES.json',initial_hashes)
        atomic_json(out/'SETUP.json',dict(neurons=len(roots),leg_order=list(LEGS),weight_native=weight_native,timing=timing,neural_parameters=dataclasses.asdict(ConductanceParams()),mapping_digest=dec.mapping_digest,source_hashes=identities))
        atomic_npz(dest/'WEIGHTS.npz',weights=weights)
        atomic_json(out/'WEIGHT_SOURCE_AUDIT.json',dict(source=str(BASELINE),trajectory_sha256=sha(BASELINE),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),dtype=str(weights.dtype),shape=list(weights.shape),exact_copy=True,arithmetic=False,update=False,deployment=False))
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
        def scores(data):
            result = score_formal(data,weight_native)
            for phase,span in support_spans(data,weight_native).items(): result[phase]['longest_support_ms'] = span
            proxy = objective.proxy(data,scales)
            if set(proxy)!= {'total','parts'} or set(proxy['parts'])!=set(GATE_KEYS): raise ValueError('proxy schema')
            result.update(optimization_loss=float(proxy['total']),proxy_parts={k:float(proxy['parts'][k]) for k in GATE_KEYS})
            if not np.isfinite([result['loss'],result['optimization_loss'],*result['proxy_parts'].values()]).all(): raise ValueError('nonfinite score')
            return result
        historical = scores(baseline); score = scores(arrays)
        violations = np.asarray(objective.violations(arrays),float); scale_array = np.asarray([scales[k] for k in GATE_KEYS])
        if violations.shape!=(4001,6) or not np.isfinite(violations).all() or np.any(violations<0): raise ValueError('invalid violations')
        atomic_npz(dest/'PROXY_SERIES.npz',time=arrays['time'],phase=arrays['phase'],gate_keys=np.asarray(GATE_KEYS),scales=scale_array,violations=violations,contributions=(violations/scale_array)**2)
        record = dict(name=ARM,**score,trajectory_path=str(dest/'TRAJECTORY.npz'),trajectory_sha256=sha(dest/'TRAJECTORY.npz'),fresh_simulation=True,
            weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),simulation_calls=dict(neural_ticks=800,physics_steps=4000,explicit_forward=4001),acceptance_updates=False,deployment=False)
        atomic_json(dest/'RESULT.json',record)
        atomic_json(out/'PHYSICAL_DESCRIPTIVE_COMPARISON.json',dict(baseline=historical,current=record,original_physical_acceptance=acceptance(historical,record),descriptive_only=True,acceptance_updates=False,deployment=False))
        if reserved!=caps or completed!=caps or ledger['active'] is not None: raise RuntimeError('exact cap ledger mismatch')
        if identities!={str(p):sha(p) for p in paths} or any(sha(p)!=h for p,h in initial_hashes.items()): raise RuntimeError('input/initial evidence changed')
        invariants(); end_cb = callback_snapshot(mj)
        if end_cb!=before_cb: raise RuntimeError('callbacks changed')
        atomic_json(out/'CALLBACKS_END.json',end_cb)
        atomic_json(out/'MODEL_END_CHECK.json',dict(passed=True,snapshot=numeric_model_snapshot(m),matches_after=True,s92_after_equal=True,callbacks_none=True,source_hashes_unchanged=True))
        atomic_json(out/'SUMMARY.json',dict(status='ONE_TICK_CONSUMPTION_ARM_COMPLETE',evaluations=[record],fresh_evaluations=1,ledger=ledger,timing=timing,
            initial_state_hashes=initial_hashes,acceptance_updates=False,selection=False,update=False,deployment=False,behavior_claim='NONE',physical_units='NOT_ESTABLISHED',automatic_resume=False))
    except BaseException as exc:
        atomic_json(out/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),ledger=ledger,automatic_resume=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    args = parser().parse_args()
    if args.offline_check:
        offline_check()
    else:
        run(args)
