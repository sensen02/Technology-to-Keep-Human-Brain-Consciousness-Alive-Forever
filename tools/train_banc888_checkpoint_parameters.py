"""S4 finite conditional parameter fit. Explicit admission; exact restore gate first.
No tier construction, prewarm replay, controller, model edits, or production deployment.
"""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, hashlib, argparse, traceback, time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
ADMISSION = Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
OUT = Path('/home/sensen/Desktop/engine修复验收/s4-banc888-closedloop-parameter-fit/20261101T222509Z')
MODEL_SHA = '3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
DT = .0000125
LEGS = ['LF','LM','LH','RF','RM','RH']
PENALTY = 1000000.

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def dump(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert)+'\n')

def load_np(path):
    with np.load(path) as f: return {k:f[k].copy() for k in f.files}

def array_hash(arrays):
    h = hashlib.sha256()
    for k, v in sorted(arrays.items()):
        a = np.asarray(v); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = dict(stage='S4 conditional training at release, NOT autonomous success or full-prewarm candidate validation',
        source=str(SOURCE),admission=str(ADMISSION),model_sha256=MODEL_SHA,
        restore='One complete saved dynamic initial state per trial; actual MjData integration fields, warmed network, decoder rates/pending/clocks. Extra mjSTATE_HISTORY must have size zero or gate aborts. No intermediate body writes.',
        gate=dict(duration_s=.020,body_steps=1600,neural_ticks=40,body_endpoints=40,command_windows=20,
                  state_tolerance_abs=1e-10,spikebits_commands_rates='exact',
                  fields=['qpos','qvel','act','ctrl','foot_force','com'],failure='First differences and counts saved; no training unless all pass; no original 2s rerun'),
        schedule=dict(body_dt_s=DT,neural_dt_s=.0005,command_dt_s=.001,sample_dt_s=.0005,
                      observation='Original entry mj_forward once; every mj_step followed by live mj_forward then native loss record, full state sampled .5ms. Read-only observer has no forward/Jac. Initial observation excluded from loss means.'),
        parameters=dict(names=['exc_gain','inh_gain','threshold_shift_mV'],gain_bounds=[.5,2.],theta_bounds_mV=[-3,3],
                        z='[log(exc)/log(2), log(inh)/log(2), theta/3] clipped [-1,1]',
                        fixed='roots, sparse masks/EI, delays, intrinsics, muscle model, reference posture, sensory and motor rules, gravity',
                        intervention='Only future synaptic event strength and effective threshold change; inherited ge/gi/ring/state/rates/pending preserved'),
        training=dict(duration_s=.050,body_steps=4000,neural_ticks=100,updates=15,max_trials=46,seed=1413,
                      order='neutral baseline, then plus/minus/updated-center per update; baseline1 + 15*3 = 46, gate separate',
                      a='0.1/(k+1)^0.602',c='0.05/(k+1)^0.101',
                      gradient='(Lplus-Lminus)/(2*c)*delta (nominal c even if projected); center=clip(z-a*gradient,-1,1)',
                      select='Minimum completed finite successful TRAINING loss, first trial wins ties; never select by validation or another metric'),
        loss=dict(h0='Release COMz from restored body entry forward',W='sum(compiled body_mass)*abs(gravity_z)',
                  terms=['mean(((COMz-h0)/h0)^2)','mean((sum(signed foot_Fz)/W-1)^2)',
                         'mean(nonfoot_positive_signed_Fz/W)', '.01*mean(root_angle_delta_rad^2)',
                         '.001*mean(ctrl^2)', '.001*sum(log(exc)^2+log(inh)^2+(theta/3)^2)'],
                  sampling='All five physical terms use every 12.5us body endpoint after existing liveforward; save native COMz, per-foot signed Fz, per-contact positive nonfoot sum, rotation angle squared, ctrl mean square. No extra forward.',
                  nonfoot='Sum max(signed floor-contact Fz,0) over individual NONFOOT contacts, then mean at every 12.5us endpoint; no candidate count or gap proxy',
                  root_angle='2*acos(clip(abs(dot(current free-root quaternion,release quaternion)),0,1))',
                  failed_loss=PENALTY,hard_fail='nonfinite state/network/metrics, MuJoCo warnings, external forces, ctrl outside [0,1], changed source/model/mask; stop that trial, preserve partial arrays and finite penalty, no retry'),
        generalization=dict(duration_s=.150,rollouts=1,compare='Cached actual original new-interface 150ms endpoints; no neutral physical rerun',
                            interpretation='Conditional unseen time extension. Report height, signed foot support, nonfoot positive support, first nonfoot, root displacement and finite support duration. No walking/standing claim.',
                            sufficient_improvement='Training loss strictly lower AND 150ms loss strictly lower AND first nonfoot later by >=.5ms AND mean signed foot_Fz/W higher AND mean nonfoot-positive-Fz/W lower. Descriptive support criterion foot_Fz/W>=.8 and nonfoot positive=0; no four-feet gate.'),
        resources='0 tiers, <=1 model compile in this process; gate+46 train+1 validation if admitted. Count actual fresh data/network/steps/forward/contactForce. No full prewarm or extra parameter/duration trials.',
        repairs='If gate fails, retain failure. Any restore/schedule correction must have a separately preregistered timestamp attempt and executed/delivered source hashes before execution.',
        importance={'runner,preregistration,source/checkpoint identities,gate,trial arrays/history,selection,validation,verification':'核心必需','REPORT,manifest,inventory,cleanup':'重要辅助'},
        cleanup='No known dispensable task outputs; frozen source and failed scientific attempts retained')
    p = OUT/'PREREGISTRATION.json'
    if p.exists(): assert json.loads(p.read_text()) == spec
    else: dump(p,spec)
    return spec


def main():
    prepare()
    if (OUT/'EXECUTION.json').exists(): raise RuntimeError('Refuse overwrite or rerun')
    import mujoco
    from scipy.sparse import load_npz
    from engine.embodied.runtime_pool_guard import load_runtime_pool
    from engine.embodied.motor_output import build_motor_manifest, MotorOutput
    from engine.embodied.sensory_channels import channel_index_map, ChannelSensoryEncoder, ChannelConfig
    from engine.embodied.continuous_handover import run_neural_continuous, state_snapshot
    from engine.neural_cond import ConductanceNetwork
    from engine.neural_tuning import FrozenConnectomeTuning, TuningParameters, STATE_FIELDS
    counters = dict(model_compile=0,tier_build=0,data_instances=0,network_instances=0,body_steps=0,
                    neural_steps=0,forward=0,contact_force=0,guard_calls=0,training_trials=0,gate_trials=0,validation_trials=0)
    start = time.time(); error = None; histories=[]; result={}
    paths = [Path(__file__), SOURCE/'RUN/MODEL.xml', SOURCE/'RUN/HANDOVER_AFTER.npz', SOURCE/'RUN/NEURAL_BOUNDARY.npz',
             SOURCE/'PREWARM_FINAL_STATE.npz', SOURCE/'PREWARM_CLOCK.json', SOURCE/'ACTUAL_We.npz',SOURCE/'ACTUAL_Wi.npz',
             SOURCE/'IDENTITIES.npz',SOURCE/'RUN/MAPPING.json',SOURCE/'SENSORY_GROUPS.json',SOURCE/'RUN/BODY.npz',
             SOURCE/'RUN/NEURAL.npz',SOURCE/'RUN/NEURAL_RECORDS.json',SOURCE/'NEURAL_RASTER.npz',
             ADMISSION/'VERSION_CONTRACT.json',ADMISSION/'MN_POOLS_SNAPSHOT888.json',ADMISSION/'SENSORY_LOOKUP_SNAPSHOT888.json']
    paths += [ROOT/p for p in ['engine/neural_tuning.py','engine/neural_cond.py','engine/embodied/continuous_handover.py',
            'engine/embodied/motor_output.py','engine/embodied/sensory_channels.py','engine/embodied/closure_clock.py',
            'engine/embodied/runtime_pool_guard.py','check_pool_version_contract.py','run_neural_muscle_closure.py']]
    hashes={str(p):sha(p) for p in paths}
    dump(OUT/'SOURCE_IDENTITY.json',dict(head='2d34f0ea3e131c1f2374e55c9b5297493c26be0a',sha256=hashes))
    (OUT/'EXECUTION_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    for sourcepath in paths:
        if sourcepath.suffix=='.py' and sourcepath.is_relative_to(ROOT):
            target=OUT/'SOURCES'/sourcepath.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(sourcepath.read_bytes())
    import importlib.metadata
    dump(OUT/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,packages={p:importlib.metadata.version(p) for p in ['mujoco','numpy','scipy']},
        thread_env={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE']}))
    contract=json.loads((ADMISSION/'VERSION_CONTRACT.json').read_text())
    assert sha(ADMISSION/'SENSORY_LOOKUP_SNAPSHOT888.json')==contract['sensory_lookup_sha256']
    pool=load_runtime_pool(ADMISSION/'MN_POOLS_SNAPSHOT888.json',version_contract=contract);counters['guard_calls']+=1
    assert sha(SOURCE/'RUN/MODEL.xml')==MODEL_SHA
    physical=load_np(SOURCE/'RUN/HANDOVER_AFTER.npz');boundary=load_np(SOURCE/'RUN/NEURAL_BOUNDARY.npz')
    prewarm=load_np(SOURCE/'PREWARM_FINAL_STATE.npz');clock=json.loads((SOURCE/'PREWARM_CLOCK.json').read_text())
    for k in boundary: assert np.array_equal(boundary[k],prewarm[k]),k
    assert (int(prewarm['steps']),float(prewarm['time_s']),int(prewarm['ring_pos']))==(2200,1.1,0)
    assert clock['net_scalars']['t_ms']==1100. and clock['decoder_elapsed_s']==1.1 and clock['decoder_interval']==1099
    ids=load_np(SOURCE/'IDENTITIES.npz');roots=ids['root_ids'];n=len(roots)
    groups,_=channel_index_map(n,roots,lookup_path=ADMISSION/'SENSORY_LOOKUP_SNAPSHOT888.json')
    def plain(v):
        if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
        if isinstance(v,(np.ndarray,list,tuple)):return [plain(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    assert plain(groups)==json.loads((SOURCE/'SENSORY_GROUPS.json').read_text())
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    spec=_load_mjcf(SOURCE/'RUN/MODEL.xml')
    dump(OUT/'ASSET_IDENTITY.json',dict(loader='flygym.compose.fly.musculoskeletal._load_mjcf',
        mesh_sha256={str(mesh.file):sha(mesh.file) for mesh in spec.meshes if mesh.file}))
    model=spec.compile();counters['model_compile']+=1
    assert model.jnt_type[0]==mujoco.mjtJoint.mjJNT_FREE and model.opt.timestep==float(model.opt.timestep)
    assert abs(model.opt.timestep-DT)<1e-15
    history_size=mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_HISTORY)
    integration_size=mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION)
    dump(OUT/'INTEGRATION_AUDIT.json',dict(mujoco=mujoco.__version__,integration_size=integration_size,history_size=history_size,
        saved_fields=list(physical),extra_history_recoverable=history_size==0,gravity=model.opt.gravity,
        integrator=int(model.opt.integrator),release_time=physical['time']))
    if history_size:raise RuntimeError('Checkpoint lacks nonempty MuJoCo HISTORY; cannot restore faithfully')
    manifest=build_motor_manifest(pool,roots,[model.actuator(i).name for i in range(model.nu)])
    assert manifest==json.loads((SOURCE/'RUN/MAPPING.json').read_text())
    dump(OUT/'MAPPING.json',manifest);dump(OUT/'SENSORY_GROUPS.json',groups)
    joints={};geom={}
    for leg in ['lf','lm','lh','rf','rm','rh']:
        js=[j for j in range(model.njnt) if model.joint(j).name.lower().startswith('joint_'+leg)]
        tib=[j for j in js if 'tibia' in model.joint(j).name.lower()];assert len(js)==7 and len(tib)==1
        joints[leg]=dict(q=[int(model.jnt_qposadr[j]) for j in js],tib_q=int(model.jnt_qposadr[tib[0]]),tib_v=int(model.jnt_dofadr[tib[0]]))
    for g in range(model.ngeom):
        nm=model.geom(g).name.lower();bn=model.body(int(model.geom_bodyid[g])).name.lower()
        matches=[l for l in joints if 'tars' in nm+bn and (nm.startswith(l) or bn.startswith(l))]
        if len(matches)==1:geom[g]=matches[0]
    assert set(geom.values())==set(joints)
    dump(OUT/'BODY_IDENTITIES.json',dict(joints=joints,geom_leg=geom,joint_names=[model.joint(j).name for j in range(model.njnt)],
         geom_names=[model.geom(g).name for g in range(model.ngeom)],actuator_names=[model.actuator(i).name for i in range(model.nu)]))
    # pybind fields are descriptors; dir enumerates all compiled numerical arrays.
    model_arrays={k:getattr(model,k).copy() for k in dir(model) if isinstance(getattr(model,k),np.ndarray)}
    model_arrays['gravity']=model.opt.gravity.copy()
    model_hash=array_hash(model_arrays);np.savez_compressed(OUT/'COMPILED_MODEL_ARRAYS.npz',**model_arrays)
    body=load_np(SOURCE/'RUN/BODY.npz');reference=body['qpos'][0].copy();release_index=int(np.argmin(abs(body['time']-float(physical['time']))))
    assert abs(body['time'][release_index]-float(physical['time']))<1e-10
    baseline={k:body[k][release_index:release_index+301].copy() for k in body}
    np.savez_compressed(OUT/'CACHED_NEUTRAL_150MS.npz',**baseline)
    weights=[load_npz(SOURCE/('ACTUAL_'+name+'.npz')) for name in ['We','Wi']]
    tuning=FrozenConnectomeTuning(roots,*weights,dt_ms=.5,threshold_offset_mV=boundary['threshold_offset_mV'],gain_bounds=(.5,2.),threshold_bounds_mV=(-3.,3.))
    snapshot=dict(identity=tuning.contract()['identity'],root_ids=roots.copy(),steps=2200,t_ms=1100.,ring_i=0,
                  **{k:boundary[k].copy() for k in STATE_FIELDS})
    dump(OUT/'TUNING_CONTRACT.json',tuning.contract())
    np.savez_compressed(OUT/'INITIAL_CHECKPOINT.npz',**{'body_'+k:v for k,v in physical.items()},**{'neural_'+k:v for k,v in boundary.items()},reference_qpos=reference)
    checkpoint_hash=array_hash({**{'body_'+k:v for k,v in physical.items()},**{'neural_'+k:boundary[k] for k in STATE_FIELDS},'rates':boundary['decoder_rates'],'pending':boundary['pending']})
    weight=float(model.body_mass.sum()*abs(model.opt.gravity[2]));h0=float(baseline['com'][0,2]);q0=physical['qpos'][3:7].copy()
    context=dict(checkpoint_hash=checkpoint_hash,model_array_hash=model_hash,h0=h0,W=weight,q0=q0,sample_dt=.0005)
    dump(OUT/'LOSS_CONTEXT.json',context)
    orig_step=mujoco.mj_step;orig_forward=mujoco.mj_forward;orig_force=mujoco.mj_contactForce;orig_ns=ConductanceNetwork.step
    active_raster=[]
    def st(*a,**kw):counters['body_steps']+=1;return orig_step(*a,**kw)
    def fw(*a,**kw):counters['forward']+=1;return orig_forward(*a,**kw)
    def cf(*a,**kw):counters['contact_force']+=1;return orig_force(*a,**kw)
    def ns(net,*a,**kw):
        counters['neural_steps']+=1;sp=orig_ns(net,*a,**kw);active_raster.append(sp.copy())
        if any(not np.isfinite(v).all() for v in vars(net).values() if isinstance(v,np.ndarray)):raise FloatingPointError('nonfinite neural')
        return sp
    mujoco.mj_step=st;mujoco.mj_forward=fw;mujoco.mj_contactForce=cf;ConductanceNetwork.step=ns
    def read_support(d):
        foot=np.zeros((6,3));nf=np.zeros(3);positive=0.;nc=0;f=np.zeros(6)
        for j in range(d.ncon):
            co=d.contact[j];a,b=int(co.geom1),int(co.geom2)
            pa=model.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=model.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
            if not(pa or pb):continue
            g=b if pa else a;cf(model,d,j,f);force=(co.frame.reshape(3,3).T@f[:3])*(1 if pa else -1)
            name=model.geom(g).name;leg=next((i for i,l in enumerate(LEGS) if name.startswith(l) and 'Tars' in name),None)
            if leg is None:nf+=force;nc+=1;positive+=max(float(force[2]),0.)
            else:foot[leg]+=force
        return foot,nf,positive,nc
    def components(a,z):
        foot=a['native_foot_Fz'].sum(axis=1)/weight
        lg=np.asarray(z)[:2]*np.log(2.)
        return dict(com_height=float(np.mean(((a['native_COMz']-h0)/h0)**2)),foot_support=float(np.mean((foot-1)**2)),
                    nonfoot_positive=float(np.mean(a['native_nonfoot_positive_Fz']/weight)),root_rotation=float(.01*np.mean(a['native_root_angle_squared'])),
                    control=float(.001*np.mean(a['native_ctrl_mean_squared'])),regularization=float(.001*(np.sum(lg**2)+float(z[2])**2)))
    def trial(label,z,duration,role):
        nonlocal active_raster
        path=OUT/label;path.mkdir();active_raster=[];rows=[];native=[];records=[];before_counts=counters.copy();tstart=time.time();failure=None;net=None
        counters[role+'_trials']+=1
        z=np.asarray(z,dtype=float);x=[float(2.**z[0]),float(2.**z[1]),float(3*z[2])]
        dump(path/'PARAMETERS.json',dict(z=z,x=x,names=['exc_gain','inh_gain','threshold_shift_mV'],role=role,checkpoint_hash=checkpoint_hash))
        try:
            d=mujoco.MjData(model);counters['data_instances']+=1
            for k,v in physical.items():
                if k=='time':d.time=float(v)
                else:getattr(d,k)[:]=v
            assert all(np.array_equal(state_snapshot(d)[k],v) for k,v in physical.items())
            net=tuning.spawn(TuningParameters(*x),snapshot=snapshot);counters['network_instances']+=1
            encoder=ChannelSensoryEncoder(groups,n,ChannelConfig());decoder=MotorOutput(manifest)
            decoder.rates[:]=boundary['decoder_rates'];decoder.elapsed_s=1.1;decoder.interval=1099
            actual_hash=array_hash({**{'body_'+k:v for k,v in state_snapshot(d).items()},**{'neural_'+k:getattr(net,k) for k in STATE_FIELDS},'rates':decoder.rates,'pending':boundary['pending']})
            assert actual_hash==checkpoint_hash
            initial_integration=np.empty(integration_size);mujoco.mj_getState(model,d,initial_integration,mujoco.mjtState.mjSTATE_INTEGRATION)
            np.savez_compressed(path/'INITIAL_INTEGRATION_STATE.npz',state=initial_integration)
            obs_count=0;thorax=model.body('Thorax').id
            def observe(m,dd):
                nonlocal obs_count
                f,nf,pos,nc=read_support(dd)
                if obs_count>0:
                    angle=2*np.arccos(np.clip(abs(float(dd.qpos[3:7]@q0)),0,1))
                    native.append(dict(native_time=float(dd.time),native_COMz=float(dd.subtree_com[thorax,2]),
                        native_foot_Fz=f[:,2].copy(),native_nonfoot_Fz=float(nf[2]),native_nonfoot_positive_Fz=pos,
                        native_root_angle_squared=float(angle**2),native_root_quaternion=dd.qpos[3:7].copy(),
                        native_ctrl_mean_squared=float(np.mean(dd.ctrl**2))))
                if obs_count%40==0:
                    rows.append(dict(time=float(dd.time),qpos=dd.qpos.copy(),qvel=dd.qvel.copy(),act=dd.act.copy(),ctrl=dd.ctrl.copy(),
                        qacc_warmstart=dd.qacc_warmstart.copy(),com=dd.subtree_com[thorax].copy(),foot_force=f,nonfoot_force=nf,
                        nonfoot_positive_Fz=pos,nonfoot_count=nc,warning=dd.warning.number.copy(),qfrc_applied=dd.qfrc_applied.copy(),xfrc_applied=dd.xfrc_applied.copy()))
                obs_count+=1
                if np.any(dd.ctrl<0) or np.any(dd.ctrl>1):raise AssertionError('ctrl bounds')
            ret=run_neural_continuous(model,d,dict(n=n),manifest,groups,duration,sensory_reference_qpos=reference,
                joints=joints,geom_leg=geom,body_dt_s=DT,neural_dt_s=.0005,command_dt_s=.001,
                body_observer=observe,interval_observer=records.append,retain_currents=True,
                runtime_state=dict(runtime=(net,encoder,decoder),pending=boundary['pending'].copy(),body_time=float(d.time)))
            assert len(rows)==round(duration/.0005)+1
            assert all(np.array_equal(ret['handover_before'][k],physical[k]) for k in physical)
            assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
            for a,b in zip((net.We,net.Wi),weights):
                assert np.array_equal(a.indices,b.indices) and np.array_equal(a.indptr,b.indptr)
            final={k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)}
            final.update(decoder_rates=decoder.rates,pending=ret['runtime_state']['pending'],steps=net.steps,t_ms=net.t_ms,ring_i=net._ring_i)
            np.savez_compressed(path/'FINAL_NEURAL_STATE.npz',**final)
        except BaseException: failure=traceback.format_exc()
        arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]} if rows else {}
        if native: arrays.update({k:np.asarray([r[k] for r in native]) for k in native[0]})
        arrays.update(spikes=np.asarray(active_raster,bool),commands=np.asarray([r['applied_command'] for r in records]),
                      decoded_next=np.asarray([r['decoded_next_command'] for r in records]),rates=np.asarray([r['decoder']['rates_hz'] for r in records]),
                      neural_counts=np.asarray([np.asarray(active_raster[2*i:2*i+2]).sum(axis=0) for i in range(len(records))]),
                      sensory_current=np.asarray([r['sensory_current_nA'] for r in records]))
        if failure is None:
            try:
                terms=components(arrays,z)
                assert all(np.isfinite(v) for v in terms.values())
                assert all(np.isfinite(v).all() for v in arrays.values())
                loss=sum(terms.values())
            except BaseException:failure=traceback.format_exc()
        if failure is not None:terms={};loss=PENALTY
        np.savez_compressed(path/'TRAJECTORY.npz',**arrays);dump(path/'EVENTS.json',records)
        summary=dict(label=label,role=role,z=z,x=x,loss=loss,terms=terms,failed=failure is not None,error=failure,
                     checkpoint_hash=checkpoint_hash,counts={k:counters[k]-before_counts[k] for k in counters},wall_seconds=time.time()-tstart)
        dump(path/'RESULT.json',summary)
        print('TRIAL',label,'loss',loss,'failed',bool(failure),'seconds',round(time.time()-tstart,2),flush=True)
        return summary,arrays
    try:
        gate,ga=trial('GATE_20MS',np.zeros(3),.020,'gate')
        checks={};first=[]
        for field,oldfield in [('qpos','qpos'),('qvel','qvel'),('act','activation'),('ctrl','command'),('foot_force','foot_force'),('com','com')]:
            a=ga.get(field,np.empty(0))[1:];b=baseline[oldfield][1:41]
            same=a.shape==b.shape
            diff=float(np.max(abs(a-b))) if same else None
            checks[field]=dict(bitwise=bool(same and np.array_equal(a,b)),max_abs=diff,pass_gate=bool(same and diff<=1e-10))
            if same and np.any(a!=b):
                idx=tuple(np.argwhere(a!=b)[0]);first.append(dict(field=field,index=idx,actual=float(a[idx]),expected=float(b[idx])))
        oldr=load_np(SOURCE/'NEURAL_RASTER.npz')['spikes'][2200:2240]
        oldn=load_np(SOURCE/'RUN/NEURAL.npz')
        oldrecords=json.loads((SOURCE/'RUN/NEURAL_RECORDS.json').read_text())[:20]
        for field,a,b in [('spikes',ga['spikes'],oldr),('commands',ga['commands'],oldn['commands'][:20]),
                           ('rates',ga['rates'],np.asarray([r['decoder']['rates_hz'] for r in oldrecords])),
                           ('decoded_next',ga['decoded_next'],np.asarray([r['decoded_next_command'] for r in oldrecords]))]:
            same=a.shape==b.shape and np.array_equal(a,b);checks[field]=dict(exact=bool(same),pass_gate=bool(same))
            if a.shape==b.shape and not same:
                idx=tuple(np.argwhere(a!=b)[0]);first.append(dict(field=field,index=idx,actual=float(a[idx]),expected=float(b[idx])))
        passed=not gate['failed'] and all(c['pass_gate'] for c in checks.values())
        dump(OUT/'GATE_RESULT.json',dict(passed=passed,checks=checks,first_differences=first,actual_counts=gate['counts'],source='actual new-interface RUN physical trajectory'))
        if not passed:
            result=dict(status='GATE_FAILED_NO_TRAINING',gate_passed=False)
        else:
            rng=np.random.default_rng(1413);z=np.zeros(3)
            base,_=trial('TRAIN_000_NEUTRAL',z,.050,'training');histories.append(base)
            for k in range(15):
                a=.1/(k+1)**.602;c=.05/(k+1)**.101;delta=rng.choice([-1.,1.],size=3)
                plus,_=trial('TRAIN_%03d_PLUS'%(3*k+1),np.clip(z+c*delta,-1,1),.050,'training');histories.append(plus)
                minus,_=trial('TRAIN_%03d_MINUS'%(3*k+2),np.clip(z-c*delta,-1,1),.050,'training');histories.append(minus)
                gradient=(plus['loss']-minus['loss'])/(2*c)*delta
                oldz=z.copy();z=np.clip(z-a*gradient,-1,1)
                center,_=trial('TRAIN_%03d_CENTER'%(3*k+3),z,.050,'training');histories.append(center)
                dump(OUT/('UPDATE_%02d.json'%k),dict(k=k,a=a,c=c,delta=delta,old_z=oldz,gradient=gradient,new_z=z,
                    plus=plus['label'],minus=minus['label'],center=center['label']))
                dump(OUT/'TRAINING_HISTORY.json',histories)
            successful=[h for h in histories if not h['failed']]
            if not successful:raise RuntimeError('All training trials failed; no candidate')
            best=min(successful,key=lambda h:h['loss'])
            dump(OUT/'SELECTED_CANDIDATE.json',dict(selected=best,criterion='Minimum successful completed training loss, first wins ties',training_trials=len(histories),conditional=True,requires_full_fixed_parameter_prewarm_validation=True))
            val,va=trial('VALIDATION_150MS',best['z'],.150,'validation')
            # Cached neutral nonfoot signed sum equals sum of positive individual supports
            # only if all floor contact vertical supports are nonnegative; recomputed separately from original contact rows below.
            native_positive=np.zeros(12000);seen=np.zeros(12000,bool);cached_native=[]
            for chunkpath in sorted((SOURCE/'NATIVE').glob('chunk_*.npz')):
                chunk=load_np(chunkpath)
                mask=(chunk['step']>104000)&(chunk['step']<=116000)&(chunk['kind']==0)
                for si in np.flatnonzero(mask):
                    ci=int(chunk['step'][si]-104001);contacts=chunk['contact_rows'][chunk['contact_offsets'][si]:chunk['contact_offsets'][si+1]]
                    floors=set(np.flatnonzero(model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE).tolist());positive=0.
                    for co in contacts:
                        g1,g2=int(co[0]),int(co[1])
                        if (g1 in floors or g2 in floors) and int(co[4])<0:
                            positive+=max(float(co[17])*(1 if g1 in floors else -1),0.)
                    native_positive[ci]=positive;seen[ci]=True
                    quat=chunk['root'][si,3:7];angle=2*np.arccos(np.clip(abs(float(quat@q0)),0,1))
                    cached_native.append(dict(native_time=float(chunk['time'][si]),native_COMz=float(chunk['com'][si,2]),
                        native_foot_Fz=chunk['foot_force'][si,:,2].copy(),native_nonfoot_Fz=float(chunk['nonfoot_force'][si,2]),
                        native_nonfoot_positive_Fz=positive,native_root_angle_squared=float(angle**2),native_root_quaternion=quat.copy(),
                        native_ctrl_mean_squared=float(np.mean(chunk['ctrl'][si]**2))))
            assert seen.all() and len(cached_native)==12000
            cached=dict(qpos=baseline['qpos'],com=baseline['com'],foot_force=baseline['foot_force'],ctrl=baseline['command'])
            cached.update({k:np.asarray([r[k] for r in cached_native]) for k in cached_native[0]})
            neutral_terms=components(cached,np.zeros(3));neutral_loss=sum(neutral_terms.values())
            np.savez_compressed(OUT/'CACHED_NEUTRAL_LOSS_INPUTS.npz',**cached)
            def metrics(a):
                foot=a['native_foot_Fz'].sum(axis=1)/weight;nf=a['native_nonfoot_positive_Fz']/weight
                idx=np.flatnonzero(nf>0);supported=(foot>=.8)&(nf==0)
                prefix=next((i for i,v in enumerate(supported) if not v),len(supported))
                return dict(mean_foot_Fz_W=float(foot.mean()),mean_nonfoot_positive_Fz_W=float(nf.mean()),
                    first_nonfoot_s=None if not len(idx) else float((idx[0]+1)*DT),
                    supported_sample_time_s=float(supported.sum()*DT),initial_continuous_support_s=float(prefix*DT),
                    final_com_z=float(a['com'][-1,2]),min_com_z=float(a['com'][1:,2].min()),
                    forward_displacement=float(a['qpos'][-1,0]-physical['qpos'][0]))
            nm=metrics(cached);vm=metrics(va) if not val['failed'] else None
            improved=False
            if vm and nm['first_nonfoot_s'] is not None and vm['first_nonfoot_s'] is not None:
                improved=bool(best['loss']<base['loss'] and val['loss']<neutral_loss and vm['first_nonfoot_s']>=nm['first_nonfoot_s']+.0005-1e-12 and vm['mean_foot_Fz_W']>nm['mean_foot_Fz_W'] and vm['mean_nonfoot_positive_Fz_W']<nm['mean_nonfoot_positive_Fz_W'])
            result=dict(status='COMPLETED_CONDITIONAL_FIT',gate_passed=True,training_trials=len(histories),selected=best['label'],
                baseline_training_loss=base['loss'],selected_training_loss=best['loss'],validation_loss=val['loss'],neutral_validation_loss=neutral_loss,
                neutral_validation_terms=neutral_terms,neutral_metrics=nm,candidate_metrics=vm,sufficient_improvement=improved,
                autonomous_standing_or_walking=False,full_prewarm_fixed_parameter_candidate_run=False)
        dump(OUT/'RESULT.json',result)
    except BaseException:error=traceback.format_exc()
    finally:
        mujoco.mj_step=orig_step;mujoco.mj_forward=orig_forward;mujoco.mj_contactForce=orig_force;ConductanceNetwork.step=orig_ns
        unchanged={p:sha(p)==h for p,h in hashes.items()}
        final_model={k:getattr(model,k).copy() for k in dir(model) if isinstance(getattr(model,k),np.ndarray)};final_model['gravity']=model.opt.gravity.copy()
        model_unchanged=array_hash(final_model)==model_hash
        if not all(unchanged.values()) or not model_unchanged:error=(error or '')+'Source or model identity changed'
        dump(OUT/'EXECUTION.json',dict(counters=counters,wall_seconds=time.time()-start,error=error,exit_code=int(error is not None),source_unchanged=unchanged,compiled_model_unchanged=model_unchanged))
    if error:raise RuntimeError(error)
    print('FIT_FINISHED',result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare();print('PREPARED_NO_SIMULATION',str(OUT),flush=True)
    else:main()
