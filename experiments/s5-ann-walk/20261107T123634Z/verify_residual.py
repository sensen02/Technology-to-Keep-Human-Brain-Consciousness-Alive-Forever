"""Independent two-ANN/sensory audit with unchanged frozen physics acceptance.
Parent owns replay execution. No training/controller/sensory-runtime imports here.
Full effective verifier and provenance are saved before any requested replay.
"""
import argparse,hashlib,importlib.util,inspect,json,sys,types
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
HERE=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/verification_method')
VERIFY=ROOT/'experiments/s5-ann-walk/20261107T073258Z/verify_walk.py'
ASSESS=ROOT/'experiments/s5-ann-walk/20261107T080221Z/assess_walk.py'
FROZEN_BASE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T111149Z/MODEL_VELOCITY.npz')
BASE_HASH='08732229ce06ba201f88fd67d29aefdc50ccdf95e5a522a5125986748c6245f7'
VHASH='78e1b838f0a150c5fbae2604cbad59ce49809b486066f072e05f55ba9e1e862e'
AHASH='c01bcdfa1b526693bb4a8f7645c8f9517a1d284f4f2832855ed8fab3a4ebcf28'
JOINTS=('joint_RHCoxa_pitch','joint_RHTrochanter_yaw','joint_LHCoxa_pitch','joint_LMCoxa_yaw','joint_RMCoxa_yaw','joint_RHTrochanter_roll')
W=24.446354755878005

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def error(a,b):
    a,b=np.asarray(a),np.asarray(b);require(a.shape==b.shape,'shape mismatch '+str((a.shape,b.shape)));return float(abs(a.astype(float)-b.astype(float)).max()) if a.size else 0.
def encode(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)
def stable(path,text):
    if path.exists():require(path.read_text()==text,'Preserve differing prior verifier method '+str(path))
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

def load_weights(path):
    require(sha(FROZEN_BASE)==BASE_HASH,'Frozen base ANN changed');base=load(FROZEN_BASE);weights=load(path)
    for key,value in base.items():require(key in weights and np.array_equal(weights[key],value),'Base model field changed: '+key)
    require(str(weights['base_sha256'].item())==BASE_HASH,'Residual declares wrong base hash')
    for key,shape in {'actor_w0':(128,477),'actor_b0':(128,),'actor_w1':(128,128),'actor_b1':(128,),'actor_w2':(6,128),'actor_b2':(6,),'indices':(6,)}.items():
        require(key in weights and weights[key].shape==shape,'Wrong actor field '+key)
        require(np.isfinite(weights[key]).all(),'Nonfinite actor field '+key)
    require(float(weights['delta_scale'])==.003,'Undeclared residual delta scale')
    require(np.array_equal(weights['indices'],np.asarray(weights['indices'],int)) and len(set(weights['indices'].tolist()))==6,'Invalid residual indices')
    return weights

def features(plant,data,target,loads):
    """Independent declared183 formula, no time/worldpos/rootheight/reference."""
    q=data.qpos[plant.qa];v=data.qvel[plant.va];half=np.maximum(np.ptp(plant.m.jnt_range[plant.jids],axis=1)/2,.1)
    q0=plant.q0[plant.qa];R=data.xmat[plant.thorax].reshape(3,3)
    x=np.r_[(q-q0)/half,(np.asarray(target)-q)/half,v/10.,(np.asarray(target)-q0)/half,
             np.asarray(loads)/W,R.T@np.array([0.,0.,-1.]),R.T@data.qvel[:3],data.qvel[3:6]]
    require(x.shape==(183,) and np.isfinite(x).all(),'Malformed physical features')
    return np.clip(x,-10,10).astype(np.float32)

def positive_loads(mj,plant,data):
    loads=np.zeros(6)
    for i in range(data.ncon):
        c=data.contact[i];ga,gb=int(c.geom1),int(c.geom2)
        if not(plant.planes[ga] or plant.planes[gb]):continue
        leg=int(plant.feet[gb if plant.planes[ga] else ga])
        if leg<0:continue
        force=np.zeros(6);mj.mj_contactForce(plant.m,data,i,force)
        normal=float((c.frame.reshape(3,3).T@force[:3])[2])*(1 if plant.planes[ga] else -1)
        loads[leg]+=max(0.,normal)
    return loads

def observe(weights,history,target):
    x=(np.asarray(target,np.float32)-weights['center'])/weights['scale'];K=int(weights['K'])
    return np.repeat(x[None],K,axis=0) if history is None else np.concatenate((history[1:],x[None]),axis=0)

def predict(weights,history,sensory,target):
    lags=np.asarray(weights['lags'],int);flat=history[lags].reshape(-1).copy();inputs=np.r_[sensory,flat].astype(np.float32)
    h=np.tanh(inputs@weights['actor_w0'].T+weights['actor_b0']);h=np.tanh(h@weights['actor_w1'].T+weights['actor_b1'])
    mean=h@weights['actor_w2'].T+weights['actor_b2'];residual=np.tanh(mean)*float(weights['delta_scale'])
    z=flat.reshape(1,-1);z=np.tanh(z@weights['w0'].T+weights['b0']);z=np.tanh(z@weights['w1'].T+weights['b1'])
    base_next=weights['center']+weights['scale']*(z@weights['w2'].T+weights['b2'])[0]
    nextq=np.asarray(base_next,np.float64).copy();nextq[np.asarray(weights['indices'],int)]+=residual
    rate=np.clip((nextq-target)/.005,-30,30)
    return dict(features=sensory,input=inputs,history=flat,mean=mean,residual_target=residual,base_prediction=base_next,predicted_rate=rate)

def replay_features(mj,p,data,target,weights,history,trace,index):
    """Called before the command update on every independent replay policy tick."""
    history=observe(weights,history,target);loads=positive_loads(mj,p,data);sensory=features(p,data,target,loads)
    flat=history[np.asarray(weights['lags'],int)].reshape(-1);inputs=np.r_[sensory,flat].astype(np.float32)
    fields=dict(features=sensory,input=inputs,history=flat,actual_q=data.qpos[p.qa],actual_qvel=data.qvel[p.va],foot_load=loads,time=np.asarray(float(data.time)))
    return history,{key:error(value,trace[key][index]) for key,value in fields.items()}

def validate_native(model_path,arrays,cap):
    require(cap==30.,'Residual runtime requires native30rad/s cap');w=load_weights(model_path)
    sys.path.insert(0,str(BASE));import mujoco as mj
    from plant_probe import Plant
    from incremental_actuator import DEFAULT_MODEL_PATH
    require(sha(DEFAULT_MODEL_PATH)=='517cec0eed476f825535b56858b434c949bcbf8ed18ce954189689c784d1d2d1','Original full physical model changed')
    p=Plant();native=np.minimum(-p.m.actuator_ctrlrange[:,0],p.m.actuator_ctrlrange[:,1]);fl=p.m.actuator_forcelimited.astype(bool)
    native[fl]=np.minimum(native[fl],np.minimum(-p.m.actuator_forcerange[fl,0],p.m.actuator_forcerange[fl,1]))
    require(np.array_equal(arrays['limits'],native),'Native torque caps differ')
    require(np.allclose(arrays['kp'],native/.2*128,rtol=1e-12,atol=1e-12),'Wrong gain128 kp')
    require(np.allclose(arrays['kd'],native*.005*np.sqrt(128),rtol=1e-12,atol=1e-12),'Wrong gain128 kd')
    require(np.array_equal(w['lo'],p.m.jnt_range[p.jids,0]) and np.array_equal(w['hi'],p.m.jnt_range[p.jids,1]),'Wrong model integrator bounds')
    names=[p.m.joint(int(j)).name for j in p.jids];expected=np.array([names.index(j) for j in JOINTS])
    require(np.array_equal(w['indices'],expected),'Residual joint-name/index mismatch')
    return p,w,dict(passed=True,base_hash=BASE_HASH,all_base_fields_bitwise=True,gain128=True,native_caps_exact=True,residual_scale=.003,indices=expected.tolist(),joints=JOINTS,model_sha256=sha(DEFAULT_MODEL_PATH))

def policy_audit(folder,commands,policy,model_path,result,handover,cap):
    arrays=load(folder/'COMMANDS.npz');p,w,validation=validate_native(model_path,arrays,cap)
    require(result.get('neural_architecture')=='frozenlearnedhistorymotorANN +learnedproprioceptiveresidualANN','Undeclared residual architecture')
    trace=load(folder/'RESIDUAL_POLICY.npz');body=load(folder/'BODY.npz')
    require(sha(folder/'RESIDUAL_POLICY.npz')==result.get('residual_trace_sha256'),'Residual trace hash mismatch')
    import mujoco as mj
    data=p.fresh();history=None;warm=None;maxima={};targets=policy['targets'];actions=policy['actions'];n=len(targets)
    require(targets.shape==actions.shape==(len(commands[::5]),42),'Wrong policy shape')
    require(all(len(a)==n for a in trace.values()),'Residual trace length mismatch')
    require(all(np.isfinite(a).all() for a in trace.values()),'Nonfinite residual policy trace')
    for key,tail in {'features':(183,),'input':(477,),'history':(294,),'mean':(6,),'residual_target':(6,),'base_prediction':(42,),'predicted_rate':(42,),'actual_q':(42,),'actual_qvel':(42,),'foot_load':(6,)}.items():
        require(trace[key].shape==(n,*tail),'Wrong residual trace shape '+key)
    action_error=0.
    for i,target in enumerate(targets):
        tick=i*5
        if i:
            data.qpos[:]=body['qpos'][i-1];data.qvel[:]=body['qvel'][i-1];data.ctrl[:]=body['ctrl'][i-1];data.time=float(body['time'][i-1]);data.qacc_warmstart[:]=0;mj.mj_forward(p.m,data)
        history=observe(w,history,target);loads=positive_loads(mj,p,data);sensory=features(p,data,target,loads);pred=predict(w,history,sensory,target)
        pred.update(actual_q=data.qpos[p.qa],actual_qvel=data.qvel[p.va],foot_load=loads,time=np.asarray(float(data.time)))
        for key,value in pred.items():maxima[key]=max(maxima.get(key,0.),error(value,trace[key][i]))
        if tick==round(handover/.001):warm=history.copy()
        if tick<round(handover/.001) or result['controller']=='teacher':continue
        expected=np.zeros(42) if result['controller'] in ('hold','motor_cut') else pred['predicted_rate']
        action_error=max(action_error,error(expected,actions[i]))
    q=commands[:,1:43];rate,perturb=commands[:,43:85],commands[:,85:127]
    nxt=np.clip(q[:-1]+.001*np.clip(rate[:-1]+perturb[:-1],-cap,cap),w['lo'],w['hi'])
    checks=dict(target_integrator=error(nxt,q[1:])<1e-10,policy_targets_match_commands=error(targets,q[::5])<1e-12,
        policy_actions_match_commands=error(actions,rate[::5])<1e-12,rate_held_between_policy_ticks=error(rate,np.repeat(actions,5,axis=0)[:len(rate)])<1e-12,
        bounded_policy_rate=bool(abs(rate).max()<=cap+1e-10),bounded_targets=bool(np.all(q>=w['lo']-1e-10) and np.all(q<=w['hi']+1e-10)),
        independently_reconstructed_policy=action_error<5e-4,model_hash_matches=sha(model_path)==result.get('model_hash'),
        both_networks_and_full_sensory_trace=all(v<=(5e-4 if key=='predicted_rate' else 1e-5) for key,v in maxima.items()),
        residual_bound=bool(abs(trace['residual_target']).max()<=.003+1e-10))
    hand_errors={};hand=folder/'HANDOVER.npz'
    if hand.exists() and warm is not None:
        h=load(hand);hand_errors=dict(history=error(h['history'],warm),target=error(h['target'],q[round(handover/.001)]));checks['real_warm_history_preserved']=hand_errors['history']<1e-6 and hand_errors['target']<1e-12
    else:checks['real_warm_history_preserved']=False
    return dict(passed=all(checks.values()),checks=checks,K=int(w['K']),lags=w['lags'],integration_max_error=error(nxt,q[1:]),policy_rate_max_error=action_error,
        handover_errors=hand_errors,network_and_sensory_max_errors=maxima,native_validation=validation,
        rate_tolerance_rad_s=5e-4,target_tolerance_rad=1e-10,limitation='Warm teacher labels are not reconstructed. Neural outputs duringwarm are logged but not applied; no teacherafterhandover inann mode. Saved-state sensory reconstruction usesFK only; independentreplay checks actualintegratedstates separately.')

def build(write):
    require(sha(VERIFY)==VHASH and sha(ASSESS)==AHASH,'Frozen verifier/heading source changed')
    spec=importlib.util.spec_from_file_location('frozen_residual_base',VERIFY);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    text=VERIFY.read_text();oldbody=inspect.getsource(old.policy_audit)
    require(text.count(oldbody)==1,'Policy function source match is not unique')
    text=text.replace(oldbody,"def policy_audit(*args, **kwargs):\n    return _support.policy_audit(*args, **kwargs)\n",1)
    initial="    kp, kd, limits = [command_arrays[k] for k in ('kp', 'kd', 'limits')]"
    require(text.count(initial)==1,'Replay initialization patch is not unique')
    text=text.replace(initial,initial+"\n    residual_weights = _support.load_weights(model_path)\n    residual_trace = load_npz(folder/'RESIDUAL_POLICY.npz')\n    residual_history = None",1)
    loop='    for k, command in enumerate(commands):'
    insert=loop+"\n        if k % 5 == 0:\n            residual_history, residual_errors = _support.replay_features(mj, p, d, target, residual_weights, residual_history, residual_trace, k//5)\n            for key, value in residual_errors.items():\n                errors['residual_'+key] = max(errors.get('residual_'+key, 0.), value)"
    require(text.count(loop)==1,'Replay sensory insertion is not unique');text=text.replace(loop,insert,1)
    header=('import importlib.util as _residual_import, hashlib as _residual_hash\n'
        f"_residual_path={str(Path(__file__).resolve())!r}\n"
        f"assert _residual_hash.sha256(open(_residual_path,'rb').read()).hexdigest()=={sha(__file__)!r}\n"
        "_residual_spec=_residual_import.spec_from_file_location('independent_residual_support',_residual_path)\n"
        "_support=_residual_import.module_from_spec(_residual_spec);_residual_spec.loader.exec_module(_support)\n")
    future='from __future__ import annotations\n'
    require(text.count(future)==1,'Frozen future-import anchor changed');text=text.replace(future,future+header,1)
    effective=OUT/'EFFECTIVE_RESIDUAL_VERIFIER.py';provenancepath=OUT/'PROVENANCE.json'
    sources=[VERIFY,ASSESS,Path(__file__),FROZEN_BASE,HERE/'run_residual.py',BASE/'incremental_actuator.py',ROOT/'experiments/s5-ann-walk/20261107T091243Z/proprioception.py']
    provenance=dict(method='independent-two-ANN-plus-proprioception-v1',source_hashes={str(p):sha(p) for p in sources},effective_source=str(effective),effective_sha256=hashlib.sha256(text.encode()).hexdigest(),
        adaptation='Onecompletepolicy_auditfunctionreplacement; replayinitialization and pre-update5ms sensorychecks inserted. OriginalPDintegration/motorcut/materialvelocity/D/steps/heading/slipcriteria unchanged.',
        sensory='Independent183formula pluspositiveFncontactloads and genuine40frameown-targethistory, concatenated183+294. No runtimeorPPOsolver imported.',provenance_before_replay=write)
    if write:stable(effective,text);stable(provenancepath,json.dumps(provenance,indent=2)+'\n')
    module=types.ModuleType('effective_residual_physics_verifier');module.__file__=str(effective if write else VERIFY);exec(compile(text,str(effective),'exec'),module.__dict__)
    return module,provenance,effective,provenancepath

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv or '--help' in argv or '-h' in argv:module,*_=build(False);return module.main(argv)
    parse=argparse.ArgumentParser(add_help=False);parse.add_argument('folder',type=Path);parse.add_argument('--model',type=Path);parse.add_argument('--cap',type=float,default=30.);parse.add_argument('--handover',type=float,default=1.)
    args,_=parse.parse_known_args(argv);require(args.handover==1.,'Residual runtime declares1s physical teacher handover')
    folder=args.folder.resolve(strict=True);result=json.loads((folder/'RESULT.json').read_text());model=args.model or Path(result['model'])
    validate_native(model,load(folder/'COMMANDS.npz'),args.cap)
    no_write='--no-write' in argv;require(not(no_write and '--replay' in argv),'Replay requires saved effective provenance')
    if '--replay' in argv and '--replay-stride' not in argv:argv+=['--replay-stride','40']
    module,provenance,effective,provpath=build(not no_write);base_main=module.main
    if no_write:return base_main(argv)
    spec=importlib.util.spec_from_file_location('residual_frozen_heading',ASSESS);assessor=importlib.util.module_from_spec(spec);spec.loader.exec_module(assessor);assessor.v=module
    def annotated(arguments):
        code=base_main(arguments);path=folder/'VERIFICATION.json';report=json.loads(path.read_text());report['residual_method_provenance']=provenance
        report['source_hashes'].update(provenance['source_hashes']);report['source_hashes'].update({str(effective):sha(effective),str(provpath):sha(provpath),str(folder/'RESIDUAL_POLICY.npz'):sha(folder/'RESIDUAL_POLICY.npz')})
        report['physics_replay']['proprioception_reconstructed_at_actual_policy_state']=bool(report['physics_replay']['performed'])
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False,default=encode)+'\n');return code
    module.main=annotated;previous=sys.argv
    try:sys.argv=[str(Path(__file__)),*argv];return assessor.main()
    finally:sys.argv=previous
if __name__=='__main__':raise SystemExit(main())
