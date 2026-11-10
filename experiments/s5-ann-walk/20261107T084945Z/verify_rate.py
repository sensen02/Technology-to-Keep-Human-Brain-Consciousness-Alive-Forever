"""Declared integrator-rate servo audit + auditable adapted frozen walk replay.

CLI and heading assessment are inherited from080221/assess_walk.py. No replay
occurs unless --replay is explicitly supplied. Only replay target-rate tracking
and torque change; native/body/policy/integrator/slip/heading checks stay frozen.
Rate-method provenance is added to VERIFICATION before GOAL_ASSESSMENT hashes it.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
HERE=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T084945Z')
VERIFY=ROOT/'experiments/s5-ann-walk/20261107T073258Z/verify_walk.py'
ASSESS=ROOT/'experiments/s5-ann-walk/20261107T080221Z/assess_walk.py'
MODEL=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml')
VERIFY_SHA='78e1b838f0a150c5fbae2604cbad59ce49809b486066f072e05f55ba9e1e862e'
ASSESS_SHA='c01bcdfa1b526693bb4a8f7645c8f9517a1d284f4f2832855ed8fab3a4ebcf28'
MODEL_SHA='517cec0eed476f825535b56858b434c949bcbf8ed18ce954189689c784d1d2d1'
EFFECTIVE=OUT/'EFFECTIVE_REPLAY.py'
PROVENANCE=OUT/'REPLAY_PROVENANCE.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def require(condition,message):
    if not condition:raise ValueError(message)


def stable_write(path,text):
    if path.exists():
        require(path.read_text()==text,'Existing method provenance differs; preserve it and use a new timestamp: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)


def load_assessor():
    require(sha(VERIFY)==VERIFY_SHA,'Frozen physics verifier changed')
    require(sha(ASSESS)==ASSESS_SHA,'Frozen heading assessment changed')
    spec=importlib.util.spec_from_file_location('frozen_heading_assessment',ASSESS)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare_replay(assessment,write=True):
    """Compile the count-checked adaptation without executing a physics step."""
    text=inspect.getsource(assessment.v.replay)
    old='        target = np.clip(target+CDT*np.clip(command[43:85]+command[85:127], -cap, cap), w[\'lo\'], w[\'hi\'])'
    new=('        previous_target = target.copy()\n'+old+
         '\n        effective_rate = (target - previous_target) / CDT')
    require(text.count(old)==1,'Frozen replay target-update patch is not unique')
    text=text.replace(old,new)
    old_torque='            raw = kp*(target-d.qpos[p.qa])-kd*d.qvel[p.va]'
    new_torque='            raw = kp*(target-d.qpos[p.qa])+kd*(effective_rate-d.qvel[p.va])'
    require(text.count(old_torque)==1,'Frozen replay torque patch is not unique')
    text=text.replace(old_torque,new_torque)
    # Full standalone effective source, including dependencies, saved before replay.
    header=(f'"""Auditable replay derived from {VERIFY}; only integrator rate and PD derivative reference changed."""\n'
        'import importlib.util, hashlib, json, sys\nfrom pathlib import Path\n'
        f'_path=Path({str(VERIFY)!r})\n'
        f'assert hashlib.sha256(_path.read_bytes()).hexdigest()=={VERIFY_SHA!r}\n'
        '_spec=importlib.util.spec_from_file_location("effective_frozen_dependencies",_path)\n'
        '_v=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_v)\n'
        'np=_v.np;DT=_v.DT;CDT=_v.CDT;BASE=_v.BASE\n'
        'load_npz=_v.load_npz;maximum_error=_v.maximum_error\n'
        'contact_row_and_material_velocity=_v.contact_row_and_material_velocity\n\n')
    effective=header+text
    files=[VERIFY,ASSESS,Path(__file__),HERE/'rate_servo.py',HERE/'run_rate.py',BASE/'incremental_actuator.py',MODEL]
    provenance=dict(method='declared-integrator-effective-rate-replay-v1',
        formula='tau=clip(kp*(target-q)+kd*((new_target-previous_target)/0.001-qvel), +/-original_native_limit)',
        adaptation='Count-checked two literal changes in frozen replay: capture previous target/effective clipped integrator rate; change derivative reference. All integration and checks otherwise unchanged.',
        effective_source=str(EFFECTIVE),effective_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        source_hashes={str(p):sha(p) for p in files},frozen_heading_thresholds_unchanged=True,
        teacher_never_counts_as_neural_success=True,
        provenance_written_before_replay=bool(write))
    if write:
        stable_write(EFFECTIVE,effective)
        stable_write(PROVENANCE,json.dumps(provenance,indent=2)+'\n')
    namespace={'__name__':'auditable_rate_replay','__file__':str(EFFECTIVE)}
    exec(compile(effective,str(EFFECTIVE),'exec'),namespace)
    return namespace['replay'],provenance


def validate_records(result,arrays,model_path,cap=30.):
    """Independent declared-mode/native-cap/keyframe-inertia validation; no steps."""
    mode=result.get('actuator_damping_mode')
    require(mode in ('same','critical'),'Undeclared actuator_damping_mode')
    if 'actuator_mode' in result:
        require(result['actuator_mode']==mode,'Conflicting or undeclared actuator_mode')
    require(result.get('actuator_derivative_target')=='integrator_effective_rate','Undeclared derivative reference')
    require(result.get('actuator_gain_scale')==32.,'Only declared gain scale32 is accepted')
    ratio=result.get('actuator_damping_ratio')
    require((mode=='same' and ratio is None) or (mode=='critical' and ratio==.7),'Undeclared damping ratio')
    require(cap==30.,'Declared physical command-rate cap is30rad/s')
    require(sha(MODEL)==MODEL_SHA,'Original physical model identity changed')
    sys.path.insert(0,str(BASE))
    import mujoco as mj
    from incremental_actuator import load_full_model,validate_full_model
    m=load_full_model();validate_full_model(m)
    require(abs(m.opt.timestep-.0000125)<1e-15,'Wrong native physics timestep')
    d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d)
    jids=m.actuator_trnid[:,0];qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    require(len(set(jids.tolist()))==42 and np.all(va>=6),'Actuator/root mapping mismatch')
    native=np.minimum(-m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    forced=m.actuator_forcelimited.astype(bool)
    native[forced]=np.minimum(native[forced],np.minimum(-m.actuator_forcerange[forced,0],m.actuator_forcerange[forced,1]))
    limits=np.asarray(arrays['limits']);kp=np.asarray(arrays['kp']);kd=np.asarray(arrays['kd'])
    require(limits.shape==kp.shape==kd.shape==(42,),'Malformed actuator gains/limits')
    require(np.isfinite(np.r_[limits,kp,kd]).all(),'Nonfinite actuator declaration')
    require(np.array_equal(limits,native),'Recorded torque limits differ from original native caps')
    expected_kp=native/.2*32.
    matrix=np.empty((m.nv,m.nv));mj.mj_fullM(m,matrix,d.qM)
    inertia=np.diag(matrix)[va].copy()
    require(np.all(inertia>0) and np.isfinite(inertia).all(),'Invalid fresh-keyframe joint inertia')
    expected_kd=native*.005*np.sqrt(32.) if mode=='same' else 2*.7*np.sqrt(expected_kp*inertia)
    require(np.allclose(kp,expected_kp,rtol=1e-12,atol=1e-12),'Recorded kp differs from declared gain32')
    require(np.allclose(kd,expected_kd,rtol=1e-12,atol=1e-12),'Recorded kd differs from declared mode/keyframe inertia')
    weights=load(model_path)
    require(np.array_equal(weights['lo'],m.jnt_range[jids,0]) and np.array_equal(weights['hi'],m.jnt_range[jids,1]),
            'Model metadata bounds differ from native integrator bounds')
    return dict(passed=True,mode=mode,derivative_reference='actual clipped target increment /1ms',gain_scale=32.,
        damping_ratio=ratio,torque_limits_exact_native=True,expected_kp=expected_kp.tolist(),expected_kd=expected_kd.tolist(),
        fresh_keyframe_Mjj=inertia.tolist(),kp_max_error=float(abs(kp-expected_kp).max()),kd_max_error=float(abs(kd-expected_kd).max()),
        qpos_addresses=qa.tolist(),velocity_addresses=va.tolist(),physics_steps=0,
        model=dict(nq=m.nq,nv=m.nv,nu=m.nu,nmesh=m.nmesh,neq=m.neq,gravity=m.opt.gravity.tolist(),timestep=m.opt.timestep),
        limitation='Diagonal-inertia damping is an isolated-joint engineering estimate, not proof of critical damping for the coupled contacting body.')


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    assessment=load_assessor();v=assessment.v;base_main=v.main
    if not argv or '--help' in argv or '-h' in argv:return base_main(argv)
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('folder',type=Path);parser.add_argument('--model',type=Path);parser.add_argument('--cap',type=float,default=30.)
    args,_=parser.parse_known_args(argv)
    folder=args.folder.resolve(strict=True);result=json.loads((folder/'RESULT.json').read_text())
    arrays=load(folder/'COMMANDS.npz');model=args.model or Path(result['model'])
    validation=validate_records(result,arrays,model,args.cap)
    if '--no-write' in argv:
        require('--replay' not in argv,'--replay requires saved effective provenance; combine neither with --no-write')
        # Do not let frozen assessor consume stale reports or write GOAL in read-only mode.
        return base_main(argv)
    effective,provenance=prepare_replay(assessment,write=True)
    def checked_replay(folder,commands,command_arrays,body,native,model_path,result,handover,cap,stride):
        validate_records(result,command_arrays,model_path,cap)
        return effective(folder,commands,command_arrays,body,native,model_path,result,handover,cap,stride)
    v.replay=checked_replay
    def main_with_method_metadata(arguments):
        code=base_main(arguments)
        path=folder/'VERIFICATION.json';report=json.loads(path.read_text())
        report['rate_actuator_validation']=validation
        report['replay_method_provenance']=provenance
        report['source_hashes'].update(provenance['source_hashes'])
        report['source_hashes'][str(EFFECTIVE)]=sha(EFFECTIVE)
        report['source_hashes'][str(PROVENANCE)]=sha(PROVENANCE)
        report['physics_replay']['actuator_torque_formula']=provenance['formula']
        report['physics_replay']['effective_source_sha256']=sha(EFFECTIVE)
        # This write precedes the frozen assessor's input_sha256 computation.
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        return code
    v.main=main_with_method_metadata
    previous=sys.argv
    try:
        sys.argv=[str(Path(__file__)),*argv]
        return assessment.main()
    finally:sys.argv=previous


if __name__=='__main__':raise SystemExit(main())
