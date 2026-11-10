"""Conditional-integrator independent array audit and full replay adapter.
No controller imports. Array checks use logged masks; replay recomputes every mask
from its independent physical state. Frozen neural/D/steps/slip/heading stay intact.
"""
import argparse,hashlib,importlib.util,json,sys,types
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
HERE=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T103309Z')
VERIFY=ROOT/'experiments/s5-ann-walk/20261107T073258Z/verify_walk.py'
ASSESS=ROOT/'experiments/s5-ann-walk/20261107T080221Z/assess_walk.py'
VHASH='78e1b838f0a150c5fbae2604cbad59ce49809b486066f072e05f55ba9e1e862e'
AHASH='c01bcdfa1b526693bb4a8f7645c8f9517a1d284f4f2832855ed8fab3a4ebcf28'
CHASH='7bfcb657303be869fbb374d8ab2eebd37229a5e7c637eb4f07e8d8d06d8c5f1a'
EFFECTIVE=OUT/'EFFECTIVE_VERIFIER.py';PROVENANCE=OUT/'VERIFIER_PROVENANCE_V2.json'
MODEL=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml')
MODEL_HASH='517cec0eed476f825535b56858b434c949bcbf8ed18ce954189689c784d1d2d1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def require(ok,message):
    if not ok:raise ValueError(message)
def replace(text,old,new):
    require(text.count(old)==1,'Frozen adaptation must match exactly once: '+old);return text.replace(old,new)
def stable_write(path,text):
    if path.exists():require(path.read_text()==text,'Preserve mismatched prior method artifact '+str(path))
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

def build(write=True):
    require(sha(VERIFY)==VHASH and sha(ASSESS)==AHASH,'Frozen verifier/assessment source changed')
    require(sha(HERE/'conditional_integrator.py')==CHASH,'Frozen conditional controller changed')
    text=VERIFY.read_text()
    old="    nxt = np.clip(q[:-1]+CDT*np.clip(rate[:-1]+perturb[:-1], -cap, cap), w['lo'], w['hi'])"
    new="    antiwindup = load_npz(folder/'ANTIWINDUP.npz')\n    effective = np.where(antiwindup['torque_block_mask'][:-1], 0., np.clip(rate[:-1]+perturb[:-1], -cap, cap))\n    nxt = np.clip(q[:-1]+CDT*effective, w['lo'], w['hi'])"
    text=replace(text,old,new)
    old="    kp, kd, limits = [command_arrays[k] for k in ('kp', 'kd', 'limits')]"
    new=old+"\n    antiwindup = load_npz(folder/'ANTIWINDUP.npz')\n    blocked_ticks = blocked_updates = 0"
    text=replace(text,old,new)
    old="        target = np.clip(target+CDT*np.clip(command[43:85]+command[85:127], -cap, cap), w['lo'], w['hi'])"
    new="""        previous_target = target.copy()
        requested = command[43:85]+command[85:127]
        limited = np.clip(requested, -cap, cap)
        raw_before = kp*(previous_target-d.qpos[p.qa])-kd*d.qvel[p.va]
        enabled = float(d.time) >= 1.-1e-9
        mask = (((raw_before >= limits)&(limited > 0))|((raw_before <= -limits)&(limited < 0))) if enabled else np.zeros(42, bool)
        effective = np.where(mask, 0., limited)
        target = np.clip(previous_target+CDT*effective, w['lo'], w['hi'])
        blocked_ticks += int(mask.any()); blocked_updates += int(mask.sum())
        compare = dict(time=float(d.time), command_tick=k+1, requested_rate=requested,
            limited_requested_rate=limited, effective_input_rate=effective,
            requested_rate_clip_mask=requested!=limited, torque_block_mask=mask,
            raw_torque_before=raw_before, target_before=previous_target, target_after=target,
            blocked_command_ticks=blocked_ticks, blocked_joint_updates=blocked_updates, enabled=enabled)
        for key, actual in compare.items():
            error_key = 'antiwindup_'+key
            discrepancy = maximum_error(np.asarray(actual, float), np.asarray(antiwindup[key][k], float))
            errors[error_key] = max(errors.get(error_key, 0.), discrepancy)"""
    text=replace(text,old,new)
    sources=[VERIFY,ASSESS,HERE/'conditional_integrator.py',HERE/'run_conditional.py',Path(__file__),BASE/'incremental_actuator.py',MODEL]
    provenance=dict(method='conditional-current-raw-PD-integration-v1',effective_source=str(EFFECTIVE),
        effective_sha256=hashlib.sha256(text.encode()).hexdigest(),source_hashes={str(p):sha(p) for p in sources},
        formula='limited=clip(requested,+/-30); iftime>=1s: block=(raw>=cap & limited>0)|(raw<=-cap & limited<0); target=clip(target+0.001*where(block,0,limited),nativebounds); raw=kp*(target-q)-kd*qvel eachphysicsstep',
        adaptation='Exactly three count-checked source blocks: policy target integration uses logged masks; replay loads trace; replay independently recomputes masks/integration and checks every1ms trace field.',
        array_masks_trusted_only_for_array_check=True,physics_recomputes_masks=True,
        provenance_saved_before_replay=write,heading_and_all_acceptance_thresholds_unchanged=True,
        validation_revision='V2 adds originalXMLhash plusfinite/shape trace validation; effective replay/policy code identical to preservedV1provenance.')
    if write:stable_write(EFFECTIVE,text);stable_write(PROVENANCE,json.dumps(provenance,indent=2)+'\n')
    module=types.ModuleType('independent_conditional_verifier');module.__file__=str(EFFECTIVE if write else VERIFY)
    exec(compile(text,str(EFFECTIVE),'exec'),module.__dict__)
    return module,provenance


def validate(folder,result,commands,aw,model_path,cap=30.,handover=1.):
    require(result.get('actuator_interface')=='conditional_raw_PD_outward_block','Undeclared actuator_interface')
    require(result.get('gain_scale')==128. and result.get('conditional_from_s')==1.,'Undeclared gain/conditional activation')
    require(cap==30. and handover==1.,'Conditional diagnostic declares30rad/s and1s activation')
    require(sha(folder/'ANTIWINDUP.npz')==result.get('antiwindup_sha256'),'Antiwindup trace hash mismatch')
    sys.path.insert(0,str(BASE));import mujoco as mj
    from incremental_actuator import load_full_model,validate_full_model,DEFAULT_MODEL_PATH
    require(sha(MODEL)==MODEL_HASH,'Original full physical model identity changed')
    m=load_full_model();validate_full_model(m);jids=m.actuator_trnid[:,0];qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    native=np.minimum(-m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);force=m.actuator_forcelimited.astype(bool)
    native[force]=np.minimum(native[force],np.minimum(-m.actuator_forcerange[force,0],m.actuator_forcerange[force,1]))
    require(np.array_equal(commands['limits'],native),'Recorded limits differ from original native caps')
    require(np.allclose(commands['kp'],native/.2*128.,rtol=1e-12,atol=1e-12),'Wrong gain128 kp')
    require(np.allclose(commands['kd'],native*.005*np.sqrt(128),rtol=1e-12,atol=1e-12),'Wrong gain128 kd')
    model=load(model_path);require(np.array_equal(model['lo'],m.jnt_range[jids,0]) and np.array_equal(model['hi'],m.jnt_range[jids,1]),'Model bounds differ from native actuator')
    rows=commands['rows'];n=len(rows);require(all(len(a)==n for a in aw.values()),'Trace length mismatch')
    require(all(np.isfinite(a).all() for a in aw.values()),'Nonfinite antiwindup trace')
    require(all(np.asarray(aw[key]).shape==(n,42) for key in ('requested_rate','limited_requested_rate','effective_input_rate','requested_rate_clip_mask','torque_block_mask','raw_torque_before','target_before','target_after')),'Malformed antiwindup vector shape')
    q=rows[:,1:43];requested=rows[:,43:85]+rows[:,85:127];limited=np.clip(requested,-cap,cap)
    mask=np.asarray(aw['torque_block_mask'],bool);enabled=rows[:,0]>=1.-1e-9
    raw=aw['raw_torque_before'];logged_rule=(((raw>=native)&(limited>0))|((raw<=-native)&(limited<0)))&enabled[:,None]
    effective=np.where(mask,0.,limited);after=np.clip(q+.001*effective,model['lo'],model['hi'])
    error=lambda a,b:float(np.max(abs(np.asarray(a,float)-np.asarray(b,float))))
    checks=dict(clock=error(aw['time'],rows[:,0])<1e-10,tick_number=error(aw['command_tick'],np.arange(1,n+1))==0,
        activation=bool(np.array_equal(aw['enabled'],enabled)),warmup_mask_zero=not mask[~enabled].any(),
        mask_matches_logged_raw_rule=bool(np.array_equal(mask,logged_rule)),requested=error(aw['requested_rate'],requested)<1e-12,
        limited_requested=error(aw['limited_requested_rate'],limited)<1e-12,effective=error(aw['effective_input_rate'],effective)<1e-12,
        requested_clip_mask=bool(np.array_equal(aw['requested_rate_clip_mask'],requested!=limited)),
        target_before=error(aw['target_before'],q)<1e-12,target_after=error(aw['target_after'],after)<1e-12,
        next_tick_target=error(after[:-1],q[1:])<1e-10,zero_commands_exact=bool(np.array_equal(aw['target_after'][~requested.any(axis=1)],q[~requested.any(axis=1)])),
        blocked_ticks=error(aw['blocked_command_ticks'],np.cumsum(mask.any(axis=1)))==0,
        blocked_updates=error(aw['blocked_joint_updates'],np.cumsum(mask.sum(axis=1)))==0)
    require(all(checks.values()),'Conditional trace array checks failed: '+str([k for k,v in checks.items() if not v]))
    return dict(passed=True,checks=checks,native_caps_exact=True,gain=128.,activation_s=1.,
        torque_blocked_joint_updates=int(mask.sum()),blocked_command_ticks=int(mask.any(axis=1).sum()),
        trace_hash=sha(folder/'ANTIWINDUP.npz'),model_hash=sha(DEFAULT_MODEL_PATH),qpos_addresses=qa.tolist(),velocity_addresses=va.tolist(),
        limitation='Array mask/raw consistency is not independent physical-state verification. Only full replay recomputes rawtorque andmask from independently integratedstate.')


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv or '--help' in argv or '-h' in argv:
        module,_=build(write=False);return module.main(argv)
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('folder',type=Path);parser.add_argument('--model',type=Path)
    parser.add_argument('--cap',type=float,default=30.);parser.add_argument('--handover',type=float,default=1.)
    args,_=parser.parse_known_args(argv);folder=args.folder.resolve(strict=True)
    result=json.loads((folder/'RESULT.json').read_text());commands=load(folder/'COMMANDS.npz');aw=load(folder/'ANTIWINDUP.npz');model=args.model or Path(result['model'])
    validation=validate(folder,result,commands,aw,model,args.cap,args.handover)
    no_write='--no-write' in argv;require(not(no_write and '--replay' in argv),'Replay requires saved provenance; --no-write+--replay unsupported')
    if '--replay' in argv and '--replay-stride' not in argv:argv+=['--replay-stride','40']
    module,provenance=build(write=not no_write);base_main=module.main
    if no_write:return base_main(argv)
    spec=importlib.util.spec_from_file_location('conditional_frozen_heading',ASSESS);assessment=importlib.util.module_from_spec(spec);spec.loader.exec_module(assessment)
    assessment.v=module
    def with_metadata(arguments):
        code=base_main(arguments);path=folder/'VERIFICATION.json';report=json.loads(path.read_text())
        report['conditional_interface_validation']=validation;report['conditional_method_provenance']=provenance
        report['source_hashes'].update(provenance['source_hashes']);report['source_hashes'].update({str(EFFECTIVE):sha(EFFECTIVE),str(PROVENANCE):sha(PROVENANCE),str(folder/'ANTIWINDUP.npz'):sha(folder/'ANTIWINDUP.npz')})
        report['physics_replay']['conditional_mask_recomputed']=bool(report['physics_replay']['performed'])
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return code
    module.main=with_metadata;previous=sys.argv
    try:sys.argv=[str(Path(__file__)),*argv];return assessment.main()
    finally:sys.argv=previous

if __name__=='__main__':raise SystemExit(main())
