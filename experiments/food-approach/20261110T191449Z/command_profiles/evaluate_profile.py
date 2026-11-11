"""Prepared fixed command-profile evaluation. Import/offline checks do not run physics."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
PROFILES={'neutral':(0.,0.),'halfleft':(.5,.5),'halfright':(-.5,-.5),
          'left_to_right':(.5,-.5),'right_to_left':(-.5,.5)}


def command_at(profile,autonomous_decision):
    if profile not in PROFILES or not isinstance(autonomous_decision,(int,np.integer)) or not 0<=autonomous_decision<1920:
        raise ValueError('fixed profile and autonomous decision0..1919 required')
    return np.array([0.,PROFILES[profile][int(autonomous_decision>=960)]],dtype=np.float64)


def yaw_report(profile,yaw):
    yaw=np.unwrap(np.asarray(yaw,dtype=np.float64))
    if yaw.shape!=(15400,) or not np.isfinite(yaw).all():raise ValueError('requires complete15400 actual endpoint yaw samples')
    parts=[float(yaw[10599]-yaw[5799]),float(yaw[15399]-yaw[10599])]
    whole=float(yaw[-1]-yaw[5799]);turns=PROFILES[profile]
    if profile=='neutral':passed=True;criterion='no directional criterion'
    elif profile in ('halfleft','halfright'):
        passed=whole*np.sign(turns[0])>=.005;criterion='whole signed >=.005rad'
    else:
        passed=all(delta*np.sign(turn)>=.002 for delta,turn in zip(parts,turns));criterion='each4.8s signed >=.002rad including switching transient'
    return dict(whole_rad=whole,first4p8_rad=parts[0],second4p8_rad=parts[1],requested_turns=list(turns),criterion=criterion,passed=bool(passed),boundary_trace_indices=[5799,10599,15399])


def pin_imports():
    for name in ('physics','static_teacher','wave_reference','models','student_wave','identity','cycle_reward'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):
            raise RuntimeError('conflicting baseline module '+name)
    sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT))
    import checked_motor
    import conditional_motor
    import student_wave as sw
    import identity
    if Path(checked_motor.__file__).resolve()!=ROOT/'checked_motor.py':raise RuntimeError('wrong strict wrapper')
    if Path(conditional_motor.__file__).resolve()!=ROOT/'conditional_motor.py':raise RuntimeError('wrong conditional module')
    for module in (sw,identity):
        if Path(module.__file__).resolve().parent!=BASE:raise RuntimeError('wrong baseline import')
    return checked_motor,conditional_motor,sw,identity


def evaluate(args):
    import torch
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    checked,cm,sw,identity=pin_imports()
    checkpoint=Path(args.checkpoint)
    if not checkpoint.is_absolute():raise ValueError('--checkpoint must be absolute')
    out=Path(args.out)
    if not out.is_absolute():raise ValueError('--out must be absolute')
    if out.exists():raise FileExistsError(out)
    prereg=json.loads((HERE/'PREREG.json').read_text())
    if prereg['profiles']!=list(PROFILES) or prereg['turns_first_second_4p8s']!={k:list(v) for k,v in PROFILES.items()} or prereg['forward']!=0:
        raise ValueError('fixed prereg profile contract changed')
    checks=json.loads((ROOT/'CONDITIONAL_CHECKS.json').read_text())
    if not checks['all_passed']:raise ValueError('interface checks failed')
    for path,digest in checks['source_hashes'].items():
        if cm.sha256(path)!=digest:raise ValueError('interface check source drift '+path)
    model,payload=checked.load_checked_motor(checkpoint,args.sha256)
    sw.teacher_preflight();sw.student_preflight()
    core=sw.Core(adhesion=False)
    original,_=sw.load_student(cm.CHECKPOINT,core) # Checks original actual compiled/runtime identity.
    del original
    audit_path=ROOT/'odor/HEAD_GEOMETRY_AUDIT.json'
    audit=json.loads(audit_path.read_text())
    head=core.plant.body('Head').id
    if audit['status']!='PASS' or head!=audit['head']['body_id']:raise ValueError('head audit identity')
    offsets=audit['sensor_offsets']
    fwd=float(offsets['forward_offset']);lateral=float(offsets['lateral_offset'])
    local=np.array([[fwd,lateral,0.],[fwd,-lateral,0.]])
    paths=[Path(__file__),HERE/'PREREG.json',ROOT/'checked_motor.py',ROOT/'conditional_motor.py',ROOT/'CONDITIONAL_CHECKS.json',ROOT/'PREREG.json',checkpoint,cm.CHECKPOINT,audit_path]
    ident=identity.capture(core,dict(kind='fixed highlevel neural command profile',cli=vars(args),prereg=prereg,checkpoint_sha256=args.sha256,food_inputs=False,actor_target_queries=False),extra_paths=paths)
    out.mkdir(parents=True,exist_ok=False);identity.atomic_json(out/'IDENTITY.json',ident)
    trace=sw.Trace(full=True);geometry=[]
    def tick(action,source):
        trace.tick(core,action,source)
        rot=core.d.xmat[core.th].reshape(3,3)
        hrot=core.d.xmat[head].reshape(3,3);headpos=core.d.xpos[head].copy()
        sensors=headpos+local@hrot.T
        geometry.append(dict(time=float(core.d.time),thorax_yaw=float(np.arctan2(rot[1,0],rot[0,0])),head_xyz=headpos,
            sensor_xyz=sensors,sensor_midpoint_xyz=sensors.mean(axis=0),control_source=int(source)))
    for _ in range(1000):tick(np.zeros(42),sw.SOURCE['settle'])
    teacher=sw.StaticTeacher(core,sw.teacher_config(3))
    previous=np.zeros(42);state=model.initial_state(1)
    observer,events=sw.tracker(core),[];decisions=[];boundaries={};teacher_calls=0
    for k in range(2880):
        warm=k<960
        if k in (960,1920):
            before=sw.continuity_snapshot(core,previous,state)
            if k==960:observer,events=sw.tracker(core),[]
            after=sw.continuity_snapshot(core,previous,state)
            exact={key:bool(np.array_equal(before[key],after[key])) for key in before}
            if not all(exact.values()):raise ValueError('boundary state discontinuity')
            boundaries['handover' if k==960 else 'halfway']=dict(time_s=float(core.d.time),all_exact=True,exact=exact,
                before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),no_reset=True)
        x,_=sw.observations(core,previous,(2880-k)*.005)
        cmd=np.zeros(2) if warm else command_at(args.profile,k-960)
        hb=state.clone()
        with torch.no_grad():
            mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(cmd[None],dtype=torch.float64),state)
        policy=mean[0].tanh().numpy()
        if warm:
            action,_=teacher.action(core,k*.005);source=sw.SOURCE['teacher'];teacher_calls+=1
        else:action=policy;source=sw.SOURCE['student']
        decisions.append(dict(time=float(core.d.time),X238=x,command=cmd.copy(),hidden_before=hb[0].numpy(),hidden_after=state[0].numpy(),
            mean=mean[0].numpy(),policy_action=policy,actually_applied_action=action.copy(),control_source=source))
        for _ in range(5):
            tick(action,source)
            if not warm:events.extend(observer.measure(core)['events'])
        previous=action.copy()
    source_counts={str(source):trace.sources.count(source) for source in (sw.SOURCE['settle'],sw.SOURCE['teacher'],sw.SOURCE['student'])}
    expected={str(sw.SOURCE['settle']):1000,str(sw.SOURCE['teacher']):4800,str(sw.SOURCE['student']):9600}
    if source_counts!=expected or len(decisions)!=2880 or teacher_calls!=960:raise ValueError('actual phase counts mismatch')
    for row in decisions[960:]:
        if row['control_source']!=sw.SOURCE['student'] or not np.array_equal(row['policy_action'],row['actually_applied_action']):
            raise ValueError('autonomous output not direct neural action')
    auto=trace.summary(core,sw.SOURCE['student']);cycles=sw.cycle_report(observer,events)
    yaws=yaw_report(args.profile,[r['thorax_yaw'] for r in geometry])
    diagnostics={}
    for key in ('head_xyz','sensor_midpoint_xyz'):
        a=np.array([r[key] for r in geometry]);start,mid,end=a[5799],a[10599],a[15399]
        diagnostics[key]=dict(whole_delta=(end-start).tolist(),first4p8_delta=(mid-start).tolist(),second4p8_delta=(end-mid).tolist(),
            autonomous_path_length_xy=float(np.linalg.norm(np.diff(a[5799:,:2],axis=0),axis=1).sum()))
    traceinfo=trace.save(out/'trace_1ms.npz')
    np.savez_compressed(out/'neural_decisions.npz',**{key:np.asarray([r[key] for r in decisions]) for key in decisions[0]})
    np.savez_compressed(out/'geometry_1ms.npz',**{key:np.asarray([r[key] for r in geometry]) for key in geometry[0]})
    identity.verify(ident)
    checked.load_checked_motor(checkpoint,args.sha256) # Recheck embedded training source/data provenance after execution.
    result=dict(status='MEASURED_NOT_FULL_NATIVE_CERTIFICATION',profile=args.profile,checkpoint=str(checkpoint),checkpoint_sha256=args.sha256,
        eligible=bool(auto['all_gates'] and cycles['all_legs_completed'] and yaws['passed']),autonomous=auto,autonomous_cycles=cycles,yaw=yaws,
        boundaries=boundaries,source_counts_1ms=source_counts,decisions=dict(total=2880,teacher_prime=960,autonomous=1920),
        teacher_calls=teacher_calls,teacher_calls_after_handover=0,physical_steps=len(trace.rec)*80,extra_forces=False,trace=traceinfo,
        neural_decisions_sha256=cm.sha256(out/'neural_decisions.npz'),geometry_sha256=cm.sha256(out/'geometry_1ms.npz'),identity_sha256=ident['identity_sha256'],
        geometry_diagnostic_only=diagnostics,geometry_offsets=dict(forward=fwd,lateral=lateral),
        scope='Original1ms endpoint all5 gates+whole-window6cycles; fixed forward0; no native certification or food-success claim. Supplied command schedule is a neural input only; no scripted gait or physical support.')
    identity.atomic_json(out/'RESULT.json',result)
    print(json.dumps(dict(out=str(out),profile=args.profile,eligible=result['eligible'],gates=auto['gates'],yaw=yaws,cycles=cycles['complete_per_leg'])))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--profile',choices=list(PROFILES),required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261110)
    args=p.parse_args()
    if len(args.sha256)!=64 or any(c not in '0123456789abcdef' for c in args.sha256):raise ValueError('lowercase SHA256 required')
    evaluate(args)

if __name__=='__main__':main()
