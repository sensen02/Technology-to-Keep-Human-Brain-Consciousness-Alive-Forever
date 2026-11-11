"""Round2 fixed1500 slow residual physical evaluator. No physical imports on import."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TRAIN=ROOT/'train'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
BASE_CHECKPOINT=BASE/'BC_GRU_ROLE_3000/checkpoint_002500.pt'
ORIGINAL=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500'
HEAD_AUDIT=ROOT.parent/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
CHECKPOINT=TRAIN/'TRAIN/checkpoint_001500.pt'
COMMANDS={'neutral':[0.,0.],'left':[0.,1.],'right':[0.,-1.]}


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def command_at(condition,decision):
    if condition not in COMMANDS or not isinstance(decision,(int,np.integer)) or not 0<=decision<2880:raise ValueError('fixed condition/decision0..2879 required')
    elapsed=max(0.,(int(decision)-960)*.005)
    u=min(elapsed,1.);blend=u*u*(3.-2.*u)
    return np.asarray(COMMANDS[condition],dtype=np.float64)*blend


def score(condition,auto,cycles,yaw):
    direction=condition=='neutral' or yaw*(1 if condition=='left' else -1)>=.01
    flags=dict(original_five=bool(auto['all_gates']),all_six=bool(cycles['all_legs_completed']),strict_nonfoot_zero=bool(auto['nonfoot_force_max_native']==0.),turn_sign_magnitude=bool(direction))
    flags['eligible']=bool(all(flags.values()))
    return flags


def pin():
    for name in ('physics','static_teacher','student_wave','models','identity','cycle_reward','wave_reference'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):raise ValueError('wrong baseline module '+name)
    sys.path.insert(0,str(BASE));sys.path.insert(0,str(TRAIN))
    import student_wave as sw
    import identity
    spec=importlib.util.spec_from_file_location('_round2_checked_model_io',TRAIN/'model_io.py')
    io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)
    return sw,identity,io


def flatten(state):
    import torch
    if not isinstance(state,(tuple,list)) or len(state)!=2 or state[0].shape[-1]!=128 or state[1].shape[-1]!=64:raise ValueError('explicit base128/residual64 tuple required')
    return torch.cat(tuple(state),dim=-1)


def checked_load(io,path,digest):
    if not Path(path).is_absolute() or Path(path).resolve()!=CHECKPOINT:raise ValueError('only exact final1500 checkpoint path')
    if len(digest)!=64 or any(ch not in '0123456789abcdef' for ch in digest):raise ValueError('explicit lowercase checkpoint SHA256 required')
    model,payload=io.load_checked(path,digest,require_final=True)
    if payload['curve'][-1]['step']!=1500 or payload['config']['physical_eval_only_step']!=1500:raise ValueError('actual fixed1500 required')
    return model.eval(),payload


def source_paths():
    paths=[HERE/n for n in ('evaluate_motor.py','verify_actual.py','pipeline.py','EVAL_PREREG.json','check_prepared.py','READINESS.json')]
    paths += [ROOT/'ROUND2_PLAN.json',HEAD_AUDIT]
    paths += sorted(TRAIN.glob('*.py'))+sorted(TRAIN.glob('*.json'))
    if TRAIN/'model_io.py' not in paths:raise ValueError('training loader absent')
    return list(dict.fromkeys(p.resolve() for p in paths))


def parser():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--condition',choices=list(COMMANDS),required=True);p.add_argument('--out',required=True)
    p.add_argument('--training-complete',action='store_true');p.add_argument('--native',action='store_true')
    return p


def evaluate(args):
    if not args.training_complete:raise ValueError('code-only: explicit parent --training-complete authorization required')
    if not Path(args.out).is_absolute() or Path(args.out).resolve()!=HERE/('EVAL_'+args.condition):raise ValueError('fresh fixed evaluation/EVAL_condition output required')
    preparation=json.loads((HERE/'READINESS.json').read_text())
    if not preparation['all_passed']:raise ValueError('offline preparation checks required')
    for path,digest in preparation['source_hashes'].items():
        if sha256(path)!=digest:raise ValueError('offline checked source drift '+path)
    import torch
    torch.set_num_threads(1);torch.manual_seed(20261110)
    sw,identity,io=pin();model,payload=checked_load(io,args.checkpoint,args.sha256)
    sw.teacher_preflight();sw.student_preflight()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    paths=source_paths();sources={str(p):sha256(p) for p in paths}
    prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
    if prereg['conditions']!=COMMANDS or prereg['checkpoint_step']!=1500:raise ValueError('evaluation prereg drift')
    c=None;tr=None;recorder=None;rows=[];geometry=[];result=None;error=None;ident=None
    def dump(name,value):identity.atomic_json(out/name,jsonable(value))
    try:
        recorder=sw.NativeRecorder(out/'native') if args.native else None
        c=sw.Core(adhesion=False,recorder=recorder)
        original,_=sw.load_student(BASE_CHECKPOINT,c)
        if not all(torch.equal(v,model.base.state_dict()[k]) for k,v in original.actor.state_dict().items()):raise ValueError('frozen base differs actual original runtime')
        import mujoco as mj
        mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
        if mesh.sum()!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('complete71 opaque body required')
        ident=identity.capture(c,dict(kind='round2 fixed1500 slow residual actual physical evaluation',cli=vars(args),condition=args.condition,evaluation_source_hashes=sources,timing=prereg['timing'],state_layout=[128,64],command_formula=prereg['command']),extra_paths=paths+[Path(args.checkpoint).resolve()])
        dump('IDENTITY.json',ident)
        np.savez_compressed(out/'runtime_geometry.npz',geom_type=c.plant.geom_type,geom_rgba=c.plant.geom_rgba)
        audit=json.loads(HEAD_AUDIT.read_text());head=c.plant.body('Head').id
        if audit['status']!='PASS' or head!=audit['head']['body_id']:raise ValueError('head audit identity mismatch')
        local=np.asarray(audit['sensor_offsets']['local_xyz'],dtype=np.float64)
        tr=sw.Trace(full=True)
        def tick(action,source):
            tr.tick(c,action,source)
            rot=c.d.xmat[c.th].reshape(3,3);hrot=c.d.xmat[head].reshape(3,3);pos=c.d.xpos[head].copy()
            sensors=pos+local@hrot.T
            geometry.append(dict(time=float(c.d.time),thorax_yaw=float(np.arctan2(rot[1,0],rot[0,0])),head_yaw=float(np.arctan2(hrot[1,0],hrot[0,0])),head_xyz=pos,head_rotation=hrot.copy(),sensor_xyz=sensors,sensor_midpoint_xyz=sensors.mean(axis=0),control_source=int(source)))
        for _ in range(1000):tick(np.zeros(42),sw.SOURCE['settle'])
        teacher=sw.StaticTeacher(c,sw.teacher_config(3))
        previous=np.zeros(42);state=model.initial_state(1);observer=sw.tracker(c);events=[];transition=None;teacher_calls=0
        for k in range(2880):
            warm=k<960
            if k==960:
                before=sw.continuity_snapshot(c,previous,flatten(state));observer=sw.tracker(c);events=[]
                after=sw.continuity_snapshot(c,previous,flatten(state));exact={key:bool(np.array_equal(before[key],after[key])) for key in before}
                if not all(exact.values()):raise ValueError('withdrawal continuity failure')
                transition=dict(time_s=float(c.d.time),all_exact=True,exact=exact,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),no_reset=True,state_dims=[128,64])
                dump('HANDOVER.json',transition)
            x,_=sw.observations(c,previous,(2880-k)*.005);command=command_at(args.condition,k);hb=flatten(state).clone()
            with torch.no_grad():mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(command[None],dtype=torch.float64),state)
            policy=mean[0].tanh().numpy()
            if warm:action,_=teacher.action(c,k*.005);teacher_calls+=1;source=sw.SOURCE['teacher']
            else:action=policy;source=sw.SOURCE['student']
            rows.append(dict(time=float(c.d.time),X238=x,command=command.copy(),hidden_before=hb[0].numpy(),hidden_after=flatten(state)[0].numpy(),mean=mean[0].numpy(),policy_action=policy,actually_applied_action=action.copy(),control_source=int(source)))
            for _ in range(5):
                tick(action,source)
                if not warm:events.extend(observer.measure(c)['events'])
            previous=action.copy()
        auto=tr.summary(c,sw.SOURCE['student']);cycles=sw.cycle_report(observer,events)
        yaw=np.unwrap(np.asarray([g['thorax_yaw'] for g in geometry]));delta=float(yaw[-1]-yaw[5799]);criteria=score(args.condition,auto,cycles,delta)
        result=dict(status='ENGINEERING_ONLY_NOT_FULL_NATIVE_OR_FOOD_CERTIFICATION',engineering_pass=criteria['eligible'],criteria=criteria,condition=args.condition,checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=args.sha256,
                    autonomous=auto,autonomous_cycles=cycles,yaw_change_rad=delta,transition=transition,state_layout=[128,64],source_hashes=sources,identity_sha256=ident['identity_sha256'],
                    physical_steps=len(tr.rec)*80,teacher_calls=teacher_calls,teacher_calls_after_handover=0,extra_forces=False,mesh_count=71,all_mesh_opaque=True,native_audit='NOT_PERFORMED',food_success=False,geometry_diagnostic_only=True)
        if args.condition=='neutral':
            with np.load(ORIGINAL/'neural_decisions.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
            parity={key:bool(np.array_equal(np.asarray([r[key] for r in rows]),old[key])) for key in ('X238','mean','policy_action','actually_applied_action','control_source')}
            for key in ('hidden_before','hidden_after'):parity['base_'+key]=bool(np.array_equal(np.asarray([r[key][:128] for r in rows]),old[key]))
            if not all(parity.values()):raise ValueError('neutral baseline neural parity failure')
            result['neutral_baseline_neural_parity']=parity
        identity.verify(ident);identity.verify_runtime(ident,c)
        for p,h in sources.items():
            if sha256(p)!=h:raise ValueError('source drift '+p)
        checked_load(io,args.checkpoint,args.sha256)
    except BaseException as exc:error=exc
    finally:
        artifacts={}
        try:
            if tr is not None and tr.rows:artifacts['trace']=tr.save(out/'trace_1ms.npz')
            if rows:
                path=out/'neural_decisions.npz';np.savez_compressed(path,**{key:np.asarray([r[key] for r in rows]) for key in rows[0]})
                artifacts['neural_decisions']=dict(path=str(path),sha256=sha256(path),rows=len(rows))
            if geometry:
                path=out/'geometry_1ms.npz';np.savez_compressed(path,**{key:np.asarray([g[key] for g in geometry]) for key in geometry[0]})
                artifacts['geometry']=dict(path=str(path),sha256=sha256(path),rows=len(geometry))
            if (out/'runtime_geometry.npz').exists():artifacts['runtime_geometry']=dict(path=str(out/'runtime_geometry.npz'),sha256=sha256(out/'runtime_geometry.npz'))
            if recorder is not None:artifacts['native']=recorder.close()
            if error is not None:dump('FAILURE.json',dict(engineering_pass=False,error_type=type(error).__name__,error=str(error),completed_endpoints=0 if tr is None else len(tr.rec),attempted_decisions=len(rows),artifacts=artifacts,source_hashes=sources))
            elif result is not None:result.update(artifacts);dump('RESULT.json',result)
        except BaseException as saving:
            if error is None:error=saving
            else:print('Additional save failure: '+repr(saving),file=sys.stderr,flush=True)
    if error is not None:raise error
    print(json.dumps(jsonable(dict(condition=args.condition,engineering_pass=result['engineering_pass'],gates=result['autonomous']['gates'],slip=result['autonomous']['slip_ratio_max'],yaw=result['yaw_change_rad']))),flush=True)
    return result

if __name__=='__main__':evaluate(parser().parse_args())
