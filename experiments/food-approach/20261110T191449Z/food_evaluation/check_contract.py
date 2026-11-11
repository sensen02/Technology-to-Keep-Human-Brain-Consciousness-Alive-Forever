"""Offline draft boundary/loader checks. No physics, no training."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import json
import sys
import tempfile
import numpy as np
from contract import validate_layout,source_center,qualification,Arrival,digest,layout_from
from evaluate_food import absolute_inputs,load_behavior,prefix_digest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    checks={}
    d=dict(layout_id='OFFLINE_ONLY',offset_forward=.4,offset_lateral=.1,patch_radius=.02,sigma=.5,amplitude=1.,noise_std=0.,onset_elapsed_s=0.,rise_s=.2,arrival_dwell_s=.003,seed=20261110)
    checks['explicit_layout_valid']=validate_layout(d)==d
    checks['handover_offset_world']=np.allclose(source_center([10,20],np.eye(3),d),[10.4,20.1])
    rot=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    checks['head_relative_rotated_offset']=np.allclose(source_center([10,20],rot,d),[9.9,20.4])
    def rejected(fn):
        try:fn()
        except (ValueError,KeyError,TypeError):return True
        return False
    for key,value in [('sigma',0.),('noise_std',-1.),('onset_elapsed_s',-1.),('amplitude',float('nan')),('seed',True)]:
        bad=deepcopy(d);bad[key]=value
        checks['reject_'+key]=rejected(lambda:validate_layout(bad))
    bad=deepcopy(d);bad['offset_forward']=0;bad['offset_lateral']=0
    checks['reject_start_inside_patch']=rejected(lambda:validate_layout(bad))
    goodq=dict(all_conditions_eligible=True,checkpoint='/tmp/motor.pt',checkpoint_sha256='a'*64)
    checks['qualification_exact_binding']=qualification(goodq,'/tmp/motor.pt','a'*64)
    checks['qualification_reject_wrong_digest']=not qualification(goodq,'/tmp/motor.pt','b'*64)
    checks['qualification_reject_wrong_path']=not qualification(goodq,'/tmp/other.pt','a'*64)
    checks['qualification_requires_true_bool']=not qualification({**goodq,'all_conditions_eligible':1},'/tmp/motor.pt','a'*64)
    arrival=Arrival(.003)
    arrival.tick(.001,True,0);arrival.tick(.002,True,0)
    checks['dwell_no_early_success']=not arrival.report()['arrived']
    arrival.tick(.003,True,0)
    checks['dwell_exact_endpoint_success']=arrival.report()['arrived']
    arrival=Arrival(.003);arrival.tick(.001,True,.01);arrival.tick(.002,True,.01)
    checks['preonset_not_counted']=arrival.count==0
    arrival.tick(.011,True,.01);arrival.tick(.012,False,.01);arrival.tick(.013,True,.01)
    checks['dwell_requires_contiguous']=arrival.count==1 and not arrival.report()['arrived']
    args=SimpleNamespace(motor_checkpoint='/tmp/m.pt',behavior_checkpoint='/tmp/b.pt',qualification='/tmp/q.json',layout='/tmp/l.json',out='/tmp/out',motor_sha256='a'*64,behavior_sha256='b'*64,control='normal',replay_file=None,replay_sha256=None)
    absolute_inputs(args);checks['explicit_cli_paths_digests']=True
    bad=deepcopy(args);bad.control='replay'
    checks['replay_requires_frozen_file']=rejected(lambda:absolute_inputs(bad))
    bad=deepcopy(args);bad.motor_checkpoint='relative.pt'
    checks['reject_relative_checkpoint']=rejected(lambda:absolute_inputs(bad))
    fake=SimpleNamespace(rows=[dict(time=float(i),qpos=np.array([1.,2.]),qvel=np.array([0.]),target=np.array([.1]),actualctrl=np.array([.2]),action=np.array([.3]),control_source=0) for i in range(5800)])
    prefix=prefix_digest(fake)
    checks['physical_prefix_digest_stable']=prefix==prefix_digest(fake)
    fake.rows[-1]['qpos'][0]+=.001
    checks['physical_prefix_includes_final_interval']=prefix!=prefix_digest(fake)
    checks['incomplete_physical_prefix_rejected']=rejected(lambda:prefix_digest(SimpleNamespace(rows=fake.rows[:-1])))
    import torch
    sys.path.insert(0,str(ROOT));import steering_behavior
    with tempfile.TemporaryDirectory(prefix='food-eval-offline-') as td:
        td=Path(td);layoutfile=td/'layout.json';layoutfile.write_text(json.dumps(d))
        checks['layout_parser_exact']=layout_from(layoutfile)==d
        model=steering_behavior.SteeringBehavior()
        payload=dict(format_version=1,kind='food_steering_behavior',identity=model.identity(),state_dict=model.state_dict(),
            config=dict(source_hashes={str(ROOT/'steering_behavior.py'):digest(ROOT/'steering_behavior.py')},data_hashes={str(layoutfile):digest(layoutfile)}))
        ck=td/'behavior.pt';torch.save(payload,ck)
        loaded,_=load_behavior(ck,digest(ck),steering_behavior)
        checks['behavior_schema_reload_exact']=all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
        checks['behavior_wrong_sha_rejected']=rejected(lambda:load_behavior(ck,'0'*64,steering_behavior))
        payload['kind']='wrong';torch.save(payload,ck)
        checks['behavior_wrong_kind_rejected']=rejected(lambda:load_behavior(ck,digest(ck),steering_behavior))
        payload['kind']='food_steering_behavior';payload['config']['data_hashes'][str(layoutfile)]='0'*64;torch.save(payload,ck)
        checks['behavior_data_drift_rejected']=rejected(lambda:load_behavior(ck,digest(ck),steering_behavior))
        state=model.initial_state(1)
        command,_=model.step(torch.tensor([[.2,.8]],dtype=torch.float64),state)
        checks['structural_forward_zero']=float(command[0,0])==0.
    for path in HERE.glob('*.py'):
        compile(path.read_text(),str(path),'exec');checks['syntax:'+path.name]=True
    checks['no_physics_import_or_step']=not any(name in sys.modules for name in ('mujoco','physics','student_wave'))
    report=dict(all_passed=all(bool(v) for v in checks.values()),checks={k:bool(v) for k,v in checks.items()},physical_steps=0,training_updates=0,
        source_sha256={str(p):digest(p) for p in HERE.iterdir() if p.suffix in ('.py','.json') and p.name!='OFFLINE_CHECKS.json'},
        limitations='Draft contract/loader tests only, no qualified candidate or actual layout suite; evaluator has not been physically run.')
    out=HERE/'OFFLINE_CHECKS.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),physical_steps=0,out=str(out))))
    if not report['all_passed']:raise RuntimeError('offline contract failed')

if __name__=='__main__':main()
