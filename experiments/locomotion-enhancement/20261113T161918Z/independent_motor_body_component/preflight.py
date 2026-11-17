"""Parent final600 motor first, then oneCore0integration componentinterface probe."""
import argparse
import sys
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-preflight',action='store_true');a=p.parse_args()
    if not a.execute_parent_preflight:p.error('Parent actualcomponent preflightrequired')
    cpu1();sources=frozen();verify(sources)
    if (ROOT/'PREFLIGHT.json').exists():raise FileExistsError(ROOT/'PREFLIGHT.json')
    m,payload=load_motor()
    if any(k in sys.modules for k in ('mujoco','physics','flygym','student_wave')):raise ValueError('Physicalbackend beforestrictNNload')
    sw,physics,identity,static,Peak,ast_ids=physical_stack()
    import numpy as np,torch,mujoco as mj
    recorder=Peak();c=physics.Core(adhesion=False,recorder=recorder);initial=snapshot(c);state=m.initial_state(1);before=tuple(s.clone() for s in state);checks={}
    checks['actualfinal600_state352']=payload['format_version']=='INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1' and payload['step']==600 and [s.shape[-1] for s in state]==[128,64,64,96]
    checks['original789f_frozen']=all(not p.requires_grad for p in m.old.parameters()) and m.verify_old()
    checks['new8trained_checkpoint']=len(payload['optimizer_state_dict']['state'])==8 and all(float(v['step'])==600 for v in payload['optimizer_state_dict']['state'].values())
    mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
    checks['full71opaque42finitePD']=bool(mesh.sum()==71 and np.all(c.plant.geom_rgba[mesh,3]==1.) and c.plant.nu==42 and np.isfinite(c.actuator.kp).all() and np.all(c.actuator.kp>0) and np.all(c.actuator.torque_limit>0) and np.isfinite(c.actuator.target_bounds_rad).all())
    length=body_length(c);checks['sevenbodycompiledextent']=len(length['segments'])==7 and length['length_native']>0
    x=torch.tensor(c.obs238(np.zeros(42),target_speed=.075)[None],dtype=torch.float64)
    with torch.no_grad():
        prime_mean,primed=m.prime_old(x,x.new_tensor([[1.,0.]]),state)
        checks['warm_new96held0']=not bool(primed[3].abs().max()) and prime_mean.shape==(1,42)
        for condition,command in CONDITIONS.items():
            mean,after,residual=m.step_details(x,x.new_tensor([command]),primed)
            checks[condition+'_finite42_state352']=bool(mean.shape==(1,42) and residual.shape==(1,42) and all(s.shape==(1,n) for s,n in zip(after,(128,64,64,96))) and torch.isfinite(mean).all() and torch.isfinite(residual).all() and residual.abs().max()<=6.)
    assert_same(initial,snapshot(c));checks['inputstate_notmutated']=all(torch.equal(a,b) for a,b in zip(state,before))
    warm=static.StaticTeacher(c,sw.teacher_config(3));action,diag=warm.action(c,0.)
    checks['oldwarm_finite42']=bool(action.shape==(42,) and np.isfinite(action).all() and np.max(np.abs(action))<=1.)
    assert_same(initial,snapshot(c))
    # Puremetric observer must not mutate physical state; no actual load generation.
    cr=module('_component_metric_only_preflight',BASE/'cycle_reward.py');observer=cr.CycleTracker(cr.CycleConfig(timeout=4.)).reset(start_time=float(c.d.time));measure=c.measure();observer.update(dict(time=float(c.d.time),loads=measure['loads'],padz=measure['padz'],foot=measure['foot']))
    assert_same(initial,snapshot(c));checks['observer_no_actualsideeffect']=True
    checks['zero_integration']=float(c.d.time)==0. and c.t==0 and recorder.count==0
    ident=identity.capture(c,dict(kind=FORMAT+'_ZERO_INTEGRATION_PREFLIGHT',source_hashes=sources,neural_state352=True),extra_paths=list(sources));identity.verify_runtime(ident,c);verify(sources)
    save(ROOT/'PREFLIGHT.json',dict(all_passed=bool(all(checks.values())),checks=checks,Core_constructions=1,physical_integration_steps=0,source_hashes=sources,actual_checkpoint_sha256=read(ROOT/'INPUT_BINDINGS.json')['checkpoint_sha256'],identity=ident,body_length=length,selected_legacy_ast=ast_ids,scope='One unintegratedCore/finalNNinterface only; all4bodybehaviors unknown, no fullneckdeployment'))
    if not all(checks.values()):raise ValueError('Componentinterface preflightfailed')
    print('COMPONENT_ZERO_INTEGRATION_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
