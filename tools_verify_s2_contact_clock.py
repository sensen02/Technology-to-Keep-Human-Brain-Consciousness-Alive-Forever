"""Read-only independent archive arithmetic; no stepping and no runner helpers."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco
from tools_s2_switched_activation import reference
import engine.canonical_physics as cp


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();out=args.directory.resolve()
    report=json.loads((out/'report.json').read_text());checks=[];summaries=[];loaded={}
    def check(name,value):
        checks.append(dict(name=name,pass_check=bool(value)))
    check('exact eight cases',len(report['records'])==8)
    expected=np.arange(1,401)*2.5e-5
    pose_path=next(Path(s) for s in report['source_hashes'] if s.endswith('initial_pose.npz'))
    pose=np.load(pose_path,allow_pickle=False)['projected_seated_qpos']
    for path,h in report['source_hashes'].items():
        check('source '+path,hashlib.sha256(Path(path).read_bytes()).hexdigest()==h)
    for r in report['records']:
        key=(r['branch'],r['case'],r['dt_s']);a=dict(np.load(out/r['npz'],allow_pickle=False));loaded[key]=a
        tag=r['npz'];meta=r['compiled'];dt=r['dt_s'];n=round(.01/dt);idx=r['target_actadr']
        check(tag+' finite',all(np.isfinite(v).all() for v in a.values()))
        check(tag+' expected',np.array_equal(a['expected_time_s'],expected))
        check(tag+' endpoint clock',np.allclose(a['time_s'],expected,rtol=0,atol=1e-14))
        check(tag+' all step clocks',np.allclose(a['step_time_s'],np.arange(n)*dt,rtol=0,atol=1e-14))
        check(tag+' initial pose',np.array_equal(a['initial_qpos'],pose))
        check(tag+' initial velocity',np.array_equal(a['initial_qvel'],np.zeros(meta['nv'])))
        init=np.zeros(meta['na']);init[idx]=.1 if r['case']=='decay' else 0.
        check(tag+' initial act',np.array_equal(a['initial_act'],init))
        check(tag+' initial ctrl',np.array_equal(a['initial_ctrl'],np.zeros(meta['nu'])))
        ctrl=np.zeros((n,meta['nu']))
        if r['case']=='pulse':ctrl[:round(.001/dt),0]=.1
        check(tag+' complete submitted schedule',np.array_equal(a['step_ctrl'],ctrl))
        ratio=round(2.5e-5/dt)
        check(tag+' endpoint ctrl',np.array_equal(a['ctrl'],ctrl[ratio-1::ratio]))
        check(tag+' RK4',meta['integrator']==1 and meta['timestep_s']==dt)
        floor=.0001 if r['branch']=='native_floor' else 0.
        check(tag+' compiled floor',np.all(np.asarray(meta['actuator_ctrlrange'])[:,0]==floor))
        check(tag+' target mapping',meta['activation_indices'][0]==[idx] and r['target_actuator']==0)
        check(tag+' dynprm',meta['actuator_dynprm'][0][:3]==[.0001,.0004,0.])
        ref=reference(expected,r['case'],floor);strict=reference(expected,r['case'],floor,True)
        check(tag+' stored reference',np.allclose(a['reference'],ref,rtol=0,atol=1e-14))
        error=float(np.max(np.abs(a['activation']-ref)))
        check(tag+' engineering error',abs(error-r['activation_max_error'])<1e-14)
        check(tag+' engineering decision',r['engineering_screen_pass']==(error<=.001))
        check(tag+' stricter reference agreement',np.max(np.abs(ref-strict))<1e-12)
        check(tag+' warnings',not np.any(a['warning_counts']))
        contacts=json.loads((out/r['contacts']).read_text())
        check(tag+' contacts count',len(contacts)==400)
        check(tag+' contacts finite/pair direction',all(len(v)==3 and np.isfinite(v).all() and int(k.split(',')[0])<=int(k.split(',')[1]) for row in contacts for k,v in row.items()))
        summaries.append(dict(npz=tag,activation_max_error=error,screen_pass=error<=.001,strict_reference_max_difference=float(np.max(np.abs(ref-strict)))))
    m=cp.load_model()
    for r in report['comparisons']:
        branch,case=r['branch'],r['case'];a=loaded[(branch,case,2.5e-5)];b=loaded[(branch,case,1.25e-5)]
        delta=np.zeros((400,m.nv))
        for i in range(400):mujoco.mj_differentiatePos(m,delta[i],1.,a['qpos'][i],b['qpos'][i])
        check(branch+case+' labels cover DOFs',[x['dof'] for x in r['dofs']]==list(range(m.nv)))
        for x in r['dofs']:
            i=x['dof'];j=int(m.dof_jntid[i]);kind=int(m.jnt_type[j]);component=i-int(m.jnt_dofadr[j])
            unit='model_length' if kind==2 or (kind==0 and component<3) else 'rad'
            check(f'{branch}{case} dof{i}',x['unit']==unit and x['component']==component and x['joint']==mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) and x['velocity_unit']==unit+'/s' and abs(x['pose_max_difference']-np.max(np.abs(delta[:,i])))<1e-14 and abs(x['velocity_max_difference']-np.max(np.abs(a['qvel'][:,i]-b['qvel'][:,i])))<1e-14 and abs(x['constraint_generalized_force_max_difference']-np.max(np.abs(a['qfrc_constraint'][:,i]-b['qfrc_constraint'][:,i])))<1e-14)
        rec=[x for x in report['records'] if x['branch']==branch and x['case']==case]
        ac=json.loads((out/rec[0]['contacts']).read_text());bc=json.loads((out/rec[1]['contacts']).read_text())
        presence=0;series=[]
        for c,d in zip(ac,bc):
            presence+=len(set(c)^set(d));series.append(max((np.linalg.norm(np.asarray(c.get(pair,[0,0,0]))-d.get(pair,[0,0,0])) for pair in set(c)|set(d)),default=0.))
        check(branch+case+' pair-force series',np.allclose(series,r['force_difference_series'],rtol=0,atol=1e-14))
        check(branch+case+' pair-force max',abs(max(series)-r['pair_force_max_norm_difference'])<1e-14)
        check(branch+case+' presence',presence==r['pair_presence_changes'])
        check(branch+case+' ncon',int(np.sum(a['ncon']!=b['ncon']))==r['ncon_unequal_endpoints'])
    passed=all(c['pass_check'] for c in checks)
    with (out/'independent-check.json').open('x') as f:json.dump(dict(pass_check=passed,check_count=len(checks),checks=checks,activation=summaries,scope='Independent array arithmetic, live source hashes, no force extraction re-simulation; not biological/standing/convergence verification'),f,indent=2)
    print(json.dumps(dict(pass_check=passed,check_count=len(checks),activation=summaries),indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
