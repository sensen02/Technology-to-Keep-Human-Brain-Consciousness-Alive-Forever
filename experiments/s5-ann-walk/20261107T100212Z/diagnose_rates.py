"""Independent instantaneous position-task rate diagnosis; FK/LP only, no mj_step.
Pseudoinverse is minimum2norm, NOT minimummaxnorm. BoundedLP verifies localinstant
feasibility only, not a continuous smooth path or finite-horizon joint feasibility.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linprog
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T095519Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T100212Z')
LEGS=('LF','LM','LH','RF','RM','RH')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def periodic(p,q):return PchipInterpolator(np.r_[p[-3:-1]-2*np.pi,p,p[1:3]+2*np.pi],np.concatenate((q[-3:-1],q,q[1:3]),axis=0),axis=0,extrapolate=False)
def stats(q):
    a=np.asarray(q,float)
    return dict(max_abs=float(abs(a).max()),RMS=float(np.sqrt(np.mean(a*a))))

def main():
    sys.path.insert(0,str(BASE));import mujoco as mj
    from plant_probe import Plant
    from incremental_actuator import DEFAULT_MODEL_PATH
    paths=[SOURCE/'FK_EVIDENCE.npz',SOURCE/'CORRECTIONS.npz',SOURCE/'METADATA.json',Path(__file__),DEFAULT_MODEL_PATH]
    before={str(p):sha(p) for p in paths}
    e=load(paths[0]);c=load(paths[1]);meta=json.loads(paths[2].read_text())
    phase=e['phase'];time=phase/(2*np.pi);curve=periodic(c['phase'],c['source']+c['final'])
    assert np.allclose(curve(phase),e['qpos_hinges'],atol=1e-12,rtol=0)
    actual_velocity=curve(phase,1)*2*np.pi
    target_velocity=np.gradient(e['target_world'],time,axis=0,edge_order=2)
    offsets=np.asarray(meta['phase_offsets']);duty=float(meta['parameters']['duty'])
    local=(time[:,None]+offsets)%1;swing=(local-duty)/(1-duty)
    p=Plant();m=p.m;d=mj.MjData(m);q0=p.q0.copy();lo=m.jnt_range[p.jids,0];hi=m.jnt_range[p.jids,1];q0[p.qa]=np.clip(q0[p.qa],lo,hi)
    names=np.array([m.joint(int(j)).name for j in p.jids]);ids=[np.flatnonzero(np.char.startswith(names,'joint_'+leg)) for leg in LEGS]
    assert all(len(i)==7 for i in ids) and np.array_equal(p.qa,e['qpos_addresses'])
    selected=np.arange(0,len(phase)-1,8);cache={};calls=0
    def evaluate(i,j):
        nonlocal calls
        if (i,j) in cache:return cache[i,j]
        d.qpos[:]=q0;d.qpos[p.qa]=e['qpos_hinges'][i];mj.mj_forward(m,d);calls+=1
        assert d.time==0 and np.array_equal(d.qpos[:7],p.q0[:7])
        jp=np.zeros((3,48));mj.mj_jac(m,d,jp,None,d.xpos[p.bids[j]],p.bids[j]);J=jp[:,p.va[ids[j]]]
        v=target_velocity[i,j];minimum=np.linalg.pinv(J,rcond=1e-12)@v;actual=actual_velocity[i,ids[j]]
        singular=np.linalg.svd(J,compute_uv=False);null=(np.eye(7)-np.linalg.pinv(J,rcond=1e-12)@J)@actual
        row=dict(dense_index=int(i),global_cycle=float(time[i]),local_swing_fraction=float(swing[i,j]),nominal_swing=bool(not e['nominal_stance'][i,j]),
            target_velocity_world=v.tolist(),target_speed=float(np.linalg.norm(v)),
            minimum_2norm_qdot=minimum.tolist(),actual_qdot=actual.tolist(),qdot_min_max_abs=float(abs(minimum).max()),
            actual_qdot_max_abs=float(abs(actual).max()),position_velocity_residual=float(np.linalg.norm(J@minimum-v)),
            actual_position_velocity_residual=float(np.linalg.norm(J@actual-v)),
            actual_velocity_position_nullspace_fraction=float(np.linalg.norm(null)/max(np.linalg.norm(actual),1e-15)),
            singular_values=singular.tolist(),condition=float(singular[0]/singular[-1]) if singular[-1]>1e-15 else None)
        cache[i,j]=(row,J);return row,J
    summaries=[];lp_summary=[];all_records=[]
    for j,leg in enumerate(LEGS):
        records=[evaluate(int(i),j)[0] for i in selected]
        sw=[row for row in records if row['nominal_swing']]
        groups=dict(all=records,swing=sw,early_lift=[r for r in sw if 0<=r['local_swing_fraction']<=.25],
            horizontal_plateau=[r for r in sw if .25<r['local_swing_fraction']<.75],late_landing=[r for r in sw if .75<=r['local_swing_fraction']<=1])
        summary={}
        for label,rows in groups.items():
            if not rows:continue
            worst=max(rows,key=lambda r:r['actual_qdot_max_abs'])
            inherent=max(rows,key=lambda r:r['qdot_min_max_abs'])
            summary[label]=dict(samples=len(rows),minimum_2norm_rates=stats([r['minimum_2norm_qdot'] for r in rows]),
                actual_rates=stats([r['actual_qdot'] for r in rows]),minimum_singular_value=min(r['singular_values'][-1] for r in rows),
                maximum_position_velocity_residual=max(r['position_velocity_residual'] for r in rows),
                worst_actual=worst,worst_minimum_2norm=inherent)
        summaries.append(dict(leg=leg,joints=names[ids[j]].tolist(),groups=summary))
        all_records.extend(dict(leg=leg,**r) for r in records)
        candidates=np.flatnonzero(~e['nominal_stance'][:-1,j]);pick=candidates[np.unique(np.rint(np.linspace(0,len(candidates)-1,64)).astype(int))]
        outcomes=[]
        for i in pick:
            row,J=evaluate(int(i),j);q=e['qpos_hinges'][i,ids[j]]
            for cap in (20.,30.):
                lower=np.full(7,-cap);upper=np.full(7,cap)
                lower[q<=lo[ids[j]]+1e-7]=0.;upper[q>=hi[ids[j]]-1e-7]=0.
                result=linprog(np.zeros(7),A_eq=J,b_eq=target_velocity[i,j],bounds=list(zip(lower,upper)),method='highs')
                residual=float(abs(J@result.x-target_velocity[i,j]).max()) if result.success else None
                outcomes.append(dict(index=int(i),global_cycle=float(time[i]),swing_fraction=float(swing[i,j]),cap=cap,
                    feasible=bool(result.success and residual<=1e-6),solver_status=int(result.status),solver_message=result.message,
                    equality_residual_max=residual,qdot=result.x.tolist() if result.success else None,
                    at_bound_direction_restrictions=int(np.sum((lower==0)|(upper==0))),qdot_min_max_abs=row['qdot_min_max_abs']))
        counts={str(cap):dict(feasible=sum(r['feasible'] for r in outcomes if r['cap']==cap),tested=sum(r['cap']==cap for r in outcomes),
            infeasible=[r for r in outcomes if r['cap']==cap and not r['feasible']]) for cap in (20.,30.)}
        lp_summary.append(dict(leg=leg,counts=counts,all_results=outcomes))
        print(json.dumps(dict(leg=leg,minimum_2norm_swing_max=summary['swing']['minimum_2norm_rates']['max_abs'],actual_swing_max=summary['swing']['actual_rates']['max_abs'],LP={k:dict(feasible=v['feasible'],tested=v['tested']) for k,v in counts.items()})),flush=True)
    # Exact joint-rate peak is additionally checked, outside the512frame sample.
    maxrate=0.;peak=None
    for segment in range(len(curve.x)-1):
        left,right=curve.x[segment:segment+2]
        if left<-1e-12 or right>2*np.pi+1e-12:continue
        a,b,cc=curve.c[:3,segment];h=right-left
        for k in range(42):
            roots=[0.,h]
            if abs(a[k])>1e-20:
                value=-b[k]/(3*a[k])
                if 0<value<h:roots.append(value)
            for value in roots:
                rate=float(abs(3*a[k]*value*value+2*b[k]*value+cc[k])*2*np.pi)
                if rate>maxrate:maxrate=rate;peak=dict(joint=str(names[k]),phase=float(left+value),global_cycle=float((left+value)/(2*np.pi)))
    # Preserve exactpeak metadata; sampledJacobians characterize neighboringstates.
    report=dict(stage='S5 independent instantaneous POSITION-task rate feasibility',source=str(SOURCE/'FK_EVIDENCE.npz'),
        method=dict(global_FK_grid=512,LP_swing_frames_per_leg=64,target_velocity='central finite difference of4097saved intendedworldpositions versusphase/2pi at1Hz; second-order endpoints',
            actual_qdot='exact derivative of periodicPCHIP reconstructed from513finalknots',
            minimum_2norm='pinv(Jpos3x7)@target_velocity, byname7nativeactuators; free root fixed only forFK',
            boundedLP='scipy.linprog highs feasibility Jpos*qdot=vtarget, bounds+/-20 or30; ifqwithin1e-7 ofnativebound outwardvelocityforbidden',
            no_orientation_constraint=True),
        per_leg=summaries,LP=lp_summary,sampled_rows=all_records,
        exact_actual_rate_peak=dict(rad_s=maxrate,**peak),counts=dict(FK_forward=calls,LP_calls=sum(len(x['all_results']) for x in lp_summary),physics_steps=0,neural_steps=0),
        input_hashes=before,source_hashes_unchanged=all(sha(p)==h for p,h in before.items()),
        limitations=['Pseudoinverse minimizes2norm, not maximumjointvelocity; only boundedLPtests localmaxcaps atselectedposes.',
            'LP ignores orientation, acceleration, nextstatebounddistance, contact, dynamicstability andtrajectorycontinuity; localfeasibility doesnotestablish a globallysmooth boundedratepath.',
            'Targetvelocityfinite differences can blur corners or spikes; rawtargethasdeclaredboundarynonsmoothness. Exactactualratepeak isreportedseparately.',
            'A differentjointbranch mayhave a differentJacobian. No sourcegeometry, GAIT, thresholds, driver orphysicsstatewasmodified.'])
    output=OUT/'INSTANTANEOUS_RATE_DIAGNOSIS.json';OUT.mkdir(parents=True,exist_ok=True)
    if output.exists():output.read_bytes();raise FileExistsError('preserve existing diagnostic')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),exact_peak=report['exact_actual_rate_peak'],physics_steps=0,counts=report['counts'])),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
