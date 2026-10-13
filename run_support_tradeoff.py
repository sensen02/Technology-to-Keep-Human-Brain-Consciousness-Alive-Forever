"""Bounded same-pose COM/full acceleration tradeoff; isolated mechanics, not neural control."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import run_support_force_audit as audit
s=audit.search
OUT=s.HERE/'outputs/workspace_recheck/support_tradeoff.json'


def residual(q, com_j, g, weight):
    return np.r_[s.dimensionless(q,g), np.sqrt(weight)*(com_j@q)[2]/g]


def main(report):
    start=time.monotonic(); deadline=start+90
    before=s.cp.sha256_file(s.cp.CANONICAL_XML)
    assert before==s.cp.CANONICAL_XML_SHA256
    doc=json.loads(audit.SOURCE.read_text()); model=s.cp.load_model()
    covered,_,_=s.coverage(model); g=float(np.linalg.norm(model.opt.gravity))
    rows=[]; integrations=[]; baselines=[]
    class Budget(Exception): pass
    for source in audit.selected_rows(doc):
        ev=s.FixedPose(model,source['qpos']); ev.acceleration(ev.floor)
        _,weighted=audit.com_acceleration(model,ev.data)
        cj=weighted/model.body_mass.sum()
        baseline={}
        for condition in ('covered74','all90'):
            a=np.array(source[condition]['activation']); q=ev.acceleration(a)
            angular=float(np.sqrt(np.mean(q[3:]**2)))
            baseline[condition]=(a,angular)
            baselines.append(dict(pose=source['label'],condition=condition,angular_rms_rad_s2=angular))
        previous={}
        for weight in (0,1,10,100):
            current={}
            for condition,free in [('covered74',covered),('all90',ev.muscles)]:
                seeds=[baseline[condition][0]]
                if condition in previous: seeds.append(previous[condition])
                if condition=='all90': seeds.append(current['covered74'])
                def score(a):
                    q=ev.acceleration(a); r=residual(q,cj,g,weight)
                    return float(r@r),q
                scored=[(score(a)[0],a.copy()) for a in seeds]
                best_cost,best_a=min(scored,key=lambda x:x[0]); seed_cost=best_cost
                calls=ev.calls
                def fun(x):
                    nonlocal best_cost,best_a
                    if time.monotonic()>deadline: raise Budget()
                    a=ev.command(free,x); q=ev.acceleration(a); r=residual(q,cj,g,weight)
                    cost=float(r@r)
                    if cost<best_cost: best_cost,best_a=cost,a.copy()
                    return r
                try:
                    sol=least_squares(fun,best_a[free],bounds=(ev.floor[free],ev.upper[free]),max_nfev=100,ftol=1e-6,xtol=1e-6,gtol=1e-6)
                    solver=dict(status=int(sol.status),message=sol.message,nfev=sol.nfev)
                except Budget: solver=dict(status='TIME_BUDGET',nfev=None)
                solver['forward_calls']=ev.calls-calls
                snap=audit.snapshot(model,source['qpos'],best_a,condition)
                assert all(c['within_tolerance'] for c in snap['checks'].values())
                q=np.array(snap['raw_vectors']['qacc']); angular=float(np.sqrt(np.mean(q[3:]**2)))
                ratio=angular/baseline[condition][1]; com=float(snap['COM_acceleration_mm_s2'][2]/g)
                rows.append(dict(pose=source['label'],condition=condition,weight=weight,angular_ratio=ratio,angular_rms_rad_s2=angular,support=snap['support_over_weight'],COM_z_over_g=com,full_rms=float(np.sqrt(np.mean(s.dimensionless(q,g)**2))),objective=best_cost,seed_objective=seed_cost,activation=best_a.tolist(),solver=solver,screen_pass=bool(ratio<=1.1 and abs(com)<=.01),force_audit=snap))
                previous[condition]=best_a; current[condition]=best_a
                print(source['label'],condition,weight,'support',snap['support_over_weight'],'angular ratio',ratio,flush=True)
        for condition in ('covered74','all90'):
            candidates=[r for r in rows if r['pose']==source['label'] and r['condition']==condition and r['angular_ratio']<=1.1]
            a0=baseline[condition][0]; basecom=abs((cj@ev.acceleration(a0))[2]/g)
            chosen=min(candidates,key=lambda r:abs(r['COM_z_over_g'])) if candidates else None
            if chosen is not None and abs(chosen['COM_z_over_g'])<basecom:
                command=chosen['activation']; selection=f"weight={chosen['weight']}; smallest |COM_z/g| within angular screen"
            else: command=a0; selection='original baseline retained'
            integrations.append(dict(pose=source['label'],condition=condition,selection=selection,baseline=s.integrate(model,source['qpos'],a0),candidate=s.integrate(model,source['qpos'],command)))
    after=s.cp.sha256_file(s.cp.CANONICAL_XML); assert before==after
    result=dict(status='COMPLETED_BOUNDED_MECHANICAL_TEST',rows=rows,baselines=baselines,integrations=integrations,method=dict(weights=[0,1,10,100],L_mm=1,screen='angular RMS <=1.1 original same-condition baseline AND |COM_z/g|<=.01; engineering only',search_budget_s=90,max_nfev=100),provenance=dict(xml_before=before,xml_after=after,source=str(audit.SOURCE),source_sha256=s.cp.sha256_file(audit.SOURCE),code_sha256=s.cp.sha256_file(Path(__file__))),memory_guard=report('completed'),wall_s=time.monotonic()-start,limits=['Local finite search, not infeasibility proof','Independent muscles relax neural pool constraints','Constant command integration is not neural success','No physical parameters changed'])
    def safe(v):
        if isinstance(v,dict): return {k:safe(x) for k,x in v.items()}
        if isinstance(v,list): return [safe(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v): return str(v)
        return v
    OUT.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n')
    print(OUT,flush=True)


if __name__=='__main__':
    with s.mg.guarded_run('support_tradeoff',**s.mg.BUDGETS['neural_muscle_closure']) as report: main(report)
