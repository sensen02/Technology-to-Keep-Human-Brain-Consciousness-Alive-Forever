"""S2/S5: absolute-derivative audit, one conditional native equilibrium continuation.
Run with venv_body/bin/python -B; --verify performs a separate fresh validation.
No physical or neural integration. Original search and evidence are read-only.
"""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, time, traceback, shutil
import numpy as np
import mujoco
from scipy.optimize import least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf
from search_native_muscle_equilibrium import S, EXPECTED, LEGS, sha, dump, parameter_digest

OLD = Path('/home/sensen/Desktop/engine修复验收/s5-native-equilibrium-search/20261101T193630Z')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-derivative-continuation/20261101T200110Z')
BLOCKS = {'qacc':slice(0,48), 'clearance':slice(48,89), 'side':slice(89,125),
          'load':slice(125,126), 'regularizer':slice(126,169)}

class Experiment:
    def __init__(self, verify=False):
        self.verify=verify
        self.counts=dict(model_load_attempt=1,model_load=0,mj_resetData=0,forward=0,
                         geom_distance=0,contact_force=0,least_squares=0,jacobian_calls=0,
                         body_step=0,neural_step=0)
        self.roles={}; self.best=None; self.feasible=None
        assert sha(S/'MODEL.xml')==EXPECTED
        self.m=_load_mjcf(str(S/'MODEL.xml')).compile(); self.counts['model_load']=1
        m=self.m
        assert (m.nq,m.nv,m.nu,m.na)==(49,48,90,90)
        self.before,self.before_rows=parameter_digest(m)
        self.d=mujoco.MjData(m)
        self.v=json.loads((OLD/'VARIABLES.json').read_text())
        v=self.v
        self.lo=np.array(v['lower']); self.hi=np.array(v['upper']); self.G=np.array(v['group_matrix'])
        self.qa=np.array(v['hinge_qpos_addresses'],int); self.mapped=self.G.sum(1)>0
        assert self.G.shape==(90,66) and self.mapped.sum()==74 and len(self.lo)==109
        with np.load(S/'BODY.npz') as z:self.qseed=z['qpos'][v['seed_index']].copy()
        self.seedpose=np.r_[self.qseed[self.qa],self.qseed[2]]
        self.floor=m.geom('floor').id; self.thorax=m.body('Thorax').id
        self.foot={m.geom(n).id:LEGS.index(l) for n,l in v['foot_geom_to_leg'].items()}
        self.nonfoot=np.array([m.geom(n).id for n in v['nonfoot_geoms']])
        self.sideb=np.array([m.body(n).id for n in v['side_bodies']])
        self.sign=np.array([1 if n.startswith(('LF','LM','LH')) else -1 for n in v['side_bodies']])
        self.weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        assert self.weight==v['weight_native']
        hinges=np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
        assert np.array_equal(m.jnt_qposadr[hinges],self.qa)
        assert np.array_equal(self.lo[:42],m.jnt_range[hinges,0]) and np.array_equal(self.hi[:42],m.jnt_range[hinges,1])
        assert np.array_equal(m.actuator_actadr,np.arange(90))
        mapping=json.loads((S/'MAPPING.json').read_text())['rows']; groups={}
        for row in mapping:
            assert m.actuator(int(row['actuator_index'])).name==row['actuator']
            if row['local_indices']:groups.setdefault(tuple(row['local_indices']),[]).append(int(row['actuator_index']))
        expected_G=np.zeros_like(self.G)
        for j,ids in enumerate(groups.values()):expected_G[ids,j]=1
        assert np.array_equal(self.G,expected_G)
        self.log=(OUT/('VERIFY_EVALUATIONS.jsonl' if verify else 'EVALUATIONS.jsonl')).open('x')

    def evaluate(self,x,role='optimizer',retain=False):
        m,d=self.m,self.d; x=np.asarray(x)
        self.roles[role]=self.roles.get(role,0)+1
        mujoco.mj_resetData(m,d);self.counts['mj_resetData']+=1
        d.qpos[:]=self.qseed;d.qpos[self.qa]=x[:42];d.qpos[2]=x[42];d.qvel[:]=0
        activation=np.full(90,.0001);activation[self.mapped]=(self.G@x[43:])[self.mapped]
        d.act[:]=activation;d.ctrl[:]=activation;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
        mujoco.mj_forward(m,d);self.counts['forward']+=1
        distances=np.array([mujoco.mj_geomDistance(m,d,int(g),self.floor,1000.,None) for g in self.nonfoot])
        self.counts['geom_distance']+=len(self.nonfoot)
        side=((d.xpos[self.sideb]-d.xpos[self.thorax])@d.xmat[self.thorax].reshape(3,3)[:,1])*self.sign
        ff=np.zeros((6,3));nf=np.zeros(3);nfmax=0.;contacts=[];topology=[]
        for ci in range(d.ncon):
            c=d.contact[ci];a,b=int(c.geom1),int(c.geom2)
            topology.append([a,b,int(c.dim),int(c.efc_address)])
            if self.floor not in (a,b):continue
            f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);self.counts['contact_force']+=1
            world=c.frame.reshape(3,3).T@f[:3]*(1 if a==self.floor else -1)
            other=b if a==self.floor else a
            if other in self.foot:ff[self.foot[other]]+=world
            else:nf+=world;nfmax=max(nfmax,float(abs(world).max()))
            contacts.append(dict(geoms=[a,b],dist=float(c.dist),pos=c.pos.tolist(),frame=c.frame.tolist(),local_force=f.tolist(),world_force=world.tolist()))
        load=float(ff[:,2].sum()/self.weight)
        r=np.r_[d.qacc/9810.,100*np.maximum(0,.02-distances),100*np.maximum(0,.02-side),10*(load-1),1e-6*(x[:43]-self.seedpose)]
        vd=[];normal=d.geom_xmat[self.floor].reshape(3,3)[:,2];origin=d.geom_xpos[self.floor]
        for g in self.nonfoot:
            mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
            verts=m.mesh_vert[start:start+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            vd.append(float(np.min((verts-origin)@normal)))
        vd=np.array(vd);cross=float(abs(vd-distances).max())
        arrays=dict(x=x.copy(),residual=r.copy(),qpos=d.qpos.copy(),qvel=d.qvel.copy(),activation=d.act.copy(),ctrl=d.ctrl.copy(),
                    act_dot=d.act_dot.copy(),qacc=d.qacc.copy(),qfrc_applied=d.qfrc_applied.copy(),xfrc_applied=d.xfrc_applied.copy(),
                    actuator_force=d.actuator_force.copy(),qfrc_actuator=d.qfrc_actuator.copy(),qfrc_constraint=d.qfrc_constraint.copy(),
                    qfrc_passive=d.qfrc_passive.copy(),qfrc_bias=d.qfrc_bias.copy(),foot_force=ff,nonfoot_force=nf,
                    nonfoot_distances=distances,compiled_vertex_distances=vd,body_side=side,solver_niter=d.solver_niter.copy())
        finite=all(np.isfinite(a).all() for a in arrays.values())
        gates=dict(all48_acceleration=bool(np.all(abs(d.qacc)<=.1)),nonfoot_clearance=bool(np.all(distances>=.02)),
                   body_sides=bool(np.all(side>=.02)),foot_load=.99<=load<=1.01,nonfoot_force_zero=nfmax==0,
                   all109_bounds=bool(np.all(x>=self.lo) and np.all(x<=self.hi)),
                   activation_bounds=bool(np.all(d.act>=.0001) and np.all(d.act<=1)),
                   pool_binding=bool(np.array_equal(d.act[self.mapped],(self.G@x[43:])[self.mapped])),
                   unmapped_floor=bool(np.all(d.act[~self.mapped]==.0001)),act_ctrl_equal=bool(np.array_equal(d.act,d.ctrl)),
                   activation_steady=bool(np.max(abs(d.act_dot))<=1e-10),external_forces_zero=bool(not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)),
                   finite=finite,no_warnings=not np.any(d.warning.number),geom_distance_cutoff_zero=bool(np.all(distances<1000)),mesh_plane_crosscheck=cross<=1e-8)
        cost=float(r@r)
        row=dict(evaluation=self.counts['forward'],role=role,cost=cost,x=x.tolist(),residual=r.tolist(),qacc=d.qacc.tolist(),
                 max_abs_qacc=float(abs(d.qacc).max()),foot_load_ratio=load,min_nonfoot_distance=float(distances.min()),min_side=float(side.min()),
                 nonfoot_force_max_abs=nfmax,act_dot_max_abs=float(abs(d.act_dot).max()),crosscheck_error=cross,
                 solver_niter=d.solver_niter.tolist(),contact_topology=topology,contacts=contacts,gates=gates,
                 observed_physics_feasible=all(gates.values()))
        self.log.write(json.dumps(row,allow_nan=False)+'\n');self.log.flush()
        if not finite:raise FloatingPointError('nonfinite evaluation')
        if not self.verify and role!='final_validation':
            if self.best is None or cost<self.best['cost']:
                self.best=dict(cost=cost,x=x.copy(),evaluation=row['evaluation'])
                np.savez_compressed(OUT/'BEST_INCREMENTAL.npz',**arrays)
            if all(gates.values()) and (self.feasible is None or cost<self.feasible['cost']):
                self.feasible=dict(cost=cost,evaluation=row['evaluation'])
                np.savez_compressed(OUT/'BEST_FEASIBLE_OBSERVED.npz',**arrays)
                print('FEASIBLE_CANDIDATE',json.dumps(self.feasible),flush=True)
        return (r,row,arrays) if retain else r

    def final(self,x,prefix='FINAL'):
        self.d=mujoco.MjData(self.m)
        r,row,arrays=self.evaluate(x,'independent_validation' if self.verify else 'final_validation',True)
        m,d=self.m,self.d
        moment=np.zeros((m.nu,m.nv))
        for j in range(m.nu):
            a=int(d.moment_rowadr[j]);n=int(d.moment_rownnz[j]);moment[j,d.moment_colind[a:a+n]]=d.actuator_moment[a:a+n]
        reconstruction=moment.T@d.actuator_force
        err=float(abs(reconstruction-d.qfrc_actuator).max())
        after,rows=parameter_digest(m)
        gates=row['gates'].copy();gates.update(moment_reconstruction=err<=1e-9,model_parameters_unchanged=after==self.before,xml_unchanged=sha(S/'MODEL.xml')==EXPECTED)
        arrays.update(moment=moment,reconstructed_qfrc_actuator=reconstruction,group_matrix=self.G,seed_qpos=self.qseed,hinge_qpos_addresses=self.qa)
        np.savez_compressed(OUT/(prefix+'_STATE.npz'),**arrays)
        result=dict(passed=all(gates.values()),gates=gates,metrics=row,moment_reconstruction_error=err,model_before=self.before,model_after=after)
        dump(OUT/(prefix+'_GATE.json'),result)
        dump(OUT/(prefix+'_MODEL_PARAMETERS.json'),dict(before=self.before,after=after,fields=rows))
        return result


def diagnose(e,x):
    base,row,arrays=e.evaluate(x,'A_fresh_best',True)
    with np.load(OLD/'FINAL_STATE.npz') as old:
        errors={k:float(np.max(abs(arrays[k]-old[k]))) for k in ['x','residual','qpos','qacc','activation','foot_force','nonfoot_distances','body_side']}
    dump(OUT/'A_REPRODUCTION.json',dict(errors=errors,metrics=row))
    assert max(errors.values())<=1e-8
    epss=[1e-5,1e-6,1e-7];js=[];mismatches=[];stats=[]
    assert np.all(x-1e-5>=e.lo) and np.all(x+1e-5<=e.hi)
    for eps in epss:
        jp=np.zeros((169,109));jm=np.zeros_like(jp)
        for j in range(109):
            xp=x.copy();xm=x.copy();xp[j]+=eps;xm[j]-=eps
            jp[:,j]=(e.evaluate(xp,f'A_eps{eps}_plus')-base)/eps
            jm[:,j]=(base-e.evaluate(xm,f'A_eps{eps}_minus'))/eps
        jac=(jp+jm)/2;norm=np.linalg.norm(jac,axis=0)
        mismatch=np.linalg.norm(jp-jm,axis=0)/np.maximum(np.maximum(np.linalg.norm(jp,axis=0),np.linalg.norm(jm,axis=0)),1e-8)
        js.append(jac);mismatches.append(mismatch)
        blocks={name:dict(sse=float(base[sl]@base[sl]),gradient_sse=2*jac[sl].T@base[sl],column_norms=np.linalg.norm(jac[sl],axis=0)) for name,sl in BLOCKS.items()}
        stats.append(dict(eps=eps,column_norms=norm,gradient_sse=2*jac.T@base,one_sided_relative_difference=mismatch,blocks=blocks))
        np.savez_compressed(OUT/f'A_JAC_{eps:.0e}.npz',eps=eps,x=x,residual=base,jacobian=jac,forward=jp,backward=jm,one_sided_relative_difference=mismatch)
    pairs=[]
    for i in range(2):
        rel=np.linalg.norm(js[i]-js[i+1],axis=0)/np.maximum(np.maximum(np.linalg.norm(js[i],axis=0),np.linalg.norm(js[i+1],axis=0)),1e-8)
        stable=bool(np.isfinite(js[i]).all() and np.isfinite(js[i+1]).all() and rel.max()<=.1 and max(mismatches[i].max(),mismatches[i+1].max())<=.2)
        pairs.append(dict(indices=[i,i+1],eps=[epss[i],epss[i+1]],column_relative_difference=rel,max_relative_difference=float(rel.max()),stable=stable))
    stable=[p for p in pairs if p['stable']]
    choice=min(stable,key=lambda p:p['max_relative_difference']) if stable else None
    decision=dict(stages=stats,pairs=pairs,continue_B=choice is not None)
    if choice:
        idx=choice['indices'][1];norm=np.linalg.norm(js[idx],axis=0);floor=max(1e-8,float(norm.max())*1e-8)
        decision.update(selected_absolute_eps=epss[idx],x_scale=np.clip(1/np.maximum(norm,floor),1e-8,1e8),column_norm_floor=floor)
    dump(OUT/'A_DERIVATIVE_DIAGNOSTICS.json',decision)
    return decision


def run():
    assert (OUT/'PREREGISTRATION.json').is_file()
    assert not (OUT/'EXECUTION.json').exists() and not (OUT/'EVALUATIONS.jsonl').exists()
    source=Path(__file__).resolve();original=source.with_name('search_native_muscle_equilibrium.py')
    shutil.copyfile(source,OUT/source.name);shutil.copyfile(original,OUT/'BASE_search_native_muscle_equilibrium.py')
    shutil.copyfile(OLD/'VARIABLES.json',OUT/'VARIABLES.json')
    inputs={str(p):sha(p) for p in [source,original,S/'MODEL.xml',S/'BODY.npz',S/'MAPPING.json',OLD/'VARIABLES.json',OLD/'FINAL_STATE.npz',OUT/'PREREGISTRATION.json']}
    dump(OUT/'INPUT_IDENTITIES.json',inputs)
    e=None;error=None;gate=None;sol=None;started=time.time()
    try:
        e=Experiment();dump(OUT/'MODEL_PARAMETERS_BEFORE.json',dict(sha256=e.before,fields=e.before_rows))
        with np.load(OLD/'FINAL_STATE.npz') as z:x=z['x'].copy()
        decision=diagnose(e,x)
        if decision['continue_B']:
            eps=decision['selected_absolute_eps'];scale=np.array(decision['x_scale'])
            dump(OUT/'B_DECISION.json',dict(absolute_eps=eps,x_scale=scale,max_nfev=150,start=x,reason='preregistered stability test passed'))
            def jacobian(at):
                e.counts['jacobian_calls']+=1
                out=np.empty((169,109));base=None
                for j in range(109):
                    if at[j]-eps>=e.lo[j] and at[j]+eps<=e.hi[j]:
                        plus=at.copy();minus=at.copy();plus[j]+=eps;minus[j]-=eps
                        out[:,j]=(e.evaluate(plus,'B_jac_plus')-e.evaluate(minus,'B_jac_minus'))/(2*eps)
                    else:
                        if base is None:base=e.evaluate(at,'B_jac_base')
                        sign=1 if e.hi[j]-at[j]>=at[j]-e.lo[j] else -1
                        h=min(eps,(e.hi[j]-at[j] if sign==1 else at[j]-e.lo[j])/2)
                        assert h>0
                        p=at.copy();q=at.copy();p[j]+=sign*h;q[j]+=2*sign*h
                        out[:,j]=sign*(-3*base+4*e.evaluate(p,'B_jac_inward1')-e.evaluate(q,'B_jac_inward2'))/(2*h)
                return out
            e.counts['least_squares']+=1
            sol=least_squares(lambda at:e.evaluate(at,'B_optimizer'),x,jac=jacobian,bounds=(e.lo,e.hi),x_scale=scale,
                              method='trf',loss='linear',max_nfev=150,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            dump(OUT/'SOLVER_RESULT.json',dict(success=bool(sol.success),status=int(sol.status),message=sol.message,nfev=sol.nfev,njev=sol.njev,
                 optimality=sol.optimality,cost_half_sse=sol.cost,x=sol.x,best=e.best,observed_feasible=e.feasible))
            _,_,solver_arrays=e.evaluate(sol.x,'solver_x_validation',True)
            np.savez_compressed(OUT/'SOLVER_STATE.npz',**solver_arrays)
        else:
            dump(OUT/'B_SKIPPED.json',dict(reason='Neither adjacent absolute-difference pair meets preregistered stability thresholds. Report to parent; no blind optimizer.'))
        gate=e.final(e.best['x'])
    except Exception:
        error=traceback.format_exc();dump(OUT/'FAILURE.json',dict(error=error))
    finally:
        if e:e.log.close()
        unchanged=all(sha(Path(p))==h for p,h in inputs.items())
        execution=dict(error=error,counts=e.counts if e else None,roles=e.roles if e else None,wall_seconds=time.time()-started,
                       input_hashes_unchanged=unchanged,best=e.best if e else None,observed_feasible=e.feasible if e else None,
                       engineering_static_gate_passed=bool(gate and gate['passed'] and unchanged and error is None),exit_code=0 if error is None and unchanged else 1)
        dump(OUT/'EXECUTION.json',execution)
        print(json.dumps(execution,default=lambda a:a.tolist(),ensure_ascii=False),flush=True)
    return execution['exit_code']


def verify():
    assert not (OUT/'INDEPENDENT_VERIFICATION.json').exists()
    e=Experiment(True)
    with np.load(OUT/'FINAL_STATE.npz') as z:saved={k:z[k].copy() for k in z.files}
    result=e.final(saved['x'],'VERIFY')
    with np.load(OUT/'VERIFY_STATE.npz') as z:errors={k:float(np.max(abs(z[k]-a))) for k,a in saved.items() if a.size}
    rows=[json.loads(line) for line in (OUT/'EVALUATIONS.jsonl').open()]
    execution=json.loads((OUT/'EXECUTION.json').read_text())
    costs=np.array([r['cost'] for r in rows]);selection=[r for r in rows if r['role']!='final_validation']
    best=min(selection,key=lambda r:r['cost'])
    checks=dict(fresh_arrays_match=max(errors.values())<=1e-8,minimum_objective_selected=abs(best['cost']-float(saved['residual']@saved['residual']))<=1e-12,
                all_counts_match=len(rows)==execution['counts']['forward']==execution['counts']['mj_resetData'],
                A_probes_654=sum(r['role'].startswith('A_eps') for r in rows)==654,
                variables_bytes_unchanged=sha(OUT/'VARIABLES.json')==sha(OLD/'VARIABLES.json'),
                no_steps=execution['counts']['body_step']==execution['counts']['neural_step']==0,
                all_costs_reconstruct=all(abs(np.dot(r['residual'],r['residual'])-r['cost'])<=1e-12 for r in rows),
                all_x_in_bounds=all(np.all(np.array(r['x'])>=e.lo) and np.all(np.array(r['x'])<=e.hi) for r in rows),
                original_inputs_unchanged=all(sha(Path(p))==h for p,h in json.loads((OUT/'INPUT_IDENTITIES.json').read_text()).items()))
    e.log.close()
    dump(OUT/'INDEPENDENT_VERIFICATION.json',dict(passed=all(checks.values()),checks=checks,array_repeat_errors=errors,counts=e.counts,
         static_gate_passed=result['passed'],observed_feasible_count=sum(r['observed_physics_feasible'] for r in rows)))
    report='# S2/S5 绝对差分与原生静态平衡续算\n\n'
    report+='静态硬门：'+('通过' if result['passed'] else '未通过')+'。独立重建与记录一致性：'+('通过' if all(checks.values()) else '未通过')+'。零身体/神经步，未物理rollout；不是自主站立或行走。绝对力单位生物标定仍 NOT_ESTABLISHED。\n\n'
    report+='原模型、原目标、原VARIABLES字节不变。A固定654绝对中心差分探针；B仅在预登记导数判据通过后单次max_nfev150。全部真实评估含Jac探针纳入最佳选择，solver.x与observed feasible另存。局部失败不证明全局无解。\n\n'
    report+='执行台账：\n```json\n'+json.dumps({k:v for k,v in execution.items() if k!='best'},ensure_ascii=False,indent=2)+'\n```\n\n'
    report+='独立fresh门：\n```json\n'+json.dumps(result['gates'],ensure_ascii=False,indent=2)+'\n```\n\n'
    report+='重要程度：核心必需为源码、预登记、VARIABLES、EVALUATIONS、A_JAC/A诊断、BEST/FINAL/SOLVER状态、FINAL_GATE与INDEPENDENT_VERIFICATION；重要辅助为身份、模型指纹、EXECUTION、REPORT和清单。所有证据均保留供审计，无确认可弃文件，未执行删除。旧证据/原工具/tracker未改，无提交。\n'
    (OUT/'REPORT.zh-CN.md').write_text(report)
    dump(OUT/'MANIFEST_SHA256.json',{str(p):dict(sha256=sha(p),importance='核心必需' if p.suffix in ('.npz','.py','.jsonl') or p.name in ('PREREGISTRATION.json','FINAL_GATE.json','INDEPENDENT_VERIFICATION.json','A_DERIVATIVE_DIAGNOSTICS.json','VARIABLES.json') else '重要辅助') for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='MANIFEST_SHA256.json'})
    print(json.dumps(dict(checks=checks,counts=e.counts,static_gate_passed=result['passed']),ensure_ascii=False),flush=True)
    return 0 if all(checks.values()) else 1

if __name__=='__main__':
    sys.exit(verify() if '--verify' in sys.argv else run())
