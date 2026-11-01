"""One preregistered native3470 static joint-pose/pool-activation search.
No physical integration, neural calls, model edits, or runtime controller.
"""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, hashlib, time, traceback
import numpy as np
import mujoco
from scipy.optimize import least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf

S = Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-native-equilibrium-search/20261101T193630Z')
EXPECTED = '3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
LEGS = ('LF','LM','LH','RF','RM','RH')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

def parameter_digest(m):
    """Fingerprint all exposed model ndarrays/scalars and option scalars/arrays."""
    rows = {}
    for prefix, obj in [('model',m),('option',m.opt)]:
        for name in dir(obj):
            if name.startswith('_'): continue
            value = getattr(obj,name)
            if isinstance(value,np.ndarray):
                a=np.ascontiguousarray(value)
                rows[prefix+'.'+name]=dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
            elif isinstance(value,(int,float,bool,str)):
                rows[prefix+'.'+name]=value
    encoded=json.dumps(rows,sort_keys=True,allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(), rows

def main():
    assert (O/'PREREGISTRATION.json').is_file(), 'Preregistration must exist before execution'
    assert not (O/'EXECUTION.json').exists() and not (O/'EVALUATIONS.jsonl').exists(), 'Never restart this attempt'
    prereg=json.loads((O/'PREREGISTRATION.json').read_text())
    source=Path(__file__).resolve()
    (O/'search_native_muscle_equilibrium.py').write_bytes(source.read_bytes())
    inputs={str(p):sha(p) for p in [S/'MODEL.xml',S/'BODY.npz',S/'MAPPING.json',O/'PREREGISTRATION.json',source]}
    dump(O/'INPUT_IDENTITIES.json',inputs)
    counts=dict(model_load_attempt=0,model_load=0,mj_resetData=0,forward=0,
        optimizer_callbacks=0,final_validation=0,geom_distance=0,contact_force=0,
        least_squares=0,body_step=0,neural_step=0)
    started=time.time(); error=None; result=None; model_before=None; model_after=None
    model_rows_before=None; m=None; best=None; gate=None; model_unchanged=False
    try:
        assert inputs[str(S/'MODEL.xml')]==EXPECTED
        counts['model_load_attempt']+=1
        m=_load_mjcf(str(S/'MODEL.xml')).compile();counts['model_load']+=1
        assert (m.nq,m.nv,m.nu,m.na)==(49,48,90,90)
        model_before,model_rows_before=parameter_digest(m)
        dump(O/'MODEL_PARAMETERS_BEFORE.json',dict(sha256=model_before,fields=model_rows_before))
        d=mujoco.MjData(m)
        with np.load(S/'BODY.npz') as z:
            seed_index=int(np.argmin(abs(z['time']-.15)))
            seed_time=float(z['time'][seed_index]);qseed=z['qpos'][seed_index].copy()
        assert abs(seed_time-.15)<1e-10
        hinges=np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
        assert len(hinges)==42 and np.all(m.jnt_limited[hinges])
        qa=m.jnt_qposadr[hinges].astype(int)
        va=m.jnt_dofadr[hinges].astype(int)
        assert np.array_equal(np.sort(qa),np.arange(7,49))
        assert np.array_equal(m.actuator_actadr,np.arange(90))
        groups={};mapping=json.loads((S/'MAPPING.json').read_text())['rows']
        for row in mapping:
            assert m.actuator(int(row['actuator_index'])).name==row['actuator']
            if row['local_indices']:
                groups.setdefault(tuple(row['local_indices']),[]).append(int(row['actuator_index']))
        assert len(groups)==66
        G=np.zeros((90,66))
        for j,ids in enumerate(groups.values()): G[ids,j]=1
        mapped=G.sum(1)>0
        assert mapped.sum()==74 and np.all(G.sum(1)<=1)
        floor_id=m.geom('floor').id
        assert m.geom_type[floor_id]==mujoco.mjtGeom.mjGEOM_PLANE
        foot={}
        for g in range(m.ngeom):
            name=m.geom(g).name;bn=m.body(int(m.geom_bodyid[g])).name
            legs=[i for i,L in enumerate(LEGS) if (name.startswith(L) or bn.startswith(L)) and 'Tarsus' in name+bn]
            if len(legs)==1: foot[g]=legs[0]
        assert len(foot)==30
        nonfoot=np.array([g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and g not in foot],int)
        assert len(nonfoot)==41
        side_bodies=[];side_sign=[]
        for b in range(m.nbody):
            name=m.body(b).name
            if any(name.startswith(L) for L in LEGS) and ('Tibia' in name or 'Tarsus' in name):
                side_bodies.append(b);side_sign.append(1. if name.startswith(('LF','LM','LH')) else -1.)
        side_bodies=np.array(side_bodies,int);side_sign=np.array(side_sign)
        assert len(side_bodies)==36
        thorax=m.body('Thorax').id;weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        seed_pose=np.r_[qseed[qa],qseed[2]]
        lower=np.r_[m.jnt_range[hinges,0],qseed[2]-.25,np.full(66,.0001)]
        upper=np.r_[m.jnt_range[hinges,1],qseed[2]+.25,np.ones(66)]
        x0=np.r_[seed_pose,np.full(66,.1)]
        raw_initial=x0.copy()
        x0[:42]=np.clip(x0[:42],lower[:42],upper[:42])
        changes=np.flatnonzero(x0!=raw_initial)
        dump(O/'SEED_BOUND_PROJECTION.json',dict(raw_seed_qpos=qseed,raw_initial=raw_initial,solver_initial=x0,
            changes=[dict(variable=int(i),joint=m.joint(int(hinges[i])).name,old=float(raw_initial[i]),new=float(x0[i]),delta=float(x0[i]-raw_initial[i])) for i in changes],
            reason='Preregistered minimal native-bound projection before sole optimizer; model unchanged'))
        assert np.all(x0>=lower) and np.all(x0<=upper)
        dump(O/'VARIABLES.json',dict(seed_index=seed_index,seed_time=seed_time,hinge_names=[m.joint(int(j)).name for j in hinges],
            hinge_qpos_addresses=qa,hinge_dof_addresses=va,lower=lower,upper=upper,initial=x0,group_matrix=G,
            foot_geom_to_leg={m.geom(g).name:LEGS[l] for g,l in foot.items()},nonfoot_geoms=[m.geom(int(g)).name for g in nonfoot],
            side_bodies=[m.body(int(b)).name for b in side_bodies],side_frame='Thorax local y',weight_native=weight))
        eval_log=O/'EVALUATIONS.jsonl'
        best_feasible=None
        def evaluate(x, role='optimizer', retain=False):
            nonlocal best, best_feasible
            if role=='optimizer': counts['optimizer_callbacks']+=1
            else: counts['final_validation']+=1
            mujoco.mj_resetData(m,d);counts['mj_resetData']+=1
            d.qpos[:]=qseed;d.qpos[qa]=x[:42];d.qpos[2]=x[42]
            d.qvel[:]=0
            activation=np.full(90,.0001);activation[mapped]=(G@x[43:])[mapped]
            d.act[:]=activation;d.ctrl[:]=activation
            d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
            mujoco.mj_forward(m,d);counts['forward']+=1
            distances=np.array([mujoco.mj_geomDistance(m,d,int(g),floor_id,1000.,None) for g in nonfoot])
            counts['geom_distance']+=len(nonfoot)
            side=((d.xpos[side_bodies]-d.xpos[thorax])@d.xmat[thorax].reshape(3,3)[:,1])*side_sign
            ff=np.zeros((6,3));nf=np.zeros(3);nonfoot_max=0.;contact_rows=[]
            for ci in range(d.ncon):
                c=d.contact[ci];g1=int(c.geom1);g2=int(c.geom2)
                if floor_id not in (g1,g2):continue
                other=g2 if g1==floor_id else g1
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);counts['contact_force']+=1
                force=c.frame.reshape(3,3).T@f[:3]*(1 if g1==floor_id else -1)
                if other in foot:ff[foot[other]]+=force
                else:nf+=force;nonfoot_max=max(nonfoot_max,float(np.max(abs(force))))
                contact_rows.append(np.r_[g1,g2,c.dist,c.pos,force])
            load=float(ff[:,2].sum()/weight)
            r=np.r_[d.qacc/9810.,100*np.maximum(0.,.02-distances),100*np.maximum(0.,.02-side),10*(load-1),1e-6*(x[:43]-seed_pose)]
            if not np.isfinite(r).all() or not np.isfinite(d.actuator_force).all():
                np.savez_compressed(O/'NONFINITE_FAILURE.npz',x=x,qpos=d.qpos,qacc=d.qacc,residual=r)
                raise FloatingPointError('Nonfinite static evaluation')
            cost=float(r@r)
            row=dict(evaluation=counts['forward'],role=role,cost=cost,max_abs_qacc=float(abs(d.qacc).max()),
                min_nonfoot_distance=float(distances.min()),min_side=float(side.min()),foot_load_ratio=load,
                nonfoot_force_max_abs=nonfoot_max,act_dot_max_abs=float(abs(d.act_dot).max()),warnings=int(d.warning.number.sum()),
                geom_distance_cutoff_hits=int(np.sum(distances>=1000.)))
            observed_feasible=(row['max_abs_qacc']<=.1 and np.all(distances>=.02) and np.all(side>=.02)
                and nonfoot_max==0 and .99<=load<=1.01 and row['act_dot_max_abs']<=1e-10 and row['warnings']==0
                and np.all(x>=lower) and np.all(x<=upper) and row['geom_distance_cutoff_hits']==0)
            if role=='optimizer' and observed_feasible and (best_feasible is None or cost<best_feasible['cost']):
                best_feasible=dict(cost=cost,evaluation=counts['forward'],scope='observed physics gates; not selected instead of minimum objective; final independent validation remains required')
                np.savez_compressed(O/'BEST_FEASIBLE_OBSERVED.npz',x=x,qpos=d.qpos,activation=d.act,qacc=d.qacc,residual=r)
            row['observed_physics_feasible']=bool(observed_feasible)
            with eval_log.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if role=='optimizer' and (best is None or cost<best['cost']):
                best=dict(cost=cost,x=x.copy(),evaluation=counts['forward'])
                np.savez_compressed(O/'BEST_INCREMENTAL.npz',x=x,residual=r,qpos=d.qpos,qvel=d.qvel,activation=d.act,ctrl=d.ctrl,
                    qacc=d.qacc,foot_force=ff,nonfoot_distances=distances,body_side=side,evaluation=counts['forward'],cost=cost)
                if counts['forward']==1 or counts['forward']%100==0:print('BEST',row,flush=True)
            if retain:
                return r,row,dict(x=x.copy(),qpos=d.qpos.copy(),qvel=d.qvel.copy(),activation=d.act.copy(),ctrl=d.ctrl.copy(),
                    act_dot=d.act_dot.copy(),qacc=d.qacc.copy(),qfrc_actuator=d.qfrc_actuator.copy(),qfrc_passive=d.qfrc_passive.copy(),
                    qfrc_bias=d.qfrc_bias.copy(),qfrc_constraint=d.qfrc_constraint.copy(),qfrc_applied=d.qfrc_applied.copy(),xfrc_applied=d.xfrc_applied.copy(),
                    actuator_force=d.actuator_force.copy(),foot_force=ff,nonfoot_force=nf,nonfoot_distances=distances,
                    body_side=side,contact_rows=np.array(contact_rows).reshape(-1,9),residual=r.copy())
            return r
        counts['least_squares']+=1
        sol=least_squares(evaluate,x0,bounds=(lower,upper),**prereg['solver'])
        result=dict(success=bool(sol.success),status=int(sol.status),message=str(sol.message),nfev=int(sol.nfev),
            njev=int(sol.njev),optimality=float(sol.optimality),cost_half_sse=float(sol.cost),x=sol.x,
            selected_evaluation=best['evaluation'],selected_sse=best['cost'],best_feasible_observed=best_feasible)
        dump(O/'SOLVER_RESULT.json',result)
        d=mujoco.MjData(m)  # Independent fresh data, still the same single compiled model.
        r,row,arrays=evaluate(best['x'],role='final_validation',retain=True)
        moment=np.zeros((m.nu,m.nv))
        assert len(d.moment_rownnz)==m.nu, 'MuJoCo sparse moment rows must be actuator rows'
        for j in range(m.nu):
            start=int(d.moment_rowadr[j]);n=int(d.moment_rownnz[j])
            moment[j,d.moment_colind[start:start+n]]=d.actuator_moment[start:start+n]
        reconstruction=moment.T@d.actuator_force
        reconstruction_error=float(abs(reconstruction-d.qfrc_actuator).max())
        vertex_distances=[]
        plane_normal=d.geom_xmat[floor_id].reshape(3,3)[:,2];plane_point=d.geom_xpos[floor_id]
        for g in nonfoot:
            mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
            vertices=m.mesh_vert[start:start+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            vertex_distances.append(float(np.min((vertices-plane_point)@plane_normal)))
        vertex_distances=np.asarray(vertex_distances)
        dump(O/'MESH_DISTANCE_CROSSCHECK.json',dict(max_abs_error=float(np.max(abs(vertex_distances-arrays['nonfoot_distances']))),
            cutoff_hits=int(np.sum(arrays['nonfoot_distances']>=1000.)),compiled_vertex_distance=vertex_distances,
            geom_distance=arrays['nonfoot_distances'],method='compiled mesh vertices transformed once by geom pose; floor plane normal column2; zero extra forward/model calls'))
        arrays.update(moment=moment,reconstructed_qfrc_actuator=reconstruction,group_matrix=G,seed_qpos=qseed,hinge_qpos_addresses=qa,compiled_vertex_distances=vertex_distances)
        np.savez_compressed(O/'FINAL_STATE.npz',**arrays)
        model_after,rows_after=parameter_digest(m);model_unchanged=model_before==model_after
        dump(O/'MODEL_PARAMETERS_AFTER.json',dict(sha256=model_after,fields=rows_after))
        gate=dict(all48_acceleration=bool(np.all(abs(d.qacc)<=.1)),nonfoot_clearance=bool(np.all(arrays['nonfoot_distances']>=.02)),
            nonfoot_force_zero=row['nonfoot_force_max_abs']==0.,foot_load=.99<=row['foot_load_ratio']<=1.01,
            joint_bounds=bool(np.all(d.qpos[qa]>=m.jnt_range[hinges,0]) and np.all(d.qpos[qa]<=m.jnt_range[hinges,1])),
            body_sides=bool(np.all(arrays['body_side']>=.02)),activation_bounds=bool(np.all(d.act>=.0001) and np.all(d.act<=1.)),
            unmapped_floor=bool(np.all(d.act[~mapped]==.0001)),act_ctrl_equal=bool(np.array_equal(d.act,d.ctrl)),
            activation_steady=bool(np.max(abs(d.act_dot))<=1e-10),external_forces_zero=bool(not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)),
            moment_reconstruction=reconstruction_error<=1e-9,no_warnings=not np.any(d.warning.number),
            model_parameters_unchanged=model_unchanged,xml_unchanged=sha(S/'MODEL.xml')==EXPECTED,
             geom_distance_cutoff_zero=row['geom_distance_cutoff_hits']==0,
             mesh_plane_crosscheck=float(np.max(abs(vertex_distances-arrays['nonfoot_distances'])))<=1e-8)
        dump(O/'FINAL_GATE.json',dict(passed=all(gate.values()),gates=gate,metrics=row,moment_reconstruction_error=reconstruction_error,
            fresh_evaluation='New independent MjData, mj_resetData, full best-state reconstruction and mj_forward; same single compiled model',
            selected_sse_repeat_error=abs(row['cost']-best['cost'])))
    except Exception:
        error=traceback.format_exc();dump(O/'FAILURE.json',dict(error=error,counts=counts))
    finally:
        if m is not None and model_before is not None and model_after is None:
            model_after,rows_after=parameter_digest(m);model_unchanged=model_before==model_after
            dump(O/'MODEL_PARAMETERS_AFTER.json',dict(sha256=model_after,fields=rows_after))
        unchanged=all(Path(p).is_file() and sha(p)==h for p,h in inputs.items())
        passed=gate is not None and all(gate.values()) and unchanged and error is None
        execution=dict(counts=counts,wall_seconds=time.time()-started,source_hashes_unchanged=unchanged,
            model_parameters_before=model_before,model_parameters_after=model_after,model_parameters_unchanged=model_unchanged,
            engineering_static_gate_passed=passed,execution_exit_code=0 if error is None and unchanged else 1,error=error)
        dump(O/'EXECUTION.json',execution)
        text='# 原生3470姿态与肌肉联合静态搜索\n\n'
        text+='工程静态筛查：'+('通过' if passed else '未通过')+'。不代表自主神经站立或行走；未执行任何物理积分或神经步。\n\n'
        text+='模型与plant参数保持原样；只进行一次预登记least_squares。最终选择所有评估（含差分探针）中完整目标最小者，未根据验收门重新选择。局部失败不证明全局无解。\n\n'
        text+='调用与执行：\n```json\n'+json.dumps(execution,ensure_ascii=False,indent=2)+'\n```\n\n'
        if result is not None:text+='求解器摘要：\n```json\n'+json.dumps({k:v for k,v in result.items() if k!='x'},ensure_ascii=False,indent=2)+'\n```\n\n'
        if gate is not None:text+='逐项门：\n```json\n'+json.dumps(gate,ensure_ascii=False,indent=2)+'\n```\n\n'
        text+='核心必需：'+str(O/'PREREGISTRATION.json')+'；'+str(O/'EVALUATIONS.jsonl')+'；'+str(O/'BEST_INCREMENTAL.npz')+'；'+str(O/'FINAL_STATE.npz')+'；'+str(O/'FINAL_GATE.json')+'（失败时以实际存在文件为准）。\n\n'
        text+='重要辅助：输入身份、模型参数指纹、源码归档、执行台账及SHA256清单。没有删除历史文件或生成轨迹/视频；无tracker或commit修改。\n'
        (O/'REPORT.zh-CN.md').write_text(text)
        dump(O/'MANIFEST_SHA256.json',{str(p):sha(p) for p in sorted(O.iterdir()) if p.is_file() and p.name!='MANIFEST_SHA256.json'})
        print(json.dumps(execution,ensure_ascii=False,indent=2),flush=True)
    return 0 if error is None and unchanged else 1

if __name__=='__main__':sys.exit(main())
