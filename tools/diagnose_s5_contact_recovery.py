"""One contact response-time intervention; fresh output only, no overwrite mode."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse, importlib.util, json, sys, subprocess, time, traceback
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-baseline-recording/20261101T183002Z')
REF=Path('/home/sensen/Desktop/engine修复验收/s5-contact-eighth-step/20261101T185129Z')
DEP=ROOT/'tools/diagnose_s5_contact_resolution.py'
spec=importlib.util.spec_from_file_location('contact_helper',DEP);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
sha,dump=h.sha,h.dump

def load(path,stop=None):
    with np.load(path) as z:return {k:(z[k][:stop].copy() if stop else z[k]) for k in z.files}

class Surface:
    def __init__(self,m):
        self.m=m;self.floor=m.geom('ground_plane').id
        assert m.geom_type[self.floor]==mujoco.mjtGeom.mjGEOM_PLANE
        self.ids=np.array([m.geom('nmf/%s_tarsus%d'%(L,k)).id for L in h.LEGS for k in range(1,6)])
        assert all(m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH for g in self.ids)
    def query(self,d):
        gaps=np.array([mujoco.mj_geomDistance(self.m,d,self.floor,int(g),1000.,None) for g in self.ids])
        assert np.isfinite(gaps).all() and np.max(gaps)<999., 'unsupported/clipped geom distance'
        matrix=gaps.reshape(6,5);ix=matrix.argmin(axis=1)
        return dict(surface_gap_by_geom=gaps,surface_gap_min=matrix.min(axis=1),surface_gap_tarsus5=matrix[:,4].copy(),surface_argmin_geom=self.ids.reshape(6,5)[np.arange(6),ix])
    def vertex_projection(self,d):
        m=self.m;normal=d.geom_xmat[self.floor].reshape(3,3)[:,2];answer=[]
        for g in self.ids:
            mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];v=m.mesh_vert[a:a+m.mesh_vertnum[mid]]
            answer.append(float(np.min(v@(d.geom_xmat[g].reshape(3,3).T@normal))+(d.geom_xpos[g]-d.geom_xpos[self.floor])@normal))
        return np.array(answer)

class Runner(h.Runner):
    def __init__(self,m,a,ident):
        super().__init__(m,a,ident);self.surface=Surface(m)
    def sample(self):
        r=super().sample();r.update(self.surface.query(self.obs));return r

def phases(a,ident):
    t=a['time'];width=np.diff(t);n=len(t)-1;result={}
    for j,L in enumerate(h.LEGS):
        aid=ident['actuators'].index('nmf/%s_tarsus5-adhesion'%L);st=a['ctrl'][:-1,aid]>.5;loaded=a['signed_foot_fz'][:-1,j]>1e-9
        edges=np.r_[0,np.flatnonzero(st[1:]!=st[:-1])+1,n];starts=np.flatnonzero(loaded&~np.r_[False,loaded[:-1]]);ends=np.flatnonzero(loaded&~np.r_[loaded[1:],False])+1;rows=[]
        for k,(s,e) in enumerate(zip(edges[:-1],edges[1:])):
            overlap=np.flatnonzero((starts<e)&(ends>s));dur=float(t[e]-t[s]);gap=a['surface_gap_min'][s:e,j];w=width[s:e]
            rows.append(dict(phase='stance' if st[s] else 'swing',start_time=float(t[s]),end_time=float(t[e]),left_censored=bool(k==0),right_censored=bool(k==len(edges)-2),duration_s=dur,contact_time_s=float(w[loaded[s:e]].sum()),contact_duty=float(w[loaded[s:e]].sum()/dur),signed_fz_integral=float(a['signed_foot_fz'][s:e,j]@w),new_onsets=int(np.sum((starts>=s)&(starts<e)&(starts>0))),max_contiguous_contact_s=max((float(t[min(e,ends[i])]-t[max(s,starts[i])]) for i in overlap),default=0.),surface_gap_quantiles=np.quantile(gap,[0,.25,.5,.75,1]).tolist(),surface_nonpositive_time_s=float(w[gap<=0].sum()),surface_positive_within_margin_time_s=float(w[(gap>0)&(gap<=.001)].sum()),surface_beyond_margin_time_s=float(w[gap>.001].sum()),tarsus5_gap_quantiles=np.quantile(a['surface_gap_tarsus5'][s:e,j],[0,.25,.5,.75,1]).tolist()))
        sm={}
        for ph in ('stance','swing'):
            rs=[r for r in rows if r['phase']==ph];mask=st if ph=='stance' else ~st;dur=float(width[mask].sum());gap=a['surface_gap_min'][:-1,j][mask];q=np.quantile(gap,[0,.25,.5,.75,1]).tolist()
            sm[ph]=dict(duration_s=dur,contact_time_s=sum(r['contact_time_s'] for r in rs),contact_duty=sum(r['contact_time_s'] for r in rs)/dur,signed_fz_integral=sum(r['signed_fz_integral'] for r in rs),new_onsets=sum(r['new_onsets'] for r in rs),max_contiguous_in_any_halfcycle_fragment_s=max(r['max_contiguous_contact_s'] for r in rs),complete_halfcycles=sum(not r['left_censored'] and not r['right_censored'] for r in rs),surface_gap_quantiles=q,surface_nonpositive_time_s=sum(r['surface_nonpositive_time_s'] for r in rs),surface_positive_within_margin_time_s=sum(r['surface_positive_within_margin_time_s'] for r in rs),surface_beyond_margin_time_s=sum(r['surface_beyond_margin_time_s'] for r in rs))
        result[L]=dict(halfcycles=rows,summary=sm)
    return result

def manifest(out):
    paths=[p for p in out.iterdir() if p.is_file() and p.name not in ('MANIFEST_SHA256.json','RUN.log')]+[Path(__file__).resolve(),DEP,ROOT/'engine/embodied/contact_metrics.py']
    dump(out/'MANIFEST_SHA256.json',{str(p):dict(sha256=sha(p),bytes=p.stat().st_size,importance='重要辅助' if p.suffix in ('.md','.log') or p.name.endswith('READONLY.py') else '核心必需') for p in sorted(paths)})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out.resolve()
    if out in (BASE.resolve(),REF.resolve()) or BASE.resolve() in out.parents or REF.resolve() in out.parents:raise ValueError('frozen inputs cannot be output')
    out.mkdir(parents=True,exist_ok=True)
    if any(p.name not in ('PREREGISTRATION.json','RUN.log') for p in out.iterdir()):raise ValueError('fresh preregistered output required')
    prereg=json.loads((out/'PREREGISTRATION.json').read_text());begin=time.monotonic()
    try:
        assert sha(DEP)==prereg['dependency_sha256']
        m0=mujoco.MjModel.from_binary_path(str(REF/'model.mjb'));m=mujoco.MjModel.from_binary_path(str(REF/'model.mjb'))
        assert m.opt.timestep==.0000125 and np.all(m.pair_solref[:,0]==.0002)
        m.pair_solref[:,0]=.0004
        differences={}
        for k in dir(m0):
            v=getattr(m0,k)
            if isinstance(v,np.ndarray) and not np.array_equal(v,getattr(m,k)):
                differences[k]=dict(indices=np.argwhere(v!=getattr(m,k)).tolist(),old=v.tolist(),new=getattr(m,k).tolist())
        assert set(differences)=={'pair_solref'}
        assert differences['pair_solref']['indices']==[[i,0] for i in range(m.npair)]
        dump(out/'MODEL_DIFF.json',dict(only_change='explicit pair_solref timeconstant',arrays=differences,pairs=[dict(index=i,name=m.pair(i).name,geom1=m.geom(int(m.pair_geom1[i])).name,geom2=m.geom(int(m.pair_geom2[i])).name) for i in range(m.npair)]))
        ident=json.loads((REF/'MODEL_IDENTITY.json').read_text())
        with np.load(BASE/'trajectory.npz') as z:a={k:z[k][:1].copy() for k in ('qpos','qvel','ctrl','act','xfrc_applied','qfrc_applied','qacc_warmstart','time','filtered_load')}
        r=Runner(m,a,ident);initial=dict(integration_state=h.state(m,r.d),filtered_load=r.lf,qn=r.qn);oldini=load(REF/'INITIAL_STATE.npz')
        assert all(np.array_equal(v,oldini[k]) for k,v in initial.items())
        np.savez_compressed(out/'INITIAL_STATE.npz',**initial);mujoco.mj_saveModel(m,str(out/'model.mjb'),None)
        sources=[REF/'model.mjb',REF/'trajectory.npz',REF/'INITIAL_STATE.npz',REF/'MODEL_IDENTITY.json',REF/'MANIFEST_SHA256.json',BASE/'trajectory.npz',DEP,ROOT/'engine/embodied/contact_metrics.py']
        dump(out/'ENVIRONMENT.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=sha(__file__),python=sys.version,executable=sys.executable,mujoco=mujoco.__version__,numpy=np.__version__,MUJOCO_GL=os.getenv('MUJOCO_GL'),PYTHONDONTWRITEBYTECODE=os.getenv('PYTHONDONTWRITEBYTECODE'),inputs={str(p):sha(p) for p in sources}))
        dump(out/'MODEL_IDENTITY.json',dict(base_identity=ident,dt=.0000125,timeconstant=.0004,damping_ratio=1.,model_sha256=sha(out/'model.mjb'),initial_sha256=sha(out/'INITIAL_STATE.npz'),surface_geom_ids=r.surface.ids.tolist(),surface_geom_names=[m.geom(int(g)).name for g in r.surface.ids],plane_geom_id=r.surface.floor,surface_definition='minimum signed compiled collision mesh-plane distance; 5 tarsus geoms per leg; not body origin; separate tarsus5',state_spec=int(h.STATE)))
        reference=load(REF/'trajectory.npz',20001);surf=Surface(m0);d=mujoco.MjData(m0)
        indices=sorted(set([0,10000,20000]+[int(np.flatnonzero(reference['geometry_flag'][:,j])[0]) for j in range(6) if reference['geometry_flag'][:,j].any()]))
        errors=[];distances=[]
        for i in indices:
            d.qpos[:]=reference['qpos'][i];mujoco.mj_kinematics(m0,d);query=surf.query(d)['surface_gap_by_geom'];exact=surf.vertex_projection(d)
            errors.append(float(np.max(np.abs(query-exact))));distances.extend(query.tolist())
        dump(out/'SURFACE_PREFLIGHT.json',dict(states=indices,queries=len(distances),max_api_vs_compiled_vertex_error=max(errors),signed_range=[min(distances),max(distances)],no_integration=True,reference_reconstruction='mj_kinematics on saved qpos',no_mesh_transform_double_apply=True))
        assert max(errors)<1e-9 and min(distances)<0 and max(distances)>0
        # Sole new physical integration.
        new=r.run(.25,out/'trajectory.npz')
        # Reference: offline kinematics only, no stepping.
        surface_arrays=None
        for i in range(20001):
            d.qpos[:]=reference['qpos'][i];mujoco.mj_kinematics(m0,d);q=surf.query(d)
            if surface_arrays is None:surface_arrays={k:np.empty((20001,)+v.shape,dtype=v.dtype) for k,v in q.items()}
            for k,v in q.items():surface_arrays[k][i]=v
        np.savez_compressed(out/'REFERENCE_SURFACE.npz',time=reference['time'],**surface_arrays)
        reference.update(surface_arrays)
        # Match the new terminal held-control convention without reintegration.
        d.qpos[:]=reference['qpos'][-1];d.qvel[:]=reference['qvel'][-1];d.ctrl[:]=reference['ctrl'][-2];d.qacc_warmstart[:]=reference['qacc_warmstart'][-1];d.time=reference['time'][-1]
        mujoco.mj_forward(m0,d);observer=h.Runner(m0,a,ident)
        endpoint=observer.forces(d)
        adjustment={k:np.asarray(v).tolist() for k,v in endpoint.items()}
        adjustment.update(ctrl_held=reference['ctrl'][-2].tolist(),leg_u_held=reference['leg_u'][-2].tolist(),original_ctrl_max_change=float(np.max(np.abs(reference['ctrl'][-1]-reference['ctrl'][-2]))),policy='endpoint force static forward at same qpos/qvel, held penultimate ctrl; no duration or step')
        dump(out/'REFERENCE_ENDPOINT.json',adjustment)
        reference['ctrl'][-1]=reference['ctrl'][-2];reference['leg_u'][-1]=reference['leg_u'][-2]
        for k,v in endpoint.items():reference[k][-1]=v
        summaries={};ep={};phase={}
        for label,data in [('reference_tau0p2ms',reference),('intervention_tau0p4ms',new)]:
            summaries[label],ep[label]=h.analyze(data);phase[label]=phases(data,ident)
        dump(out/'SUMMARY.json',summaries);dump(out/'CONTACT_EPISODES.json',ep);dump(out/'PHASE_STATISTICS.json',phase);dump(out/'TRAJECTORY_DEVIATIONS.json',h.difference(reference,new,1))
        dump(out/'RUN_STATUS.json',dict(status='ONE_NEW_MECHANISM_ARM_COMPLETED',new_steps=20000,reference_steps=0,reference_kinematics_states=20001,dt=.0000125,duration_s=.25,elapsed_s=time.monotonic()-begin,no_parameter_scan=True,no_gait_claim=True))
        print('completed',json.dumps(summaries['intervention_tau0p4ms']['paths']),'elapsed_s',time.monotonic()-begin,flush=True)
        return 0
    except Exception:
        dump(out/'FAILURE.json',dict(traceback=traceback.format_exc(),elapsed_s=time.monotonic()-begin));raise
    finally:manifest(out)

if __name__=='__main__':sys.exit(main())
