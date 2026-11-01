"""S1/S5 timestep diagnosis. Fresh output only; never analyze into sealed evidence."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse, ast, hashlib, json, platform, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import mujoco
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收/s5-baseline-recording/20261101T183002Z')
OLD = Path('/home/sensen/Desktop/engine修复验收/s5-walk-push/20261101T145318Z/SCRIPTS/walk_push.py')
sys.path.insert(0, str(ROOT))
from engine.embodied.contact_metrics import contact_episodes
LEGS = ('lf','lm','lh','rf','rm','rh')
TA = ('lf','rm','lh')
STATE = mujoco.mjtState.mjSTATE_INTEGRATION

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def state(m,d):
    v=np.empty(mujoco.mj_stateSize(m,STATE)); mujoco.mj_getState(m,d,v,STATE); return v

def restore(m, a):
    d=mujoco.MjData(m)
    for k in ('qpos','qvel','ctrl','act','xfrc_applied','qfrc_applied','qacc_warmstart'):
        getattr(d,k)[:]=a[k][0]
    d.time=float(a['time'][0])
    return d

class Runner:
    def __init__(self, m, a, ident):
        self.m=m; self.d=restore(m,a); self.obs=mujoco.MjData(m)
        self.lf=a['filtered_load'][0].copy(); self.plan=ident['plan']
        self.sg={ast.literal_eval(k):v for k,v in ident['signs'].items()}
        self.qn=m.key_qpos[m.key('neutral').id].copy()
        self.root=m.body('nmf/c_thorax').id
        self.feet=[m.body('nmf/%s_tarsus5'%L).id for L in LEGS]
        self.floor=set(np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE))
        self.fly={g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('nmf/')}
        self.leggeom={g:j for g in self.fly for j,L in enumerate(LEGS) if '/%s_'%L in m.geom(g).name and 'tarsus' in m.geom(g).name}
        self.adh={i:next(j for j,L in enumerate(LEGS) if '/%s_'%L in m.actuator(i).name) for i in range(m.nu) if 'adhesion' in (m.actuator(i).name or '')}
        self.wleg=float(sum(m.body_mass[1:]))*9810.0/6
        self.alpha=1-np.exp(-m.opt.timestep/.020)
        self.u=np.zeros(6)
    def control(self):
        m,d=self.m,self.d
        stance=[]
        for j,L in enumerate(LEGS):
            s=np.sin(2*np.pi*4*d.time+(0 if L in TA else np.pi))
            st=s<0; stance.append(st)
            reflex=-min(.5,2*self.lf[j]/self.wleg) if st else 0.
            self.u[j]=float(np.clip(reflex+s,-1,1))
        for p in self.plan:
            tau=20*(self.qn[p['qadr']]-d.qpos[p['qadr']])-.5*d.qvel[p['dadr']]
            if p['leg'] and p['bias']:
                tau+=self.u[LEGS.index(p['leg'])]*abs(p['bias'])*self.sg[(p['leg'],p['name'])]
            d.ctrl[p['act']]=tau
        for i,j in self.adh.items(): d.ctrl[i]=1. if stance[j] else 0.
    def forces(self,d):
        absfz=np.zeros(6); signed=np.zeros(6); normal=np.zeros(6); support=np.zeros(3)
        cand=np.zeros(6,dtype=int); geom=np.zeros(6,dtype=bool); dist=np.ones(6); nonfoot=0
        for k in range(d.ncon):
            c=d.contact[k]; g1,g2=int(c.geom1),int(c.geom2)
            if g1 not in self.floor and g2 not in self.floor: continue
            other=g2 if g1 in self.floor else g1
            f=np.zeros(6); mujoco.mj_contactForce(self.m,d,k,f)
            world=c.frame.reshape(3,3).T@f[:3]
            flyforce=world*(int(g2 in self.fly)-int(g1 in self.fly))
            support+=flyforce
            if other in self.leggeom:
                j=self.leggeom[other]; absfz[j]+=abs(world[2]); signed[j]+=flyforce[2]; normal[j]+=max(0.,f[0])
                cand[j]+=1; geom[j]|=c.dist<=0; dist[j]=min(dist[j],float(c.dist))
            else: nonfoot+=1
        return dict(old_abs_fz=absfz,signed_foot_fz=signed,normal_force=normal,signed_support=support,candidate_count=cand,geometry_flag=geom,min_candidate_dist=dist,nonfoot=nonfoot)
    def sample(self):
        m,d=self.m,self.obs
        mujoco.mj_copyData(d,m,self.d); mujoco.mj_forward(m,d)
        r={k:np.array(getattr(d,k),copy=True) for k in ('time','qpos','qvel','ctrl','act','actuator_force','qacc_warmstart')}
        r.update(filtered_load=self.lf.copy(),leg_u=self.u.copy(),root=d.xpos[self.root].copy(),feet=d.xpos[self.feet].copy(),com=(d.xipos*m.body_mass[:,None]).sum(axis=0)/m.body_mass.sum(),warning_number=self.d.warning.number.copy(),observation_warning_number=d.warning.number.copy())
        r.update(self.forces(d)); return r
    def run(self,duration,out):
        n=round(duration/self.m.opt.timestep); arr=None; begin=time.monotonic()
        for i in range(n+1):
            if i<n: self.control()
            sample=self.sample()
            if arr is None: arr={k:np.empty((n+1,)+np.shape(v),dtype=np.asarray(v).dtype) for k,v in sample.items()}
            for k,v in sample.items(): arr[k][i]=v
            if i==n: break
            mujoco.mj_step(self.m,self.d)
            load=self.forces(self.d)['old_abs_fz']; self.lf+=self.alpha*(load-self.lf)
            if (i+1)%10000==0: print(out.name,'steps',i+1,'/',n,'wall_s',round(time.monotonic()-begin,2),flush=True)
            if (np.any(self.d.warning.number) or np.any(sample['observation_warning_number'])
                    or not all(np.isfinite(v).all() for v in sample.values())
                    or not np.isfinite(state(self.m,self.d)).all()):
                np.savez_compressed(out.with_name(out.stem+'_FAILED.npz'),
                    failure_post_step_integration_state=state(self.m,self.d),
                    failure_post_step_warnings=self.d.warning.number.copy(),
                    **{k:v[:i+1] for k,v in arr.items()})
                raise RuntimeError('warning/nonfinite: stopping arm '+str(out))
        np.savez_compressed(out,**arr)
        if (not all(np.isfinite(v).all() for v in arr.values())
                or np.any(arr['warning_number']) or np.any(arr['observation_warning_number'])):
            raise RuntimeError('invalid recorded trajectory including endpoint '+str(out))
        return arr

def analyze(a):
    t=a['time']; duration=float(t[-1]-t[0]); dt=float(np.median(np.diff(t)))
    defs={'abs_force':a['old_abs_fz']>1e-9,'signed_force':a['signed_foot_fz']>1e-9,'normal_force':a['normal_force']>1e-9,'candidate':a['candidate_count']>0,'geometry':a['geometry_flag'].astype(bool)}
    episodes={}; summary={}
    for definition, flags in defs.items():
        episodes[definition]={}; summary[definition]={}
        for j,L in enumerate(LEGS):
            r=contact_episodes(t.tolist(),flags[:,j].tolist(),a['feet'][:,j,:2].tolist())
            complete=[e['duration'] for e in r['episodes'] if not e['left_censored'] and not e['right_censored']]
            q=np.quantile(complete,[0,.25,.5,.75,1]).tolist() if complete else None
            s={k:v for k,v in r.items() if k!='episodes'}
            s.update(total_episodes=len(r['episodes']),onsets_per_s=sum(not e['left_censored'] for e in r['episodes'])/duration,complete_duration_quantiles_s=q,short_complete_lt60ms=sum(v<.060 for v in complete),mean_complete_duration_steps=(np.mean(complete)/dt if complete else None))
            episodes[definition][L]=r; summary[definition][L]=s
    paths={}
    for k in ('root','com'):
        delta=a[k][-1]-a[k][0]; path=float(np.linalg.norm(np.diff(a[k][:,:2],axis=0),axis=1).sum())
        paths[k]=dict(delta=delta.tolist(),xy_path=path,xy_efficiency=float(np.linalg.norm(delta[:2])/path) if path else None)
    fp=np.linalg.norm(np.diff(a['feet'][:,:,:2],axis=0),axis=2).sum(axis=0)
    paths['feet_xy_path']=fp.tolist(); paths['abs_root_dx_over_foot_path']=(abs(a['root'][-1,0]-a['root'][0,0])/fp).tolist()
    paths['com_z_min_ratio']=float(a['com'][:,2].min()/a['com'][0,2])
    return dict(dt=dt,duration=duration,samples=len(t),contacts=summary,paths=paths,drive_sign_changes_per_s=(np.count_nonzero(np.diff(np.sign(a['leg_u'][:-1]),axis=0),axis=0)/duration).tolist(),warning_peak=a['warning_number'].max(axis=0).tolist(),signed_support_z_mean=float(a['signed_support'][:-1,2].mean())),episodes

def baseline_geometry(a, ident):
    n=len(a['time']); a['candidate_count']=np.zeros((n,6),dtype=int); a['geometry_flag']=np.zeros((n,6),dtype=bool); a['normal_force']=np.zeros((n,6))
    gm={g:j for g,name in enumerate(ident['geoms']) for j,L in enumerate(LEGS) if '/%s_'%L in name and 'tarsus' in name}
    floor=ident['geoms'].index('ground_plane')
    for p in sorted(BASE.glob('contacts_*.npz')):
        with np.load(p) as z:
            for row in z['contacts']:
                i,g1,g2=int(row[0]),int(row[2]),int(row[3])
                other=g2 if g1==floor else g1 if g2==floor else -1
                if other in gm:
                    j=gm[other]; a['candidate_count'][i,j]+=1; a['geometry_flag'][i,j]|=row[4]<=0; a['normal_force'][i,j]+=max(0.,row[17])
    return a

def difference(a,b,stride):
    result={'time_max_abs':float(np.max(np.abs(a['time']-b['time'][::stride])))}
    for k in ('qpos','qvel','root','com','feet','ctrl','filtered_load'):
        diff=a[k]-b[k][::stride]
        if k=='ctrl': diff=diff[:-1]  # endpoint holds controls from different dt times
        result[k]=dict(max_abs=float(np.max(np.abs(diff))),rms=float(np.sqrt(np.mean(diff**2))),final_max_abs=float(np.max(np.abs(diff[-1]))))
    return result

def manifest(out):
    files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size,importance='核心必需' if p.suffix in ('.npz','.mjb','.json','.py') else '重要辅助') for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('MANIFEST_SHA256.json','RUN.log')}
    files[str(Path(__file__).resolve())]=dict(sha256=sha(__file__),importance='核心必需')
    dump(out/'MANIFEST_SHA256.json',files)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True); args=parser.parse_args(); out=args.out.resolve()
    if out==BASE or BASE in out.parents: raise ValueError('frozen baseline output prohibited')
    out.mkdir(parents=True,exist_ok=True)
    # Only a prewritten protocol is allowed; no overwrite/analyze-only entry.
    if any(p.name not in ('PREREGISTRATION.json','RUN.log') for p in out.iterdir()): raise ValueError('output must be fresh, with only preregistration and log')
    prereg=json.loads((out/'PREREGISTRATION.json').read_text())
    started=time.monotonic()
    try:
        inputs=[BASE/'trajectory.npz',BASE/'model.mjb',BASE/'MODEL_IDENTITY.json',BASE/'REPORT.zh-CN.md',OLD,ROOT/'tools/record_s5_baseline.py',ROOT/'engine/embodied/contact_metrics.py']
        ident=json.loads((BASE/'MODEL_IDENTITY.json').read_text())
        with np.load(BASE/'trajectory.npz') as z: a={k:z[k] for k in z.files}
        m=mujoco.MjModel.from_binary_path(str(BASE/'model.mjb')); runner=Runner(m,a,ident)
        env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),mujoco=mujoco.__version__,numpy=np.__version__,MUJOCO_GL=os.getenv('MUJOCO_GL'),PYTHONDONTWRITEBYTECODE=os.getenv('PYTHONDONTWRITEBYTECODE'),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=sha(__file__),inputs={str(p):sha(p) for p in inputs})
        dump(out/'ENVIRONMENT.json',env)
        np.savez_compressed(out/'INITIAL_STATE.npz',integration_state=state(m,runner.d),filtered_load=runner.lf,qn=runner.qn)
        missing={k:list(getattr(runner.d,k).shape) for k in ('act','history','plugin_state','mocap_pos','mocap_quat','userdata','eq_active')}
        if any(np.prod(shape,dtype=int)>0 for shape in missing.values()):
            raise RuntimeError('unsupported nonempty optional integration state: '+str(missing))
        dump(out/'RESTORE_AUDIT.json',dict(initial_state_sha256=sha(out/'INITIAL_STATE.npz'),state_spec=int(STATE),optional_state_shapes=missing,qn_source='neutral key_qpos',sg_source='baseline MODEL_IDENTITY.signs',model_options=str(m.opt),note='saved warmstart is from clone after forward; recovery gate empirically tests sufficiency; no hidden nonempty optional state'))
        recovery=runner.run(.020,out/'recovery_0p1ms.npz')
        n=len(recovery['time']); checks={k:float(np.max(np.abs(recovery[k]-a[k][:n]))) for k in prereg['recovery_gate']['absolute_max_tolerance']}
        flags=bool(np.array_equal(recovery['old_abs_fz'][:-1]>1e-9,a['old_abs_fz'][:n-1]>1e-9))
        # Endpoint ctrl is held in the recording convention, exclude it for control/load comparison.
        for k in ('ctrl','filtered_load','old_abs_fz'):
            checks[k]=float(np.max(np.abs(recovery[k][:-1]-a[k][:n-1])))
        passed=flags and all(checks[k]<=v for k,v in prereg['recovery_gate']['absolute_max_tolerance'].items())
        gate=dict(passed=passed,max_errors=checks,force_flags_equal=flags,steps=n-1,dt=.0001,endpoint_note='terminal ctrl held, so ctrl excludes endpoint; force endpoint also must compare using identical controls',warning_peak=runner.d.warning.number.tolist())
        dump(out/'RECOVERY.json',gate); print('RECOVERY',json.dumps(gate),flush=True)
        if not passed:
            dump(out/'VERDICT.json',dict(status='RECOVERY_FAILED_NO_REFINED_INTEGRATION',paired_convergence='NOT_ESTABLISHED'))
            return 2
        del recovery
        all_summary={}; all_ep={}; a=baseline_geometry(a,ident)
        all_summary['0p1ms'],all_ep['0p1ms']=analyze(a)
        comparisons={}; prev=None
        for label,dt in [('0p05ms',.00005),('0p025ms',.000025)]:
            model=mujoco.MjModel.from_binary_path(str(BASE/'model.mjb')); model.opt.timestep=dt
            mujoco.mj_saveModel(model,str(out/('model_'+label+'.mjb')),None)
            r=Runner(model,a,ident)
            np.savez_compressed(out/('initial_'+label+'.npz'),integration_state=state(model,r.d),filtered_load=r.lf,qn=r.qn)
            dump(out/('MODEL_IDENTITY_'+label+'.json'),dict(**{k:v for k,v in ident.items() if k!='dt'},dt=dt,source_model_sha256=sha(BASE/'model.mjb'),derived_model_sha256=sha(out/('model_'+label+'.mjb')),model_options=str(model.opt),state_spec=int(STATE),initial_state_sha256=sha(out/('initial_'+label+'.npz')),only_change='opt.timestep'))
            b=r.run(2.,out/('trajectory_'+label+'.npz'))
            all_summary[label],all_ep[label]=analyze(b)
            comparisons['baseline_vs_'+label]=difference(a,b,round(.0001/dt))
            if prev is not None:
                comparisons['0p05ms_vs_0p025ms']=difference(prev,b,2)
                fine_common=difference({k:v[::2] for k,v in prev.items()},b,4)
                comparisons['common_0p1ms_grid_fine_difference']=fine_common
                comparisons['common_grid_fine_over_coarse_rms']={
                    k:fine_common[k]['rms']/comparisons['baseline_vs_0p05ms'][k]['rms']
                    if comparisons['baseline_vs_0p05ms'][k]['rms'] else None
                    for k in ('qpos','qvel','root','com','feet','ctrl','filtered_load')}
            prev=b
        dump(out/'SUMMARY.json',all_summary); dump(out/'CONTACT_EPISODES.json',all_ep); dump(out/'TRAJECTORY_DEVIATIONS.json',comparisons)
        dump(out/'VERDICT.json',dict(status='REFINEMENT_COMPLETED_DIAGNOSTIC_ONLY',recovery_passed=True,biological_absolute_force='NOT_ESTABLISHED',gait_repair=False,elapsed_s=time.monotonic()-started))
        print('completed elapsed_s',time.monotonic()-started,flush=True)
        return 0
    except Exception:
        dump(out/'FAILURE.json',dict(traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started)); raise
    finally: manifest(out)

if __name__=='__main__': sys.exit(main())
