"""One frozen S5 algorithm run; synchronous observations on a separate MjData."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, hashlib, inspect, importlib.util, platform, subprocess, time
from pathlib import Path
import numpy as np
import mujoco
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-baseline-recording/20261101T183002Z')
OLD = Path('/home/sensen/Desktop/engine修复验收/s5-walk-push/20261101T145318Z/SCRIPTS/walk_push.py')
sys.path.insert(0, str(ROOT))
from engine.embodied.contact_metrics import contact_episodes

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

def load_old():
    spec = importlib.util.spec_from_file_location('frozen_walk_push', OLD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def confirm_sign():
    results = []
    for reverse in (False, True):
        plane = '<geom name="floor" type="plane" size="1 1 .1"/>'
        ball = '<body pos="0 0 .09"><freejoint/><geom name="ball" type="sphere" size=".1" mass="1"/></body>'
        m = mujoco.MjModel.from_xml_string('<mujoco><worldbody>' + (ball+plane if reverse else plane+ball) + '</worldbody></mujoco>')
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        c = d.contact[0]; f = np.zeros(6); mujoco.mj_contactForce(m, d, 0, f)
        sign = 1 if m.geom(int(c.geom2)).name == 'ball' else -1
        world = sign * (c.frame.reshape(3, 3).T @ f[:3])
        assert world[2] > 0 and np.isclose(world[2], d.qfrc_constraint[2])
        results.append(dict(xml_order_reversed=reverse, geom1=m.geom(int(c.geom1)).name, geom2=m.geom(int(c.geom2)).name, force_on_ball=world.tolist(), qfrc_constraint_z=float(d.qfrc_constraint[2])))
    dump('CONTACT_SIGN_CHECK.json', {'convention': 'contact frame force acts on geom2; negative on geom1', 'checks': results})

class Recorder:
    def __init__(self, old, real_step):
        self.old, self.real_step = old, real_step
        self.i = 0; self.arr = {}; self.rows = []; self.chunk = 0; self.calls = {}; self.warning_peak = None
    def setup(self, m, d, local):
        self.m, self.live, self.obs = m, d, mujoco.MjData(m)
        self.n = int(round(2 / m.opt.timestep))
        self.plan, self.qn, self.sg = local['plan'], local['qn'].copy(), local['sg']
        self.root = m.body('nmf/c_thorax').id
        self.feet = [m.body('nmf/%s_tarsus5' % leg).id for leg in self.old.LEGS]
        self.floor = set(np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
        self.fly = {g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('nmf/')}
        self.leg_for_geom = {}
        for g in self.fly:
            name = m.geom(g).name or ''
            if 'tarsus' in name:
                for j, leg in enumerate(self.old.LEGS):
                    if '/%s_' % leg in name: self.leg_for_geom[g] = j
        shapes = dict(time=(), qpos=(m.nq,), qvel=(m.nv,), ctrl=(m.nu,), act=(m.na,), actuator_force=(m.nu,), pd_p=(m.nu,), pd_d=(m.nu,), leg_drive=(m.nu,), leg_u=(6,), filtered_load=(6,), com=(3,), root=(3,), feet=(6,3), old_abs_fz=(6,), signed_foot_fz=(6,), signed_support=(3,), xfrc_applied=(m.nbody,6), qfrc_applied=(m.nv,), qacc_warmstart=(m.nv,), warning_number=(len(d.warning),), legacy_post_abs_fz=(6,), nonfoot=())
        self.arr = {k: np.zeros((self.n+1,)+s) for k,s in shapes.items()}
        mujoco.mj_saveModel(m, str(OUT/'model.mjb'), None)
        dump('MODEL_IDENTITY.json', dict(nq=m.nq,nv=m.nv,nu=m.nu,na=m.na,nbody=m.nbody,ngeom=m.ngeom,dt=float(m.opt.timestep),mass=float(m.body_mass.sum()),gravity=m.opt.gravity.tolist(),geoms=[m.geom(g).name for g in range(m.ngeom)],bodies=[m.body(b).name for b in range(m.nbody)],actuators=[m.actuator(a).name for a in range(m.nu)],plan=self.plan,signs={str(k):v for k,v in self.sg.items()},asset_identity='MJB embeds compiled meshes/textures; MJB SHA256 in manifest'))
    def sample(self, local):
        m,d,a,i = self.m,self.obs,self.arr,self.i
        mujoco.mj_copyData(d,m,self.live)
        mujoco.mj_forward(m,d)
        for key in ('time','qpos','qvel','ctrl','act','actuator_force','xfrc_applied','qfrc_applied','qacc_warmstart'):
            a[key][i] = getattr(d,key)
        a['warning_number'][i] = d.warning.number
        a['com'][i] = (d.xipos*m.body_mass[:,None]).sum(axis=0)/m.body_mass.sum()
        a['root'][i] = d.xpos[self.root]; a['feet'][i] = d.xpos[self.feet]
        a['leg_u'][i] = [local['u'][L] for L in self.old.LEGS]
        a['filtered_load'][i] = [local['lf'][L] for L in self.old.LEGS]
        for p in self.plan:
            j = p['act']; a['pd_p'][i,j] = self.old.KP*(self.qn[p['qadr']]-d.qpos[p['qadr']]); a['pd_d'][i,j] = -self.old.KD*d.qvel[p['dadr']]
            if p['leg'] and p['bias']:
                a['leg_drive'][i,j] = local['u'][p['leg']]*abs(p['bias'])*self.sg[(p['leg'],p['name'])]
        for k in range(d.ncon):
            c = d.contact[k]; g1,g2 = int(c.geom1),int(c.geom2)
            f = np.zeros(6); mujoco.mj_contactForce(m,d,k,f)
            world = c.frame.reshape(3,3).T@f[:3]
            # External net force on fly: internal fly/fly contacts cancel.
            fly_force = world * (int(g2 in self.fly)-int(g1 in self.fly))
            if (g1 in self.floor and g2 in self.fly) or (g2 in self.floor and g1 in self.fly):
                a['signed_support'][i] += fly_force
                other = g2 if g1 in self.floor else g1
                if other in self.leg_for_geom:
                    j = self.leg_for_geom[other]
                    a['old_abs_fz'][i,j] += abs(world[2]); a['signed_foot_fz'][i,j] += fly_force[2]
                else: a['nonfoot'][i] += 1
            self.rows.append([i,k,g1,g2,float(c.dist),*c.pos,*c.frame,*f,*world,*fly_force])
        self.i += 1
        if self.i % 1000 == 0: self.flush()
    def flush(self):
        if self.rows:
            np.savez_compressed(OUT/('contacts_%04d.npz'%self.chunk), contacts=np.asarray(self.rows, dtype=float))
            self.rows.clear(); self.chunk += 1
    def step(self, m, d, *args, **kwargs):
        frame = inspect.currentframe().f_back
        loc = frame.f_locals
        phase = 'walk' if frame.f_code.co_name == 'run' and 't' in loc else frame.f_code.co_name + ('_settle' if 'lf' in loc else '_calibration')
        self.calls[phase] = self.calls.get(phase,0)+1
        if phase == 'walk':
            if not self.arr: self.setup(m,d,loc)
            self.last_local = dict(loc)
            self.sample(loc)
        result = self.real_step(m,d,*args,**kwargs)
        warnings = d.warning.number.copy()
        self.warning_peak = warnings if self.warning_peak is None else np.maximum(self.warning_peak,warnings)
        if phase == 'walk':
            loads,_ = self.old.foot_load(m,d)
            self.arr['legacy_post_abs_fz'][self.i-1] = [loads[L] for L in self.old.LEGS]
        return result
    def finish(self):
        self.sample(self.last_local); self.flush()
        # Endpoint has no associated integration interval or legacy step-post load.
        self.arr['legacy_post_abs_fz'][-1] = self.arr['legacy_post_abs_fz'][-2]
        assert self.i == self.n+1
        np.savez_compressed(OUT/'trajectory.npz', **self.arr)
        dump('NUMERICAL_WARNINGS.json', dict(peak_live_warning_counts=self.warning_peak.tolist(), final_live_counts=self.live.warning.number.tolist(), final_lastinfo=self.live.warning.lastinfo.tolist(), calls=self.calls))

def analyze():
    with np.load(OUT/'trajectory.npz') as f: a = {k:f[k] for k in f.files}
    ident = json.loads((OUT/'MODEL_IDENTITY.json').read_text())
    dt=ident['dt']; n=len(a['time']); t=a['time']; rel=t-t[0]
    finite={k:bool(np.isfinite(v).all()) for k,v in a.items()}
    assert all(finite.values()) and np.allclose(np.diff(t),dt,rtol=0,atol=1e-12)
    assert abs(rel[-1]-2)<1e-9
    assert not np.any(a['xfrc_applied']) and not np.any(a['qfrc_applied'])
    motors=[p['act'] for p in ident['plan']]
    # Final endpoint retains preceding control, so decomposition check excludes it.
    err=float(np.max(np.abs(a['ctrl'][:-1,motors]-(a['pd_p']+a['pd_d']+a['leg_drive'])[:-1,motors])))
    assert err<1e-10
    model=mujoco.MjModel.from_binary_path(str(OUT/'model.mjb')); d=mujoco.MjData(model)
    sync=[]
    for i in (0,n//2,n-1):
        d.qpos[:]=a['qpos'][i]; d.qvel[:]=a['qvel'][i]; d.ctrl[:]=a['ctrl'][i]; d.time=t[i]
        mujoco.mj_forward(model,d)
        root=d.xpos[model.body('nmf/c_thorax').id]
        sync.append(float(np.max(np.abs(root-a['root'][i]))))
    assert max(sync)<1e-10
    episode={}
    for definition,key in [('old_abs','old_abs_fz'),('signed_up','signed_foot_fz')]:
        episode[definition]={}
        for stride in (1,20):
            idx=np.arange(0,n,stride)
            if idx[-1]!=n-1: idx=np.append(idx,n-1)
            legs={}
            for j,L in enumerate(('lf','lm','lh','rf','rm','rh')):
                r=contact_episodes(t[idx].tolist(),(a[key][idx,j]>1e-9).tolist(),a['feet'][idx,j,:2].tolist())
                complete=[e for e in r['episodes'] if not e['left_censored'] and not e['right_censored']]
                durations=[e['duration'] for e in complete]
                r['short_complete_count_lt60ms']=sum(v<.060 for v in durations)
                r['duration_quantiles_s']=np.quantile(durations,[0,.25,.5,.75,1]).tolist() if durations else None
                legs[L]=r
            episode[definition][str(stride)]=legs
    dump('CONTACT_EPISODES.json',episode)
    delta=a['root'][-1]-a['root'][0]
    path=float(np.linalg.norm(np.diff(a['root'][:,:2],axis=0),axis=1).sum())
    com_ratio=float(a['com'][:,2].min()/a['com'][0,2])
    legacy=json.loads((OUT/'LEGACY_RESULT.json').read_text())
    summary=dict(status='ALGORITHM_PD_DIAGNOSTIC_ONLY; neural and muscle success NOT_ESTABLISHED',samples=n,window_s=float(rel[-1]),dt=dt,root_delta=delta.tolist(),root_planar_net=float(np.linalg.norm(delta[:2])),root_planar_path=path,root_path_efficiency=float(np.linalg.norm(delta[:2])/path),com_delta=(a['com'][-1]-a['com'][0]).tolist(),com_z_min=float(a['com'][:,2].min()),com_z_min_ratio=com_ratio,signed_support_z_mean=float(a['signed_support'][:-1,2].mean()),signed_support_z_range=[float(a['signed_support'][:,2].min()),float(a['signed_support'][:,2].max())],legacy_displacement=legacy['displacement_mm'],legacy_minus_rounded_3p013=float(legacy['displacement_mm']-3.013),legacy_rounding_agrees=abs(legacy['displacement_mm']-3.013)<=.0005,drive_sign_changes_per_s=[int(np.count_nonzero(np.diff(np.sign(a['leg_u'][:-1,j]))))/2 for j in range(6)],units='native model units, historical length labelled mm; biological absolute force calibration NOT_ESTABLISHED',foot_drift='tarsus5 body-origin planar drift during flagged contact, NOT true material-point slip',support_definition='sum signed world floor contact forces acting on fly; adhesion forces are separate actuator contributions',sample_alignment='step-pre clone forward after ctrl update; final endpoint ctrl held; legacy_post_abs_fz separately preserves old post-step cache')
    dump('SUMMARY.json',summary)
    contact_count=0
    for p in sorted(OUT.glob('contacts_*.npz')):
        rows=np.load(p)['contacts']; assert rows.shape[1]==29 and np.isfinite(rows).all()
        assert np.all((rows[:,0]>=0)&(rows[:,0]<n)); contact_count+=len(rows)
    dump('VALIDATION.json',dict(exit_code=0,shapes={k:list(v.shape) for k,v in a.items()},finite=finite,dt_max_error=float(np.max(np.abs(np.diff(t)-dt))),ctrl_decomposition_max_error=err,model_reload_root_sync_errors=sync,contact_rows=contact_count,contact_columns=['sample','contact_index','geom1','geom2','dist','pos[3]','frame[9]','contact_force_torque[6]','world_force_on_geom2[3]','net_external_world_force_on_fly[3]']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(13,10))
    axs[0,0].plot(rel,a['root']-a['root'][0]); axs[0,0].set_title('Root displacement (native length)'); axs[0,0].legend(['x','y','z'])
    axs[0,1].plot(a['root'][:,0],a['root'][:,1]); axs[0,1].set_title('Root XY path; efficiency %.3f'%summary['root_path_efficiency'])
    axs[1,0].plot(rel,a['com'][:,2]/a['com'][0,2]); axs[1,0].axhline(.9,color='r',ls='--'); axs[1,0].set_title('COM z / initial')
    axs[1,1].plot(rel,a['signed_support'][:,2],lw=.5); axs[1,1].set_title('Signed ground support Fz (native force)')
    axs[2,0].imshow((a['old_abs_fz']>1e-9).T,aspect='auto',interpolation='nearest',extent=[0,2,6,0]); axs[2,0].set_yticks(np.arange(6)+.5,('lf','lm','lh','rf','rm','rh')); axs[2,0].set_title('Every-step old abs Fz contact flags')
    for stride in ('1','20'):
        ds=[e['duration']*1000 for r in episode['old_abs'][stride].values() for e in r['episodes'] if not e['left_censored'] and not e['right_censored']]
        axs[2,1].hist(ds,bins=np.linspace(0,160,41),histtype='step',label='stride '+stride)
    axs[2,1].set_title('Complete contact duration (ms)'); axs[2,1].set_yscale('log'); axs[2,1].legend()
    fig.suptitle('Single frozen algorithm PD baseline — not neural/muscle success')
    fig.tight_layout(); fig.savefig(OUT/'DIAGNOSTICS.png',dpi=150); plt.close(fig)
    return summary

def manifest():
    files={str(p):{'sha256':sha(p),'bytes':p.stat().st_size,'importance':'重要辅助' if p.suffix=='.png' or p.name=='RUN.log' else '核心必需'} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('MANIFEST_SHA256.json','RUN.log')}
    files[str(Path(__file__).resolve())]={'sha256':sha(__file__),'importance':'核心必需'}
    files[str(OLD)]={'sha256':sha(OLD),'importance':'核心必需; frozen external dependency'}
    dump('MANIFEST_SHA256.json',files)
    assert all(sha(p)==v['sha256'] for p,v in files.items())

def main():
    started=time.time(); prereg=json.loads((OUT/'PREREGISTRATION.json').read_text())
    assert sha(OLD)==prereg['historical_sha256']
    dump('ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),mujoco=mujoco.__version__,numpy=np.__version__,MUJOCO_GL=os.getenv('MUJOCO_GL'),PYTHONDONTWRITEBYTECODE=os.getenv('PYTHONDONTWRITEBYTECODE'),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=sha(__file__),historical_sha256=sha(OLD),contact_metrics_sha256=sha(ROOT/'engine/embodied/contact_metrics.py')))
    if '--analyze-only' not in sys.argv:
        assert not (OUT/'trajectory.npz').exists(), 'refuse second simulation in same evidence directory'
        confirm_sign(); old=load_old(); real_step=mujoco.mj_step; rec=Recorder(old,real_step)
        mujoco.mj_step=rec.step
        try: result=old.run(gain=1.0,grip=False,a_sweep=0.0,push_frac=0.0)
        finally: mujoco.mj_step=real_step
        dump('LEGACY_RESULT.json',result); rec.finish()
        print('single simulation finished',result,flush=True)
    summary=analyze(); manifest()
    print(json.dumps(summary,indent=2),flush=True)
    print('exit_code=0 elapsed_s=%.3f'%(time.time()-started),flush=True)

if __name__=='__main__':
    main()
