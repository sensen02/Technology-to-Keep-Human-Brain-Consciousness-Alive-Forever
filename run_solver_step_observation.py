"""One original40ms reproduction; copy-only hook between step and forward."""
from pathlib import Path
import os,inspect,json,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_quarter_second_pair import Observer,OUT as BASE
from run_one_second_motor_pair import INITIAL,sha,dump
O=Path('/home/sensen/Desktop/engine修复验收/s2-solver-step-observation/20261027T202000Z')
FIELDS=['qpos','qvel','act','ctrl','qacc','qfrc_actuator','qfrc_passive','qfrc_bias','qfrc_constraint','qfrc_applied','xfrc_applied','xipos','cvel','qM']
def snapshot(d):return {k:getattr(d,k).copy() for k in FIELDS}
def tests():
 class Fake:pass
 f=Fake()
 for i,k in enumerate(FIELDS):setattr(f,k,np.array([float(i)]))
 old={k:getattr(f,k).copy() for k in FIELDS};s=snapshot(f)
 for k in FIELDS:np.testing.assert_array_equal(getattr(f,k),old[k]);getattr(f,k)[:]=999;np.testing.assert_array_equal(s[k],old[k])
 return dict(exit_code=0,copy_does_not_modify_source=True,later_writes_cannot_overwrite_snapshots=True)
class Hook:
 def __init__(self):self.pre=None;self.rows=[];self.contacts=[]
 def before(self,m,d):self.pre=snapshot(d);self.start=float(d.time)
 def after(self,m,d):
  row=snapshot(d);row.update(time_start=self.start,time_return=float(d.time),pre_qpos=self.pre['qpos'],pre_qvel=self.pre['qvel'],pre_act=self.pre['act']);con=[]
  for k in range(d.ncon):
   co=d.contact[k];f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);con.append(dict(geom1=int(co.geom1),geom2=int(co.geom2),position=co.pos.copy(),frame=co.frame.copy(),local_force=f.copy()))
  # kinematics remain solver stage values; integrator has updated qpos/qvel/act.
  copy=mujoco.MjData(m);mujoco.mj_copyData(copy,m,d);copy.qpos[:]=self.pre['qpos'];copy.qvel[:]=self.pre['qvel'];copy.act[:]=self.pre['act']
  mass=m.body_mass.sum();acc=np.zeros(3);vel=np.zeros(3)
  for b in range(1,m.nbody):
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);jd=np.zeros_like(jp);rd=np.zeros_like(jp);mujoco.mj_jacBodyCom(m,copy,jp,jr,b);mujoco.mj_jacDot(m,copy,jd,rd,copy.xipos[b],b);acc+=m.body_mass[b]*(jp@row['qacc']+jd@copy.qvel)/mass;vel+=m.body_mass[b]*(jp@copy.qvel)/mass
  row.update(solve_com_acc=acc,solve_com_velocity=vel);self.rows.append(row);self.contacts.append(con)

def main():
 O.mkdir(parents=True,exist_ok=False);dump(O/'TESTS.json',tests());sources=[Path(__file__),Path(c.__file__),INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML),Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz')];h={str(p):sha(p) for p in sources};dump(O/'PREREGISTRATION.json',dict(source_hashes=h,duration=.04,dt_body=.0000125,dt_neural=.0005,dt_command=.005,only_change='copy-only pre-step and step-return/pre-forward observer',reproduction='bitwise equality all shared BODY fields and INTERVALS arrays vs historical prefix; exact probe event records; fail before interpreting mismatch',timing='Euler integrator: step forward dynamics at interval start then Euler integration; returned integration state new while kinematic/force arrays retained solver stage; hook snapshots pre state separately',layers=['delta generalized velocity versus returned qacc incl Euler damping caveat','solver COM acceleration J*qacc+Jdot*qvel vs external world forces','cross-step COM delta momentum minus solve external impulse as discrete kinematics remainder'],no_second_run=True))
 m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);assert m.opt.integrator==mujoco.mjtIntegrator.mjINT_EULER;c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));mapping=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],[m.actuator(i).name for i in range(m.nu)]);dump(O/'MAPPING.json',mapping);groups,_=c.channel_index_map(built['n'],built['tier']['root_ids']);obs=Observer(m,geom);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);hook=Hook();src=inspect.getsource(c.episode);src=src.replace('            mujoco.mj_step(model, d)','            hook.before(model,d)\n            mujoco.mj_step(model, d)\n            hook.after(model,d)');ns=dict(vars(c),hook=hook);exec(compile(src,__file__,'exec'),ns);ev={};np.random.seed(0)
 rec,tr,cmd,sp=ns['episode'](m,q,joints,geom,built,mapping,groups,'normal',.04,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
 obs.save(O/'BODY.npz');dump(O/'RECORDS.json',rec);np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});np.savez_compressed(O/'SOLVER_STEPS.npz',**{k:np.array([r[k] for r in hook.rows]) for k in hook.rows[0]});dump(O/'SOLVER_CONTACTS.json',hook.contacts);(O/'DERIVED_EPISODE.py').write_text(src)
 checks={}
 for name in ['BODY','INTERVALS']:
  new=np.load(O/(name+'.npz'));old=np.load(BASE/'normal'/(name+'.npz'))
  for k in new.files:checks[name+'/'+k]=bool(np.array_equal(new[k],old[k][:len(new[k])]))
 oldrec=json.loads((BASE/'normal/RECORDS.json').read_text());checks['probe_indices']=rec['probe_local_indices']==oldrec['probe_local_indices'];checks['records']=rec['records']==oldrec['records'][:8];dump(O/'REPRODUCTION.json',checks);assert all(checks.values()),'observation run prefix differs'
 assert all(sha(Path(p))==v for p,v in h.items());dump(O/'EXECUTION.json',dict(exit_code=0,command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),model_load=1,mj_forward=3203,mj_step=3200,neural_step=80,copyData=3200,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('exact40ms reproduction copied solver stages exit0')
if __name__=='__main__':main()
