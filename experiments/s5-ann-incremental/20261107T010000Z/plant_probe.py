"""Full free-body incremental actuator screening, no network success claims."""
import os,sys,json,hashlib,time,importlib.util
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np
import mujoco as mj
from incremental_actuator import IncrementalActuator,load_full_model,DEFAULT_MODEL_PATH
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path[:0]=[str(R),str(R/'tools')]
import dynamic_support_metrics as dsm
O=Path('/home/sensen/Desktop/engine修复验收/s5-ann-incremental/20261107T010000Z')
LEGACY=R/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py'
spec=importlib.util.spec_from_file_location('old_trial',LEGACY);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Plant:
 def __init__(self):
  self.m=load_full_model();self.mj=mj;m=self.m
  self.q0=m.key_qpos[0].copy();self.jids=m.actuator_trnid[:,0];self.qa=m.jnt_qposadr[self.jids];self.va=m.jnt_dofadr[self.jids];self.limits=m.actuator_ctrlrange[:,1]
  self.thorax=m.body('Thorax').id;self.bids=[m.body(l+'Tarsus5').id for l in b.LEGS]
  self.planes=m.geom_type==mj.mjtGeom.mjGEOM_PLANE
  self.feet=np.array([next((i for i,l in enumerate(b.LEGS) if (m.geom(g).name or '').startswith(l) and 'Tars' in (m.geom(g).name or '')),-1) for g in range(m.ngeom)])
 contact=b.Trial.contact
 bodyrow=b.Trial.bodyrow
 snapshot=b.Trial.snapshot
 def fresh(self):
  d=mj.MjData(self.m);mj.mj_resetDataKeyframe(self.m,d,0);mj.mj_forward(self.m,d);return d
 def actuator(self,d,scale=1.):
  return IncrementalActuator(self.m,d,kp=self.limits/.2*scale,kd=self.limits*.005*np.sqrt(scale),max_rate_rad_s=4.)

def main():
 O.mkdir(parents=True,exist_ok=True);dest=O/'plant_probe';dest.mkdir(exist_ok=False)
 protocol=dict(importance='核心必需',goal='confirm full-body incremental holding actuator before ANN training',duration_s=3.,warmup_s=1.,score_s=[1,3],gain_scales=[1,2,4],conditions=['hold','motor_cut'],gravity=[0,0,-9801],root='free, no equalities, no external forces',limits='unchanged original motor limits',success='all D support gates over final2s; holding success belongs to actuator, not neural network',hashes={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('incremental_actuator.py'),DEFAULT_MODEL_PATH]},units='NOT_ESTABLISHED')
 dump(dest/'PREREGISTRATION.json',protocol);plant=Plant();results=[]
 for scale in [1,2,4]:
  for mode in (['hold','motor_cut'] if scale==1 else ['hold']):
   d=plant.fresh();a=plant.actuator(d,scale);folder=dest/(mode+'_gain'+str(scale));folder.mkdir();dump(folder/'ACTUATOR.json',a.state_dict());rows=[];body=[];start=time.monotonic()
   for k in range(3000):
    a.update(np.zeros(42),.001)
    for sub in range(80):
     d.ctrl[:]=0 if mode=='motor_cut' and k>=1000 else a.torque(d)
     mj.mj_step(plant.m,d);mj.mj_forward(plant.m,d);rows.append(plant.contact(d))
    body.append(plant.bodyrow(d))
   rows=np.asarray(rows);assist=rows[(rows[:,0]>=.8)&(rows[:,0]<1)];release=rows[rows[:,0]>=1]
   reference=dict(foot_support_mean=float(assist[:,1:7].sum(1).mean()/dsm.WEIGHT),com_z_mean=float(assist[:,10].mean()))
   gate=dsm.evaluate_native(reference,release);gate['checks']['D1_no_nonfoot_ground_force']=bool(np.max(release[:,7])==0);gate['passed']=all(gate['checks'].values())
   r=dict(scale=scale,mode=mode,validation=gate,warnings=int(d.warning.number.sum()),finite=bool(np.isfinite(rows).all()),wall_s=time.monotonic()-start,folder=str(folder),neural_success=False)
   np.savez_compressed(folder/'NATIVE.npz',rows=rows);np.savez_compressed(folder/'BODY.npz',**{key:np.array([x[key] for x in body]) for key in body[0]});dump(folder/'RESULT.json',r);results.append(r);print(json.dumps(r),flush=True)
 dump(dest/'SUMMARY.json',results)
if __name__=='__main__':main()
