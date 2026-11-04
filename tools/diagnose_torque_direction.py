"""Where does the neural drive point compared with the torque that makes the reference gait work?

For each hinge joint the reference run records the external assist torque that was needed to make the
body follow the stride. The best neural run records the muscle activations it commands. At the same
gait phase this compares:

    tau_ref  the assist torque that made walking work, from the reference run
    tau_net  the joint torque the best run's own muscles produce from its activations

so the answer is per joint: does the network pull the same way, how large is the drive, and by how much
does it fall short. Analysis only: single mj_forward calls, no time stepping, no training.

modes: run
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s5-torque-diagnosis/20261106T060000Z')
REF=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN')
NET=Path('/home/sensen/Desktop/engine修复验收/s5-per-cell-readout-long/20261106T000000Z/VERIFY_A/RUN')
FREQ=5.0;T0=0.15;NBIN=20
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def run():
 import mujoco
 assets=Path('/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a')
 shim=O/'MODEL_SHIM'
 if not shim.exists():
  shim.mkdir(parents=True);(shim/'MODEL.xml').symlink_to(REF/'MODEL.xml')
  (shim/'meshes').mkdir(exist_ok=True)
  if not (shim/'meshes'/'stl').exists():(shim/'meshes'/'stl').symlink_to(assets,target_is_directory=True)
 os.chdir(shim)
 m=mujoco.MjModel.from_xml_path('MODEL.xml');d=mujoco.MjData(m)
 va=np.array([k for k in range(m.nv) if m.jnt_dofadr[m.dof_jntid[k]]==k and m.jnt_type[m.dof_jntid[k]]==3])
 va=np.array(sorted({int(m.jnt_dofadr[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE}))
 jnames=[m.joint(int(m.dof_jntid[k])).name for k in va]
 def torques(folder,lo,hi):
  z=np.load(folder/'BODY.npz',allow_pickle=False)
  t=np.asarray(z['time'],float);q=z['qpos'];qe=z['qvel'];act=z['activation'];assist=np.asarray(z['assist'],float)
  sel=np.flatnonzero((t>=lo)&(t<=hi))
  net=np.zeros((len(sel),len(va)));ref=np.zeros((len(sel),len(va)));mus=np.zeros((len(sel),m.nu))
  for i,k in enumerate(sel):
   d.qpos[:]=q[k];d.qvel[:]=qe[k];d.act[:]=act[k];d.ctrl[:]=act[k];mujoco.mj_forward(m,d)
   net[i]=np.asarray(d.qfrc_actuator).ravel()[va]
   mus[i]=act[k]
   ref[i]=assist[k][va]
  return t[sel],net,ref,mus
 tr,net_r,ref_r,mr=torques(REF,1.10,1.30)
 tn,net_n,ref_n,mn=torques(NET,1.30,1.50)
 ph=lambda t:((np.maximum(t-T0,0)*2*np.pi*FREQ)%(2*np.pi))
 def binned(tau,t):
  p=ph(t);idx=np.minimum((p/(2*np.pi)*NBIN).astype(int),NBIN-1)
  return np.array([tau[idx==b].mean(axis=0) if (idx==b).any() else np.zeros(tau.shape[1]) for b in range(NBIN)])
 R_ref=binned(ref_r,tr);R_net=binned(net_n,tn)
 rows=[]
 for j,name in enumerate(jnames):
  a=R_ref[:,j];b=R_net[:,j]
  s=np.linalg.norm(a);q=np.linalg.norm(b)
  agree=float(np.mean(np.sign(a[np.abs(a)>0.02])==np.sign(b[np.abs(a)>0.02]))) if (np.abs(a)>0.02).any() else None
  corr=float(np.corrcoef(a,b)[0,1]) if (a.std()>1e-9 and b.std()>1e-9) else None
  rows.append(dict(joint=name,dof=int(va[j]),ref_rms=float(s/np.sqrt(NBIN)),net_rms=float(q/np.sqrt(NBIN)),
   magnitude_ratio=float(q/s) if s>1e-9 else None,sign_agreement=agree,correlation=corr,
   ref_range=float(a.max()-a.min()),net_range=float(b.max()-b.min())))
 rows.sort(key=lambda r:(-(abs(r['correlation']) if r['correlation'] is not None else -1)))
 detail=[]
 for j,name in enumerate(jnames):
  worse=[r for r in rows if r['joint']==name][0]
  if abs(worse['ref_rms'])>0.05 and (worse['correlation'] is None or worse['correlation']<0.3):
   detail.append(dict(joint=name,ref_rms=worse['ref_rms'],net_rms=worse['net_rms'],corr=worse['correlation'],agree=worse['sign_agreement']))
 summary=dict(method='per joint comparison of the assist torque that made the reference stride work against the joint torque the best run own muscles produce, binned by gait phase',
  bins=NBIN,freq_hz=FREQ,reference_window=[1.10,1.30],network_window=[1.30,1.50],
  joints=len(va),mean_abs_correlation=float(np.nanmean([abs(r['correlation']) for r in rows if r['correlation'] is not None])),
  joints_corr_below_0_3=sum(1 for r in rows if r['correlation'] is not None and r['correlation']<0.3),
  worst_joints=detail[:12],
  tables=rows,
  reading='a joint with a large ref_rms but a correlation near zero or negative is a joint where the network pulls the wrong way or at the wrong time, which the current readout cannot fix by scaling')
 dump(O/'TORQUE_DEFICIT.json',summary)
 print('关节数 %d | 平均 |相关| %.3f | 相关<0.3 的关节 %d 个'%(len(va),summary['mean_abs_correlation'],summary['joints_corr_below_0_3']))
 print()
 print('%-26s %9s %9s %7s %7s'%('关节','参考RMS','网络RMS','相关','符号一致'))
 for r in detail[:12]:
  print('%-26s %9.3f %9.3f %7s %7s'%(r['joint'],r['ref_rms'],r['net_rms'],
   ('%.2f'%r['corr']) if r['corr'] is not None else 'na',('%.2f'%r['agree']) if r['agree'] is not None else 'na'))
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'run':run}[sys.argv[1]]()
