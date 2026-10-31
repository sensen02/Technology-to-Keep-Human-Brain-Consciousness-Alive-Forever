"""Static diagnostic: native muscle floor force at the settled pose, and a declared
length-range recentering (engineering calibration, NOT physiology). No integration."""
from pathlib import Path
import json,hashlib,time,traceback
import numpy as np, mujoco
from scipy.optimize import lsq_linear
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-lengthrange-recenter/20261028T088000Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);start=time.time();counts=dict(model_load=0,forward=0,body_step=0,neural_step=0);w=np.load(S/'BODY.npz');t=w['time'];i=int(np.argmin(abs(t-1.199)));q=w['qpos'][i].copy();T=w['assist'][i,6:48].copy();weight=float(json.loads((S/'GAINS.json').read_text())['weight'])
 dump(O/'PREREGISTRATION.json',dict(source_model=str(S/'MODEL.xml'),source_sha256=sha(S/'MODEL.xml'),pose_time=float(t[i]),pose_qpos_sha256=hashlib.sha256(q.tobytes()).hexdigest(),scope='Static floor/authority diagnostic plus declared length-range recentering so each muscle sits at a chosen normalized length at the settled pose. Engineering calibration, NOT physiology. No body integration.',targets=[1.0,1.1],grid_targets=[0.9,1.0,1.1,1.2],objective='min ||A_leg x + floor_leg - PD_tau_leg|| / bodyweight, 0<=x<=1, all90 muscles'))
 m=_load_mjcf(str(S/'MODEL.xml')).compile();counts['model_load']+=1;d=mujoco.MjData(m);leg=np.arange(6,48);rows=[]
 def probe(mm,dd):
  dd.qpos[:]=q;dd.qvel[:]=0;dd.act[:]=.0001;dd.ctrl[:]=0;dd.qfrc_applied[:]=0;dd.xfrc_applied[:]=0;dd.qacc_warmstart[:]=0;mujoco.mj_forward(mm,dd);counts['forward']+=1
  base=dd.qfrc_actuator.copy();A=np.zeros((48,90))
  for j in range(90):
   x=np.zeros(90);x[j]=1;dd.act[:]=.0001+.9999*x;mujoco.mj_forward(mm,dd);counts['forward']+=1;A[:,j]=dd.qfrc_actuator-base
  return A,base
 def evaluate(mm,name,extra):
  A,base=probe(mm,d)
  sol=lsq_linear(A[leg],T-base[leg],bounds=(0,1));pred=A[leg]@sol.x+base[leg]
  return dict(name=name,**extra,model_sha256=None,floor_norm_over_weight=float(np.linalg.norm(base[leg])/weight),pd_torque_norm_over_weight=float(np.linalg.norm(T)/weight),best_fit_residual_over_weight=float(np.linalg.norm(pred-T)/weight),relative_residual=float(np.linalg.norm(pred-T)/np.linalg.norm(T)),activation_max=float(sol.x.max()),active_fraction=int((sol.x>1e-3).sum()),active_force_matrix=A,floor_force=base)
 rows.append({k:v for k,v in evaluate(m,'original',{}).items() if k not in('active_force_matrix','floor_force')});dump(O/'RESULTS.json',rows);print(json.dumps(rows[-1]),flush=True)
 # Build recentered models: translate lengthrange so normalized length at this pose is the target.
 base_lens=d.actuator_length.copy() if False else None
 for target in [0.9,1.0,1.1,1.2]:
  mm=_load_mjcf(str(S/'MODEL.xml')).compile() if False else None
  spec=mujoco.MjSpec.from_file(str(S/'MODEL.xml')) if False else None
  # Use a fresh model built from a rewritten XML instead (lengthrange lives in the XML).
  import xml.etree.ElementTree as ET
  tree=ET.parse(S/'MODEL.xml');root=tree.getroot();mm=_load_mjcf(str(S/'MODEL.xml')).compile();dd=mujoco.MjData(mm);dd.qpos[:]=q;dd.qvel[:]=0;mujoco.mj_forward(mm,dd);counts['forward']+=1;L=dd.actuator_length.copy()
  changes=[]
  for ax,el in zip(root.findall('./actuator/*'),root.findall('./actuator/*')):
   gp=np.array(el.get('gainprm').split(),float);lmin,lmax=gp[0],gp[1];lr=np.array(el.get('lengthrange').split(),float);L0=(lr[1]-lr[0])/(lmax-lmin);cur=(L[len(changes)]-lr[0]+lmin*L0)/L0;delta=L[len(changes)]-target*L0-(lr[0]-lmin*L0);new=lr+delta;el.set('lengthrange',' '.join(format(v,'.17g') for v in new));changes.append(dict(actuator=el.get('name'),old_lengthrange=lr,new_lengthrange=new,L0=L0,length_now=float(L[len(changes)-1]),normalized_before=float(cur),delta=float(delta)))
  out=O/f'MODEL_recenter_{target}.xml';tree.write(out,encoding='utf8',xml_declaration=True);mm2=_load_mjcf(str(out)).compile();counts['model_load']+=1
  r={k:v for k,v in evaluate(mm2,f'recenter_{target}',dict(target_normalized_length=target,model_path=str(out))).items() if k not in('active_force_matrix','floor_force')};r['model_sha256']=sha(out);rows.append(r);dump(O/'RESULTS.json',rows);dump(O/f'LENGTHRANGE_LEDGER_{target}.json',dict(pose_time=float(t[i]),target=target,changes=changes,note='Translation only; L0 width preserved. Engineering calibration, NOT physiology.'));print(json.dumps(r),flush=True)
 dump(O/'EXECUTION.json',dict(counts=counts,runtime_s=time.time()-start,payload_saved=True))
if __name__=='__main__':main()
