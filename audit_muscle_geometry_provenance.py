"""Source anchored geometry provenance and pose-dependent moment-arm authority, no integration."""
from pathlib import Path
import json,hashlib,time,xml.etree.ElementTree as ET
import numpy as np,mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf
P=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-geometry-provenance/20261028T083000Z');B=Path('/home/sensen/Desktop/engine修复验收/s4-integrated-repaired-closure/20261028T032000Z/INTEGRATED_MODEL.xml');T=Path('/home/sensen/Desktop/engine修复验收/s5-compatible-stance-handover/20261028T080000Z');C=P/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):
 def enc(v):return v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)
 (O/n).write_text(json.dumps(x,indent=2,default=enc)+'\n')
def sites(p):return {s.get('name'):(b.get('name'),s.get('pos')) for b in ET.parse(p).getroot().iter('body') for s in b.findall('site')}
def main():
 O.mkdir(parents=True,exist_ok=False);start=time.time();src=[Path(__file__),C,B,T/'MODEL.xml',T/'HANDOVER_BEFORE.npz',P/'tools_mirror_muscles.py',P/'tools_stand_and_ground.py',P/'tools_repair_moment_arms.py',P/'run_integrated_repaired_closure.py'];dump('PREREGISTRATION.json',dict(source_hashes={str(p):sha(p) for p in src},scope='No model modifications. Compare current sites to canonical and evaluate one unchanged080 model at three saved poses:032keyframe,080initialkeyframe,080actualhandover.',calls=dict(model_load=1,forward=3,body_steps=0,neural_steps=0),interpretation='Bidirectional moment arms are only necessary geometrical authority, not force feasibility or behavior. No new attachments proposed.'))
 sc,sb,st=sites(C),sites(B),sites(T/'MODEL.xml');dump('SITE_CHAIN_CHECK.json',dict(canonical_sites=len(sc),base032_sites=len(sb),trial080_sites=len(st),canonical_to032_all_equal=sc==sb,base032_to080_all_equal=sb==st,canonical_sha256=sha(C),chain='032builder reads canonical and alters joint ranges/springrefs, wings, lengthranges, keyframe only. Warmup builder alters stance/support targets, springrefs, lengthranges, contact solver only. Sites remain inherited.'))
 m=_load_mjcf(str(T/'MODEL.xml')).compile();d=mujoco.MjData(m);counts=dict(model_load=1,forward=0,body_steps=0,neural_steps=0);qa=np.array([m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]);va=np.array([m.jnt_dofadr[j] for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]);baseq=np.fromstring(ET.parse(B).getroot().find('keyframe/key').get('qpos'),sep=' ');rows=[];arrays={}
 for label,q in [('base032_keyframe',baseq),('trial080_initial',m.key_qpos[0].copy()),('trial080_handover',np.load(T/'HANDOVER_BEFORE.npz')['qpos'])]:
  d.qpos[:]=q;d.qvel[:]=0;d.act[:]=.0001;d.ctrl[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1;moment=np.zeros((m.nu,m.nv))
  for a in range(m.nu):
   ad=int(d.moment_rowadr[a]);nn=int(d.moment_rownnz[a]);moment[a,d.moment_colind[ad:ad+nn]]=d.actuator_moment[ad:ad+nn]
  # Muscle active force is nonpositive, so generalized active direction is negative tendon moment.
  authority=-moment[:,va].T;detail=[]
  for k,v in enumerate(va):
   r=authority[k];pos=int(np.sum(r>1e-8));neg=int(np.sum(r< -1e-8));detail.append(dict(joint=m.joint(int(m.dof_jntid[v])).name,q=float(q[qa[k]]),positive_moment_arms=pos,negative_moment_arms=neg,min=float(r.min()),max=float(r.max()),bidirectional=bool(pos and neg)))
  row=dict(label=label,hinge_deviation_from032=float(np.linalg.norm(q[qa]-baseq[qa])),bidirectional_joint_count=sum(r['bidirectional'] for r in detail),one_way_or_unactuated=[r['joint'] for r in detail if not r['bidirectional']],joints=detail);rows.append(row);arrays[label+'_qpos']=q.copy();arrays[label+'_moment']=moment
 dump('POSE_AUTHORITY.json',rows);np.savez_compressed(O/'MOMENT_ARRAYS.npz',**arrays);dump('EXECUTION.json',dict(counts=counts,exit_code=0,runtime_s=time.time()-start,source_unchanged=all(sha(p)==h for p,h in json.loads((O/'PREREGISTRATION.json').read_text())['source_hashes'].items())));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps([{k:v for k,v in r.items() if k!='joints'} for r in rows]),flush=True)
if __name__=='__main__':main()
