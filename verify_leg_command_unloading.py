"""Independent S150 raw-contact sum and native muscle/joint projection audit."""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from verify_muscle_reconstruction import evaluate
ROOT=Path('/home/sensen/Desktop/engine修复验收');PATHS={'S146':ROOT/'s146-secant-pose-free-physical/20261024T203645678947Z/init80ms','S149':ROOT/'s149-half-gain-physical/20261024T213108429114Z/init80ms','S148':ROOT/'s148-zero-control-ablation/20261024T212511892865Z/init80ms'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);out=pa.parse_args().output;summary={};xml=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml');nodes=list(ET.parse(xml).getroot().find('actuator'))
 for name,p in PATHS.items():
  with np.load(p/'TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
  n=11 if name=='S148' else 78;meta=json.loads((p/'COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];gm={int(k):v for k,v in meta['geom_map'].items()};floors=set(meta['floor_geom_ids']);force=np.zeros((n,6,3));counts=np.zeros((n,6),int);depth=np.zeros((n,6));positions=[]
  for t in range(n):
   for c in range(int(a['contact_ncon'][t])):
    g1,g2=map(int,a['contact_geom_ids'][t,c]);foot=g2 if g1 in floors else g1 if g2 in floors else -1
    if foot not in gm:continue
    j=legs.index(gm[foot]);sign=1 if g1 in floors else -1;f=sign*a['contact_frame'][t,c].T@a['contact_force_raw6'][t,c,:3];force[t,j]+=f;counts[t,j]+=int(a['contact_dist'][t,c]<0);depth[t,j]=max(depth[t,j],-float(a['contact_dist'][t,c]));positions.append(a['contact_pos_world'][t,c])
  err=float(abs(force-a['foot_force'][:n]).max());assert err<1e-10;assert np.array_equal(counts,a['contacts'][:n]);assert abs(force[:,:,2].sum(1)/abs(a['gravity_force_world'][:n,2])-a['load_ratio'][:n]).max()<1e-12
  gain={'S146':1,'S149':.5,'S148':0}[name];assert np.array_equal(a['applied'],gain*a['decoded'])
  for t in range(1,n):assert np.array_equal(a['ctrl'][t],a['applied'][(t-1)//20])
  forceerr=0.;active=np.zeros((n,90));bias=active.copy()
  for t in range(n):
   for j,node in enumerate(nodes):
    f,g,b=evaluate(a['actuator_length'][t,j],a['actuator_velocity'][t,j],a['act'][t,j],node);forceerr=max(forceerr,abs(f-a['actuator_force'][t,j]));active[t,j]=g*a['act'][t,j];bias[t,j]=b
  assert forceerr<1e-10;torque=np.einsum('tud,tu->td',a['actuator_moment'][:n],active+bias);terr=float(abs(torque-a['qfrc_actuator'][:n]).max());assert terr<1e-10
  peaks={leg:dict(peak_time_ms=float(a['time'][np.argmax(force[:,j,2])]*1000),Fz_initial=float(force[0,j,2]),Fz_final=float(force[-1,j,2]),negative_contacts_final=int(counts[-1,j]),penetration_final=float(depth[-1,j])) for j,leg in enumerate(legs)}
  momentmeta=json.loads((p/'MOMENT_METADATA.json').read_text());joint_names=momentmeta['joint_names'];dof_jntid=momentmeta['dof_jntid'];mapping=[]
  for j,node in enumerate(nodes):
   dofs=np.flatnonzero(np.any(abs(a['actuator_moment'][:n,j])>1e-14,axis=0));mapping.append(dict(actuator_index=j,actuator=node.get('name'),tendon=node.get('tendon'),supported_dofs=dofs.tolist(),joint_ids=sorted({int(dof_jntid[k]) for k in dofs}),joints=sorted({joint_names[dof_jntid[k]] for k in dofs})))
  summary[name]=dict(native_transmission_mapping=mapping,contact_force_maxabs=err,muscle_force_maxabs=forceerr,joint_torque_maxabs=terr,actual_activation_max=float(a['act'][:n].max()),actual_activation_max_entire_archive=float(a['act'].max()),reconstruction_error_field_not_actual_activation=True,leg_peak_order=sorted(peaks,key=lambda k:peaks[k]['peak_time_ms']),legs=peaks,hashes={str(x):sha(x) for x in (p/'TRAJECTORY.npz',p/'COM_CONTACT_METADATA.json',p/'MOMENT_METADATA.json')})
 report=dict(passed=True,arms=summary,physical_units='NOT_ESTABLISHED',model_load=0,forward=0,step=0,verifier_sha256=sha(__file__))
 with (out/'PARENT_LEG_INDEPENDENT_CHECK.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
