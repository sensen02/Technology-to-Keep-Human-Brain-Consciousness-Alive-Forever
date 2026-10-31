"""Independent S148 zero-interface ablation saved-array verifier; no simulator."""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from verify_com_contact_probe import reconstruct_fir
from verify_muscle_reconstruction import evaluate
REF=Path('/home/sensen/Desktop/engine修复验收/s146-secant-pose-free-physical/20261024T203645678947Z')
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;arm=p/'init80ms';a=load(arm/'TRAJECTORY.npz');b=load(REF/'init80ms/TRAJECTORY.npz');i=load(arm/'INITIALIZATION.npz');ib=load(REF/'init80ms/INITIALIZATION.npz');n=len(a['time']);assert 1<=n<=11 and np.allclose(a['time'],np.arange(n)*25e-6,rtol=0,atol=1e-15)
 for k in ('qpos','qvel','act','ctrl','qacc','qfrc_actuator','qfrc_passive','qfrc_constraint','com_acceleration_world','contact_dist'):
  assert np.array_equal(a[k][0],b[k][0]),k
 for k in i:assert np.array_equal(i[k],ib[k]),k
 model=load(arm/'NATIVE_MODEL_ARRAYS.npz');reference_model=load(REF/'init80ms/NATIVE_MODEL_ARRAYS.npz')
 for k in reference_model:assert np.array_equal(model[k],reference_model[k]),k
 assert not a['ctrl'].any() and not a['applied'].any() and not a['act'].any();assert a['decoded'].any();assert not a['qfrc_applied'].any() and not a['xfrc_applied'].any()
 hashes=json.loads((p/'INPUT_HASHES.json').read_text());paths={str(Path(k).resolve()) for k in hashes if Path(k).name=='MANIFEST.json'};assert len(paths)==1;manifest=Path(next(iter(paths)));rows=sorted(json.loads(manifest.read_text())['rows'],key=lambda r:r['actuator_index']);pools=sorted({tuple(sorted(r['local_indices'])) for r in rows if r['local_indices']});lookup={x:j for j,x in enumerate(pools)};mapping=np.array([lookup.get(tuple(sorted(r['local_indices'])),-1) for r in rows]);assert np.array_equal(mapping,a['actuator_pool'])
 fs,us,hist=reconstruct_fir(np.concatenate((i['spikes'],a['spikes'])),pools,mapping,a['weights'])
 for x,y in ((fs[:160],i['features']),(us[:160],i['decoded']),(fs[160:],a['features']),(us[160:],a['decoded']),(hist,load(arm/'NEURAL_FINAL.npz')['FIR_history'])):assert abs(x-y).max()<1e-14
 native=load(arm/'CONSTRAINT_ROWS.npz');errs=[]
 for k in range(n):errs.append(float(abs(native[f'sample{k:03d}_efc_J_dense'].T@native[f'sample{k:03d}_efc_force']-a['qfrc_constraint'][k]).max()))
 assert max(errs)<1e-10;assert abs(a['qfrc_actuator']+a['qfrc_passive']+a['qfrc_constraint']-a['qfrc_bias']-a['Ma']).max()<1e-10
 assert abs(a['com_acceleration_world']-(a['contact_force_world']+a['gravity_force_world'])/a['body_mass'][0].sum()).max()<1e-9
 orig=load(REF/'POSE_INITIALIZATION.npz')['original_qpos'];quat=a['qpos'][:,3:7];angle=2*np.arccos(np.clip(abs(quat@orig[3:7])/(np.linalg.norm(quat,axis=1)*np.linalg.norm(orig[3:7])),0,1));gates=np.column_stack((a['qpos'][:,2]/orig[2]>=.85,angle<=.35,np.count_nonzero(a['contacts']>0,axis=1)>=3,a['load_ratio']>=.8,a['nonfoot_depth']<=1e-12,a['limit_excess']<=1e-6));assert np.array_equal(gates,a['six_gates']);bad=np.flatnonzero(~gates.all(1));assert not len(bad) or (bad[0]==n-1)
 xml=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml');nodes=list(ET.parse(xml).getroot().find('actuator'));active=np.zeros((n,90));bias=active.copy()
 for k in range(n):
  for j,node in enumerate(nodes):
   f,g,v=evaluate(a['actuator_length'][k,j],a['actuator_velocity'][k,j],a['act'][k,j],node);assert abs(f-a['actuator_force'][k,j])<1e-10;active[k,j]=g*a['act'][k,j];bias[k,j]=v
 assert not active.any();assert abs(np.einsum('tud,tu->td',a['actuator_moment'],bias)-a['qfrc_actuator']).max()<1e-10
 t=a['time'];report=dict(passed=True,samples=n,steps=n-1,neural_ticks=160+len(a['spikes']),initial_state_and_PRE160_exact=True,actual_ctrl_zero=True,activation_and_active_force_zero=True,first_failure_sample=int(bad[0]) if len(bad) else None,physical_final_ms=float(t[-1]*1000),native_JTforce_maxabs=max(errs),common_window_comparison=dict(reference_final_COM_vz=float(b['com_velocity_world'][n-1,2]),ablation_final_COM_vz=float(a['com_velocity_world'][-1,2]),reference_final_COM_az=float(b['com_acceleration_world'][n-1,2]),ablation_final_COM_az=float(a['com_acceleration_world'][-1,2])),prewarm_success=False,physical_units='NOT_ESTABLISHED',hashes={str(x):sha(x) for x in (Path(__file__),arm/'TRAJECTORY.npz',arm/'INITIALIZATION.npz',manifest)})
 with (p/'ZERO_CONTROL_INDEPENDENT_CHECK.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
