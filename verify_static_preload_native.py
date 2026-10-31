"""S143 independent native constraint/contact and selection supplement."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path('/home/sensen/Desktop/engine修复验收/s143-initial-contact-preload/20261024T202354366515Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 r=json.loads((P/'RESULT.json').read_text());m=json.loads((P/'METADATA.json').read_text());rows=[]
 assert json.loads((P/'MODEL_NUMERIC_BEFORE.json').read_text())==json.loads((P/'MODEL_NUMERIC_AFTER.json').read_text())
 for i in range(3):
  with np.load(P/f'STATIC_{i}.npz') as z:a={k:z[k].copy() for k in z.files}
  err=float(abs(a['constraint__efc_J_dense'].T@a['constraint__efc_force']-a['qfrc_constraint']).max());assert err<=1e-10
  assert abs(a['qfrc_actuator']+a['qfrc_passive']+a['qfrc_constraint']-a['qfrc_bias']-a['Ma']).max()<=1e-10
  gates=np.array([a['qpos'][2]/a['original_qpos'][2]>=.85,a['root_angle_rad']<=.35,np.count_nonzero(a['contacts'])>=3,a['load_ratio']>=.8,a['nonfoot_depth']<=1e-12,a['limit_excess']<=1e-6]);assert np.array_equal(gates,a['six_gates'])
  mass=a['body_mass'].sum();az=float(a['external_force_world'][2]/mass);assert abs(az-a['com_acceleration_world'][2])<=1e-9
  rows.append(dict(index=i,dz=float(a['dz']),eligible=bool(gates.all()),abs_COM_az=abs(az),native_constraint_error=err))
 selected=min((x for x in rows if x['eligible']),key=lambda x:(x['abs_COM_az'],x['dz']));assert selected['index']==r['selected']['index']==0
 report=dict(passed=True,selection=selected,rows=rows,dynamic_executed=False,model_arrays_unchanged=True,hashes={str(f):sha(f) for f in [Path(__file__),P/'METADATA.json',P/'RESULT.json',*[P/f'STATIC_{i}.npz' for i in range(3)]]})
 with (P/'independent/NATIVE_SELECTION_CHECK.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
