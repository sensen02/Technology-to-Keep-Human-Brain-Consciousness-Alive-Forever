"""S142 saved-array muscle work attribution; no simulation or control."""
from pathlib import Path
import hashlib,json,csv,xml.etree.ElementTree as ET
import numpy as np
from verify_muscle_reconstruction import evaluate
ROOT=Path('/home/sensen/Desktop/engine修复验收/s142-free-physical-prewarm/20261024T201419378677Z/init80ms')
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 with np.load(ROOT/'TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 with np.load(ROOT/'NATIVE_MODEL_ARRAYS.npz') as z:m={k:z[k].copy() for k in z.files}
 nodes=list(ET.parse(XML).getroot().find('actuator'));assert len(nodes)==90
 for key in ('gainprm','biasprm','dynprm'):
  vals=[list(map(float,n.get(key).split())) for n in nodes];x=np.array([v+[0.]*(10-len(v)) for v in vals]);assert np.array_equal(x,m['actuator_'+key])
 assert np.array_equal(np.array([list(map(float,n.get('lengthrange').split())) for n in nodes]),m['actuator_lengthrange'])
 active=np.zeros_like(a['act']);bias=active.copy();total=active.copy()
 for i in range(len(a['time'])):
  for j,node in enumerate(nodes):
   f,g,b=evaluate(a['actuator_length'][i,j],a['actuator_velocity'][i,j],a['act'][i,j],node)
   total[i,j]=f;active[i,j]=g*a['act'][i,j];bias[i,j]=b
 err=float(abs(total-a['actuator_force']).max());assert err<=1e-10
 projection=np.einsum('tud,tu->td',a['actuator_moment'],total);assert abs(projection-a['qfrc_actuator']).max()<=1e-10
 powers={'muscle_active_gain_times_activation':(active*a['actuator_velocity']).sum(1),'muscle_passive_bias':(bias*a['actuator_velocity']).sum(1),'total_muscle':(total*a['actuator_velocity']).sum(1),'joint_passive':a['signed_power_passive'],'contact_constraint':a['signed_power_contact_rows'],'noncontact_constraint':a['signed_power_noncontact_constraint']}
 assert abs(powers['total_muscle']-a['signed_power_actuator']).max()<=1e-10
 work={k:float(np.trapezoid(v,a['time'])) for k,v in powers.items()}
 contact_impulse=np.trapezoid(a['contact_force_world'],a['time'],axis=0);gravity_impulse=np.trapezoid(a['gravity_force_world'],a['time'],axis=0)
 momentum=a['com_momentum_world'][-1]-a['com_momentum_world'][0]
 with (ROOT/'SIGNED_WORK_BUDGET.csv').open('x') as f:
  w=csv.writer(f);w.writerow(['physical_s',*powers]);w.writerows(zip(a['time'],*powers.values()))
 report=dict(passed=True,force_maxabs_error=err,work_native=work,contact_impulse_world=contact_impulse.tolist(),gravity_impulse_world=gravity_impulse.tolist(),COM_momentum_change_world=momentum.tolist(),quadrature_momentum_residual=(momentum-contact_impulse-gravity_impulse).tolist(),interpretation='Active gain*activation separated from passive muscle bias and joint passive forces. Signed trapezoidal native work, not stored elastic energy. Positive contact work is compatible with initial soft-contact transient but cannot uniquely identify elastic release or exclude dissipation/integration effects. No physical energy-unit calibration; NOT_ESTABLISHED.',hashes={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('verify_muscle_reconstruction.py'),XML,ROOT/'TRAJECTORY.npz',ROOT/'NATIVE_MODEL_ARRAYS.npz')},model_load=0,forward=0,step=0)
 with (ROOT/'SIGNED_WORK_BUDGET.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
