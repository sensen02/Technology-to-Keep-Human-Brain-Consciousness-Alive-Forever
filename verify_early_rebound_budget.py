"""Independent S147 saved-array early COM/force reconstruction; no simulation."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import xml.etree.ElementTree as ET
from verify_muscle_reconstruction import evaluate
SOURCES={'S142':Path('/home/sensen/Desktop/engine修复验收/s142-free-physical-prewarm/20261024T201419378677Z/init80ms'),'S146':Path('/home/sensen/Desktop/engine修复验收/s146-secant-pose-free-physical/20261024T203645678947Z/init80ms')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);out=pa.parse_args().output;rows={}
 for name,p in SOURCES.items():
  with np.load(p/'TRAJECTORY.npz') as z:a={k:z[k][:11].copy() if z[k].shape and z[k].shape[0]==78 else z[k].copy() for k in z.files}
  t=a['time'];assert len(t)==11 and abs(t[-1]-.00025)<1e-15
  force=a['contact_force_world']+a['gravity_force_world'];mass=a['body_mass'][0].sum();acc=force/mass;assert abs(acc-a['com_acceleration_world']).max()<1e-9
  assert abs(a['qfrc_actuator']+a['qfrc_passive']+a['qfrc_constraint']-a['qfrc_bias']-a['Ma']).max()<1e-10
  assert np.max(abs(np.einsum('tud,tu->td',a['actuator_moment'],a['actuator_force'])-a['qfrc_actuator']))<1e-10
  nodes=list(ET.parse('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml').getroot().find('actuator'));active=np.zeros((11,90));bias=active.copy()
  for i in range(11):
   for j in range(90):
    f,g,b=evaluate(a['actuator_length'][i,j],a['actuator_velocity'][i,j],a['act'][i,j],nodes[j]);active[i,j]=g*a['act'][i,j];bias[i,j]=b;assert abs(f-a['actuator_force'][i,j])<1e-10
  work={k:float(np.trapezoid(v,t)) for k,v in dict(active=np.sum(active*a['actuator_velocity'],axis=1),muscle_bias=np.sum(bias*a['actuator_velocity'],axis=1),contact=a['signed_power_contact_rows'],joint_passive=a['signed_power_passive']).items()}
  residual=a['com_momentum_world'][-1]-a['com_momentum_world'][0]-np.trapezoid(force,t,axis=0)
  rows[name]=dict(work_native=work,load_ratio_initial=float(a['load_ratio'][0]),load_ratio_25us=float(a['load_ratio'][1]),load_ratio_250us=float(a['load_ratio'][-1]),COM_az_initial=float(acc[0,2]),COM_az_25us=float(acc[1,2]),COM_az_250us=float(acc[-1,2]),max_activation_25us=float(a['act'][1].max()),max_activation_250us=float(a['act'][-1].max()),saturated_first_commands=int(np.count_nonzero(a['applied'][0]==1)),momentum_quadrature_residual=residual.tolist(),actuator_work_native=float(np.trapezoid(np.sum(a['actuator_force']*a['actuator_velocity'],axis=1),t)),hashes={str(p/'TRAJECTORY.npz'):sha(p/'TRAJECTORY.npz')})
 report=dict(passed=True,rows=rows,model_load=0,forward=0,step=0,physical_units='NOT_ESTABLISHED',interpretation='Internal muscles cannot be assigned net whole-body external force. Temporal concurrence is not a causal ablation.',verifier_sha256=sha(__file__))
 with (out/'PARENT_EARLY_REBOUND_INDEPENDENT_CHECK.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
