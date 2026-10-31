"""S152 independent force/velocity power identities and between-run command audit."""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from verify_muscle_reconstruction import evaluate
R=Path('/home/sensen/Desktop/engine修复验收');PATHS={'S149':R/'s149-half-gain-physical/20261024T213108429114Z/init80ms','S151':R/'s151-lf-channel-gain/20261024T214136681607Z/init80ms','S146':R/'s146-secant-pose-free-physical/20261024T203645678947Z/init80ms'}
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;nodes=list(ET.parse('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml').getroot().find('actuator'));data={};report={}
 for arm,path in PATHS.items():
  a=load(path/'TRAJECTORY.npz');n=78 if arm=='S146' else 80;data[arm]=a;active=np.zeros((n,90));bias=active.copy()
  for t in range(n):
   for j,node in enumerate(nodes):
    f,g,b=evaluate(a['actuator_length'][t,j],a['actuator_velocity'][t,j],a['act'][t,j],node);active[t,j]=g*a['act'][t,j];bias[t,j]=b;assert abs(f-a['actuator_force'][t,j])<1e-10
  velocity=np.einsum('tud,td->tu',a['actuator_moment'][:n],a['qvel'][:n]);verr=float(abs(velocity-a['actuator_velocity'][:n]).max());assert verr<1e-10
  contributions=active[:,:,None]*a['actuator_moment'][:n];pos=np.maximum(contributions,0).sum(1);neg=np.minimum(contributions,0).sum(1);net=contributions.sum(1);assert abs(pos+neg-net).max()<1e-10
  scalar=(active+bias)*a['actuator_velocity'][:n];projected=np.sum((active+bias)[:,:,None]*a['actuator_moment'][:n]*a['qvel'][:n,None,:],2);perr=float(abs(scalar-projected).max());assert perr<1e-10;assert abs(scalar.sum(1)-a['signed_power_actuator'][:n]).max()<1e-10
  t=a['time'][:n];power=active*a['actuator_velocity'][:n];work=np.trapezoid(power,t,axis=0);negative=np.trapezoid(np.minimum(power,0),t,axis=0);positive=np.trapezoid(np.maximum(power,0),t,axis=0);assert abs(positive+negative-work).max()<1e-12
  opposing=np.any((pos>1e-12)&(neg< -1e-12),axis=0);indices=np.argsort(negative)[:8];report[arm]=dict(velocity_projection_maxabs=verr,power_projection_maxabs=perr,opposing_dofs=np.flatnonzero(opposing).tolist(),active_work=float(work.sum()),active_absorbed_signed_work=float(negative.sum()),negative_work_rank=[dict(index=int(j),name=nodes[j].get('name'),negative_work=float(negative[j]),net_work=float(work[j])) for j in indices],actual_activation_max=float(a['act'][:n].max()),input_sha256=sha(path/'TRAJECTORY.npz'))
 a=data['S149'];b=data['S151'];other=np.arange(90)!=4;changes={}
 for key in ('spikes','features','decoded','applied','ctrl','act','actuator_force','actuator_velocity'):
  x=a[key];y=b[key];changes[key]=dict(equal=bool(np.array_equal(x,y)),maxabs=float(abs(x.astype(float)-y.astype(float)).max()))
  if x.shape[-1]==90:changes[key]['other89_maxabs']=float(abs(x[...,other]-y[...,other]).max())
 result=dict(passed=True,arms=report,changes_S149_S151=changes,physical_units='NOT_ESTABLISHED',interpretation='Opposite torque, multijoint transfer and negative work are distinct; no harmfulness or additive COM force inference.',verifier_sha256=sha(__file__),model_load=0,step=0)
 with (p/'PARENT_TENDON_WORK_INDEPENDENT_CHECK.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
