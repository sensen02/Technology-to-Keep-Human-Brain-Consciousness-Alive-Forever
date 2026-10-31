"""Independent S153 four-corner scalar formula, fixed geometry torque and power audit."""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from verify_muscle_reconstruction import evaluate
R=Path('/home/sensen/Desktop/engine修复验收');PATHS={'S149':R/'s149-half-gain-physical/20261024T213108429114Z/init80ms','S151':R/'s151-lf-channel-gain/20261024T214136681607Z/init80ms'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;xml=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml');nodes=list(ET.parse(xml).getroot().find('actuator'));data={k:load(v/'TRAJECTORY.npz') for k,v in PATHS.items()};report={};forces={}
 for host,a in data.items():
  M=a['actuator_moment'];v=a['qvel'];assert len(a['time'])==80
  for donor,b in data.items():
   active=np.zeros((80,90));bias=active.copy()
   for t in range(80):
    for j,node in enumerate(nodes):f,g,bi=evaluate(b['actuator_length'][t,j],b['actuator_velocity'][t,j],a['act'][t,j],node);active[t,j]=g*a['act'][t,j];bias[t,j]=bi
   force=active+bias;forces[host,donor]=force;torque=np.einsum('tud,tu->td',M,force);power=force*b['actuator_velocity'];nativepower=force*np.einsum('tud,td->tu',M,v);mismatch=float(abs(power-nativepower).max());norm=np.sum(M*M,2);assert np.all(norm>0);synthetic=v[:,None,:]+M*((b['actuator_velocity']-np.einsum('tud,td->tu',M,v))/norm)[:,:,None];projected=np.sum(M*force[:,:,None]*synthetic,2);err=float(abs(projected-power).max());assert err<1e-10
   actualerr=float(abs(force-b['actuator_force']).max());other=np.arange(90)!=4;othererr=float(abs(force[:,other]-b['actuator_force'][:,other]).max());assert othererr<1e-10
   if host==donor:assert actualerr<1e-10;assert abs(torque-a['qfrc_actuator']).max()<1e-10
   report[host+'_act_M__'+donor+'_LV']=dict(other89_force_vs_donor_residual=othererr,all90_force_vs_donor_difference=actualerr,compatible_per_actuator_power_residual=err,donor_power_vs_host_physical_velocity_mismatch=mismatch,synthetic_velocities_not_single_physical_state=True,scalar_total_work=float(np.trapezoid(power.sum(1),a['time'])),active_work=float(np.trapezoid((active*b['actuator_velocity']).sum(1),a['time'])))
 map_path=R/'s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json';mapping=json.loads(map_path.read_text());assert len(mapping)==90
 for arm,a in data.items():
  for j,item in enumerate(mapping):
   assert item['actuator']==j and item['name']==nodes[j].get('name') and item['tendon']==nodes[j].get('tendon');support=np.flatnonzero(np.any(abs(a['actuator_moment'][:,j])>1e-12,axis=0));assert set(support).issubset(set(item['dofs']))
 a=data['S149'];b=data['S151'];f0=forces['S149','S149'];fc=forces['S149','S151'];f1=forces['S151','S151'];m0=a['actuator_moment'];m1=b['actuator_moment'];terms=[np.einsum('tud,tu->td',m0,fc-f0),np.einsum('tud,tu->td',m0,f1-fc),np.einsum('tud,tu->td',m1-m0,f1)];delta=b['qfrc_actuator']-a['qfrc_actuator'];terr=float(abs(sum(terms)-delta).max());assert terr<1e-10
 assert np.array_equal(a['act'][:,np.arange(90)!=4],b['act'][:,np.arange(90)!=4]);result=dict(passed=True,corners=report,torque_delta_three_term_residual=terr,formula_tolerance=1e-10,input_hashes={str(PATHS[k]/'TRAJECTORY.npz'):sha(PATHS[k]/'TRAJECTORY.npz') for k in PATHS},xml_sha256=sha(xml),mapping_source=str(map_path),mapping_sha256=sha(map_path),physical_calls=0,conclusion='Known scalar formula and actual geometry explain saved differences; not a causal unloading localization.',verifier_sha256=sha(__file__))
 with (p/'PARENT_MATCHED_STATE_INDEPENDENT_CHECK.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
