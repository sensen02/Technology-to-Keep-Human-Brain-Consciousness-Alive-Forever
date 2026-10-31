"""S156 independent offline audit, adds no model/forward calls."""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from verify_muscle_reconstruction import evaluate
SRC=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms')
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;a=load(SRC/'TRAJECTORY.npz');meta=json.loads((SRC/'COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];gm={int(k):v for k,v in meta['geom_map'].items()};floors=set(meta['floor_geom_ids']);mapping=json.loads(Path('/home/sensen/Desktop/engine修复验收/s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json').read_text());nodes=list(ET.parse('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml').getroot().find('actuator'));results=[];derived={};report={}
 for sample in (12,60):
  baseline=load(p/f'state{sample:03d}_baseline_gain1.00.npz');resp=np.zeros((6,6));az=np.zeros(6)
  for field in ('qacc','qfrc_actuator','qfrc_constraint','com_acceleration_world','foot_force'):assert abs(baseline[field]-a[field][sample]).max()<1e-9
  for leg,gain in [('baseline',1.)]+[(l,g) for l in legs for g in (.95,1.05)]:
   file=p/f'state{sample:03d}_{leg}_gain{gain:.2f}.npz';x=load(file);expected=a['act'][sample].copy()
   if leg!='baseline':expected[[j for j,r in enumerate(mapping) if r['leg']==leg]]*=gain
   assert expected.min()>=0 and expected.max()<=1;assert np.array_equal(x['act'],expected)
   for k in ('qpos','qvel','ctrl','qfrc_applied','xfrc_applied'):assert np.array_equal(x[k],a[k][sample]),k
   assert np.array_equal(x['qacc_warmstart_input'],a['native_qacc_warmstart'][sample]);f=np.zeros((6,3))
   for c in range(int(x['contact_ncon'])):
    g1,g2=map(int,x['contact_geom_ids'][c]);foot=g2 if g1 in floors else g1 if g2 in floors else -1
    if foot in gm:f[legs.index(gm[foot])]+=(1 if g1 in floors else -1)*(x['contact_frame'][c].T@x['contact_force_raw6'][c,:3])
   ferr=float(abs(f-x['foot_force']).max());assert ferr<1e-10;active=np.zeros(90);bias=np.zeros(90)
   for j,node in enumerate(nodes):force,g,b=evaluate(x['actuator_length'][j],x['actuator_velocity'][j],x['act'][j],node);active[j]=g*x['act'][j];bias[j]=b;assert abs(force-x['actuator_force'][j])<1e-10
   M=a['actuator_moment'][sample];terr=float(abs(M.T@(active+bias)-x['qfrc_actuator']).max());assert terr<1e-10;assert abs(x['M']@x['qacc']-(x['qfrc_actuator']+x['qfrc_passive']+x['qfrc_constraint']-x['qfrc_bias'])).max()<1e-9;assert abs(x['efc_J_dense'].T@x['efc_force']-x['qfrc_constraint']).max()<1e-9;assert abs(x['com_acceleration_world']-(x['contact_force_world']+x['gravity_force_world'])/x['body_mass'].sum()).max()<1e-9
   derived[file.stem+'_active_force']=active;derived[file.stem+'_bias_force']=bias;results.append(dict(file=file.name,contact_maxabs=ferr,torque_maxabs=terr,act_clipped=False))
  for j,leg in enumerate(legs):
   plus=load(p/f'state{sample:03d}_{leg}_gain1.05.npz');minus=load(p/f'state{sample:03d}_{leg}_gain0.95.npz');resp[:,j]=(plus['foot_force'][:,2]-minus['foot_force'][:,2])/.1;az[j]=(plus['com_acceleration_world'][2]-minus['com_acceleration_world'][2])/.1
  report[str(sample)]=dict(time_ms=float(a['time'][sample]*1000),dFz_d_fraction=resp.tolist(),dCOMaz_d_fraction=az.tolist(),LM_own=float(resp[legs.index('lm'),legs.index('lm')]),RM_own=float(resp[legs.index('rm'),legs.index('rm')]))
 with (p/'PARENT_ACTIVE_BIAS_RECONSTRUCTION.npz').open('xb') as f:np.savez_compressed(f,**derived)
 hashes=json.loads((p/'INPUT_HASHES.json').read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items())
 with (p/'PARENT_STATIC_INDEPENDENT_CHECK.json').open('x') as f:json.dump(dict(passed=True,raw_records=len(results),records=results,responses=report,model_load=0,forward=0,step=0),f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
