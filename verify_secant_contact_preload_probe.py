"""S145 independent full precision secant and static native evidence check."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output.resolve();r=json.loads((p/'RESULT.json').read_text());calc=json.loads((p/'SECANT_CALCULATION.json').read_text());e=[load(x['snapshot']) for x in calc['endpoints']];z0,z1=[float(x['dz']) for x in e];a0,a1=[float(x['com_acceleration_world'][2]) for x in e];offset=z0-a0*(z1-z0)/(a1-a0);assert np.isfinite(offset) and min(z0,z1)<offset<max(z0,z1);assert offset==calc['zstar']
 for path,digest in r['source_hashes'].items():assert sha(path)==digest,path
 x=load(p/'STATIC_SECANT.npz');q=e[0]['qpos'].copy();q[2]+=offset;assert np.array_equal(q,x['qpos']) and float(x['dz'])==offset
 for k in ('qvel','act','ctrl','qfrc_applied','xfrc_applied','warnings'):assert not x[k].any()
 assert json.loads((p/'MODEL_NUMERIC_BEFORE.json').read_text())==json.loads((p/'MODEL_NUMERIC_AFTER.json').read_text())
 residual=float(abs(x['constraint__efc_J_dense'].T@x['constraint__efc_force']-x['qfrc_constraint']).max());assert residual<=1e-10
 assert abs(x['qfrc_actuator']+x['qfrc_passive']+x['qfrc_constraint']-x['qfrc_bias']-x['Ma']).max()<=1e-10
 acc=np.einsum('bkd,d->bk',x['body_com_jacp'],x['qacc'])+np.einsum('bkd,d->bk',x['body_com_jacdot'],x['qvel']);com=(acc*x['body_mass'][:,None]).sum(0)/x['body_mass'].sum();assert abs(com-x['com_acceleration_world']).max()<=1e-9;assert abs(com-x['external_force_world']/x['body_mass'].sum()).max()<=1e-9
 gates=np.array([q[2]/e[0]['qpos'][2]>=.85,x['root_angle_rad']<=.35,np.count_nonzero(x['contacts'])>=3,x['load_ratio']>=.8,x['nonfoot_depth']<=1e-12,x['limit_excess']<=1e-6]);assert np.array_equal(gates,x['six_gates']);eligible=bool(gates.all() and abs(com[2])<abs(a0));assert eligible==r['dynamic_eligible_pending_independent_verifier']
 assert abs(x['contact_dist']-(e[0]['contact_dist']+offset)).max()<=1e-12;assert np.array_equal(x['contact_geom_ids'],e[0]['contact_geom_ids'])
 report=dict(passed=True,dynamic_eligible=eligible,offset=offset,reference_COM_az=a0,actual_COM_az=float(com[2]),load_ratio=float(x['load_ratio']),constraint_maxabs=residual,root_height_only=True,model_unchanged=True,prewarm_success=False,S119_same_condition_candidate=False,physical_units='NOT_ESTABLISHED',verifier=str(Path(__file__)),verifier_sha256=sha(__file__),hashes={str(f):sha(f) for f in (p/'RESULT.json',p/'STATIC_SECANT.npz',p/'SECANT_CALCULATION.json')})
 with (p/'STATIC_SECANT_INDEPENDENT_CHECK.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
