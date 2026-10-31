"""S144 offline independent geometry/native dynamics and original-gate verifier."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
OLD=Path('/home/sensen/Desktop/engine修复验收/s143-initial-contact-preload/20261024T202354366515Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();p=args.output.resolve();r=json.loads((p/'RESULT.json').read_text());old=json.loads((OLD/'RESULT.json').read_text());meta=json.loads((p/'METADATA.json').read_text());d=old['d'];assert d>0 and r['d']==d
 for name,digest in r['source_hashes'].items():assert sha(name)==digest,name
 assert json.loads((p/'MODEL_NUMERIC_BEFORE.json').read_text())==json.loads((p/'MODEL_NUMERIC_AFTER.json').read_text())
 base=load(OLD/'STATIC_0.npz');rows=[]
 for i in range(3):
  a=load(p/f'STATIC_{i}.npz');offset=-d*i/4;expected=base['qpos'].copy();expected[2]+=offset
  assert np.array_equal(a['qpos'],expected);assert float(a['dz'])==offset
  for k in ('qvel','act','ctrl','qfrc_applied','xfrc_applied','warnings'):assert not a[k].any(),k
  assert float(a['time'])==0
  err=float(abs(a['constraint__efc_J_dense'].T@a['constraint__efc_force']-a['qfrc_constraint']).max());assert err<=1e-10
  assert abs(a['qfrc_actuator']+a['qfrc_passive']+a['qfrc_constraint']-a['qfrc_bias']-a['Ma']).max()<=1e-10
  acc=np.einsum('bkd,d->bk',a['body_com_jacp'],a['qacc'])+np.einsum('bkd,d->bk',a['body_com_jacdot'],a['qvel']);com=(acc*a['body_mass'][:,None]).sum(0)/a['body_mass'].sum();assert np.max(abs(com-a['com_acceleration_world']))<=1e-9
  assert abs(com-a['external_force_world']/a['body_mass'].sum()).max()<=1e-9
  angle=2*np.arccos(np.clip(abs(a['qpos'][3:7]@base['qpos'][3:7])/(np.linalg.norm(a['qpos'][3:7])*np.linalg.norm(base['qpos'][3:7])),0,1));gates=np.array([a['qpos'][2]/base['qpos'][2]>=.85,angle<=.35,np.count_nonzero(a['contacts'])>=3,a['load_ratio']>=.8,a['nonfoot_depth']<=1e-12,a['limit_excess']<=1e-6]);assert np.array_equal(gates,a['six_gates'])
  if i==0:
   mask=a['bearing_contact_mask'].astype(bool)&(a['contact_foot_leg_index']>=0)&(a['contact_normal_force_on_foot_world'][:,2]>0)&(a['contact_dist']<0);assert float(np.max(-a['contact_dist'][mask]))==d
   for key in ('qpos','qvel','act','ctrl','qacc','contact_dist','contact_geom_ids','com_acceleration_world'):assert np.array_equal(a[key],base[key]),key
  az=float(com[2]);assert abs(az-r['results'][i]['com_az'])<=1e-9;assert bool(gates.all())==r['results'][i]['eligible']
  rows.append(dict(index=i,dz=offset,eligible=bool(gates.all()),COM_az=az,abs_COM_az=abs(az),load_ratio=float(a['load_ratio']),constraint_maxabs=err,contacts=a['contacts'].tolist(),bearing_contacts=a['bearing_contacts'].tolist(),max_penetration=float(max(0,-a['contact_dist'].min()))))
 selected=min((x for x in rows if x['eligible']),key=lambda x:(x['abs_COM_az'],abs(x['dz'])));assert selected['index']==r['selected']['index'];eligible=selected['index']!=0 and selected['abs_COM_az']<rows[0]['abs_COM_az'];assert eligible==r['dynamic_eligible_pending_independent_verifier'];assert not r['dynamic_executed']
 report=dict(passed=True,rows=rows,selected=selected,dynamic_eligible=eligible,dynamic_executed=False,prewarm_success=False,physical_units='NOT_ESTABLISHED',verifier=str(Path(__file__)),verifier_sha256=sha(__file__),hashes={str(f):sha(f) for f in [p/'RESULT.json',OLD/'RESULT.json',*[p/f'STATIC_{i}.npz' for i in range(3)]]})
 with (p/'DOWNWARD_STATIC_INDEPENDENT_CHECK.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
