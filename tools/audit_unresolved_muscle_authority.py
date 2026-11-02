"""Actual mapping gaps and saved full-cycle force-direction authority; no models/LP."""
from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收')
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
S=ROOT/'s2-reference-muscle-operating-range/20261102T155412Z'
O=ROOT/'s2-unresolved-authority-audit/20261102T170741Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),B/'RUN/MAPPING.json',B/'IDENTITIES.npz',S/'AUDIT.npz'];hashes={str(p):sha(p) for p in files};dump(O/'PREREGISTRATION.json',dict(scope='Existingmapping andsavedstate coordinateauthority only;0model/step/LP',source_sha256=hashes,threshold=1e-12,activation='Theoretical90independent vs74independent;not feasibleequilibrium or controller,not inventedMN'))
 rows=json.loads((B/'RUN/MAPPING.json').read_text())['rows'];counts=collections.Counter(row['status'] for row in rows);assert dict(counts)==dict(complete=67,unresolved=16,partial=7)
 with np.load(S/'AUDIT.npz') as z:a={k:z[k] for k in ['time','coefficient_42x90','moment_42x90','gain_zero_velocity','joint_names','activation','command']}
 mask=np.array([not r['local_indices'] for r in rows]);assert int(mask.sum())==16
 reasons=collections.Counter();gap=[]
 for u in np.flatnonzero(mask):
  r=rows[u];annotation=r['source_annotation'];reason=annotation['reason'];reasons[reason]+=1
  affected=np.flatnonzero(abs(a['coefficient_42x90'][:,:,u]).max(0)>1e-12)
  gap.append(dict(actuator_index=int(u),actuator=r['actuator'],leg=r['leg'],source_target=r['muscle_target'],reason=reason,source_annotation=annotation,local_indices=r['local_indices'],pool_root_ids=r['pool_root_ids'],nonzero_dynamic_coefficient_joints=[str(a['joint_names'][j]) for j in affected],command_all_zero=bool(not a['command'][:,u].any()),activation_floor_all=bool(np.max(abs(a['activation'][:,u]-1e-4))<=1e-12)))
 out=[];dynamic=a['coefficient_42x90'];stationary=a['moment_42x90']*a['gain_zero_velocity'][:,None,:]
 for kind,A in [('actual_velocity',dynamic),('same_length_zero_velocity',stationary)]:
  for j,name in enumerate(a['joint_names']):
   mapped=A[:,j,~mask];unmapped=A[:,j,mask];pos74=np.maximum(mapped,0).sum(1);neg74=np.minimum(mapped,0).sum(1);pos16=np.maximum(unmapped,0).sum(1);neg16=np.minimum(unmapped,0).sum(1)
   out.append(dict(condition=kind,joint=str(name),new_positive_direction_samples=int(((pos74<=1e-12)&(pos16>1e-12)).sum()),new_negative_direction_samples=int(((neg74>=-1e-12)&(neg16<-1e-12)).sum()),missing_positive_coefficient_max=float(pos16.max()),missing_negative_coefficient_min=float(neg16.min()),boundary_mapped_positive=float(pos74[-1]),boundary_missing_positive=float(pos16[-1]),boundary_mapped_negative=float(neg74[-1]),boundary_missing_negative=float(neg16[-1])))
 with (O/'COORDINATE_AUTHORITY.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 result=dict(passed=True,source_sha256=hashes,status_counts=dict(counts),missing=gap,source_reason_counts=dict(reasons),coordinates=out,samples=401,calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0,LP=0),limits='Missingmappingdoesnotmeanmuscleabsent. Thesearesignedactivecoefficients,not nettorque or staticwitness. Samevelocityzero ismathematicalcontrast,not actualtrajectory. Extraauthorityisnotverifiedanatomy or permissiontoinventMN. Full90alreadyincludesunmappedmuscles;cannotassert mappingaddspositiveability wherefull90lacksit.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k in ['passed','status_counts','source_reason_counts','samples','calls']},indent=2))
if __name__=='__main__':main()
