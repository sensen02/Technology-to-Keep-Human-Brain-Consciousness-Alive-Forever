"""Independent saved-moment/shared-pool audit, no optimization/dynamics."""
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s2-early-muscle-support/20261102T053639Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
O=ROOT/'s4-antagonist-pool-coupling/20261102T061756Z'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 inputs=[Path(__file__),S/'TRAIN_000_BASELINE_AUDIT.npz',S/'IDENTITY_AND_MAPPING.json',B/'RUN/MAPPING.json',B/'IDENTITIES.npz'];hashes={str(p):sha(p) for p in inputs}
 dump(O/'PREREGISTRATION.json',dict(scope='Independent origin/moment/pool overlap audit before closedloop neural pool trial',source_sha256=hashes,target_joints=['joint_LFTibia_pitch','joint_LMCoxa_yaw'],definition='Present nonempty readout pools grouped by exactMNset; contraction coefficient is sum(moment*gain) over all muscles sharing that pool. Sign is coefficient,not muscle name. Compare releaseboundary andfirst5mssignedmean; no policy,fit or modelcalls.'))
 meta=json.loads((S/'IDENTITY_AND_MAPPING.json').read_text());mapping=json.loads((B/'RUN/MAPPING.json').read_text())['rows'];ids=np.load(B/'IDENTITIES.npz')['root_ids']
 with np.load(S/'TRAIN_000_BASELINE_AUDIT.npz') as z:a={k:z[k] for k in ['moment_42x90','actuator_gain','relative_time_s']}
 group={}
 for row in mapping:
  key=tuple(row['local_indices'])
  if key:group.setdefault(key,[]).append(row['actuator_index'])
 assert len(group)==66
 coefficients=a['moment_42x90']*a['actuator_gain'][:,None,:];targets={};allrows=[];selected={}
 for joint in ['joint_LFTibia_pitch','joint_LMCoxa_yaw']:
  j=meta['joint_names'].index(joint);leg=joint[6:8];rows=[]
  for gi,(key,muscles) in enumerate(group.items()):
   values=coefficients[:,j,muscles].sum(1)
   if max(abs(values))<1e-12:continue
   row=dict(target=joint,pool_index=gi,MN_local_indices=list(key),MN_root_ids=[int(ids[i]) for i in key],muscles=[mapping[i]['actuator'] for i in muscles],coefficient_boundary=float(values[40]),coefficient_first5_mean=float(values[40:50].mean()),coefficient_first5_min=float(values[40:50].min()),coefficient_first5_max=float(values[40:50].max()),all_mapped_muscle_legs=sorted({mapping[i]['leg'] for i in muscles}))
   rows.append(row);allrows.append(row)
  pos=[row for row in rows if row['coefficient_boundary']>0];neg=[row for row in rows if row['coefficient_boundary']<0]
  strongest_pos=max(pos,key=lambda r:r['coefficient_boundary']) if pos else None;strongest_neg=min(neg,key=lambda r:r['coefficient_boundary']) if neg else None
  pair=[row for row in [strongest_pos,strongest_neg] if row];conflict=sorted(set(pair[0]['MN_local_indices'])&set(pair[1]['MN_local_indices'])) if len(pair)==2 else []
  targets[joint]=dict(nonzero_pools=len(rows),positive_pools=len(pos),negative_pools=len(neg),strongest_positive=strongest_pos,strongest_negative=strongest_neg,antagonist_MN_overlap=conflict)
  for row in pair:selected[(joint,row['pool_index'])]=row
 with (O/'POOL_COEFFICIENTS.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['joint','pool_index','muscles','MNroots','coefficient_release','coefficient_mean0_5ms','min0_5ms','max0_5ms'])
  for row in allrows:writer.writerow([row['target'],row['pool_index'],'|'.join(row['muscles']),'|'.join(map(str,row['MN_root_ids'])),row['coefficient_boundary'],row['coefficient_first5_mean'],row['coefficient_first5_min'],row['coefficient_first5_max']])
 # Every pool whose members overlap a selectedMN can be affected by a threshold shift.
 collateral=[]
 for (joint,gi),chosen in selected.items():
  nodes=set(chosen['MN_local_indices']);overlaps=[]
  for other,(key,muscles) in enumerate(group.items()):
   shared=sorted(nodes&set(key))
   if shared:overlaps.append(dict(pool_index=other,MN_overlap=shared,muscles=[mapping[i]['actuator'] for i in muscles],is_selected_exact_pool=other==gi))
  collateral.append(dict(target=joint,selected_pool=gi,overlapping_readout_pools=overlaps))
 report=dict(passed=True,source_sha256=hashes,targets=targets,collateral_readout=collateral,counts=dict(pools=66,readout_neurons=204),calls=dict(model_compile=0,forward=0,neural_step=0,body_step=0,training=0),limits='Pool contraction is hypothetical simultaneousactivation,not instantaneouscommandcausality. Positive vsnegative torque does not say which is beneficial at all gait phases. Threshold changes may affect everyreadout sharingMN; body validation is separate.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','collateral_readout']},indent=2))
if __name__=='__main__':main()
