"""Read-only parameter-group authority audit from actual MN matrix and manifest."""
from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1]
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
T=Path('/home/sensen/Desktop/engine修复验收/s4-invariant-motion-target/20261102T023600Z/INVARIANT_BENDING_TARGETS.npz')
O=Path('/home/sensen/Desktop/engine修复验收/s4-bending-group-authority/20261102T042848Z')
LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);inputs=[Path(__file__),B/'RUN/MAPPING.json',B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',B/'IDENTITIES.npz',T];hashes={str(p):sha(p) for p in inputs}
 dump(O/'PREREGISTRATION.json',dict(scope='Independent S4 audit of12constant MNparameter grouping before learning; no network or optimization',source_sha256=hashes,rules='Sixleg postsynaptic readoutMN groups; node shared across muscles in sameleg belongs once, crossleg identity conflict forbids this grouping. Every incoming existing E/I edge to each MN uses sameleg positive gain. Threshold shift only on groupMN. Preserve samepooldecoder.',limits='A group parameter is a model knob, not a complete actuator control basis or a measured biological property. Counts alone do not prove dynamic or physical authority.'))
 mapping=json.loads((B/'RUN/MAPPING.json').read_text());ids=np.load(B/'IDENTITIES.npz');roots=ids['root_ids'];n=len(roots);owner={};conflicts=[];rows=[]
 E=sparse.load_npz(B/'ACTUAL_We.npz').tocsr();I=sparse.load_npz(B/'ACTUAL_Wi.npz').tocsr()
 for row in mapping['rows']:
  leg=row['actuator'][:2];assert leg in LEGS
  for i in row['local_indices']:
   if i in owner and owner[i]!=leg:conflicts.append(dict(local_index=i,prior_leg=owner[i],new_leg=leg))
   else:owner[i]=leg
 assert not conflicts
 masks=np.zeros((6,n),bool);summary=[]
 for li,leg in enumerate(LEGS):
  indices=sorted(i for i,l in owner.items() if l==leg);masks[li,indices]=True;mr=[row for row in mapping['rows'] if row['actuator'].startswith(leg)]
  incomingE=np.diff(E.indptr)[indices];incomingI=np.diff(I.indptr)[indices];zero=[]
  for i in indices:
   row=dict(leg=leg,local_index=int(i),root_888=int(roots[i]),incoming_exc=int(E.indptr[i+1]-E.indptr[i]),incoming_inh=int(I.indptr[i+1]-I.indptr[i]),muscles_sharing_cell=sum(i in row['local_indices'] for row in mr))
   rows.append(row)
   if row['incoming_exc']+row['incoming_inh']==0:zero.append(int(roots[i]))
  summary.append(dict(leg=leg,readout_MN=len(indices),incoming_exc=int(incomingE.sum()),incoming_inh=int(incomingI.sum()),MN_without_modeled_incoming=len(zero),MN_without_modeled_incoming_roots=zero,actuators=len(mr),complete=sum(r['status']=='complete' for r in mr),partial=sum(r['status']=='partial' for r in mr),unresolved=sum(not r['local_indices'] for r in mr),unique_present_pools=len({tuple(r['local_indices']) for r in mr if r['local_indices']})))
 assert masks.sum()==204 and np.all(masks.sum(0)<=1)
 target=np.load(T);assert target['angles_rad'].shape==(37,6,2) and target['angle_valid'].all()
 # Native global parameter scope is explicit and disjoint; no runtime matrix built.
 np.savez_compressed(O/'MN_GROUP_MASKS.npz',root_ids=roots,leg_names=np.array(LEGS),post_MN_mask=masks,source_group=np.where(masks.any(0),np.argmax(masks,axis=0),-1),threshold_shift_bound_mV=np.array([-2.,2.]),incoming_gain_bounds=np.array([.5,2.]))
 with (O/'GROUP_MEMBER_AUTHORITY.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result=dict(passed=True,source_sha256=hashes,neurons=n,distinct_readout_MN=len(owner),crossleg_conflicts=conflicts,groups=summary,unique_present_pools=sum(r['unique_present_pools'] for r in summary),total_selected_incoming_exc=sum(r['incoming_exc'] for r in summary),total_selected_incoming_inh=sum(r['incoming_inh'] for r in summary),parameter_names=[f'{leg}_MN_incoming_gain' for leg in LEGS]+[f'{leg}_MN_threshold_shift_mV' for leg in LEGS],no_extra_neurons_or_synapses=True,calls=dict(model_compile=0,network=0,neural_step=0,body_step=0,training=0),limits='MNincominggain cannot alter missing edges or decode additional16unresolved muscles. Sharedpools remain bound. Thresholds/gains do not uniquely specify sixleg trajectories or establish standing.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
