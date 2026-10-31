"""S120 saved-array schema audit only. No simulator or neural engine imports."""
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np
B=Path('/home/sensen/Desktop/engine修复验收');OUT=B/'s120-constraint-identifiability/20261024T071500Z'
REFS=dict(S113=B/'s113-com-timestep/20261024T060500Z/dt25us',S115=B/'s115-extended-initialization/20261024T062500Z/init80ms',S117=B/'s117-single-weight-attenuation/20261024T064500Z/w36b4_down',S118=B/'s118-secant-weight-step/20261024T065500Z/w36b4_secant')
C=B/'s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(n,rows):
 with (OUT/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 hashes={str(C):sha(C),str(Path(__file__).resolve()):sha(__file__)};inventory={};gaps=[];timeline=[];facts={}
 required=dict(nefc='valid row count and offsets per state',efc_type='per-row constraint type',efc_id='per-row object ID with model/type dictionary',efc_J='exact constraint Jacobian row columns for nv48; dense or full CSR',efc_force='solved row scalar force same sampled forward state',contact_efc_address='per-contact first row index or inactive marker',efc_row_contact_index='explicit row-to-contact map or reconstructable address/dimension/cone metadata',efc_row_joint_id='limit row-to-joint/type mapping',body_jacr='angular world Jacobian needed with COM translational Jacobian for arbitrary-point wrench projection')
 for label,p in REFS.items():
  for mapping in (js(p.parent/'INPUT_HASHES.json'),js(p/'RESULT.json')['hashes']):
   for name,digest in mapping.items():assert Path(name).is_file() and sha(name)==digest,name
  for f in p.iterdir():
   if f.is_file():hashes[str(f.resolve())]=sha(f)
  schemas={}
  for f in p.glob('*.npz'):
   with np.load(f,allow_pickle=False) as z:schemas[f.name]={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
  inventory[label]=dict(path=str(p),npz_fields=schemas,sidecars={f.name:js(f) for f in p.glob('*.json') if f.name not in ('RESULT.json',)})
  with np.load(p/'TRAJECTORY.npz') as z:a={k:z[k] for k in z.files}
  keys=set().union(*(set(s) for s in schemas.values()));assert not any(k in keys for k in ('efc_type','efc_id','efc_J','efc_force','contact_efc_address'))
  for field,need in required.items():gaps.append(dict(sample=label,field=field,present=field in keys,need=need,consequence='cannot identify contact/limit/other generalized constraint split' if field!='body_jacr' else 'COM jacp alone is not contact-point translational + angular Jacobian'))
  total=a['qfrc_actuator']+a['qfrc_passive']-a['qfrc_bias']+a['qfrc_constraint']+a['qfrc_applied'];assert np.max(abs(total-a['Ma']))<1e-10
  t=a['time'];first=float(t[np.flatnonzero(a['load_ratio']<.8)[0]])
  for i in np.flatnonzero((t>=min(first,.0017)-1e-12)&(t<=.002825+1e-12)):
   for d in (6,7,8):
    tau=a['actuator_force'][i]*a['actuator_moment'][i,:,d];assert abs(tau.sum()-a['qfrc_actuator'][i,d])<1e-10
    timeline.append(dict(sample=label,state=int(i),time_ms=float(t[i]*1000),dof=d,roll_q=float(a['qpos'][i,9]),roll_velocity=float(a['qvel'][i,8]),load_ratio=float(a['load_ratio'][i]),global_limit=float(a['limit_excess'][i]),act2_force=float(a['actuator_force'][i,2]),act2_torque=float(tau[2]),actuator_total=float(tau.sum()),passive=float(a['qfrc_passive'][i,d]),bias=float(a['qfrc_bias'][i,d]),constraint_total=float(a['qfrc_constraint'][i,d]),Ma=float(a['Ma'][i,d]),qacc=float(a['qacc'][i,d]),row_split='NOT_IDENTIFIABLE'))
  i=100;facts[label]=dict(first_load_ms=first*1000,first_limit_ms=float(t[np.flatnonzero(a['limit_excess']>1e-6)[0]]*1000),peak_limit=float(a['limit_excess'].max()),peak_ms=float(t[a['limit_excess'].argmax()]*1000),time2p5=dict(roll_q=float(a['qpos'][i,9]),roll_velocity=float(a['qvel'][i,8]),act2_tau8=float(a['actuator_force'][i,2]*a['actuator_moment'][i,2,8]),total_actuator8=float(a['qfrc_actuator'][i,8]),total_constraint8=float(a['qfrc_constraint'][i,8]),passive8=float(a['qfrc_passive'][i,8]),bias8=float(a['qfrc_bias'][i,8]),Ma8=float(a['Ma'][i,8])),max_generalized_balance_residual=float(abs(total-a['Ma']).max()))
 assert not any(n.startswith(('mujoco','torch')) for n in sys.modules)
 OUT.mkdir(parents=True,exist_ok=False)
 save('INPUT_HASHES.json',hashes);save('SAVED_FIELD_INVENTORY.json',inventory);table('FIELD_GAPS.csv',gaps);table('ALIGNED_TOTAL_BALANCE.csv',timeline)
 save('IDENTIFIABILITY_RESULT.json',dict(identifiable=False,row_decomposition_possible=False,contact_wrench_saved=True,aggregate_balance_reconstructable=True,facts=facts,zero_model_loads=True,zero_forward=True,zero_steps=True,zero_neural_ticks=True,weight_writes=False,constraints_sha256=sha(C),conclusion='Total constraint changes measurable but contact/limit/other contributions not separable; no residual labeled joint limit, no new parameter proposal.'))
 save('MINIMUM_NEXT_RECORDING.json',dict(scope='future necessary recording only, NOT an experiment started or authorized by this file',per_state=['time/source and physical tick, nv, nefc, active row offsets', 'efc_type[nefc], efc_id[nefc] with enum and object ID names', 'efc_force[nefc] and efc_J[nefc,nv] exact same forward state; if sparse save CSR rownnz,rowadr,colind and values', 'contact count/validity, geom/body IDs, position/frame/dim and efc_address per contact; exact cone type and resulting row count or explicit row-to-contact index', 'map each noncontact row to joint/tendon/equality/other object according to efc_type; preserve unclassified rows', 'qfrc_constraint[nv] plus existing qpos/qvel/qacc/Ma/passive/bias/actuator and contact/force arrays'],validation='Sum Jrow.T * efc_force by original row types; sum all categories must reproduce saved qfrc_constraint on all48 DOFs, including6/7/8. Never slice by ncon; inactive/condim/cone and sparse addresses may change.',optional='Contact-point translational+angular Jacobians if independently validating wrench projection; not a substitute for row type/id audit',sampling='native25us exactly aligned after same forward state; serialize copies and ragged counts; no new dynamics tolerance/framework'))
 assert all(sha(name)==h for name,h in hashes.items())
 save('ARTIFACT_HASHES.json',{str(f):sha(f) for f in OUT.iterdir() if f.is_file()})
 print(json.dumps(dict(passed=True,identifiable=False,field_gap_rows=len(gaps),aligned_rows=len(timeline),bound_files=len(hashes),output=str(OUT))))
if __name__=='__main__':main()
