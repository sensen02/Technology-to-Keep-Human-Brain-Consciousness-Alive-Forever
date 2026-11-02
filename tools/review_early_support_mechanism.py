"""Independent saved-array mechanics audit; no engine, solver or new intervention."""
from pathlib import Path
import json,hashlib,math,csv
import numpy as np
S=Path('/home/sensen/Desktop/engine修复验收/s2-early-muscle-support/20261102T053639Z')
T=Path('/home/sensen/Desktop/engine修复验收/s4-bending-closedloop-training/20261102T042848Z')
O=Path('/home/sensen/Desktop/engine修复验收/s2-early-support-parent-review/20261102T053639Z')
LABELS=['TRAIN_000_BASELINE','TRAIN_017_MINUS']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'RESULT.json').exists(),'No completed scientificreview overwrite'
 files=[Path(__file__),S/'RESULT.json',S/'IDENTITY_AND_MAPPING.json',S/'PHASE_LEG_CORRECTION.json',S/'GAIN_ACTIVATION_ADDENDUM.json']+[S/(label+suffix) for label in LABELS for suffix in ['_AUDIT.npz','_GAIN_ACTIVATION.npz']]
 hashes={str(p):sha(p) for p in files};dump(O/('CACHED_ARRAY_REVIEW_METHOD.json' if (O/'PREREGISTRATION.json').exists() else 'PREREGISTRATION.json'),dict(scope='Independent completedmuscle reconstruction arithmetic and savedtime audit only; no model/forward/step/optimizer',source_sha256=hashes,checks='Momentmatrix products, gainactivationbiasforce, exact gain/activation/moment algebra, corrected perleg grouping, sourcebody times and preserved assists. Not an intervention on physics or proof of a unique cause.'))
 meta=json.loads((S/'IDENTITY_AND_MAPPING.json').read_text());result=json.loads((S/'RESULT.json').read_text());correction=json.loads((S/'PHASE_LEG_CORRECTION.json').read_text());summary={};maxforce=maxtorque=maxdecomp=0.
 for label in LABELS:
  a=arrays(S/(label+'_AUDIT.npz'));d=arrays(S/(label+'_GAIN_ACTIVATION.npz'));old=arrays(T/label/'RUN/BODY.npz');ids=a['source_indices'];assert len(ids)==101 and np.array_equal(a['qpos'],old['qpos'][ids]) and np.array_equal(a['command'],old['command'][ids]) and np.array_equal(a['activation'],old['activation'][ids])
  assert np.array_equal(a['native_foot_force'],old['foot_force'][ids]) and np.array_equal(a['native_com'],old['com'][ids])
  for t in range(101):
   for u in range(90):
    f=float(a['actuator_gain'][t,u])*float(a['activation'][t,u])+float(a['actuator_bias'][t,u]);maxforce=max(maxforce,abs(f-float(a['force'][t,u])))
   for j in range(48):
    torque=math.fsum(float(a['moment_90x48'][t,u,j])*float(a['force'][t,u]) for u in range(90));maxtorque=max(maxtorque,abs(torque-float(a['qfrc_actuator'][t,j])))
  assert maxforce<1e-10 and maxtorque<1e-10
  gain=a['actuator_gain'];activation=a['activation'];dg=(gain-gain[40])*(activation+activation[40])/2;da=(activation-activation[40])*(gain+gain[40])/2
  maxdecomp=max(maxdecomp,float(abs(dg-d['delta_force_gain']).max()),float(abs(da-d['delta_force_activation']).max()),float(abs(dg+da-(a['active_force']-a['active_force'][40])).max()))
  current=a['moment_42x90'];total=np.einsum('tju,tu->tj',current,dg+da)+np.einsum('tju,u->tj',current-current[40],a['active_force'][40]);assert abs(total-(a['qfrc_active'][:,6:]-a['qfrc_active'][40,6:])).max()<1e-10
  native=a['native_release_rows'];nf=np.flatnonzero(native[:,7]>0);first=float(native[nf[0],0]-1.3);assert abs(first*1000-result['variants'][label]['first_nonfoot_ms'])<1e-10
  assert not a['assist'][40:].any()
  before=np.load(T/label/'RUN/HANDOVER_BEFORE.npz');assert np.array_equal(a['removed_assist'],before['qfrc_applied'])
  legs=[]
  for leg in meta['legs']:
   idx=[j for j,name in enumerate(meta['joint_names']) if name.startswith('joint_'+leg)];assert len(idx)==7
   fullidx=np.array(idx)+6;tau=a['qfrc_actuator'][40,fullidx];active=a['qfrc_active'][40,fullidx];assist=a['removed_assist'][fullidx];norm=float(np.linalg.norm(tau)*np.linalg.norm(assist));cos=None if norm==0 else float(np.dot(tau,assist)/norm)
   phases={}
   for name,start,end in [('0_5ms',40,50),('5_10ms',50,60),('10_18ms',60,76)]:
    phases[name]=dict(active_torque_L1_mean=float(abs(a['qfrc_active'][start:end][:,fullidx]).sum(1).mean()),activation_mean=float(a['activation'][start:end][:,[i for i,row in enumerate(meta['mapping']['rows']) if row['leg']==leg]].mean()))
   legs.append(dict(leg=leg,boundary_torque_vs_removed_assist_cosine=cos,boundary_active_torque_L1=float(abs(active).sum()),boundary_total_muscle_torque_L1=float(abs(tau).sum()),removed_algorithm_assist_L1=float(abs(assist).sum()),phases=phases))
  early=(a['neural_release_time']>1e-10)&(a['neural_release_time']<=.018+1e-10)
  summary[label]=dict(first_nonfoot_s=first,boundary_activation_mean=float(a['activation'][40].mean()),boundary_low_activation_count=int((a['activation'][40]<.01).sum()),boundary_active_torque_L1=float(abs(a['qfrc_active'][40,6:]).sum()),removed_algorithm_assist_L1=float(abs(a['removed_assist'][6:]).sum()),boundary_COM_velocity=a['com_velocity'][40].tolist(),early18ms_readout_MN_spikes=int(a['MN_pulse_bits'][early].sum()),corrected_per_leg=legs,interpretation='Removedassist is prioralgorithm action,not independently requiredstatic torque. Normsizes cannot prove physical insufficiency; opposingdirections and timing are conditionalobservations. Native gain mixes length/velocity factors.')
 assert maxdecomp<1e-10 and all(sha(Path(p))==h for p,h in hashes.items())
 dump(O/'RESULT.json',dict(passed=True,source_sha256=hashes,scalar_force_max_error=maxforce,scalar_moment_torque_max_error=maxtorque,gain_activation_force_decomposition_max_error=maxdecomp,variants=summary,original_perleg_startswith_field_valid=False,corrected_leg_membership_exact=True,calls=dict(model_compile=0,forward=0,Jacobian=0,body_step=0,network=0,neural_step=0,training=0),limits='101sharedframes per variant; footforce takenfromtruearchives,not claimedexactsolverstate restoration. Active/passive/algebra decomposition is descriptive,not a unique causal intervention. No argument to inventconnections or enlarge muscleforces.'));print(json.dumps(dict(passed=True,maxforce=maxforce,maxtorque=maxtorque,maxdecomp=maxdecomp,variants={k:{key:v for key,v in val.items() if key!='corrected_per_leg'} for k,val in summary.items()}),indent=2))
if __name__=='__main__':main()
