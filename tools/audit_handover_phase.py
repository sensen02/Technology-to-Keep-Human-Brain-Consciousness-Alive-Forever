"""Saved-trajectory handover phase audit; no rollout or preferred phase selection."""
from pathlib import Path
import json,hashlib,math,csv
import numpy as np
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-handover-phase-audit/20261102T071409Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 sources=[Path(__file__),B/'RUN/BODY.npz',B/'RUN/PREREGISTRATION.json',B/'RUN/GAINS.json'];hashes={str(p):sha(p) for p in sources}
 dump(O/'PREREGISTRATION.json',dict(scope='Offline originalwarmup phase/time audit,no selectednewhandover or dynamicprediction',source_sha256=hashes,window=[1.1,1.3],cycle='Declared5Hz algorithmreference only; not measuredanimalperiod',diagnostics='Savedfoot/nonfoot/COM/assist plus finite-difference COMvelocity; endpointphasecompare descriptive, never mark an untested alternative phase successful.',calls=dict(model_compile=0,body_step=0,neural_step=0,optimizer=0)))
 with np.load(B/'RUN/BODY.npz',allow_pickle=False) as z:b={k:z[k] for k in z.files}
 t=b['time'];idx=np.flatnonzero((t>=1.1-1e-10)&(t<=1.3+1e-10));assert len(idx)==401
 W=float(json.loads((B/'RUN/GAINS.json').read_text())['weight']);vz=np.gradient(b['com'][:,2],t);rootv=b['qvel'][:,:3]
 values=[]
 for i in idx:
  values.append(dict(time=float(t[i]),reference_phase_rad=float(((max(float(t[i])-.15,0)*2*math.pi*5)%(2*math.pi))),foot_load=float(b['foot_force'][i,:,2].sum()/W),nonfoot_count=int(b['nonfoot_count'][i]),COMz=float(b['com'][i,2]),COMvz_fd=float(vz[i]),rootvz=float(rootv[i,2]),assist_L1=float(abs(b['assist'][i,6:]).sum()),rootassist_max=float(abs(b['assist'][i,:6]).max())))
 with (O/'WARMUP_PHASE_SAMPLES.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 # At1.3s BODY is afterwithdrawal; rawprewithdraw assistance exists separately.
 body_boundary=values[-1];before=np.load(B/'RUN/HANDOVER_BEFORE.npz');assert abs(float(before['time'])-1.3)<1e-10
 cycles=[]
 for start in [1.1,1.15,1.2,1.25]:
  m=(t>=start-1e-10)&(t<start+.05-1e-10);cycles.append(dict(start_s=start,end_s=start+.05,frames=int(m.sum()),COMz_min=float(b['com'][m,2].min()),COMz_max=float(b['com'][m,2].max()),COMvz_fd_min=float(vz[m].min()),COMvz_fd_max=float(vz[m].max()),mean_foot_load=float(b['foot_force'][m,:,2].sum(1).mean()/W),nonfoot_frames=int((b['nonfoot_count'][m]>0).sum())))
 result=dict(passed=True,source_sha256=hashes,declared_reference_frequency_hz=5.,warmup_complete_reference_cycle_s=.2,phase_at_withdrawal_rad=body_boundary['reference_phase_rad'],last_cycle_statistics=cycles,withdrawal_BODY_after=body_boundary,prewithdraw_assist_L1=float(abs(before['qfrc_applied'][6:]).sum()),prewithdraw_rootassist_max=float(abs(before['qfrc_applied'][:6]).max()),COMz_last_cycle_range=[float(b['com'][idx,2].min()),float(b['com'][idx,2].max())],last_cycle_COMvz_fd_range=[float(vz[idx].min()),float(vz[idx].max())],calls=dict(model_compile=0,body_step=0,neural_step=0,optimizer=0),limits='FDCOMvelocity is descriptive and not exactsaved COMJac velocity; boundary gradient includes firstreleased frame. BODY1.3s is withdrawalafter andassistance0, prewithdrawsnapshot kept separate. Phase0 atreferenceclock is algorithmic,not animal phase or proof of a safe alternate release. No phase selected or simulated.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','last_cycle_statistics']},indent=2))
if __name__=='__main__':main()
