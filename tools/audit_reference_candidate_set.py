"""Freeze saved reference poses before conditional capacity selection, no physics."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收')
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z/RUN'
M=ROOT/'s2-reference-muscle-operating-range/20261102T155412Z'
O=ROOT/'s2-reference-candidate-set/20261102T182109Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),B/'BODY.npz',B/'MODEL.xml',M/'COMPILED_ACTUAL_ARRAYS.npz'];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(scope='Freeze33observedassistedreferenceposes only,not newanimaldata/IK/control or feasibilitychoice',source_sha256=hashes,indices='rint(linspace(2200,2600,33)); final1.3posecapacitycached,no repeatedsolve',selection='Staticcandidate must havecheckednativeprimalwitness. Prefer firstsourceindexnative66; fallbackfirstnative90onlyasnonneuralupperbounddiagnostic. Unsuccessfulouteronly isnoteligible.',limits='Do notclip savedq; nominaljointlimitsareoriginalsoftconstraints. Measuredovershoot/activeconstraintmustbehandled explicitly in staticcapture,not hidden. No arbitrarynewpose or forceparameter.'))
 with np.load(B/'BODY.npz') as z:b={k:z[k] for k in ['time','qpos','qvel','nonfoot_count','com','foot_contact','assist','external']}
 with np.load(M/'COMPILED_ACTUAL_ARRAYS.npz') as z:m={k:z[k] for k in ['jnt_type','jnt_qposadr','jnt_range']}
 ix=np.rint(np.linspace(2200,2600,33)).astype(int);assert len(set(ix))==33 and ix[0]==2200 and ix[-1]==2600
 times=b['time'][ix];desired=np.linspace(1.1,1.3,33);joints=np.flatnonzero(m['jnt_type']==3);adr=m['jnt_qposadr'][joints];q=b['qpos'][ix][:,adr];lo=m['jnt_range'][joints,0];hi=m['jnt_range'][joints,1];lower=np.maximum(lo-q,0);upper=np.maximum(q-hi,0)
 assert len(joints)==42 and abs(times-desired).max()<=.00025+1e-10
 np.savez_compressed(O/'FROZEN_CANDIDATE_SET.npz',source_BODY_indices=ix,actual_body_time=times,nominal_phase_time=desired,qpos=b['qpos'][ix],qvel_actual=b['qvel'][ix],lower_nominal_range_overshoot=lower,upper_nominal_range_overshoot=upper,nominal_range_lower=lo,nominal_range_upper=hi)
 records=[dict(candidate=i,source_index=int(k),time=float(times[i]),nominal_phase_rad=float(((float(times[i])-.15)*2*np.pi*5)%(2*np.pi)),footcontact_legs=int(b['foot_contact'][k].sum()),nonfoot_count=int(b['nonfoot_count'][k]),COMz=float(b['com'][k,2]),nominal_range_overshoot_max=float(max(lower[i].max(),upper[i].max())),reuse_prior_capacity=i==32) for i,k in enumerate(ix)]
 result=dict(passed=True,source_sha256=hashes,candidates=33,cached_final_candidate=32,new_candidate_poses=32,time_quantization_max_s=float(abs(times-desired).max()),records=records,source='Originalsame3470physicalassistedtrajectory followingrealderivedengineeringreference,not raw measuredanimalposes',calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0,linprog=0),limits='Savedvelocityretainedforprovenance; conditionalstaticcaptureuseszero velocity openly. Staticwitness doesnotestablishactualsoft-contact or neuralcontrol.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','records']},indent=2))
if __name__=='__main__':main()
