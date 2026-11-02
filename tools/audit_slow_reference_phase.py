"""Analytical reference clock matching; no gait generation or simulation."""
from pathlib import Path
import json,hashlib,math
import numpy as np
R=Path(__file__).resolve().parents[1];O=Path('/home/sensen/Desktop/engine修复验收/s5-slow-phase-contract/20261102T165227Z')
def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),R/'run_physical_walk_prewarm.py',R/'PLASTICITY_HYPOTHESIS.zh-CN.md'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 result=dict(scope='Prospective analytical phasecontract only; not modelrun or physicalsuccess',source_sha256=hashes,baseline=dict(frequency_hz=5.,withdrawal_s=1.3,neural_start_s=.2,neural_prewarm_s=1.1,last_reference_cycle=[1.1,1.3]),candidate=dict(frequency_hz=2.5,withdrawal_s=1.25,neural_start_s=.2,neural_prewarm_s=1.05,last_reference_cycle=[.85,1.25]),phase_anchor_s=.15,release_s=.1,interpretation='Sameinitialphaseandterminalphase,halfspeedreference; prewarmdurationdiffers50ms so not identicalneuralhistory. No phase/gainsearch, no STDP, no newanatomy. Countsactualbodyvsnominalsourceperiod separately.',calls=dict(model_compile=0,body_step=0,neural_step=0,optimizer=0))
 phase=lambda t,f:((t-.15)*2*math.pi*f)%(2*math.pi)
 assert abs(phase(1.3,5)-phase(1.25,2.5))<1e-13
 theta=np.linspace(0,2*math.pi,101);old=1.1+theta/(2*math.pi*5);new=.85+theta/(2*math.pi*2.5)
 assert np.max(abs(np.unwrap((old-.15)*2*math.pi*5)-np.unwrap((new-.15)*2*math.pi*2.5)-6*math.pi))<1e-12
 np.savez_compressed(O/'MATCHED_PHASE_CLOCKS.npz',relative_phase=theta,baseline_body_time=old,candidate_body_time=new,baseline_phase=np.mod((old-.15)*2*math.pi*5,2*math.pi),candidate_phase=np.mod((new-.15)*2*math.pi*2.5,2*math.pi))
 result['terminal_phase_rad']=phase(1.3,5);result['matched_samples']=101;result['nominal_body_steps']=108000;result['nominal_neural_ticks']=2300
 (O/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
