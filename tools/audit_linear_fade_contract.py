"""Offline declared fade multiplier/phase contract; not a controller implementation."""
from pathlib import Path
import json,hashlib,ast,math
import numpy as np
R=Path(__file__).resolve().parents[1]
O=Path('/home/sensen/Desktop/engine修复验收/s5-fade-contract-audit/20261102T074033Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),R/'run_physical_walk_prewarm.py',R/'PLASTICITY_HYPOTHESIS.zh-CN.md'];hashes={str(p):sha(p) for p in files}
 spec=dict(scope='Purearray contract of one declared linearfade experiment, no model/network or newcontroller',source_sha256=hashes,body_dt_s=.0000125,fade_start_s=1.1,fade_end_s=1.3,release_end_s=1.4,weight_updates=False,training=False,meaning='ExistingjointPDwarmup assist multiplied bydeclaredremain duringlast.2s; allfadeisassisted,strictautonomousmetricsonlyafter1.3s. This audit neverwritesactions or executesphysics.')
 (O/'PREREGISTRATION.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
 # Use integer tick clock for analytical schedule. Actual run arithmetic may differ <1e-10.
 steps=np.arange(88000,112001);times=steps*.0000125;multiplier=np.clip((1.3-times)/.2,0,1);assisted=steps<104000
 assert steps[0]==88000 and steps[-1]==112000 and abs(multiplier[0]-1)<1e-14 and abs(multiplier[16000])<1e-14 and not multiplier[16000:].any()
 assert np.all(np.diff(multiplier)<=1e-14)
 np.savez_compressed(O/'DECLARED_PHASE_CONTRACT.npz',body_step=steps,time_s=times,assist_multiplier=multiplier,assisted_phase=assisted,autonomous_phase=~assisted)
 source=(R/'run_physical_walk_prewarm.py').read_text();tree=ast.parse(source)
 assert 'dd.qfrc_applied[va]*=float(np.clip(remain,0.,1.))' in source
 assert 'remain=(duration+warmup_settle_s+warmup_hold_s-dd.time)/assistance_fade_s' in source
 result=dict(passed=True,source_sha256=hashes,fade_tick_intervals=16000,strict_release_tick_intervals=8000,fade_all_assisted=True,actual_existing_source_multiplies_generalized_torque=True,default_source_modified=False,STDP_or_other_plasticity_enabled=False,mechanical_support_established=False,calls=dict(model_compile=0,body_step=0,network=0,neural_step=0,training=0),limits='Scheduleidentity only,not evidence actualforces follow it. Candidatebodymust log actualassistanttorques and phaseevents; fadingtime cannot countasautonomoussuccess.')
 assert all(sha(Path(f))==h for f,h in hashes.items());(O/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
