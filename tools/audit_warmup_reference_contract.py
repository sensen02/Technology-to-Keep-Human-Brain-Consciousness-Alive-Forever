"""Read-only accounting of real-derived reference vs engineering warmup transformations."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN')
S=Path('/home/sensen/Desktop/engine修复验收/s5-standard-gait-reference/20261028T040000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s1-warmup-reference-contract/20261102T155412Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 paths=[Path(__file__),R/'run_physical_walk_prewarm.py',R/'prepare_standard_gait_reference.py',S/'REFERENCE_METADATA.json',S/'STANDARD_GAIT_FK.npz',B/'GAIT_RETARGET.json',B/'MODEL_CHANGES.json',B/'STANCE_SUPPORT_RETARGET.json'];hashes={str(p):sha(p) for p in paths}
 dump(O/'PREREGISTRATION.json',dict(scope='Source/transform identity audit,not physicalsimulation or reconstruction ofnewanimal data',source_sha256=hashes,question='Whichparts ofwarmup are measured-derived andwhich are explicitly engineered? Keep allengineeredamplitudes/phases/referencefrequency separatefrom measuredanimaltruth.',calls=dict(model_compile=0,body_step=0,neural_step=0,optimizer=0)))
 meta=json.loads((S/'REFERENCE_METADATA.json').read_text());gait=json.loads((B/'GAIT_RETARGET.json').read_text());model=json.loads((B/'MODEL_CHANGES.json').read_text());stance=json.loads((B/'STANCE_SUPPORT_RETARGET.json').read_text())
 with np.load(S/'STANDARD_GAIT_FK.npz',allow_pickle=False) as z:phase=z['phase'];foot=z['foot_body'];hips=z['hip_body']
 assert len(phase)==33 and np.array_equal(foot[0],foot[-1]) and foot.shape==(33,6,3)
 result=dict(passed=True,source_sha256=hashes,real_derived_source=dict(description=meta['source'],source_original_processed_samples=meta['samples_per_step'],sample_dt_s=meta['sample_dt_s'],processed_cycle_s=meta['processed_cycle_duration_s'],processed_frequency_hz=1/meta['processed_cycle_duration_s'],raw_continuous_sixleg=False,points=meta['foot_definition'],absolute_source_units=meta['source_units_SI']),engineering_transform=dict(actual_reference_frequency_hz=5.,actual_cycle_s=.2,x_amplitude_native=.30,y_amplitude_native=.06,z_amplitude_native=.45,per_leg_peak_alignment=True,LF_z_profile_reused_for_alllegs=True,explicit_swing_profile_replaced=True,declared_tripod_offsets_rad=[0.,float(np.pi),0.,float(np.pi),0.,float(np.pi)],initial_stance_middle_x_extension_native=.35,initial_stance_hind_x_extension_native=.85,actual_gait_retarget_summary=gait,stance_retarget=stance,model_changes=model),interpretation='Physicalwarmup is real-derived but heavily engineered,not a direct synchronized replay ofan observed flywalkingbout. Sourceprocessedcycle and5Hz appliedcycle are different. Successfulassistedtracking doesnot validate muscleability or neurohandover. No alternativetrajectory selected, no qpos or controls written.',calls=dict(model_compile=0,body_step=0,neural_step=0,optimizer=0))
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','engineering_transform']},indent=2))
if __name__=='__main__':main()
