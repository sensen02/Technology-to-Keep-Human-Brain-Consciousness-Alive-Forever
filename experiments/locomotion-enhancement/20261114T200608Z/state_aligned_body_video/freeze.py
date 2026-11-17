"""FIRSTREADY stdlib-only final source freeze, no real900 qualification."""
from pathlib import Path
import sys
from contract import ROOT,BODY,CAL,BATCH,read,save,sha,require
FILES=('contract.py','render.py','mp4_timing.py','deliver.py','check_codeonly.py','freeze.py','PREREG.json','README.zh-CN.md','CODECHECK.json','INDEPENDENT_REVIEW.json','SOURCE_SCHEMA.json')

def main():
 require(not (ROOT/'READY.json').exists(),'FIRSTREADY immutable; new render scope for correction')
 check=read(ROOT/'CODECHECK.json');review=read(ROOT/'INDEPENDENT_REVIEW.json')
 require(all(v is True for v in check['checks'].values()) and review['all_findings_resolved_before_FIRSTREADY'] is True,'Final STD/review required')
 old=ROOT.parent.parent/'20261114T010835Z/target_feedback_body_video'
 for name in ('render.py','mp4_timing.py'):require(sha(ROOT/name)==sha(old/name)==check['byteequiv_sha256'][name],'Actual116 byteequal source')
 refs=[old/'READY.json',old/'render.py',old/'mp4_timing.py',old/'contract.py',old/'deliver.py']
 refs += [BODY/n for n in ('PREREG.json','CODECHECK.json','adapter.py','prepare.py','common.py','runner.py','audit.py','run.py','preflight.py','static_check.py','OLD_SOURCE_SNAPSHOT.json')]
 refs += [BATCH/n for n in ('contracts.py','audit.py','staticcheck.py','FROZEN.json','PREREG.json')]
 refs += [CAL/n for n in ('equivalence.py','prepare.py','run.py','PREREG.json','sources.py','common.py')]
 schema=dict(body=str(BODY),producer=str(CAL),batch=str(BATCH),FORMAT='STATE_ALIGNED_ACTION_MOTOR_COMPONENT_BODY_V1',motor_format='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1',final_step=900,save_steps=[600,750,900],actualSHA='UNKNOWN_NOT_READ',cases=['walk','stop','left_pivot','right_pivot'],future_paths={c:str(BODY/'trials'/c) for c in ('walk','stop','left_pivot','right_pivot')},unchanged_trace_schema='1ms qpos49/qvel48/time/actualctrl42/com3/loads6/controlsource0,2,3;geometryafter1s;neuralstate352',audit='All new sameSHA VERIFY checks true inclNN352/source/native/compiled; no behavioralPASS admission requirement',producer_qualification='New STD contracts.final_qualification:3stages/save600750900/reuseactual96 originaltimingFAIL/resourcewall9000/currentequivalence/EVALindependent/artifactSHA',batch_qualification='RUN_actualSHA/contracts.accept_run:VERIFYallchecks/PASS+completion600s1800MiB/exactartifactSHA;12rows1920/72comparisons/conditional6+TF4 B1B12micro4 <=1e-12;SOURCE_LEDGER allSHA',body_qualification='JOBS9completeexit0+RESULT4/VERIFYsame900SHA+SUMMARY+source/proofhash',render_and_timing='Exactbyteequal actual116 source',runtime='1directMjSpec compile,0Core0NN0step,savedstateFKonly;71opaque/compiledSHA/fullframe/fullPTS/fullDecode',futureactualread=0)
 save(ROOT/'SOURCE_SCHEMA.json',schema)
 ready=dict(status='READYSTD_FROZEN_CONDITIONAL_NO_ACTUAL_BODY_READ',FIRSTREADY=True,checks=check['checks'],negative_mock_cases=check['negative_mock_cases'],independent_review=str(ROOT/'INDEPENDENT_REVIEW.json'),byteequiv_sha256=check['byteequiv_sha256'],source_hashes={str(ROOT/n):sha(ROOT/n) for n in FILES},reference_source_hashes={str(p):sha(p) for p in refs},body=str(BODY),producer=str(CAL),batch=str(BATCH),actual900_SHA='UNKNOWN_NOT_READ_REJECT_UNTIL_ACTUAL_COMPLETED',source_candidate=3,model_loads=0,NN_forwards=0,Core_constructions=0,model_compiles=0,physics_steps=0,render_frames=0,PT_payloads_read=0,NPZ_payloads_read=0,futureactualreads=0,desktop_published=False,tracker_modified=False,immutable_after_FIRSTREADY=True)
 require(not any(k in sys.modules for k in ('numpy','torch','mujoco','PIL','physics','flygym','model')),'No heavy prep imports')
 save(ROOT/'READY.json',ready);print(str(ROOT/'READY.json'))
if __name__=='__main__':main()
