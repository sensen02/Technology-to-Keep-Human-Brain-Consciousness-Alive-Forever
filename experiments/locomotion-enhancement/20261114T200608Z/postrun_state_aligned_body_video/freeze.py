"""Postrun FIRSTREADY. Only stdlib source/JSON/opaque hash, no actual qualification."""
from pathlib import Path
import sys
from contract import ROOT,BODY,CAL,BATCH,POSTRUN,BINDER,read,save,sha,require,verify_map
FILES=('contract.py','render.py','mp4_timing.py','deliver.py','check_codeonly.py','freeze.py','PREREG.json','README.zh-CN.md','CODECHECK.json','INDEPENDENT_REVIEW.json','SOURCE_SCHEMA.json')

def main():
 require(not (ROOT/'READY.json').exists(),'Immutable FIRSTREADY; correction requires new video scope')
 check=read(ROOT/'CODECHECK.json');review=read(ROOT/'INDEPENDENT_REVIEW.json');binder=read(BINDER/'CODECHECK.json')
 require(all(v is True for v in check['checks'].values()) and review['all_findings_resolved_before_FIRSTREADY'] is True,'STDchecks and independent readreview')
 require(binder['status']=='FIRSTREADY_FROZEN_EXTERNAL_POSTRUN_BINDING_ONLY_ACTUAL_POSTRUN_BATCH_NOT_QUALIFIED' and binder['all_passed'] is True and all(v is True for v in binder['checks'].values()),'Wait for actual external binder FIRSTREADY')
 verify_map(binder['source_hashes'])
 old=ROOT.parent/'state_aligned_body_video';actual=ROOT.parent.parent/'20261114T010835Z/target_feedback_body_video'
 for name in ('render.py','mp4_timing.py'):require(sha(ROOT/name)==sha(old/name)==sha(actual/name)==check['byteequiv_sha256'][name],'Exact actual116/statealigned renderer bytes')
 refs=[old/'READY.json',actual/'READY.json',actual/'render.py',actual/'mp4_timing.py',ROOT.parent/'FUNCTIONAL_MOTION_REPORT_PREREG.json',BINDER/'CODECHECK.json']
 refs += [BODY/n for n in ('PREREG.json','CODECHECK.json','adapter.py','prepare.py','common.py','runner.py','audit.py','run.py','preflight.py','static_check.py','OLD_SOURCE_SNAPSHOT.json')]
 refs += [BATCH/n for n in ('contracts.py','audit.py','staticcheck.py','FROZEN.json','PREREG.json')]
 refs += [POSTRUN/n for n in ('metadata.py','run.py','runtime.py','CODECHECK.json','PREREG.json')]
 refs += [BINDER/n for n in ('prepare.py','staticcheck.py','PREREG.json','README.zh-CN.md')]
 refs += [CAL/n for n in ('equivalence.py','PREREG.json','common.py','sources.py')]
 schema=dict(mode='POSTRUN_SAVED_CHECKPOINT_ONLY',original_pipeline='ActualterminalFAILED124 required, RUNNING/unknown/originalsuccess rejected, originalJOBS retained',modelCAL=str(CAL),BODY=str(BODY),POSTRUN=str(POSTRUN),BATCH=str(BATCH),BINDER=str(BINDER),model_format='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1',body_format='STATE_ALIGNED_ACTION_MOTOR_COMPONENT_BODY_V1',final_step=900,state_dims=[128,64,64,96],source3=True,strict_model_domain='fresh900 actualsource/data strictpayloadcfg; originalCALcommon/model',qualification_domain='postrun widerTRtoolsource/data/proof separatelybound',qualification='eca actual contracts.final_qualification/accept_run plus afb postrun_qualification/batch_qualification, no body-unexecutedqualification call',resource='postrun3600/RSS1800; independentpostrunbatch1800/RSS1800 completionallartifacts/noFAILURE',body='ExistingBODYJOBS9stage0/fourRESULTVERIFYsame900SHA/canonicalcompiled/allsourcehash/SUMMARY; FAIL/earlystop renderable',video_math='Exactbyteequal actual116+statealigned;0NN/Core/integration;1directMjSpec;71opaque42PD/compiledSHA/savedFK/PTS/fullDecode/trueearlystop',functional='Originaltaskoverlayunchanged;secondaryprereg linkonly, never substitute functionalPASS',actualqualification='UNKNOWN_NOT_CALLED',actual_video='NOT_RENDERED')
 save(ROOT/'SOURCE_SCHEMA.json',schema)
 ready=dict(status='READYSTD_FROZEN_CONDITIONAL_NO_ACTUAL_BODY_READ',FIRSTREADY=True,qualification_mode='POSTRUN_SAVED_CHECKPOINT_ONLY',checks=check['checks'],negative_mock_cases=check['negative_mock_cases'],byteequiv_sha256=check['byteequiv_sha256'],source_hashes={str(ROOT/n):sha(ROOT/n) for n in FILES},reference_source_hashes={str(p):sha(p) for p in refs},independent_review=str(ROOT/'INDEPENDENT_REVIEW.json'),body=str(BODY),CAL_original=str(CAL),postrun=str(POSTRUN),batch=str(BATCH),binder_CODECHECK_sha256=sha(BINDER/'CODECHECK.json'),actualpostrun_batch_body_video='UNKNOWN_NOT_READ_OR_EXECUTED',NN_forwards=0,Core_constructions=0,model_compiles=0,physics_steps=0,render_frames=0,PT_payloads_read=0,NPZ_payloads_read=0,futureactualreads=0,desktop_published=False,tracker_modified=False,immutable_after_FIRSTREADY=True)
 require(not any(k in sys.modules for k in ('torch','numpy','mujoco','PIL','physics','model','flygym')),'No heavy imports')
 save(ROOT/'READY.json',ready);print(str(ROOT/'READY.json'))
if __name__=='__main__':main()
