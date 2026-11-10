"""Verify completed fixed physical-learning study without executing new trials."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z');LEARN=OUT/'learning'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
study=load(LEARN/'STUDY_RESULT.json');comparison=load(OUT/'PHYSICAL_COMPARISON.json');approval=load(HERE/'learning/START_AUTHORIZATION.json')
for path,h in approval['reviewed_sha256'].items():assert sha(path)==h,path
assert study['training_trials']==53 and study['validation_trials']==2 and len(study['generations'])==4
records=list((LEARN/'training').glob('*/RESULT.json'))+list((LEARN/'evaluation').glob('*/RESULT.json'));assert len(records)==55
physical=0;neural=0
for path in records:
    r=load(path);physical+=r['native_steps'];neural+=r['neural_steps'];assert r['valid'] and not r.get('software_exception',False)
    for name,h in r['files'].items():assert sha(path.parent/name)==h,(path,name)
assert physical==study['actual_calls']['actual_native_steps'] and neural==study['actual_calls']['actual_neural_steps']
for i in range(4):
    det=load(LEARN/f'generation_{i}/INCUMBENT_DETERMINISM.json');assert det['max_abs_error']<=1e-9
finals=[]
for name in ['selected','zero']:
    p=LEARN/'evaluation'/name;r=load(p/'RESULT.json');v=load(p/'VERIFICATION.json');video=load(LEARN/'evaluation/selected/VIDEO_MANIFEST.json')
    if name=='zero':
        reference=load(p/'VIDEO_REFERENCE.json');assert reference['shared_BODY_sha256']==sha(p/'BODY.npz') and reference['video_sha256']==video['video_sha256']
    assert v['status']=='REPLAY_PASS' and max(v['max_errors'].values())==0 and not any(v['warning_counts'])
    assert video['mesh_count']==71 and video['render_COM_error']<1e-9 and max(video['seam_max_errors'].values())<1e-9 and video['visual_review']!='pending'
    assert sha(video['video'])==video['video_sha256'] and sha(p/'BODY.npz')==video['release_BODY_sha256']
    finals.append(dict(condition=name,reward=r['reward'],support_pass=r['physical_gate']['passed'],native_steps=r['native_steps'],neural_steps=r['neural_steps'],physics_replay_max_error=max(v['max_errors'].values()),motion=comparison[name],video=video['video']))
verification=load(OUT/'verification/VERIFICATION.json')
assert verification['status']=='PASS_INDEPENDENT_SELECTED_ZERO_REPLAY' and verification['completed_neural_steps']==20000
assert load(LEARN/'INDEPENDENT_STUDY_VERIFICATION.json')['status']=='PASS_INDEPENDENT_STUDY_RECONSTRUCTION_NOT_WALKING'
report=OUT/'物理反馈训练_本轮结果.zh-CN.md'
result=dict(status='ROUND10_DIRECT_PHYSICAL_LEARNING_EXECUTED_PLATFORM_INCOMPLETE',training_trials=53,validation_trials=2,optimizer_generations=4,selected_training_episode=study['selected_training_episode'],baseline_training_reward=study['baseline_training_reward'],selected_training_reward=study['selected_training_reward'],study_native_steps=physical,study_neural_steps=neural,physical_replay_steps=sum(x['native_steps'] for x in finals),foundation_completed_neural_steps=4060,neural_verification_steps=verification['completed_neural_steps'],all_completed_neural_steps=neural+4060+verification['completed_neural_steps'],all_native_physical_steps=physical+sum(x['native_steps'] for x in finals),selected_is_unchanged_zero=True,walking_pass=False,finals=finals,biological_calibration='NOT_ESTABLISHED',complete_brain=False,feeding=False,injury_experiments=0,goal='active',report=str(report),report_sha256=sha(report),source_hashes={str(p):sha(p) for p in HERE.glob('*.py')})
with (OUT/'ROUND10_STATUS.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps({k:result[k] for k in ['status','selected_training_episode','baseline_training_reward','selected_training_reward','study_native_steps','study_neural_steps','physical_replay_steps']},indent=2))
