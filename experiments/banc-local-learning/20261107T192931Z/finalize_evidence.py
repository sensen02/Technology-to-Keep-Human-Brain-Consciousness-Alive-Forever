"""Seal Round9 evidence and distinguish mechanism checks from behavior failure."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z');OLD=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/output_hold')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
p=OUT/'closed_loop';r=load(p/'RESULT.json');v=load(p/'VERIFICATION.json');protocol=load(p/'PROTOCOL.json');video=load(p/'VIDEO_MANIFEST.json');comp=load(OUT/'learning/COMPARISON.json')
assert not r['physical_gate']['passed'] and r['neural_voltage_range_mV'][0]>=-90
assert v['status']=='REPLAY_PASS' and max(v['max_errors'].values())==0
assert not any(v['warning_counts']) and r['warmup_qpos_max_error']==0
for name,h in r['files'].items():assert sha(p/name)==h,name
for name in ['INDEPENDENT_VERIFICATION.json','PILOT_VERIFICATION.json']:assert load(OUT/'learning'/name)['status'].startswith('PASS')
assert comp['primary_fixed_before_results']=='newhistory' and not any(x['temporal_better_than_zero'] for x in comp['cases'].values())
assert sha(protocol['readout_path'])==protocol['readout_sha256']
assert sha(HERE/'learning/run_closed_loop.py')==protocol['runner_sha256']
assert sha(HERE/'dynamics/module.py')==protocol['network_sha256']
assert sha(HERE/'learning/runtime.py')==protocol['runtime_sha256']
assert sha(HERE/'dynamics/SOURCE_TRACE.npz')=='8a73c8a221b0f752e7c53f9e129fcf9747c42dc3ead6209a7ce912f430e70847'
assert sha(HERE/'learning/cases/newhistory/MODEL.npz')=='0474b070a8576229a2878f890225f4b081684ca85cef9e6dca9beb8c240118c6'
assert sha(p/'MODEL.mjb')==sha(OLD/'MODEL.mjb')
with np.load(p/'INITIAL.npz') as new,np.load(OLD/'INITIAL.npz') as old:
    assert new.files==old.files
    for key in new.files:assert np.array_equal(new[key],old[key]),key
with np.load(p/'COMMANDS.npz') as new,np.load(OLD/'COMMANDS.npz') as old:
    assert np.array_equal(new['rows'][:1000],old['rows'][:1000]) and not old['rows'][1000:,1:43].any()
assert sha(p/'BODY.npz')==video['body_sha256'] and sha(video['video'])==video['video_sha256']
assert video['frames']==360 and video['mesh_count']==71 and video['render_COM_error']==0 and video['visual_review']!='pending'
report=OUT/'适应电导与历史读出_本轮结果.zh-CN.md'
result=dict(status='ROUND9_MECHANISM_CHECKS_PASS_LEARNING_AND_WALKING_FAIL_PLATFORM_INCOMPLETE',mechanism_voltage_check_pass=True,biological_calibration='NOT_ESTABLISHED',walking_pass=False,physical_support_pass=False,temporal_improved_cases=[],primary_case='newhistory',cases=comp['cases'],physical_native_steps_including_replay=480000,completed_neural_steps=33212,rejected_net_calls_not_steps=24,new_fit_solves=126,physical_replay_all_max_errors_zero=True,historical_hold_model_bytes_and_initial_arrays_and_warm_commands_identical=True,historical_hold_rerun=False,video=video,report=str(report),report_sha256=sha(report),goal='active',injury_experiments=0,brain_implemented=False,feeding_implemented=False,source_hashes={str(x):sha(x) for x in [HERE/'audit_named_dn.py',HERE/'render_pilot.py',HERE/'finalize_evidence.py',HERE/'dynamics/module.py',HERE/'learning/runtime.py',HERE/'learning/run_closed_loop.py']})
with (OUT/'ROUND9_STATUS.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps({k:result[k] for k in ['status','mechanism_voltage_check_pass','walking_pass','physical_support_pass','completed_neural_steps','physical_native_steps_including_replay','historical_hold_model_bytes_and_initial_arrays_and_warm_commands_identical']},indent=2))
