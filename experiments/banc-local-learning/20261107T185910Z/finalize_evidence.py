"""Verify completed local-neural milestone artifacts; do not run dynamics."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;OUT=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
train=load(OUT/'readout/RESULT.json')
for name,h in train['files'].items():assert sha(OUT/'readout'/name)==h
for name,h in train['source_hashes'].items():assert sha(name)==h,name
chosen=load(HERE/'dynamics/CHOSEN_CONFIG.json');assert chosen['status']=='CALIBRATION_FAILED'
assert train['temporal_RMS_rad_s']>train['zero_baseline_temporal_RMS_rad_s']
for filename in ['VERIFY_READOUT.json','VERIFY_CLOSED_LOOP.json','VERIFY_OUTPUT_HOLD.json']:
    v=load(OUT/'verification'/filename);assert v['status'].startswith('PASS')
episodes=[]
for tag in ['closed_loop','output_hold']:
    p=OUT/tag;r=load(p/'RESULT.json');v=load(p/'VERIFICATION.json');video=load(p/'VIDEO_MANIFEST.json');protocol=load(p/'PROTOCOL.json')
    for name,h in r['files'].items():assert sha(p/name)==h,(tag,name)
    assert sha(protocol['body_source'])==protocol['body_source_sha256']
    assert sha(HERE/'run_closed_loop.py')==protocol['runner_sha256']
    assert sha(OUT/'readout/MODEL.npz')==protocol['readout_sha256']
    assert v['status']=='REPLAY_PASS' and max(v['max_errors'].values())==0 and not any(v['warning_counts'])
    assert r['physical_gate']['passed'] and r['warmup_qpos_max_error']==0
    assert video['mesh_count']==71 and video['render_COM_error']==0 and video['visual_review']!='pending'
    assert sha(p/'BODY.npz')==video['body_sha256'] and sha(video['video'])==video['video_sha256']
    episodes.append(dict(condition=r['condition'],physical_steps=r['native_steps'],neural_steps=r['neural_steps'],support_screen_pass=True,walking_pass=False,warmup_source_error=0,physical_replay_all_errors_zero=True,slip_ratios=r['slip_to_pathspeed_ratio'],neural_min_5ms_sample_mV=r['neural_voltage_range_mV'][0],video=video['video']))
result=dict(status='ROUND8_LOCAL_NEURAL_LOOP_EXECUTED_NOT_STABLE_WALKING_OR_PLATFORM_COMPLETE',episodes=episodes,source_neuron_identities=20000,active_strict_original_edges=435126,qualified_sensory_ports=556,uniqueleg_MN_outputs=323,
    physical_steps_including_replays=sum(x['physical_steps'] for x in episodes)*2,neural_steps_including_all_probes_and_replays=69762,readout_fit_solves=42,readout_parameters=2261,intrinsic_neural_parameter_training=False,
    temporal_prediction_improved=False,calibration_status=chosen['status'],support_neural_necessity_established=False,brain_implemented=False,feeding_implemented=False,injury_experiments=0,goal='active',
    next='Versioned adaptation-mechanism/parameter diagnostics and information/temporal-readout learning; retain failures and actual neural path. No external phase or hidden brain-to-joint shortcuts.',
    report=str(OUT/'局部神经闭环与学习_本轮结果.zh-CN.md'),report_sha256=sha(OUT/'局部神经闭环与学习_本轮结果.zh-CN.md'),source_hashes={str(p):sha(p) for p in HERE.glob('*.py')})
with (OUT/'ROUND8_STATUS.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(result,indent=2,ensure_ascii=False))
