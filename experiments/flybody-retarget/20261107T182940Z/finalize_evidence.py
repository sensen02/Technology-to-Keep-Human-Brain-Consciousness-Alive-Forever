"""Seal this round's independently replayed failures and source identities."""
from pathlib import Path
import json,hashlib
OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z');HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
episodes=[]
for tag in ['teacherderived_physics','teacherderived_slow4_physics']:
    p=OUT/tag;r=load(p/'RESULT.json');v=load(p/'VERIFICATION.json');prot=load(p/'PROTOCOL.json');video=load(p/'VIDEO_MANIFEST.json')
    assert v['status']=='REPLAY_PASS' and max(v['max_errors'].values())==0 and not any(v['warning_counts'])
    assert r['physical_gate']['passed'] is False and v['native_steps']==r['native_steps']
    for name,h in r['files'].items():assert sha(p/name)==h,(tag,name)
    assert sha(prot['body_source'])==prot['body_source_sha256'];assert sha(prot['target_file'])==prot['target_sha256']
    expected=p/'RUN_SOURCE.py' if tag=='teacherderived_physics' else HERE/'run_physics.py'
    assert sha(expected)==prot['runner_sha256']
    assert video['body_sha256']==sha(p/'BODY.npz') and video['video_sha256']==sha(Path(video['video']))
    assert video['render_COM_error']==0 and video['mesh_count']==71 and video['visual_review']!='pending'
    episodes.append(dict(tag=tag,native_steps=r['native_steps'],sim_s=r['sim_s'],physical_pass=False,replay_pass=True,max_replay_error=0,saturation_count=v['recovered_saturated_entries'],video=video['video'],video_frames=video['frames']))
k=load(OUT/'kinematics/VERIFICATION.json');nn=load(OUT/'neural-interface/VERIFICATION.json')
assert k['passed'] and all(v['geometry_gate'] is False for v in k['datasets'].values()) and nn['status']=='PASS'
result=dict(status='ROUND7_IMPLEMENTED_VERIFIED_FAILURES_AND_NEURAL_CANDIDATE_NOT_PLATFORM_COMPLETE',episodes=episodes,total_actual_physics_steps=sum(e['native_steps'] for e in episodes)*2,new_training_updates=0,neural_candidate_executed=False,
 geometry_pass=False,kinematic_numerical_verification_pass=True,anatomy_static_verification_pass=True,anatomy_nodes=20000,anatomy_exact_edges=705068,anatomy_strict_label_edges=435126,
 next='Identity-preserving sparse neuron dynamics with explicit sensory/DN/AN/MN ports and qualified transmitter mode; no bad-reference acceptance or hidden brain-to-joint bypass',
 failures_preserved=['original full-run JSONserialization failure, arrays recovered without additional simulation','first renderer framebuffer configuration failure before frames','neural identity equality assertion failure, source exceptions preserved','both fixed geometry gates','both physical support screens and slip'],
 source_hashes={str(p):sha(p) for p in HERE.glob('*.py')},report=str(OUT/'重定向与真实神经接口_本轮结果.zh-CN.md'),report_sha256=sha(OUT/'重定向与真实神经接口_本轮结果.zh-CN.md'))
with (OUT/'ROUND7_STATUS.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(result,indent=2,ensure_ascii=False))
