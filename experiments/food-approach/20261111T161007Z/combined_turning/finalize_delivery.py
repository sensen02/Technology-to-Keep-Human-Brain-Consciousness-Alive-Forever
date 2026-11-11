"""Parent final conjunction and desktop copies; no neural/model/physics imports."""
from pathlib import Path
import json,hashlib,math,subprocess
HERE=Path(__file__).resolve().parent
DESKTOP=Path('/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110')
EXPECTED='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save_new(p,value):
    with Path(p).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
assert sha(HERE/'COMBINED.pt')==EXPECTED
prep=load(HERE/'PREPARATION.json');assert prep['all_passed'] and all(prep['checks'].values())
rows=[];checked_sources={};copies=[]
for condition in ('neutral','left','right'):
    run=HERE/('EVAL_'+condition);r=load(run/'RESULT.json');a=load(HERE/('VERIFY_'+condition+'.json'))
    assert r['checkpoint_sha256']==a['checkpoint_sha256']==EXPECTED
    assert a['integrity_all_pass'] and all(a['checks'].values()) and r['engineering_pass']==a['engineering_pass']
    assert r['state_layout']==[128,64,64] and r['transition']['all_exact'] and r['transition']['no_reset']
    for k in ('full2880_threehidden_decisions','full256_hidden_chain','actual2880_checkpoint_threehidden_exact'):assert a['checks'][k]
    assert r['physical_steps']==1232000 and r['native']['post_native_steps']==768000 and r['native']['post_nonfoot_peak_native']==0 and r['native']['extra_force_max']==0
    assert r['autonomous_cycles']['all_legs_completed']
    for k,n in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:assert sha(run/n)==r[k]['sha256']
    checked_sources.update(r['source_hashes'])
    rows.append(dict(condition=condition,checkpoint_sha256=EXPECTED,actual_yaw_deg=math.degrees(r['yaw_change_rad']),slip=r['autonomous']['slip_ratio_max'],original_five=r['autonomous']['gates'],criteria=r['criteria'],engineering_pass=r['engineering_pass'],cycles=r['autonomous_cycles']['complete_per_leg'],native_nonfoot_peak=0.,extra_force_max=0.,audit_checks=len(a['checks']),result=str(run/'RESULT.json'),result_sha256=sha(run/'RESULT.json'),audit=str(HERE/('VERIFY_'+condition+'.json')),audit_sha256=sha(HERE/('VERIFY_'+condition+'.json'))))
    if condition=='neutral':continue
    out=HERE/'visual'/('VIDEO_'+condition);m=load(out/'VIDEO.json')
    assert m['checkpoint_sha256']==EXPECTED and m['model_format']=='COMBINED_TURN_V1'
    assert m['frames']==m['decoded_frames']==770 and m['fps']==50 and m['physical_seconds']==m['playback_seconds']==15.4
    assert m['physics_steps']==m['integration_attempts']==0 and m['all_frame_fullbody_containment'] and m['all_mesh_alpha1'] and m['mesh_count']==71
    assert m['all_qpos_qvel_exact'] and m['all_heading_com_exact'] and m['source_unchanged'] and m['compiled_runtime_unchanged']
    assert m['ffmpeg_exit']==m['ffprobe_exit']==m['full_decode_exit']==0
    for path,digest in m['source_hashes'].items():assert sha(path)==digest,path
    direction='左转' if condition=='left' else '右转';stem='大角度'+direction+'-同一神经模型-待你判定'
    for key,suffix in [('video','.mp4'),('montage','-蒙太奇.png')]:
        src=Path(m[key]);assert sha(src)==m[key+'_sha256'];dst=DESKTOP/(stem+suffix)
        with src.open('rb') as fi,dst.open('xb') as fo:
            while chunk:=fi.read(1024*1024):fo.write(chunk)
        assert sha(dst)==sha(src)
        copies.append(dict(source=str(src),desktop=str(dst),sha256=sha(dst),bytes=dst.stat().st_size))
    proc=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames:format=duration','-of','json',m['video']],capture_output=True,text=True,check=True)
    meta=json.loads(proc.stdout);s=meta['streams'][0];assert (s['width'],s['height'],s['r_frame_rate'],int(s['nb_frames']))==(960,640,'50/1',770) and float(meta['format']['duration'])==15.4
for path,digest in checked_sources.items():assert sha(path)==digest,path
report=dict(status='VERIFIED_SAME_MODEL_LARGE_TURN_SAMPLES_DELIVERED_USER_REVIEW_PENDING',checkpoint=str(HERE/'COMBINED.pt'),checkpoint_sha256=EXPECTED,state_dims=[128,64,64],state_total=256,shared_original_walking_base_preserved=True,all_three_actual_integrity_pass=True,all_three_engineering_qualified=all(r['engineering_pass'] for r in rows),formal_qualification=False,user_visual_acceptance='PENDING',rows=rows,copies=copies,video_verification=dict(frames_each=770,decoded_each=770,fps=50,physical_play_seconds=15.4,settle_seconds=1.,teacher_seconds=4.8,ANN_seconds=9.6,mesh_count=71,all_frames_fullbody=True,render_integration=0,parent_opened_both_montages=True,source_pose_COM_yaw_runtime_unchanged=True),scope='One nominal complete actual run per neutral/left/right. Replacement ANN, declared complete derived body and bounded PD. Both neural branches trained separately and composed, not naturally jointly trained. RIGHT posthoc visual component choice disclosed in AMENDMENT04. Strict engineering angle/slip FAIL retained. No food, biological, original connectome, continuous command range or full native D1-D8 qualification.',interruption='Worker turn failed after all physics/audits collected; parent independently verified final conjunction and delivered artifacts; no physical rerun.')
save_new(HERE/'FINAL_SUMMARY.json',report)
print(json.dumps(dict(summary=str(HERE/'FINAL_SUMMARY.json'),same_checkpoint=EXPECTED,engineering_qualified=False,copies=copies),ensure_ascii=False,indent=2))
