from pathlib import Path
import json,hashlib,shutil,math
HERE=Path(__file__).resolve().parent
DESK=Path('/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
rows=[];copies=[]
for condition,label in [('left','待你判定-左转-实际最佳神经轨迹'),('right','待你判定-右转-同一神经模型实际轨迹')]:
 out=HERE/('VIDEO_'+condition);v=load(out/'VIDEO.json');r=load(Path(v['source_run'])/'RESULT.json')
 assert v['frames']==v['decoded_frames']==770 and v['physics_steps']==v['integration_attempts']==0
 assert v['ffmpeg_exit']==v['ffprobe_exit']==v['full_decode_exit']==0
 assert v['all_frame_fullbody_containment'] and v['all_qpos_qvel_exact'] and v['all_heading_com_exact'] and v['source_unchanged'] and v['compiled_runtime_unchanged']
 for p,h in v['source_hashes'].items():assert sha(p)==h
 for p,h in r['source_hashes'].items():assert sha(p)==h
 for source,name,expected in [(Path(v['video']),label+'.mp4',v['video_sha256']),(Path(v['montage']),label+'-蒙太奇.png',v['montage_sha256'])]:
  dest=DESK/name;assert not dest.exists();assert sha(source)==expected;shutil.copyfile(source,dest);assert sha(dest)==expected
  copies.append(dict(source=str(source),desktop=str(dest),sha256=expected,bytes=dest.stat().st_size))
 rows.append(dict(condition=condition,run=v['source_run'],checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],slip=r['autonomous']['slip_ratio_max'],yaw_rad=r['yaw_change_rad'],yaw_deg=math.degrees(r['yaw_change_rad']),forward_per_s=r['autonomous']['forward_signed_native_per_s'],nonfoot=r['autonomous']['nonfoot_force_max_native'],gates=r['autonomous']['gates'],engineering_pass=r['engineering_pass'],cycles=r['autonomous_cycles']['complete_per_leg'],video_manifest=str(out/'VIDEO.json'),manifest_sha256=sha(out/'VIDEO.json'),montage_actually_opened=True))
assert rows[0]['checkpoint_sha256']==rows[1]['checkpoint_sha256']=='6a6c23feb9a4b36fcf761f62f9d63309b5c81763f950afe520f42c09a5855e2f'
right=load(HERE/'EVAL_right/RESULT.json');audit=load(HERE/'VERIFY_right.json')
assert right['physical_steps']==1232000 and right['right_parent_preservation']['all_exact'] and audit['integrity_all_pass'] and all(audit['checks'].values())
readme=DESK/'待你判定-左右实际视频说明.zh-CN.md';assert not readme.exists();shutil.copyfile(HERE/'说明.zh-CN.md',readme);assert sha(readme)==sha(HERE/'说明.zh-CN.md')
copies.append(dict(source=str(HERE/'说明.zh-CN.md'),desktop=str(readme),sha256=sha(readme),bytes=readme.stat().st_size))
result=dict(status='DELIVERED_FOR_USER_VISUAL_JUDGMENT',scope='S2/S5 actual saved physical video; no training/optimization; original left FAIL retained',rows=rows,copies=copies,video=dict(width=960,height=640,fps=50,frames_each=770,decoded_frames_each=770,physical_seconds=15.4,playback_seconds=15.4,settle_seconds=1.,teacher_seconds=4.8,ANN_seconds=9.6,mesh_count=71,mesh_alpha=1,all_frames_fullbody=True,all_qpos_qvel_COM_yaw_verified=True,source_hashes_unchanged=True,render_physics_steps=0,trajectory_corrections=0),jobs=[dict(id='bash-1',task='one full same-model right physical plus independent audit',exit=0),dict(id='bash-2',task='first render failed before frame0 due framebuffer',exit=1,record=str(HERE/'RENDER_ATTEMPT1_FAILURE.json')),dict(id='bash-3',task='left render+ffprobe+full decode',exit=0),dict(id='bash-4',task='right render+ffprobe+full decode',exit=0)],right=dict(actual_physical_steps=1232000,audit_checks=len(audit['checks']),all_audit_checks_pass=True,ancestor_right_all_saved_arrays_exact=True,new_actual_same_checkpoint_not_ancestor_video=True),resources=dict(max_physics_or_renderer=1,physical_worker_threads=1,observed_physical_RSS_kB=761812,ffmpeg_encoder_threads=1,streaming=True),cleanup='Only empty failed staging removed; no useless temporary products; failures recorded; historical evidence untouched',tracker='parent responsibility; not edited by worker',limitations=['left original slip threshold FAIL remains','substitute ANN / derived full body / finite PD','full native high-frequency D1-D8 not performed in this delivery','not food success, connectome or biology certification','visual acceptance belongs to user'],record_hashes={str(HERE/n):sha(HERE/n) for n in ['PREREG.json','right_actual.py','render.py','finalize.py','VERIFY_right.json','CHECKLIST.zh-CN.md','说明.zh-CN.md','RENDER_ATTEMPT1_FAILURE.json']})
assert not (HERE/'DELIVERY.json').exists()
(HERE/'DELIVERY.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(delivery=str(HERE/'DELIVERY.json'),copies=copies,metrics=rows),ensure_ascii=False,indent=2))
