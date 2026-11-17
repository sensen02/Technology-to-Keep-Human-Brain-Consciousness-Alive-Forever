"""Post-run metadata-only diagnostic: stdlib JSON/math, no RGB/NPZ/model/physics."""
from pathlib import Path
import json,math,hashlib,sys
ROOT=Path(__file__).resolve().parent;LIB=ROOT.parent/'library/LIBRARY.json'
CASES=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right')
WINDOWS={'onset_and_hold_ANN1_6to2_6':(1.6,2.6),'pre_switch_ANN4_0to4_8':(4.,4.8),'post_switch_ANN4_82to5_82':(4.82,5.82),'late_ANN7_0to9_2':(7.,9.2)}
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def geometry(row,target_key):
 pos=[sum(eye['pos'][i] for eye in row['camera_metadata'])/2 for i in range(3)]
 rotation=row['actual_head_rotation'];ex=[rotation[0][0],rotation[1][0]];norm=math.hypot(*ex);ex=[v/norm for v in ex];ey=[-ex[1],ex[0]]
 target=row[target_key];delta=[target[i]-pos[i] for i in range(2)];forward=sum(a*b for a,b in zip(ex,delta));lateral=sum(a*b for a,b in zip(ey,delta))
 return dict(bearing_rad=math.atan2(lateral,forward),forward_depth_native=forward,lateral_native=lateral,planar_distance_native=math.hypot(*delta))
def range_stats(values):
 if not values:return None
 values=sorted(values);return dict(minimum=values[0],maximum=values[-1],mean=sum(values)/len(values),sample_count=len(values))
def aggregate(rows,support=None):
 if not rows:return dict(sample_count=0,geometry_stats=None)
 result=dict(sample_count=len(rows),geometry_stats={key:range_stats([r[key] for r in rows]) for key in ('bearing_rad','forward_depth_native','lateral_native','planar_distance_native')},target_behind_head_samples=sum(r['forward_depth_native']<=0 for r in rows))
 if support:
  lo,hi=support['bearing_rad']['minimum'],support['bearing_rad']['maximum'];dlo,dhi=support['forward_depth_native']['minimum'],support['forward_depth_native']['maximum']
  result.update(training_bearing_support_rad=[lo,hi],bearing_outside_training_samples=sum(not lo-1e-10<=r['bearing_rad']<=hi+1e-10 for r in rows),bearing_outside_training_fraction=sum(not lo-1e-10<=r['bearing_rad']<=hi+1e-10 for r in rows)/len(rows),forward_depth_outside_training_samples=sum(not dlo-1e-10<=r['forward_depth_native']<=dhi+1e-10 for r in rows),training_forward_depth_support_native=[dlo,dhi])
 return result

def main():
 library=read(LIB);inputs={str(LIB):sha(LIB)};training={};alltrain=[]
 for scene in ('center','left','right'):
  selected=[r for r in library['rows'] if r['split']=='train' and r['stimulus']==scene and r['environment_target_world_xyz'] is not None]
  rows=[geometry(r,'environment_target_world_xyz') for r in selected];training[scene]=aggregate(rows);alltrain.extend(selected)
 results=[]
 for name in CASES:
  path=ROOT/'trials'/name/'CAPTURES.json';inputs[str(path)]=sha(path);captures=read(path);scenes={}
  for scene in ('center','left','right'):
   selected=[]
   for cap in captures:
    if cap['stimulus']!=scene:continue
    g=geometry(cap,'target_center_xyz');g.update(capture_elapsed_ANN_s=cap['capture_time_s']-5.8,release_elapsed_ANN_s=cap['release_time_s']-5.8,capture_id=cap['capture_id']);selected.append(g)
   if not selected:continue
   support=training[scene]['geometry_stats'];overall=aggregate(selected,support);windows={}
   for key,(start,end) in WINDOWS.items():windows[key]=aggregate([r for r in selected if start-1e-10<=r['capture_elapsed_ANN_s']<end-1e-10],support)
   first=selected[0];last=selected[-1]
   scenes[scene]=dict(overall=overall,windows=windows,first_visible_scene_geometry=first,last_visible_scene_geometry=last,worldlabel_vs_current_head_sign_disagrees_samples=sum((r['bearing_rad']>0)!=(scene=='left') for r in selected) if scene in ('left','right') else None)
  results.append(dict(condition=name,capture_rows=len(captures),scenes=scenes))
 report=dict(kind='METADATA_ONLY_ACTUAL_HEAD_TARGET_DOMAIN_REVIEW_V1',scope='Posthoc description of first five completed actual body cases only; no causal mechanism proof or policy input changes',inputs=inputs,formula='eye-midpoint=mean(actual camera_metadata pos); ex=normalized horizontal actual_head_rotation[:,0], ey=[-ex_y,ex_x]; delta=worldtarget-eye-midpoint; bearing=atan2(dot(ey,delta),dot(ex,delta)), forwarddepth=dot(ex,delta)',window_basis='actual camera capture time minus5.8, [start,end) with1e-10 boundary tolerance; release timestamps separately reported. These are geometry diagnostics, not actor inputs or retimed neural gates.',training_support=training,training_visible_metadata_rows=len(alltrain),training_unique_config_pose_count=len({(r['config_id'],r['pose_id']) for r in alltrain}),actual_cases=results,RGB_arrays_loaded=0,NPZ_arrays_loaded=0,model_forward_calls=0,Core_constructions=0,render_calls=0,training_updates=0,physical_steps=0,limitations=['World target scene labels do not guarantee image visibility. Horizontal bearing alone does not prove target is out of a pitched binocular frustum or occluded.','Broad static min/max support is not learned support or a generalization certificate; no command arrays were loaded.','Temporal geometry crossing worldlabel/headrelative sign does not by itself prove an incorrect neural command; inspect actual images and recorded commands separately.','No new candidate, runtime rule, source edit or parameter change is performed.'])
 for name in ('numpy','torch','mujoco','physics','student_wave','flygym'):assert name not in sys.modules
 out=ROOT/'METADATA_DOMAIN_REVIEW.json'
 with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 lines=['# 已完成5回合的头部相对目标角度：仅JSON元数据诊断','','本次仅读训练库与5个已完成回合的JSON，用标准库math计算，未读RGB/NPZ、未运行模型、Core、渲染、训练或物理。当前父真实回合继续，本文不是完整功能验收。','','角度基于双眼位置中点与实际头部水平+x/+y，不是神经输入。时间窗按capture时刻相对5.8s交接，release时刻另记；不改变已有反应时序门。','','| 训练场景 | 行数 | bearing范围(rad) | forwarddepth范围 |','|---|---:|---|---|']
 for scene,t in training.items():
  b=t['geometry_stats']['bearing_rad'];d=t['geometry_stats']['forward_depth_native'];lines.append(f"| {scene} | {t['sample_count']} | {b['minimum']:.6f} .. {b['maximum']:.6f} | {d['minimum']:.6f} .. {d['maximum']:.6f} |")
 lines+=['','| 实际回合/场景 | 时段 | 图像采样数 | bearing范围(rad) | 超训练角度范围比例 | forwarddepth范围 |','|---|---|---:|---|---:|---|']
 for case in results:
  for scene,s in case['scenes'].items():
   for window,row in [('全部source可见标签时刻',s['overall']),*s['windows'].items()]:
    if not row['sample_count']:continue
    b=row['geometry_stats']['bearing_rad'];d=row['geometry_stats']['forward_depth_native'];lines.append(f"| {case['condition']}/{scene} | {window} | {row['sample_count']} | {b['minimum']:.6f} .. {b['maximum']:.6f} | {row['bearing_outside_training_fraction']:.3f} | {d['minimum']:.6f} .. {d['maximum']:.6f} |")
 lines+=['','这些是事后几何范围描述，不证明训练失败原因。场景标签left/right是世界固定地标身份，不保证目标在当前头部同侧或图像中可见。水平bearing超训练范围不能单独证明出FOV；需结合实际双眼图、俯仰、自遮挡和命令时序。没有读取神经输出NPZ，不能据此判命令正确/错误。','','原旧近域/远域失败与原滑移门保持，未开新候选、未改冻结来源或判据。完整数值、首末帧几何、输入SHA见：',str(out)]
 with (ROOT/'METADATA_DOMAIN_REVIEW.zh-CN.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print(json.dumps(dict(metadata_only=True,training_unique_config_pose_count=report['training_unique_config_pose_count'],cases=len(results),output=str(out),heavy_imports=False)))
if __name__=='__main__':main()
