"""Parent stdlib desktop copy only AFTER actual opening of all4montages and4graphs."""
from pathlib import Path
import argparse,shutil
from contract import ROOT,CASES,read,save,sha,require,verify_map
DESKTOP=Path('/home/sensen/Desktop/engine修复验收/运动增强/20261114T010835Z/目标反馈修复-纯运动组件身体验收')
LABELS={'walk':'直行','stop':'停止','left_pivot':'左转枢转诊断','right_pivot':'右转枢转诊断'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-reviewed-all4-montages-and4-graphs',action='store_true');a=p.parse_args()
 require(a.parent_reviewed_all4_montages_and4_graphs,'Parent must actually open all4montages/all4graphs first')
 out=ROOT/'OUTPUT';d=read(out/'DELIVERY_PENDING_REVIEW.json');require(d['model_compiles']==1 and d['Core_constructions']==d['NN_forwards']==d['physics_steps']==0,'Read-only reconstruction evidence')
 verify_map(d['source_hashes']);require(not (out/'DELIVERY.json').exists(),'No duplicate publication')
 DESKTOP.mkdir(parents=True,exist_ok=True);copies=[];lines=['# 目标反馈修复：纯运动组件同次保存状态视频','',f'共同实际模型 SHA256：{d["checkpoint_sha256"]}','', '此交付是source3常量[speed,yaw]运动组件诊断。包含实际稳定、算法教师预热和撤辅后纯运动ANN阶段；不是人工脑—真实颈部部署，不是成品新脑或真实颈部枢转验收。','', '| 任务 | 实际总长s | ANN窗口s | 实际任务门 | 原五门 | 早停 |','|---|---:|---:|---|---|---|']
 for row in d['cases']:
  condition=row['condition'];folder=out/condition
  require(row['checkpoint_sha256']==d['checkpoint_sha256'] and row['decoded_frames']==row['frames'] and row['ffmpeg_exit']==row['ffprobe_exit']==row['full_decode_exit']==0,'All four verified same model')
  for key,suffix in [('video','.mp4'),('montage','-蒙太奇.png'),('graph','-同步保存指标.png')]:
   source=Path(row[key]);require(sha(source)==row[key+'_sha256'],'Actual artifact SHA');dest=DESKTOP/(LABELS[condition]+'-同次纯运动组件'+suffix);require(not dest.exists(),'No historical overwrite: '+str(dest));shutil.copyfile(source,dest);require(sha(dest)==sha(source),'Desktop SHA');copies.append(dict(source=str(source),desktop=str(dest),sha256=sha(dest)))
  metric=row['candidate_metrics'];task='N/A（未接管）' if metric is None else str(metric['task_pass_candidate']);legacy=str(row['original_five']['all_gates'])
  lines.append(f'| {LABELS[condition]} | {row["physical_seconds"]:.3f} | {row["ANN_seconds"]:.3f} | {task} | {legacy} | {row["early_stop"] is not None} |')
 lines+=['','失败和旧五门结果均原样保留。早停只展示实际保存窗口，无延长至9.6s。视频按50fps主要节拍取各20ms区间末端保存状态；末尾不足20ms时最后帧时长截短，末个实际qpos已展示，ffprobe逐帧PTS和电影终点与物理总长一致。','', '全71mesh不透明、逐帧全身轮廓在画面内，保存qpos/qvel/ctrl/COM/yaw身份核验。只重建一个MjSpec模型、只mj_forward，无Core、无神经前向、无积分。重建接触力不用于指标。同步图和文字只来自同次RESULT/trace/geometry及保存native峰值。','', '父已实际打开四张解码蒙太奇和四张同步图；四视频全解码、ffprobe、源hash和桌面hash验证完成。视频不构成额外行为通过判断。旧789f/f25b及历史科学证据未修改或重跑。','', '证据：'+str(out/'DELIVERY.json')]
 readme=DESKTOP/'README.zh-CN.md';require(not readme.exists(),'No README overwrite');readme.write_text('\n'.join(lines)+'\n')
 save(out/'DELIVERY.json',dict(status='DELIVERED_SAVED_ACTUAL_COMPONENT_WINDOWS',checkpoint_sha256=d['checkpoint_sha256'],cases=d['cases'],copies=copies,README=str(readme),README_sha256=sha(readme),parent_opened_all4montages_and4graphs=True,task_acceptance='Original same-run gates retained, video does not declare pass',NN_forwards=0,Core_constructions=0,physics_steps=0))
 print(str(out/'DELIVERY.json'))
if __name__=='__main__':main()
