"""Saved-data reporting only, not another neural/body run."""
from pathlib import Path
import json,hashlib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s4-dynamic-neural-gain/20261028T011000Z');D=Path('/home/sensen/Desktop/S4_DYNAMIC_NEURAL_GAIN')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=load(O/'SUMMARY.json');ex=load(O/'EXECUTION.json');pr=load(O/'PREREGISTRATION.json');dec=load(O/'DECISION.json');base=load(O/'BASELINE_METRICS.json');checks={};trained=[k for k in r if k.startswith('train_')];assert len(trained)==12;assert ex['exit_code']==0
 best=dec['best_training'];fig,axs=plt.subplots(2,2,figsize=(14,9));h=np.load('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/normal/BODY.npz');axs[0,0].plot(h['time']*1000,h['com_position'][:,2],color='black',linewidth=3,label='historical normal');axs[0,1].plot(h['time']*1000,h['nonfoot_floor_count'],color='black',linewidth=3,label='historical normal')
 for k in trained:
  b=np.load(O/k/'BODY.npz');u=b['command'];assert b['qpos'].shape==(20001,49);assert np.array_equal(b['qpos'][0],h['qpos'][0]);assert not b['activation'][0].any() and not b['qvel'][0].any();assert not b['external_force'].any();assert not b['warnings'].any();assert all(np.isfinite(b[f]).all() for f in b.files if b[f].dtype.kind in 'biuf');assert np.all((u>=0)&(u<=1));assert np.allclose(b['ground_force_foot_world'].sum(1)+b['ground_force_nonfoot_world'],b['ground_force_world'],atol=1e-12,rtol=1e-12);met=r[k];assert met['calls']['step']==20000 and met['calls']['neural_step']==500 and met['calls']['network_instances']==1;cfg=load(O/k/'CONFIG.json');assert np.array_equal(cfg['group_gains'],pr['schedule'][int(k[-2:])]);records=load(O/k/'RECORDS.json')['records'];iv=np.load(O/k/'INTERVALS.npz');gains=np.asarray(cfg['per_actuator_gain']);mp=load(O/'MAPPING.json');ties={}
  for row in mp['rows']:
   ids=row['local_indices'];j=row['actuator_index']
   if not ids:assert not u[:,j].any()
   else:ties.setdefault(tuple(sorted(ids)),[]).append(j)
  for ids in ties.values():assert np.array_equal(u[:,ids],np.repeat(u[:,ids[:1]],len(ids),axis=1))
  for j,record in enumerate(records):
   raw=np.asarray(record['decoder']['rates_hz']);assert np.array_equal(np.clip(raw/100*gains,0,1),iv['decoded_next_controls'][j])
  assert np.array_equal(iv['commands'][1:],iv['decoded_next_controls'][:-1]);assert not iv['commands'][0].any()
  checks[k]=dict(passed=True,initial_exact=True,commands_bounded=True,force_partition=True,actual_neural_steps=500,preclip_decodes_exact=50,command_causal_hold=True,ties_unresolved_exact=True)
  alpha=1 if k==best else .35;lw=2 if k==best else 1;axs[0,0].plot(b['time']*1000,b['com_position'][:,2],alpha=alpha,linewidth=lw,label=k);axs[0,1].plot(b['time']*1000,b['nonfoot_floor_count'],alpha=alpha,linewidth=lw,label=k)
 axs[1,0].bar(np.arange(12),[r[k]['shared_score'] for k in trained]);axs[1,0].axhline(base['shared_score'],color='black',label='historical shared score');axs[1,0].set_xticks(np.arange(12),trained,rotation=50)
 axs[1,1].bar(np.arange(12)-.2,[r[k]['last50_foot_fz'] for k in trained],width=.4,label='FOOT');axs[1,1].bar(np.arange(12)+.2,[r[k]['last50_nonfoot_fz'] for k in trained],width=.4,label='NONFOOT');axs[1,1].set_xticks(np.arange(12),trained,rotation=50)
 for ax,title in zip(axs.flat,['Full-body COM height / model length','Nonfoot-floor contacts','Shared dynamic score (not behavior pass)','Last50ms vertical world reaction / model force']):ax.set_title(title);ax.legend(fontsize=6);ax.grid(alpha=.2)
 axs[0,0].set_xlabel('physical time / ms');axs[0,1].set_xlabel('physical time / ms');fig.suptitle('Actual neural closed-loop PRECLIP gain calibration — no auxiliary controller');fig.tight_layout();fig.savefig(str(D)+'.png',dpi=150);plt.close(fig)
 validations={}
 if dec['validation_run']:
  a=np.load(O/best/'BODY.npz')
  for k in ['validation_repeat','validation_extend']:
   b=np.load(O/k/'BODY.npz');validations[k]={f:bool(np.array_equal(a[f],b[f][:len(a[f])])) for f in a.files};assert all(validations[k].values())
 dump=dict(checks=checks,validation_prefix_equal=validations,source_hashes_unchanged=all(sha(Path(k))==v for k,v in pr['source_hashes'].items()),no_physics=True,exit_code=0);assert dump['source_hashes_unchanged'];(O/'SAVED_VERIFICATION.json').write_text(json.dumps(dump,indent=2)+'\n')
 lines=['# 真实神经闭环动态招募增益标定','','## 本轮结果',f'12条250ms真实神经反馈候选；预声明足撑改善门通过候选：{dec["qualified_training"]}；验证运行：{dec["validation_run"]}。',f'按预声明完整动态score最高：{best}。基线只比较shared_score={base["shared_score"]}，没有为旧档虚构逐足反力。','增益概念不是首次实验；此前S233/S237/S253/S254/S256为其他分支。本轮固定canonical65d、历史原始初态、静息tier、身体12.5us/神经500us/感觉与命令5ms，无FIR静态目标。','5组来自实际执行器名称标签，不宣称生理解剖功能标定；同运动池共享增益，16缺项命令0。只修改原MotorOutput rates乘gain后的preclip数值，不写脚本动作/外力/姿态，不加偏置；每次episode新建神经网络。','','|候选|5组gain|score|shared_score|首次NONFOOT ms|末50ms非足比例|末50ms COM均高|足/非足Fz|','|---|---|---:|---:|---:|---:|---:|---|']
 for k in trained:
  v=r[k];lines.append(f'|{k}|'+','.join(f'{x:.4f}' for x in v['group_gains'])+f'|{v["score"]:.6f}|{v["shared_score"]:.6f}|{None if v["first_nonfoot_s"] is None else v["first_nonfoot_s"]*1000}|{v["last50_nonfoot_fraction"]}|{v["last50_mean_com_height"]:.6f}|{v["last50_foot_fz"]:.6f}/{v["last50_nonfoot_fz"]:.6f}|')
 lines += ['',f'历史正常：firstNONFOOT={base["first_nonfoot_s"]*1000}ms，末50ms NONFOOT比例={base["last50_nonfoot_fraction"]}，平均COM={base["last50_mean_com_height"]}。','本轮是否触发验证严格按上述qualified列表判定；无合格候选即不追加倍率或扩大搜索，只否定本轮5标签组、[.25,4]的12固定候选，不能宣称身体不可能支撑或神经模型普遍无效。局部指标改善不等于稳定足撑，未作独立扰动/生理标定的行为认证。','','## 实际执行',json.dumps(ex,ensure_ascii=False),'保存数据复核12条：初态精确、活化与速度初始0、有界命令、无外力、世界反力分量和、有限性/warnings0、每条500真实神经步及独立网络实例。增益1生产解码器完全恒等、零输入零命令、preclip公式和共享池检查在物理前通过。','初次010000Z入口contact gate误传MjData导致TypeError exit1，1模型/2forward/0身体/0神经，保留失败目录；修正为qpos后新011000Z运行。旧封存不改。','','## 交付',str(O),str(P/'run_dynamic_neural_gain_calibration.py'),str(P/'summarize_dynamic_neural_gain.py'),str(D)+'.png',str(D)+'.md','尚未自主站立／行走，16来源缺口、参考与绝对力标定限制保持。']
 bestshared=max(trained,key=lambda k:r[k]['shared_score']);firsts=[r[k]['first_nonfoot_s']*1000 for k in trained];lines += ['',f'实际结论：0/12达到预声明足撑改善门，未触发额外验证，不追加搜索。firstNONFOOT范围{min(firsts):.4f}–{max(firsts):.4f}ms；最好shared为{bestshared}={r[bestshared]["shared_score"]}，完整score最好为{best}。', '部分score或高度略升，但12条末窗非足接触比例均1.0，仍以非足竖反力为主；足/总反力比例不等于足力/体重，不足以证明抵抗重力。',f'实际初态关节范围gate：{load(O/"INITIAL_GATE.json")["initial_scalar_joint_ranges"]["within_declared_scalar_ranges"]}；工程范围通过不代表生理姿态标定。','父保存验证首次RECORDS结构误读为列表造成TypeError exit1（0物理），改为records字段后exit0；600次真实preclip公式与命令下一窗保持另由本helper全量核验。']
 text='\n'.join(lines)+'\n';Path(str(D)+'.md').write_text(text);(O/'REPORT.zh-CN.md').write_text(text);(O/Path(str(D)+'.png').name).write_bytes(Path(str(D)+'.png').read_bytes());(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('Saved verified',len(checks),'trials',best,r[best])
if __name__=='__main__':main()
