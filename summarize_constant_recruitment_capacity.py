"""Saved-only capacity diagnostic report; no new optimization/dynamics."""
from pathlib import Path
import json,hashlib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s2-constant-recruitment-capacity/20261028T001000Z');D=Path('/home/sensen/Desktop/S2_CONSTANT_RECRUITMENT_CAPACITY')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ex=json.loads((O/'EXECUTION.json').read_text());summary=json.loads((O/'SUMMARY.json').read_text());historical=json.loads((O/'HISTORICAL_METRICS.json').read_text());H=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z');fig,axs=plt.subplots(3,2,figsize=(14,12));inputs=[]
 series=[('historical_normal',H/'normal/BODY.npz'),('historical_motor_cut',H/'motor_cut/BODY.npz')]+[(k,O/k/'BODY.npz') for k in summary]
 for label,path in series:
  inputs.append(path);b=np.load(path);t=b['time']*1000
  for ax,y in zip(axs.flat,[b['com_position'][:,2],b['com_velocity'][:,2],b['thorax_position'][:,2],b['nonfoot_floor_count'],b['foot_contact'].sum(1),b['ground_force_world'][:,2]]):ax.plot(t,y,label=label)
 # Saved-only late force partition, no historical model recovery.
 fig2,ax2=plt.subplots(figsize=(10,5))
 for mode in summary:
  b=np.load(O/mode/'BODY.npz');t=b['time'][-4000:]*1000
  ax2.plot(t,b['ground_force_foot_world'][-4000:,:,2].sum(1),label=mode+' FOOT')
  ax2.plot(t,b['ground_force_nonfoot_world'][-4000:,2],linestyle='--',label=mode+' NONFOOT')
 ax2.set_title('Last50ms direct recruitment: foot vs nonfoot world reaction — not neural standing');ax2.set_xlabel('physical time / ms');ax2.set_ylabel('model force units');ax2.legend(fontsize=8);ax2.grid(alpha=.2);fig2.tight_layout();fig2.savefig(str(D)+'_LOAD.png',dpi=140);plt.close(fig2)
 titles=['COM height / model length','COM vertical velocity / model length per second','Thorax height / model length','Nonfoot-floor contact count','Foot contact legs (contact is not load)','Total world vertical reaction / model force']
 for ax,title in zip(axs.flat,titles):ax.set_title(title);ax.set_xlabel('physical time / ms');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle('Constant muscle recruitment capacity diagnostic — NOT neural standing / NOT goal pass');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(str(D)+'.png',dpi=140);plt.close(fig)
 text=['# 固定肌肉招募承载容量诊断（不是神经站立）','','## 直接结论','两个离线固定状态优化模式，候选命令只做从原始初态开始的常量输入物理rollout。没有在线反馈、姿态修正、外力、activation preload或神经运行。','优化使用历史normal15ms固定qpos/qvel；固定态设置activation仅表示稳态肌肉容量，不是动态运行初态。原始关节范围违规保留，绝对力未标定。','','| 模式 | 首次非足触地ms | 末50ms非足比例 | 末50ms平均COM高度 | 末COM高度 | 末COMvz | 有界显著改善 |','|---|---:|---:|---:|---:|---:|---|']
 for label,r in list(historical.items())+list(summary.items()):text.append('| '+label+' | '+' | '.join(str(None if r.get(k) is None else r[k]*1000 if k=='first_nonfoot_s' else r[k]) for k in ['first_nonfoot_s','last50_nonfoot_fraction','last50_mean_com_height','final_com_height','final_com_vz','substantial_engineering_support_improvement'])+' |')
 text+=['','## 固定状态目标与优化','预声明目标：COMaz/g权重1、根部局部角加速度3/4乘L/g权重.3、水平COM加速度/g权重.1、effort平方权重.01。坐标3/4不是宣称世界roll/pitch。不得把不同量纲归一权重当生理标定。','74模式按实际manifest的相同local_indices共享控制变量，未解析16命令为0；90模式独立直接肌肉控制仅作上界诊断，不补节点、不放行映射。']
 for mode in summary:
  opt=json.loads((O/mode/'OPTIMIZATION.json').read_text());chk=json.loads((O/mode/'DETERMINISTIC_CHECKS.json').read_text());text += ['',mode+': '+json.dumps({k:opt[k] for k in ['success','status','message','nfev','forward_calls','initial_loss','final_loss','initial_com_acceleration','final_com_acceleration']},ensure_ascii=False), '优化success仅指优化器停止条件；max_nfev停止或候选失败不等机械不可行。检查 '+json.dumps(chk,ensure_ascii=False)]
 text+=['','## 瓶颈决策','两条候选均未改善末窗足撑：只延后首次非足触地约5–6ms，随后下沉更深并转为非足承载。15ms固定状态局部loss大幅降低未转移到原始初态的动态释放。不能训练神经翻译层去追这两个不良目标；下一次招募假设必须在真实rollout上评价，不可只依赖单态加速度近零。不证明模型无支撑能力，也不授权立即猜映射或扩大搜索。','必须同时看足地世界力与非足接触；末态正vz、接触数或固定态loss降低均不能充当足撑。','','## 执行与范围',json.dumps(ex,ensure_ascii=False),'原历史normal/motorcut只读，不重跑；新2模式最多1000forward各，最多2条250ms。没有神经/新baseline。输入哈希见PREREGISTRATION，默认模型与历史池不改。','','## 交付',str(O),str(P/'run_constant_recruitment_capacity.py'),str(P/'summarize_constant_recruitment_capacity.py'),str(D)+'.png',str(D)+'.md','本报告是工程容量诊断，16映射缺口、动物参考与绝对力限制、未自主站立行走保持。']
 text += ['', '两模式都未获预声明显著改善；固定15ms局部loss降低没有转移为从原始初态释放后的持久足撑。不能把这两个不良常量目标拿来训练神经翻译层。优化器均到max_nfev停止，未认证收敛，更不认证全局不可行。', '末50ms足/非足竖反力及最小COM高度：']
 for mode in summary:
  b=np.load(O/mode/'BODY.npz');fz=float(b['ground_force_foot_world'][-4000:,:,2].sum(1).mean());nf=float(b['ground_force_nonfoot_world'][-4000:,2].mean());text.append(mode+': FOOTmeanFz='+str(fz)+', NONFOOTmeanFz='+str(nf)+', foot_fraction_of_totalFz='+str(fz/(fz+nf))+', minimumCOMheight='+str(float(b['com_position'][:,2].min())))
 text += ['', '父代理保存数据独立复核：首次shell/python单行语法错误exit1，未执行导入/模型/物理；改用heredoc复核exit0，确认原始qpos、act0/qvel0、常量命令、无外力/warnings0、世界力分量和。动态运行本身exit0无失败，不改其EXECUTION。']
 text.append(str(D)+'_LOAD.png');(O/Path(str(D)+'_LOAD.png').name).write_bytes(Path(str(D)+'_LOAD.png').read_bytes())
 (O/'REPORT.zh-CN.md').write_text('\n'.join(text)+'\n');Path(str(D)+'.md').write_text('\n'.join(text)+'\n');(O/Path(str(D)+'.png').name).write_bytes(Path(str(D)+'.png').read_bytes());(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('Saved-only report',D)
if __name__=='__main__':main()
