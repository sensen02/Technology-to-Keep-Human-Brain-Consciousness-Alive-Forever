"""Generate inspectable plot, concise Chinese report and hashed file inventory."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261107T002222Z')
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 v=json.loads((O/'VERIFICATION.json').read_text());assert v['passed']
 summary=json.loads((O/'SUMMARY.json').read_text());rs={r['tag']:r for r in v['results']}
 warm=load(O/'warmup3s/TRACE.npz');start=load(O/'warmup3s/START.npz')
 fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
 colors={'withdraw':'#c43d3d','sustained':'#1e78b4','withdraw_sensory_cut':'#8b55a5','withdraw_motor_cut':'#777777'}
 for col,name in enumerate(['Flexor pool','Extensor pool']):
  cnt=warm['counts'].reshape(-1,100,3).sum(axis=1)
  axes[col].plot(np.arange(30)*.1+.05,cnt[:,col],color='#222222',label='Warmup: live neural + physical loop')
  for tag,color in colors.items():
   r=rs[tag];bins=np.array(r['counts100'])
   axes[col].plot(3+np.arange(10)*.1+.05,bins[:,col],color=color,label=tag,linestyle='--' if 'cut' in tag else '-')
  axes[col].set_ylabel(name+' spikes / 100 ms');axes[col].grid(alpha=.2)
 axes[2].plot(warm['rows'][::80,0],warm['rows'][::80,1],color='#222222')
 for tag,color in colors.items():
  z=load(O/tag/'TRACE.npz');axes[2].plot(z['rows'][::80,0],z['rows'][::80,1],color=color,linestyle='--' if 'cut' in tag else '-')
 axes[2].set_ylabel('Left front tibia angle (rad)');axes[2].set_xlabel('Physical simulation time (s)');axes[2].grid(alpha=.2)
 for ax in axes:ax.axvline(3,color='black',linestyle=':');ax.axvspan(0,3,color='#dddddd',alpha=.25)
 axes[0].legend(fontsize=8,loc='upper right')
 fig.suptitle('3 s continuous warmup + 1 s handover\nFull body fixture / one active joint / 0.1 gravity / no contact')
 fig.tight_layout();fig.savefig(O/'WARMUP_RESULT.png',dpi=160);plt.close(fig)
 w=rs['withdraw']['motor_tail500'];s=rs['sustained']['motor_tail500'];c=rs['cold_none']['motor_tail500']
 verdict='局部持续活动通过预登记判据' if v['local_activity_persistence'] else '3秒预热后，撤流仍未维持局部运动神经活动'
 warm_bins=np.asarray(rs['warmup3s']['counts100']);stats=dict(last_second_motor100=warm_bins[-10:,:2].tolist(),last_second_q_min=float(warm['rows'][-80000:,1].min()),last_second_q_max=float(warm['rows'][-80000:,1].max()))
 dump(O/'WARMUP_TAIL.json',dict(importance='重要辅助',**stats))
 text=f'''# 3秒预热后的实际结果

{verdict}。

本轮实际运行了3秒连续神经—肌肉—物理预热，再从同一个暖态分出1秒撤流、持续电流、撤流并切动态感觉、撤流并切运动输出四个对照，另跑1秒冷启动参照。合计8秒物理积分。神经、身体、肌肉与滤波状态在交接处完全保留。

撤流后最后半秒，屈/伸运动池尖峰为 **{w[0]}/{w[1]}**；同暖态持续电流为 **{s[0]}/{s[1]}**；冷启动参照为 **{c[0]}/{c[1]}**。预登记要求各池都超过冷启动，并至少达到持续电流组的一半。实际判定：**{v['verdict']}**。

撤流后屈池只在 **{rs['withdraw']['last_motor_ms'][0]} ms** 留下一枚尖峰，伸池没有尖峰；此后直到1秒观察结束，两组均无发放。前运动电流确实在交接后为零，位置感觉与0.03nA感觉背景保留。切运动组仍保留肌肉去激活过程和被动张力，不可把随后机械保持或回弹称神经持续控制。

旧版核查：历史版本已有约1.3秒物理辅助、其中1.1秒网络在线的连续预热；当前250ms局部实验没有那段预热。之前短窗失败不能用来否定“预热可能有帮助”。本轮只检验当前局部接口3秒注流预热的结果，不能排除其他预热输入或完整六足物理辅助方案。

当前仍为完整身体、胸部固定、其余41关节保持、关闭接触、0.1倍重力的左前胫关节诊断。没有恢复旧版六足辅助步行，也没有自主站立或行走证据。冷启动参照与暖态身体不同，不单独用于证明神经记忆效应。只有一个神经步长0.0078125ms，不声称跨步长稳健。绝对SI力值：NOT_ESTABLISHED。

独立验证已通过：交接全部保存状态逐位相等；事件重算尖峰计数；独立滤波/数值积分重建命令；真实关节角重建感觉电流；重建肌肉激活与拉力；输入哈希未变；全部轨迹有限、有界、在关节范围内、无物理警告。新预热的前250ms还逐位复现了旧版同配置身体轨迹。

预热末1秒屈/伸每100ms计数：{stats['last_second_motor100']}。该段关节角范围：[{stats['last_second_q_min']:.9f}, {stats['last_second_q_max']:.9f}]rad。持续发放不等于严格稳态。

核心必需：`{O}/VERIFICATION.json`、`{O}/PREREGISTRATION.json`、各分支原始TRACE/START/END文件。重要辅助：`{O}/WARMUP_RESULT.png`、本报告与WARMUP_TAIL.json。源码：`{D}/seconds_warmup.py`、`{D}/verify_seconds.py`、`{D}/report_seconds.py`。图像已生成；实际read_image调用被当前模型不支持图像输入拒绝，未完成肉眼核验，数值结论来自独立原始记录验证。本轮没有确认应删除的临时或废弃文件，没有删除历史证据。
'''
 (O/'REPORT.zh-CN.md').write_text(text)
 files=sorted([p for p in O.rglob('*') if p.is_file() and p.name!='FILE_INVENTORY.json']+[p for p in D.glob('*.py')])
 inventory=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,importance='重要辅助' if p.suffix in ['.png','.md'] or p.name=='WARMUP_TAIL.json' else '核心必需') for p in files]
 dump(O/'FILE_INVENTORY.json',dict(files=inventory,deleted_files=[],cleanup='No confirmed redundant or temporary task files; raw historical evidence retained.'))
 assert all(sha(Path(r['path']))==r['sha256'] for r in inventory)
 print(json.dumps(dict(report=str(O/'REPORT.zh-CN.md'),image=str(O/'WARMUP_RESULT.png'),files=len(files),hash_mismatches=0),ensure_ascii=False))
if __name__=='__main__':main()
