"""Produce desktop report and plots from verified physical records."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plant_probe import O,R,dump,sha
D=Path(__file__).parent

def main():
 L=O/'learning';verification=json.loads((L/'VERIFICATION.json').read_text());assert verification['passed']
 base=json.loads((O/'plant_probe/SUMMARY.json').read_text());initial=json.loads((L/'EVALUATION.json').read_text());extended=json.loads((L/'EXTENDED_RESULTS.json').read_text());training=json.loads((L/'TRAINING.json').read_text())
 results=initial+extended
 fig,axs=plt.subplots(2,1,figsize=(11,8),sharex=True)
 for tag in ['extended_ann_5s','eval_brain_cut_dt0.125','eval_motor_cut_dt0.125','eval_hold_dt0.125','frozen_brain_history']:
  with np.load(L/tag/'NATIVE.npz') as z:rows=z['rows'][::80]
  axs[0].plot(rows[:,0],rows[:,10],label=tag)
  axs[1].plot(rows[:,0],rows[:,1:7].sum(1)/24.446354755878005)
 axs[0].axhline(.8*1.2072528224791905,color='black',linestyle=':',label='height floor')
 for ax in axs:ax.axvline(1,color='gray',linestyle='--');ax.grid(alpha=.2)
 axs[0].set_ylabel('COM height (native units)');axs[0].legend(fontsize=8);axs[1].set_ylabel('Foot support / body weight');axs[1].set_xlabel('Simulation time (s)')
 fig.suptitle('Full body: learned temporal ANN + incremental actuator\nNormal gravity, free root, original torque limits, real contacts');fig.tight_layout();fig.savefig(O/'STANDING_COMPARISON.png',dpi=150);plt.close(fig)
 rows=['|条件|评分时长|通过站立门|足载均值/体重|最低质心高度|非足触地|','|---|---:|---|---:|---:|---|']
 for r in results:
  v=r['validation'];m=v['release'];rows.append(f"|{r['tag']}|{r['seconds']-1:.1f}s|{'通过' if v['passed'] else '失败'}|{m['foot_support_mean']:.6f}|{m['com_z_min']:.6f}|{'无' if v['checks']['D1_no_nonfoot_ground_force'] else '有'}|")
 by={r['tag']:r for r in results};long=by['extended_ann_5s'];freeze=by['frozen_brain_history'];cold=by['cold_ann']
 deflection=[]
 for tag in ['eval_ann_dt0.125','eval_hold_dt0.125']:
  with np.load(L/tag/'BODY.npz') as z:
   q=z['qpos'];t=z['time'];ref=q[np.argmin(abs(t-1.095)),7:];tail=q[t>=2.8,7:]
   deflection.append(dict(tag=tag,post_command_disturbance_body_joint_rms_rad=float(np.sqrt(np.mean((tail-ref)**2)))))
 dump(L/'RECOVERY_COMPARISON.json',dict(importance='重要辅助',records=deflection,disturbance='100ms declared actuator command disturbance, not an external force',metric='physical joint pose difference from pre-disturbance pose, averaged last200ms'))
 brain_statement='冻结预热末脑历史仍通过，说明本次站立不需要持续变化的脑反馈；零脑输入失败不能单独证明脑在完成平衡。' if freeze['validation']['passed'] else '冻结脑历史未通过，结合零脑输入对照说明该策略对脑活动有依赖；这仍不是原脑独立站立或生物功能验证。'
 text=f'''# 增量执行器与人工神经翻译层：实际站立结果

换回增量执行器，并完成全脑活动时间卷积ANN训练与真实物理接管测试。延长评估的5秒ANN接管：**{'通过全部站立判据' if long['validation']['passed'] else '未通过全部站立判据'}**。这次是完整身体、正常重力、自由根、真实六足接触，保持原生力矩上限，没有胸部焊接、其他关节固定、外部托举或身体姿态写入。

## 完成内容

- 执行器：42关节目标角增量积分，有限PD力矩，每1ms更新增量、12.5µs更新力矩；停止肌肉执行器路线。选择首个通过全身保持门的gain2，未增加原生力矩上限。单独增量保持也能站立，因此不能把它的保持能力全部归于ANN。
- 人工翻译层：当前12000细胞运行时的全部活动，20ms因果滤波，最近4个5ms快照，经共享投影、时间卷积与人工隐藏层输出42个关节增量。输入还包含自身已积累的42个目标角偏移，不直接读取身体姿态、速度或教师指令。12000不等于完整153962细胞银行。
- 训练：3条独立受扰训练轨迹、1条开发轨迹，100轮Adam监督学习。开发误差选模型，随后在新扰动seed55闭环评估；没有继续使用旧肌肉力矩标签。训练误差不是站立证据，以下全部来自实际物理积分。
- 接管：初始1秒训练教师准备，之后完全由ANN输出增量；同时实际测试无教师冷启动，冷启动后2秒评分结果为**{'通过' if cold['validation']['passed'] else '失败'}**。未训练替代脑，因为第一条路线已实现可测试的站立系统。

{chr(10).join(rows)}

## 能力归属

{brain_statement}

有限PD是用户明确授权的增量执行器组成部分，全程存在；撤除的是教师控制，不是撤掉执行器。零增量仍保持目标，真正断电机才是零力矩。此成果应称“人工神经翻译层＋增量执行器的站立”，不能称原连接体独立学会站立。保持对照包含与ANN相同的目标命令扰动，不是外界推扰。未展示行走或从倒地翻身站起，绝对SI力值仍为NOT_ESTABLISHED。

## 验证与局限

独立验证通过：保存命令逐步重建目标积分；重新运行物理引擎重放ANN轨迹，与原身体状态相符；同步长各对照的暖态逐字段一致；两个神经步长(.125/.0625ms)的ANN都测试；原模型自由根/71mesh/正常重力与输入哈希核查通过。PyTorch/NumPy导出推理差值：{training['numpy_parity_max']:.3g}。

训练保存float16脑帧，部署使用float32；独立验证单列了这项量化引起的动作差，不声称输入逐位相同。训练/开发轨迹共享扰动前相同前缀，另存DEV_POST_DISTURBANCE.json单独报告1.1秒后误差。DATA中的5ms尖峰分箱漏最后一个窗口，不用它冒充完整尖峰总数，完整累计在END网络状态里。所有失败对照保留。

重要辅助视频：`{O}/ANN_STANDING_4x_slow.mp4`（实际轨迹，4倍慢放，标明教师与ANN阶段）。重要辅助图表：`{O}/STANDING_COMPARISON.png`。核心必需模型：`{L}/TRANSLATOR.npz`；核心必需验证：`{L}/VERIFICATION.json`；完整源码目录：`{D}/`。图像实际目视检查状态另见VISUAL_CHECK.json，不以可解码冒充肉眼验收。

规则更新：`{R}/AGENTS.md`、`{R}/GOAL_ROADMAP.zh-CN.md`。本轮无确认可删除的废弃原始证据；组件测试临时模型已由TemporaryDirectory自动清理，未删除历史模型或实验。
'''
 (O/'REPORT.zh-CN.md').write_text(text)
 paths=[p for p in O.rglob('*') if p.is_file() and p.name!='FILE_INVENTORY.json']+list(D.glob('*.py'))
 inv=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,importance='重要辅助' if p.suffix in ['.png','.mp4','.md'] else '核心必需') for p in sorted(paths)]
 dump(O/'FILE_INVENTORY.json',dict(files=inv,verified_mismatches=sum(sha(Path(r['path']))!=r['sha256'] for r in inv),cleanup='No raw evidence deleted; temporary component parity files automatically removed.'))
 print(json.dumps(dict(report=str(O/'REPORT.zh-CN.md'),files=len(inv),extended_standing=long['validation']['passed'],cold_standing=cold['validation']['passed'],frozen_brain_standing=freeze['validation']['passed']),ensure_ascii=False))
if __name__=='__main__':main()
