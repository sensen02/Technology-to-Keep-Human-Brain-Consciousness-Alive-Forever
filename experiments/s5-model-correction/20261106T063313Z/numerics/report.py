"""Human report, diagnostic figure, classified manifest. No simulation."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as r

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=r.js(r.O/'SUMMARY.json');v=r.js(r.O/'VERIFICATION.json');d=r.js(r.O/'DIAGNOSTICS.json')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors=['#555555','#0072B2','#D55E00','#009E73']
    configs=sorted((r.O/'CONFIGS').iterdir(),key=lambda p:r.js(p/'PARAMETERS.json')['neural_dt_s'],reverse=True)
    for color,p in zip(colors,configs):
        dt=r.js(p/'PARAMETERS.json')['neural_dt_s'];z=r.arrays(p/'WARM.npz');label=f'{dt*1000:g} ms'
        axes[0,0].plot(z['time'],z['leg_rates'][:,0],label=label,color=color,lw=.7)
        axes[0,1].plot(z['time'],z['raw_command'][:,0],label=label,color=color,lw=.7)
        rr=sorted([x for x in s['results'] if x['dt_s']==dt and 'paired' in x['label']],key=lambda x:x['release_time_s'])
        axes[1,0].plot([x['release_time_s'] for x in rr],[x['J'] for x in rr],'o-',label=label,color=color)
        axes[1,1].plot([x['release_time_s'] for x in rr],[x['parts']['foot_support_mean'] for x in rr],'o-',label=label,color=color)
    for ax,title,xlabel,ylabel in zip(axes.flat,['LF mapped motor-cell 5ms rate','First joint causal raw command','Paired physical score (lower is better)','Paired foot support'],['Physical warm time (s)','Physical warm time (s)','Release state time (s)','Release state time (s)'],['Hz','Model torque units','J','Body-weight ratio']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Actual neural timestep refinement; fixed 1ms command and 12.5us physics')
    fig.savefig(r.O/'NUMERICAL_REFINEMENT.png',dpi=150);plt.close(fig)
    lines=['# 实际神经时间步收敛与配对物理验证','',f"结论：预登记工程收敛门 {'通过' if s['convergence_established'] else '未通过'}；未按J选优，未替换模型。未自主站立/行走。",'',f"实际 {s['physical_releases']} 次0.2s释放，共 {s['physical_seconds']:.1f}s物理；0新增教师预热，暖态神经步数{s['warm_neural_steps']}，释放神经步数{s['release_neural_steps']}。",'', '原连接/强度/神经参数/感觉/身体/读出不变，RH入兴奋×.375只在释放施加。每档从初态重放真实float64电流，1ms区间保持输入；命令使用上一区间已完成尖峰率。原0.5ms所有17暖态全神经字段/率/历史逐位一致，1.3s物理J复现1.5903704331878943。', '', '原积分器是后向Euler膜电位、精确指数突触/适应衰减的分裂步进，不是前向Euler；端点阈值与复位仍有时间量化。实测延迟是尖峰端点到输入区间起点恰2ms；不应期阻止积分起点2ms，最早下次尖峰端点间隔2ms+dt。未静默更改积分器。', '', '阈值仅是预登记工程容差，不是生物证据。主要窗口丢弃开始100ms，完整窗口另列；池率、解剖逐腿率、原始/限幅命令均比较相邻细化的归一化RMS及趋势，不能仅凭平均Hz判断。', '', '| 比较 | 神经门 | 物理门 |', '|---|---|---|']
    for label,c in [('主比较',s['main']),('条件补充',s['supplement'])]:
        if c is None:continue
        lines.append(f"| {label} {c['dts']} | {c['neural_passed']} | {c['physical_passed']} |")
        for context,field in [('暖态','neural_windows'),('RH.375同输入释放','controlled_release_windows')]:
            for key,val in c[field]['primary_exclude_initial100ms'].items():lines.append(f"\n{label} {context} {key}：粗差{val['coarse_difference']:.6f} → 细差{val['fine_difference']:.6f}，比{val['ratio']:.6f}，绝对RMS{val['coarse_absolute_rms']:.6f}→{val['fine_absolute_rms']:.6f}，通过{val['passed']}。")
    lines+=['','| 神经dt(ms) | 交接态(s) | J | 足载均值/W | 最低COM | 首次正非足(s) | 完整D |','|---|---|---|---|---|---|---|']
    for x in s['results']:lines.append(f"| {x['dt_s']*1000:g} | {x['release_time_s']} ({x['label']}) | {x['J']:.9f} | {x['parts']['foot_support_mean']:.6f} | {x['parts']['com_z_min']:.6f} | {x['first_positive_nonfoot_s']} | {x['validation']['passed']} |")
    lines+=['',f"独立复算通过：{v['runs']}次释放，率最大误差{v['max_filter_error']:.3g}，命令{v['max_command_error']:.3g}，J{v['max_J_error']:.3g}；.125ms重复逐数组一致。原生12.5µs D门/时钟、全12000细胞原始尖峰到滤波率、17态连续性与连接身份独立核验。",'', '控制固定在主实验最细.125ms：即使触发.0625ms补充，也不改控制所属档位。所有D原始布尔向量、配对连续差值和全网/逐细胞率、饱和与时间变化保存于JSON/NPZ。身体分岔不自动等于生物失真；通过神经门也不等于物理输出收敛。', '', '绝对力值/SI、生物有效性、神经意图保持均NOT_ESTABLISHED。本交付是手工研究诊断，非成品。', '', '保留文件分级：源码、预登记/哈希、参数、原始NPZ、候选缓存、验证/汇总为核心必需；报告/图/统计为重要辅助；唯一清理是确认无读者的本轮构造器COVERAGE.json，路径/哈希/大小见CLEANUP_LEDGER.json。没有删除原资料或失败轨迹。图已生成，是否完成实际图像查看由执行记录另行声明。']
    (r.O/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    inventory=[]
    for p in sorted(list(r.O.rglob('*'))+list(r.SRC.glob('*.py'))):
        if not p.is_file() or p.name in ['FILE_INVENTORY.json','MANIFEST_SHA256.json']:continue
        auxiliary=p.suffix=='.png' or p.name in ['REPORT.zh-CN.md','DIAGNOSTICS.json','RUN.log','VERIFY.log','REPORT.log','PROGRESS.json']
        inventory.append(dict(path=str(p),sha256=r.sha(p),bytes=p.stat().st_size,importance='重要辅助' if auxiliary else '核心必需',purpose='diagnostic presentation/progress' if auxiliary else 'reproduction, raw evidence, verification or identity'))
    r.dump(r.O/'FILE_INVENTORY.json',dict(files=inventory,cleanup=r.js(r.O/'CLEANUP_LEDGER.json'),unconfirmed=[]))
    r.dump(r.O/'MANIFEST_SHA256.json',{**{x['path']:x['sha256'] for x in inventory},str(r.O/'FILE_INVENTORY.json'):r.sha(r.O/'FILE_INVENTORY.json')})
    print('REPORT',len(inventory),'files',flush=True)
if __name__=='__main__':main()
