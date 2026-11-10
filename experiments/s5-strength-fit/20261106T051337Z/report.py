"""Charts, honest report and complete classified hash inventory."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as r
O=r.O

def inventory():
    entries=[];hashes={}
    for root in [r.SRC,O]:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.name in ['FILE_INVENTORY.json','MANIFEST_SHA256.json']:continue
            auxiliary=p.suffix in ['.md','.png'] or p.name.startswith(('CLEANUP','CHANGE_LEDGER')) or p.name=='report.py'
            level='重要辅助' if auxiliary else '核心必需'
            purpose='report/chart/review and provenance explanation' if auxiliary else 'reproduction, fixed protocol, model identity, raw evidence or verification'
            entries.append(dict(path=str(p),importance=level,purpose=purpose,bytes=p.stat().st_size,sha256=r.sha(p)));hashes[str(p)]=r.sha(p)
    r.dump(O/'FILE_INVENTORY.json',dict(files=entries,excluded_self_referential=['FILE_INVENTORY.json','MANIFEST_SHA256.json'],unknown=[]))
    hashes[str(O/'FILE_INVENTORY.json')]=r.sha(O/'FILE_INVENTORY.json');r.dump(O/'MANIFEST_SHA256.json',hashes)
    assert all(r.sha(p)==h for p,h in hashes.items());print('INVENTORY',len(hashes),flush=True)

def main():
    if len(sys.argv)>1 and sys.argv[1]=='inventory':inventory();return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    s=r.js(O/'SUMMARY.json');v=r.js(O/'VERIFICATION.json');g=r.js(O/'GROUPING.json');m=s['candidate_metrics'];names=[x['candidate'] for x in m]
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');idx=np.arange(len(m));base=m[0]
    axs[0,0].plot(idx,[x['worst_J'] for x in m],'o-',label='worst train J');axs[0,0].axhline(base['worst_J'],color='k',ls='--',label='baseline');axs[0,0].set(title='Fixed 24-candidate bounded pilot',xlabel='candidate index',ylabel='worst train J (lower better)');axs[0,0].legend()
    guard=[next(z for z in s['results'] if z['candidate']==name and z['release_time_s']==1.3 and '_train_guard_' in z['label']) for name in names]
    axs[0,1].scatter([x['parts']['foot_support_mean'] for x in guard],[x['J'] for x in guard],c=['tab:green' if x['guard'] else 'tab:red' for x in m]);axs[0,1].scatter(guard[0]['parts']['foot_support_mean'],guard[0]['J'],marker='*',s=180,color='k',label='baseline');axs[0,1].set(title='1.3s guard: green passes all four metrics',xlabel='mean foot support / W',ylabel='J');axs[0,1].legend()
    dm=s['selection']['dev_metrics'];xx=np.arange(len(dm));axs[1,0].bar(xx,[x['worst_J'] for x in dm]);axs[1,0].set_xticks(xx,[x['candidate'] for x in dm]);axs[1,0].set(title='Distinct development states select model',ylabel='worst dev J')
    b=guard[0];sel=next(z for z in guard if z['candidate']==s['selected']);seen=set()
    for label,res in [('baseline',b),('selected',sel)]:
        if res['folder'] in seen:continue
        seen.add(res['folder']);n=r.arrays(Path(res['folder'])/'NATIVE.npz')['rows'];axs[1,1].plot((n[:,0]-1.3)*1000,n[:,10],label=label)
    axs[1,1].axhline(.8*1.2072528224791905,color='k',ls='--',label='D8 absolute floor');axs[1,1].set(title='Actual physical COM at 1.3s release',xlabel='time after release / ms',ylabel='COM height');axs[1,1].legend()
    fig.suptitle('Engineering strength fit; fixed topology/signs/readout; not biological calibration')
    image=O/'STRENGTH_FIT.png';fig.savefig(image,dpi=150);plt.close(fig)
    with Image.open(image) as im:im.load();shape=im.size
    r.dump(O/'CHART_VALIDATION.json',dict(path=str(image),decoded=True,size=shape,sha256=r.sha(image),visual_review='read_image must be attempted separately; decoding alone is not human visual review'))
    passed=sum(x['guard'] for x in m[1:]);text=f'''# S5 固定拓扑强度拟合首轮研究报告

本轮已完成有限工程试验，选择 **{s['selected']}**。本轮释放 {s['physical_releases']} 次，完整 D 门通过 {s['all_D_passes']} 次。不将指标改善等同自主支撑或行走；生物有效性、运动意图保持及绝对力值/SI 均为 NOT_ESTABLISHED。

实际网络12000细胞、{g['edges']}条已有边，按对称连接图RCM排序分成10个结构组，每组独立入兴奋/入抑制倍率，共20维。分组不冒称解剖模块；孤立细胞 {len(g['isolated_cells'])}，无入边细胞 {len(g['no_incoming_cells'])}。所有细胞归组，所有已有边只归一个目标组；无入边细胞没有可拟合的入边。实际注释中6个glia标签模型细胞未删除，该身份问题不由本轮校准解决。

仅调整已有边相对原权重0.9—1.1倍率，符号与拓扑固定；旧RH入兴奋0.375仍只在释放时施加一次。原5ms读出、矩阵、身体、感觉、神经参数及执行器限幅未改。在线无新身体状态输入、PD、CPG或按时切换。

一次1.4s真实物理教师预热直接记录1200个双精度电流窗，逐候选从静息神经状态重放2400步，重建全部12000细胞状态、延迟环、率与历史。旧缓存全神经/身体/率/历史及204细胞尖峰逐位相同。旧审计float32电流不用于候选暖启动；其精度/求解器状态差异单列 WARM_PHYSICAL_VALIDATION.json。新final身体状态虽然在预热中生成，释放成绩在选择冻结前未访问。

优化器为明确命名的有界自适应对角进化策略，并非CMA-ES软件：固定种子、3代×8反向配对高斯候选、精英均值与对角方差更新；20维仅24候选是有界首轮，不是穷尽搜索。训练四状态以完整D门、最差J和参数偏离罚项排序；每候选另测1.3s J/足载/最低COM/首次正非足触地四量不退步。非基线通过保护门 {passed}/24。

四个不同dev状态仅评估基线与每代一个训练决赛者，参与明确选择；dev过门数/最差J/平均J不得退步，随后按过门数、最差J加罚项、平均J选。选择前冻结SELECTION.json。final跳过={s['final_skipped']}；扩展释放运行数={s['extended_runs']}。非基线未被选中时按预登记不执行新状态final、逐边扰动或扩展，不用基线回退冒充拟合成功。

实际神经重放网络 {s['neural_replay_networks']} 个、{s['neural_replay_steps']} 步；物理总时长 {s['physical_seconds']:.6f}s含教师1.4s。失败候选和全部轨迹保留。重复/运动切断/感觉切断结果见SUMMARY.json；切断仅检验本短窗链路依赖，不等同走停意图。

独立验证：{v['runs']}条释放，从原始尖峰以scipy lfilter重算滤波和实际命令，从12.5微秒原生数据展开J/D门，逐边核对界限与拓扑，复核全物理交接及ES生成更新。最大滤波误差 {v['max_filter_error']:.3g}，最大命令误差 {v['max_command_error']:.3g}，最大J误差 {v['max_J_error']:.3g}。所选重复逐数组一致，冻结输入哈希不变。

核心必需：{O}/PREREGISTRATION.json、{O}/SUMMARY.json、{O}/VERIFICATION.json、{O}/CONFIGS、{O}/RUNS、{O}/TRUE_CURRENTS.npz、{O}/DATA、{r.SRC}/run.py、{r.SRC}/verify.py。重要辅助：本报告、{image}、父代理设计审阅、变更和清理台账。完整逐文件分类见{O}/FILE_INVENTORY.json，哈希见{O}/MANIFEST_SHA256.json。

清理仅执行已检查无读者的本轮构造器再生COVERAGE.json；每次删除路径/字节/哈希见CLEANUP_LEDGER.json。历史归档及失败证据未删，tracker由父代理负责。
'''
    (O/'REPORT.zh-CN.md').write_text(text);inventory()
if __name__=='__main__':main()
