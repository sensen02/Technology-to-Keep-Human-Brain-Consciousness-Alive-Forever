"""Comparison report, plots, dependency identity and classified delivery manifest."""
import sys
from pathlib import Path
import numpy as np
import run as r
O=r.O

def inventory():
    entries=[];hashes={};paths=list(O.rglob('*'))+list(r.SRC.glob('*.py'))
    dep=r.SRC/'.venv';dependency_paths=[dep/'pyvenv.cfg',dep/'lib/python3.12/site-packages/readonly_dependencies.pth']
    import cma
    cp=Path(cma.__file__).parent;dependency_paths+=list(cp.rglob('*.py'))+list((cp.parent/'cma-4.5.0.dist-info').glob('*'))
    dependency_paths=[p for p in dependency_paths if p.is_file()]
    r.dump(O/'DEPENDENCY_IDENTITY.json',dict(cma_version=cma.__version__,new_venv=str(dep),python=str(dep/'bin/python'),readonly_dependencies=['/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-translator/20261105T200021Z/.venv/lib/python3.12/site-packages','/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages'],files={str(p):r.sha(p) for p in dependency_paths},importance='核心必需',other_venv_files='reinstallable environment support; no recursive traversal of old dependency trees'))
    paths+=dependency_paths+[O/'DEPENDENCY_IDENTITY.json']
    for p in sorted(set(paths)):
        if not p.is_file() or p.name in ['FILE_INVENTORY.json','MANIFEST_SHA256.json']:continue
        auxiliary=p.suffix in ['.md','.png'] or p.name.startswith(('CLEANUP','CHANGE_LEDGER','PREFLIGHT','CHART')) or p.name=='report.py'
        level='重要辅助' if auxiliary else '核心必需';h=r.sha(p);hashes[str(p)]=h
        entries.append(dict(path=str(p),importance=level,purpose='report, visualization, design review or audit notes' if auxiliary else 'reproduction, protocol, source, candidate identity, raw evidence or verification',bytes=p.stat().st_size,sha256=h))
    r.dump(O/'FILE_INVENTORY.json',dict(files=entries,excluded_self_referential=['FILE_INVENTORY.json','MANIFEST_SHA256.json'],environment_support=dict(path=str(dep),importance='可再生临时',purpose='installed interpreter/pip support; retained for reproduction; actual CMA and dependency identity explicitly inventoried'),unknown=[]))
    hashes[str(O/'FILE_INVENTORY.json')]=r.sha(O/'FILE_INVENTORY.json');r.dump(O/'MANIFEST_SHA256.json',hashes);assert all(r.sha(p)==h for p,h in hashes.items());print('INVENTORY',len(hashes),flush=True)

def main():
    if len(sys.argv)>1 and sys.argv[1]=='inventory':inventory();return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    s=r.js(O/'SUMMARY.json');v=r.js(O/'VERIFICATION.json');old=r.js(r.OLD/'SUMMARY.json');m=s['candidate_metrics'];base=m[0];arms=['diagonal','cma','de'];colors=['tab:blue','tab:orange','tab:green'];stats=[]
    guard=lambda name:next(z for z in s['results'] if z['candidate']==name and z['release_time_s']==1.3 and '_train_guard_' in z['label'])
    for arm in arms:
        mm=[x for x in m if x['candidate'].startswith(arm+'_')];gg=[guard(x['candidate']) for x in mm];pop=np.array(r.js(O/f'{arm}_BATCH_0_AFTER.json')['population']);dm=[x for x in s['selection']['dev_metrics'] if x['candidate'].startswith(arm+'_')]
        stats.append(dict(arm=arm,candidates=len(mm),guard_passes=sum(x['guard'] for x in mm),best_train_worst_J=min(x['worst_J'] for x in mm),best_guard_J=min(x['J'] for x in gg),min_guard_violation=min(x['guard_violation'] for x in mm),best_representative_dev_worst_J=min(x['worst_J'] for x in dm),initial_strength_rms=float(np.sqrt(np.mean((.1*pop)**2))),initial_normalized_l2_mean=float(np.linalg.norm(pop,axis=1).mean()),unique_candidates=len(np.unique(np.array([r.js(O/'CONFIGS'/x['candidate']/'PARAMETERS.json')['x'] for x in mm]),axis=0))))
    r.dump(O/'ARM_COMPARISON.json',dict(arms=stats,baseline=base,historical_large_scale_diagonal=dict(candidates=24,initial_sd=.35,best_train_worst_J=min(x['worst_J'] for x in old['candidate_metrics'][1:]),guard_passes=sum(x['guard'] for x in old['candidate_metrics'][1:])),single_seed=True))
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for arm,color in zip(arms,colors):
        mm=[x for x in m if x['candidate'].startswith(arm+'_')];gg=[guard(x['candidate']) for x in mm]
        axs[0,0].plot(np.arange(1,25),[x['worst_J'] for x in mm],'.-',color=color,label=arm)
        axs[0,1].plot(np.arange(1,25),[x['J'] for x in gg],'.-',color=color,label=arm)
        axs[1,0].scatter([x['parts']['foot_support_mean'] for x in gg],[x['parts']['com_z_min'] for x in gg],color=color,label=arm,alpha=.7)
    axs[0,0].axhline(base['worst_J'],color='k',ls='--',label='baseline');axs[0,0].set(title='Worst train J / four states',xlabel='candidate evaluation (per arm)',ylabel='J, lower is better');axs[0,0].legend()
    bg=guard('baseline');axs[0,1].axhline(bg['J'],color='k',ls='--',label='baseline');axs[0,1].set(title='Protected 1.3s-state J',xlabel='candidate evaluation (per arm)',ylabel='J, lower is better');axs[0,1].legend()
    axs[1,0].scatter(bg['parts']['foot_support_mean'],bg['parts']['com_z_min'],marker='*',s=180,color='k',label='baseline');axs[1,0].set(title='Guard metrics (two of four)',xlabel='mean foot load / W',ylabel='minimum COM');axs[1,0].legend()
    dm=s['selection']['dev_metrics'];xx=np.arange(len(dm));axs[1,1].bar(xx,[x['worst_J'] for x in dm],color=['k']+[colors[arms.index(x['candidate'].split('_')[0])] for x in dm[1:]]);axs[1,1].set_xticks(xx,[x['candidate'] for x in dm],rotation=35,ha='right');axs[1,1].set(title='Dev scores for one representative / batch',ylabel='worst dev J');axs[1,1].axhline(dm[0]['worst_J'],color='k',ls='--')
    fig.suptitle('Offline optimizer pilot: same 20D bounds, initial scale and 24 paid candidates per arm')
    image=O/'OPTIMIZER_COMPARISON.png';fig.savefig(image,dpi=150);plt.close(fig)
    with Image.open(image) as im:im.load();shape=im.size
    r.dump(O/'CHART_VALIDATION.json',dict(path=str(image),decoded=True,size=shape,sha256=r.sha(image),visual_review='read_image attempted separately; decoding is not visual review'))
    rows='\n'.join(f"| {a['arm']} | {a['candidates']} | {a['guard_passes']} | {a['best_train_worst_J']:.9f} | {a['best_guard_J']:.9f} | {a['best_representative_dev_worst_J']:.9f} |" for a in stats)
    controls=[z for z in s['results'] if z['label'].startswith('selected_')];control_text='; '.join(f"{z['label']} J={z['J']:.9f}" for z in controls)
    report=f'''# S5 更换离线训练算法的等预算试验

已实际执行三种离线优化器各24候选，最终受保护选择为 **{s['selected']}**。共{s['physical_releases']}次真实物理释放，完整D门通过{s['all_D_passes']}次。当前仍未证明自主支撑或行走；不能将本试验失败推广为所有训练算法或±10%空间无解。

| 算法 | 候选数 | 四量保护通过 | 最好训练最差J | 最好保护J | 三代表最好dev最差J |
|---|---:|---:|---:|---:|---:|
{rows}

原基线训练最差J={base['worst_J']:.9f}，保护J={bg['J']:.9f}，dev最差J={dm[0]['worst_J']:.9f}。表中各列的最优值可能来自不同候选，不能拼成一个模型的表现。全部候选、四项保护和开发成绩见{O}/SUMMARY.json；初始实际半径见{O}/ARM_COMPARISON.json。

三种算法：旧对角ES（原更新公式、SD下限.05保留，仅初始SD从.35改为.05）；pycma4.5.0真正的全协方差CMA-ES（完整协方差C及均值、pc、ps、sigma、迭代数等关键状态逐批存档，固定版本/种子可重放；不是全部内部字段/RNG对象序列化，按共同字典序转成数值秩tell）；DE/rand/1/bin（种群8、F.5、CR.9、三个不同donor且排除目标、强制jrand、延后逐目标选择，完全相同键保留原目标）。DE第1批初始化计入评价预算，第2/3批为真正变异交叉试验。每种算法独立固定种子20261106，初始名义归一SD均.05，对应强度SD.005。ES用反向成对采样，CMA和DE用各自高斯采样；初始总体不同，适应后的半径也不相同。ES与DE因相同default_rng种子，前4个初始方向实际完全相同；仍分别计入各自评价预算，不称72个唯一方向。交叉臂逐位复核见{O}/CROSS_ARM_VERIFICATION.json。

20维对应同一RCM结构10组的入兴奋/入抑制倍率，实际边0.9–1.1倍，拓扑、符号、电参数、感觉、5ms池读出、完整身体固定。RH入兴奋.375只在交接时施加。优化器仅离线选择权重；在线仍为原神经—关节力矩—感觉闭环（当前42个理想关节力矩执行器，不称肌肉有效性已验证），没有身体/时间控制器或新增PD/CPG。

每候选均从神经初态回放旧轮实际float64 TRUE_CURRENTS，2400神经步，重建自己的完整12000细胞状态及历史。复用旧17物理状态，经基线全神经字段/历史/池率/204尖峰逐位验证，1.3s原J精确复现。新教师运行0次，不用旧事后重算float32电流。四个final身体状态旧轮已生成但未释放评分；在本轮选择前保持评分留出，不能称首次生成或从未见过记录。

公平预算：每算法24个目标候选×(训练4+保护1)=120次；每8个新评估点选一个代表测dev4，各算法12次；共享基线9次、选中重复/双切断3次，主试验408次。DE代表也只能从新8试验点中提名，dev不回传优化器。四量保护为J不升、足载/最低COM/首次正非足触地不降；dev完整D数不降且最差/均值J不升，才可选择非基线。最终选择冻结后才可能做额外终测8次、六扰动30次；只有所选final与扰动全部D通过才允许延长。本轮final跳过={s['final_skipped']}，延长次数={s['extended_runs']}。切断/重复：{control_text}。

历史大尺度ES（初始SD.35、24候选）只作上下文；新小尺度ES是初始化对照，避免把尺度变化全算作新算法收益。此次CMA只有3次更新、DE只有2代变异，20维种群8的初始仿射秩至多7；一个随机种子、一个微型种群不能支持显著性结论、全局不可训练结论或算法优劣排名。没有确认旧ES实现错误；这是检验算法/初始化是否影响结果的有限实测。三臂均为无梯度离线搜索，本轮没有检验替代梯度或其他梯度训练方法；共同的20维参数化和目标函数也可能限制结果，不能外推到全部训练算法。

物理前解析检查覆盖有界6维平移球与旋转各向异性二次函数、重复性、DE公式/延后选择/同分保留、CMA正定完整协方差与非零路径。DE在6维球也停滞：.1074797→.0386360；原草稿4倍改善门未过，冻结前改为2倍改善的实现冒烟门，明确不叫收敛；全过程0物理步，原失败和理由存{O}/PREFLIGHT_NOTES.json。解析测试通过不是实际身体训练成功。

实际物理{s['physical_seconds']:.6f}s，神经回放{s['neural_replay_networks']}套/{s['neural_replay_steps']}步。独立已存数组验证重算全部{v['runs']}条J/D、尖峰到率到命令、所有候选逐边倍率和物理交接、排名/开发选择及优化器轨迹。最大滤波误差{v['max_filter_error']:.3g}、命令误差{v['max_command_error']:.3g}，重复逐位一致，冻结输入不变。父代理另以独立ES/DE公式及直接调用CMA库核对全部9批，与执行包装分离，见{O}/PARENT_OPTIMIZER_VERIFICATION.json。生物有效性、运动意图、绝对力/SI均NOT_ESTABLISHED。

核心必需：{O}/PREREGISTRATION.json、{O}/SUMMARY.json、{O}/VERIFICATION.json、{O}/FILE_INVENTORY.json、{O}/MANIFEST_SHA256.json、{O}/CONFIGS、{O}/RUNS，以及{r.SRC}/run.py、{r.SRC}/optimizers.py、{r.SRC}/verify.py。重要辅助：本报告、{image}、设计审阅/预检/清理台账。新隔离环境依赖身份见{O}/DEPENDENCY_IDENTITY.json，旧环境只读。

实际清理：仅删除本轮构造器再生、已核实无读取用途的COVERAGE.json，路径/字节/哈希见{O}/CLEANUP_LEDGER.json；早期新建失败的Python3.14局部环境在用途确认后由venv --clear清除并重建为3.12。失败候选轨迹全部保留，未删除历史输入；无待确认文件。
'''
    (O/'REPORT.zh-CN.md').write_text(report);inventory()
if __name__=='__main__':main()
