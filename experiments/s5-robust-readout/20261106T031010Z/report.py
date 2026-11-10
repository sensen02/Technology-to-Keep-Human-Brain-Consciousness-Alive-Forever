"""Saved-data diagnostics, Chinese report, chart and complete inventory."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as r
O=r.O;SRC=r.SRC;dump=r.dump

def main():
    summary=r.js(O/'SUMMARY.json');rob=r.js(O/'ROBUSTNESS.json');base=r.arrays(O/'REPLAY_sigma0.0_seed0.npz');diag=[]
    for tau in r.TAUS:
        key=f'{tau:.3f}';record=dict(tau_s=tau,windows=[])
        for start in [0,100]:
            window=dict(start_command_index=start,perturbations=[])
            for sigma in [.001,.01]:
                for seed in [1,2,3]:
                    z=r.arrays(O/f'REPLAY_sigma{sigma}_seed{seed}.npz');metrics={}
                    for kind in ['cmd','raw','rates']:
                        a=z[kind+'_'+key][start:];b=base[kind+'_'+key][start:];diff=a-b
                        den=float(np.sqrt(np.mean(b*b)));ratio=float(np.sqrt(np.mean(diff*diff))/(den+1e-12))
                        dims=np.sqrt(np.mean(diff*diff,axis=0))/(np.sqrt(np.mean(b*b,axis=0))+1e-12)
                        metrics[kind]=dict(relative_rms=ratio,baseline_rms=den,per_dimension_relative_rms=dims,quantiles=np.quantile(dims,[0,.25,.5,.75,.9,1]))
                    window['perturbations'].append(dict(sigma=sigma,seed=seed,metrics=metrics))
            record['windows'].append(window)
        b=base['cmd_'+key][100:];ref=base['cmd_0.005'][100:];raw=base['raw_'+key][100:];rawref=base['raw_0.005'][100:]
        var=lambda a:float(np.sqrt(np.mean(np.var(a,axis=0))))
        record.update(command_temporal_std=var(b),command_std_ratio_to_5ms=var(b)/var(ref),raw_temporal_std=var(raw),raw_std_ratio_to_5ms=var(raw)/var(rawref),total_variation=float(np.mean(abs(np.diff(b,axis=0)))),baseline_saturation=float(np.mean(abs(b)>=np.array(r.js(O/'TRANSLATOR_IDENTITIES.json')['limits']))))
        diag.append(record)
    dump(O/'STABILITY_INFORMATION_TRADEOFF.json',dict(status='secondary saved-data diagnostic, not selection retuning',denominator='RMS of same-tau unperturbed signal, per channel for dimension distributions',windows='entire600ms and excluding first100ms; zero-input startup is part of replay, not warm physical release',rows=diag))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,8));colors=['black','tab:blue','tab:orange','tab:red']
    for i,tau in enumerate(r.TAUS):
        row=diag[i];x=tau*1000
        vals=[v['metrics']['cmd']['relative_rms'] for w in row['windows'] for v in w['perturbations']]
        rawvals=[v['metrics']['raw']['relative_rms'] for w in row['windows'] for v in w['perturbations']]
        ax[0,0].scatter([x]*len(vals),vals,color=colors[i],s=20);ax[0,0].scatter(x,max(rawvals),color=colors[i],marker='x',s=60)
        ax[0,1].scatter(x,row['command_std_ratio_to_5ms'],color=colors[i]);ax[0,1].scatter(x,row['raw_std_ratio_to_5ms'],color=colors[i],marker='x')
        res=next(v for v in summary['results'] if v['tau_s']==tau and v['release_time_s']==1.3 and '_dev' in v['label']);a=r.arrays(Path(res['folder'])/'NATIVE.npz')['rows'];t=a[:,0]-1.3
        ax[1,0].plot(t[::40],a[::40,1:7].sum(1)/24.446354755878005,color=colors[i],label=f'{x:g}ms, J={res["J"]:.3f}')
        ax[1,1].plot(t[::40],a[::40,10],color=colors[i],label=f'{x:g}ms')
    ax[0,0].axhline(.15,ls='--',color='gray');ax[0,0].set(title='Perturbation sensitivity: dots clipped, x raw worst',xlabel='Filter tau (ms)',ylabel='RMS difference / baseline RMS')
    ax[0,1].set(title='Temporal variation retained: dots clipped, x raw',xlabel='Filter tau (ms)',ylabel='Temporal std / 5ms temporal std')
    ax[1,0].set(title='Known 1.300s state: native foot load (plot decimated)',xlabel='Time after release (s)',ylabel='Foot load / W');ax[1,0].legend()
    ax[1,1].axhline(.8*1.2072528224791905,ls='--',color='gray');ax[1,1].set(title='Known 1.300s state: COM height',xlabel='Time after release (s)',ylabel='Model COM height');ax[1,1].legend()
    fig.suptitle('Bounded R: smoother readout is more robust, physical control regresses\nAll 23 physical releases fail full support gates; not walking')
    fig.tight_layout();fig.savefig(O/'STABILITY_CONTROL.png',dpi=160);plt.close(fig)
    lines=['# R 有界稳健读出实验：研究验证，未自主支撑/行走','',
    '**结论：50/100ms 读出通过预登记的强度扰动稳健门，但三个慢滤波候选均破坏已知较好交接态，故保留原5ms基线。测试的简单固定低通家族未兼顾稳健与物理控制；不能推广成所有R路线失败。**','',
    '本轮只用现有66个解剖运动池和原有符号矩阵；5ms被20/50/100ms直接替换，没有保留快速读出作为残差底座，没有拟合、身体状态/时刻/相位输入或撤辅后PD。SNN拓扑/强度沿用基线（包括释放时RH入兴奋0.375），身体、执行器限幅和D1–D8未变。运动204细胞的预热尖峰用于因果重建滤波记忆，释放时不清零。', '',
    '| tau | 最大命令相对RMS | 稳健门≤0.15 | 命令时间std/5ms | 1.3s J | 最差J | 保护门 |','|---|---:|---|---:|---:|---:|---|']
    for rank,d in zip(summary['ranking'],diag):
        res=next(v for v in summary['results'] if v['tau_s']==rank['tau'] and v['release_time_s']==1.3 and '_dev' in v['label'])
        lines.append(f"| {rank['tau']*1000:g}ms | {rank['max_relative_rms']:.6f} | {rank['robust']} | {d['command_std_ratio_to_5ms']:.6f} | {res['J']:.9f} | {rank['worst_J']:.9f} | {rank['guard']} |")
    lines += ['', '稳健量为六组逐边强度扰动（0.1%/1%、种子1/2/3）相对同tau无扰动命令的真正RMS比值；对完整0.6s和去掉首100ms两窗取最大。实际扰动是log-normal sigma，非所有边精确同一偏差。输入片段与前轮fragility相同，但本轮施加RH0.375、前轮脚本未施，因此不将数值视为前轮原样复现。部署物理实验不施加这些噪声。', '',
    '稳定性与控制权衡：平滑显著减少扰动差异，也减少时间变化；STABILITY_INFORMATION_TRADEOFF.json额外保留未截断力矩、66池、42关节逐维RMS及分位，避免限幅掩盖敏感性。这是该固定低通的实测权衡，不是控制信息不存在的证明；时间幅度、延迟及实际闭环轨迹同时变化，不能单独归因为延迟。', '',
    '| tau | 1.3s 足载/W | 最低COM | 首次非足正力(s) |','|---|---:|---:|---:|']
    for tau in r.TAUS:
        res=next(v for v in summary['results'] if v['tau_s']==tau and v['release_time_s']==1.3 and '_dev' in v['label']);p=res['parts']
        lines.append(f"| {tau*1000:g}ms | {p['foot_support_mean']:.9f} | {p['com_z_min']:.9f} | {p['first_nonfoot_s']:.7f} |")
    ver=r.js(O/'VERIFICATION.json')
    motor=next(x['J'] for x in summary['results'] if x['label']=='selected_motor_cut')
    sensory=next(x['J'] for x in summary['results'] if x['label']=='selected_sensory_cut')
    lines += ['', f"实际运行：7个回放网络×1200步=8400神经诊断步；4tau×5状态=20次释放，加选中基线重复/运动切断/感觉切断3次，总23次、4.6s释放；无新预热、无训练。五状态为1.225/1.275/1.300/1.325/1.375s。基线重复BODY/NATIVE/NEURAL逐位相同，运动切断J{motor:.9f}、感觉切断J{sensory:.9f}（精确数值见{O/'SUMMARY.json'}）。感觉切断保留释放前神经与滤波记忆，不能解释为从零状态无感觉。", '',
    '全部23次完整D门失败。50/100ms虽然过稳健门仍未过回归保护；20ms两门都失败。没有候选胜出，按预登记跳过四个终测状态；这些1.212/1.262/1.312/1.362s本已在历史试验使用，不能称新或未见状态。未追加视频、不声称已肉眼检查图像；图只来自真实保存数组，非行走演示。', '',
    f"独立验证已运行退出0：使用scipy.signal.lfilter从原始预热和释放尖峰复算，最大率误差{ver['max_filter_error']:.3g}、命令误差{ver['max_command_error']:.3g}、J误差{ver['max_J_error']:.3g}；逐项重新展开D门和J，核对12.5µs原生采样、质量/重力常数、正/带符号非足首次时刻一致、交接qpos/qvel连续、全部辅助/外力0、警告0。源文件前后哈希不变；父代理独立检查历史1235条哈希通过。", '',
    '边界：一个真实感觉片段、三种扰动种子、五已知物理状态、0.2s释放；没有稳定基线，切断只作链路诊断。当前为全身42理想关节力矩执行器plant，不能写成已验证的肌肉动力学或生物行走。绝对力值/SI与意图保持仍NOT_ESTABLISHED。未用不合格预热步态训练。', '',
    '下一步应检验保留功能方向的稳健聚合是否还能保留必要控制变化，或先修复感觉/真实预热目标；本轮不授权或执行S强度拟合。', '',
    '清理：实际删除构造器再生、确认当前路径无读者的COVERAGE.json一项，40247字节，删除前路径/哈希/字节数见CLEANUP_LEDGER.json；其余失败物理轨迹全部保留，无旧目录写入。', '',
    '文件分级与全路径见FILE_INVENTORY.json，所有保留产物含源码以MANIFEST_SHA256.json核验。主体数据/预登记/独立验证/源码为核心必需；本报告、图、方案评审与补充诊断为重要辅助。']
    lines += ['', '未截断诊断（两窗六扰动最大相对RMS）与去掉首100ms后的回放限幅比例：']
    for d in diag:
        rawmax=max(v['metrics']['raw']['relative_rms'] for w in d['windows'] for v in w['perturbations'])
        lines.append(f"- {d['tau_s']*1000:g}ms：未截断RMS {rawmax:.6f}，时间std比 {d['raw_std_ratio_to_5ms']:.6f}，限幅比例 {d['baseline_saturation']:.6f}。")
    lines += ['', '慢滤波下仍有大面积限幅，因此不能声称获得了非饱和的分级控制。', '', '执行环境：`/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-translator/20261105T200021Z/.venv/bin/python -B`；PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl MALLOC_ARENA_MAX=2；唯一重进程串行，prereg/run/verify/report均exit0，无失败仿真起点。']
    text='\n'.join(lines)+'\n'
    for name in ['STABILITY_INFORMATION_TRADEOFF.json','CLEANUP_LEDGER.json','FILE_INVENTORY.json','MANIFEST_SHA256.json']:
        text=text.replace(name,str(O/name))
    (O/'REPORT.zh-CN.md').write_text(text)
    before=r.js(O/'INPUT_HASHES_BEFORE.json')
    dump(O/'CHANGE_LEDGER.json',dict(source_changes=[dict(old_file=str(Path(r.rt.__file__)),old_sha256=before[str(Path(r.rt.__file__))],new_file=str(SRC/'run.py'),new_sha256=r.sha(SRC/'run.py'),reason='new subclass replaces pool rate recurrence; original source unmodified')],parameters=[dict(old_tau_s=.005,new_tau_s=tau,new_file=str(O/'CONFIGS'/f'tau_{tau:.3f}'/'PARAMETERS.json'),new_sha256=r.sha(O/'CONFIGS'/f'tau_{tau:.3f}'/'PARAMETERS.json'),reason='fixed causal lowpass replacement, no fit') for tau in r.TAUS],unchanged='SNN matrices, RH .375, sensory encoder, signed anatomical matrix, body and actuator limits',source_hashes_before=str(O/'INPUT_HASHES_BEFORE.json'),source_hashes_after=str(O/'INPUT_HASHES_AFTER.json')))
    files=sorted([p for folder in [O,SRC] for p in folder.rglob('*') if p.is_file() and p.name not in ['FILE_INVENTORY.json','MANIFEST_SHA256.json','MANIFEST_VERIFICATION.json']])
    inv=[]
    for p in files:
        auxiliary=p.suffix in ['.md','.png'] or p.name in ['CHANGE_LEDGER.json','CLEANUP_LEDGER.json','STABILITY_INFORMATION_TRADEOFF.json','HISTORICAL_INTEGRITY.json']
        inv.append(dict(path=str(p),importance='重要辅助' if auxiliary else '核心必需',purpose='报告/诊断/溯源辅助' if auxiliary else '重现、参数、原始轨迹或验证证据',bytes=p.stat().st_size,sha256=r.sha(p)))
    dump(O/'FILE_INVENTORY.json',dict(files=inv,inventory_itself='核心必需；保留文件分类索引',manifest_itself='核心必需；哈希验证入口',verification_itself='重要辅助；manifest实际核验结果',pending_confirmation=[]))
    paths=files+[O/'FILE_INVENTORY.json'];manifest={str(p):r.sha(p) for p in paths};dump(O/'MANIFEST_SHA256.json',manifest)
    errors=[p for p,h in manifest.items() if r.sha(p)!=h];assert not errors
    dump(O/'MANIFEST_VERIFICATION.json',dict(entries=len(manifest),mismatches=errors,passed=True,manifest_sha256=r.sha(O/'MANIFEST_SHA256.json'),self_hash_excluded=True))
    print('REPORT MANIFEST',len(manifest),flush=True)
if __name__=='__main__':main()
