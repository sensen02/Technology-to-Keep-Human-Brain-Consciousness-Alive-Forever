"""Post-run presentation only; not part of frozen simulation implementation."""
import numpy as np
import run as r
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    s=r.js(r.O/'SUMMARY.json');v=r.js(r.O/'VERIFICATION.json');tau=r.js(r.O/'TAU_DIAGNOSTIC.json')
    fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    keys=['pool_rates','raw_command','clipped_command']
    for i,m in enumerate(r.METHODS):
        for j,k in enumerate(keys):
            ax=axs[i,j]
            for stage,style in [('main','o--'),('supplement','s-')]:
                if m not in s[stage]:continue
                result=s[stage][m];x=list(map(int,result['windows']));y=[w['metrics'][k]['fine']['normalized_RMS'] for w in result['windows'].values()]
                ax.plot(x,y,style,label=str(result['dts_ms'][-2:])+' ms')
            ax.axhline(.15,color='gray',ls=':',label='nRMS threshold only');ax.set(xscale='log',xlabel='Window from identical state (ms)',ylabel='difference RMS / fine RMS',title=m+' / '+k);ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Identical-state local refinement; original 5 ms decoder; no physical runs')
    fig.savefig(r.O/'LOCAL_REFINEMENT.png',dpi=150);plt.close(fig)
    lines=['# 同一暖态神经局部细化诊断','', '本轮为手工研究诊断，非成品；不改变已封第一阶段，不代表物理行走完成。', '',f"实际 {len(s['results'])} 个神经病例，共 {s['neural_steps']} 步；0 新暖态、0 物理运行。原0.5ms基线全部12000细胞尖峰、池率、原始及限幅命令逐位复现。",'', '全部病例由相同0.5ms、物理1.3s暖态（神经时钟1100ms）开始；v/ge/gi/adaptation/refractory_until/spike_count及未到达事件的绝对时间一致。暖态原连接未缩放，转移后RH入兴奋仅乘0.375一次。输入为同一200×12000实际float64电流，每1ms保持；命令用前一完成区间的池率。原5ms滤波保持不变。', '', '## 电参数与有效时间常数','', 'C=0.02 nF，gleak=0.001 µS；nF/µS=ms，静息tau=20ms。以下包含每档17暖态×12000细胞的快照；不是全预热时序分布。另保存加上当前环槽到达事件后的tau，因为事件在电位积分之前加入。参数是未生理标定的示例点模型。','', '|dt ms|tau min ms|tau中位 ms|tau p95 ms|dt/tau中位|dt/tau p95|dt/tau max|','|---|---|---|---|---|---|---|']
    for row in tau['records']:
        if not row.get('all17'):continue
        t=row['tau_ms']['quantiles'];q=row['dt_over_tau']['quantiles'];lines.append(f"|{row['dt_ms']}|{t['min']:.6g}|{t['p50']:.6g}|{t['p95']:.6g}|{q['p50']:.6g}|{q['p95']:.6g}|{q['max']:.6g}|")
    lines+=['','## 局部收敛：短窗与长窗分开','', '工程门：每个窗口的pool/raw/clipped细差nRMS≤0.15，且细差/粗差≤0.9或两差均≤0.02。全12000细胞1ms计数与每窗累积计数单列诊断，不用于替换门；20ms非重叠聚合只作统计，未改实际解码器。nRMS分母是细网格参考RMS，绝对RMS/参考RMS亦全量保存。', '', '|方法/三网格ms|窗口ms|pool nRMS /收缩|raw nRMS /收缩|clip nRMS /收缩|三门同时过|','|---|---|---|---|---|---|']
    for m in r.METHODS:
        for stage in ['main','supplement']:
            if m not in s[stage]:continue
            row=s[stage][m]
            for w,result in row['windows'].items():
                cells=[f"{result['metrics'][k]['fine']['normalized_RMS']:.6f} / {result['metrics'][k]['contraction']:.6f}" for k in keys]
                lines.append('|'+m+' '+str(row['dts_ms'])+'|'+w+'|'+'|'.join(cells)+'|'+str(result['passed'])+'|')
    latest={m:s['supplement'].get(m,s['main'][m]) for m in r.METHODS}
    early={m:all(item['windows'][str(w)]['passed'] for w in [5,10]) for m,item in latest.items()}
    full={m:item['windows']['200']['passed'] for m,item in latest.items()}
    lines+=['','## 解释与下一步','',f'最细三网格首5/10ms同时通过：{early}；完整200ms通过：{full}。','', '从完全相同状态出发消除了不同长暖态的初态差异，但局部窗口仍包含阈值/复位/延迟量化及复发网络放大，不能把差值全部当作平滑ODE局部截断误差。首5/10ms与50–200ms的差异用于区分初始离散误差和后续轨迹放大；不能据长窗失败宣布脑无效。', '', 'frozen_exact仅对步内冻结电导/适应电流的电位方程精确；电导、适应衰减仍分裂，阈值/复位仍在步端。两个方法在匹配网格的全窗口差值见SUMMARY.json，不能将名称当作全ODE真解。','']
    if any(full.values()):
        recommendation='局部候选存在通过完整200ms门的方法；只可推荐进入新预登记的全暖态细网格稳健性验证，再决定是否做配对物理释放，不能由本局部结果直接采用新积分器。'
    else:
        recommendation='尚不能采用更好积分器为已收敛模型。建议进入数值稳健性分支：先固定同状态、同电流检查事件/阈值时间处理与步内电导变化的积分误差，并用更严格参考方法验证；本轮不启动额外全暖态或物理运行。仅靠继续缩小长暖态dt或替换frozen_exact，尚无足够证据保证原5ms命令稳健。'
    lines += [recommendation,'',f"独立验证通过={v['passed']}，包含scipy lfilter重构、因果命令、全网计数、适应方程、状态转移、绝对事件时刻、RH一次、时钟和所有窗口工程门重算。",'', '原始NPZ保存全12000 packed尖峰、每1ms全状态与全细胞计数、精确输入、时间以及原始/限幅命令；初末完整网络pickle保留pending ring。所有封存输入哈希前后相同。', '', '文件分级见FILE_INVENTORY.json。源码/原始数组/身份/预登记/验证为核心必需；图表与报告为重要辅助。无构造器临时产物、无实际删除文件；失败文件如有一律保留。绝对力/SI与生物有效性NOT_ESTABLISHED，未宣告总目标完成。', '', '已实际调用read_image，但当前模型无图像输入能力而被拒绝；PIL完整解码2250×1200 RGBA通过，未完成目视核验。详见FINAL_AUDIT.json。全12000细胞每窗累计数/Hz及相邻网格绝对RMS/参考RMS见WINDOW_CELL_COUNTS_RATES.npz与COUNT_RATE_DIAGNOSTICS.json。']
    (r.O/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    r.dump(r.O/'RECOMMENDATION.json',dict(early5and10ms=early,full200ms=full,recommendation=recommendation,adopted_method=None))
    r.dump(r.O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='No disposable constructor or temporary artifacts created; all run evidence retained.'))
    inventory={};hashes={}
    for root in [r.O,r.SRC]:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.name in ['FILE_INVENTORY.json','MANIFEST_SHA256.json']:continue
            inventory[str(p)]=dict(importance='重要辅助' if p.suffix in ['.png','.md'] or p.name in ['RUN.log','report.py'] else '核心必需',purpose='Presentation/diagnosis' if p.suffix in ['.png','.md'] else 'Reproduction, raw evidence or verification',bytes=p.stat().st_size);hashes[str(p)]=r.sha(p)
    r.dump(r.O/'FILE_INVENTORY.json',inventory);r.dump(r.O/'MANIFEST_SHA256.json',hashes)
    print('REPORTED',recommendation,flush=True)
if __name__=='__main__':main()
