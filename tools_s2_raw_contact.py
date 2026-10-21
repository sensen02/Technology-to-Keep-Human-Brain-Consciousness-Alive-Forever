"""Offline descriptive raw-contact windows, without cross-clock constraint matching."""
from pathlib import Path
import argparse
import json
import numpy as np


def distribution(v):
    return dict(min=float(np.min(v)), median=float(np.median(v)), max=float(np.max(v))) if len(v) else None


def histogram(v):
    keys, counts = np.unique(v, return_counts=True)
    return {str(int(k)): int(c) for k, c in zip(keys, counts)}


def point_summary(a, i, pair, cone):
    lo, hi = a['contact_offsets'][i:i+2]; elo, ehi = a['efc_offsets'][i:i+2]
    ids = [j for j in range(lo, hi) if tuple(sorted(a['contact_geom'][j])) == pair]
    rr = []
    for j in ids:
        address = int(a['contact_efc_address'][j]); dim = int(a['contact_dim'][j])
        size = 1 if dim == 1 else 2*(dim-1) if cone == 0 else dim
        if address >= 0 and not a['contact_exclude'][j]:
            assert address + size <= ehi-elo
            rr.extend(range(elo+address, elo+address+size))
    return dict(points=len(ids), distance=distribution(a['contact_dist'][ids]), local_normal=distribution(a['contact_local6'][ids,0]), local_force_norm=distribution(np.linalg.norm(a['contact_local6'][ids,:3],axis=1)), excluded=histogram(a['contact_exclude'][ids]), dimensions=histogram(a['contact_dim'][ids]), constraint_state=histogram(a['efc_state'][rr]), constraint_type=histogram(a['efc_type'][rr]), constraint_id=histogram(a['efc_id'][rr]))


def analyze(out):
    out = Path(out); manifest = json.loads((out/'manifest.json').read_text())
    datasets = [np.load(out/r['npz'],allow_pickle=False) for r in manifest['records']]
    pairs = [json.loads((out/r['pair_json']).read_text()) for r in manifest['records']]
    times = datasets[0]['time_s']; targets = ((0,26),(0,25)); series = {}
    for p in targets:
        key = f'{p[0]},{p[1]}'
        series[key] = [float(np.linalg.norm(np.asarray(pairs[0][i].get(key,[0,0,0]))-np.asarray(pairs[1][i].get(key,[0,0,0])))) for i in range(400)]
    windows = []
    for start, end in ((1.5,2.3),(7.7,8.7)):
        selected = np.flatnonzero((times*1000 >= start-1e-9)&(times*1000 <= end+1e-9))
        result = dict(window_ms=[start,end], pairs={})
        for p in targets:
            key = f'{p[0]},{p[1]}'; peak = int(selected[np.argmax(np.asarray(series[key])[selected])]); endpoints = []
            for i in selected:
                descriptions = []
                for a,r in zip(datasets,manifest['records']):
                    e0,e1 = a['efc_offsets'][i:i+2]
                    descriptions.append(dict(ncon=int(a['ncon'][i]),nefc=int(a['nefc'][i]),solver_niter=a['solver_niter'][i].tolist(), efc_state_histogram=histogram(a['efc_state'][e0:e1]), pair=point_summary(a,i,p,r['cone'])))
                endpoints.append(dict(index=int(i),time_ms=float(times[i]*1000),difference=series[key][i],clocks=descriptions))
            neighborhood=[]
            for i in range(max(0,peak-2),min(400,peak+3)):
                clocks=[]
                for a,r in zip(datasets,manifest['records']):
                    current=point_summary(a,i,p,r['cone']); previous=point_summary(a,i-1,p,r['cone']) if i else None
                    e0,e1=a['efc_offsets'][i:i+2]
                    clocks.append(dict(ncon=int(a['ncon'][i]),nefc=int(a['nefc'][i]),pair=current,pair_state_histogram_changed_from_previous=(previous is not None and previous['constraint_state'] != current['constraint_state']), all_state_histogram=histogram(a['efc_state'][e0:e1])))
                neighborhood.append(dict(index=i,time_ms=float(times[i]*1000),clocks=clocks))
            result['pairs'][key]=dict(peak_index=peak,peak_time_ms=float(times[peak]*1000),peak_difference=series[key][peak],endpoints=endpoints,neighborhood=neighborhood)
        windows.append(result)
    report=dict(windows=windows,force_difference_series=series,physical_cause='unknown',interpretation='Counts/distances/state histograms are associations only; efc rows are endpoint-local, not matched across clocks or time.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# S2原始接触观测报告','', '手动研究实施，不是成品。严格固定 native_floor pulse target0 两时钟，同初态10ms，未搜索/修改生产XML、tau或pose。每共同25us端点原始contact和efc数值档案，逐案例写盘；无pickle。','', '力/距离均模型单位，未换SI。efc行只在端点内联系contact，未跨时钟匹配约束行。状态直方图变化不是已追踪同一接触的状态转移。算术一致性不是身体收敛。','']
    for w in windows:
        lines.append(f"## 窗口 {w['window_ms']} ms")
        for key,r in w['pairs'].items():
            lines.append(f"- pair {key} 差峰 {r['peak_time_ms']:.6f}ms，世界合力欧氏差 {r['peak_difference']:.12g} model_force。")
            for clock in range(2):
                lines.append(f"  - dt {manifest['records'][clock]['dt_s']*1e6:g}us 峰±2端点：")
                for n in r['neighborhood']:
                    c=n['clocks'][clock]; s=c['pair']
                    lines.append(f"    - {n['time_ms']:.3f}ms ncon={c['ncon']} nefc={c['nefc']} pair点数={s['points']} dist={s['distance']} normal={s['local_normal']} state={s['constraint_state']} histogram变化={c['pair_state_histogram_changed_from_previous']}")
    lines += ['', '## 结论边界','观察可指出峰附近点数、距离和状态直方图是否变化，不能将关联命名为物理原因。physical_cause=unknown；未做干预、冲量/功平衡或新动力学重放。原始local6含力矩，当前pair合力只重建前三力分量。未生物标定、身体收敛、站立行走验收。','', '## 证据完整路径']
    lines += [str(out/n) for n in ('manifest.json','report.json','raw-contact.png')]+[str(out/r['npz']) for r in manifest['records']]
    (out/'原始接触观测报告.zh-CN.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(13,11))
    for col,(start,end) in enumerate(((1.5,2.3),(7.7,8.7))):
        for p in targets:
            key=f'{p[0]},{p[1]}'; axes[0,col].plot(times*1000,series[key],label=key)
            for a,r in zip(datasets,manifest['records']):
                s=[point_summary(a,i,p,r['cone']) for i in range(400)]
                label=key+f" {r['dt_s']*1e6:g}us"
                axes[1,col].plot(times*1000,[x['points'] for x in s],label=label)
                axes[2,col].plot(times*1000,[x['distance']['min'] if x['distance'] else np.nan for x in s],label=label)
        for a,r in zip(datasets,manifest['records']): axes[3,col].plot(times*1000,a['ncon'],label=f"ncon {r['dt_s']*1e6:g}us")
        for row,label in enumerate(('World pair difference (model force)','Pair contact points','Minimum distance (model length)','All contacts')):
            axes[row,col].set_xlim(start,end); axes[row,col].set_ylabel(label); axes[row,col].legend(fontsize=7);axes[row,col].grid(alpha=.2)
        axes[3,col].set_xlabel('Time (ms)')
    fig.suptitle('Fixed raw-contact observations: association NOT physical cause / convergence')
    fig.tight_layout();fig.savefig(out/'raw-contact.png',dpi=150);plt.close(fig)
    for a in datasets: a.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);analyze(parser.parse_args().out)
