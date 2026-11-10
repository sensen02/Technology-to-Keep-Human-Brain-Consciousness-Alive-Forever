"""Post-search saved-array diagnostics only, no source or experiment changes."""
import sys,pickle
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as r
O=r.O
s=r.js(O/'SUMMARY.json');rows=[]
for metric in s['candidate_metrics']:
    name=metric['candidate'];rr=[v for v in s['results'] if v['candidate']==name and '_train_guard_' in v['label']];guard=next(v for v in rr if v['release_time_s']==1.3)
    with (O/'CONFIGS'/name/'WARM_CACHE.pkl').open('rb') as f:cache=pickle.load(f)
    net=cache[1.4]['net'];warm_rate=float(net.spike_count.sum()/12000/1.2)
    zs=[r.arrays(Path(v['folder'])/'NEURAL.npz') for v in rr];sp=sum(z['spikes204'].sum() for z in zs);duration=sum(v['duration_s'] for v in rr);rate=float(sp/204/duration)
    cmds=np.concatenate([z['applied'] for z in zs]);std=float(np.sqrt(np.mean(np.var(cmds,axis=0))));tv=float(np.mean(np.concatenate([np.abs(np.diff(z['applied'],axis=0)) for z in zs])))
    rows.append(dict(candidate=name,guard_pass=metric['guard'],train_worst_J=metric['worst_J'],train_mean_J=metric['mean_J'],guard_J=guard['J'],guard_foot_mean=guard['parts']['foot_support_mean'],guard_com_min=guard['parts']['com_z_min'],guard_first_positive_nonfoot_s=guard['first_positive_nonfoot_s'],warm_all12000_mean_rate_hz=warm_rate,release_logged204_mean_rate_hz=rate,release_saturation_mean=float(np.mean([v['saturation_fraction'] for v in rr])),release_command_temporal_std_rms=std,release_command_total_variation_mean=tv))
parameters=np.array([r.js(O/'CONFIGS'/v['candidate']/'PARAMETERS.json')['multipliers'] for v in s['candidate_metrics'] if v['candidate']!='baseline'])
coverage=dict(actual_multiplier_min=float(parameters.min()),actual_multiplier_max=float(parameters.max()),dimensions_exercised=int(np.count_nonzero(np.ptp(parameters,axis=0)>0)),per_dimension_min=parameters.min(axis=0),per_dimension_max=parameters.max(axis=0),unique_candidates=len(np.unique(parameters,axis=0)))
r.dump(O/'SEARCH_PARAMETER_COVERAGE.json',coverage)
r.dump(O/'NEURAL_ACTION_DIAGNOSTICS.json',dict(scope='all12000 warm rate from full spike_count at1.4 over1.2s active neural prewarm; release rate only logged204motor cells across train+guard; std includes between-state mean variation, total variation within each episode only',rows=rows))
text='# 候选比较与神经/动作诊断\n\n只读已保存数组，无追加仿真。预热率为全部12000细胞在0.2—1.4s累计放电；释放率仅204个记录的运动细胞，不能称全网释放率。命令时间标准差跨五状态合并，包含状态间均值差；总变差只计算每条轨迹内相邻指令。\n\n'
text+='|候选|保护门|训练最差J|1.3 J|足载/W|最低COM|首次正非足/ms|预热全网Hz|释放204Hz|饱和比例|命令std|总变差|\n|---|---|---|---|---|---|---|---|---|---|---|---|\n'
for v in rows:
    first='未观察到' if v['guard_first_positive_nonfoot_s'] is None else f"{v['guard_first_positive_nonfoot_s']*1000:.3f}"
    text+=f"|{v['candidate']}|{v['guard_pass']}|{v['train_worst_J']:.6f}|{v['guard_J']:.6f}|{v['guard_foot_mean']:.6f}|{v['guard_com_min']:.6f}|{first}|{v['warm_all12000_mean_rate_hz']:.3f}|{v['release_logged204_mean_rate_hz']:.3f}|{v['release_saturation_mean']:.4f}|{v['release_command_temporal_std_rms']:.4f}|{v['release_command_total_variation_mean']:.4f}|\n"
text+=f"\n实际24个非基线候选倍率范围 {coverage['actual_multiplier_min']:.9f}—{coverage['actual_multiplier_max']:.9f}；覆盖变化维度{coverage['dimensions_exercised']}/20，唯一候选{coverage['unique_candidates']}。允许范围0.9—1.1不表示所有边或每维都测到端点。\n"
text+='\n所有这些指标均为工程诊断；低饱和、平均率或短窗J改善单独不证明协调动作、运动意图或生物标定。图排序分组共有257194条跨组边，是粗结构块而非强社区。\n'
(O/'CANDIDATE_COMPARISON.zh-CN.md').write_text(text)
print('DIAGNOSTICS',len(rows),flush=True)
