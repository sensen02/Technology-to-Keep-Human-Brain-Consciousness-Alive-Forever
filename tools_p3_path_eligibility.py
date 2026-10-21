"""Bounded read-only P3 graph audit. No simulation, repairs, or historical imports."""
from __future__ import annotations
import argparse
from array import array
from collections import deque
from datetime import datetime
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
DEFAULT_P1 = ROOT / 'outputs/experiment_platform/20261013_233237_7be93a1d'
DESKTOP = Path('/home/sensen/Desktop/engine修复验收')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def adjacency(we, wi=None):
    """Input W[target,source]; output CSR[source,target], nonzero union only."""
    w = we.copy().tocsr()
    w.eliminate_zeros()
    w.data[:] = 1
    if wi is not None:
        other = wi.copy().tocsr()
        other.eliminate_zeros()
        other.data[:] = 1
        w = w.maximum(other)
    a = w.T.tocsr()
    a.sort_indices()
    return a


def search(a, sources, max_hops=None):
    """Deterministic multi-source directed BFS; bounded cutoff is not disconnection."""
    if max_hops is not None and max_hops < 0:
        raise ValueError('negative cutoff')
    n = a.shape[0]
    dist = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    q = deque()
    for s in sorted(set(map(int, sources))):
        if not 0 <= s < n:
            raise ValueError('source outside graph')
        dist[s] = 0
        q.append(s)
    cutoff_hit = False
    while q:
        s = q.popleft()
        targets = a.indices[a.indptr[s]:a.indptr[s + 1]]
        if max_hops is not None and dist[s] >= max_hops:
            cutoff_hit |= any(dist[t] < 0 for t in targets)
            continue
        for t in targets:
            if dist[t] < 0:
                dist[t] = dist[s] + 1
                parent[t] = s
                q.append(int(t))
    return dist, parent, cutoff_hit


def witness(parent, dist, target):
    if dist[target] < 0:
        return None
    path = [int(target)]
    while parent[path[-1]] >= 0:
        path.append(int(parent[path[-1]]))
    return path[::-1]


def run(args, memory_report):
    # Only current live construction modules. Never import source_archive/history.
    from engine.embodied.adapters import build_neural_tier, TierConfig
    from engine.neck_cut_data import select_touch_tier, TouchTierConfig
    p1file = args.p1 / 'result.json'
    p1 = json.loads(p1file.read_text())
    p3 = json.loads(args.p3.read_text())
    missing = sorted({int(r) for leg in p3['per_leg'] for row in leg['partial_rows']
                      for r in row['missing_roots']})
    if len(missing) != 6:
        raise ValueError('P3 must declare precisely six unique cropped roots')
    print('Building actual current E/I tier (no dynamics)', flush=True)
    built = build_neural_tier(TierConfig(background_mean_nA=0, background_sd_nA=0))
    roots = built['tier']['root_ids']
    if not np.array_equal(roots, np.asarray(p1['tier_root_ids'], dtype=np.int64)):
        raise ValueError('live tier differs from P1 ordered root identity')
    a = built['data']
    gi = built['global_index']
    loc = built['local_of_global']
    global_by_root = {int(r): i for i, r in enumerate(a['root_ids'])}
    local_by_root = {int(r): i for i, r in enumerate(roots)}
    rawpath = ROOT / 'data/flywire/banc_connections_princeton.csv.gz'
    rows = p1['motor_manifest']['rows']
    motors = {}
    for row in rows:
        if 'leg' not in row:
            continue
        motors.setdefault(row['leg'], set()).update(int(r) for r in row['pool_root_ids']
                                                    if int(r) in local_by_root)
    target_roots = sorted(set().union(*motors.values()))
    if len(target_roots) != 202:
        raise ValueError('expected exactly 202 mapped motor targets')
    targets = [local_by_root[r] for r in target_roots]
    with np.load(args.p1 / 'traces.npz', allow_pickle=False) as z:
        current = z['normal_sensory_current_nA']
        sensory_cols = z['sensory_local_indices']
        active = set(map(int, sensory_cols[np.any(current != 0, axis=0)]))
    groups = []
    for leg, channels in p1['sensory_groups'].items():
        for channel, indices in channels.items():
            groups.append((leg.upper() + '/' + channel, indices, bool(active.intersection(indices))))
    if len(groups) != 42 or sum(g[2] for g in groups) != 12:
        raise ValueError('expected 42 groups, 12 actually injected groups')
    groups.append(('descending_actual_readout', built['readout_local'].tolist(), None))
    groups.append(('descending_all_in_tier', built['descending_local'].tolist(), None))
    strict = adjacency(built['edges']['We'], built['edges']['Wi'])
    inside = built['tier']['inside_rows'] & (a['syn'] > 0)
    raw = sparse.csr_matrix((np.ones(int(inside.sum()), dtype=np.uint8),
                            (loc[a['pre'][inside]], loc[a['post'][inside]])),
                           shape=strict.shape, dtype=bool)
    raw.sort_indices()
    # Independently project verified-eligible cached rows and compare support to actual build.
    pure = np.isin(a['ann_nt_verified'], ['acetylcholine', 'gaba', 'glutamate'])
    eligible = inside & pure[a['pre']]
    projected = sparse.csr_matrix((np.ones(int(eligible.sum()), dtype=bool),
                                  (loc[a['pre'][eligible]], loc[a['post'][eligible]])), shape=strict.shape)
    if (projected != strict.astype(bool)).nnz:
        raise ValueError('strict support does not equal actual We/Wi nonzero support')
    def node(i):
        g = int(gi[i])
        return {'root_id': str(int(roots[i])), 'super_class': str(a['ann_super_class'][g]),
                'class': str(a['ann_class'][g]), 'verified_nt': str(a['ann_nt_verified'][g]) or None}
    searches = {}
    for graph_name, graph in [('strict_actual_nonzero_EI', strict), ('raw_cached_within_tier', raw)]:
        print('Searching ' + graph_name, flush=True)
        results = []
        for name, indices, was_active in groups:
            dist, parent, truncated = search(graph, indices, args.max_hops)
            reach = [r for r in target_roots if dist[local_by_root[r]] >= 0]
            paths = {}
            for r in reach:
                path = witness(parent, dist, local_by_root[r])
                # Assert each witness is a directed edge in this exact graph.
                assert all(graph[s, t] != 0 for s, t in zip(path, path[1:]))
                paths[str(r)] = {'hops': int(dist[local_by_root[r]]), 'nodes': [node(i) for i in path]}
            results.append({'group': name, 'source_count': len(indices), 'source_root_ids':
                            [str(int(roots[i])) for i in indices], 'normal_p1_nonzero_injection': was_active,
                            'actual_nonzero_source_count': len(active.intersection(indices)) if was_active is not None else None,
                            'reachable_motor_count': len(reach), 'per_leg_reachable':
                            {leg: len(set(reach) & rr) for leg, rr in motors.items()},
                            'unreached_target_root_ids': [str(r) for r in target_roots if r not in reach],
                            'unreached_meaning': 'not found within cutoff' if truncated else 'not reachable in this bounded graph',
                            'cutoff_hit': truncated, 'witnesses': paths})
        searches[graph_name] = results
    # Selection boundary: no extra growth; reconstruct first iteration of the same rule.
    mech = built['groups_global']  # Seed identity uses exact adapter class/nerve rule.
    from engine.neck_cut_data import mechanosensory_partition
    partition = mechanosensory_partition(a)
    seeds = partition['body_route'] & np.array(['leg_nerve' in str(s) for s in a['ann_nerve']])
    first = select_touch_tier(a, {'mechanosensory': partition['mechanosensory'],
                                'body_route': seeds, 'head_route': np.zeros(len(seeds), bool)},
                              built['cut'], TouchTierConfig(max_neurons=12000))
    seedmask = first['keep_mask']
    inflow = np.bincount(a['post'][seedmask[a['pre']]],
                        weights=a['syn'][seedmask[a['pre']]].astype(float), minlength=len(seeds))
    candidates = np.flatnonzero((inflow >= 10) & ~seedmask)
    order = candidates[np.lexsort((candidates, -inflow[candidates]))]
    ranks = {int(g): rank + 1 for rank, g in enumerate(order)}
    room = 12000 - int(seedmask.sum())
    keep = seedmask.copy(); keep[order[:room]] = True
    assert np.array_equal(keep, built['tier']['keep_mask'])
    six = {}
    for r in missing:
        g = global_by_root.get(r)
        assert r not in local_by_root
        six[str(r)] = {'global_index': g, 'present_in_cached_node_universe': g is not None,
                       'super_class': str(a['ann_super_class'][g]) if g is not None else None,
                       'class': str(a['ann_class'][g]) if g is not None else None,
                       'verified_nt': (str(a['ann_nt_verified'][g]) or None) if g is not None else None,
                       'nt_output_eligible': bool(pure[g]) if g is not None else False,
                       'in_first_iteration_seed_set': bool(seedmask[g]) if g is not None else False,
                       'second_iteration_seed_inflow_synapses': float(inflow[g]) if g is not None else None,
                       'candidate_rank_1based': ranks.get(g), 'ranked_partner_seats': room,
                       'selection_reason': ('absent from cached node universe (upstream ingestion boundary, not tier rank cropping)' if g is None else
                                            'below partner threshold' if g not in ranks else 'rank exceeds retained seats'),
                       'raw_csv_incoming_from_tier': [], 'raw_csv_outgoing_to_tier': [],
                       'raw_csv_all_incoming_rows': 0, 'raw_csv_all_outgoing_rows': 0}
    with gzip.open(ROOT / 'data/flywire/banc_neurons.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            if row['Root ID'] in six:
                entry = six[row['Root ID']]
                entry['original_annotation'] = row
                entry['original_verified_nt'] = row['Verified NT type'] or None
                entry['original_predicted_nt'] = row['Predicted NT type'] or None
                entry['original_super_class'] = row['Super Class']
                entry['original_class'] = row['Class']
                if entry['global_index'] is None:
                    entry['verified_nt'] = row['Verified NT type'] or None
                    entry['super_class'] = row['Super Class']
                    entry['class'] = row['Class']
                    entry['nt_output_eligible'] = row['Verified NT type'] in ['acetylcholine', 'gaba', 'glutamate']
    # One streaming pass over original connectivity, never load all-brain CSV in memory.
    # Dense arrays store only tier/tier rows. Six-root lists retain only tier counterparts.
    source, target = array('i'), array('i')
    started = time.monotonic(); scanned = 0; skipped = None
    print('Streaming original raw CSV once', flush=True)
    with gzip.open(rawpath, 'rt') as f:
        for row in csv.DictReader(f):
            scanned += 1
            if scanned > args.max_raw_rows or time.monotonic() - started > args.raw_seconds:
                skipped = 'original CSV scan resource cutoff; raw CSV verification and six boundaries partial'
                break
            s, t = int(row['pre_root_id']), int(row['post_root_id'])
            sl, tl = local_by_root.get(s), local_by_root.get(t)
            count = float(row['syn_count'])
            if sl is not None and tl is not None and count > 0:
                if len(source) >= args.max_relevant_rows:
                    skipped = 'relevant CSV row storage cutoff; CSV verification and six boundaries partial'
                    break
                source.append(sl); target.append(tl)
            if str(t) in six:
                six[str(t)]['raw_csv_all_incoming_rows'] += 1
                if sl is not None:
                    six[str(t)]['raw_csv_incoming_from_tier'].append({'source_root_id': str(s),
                        'synapses': count, 'neuropil': row['neuropil'], 'verified_nt': str(a['ann_nt_verified'][gi[sl]]) or None,
                        'strict_source_nt_eligible': bool(pure[gi[sl]])})
            if str(s) in six:
                six[str(s)]['raw_csv_all_outgoing_rows'] += 1
                if tl is not None:
                    six[str(s)]['raw_csv_outgoing_to_tier'].append({'target_root_id': str(t),
                        'synapses': count, 'neuropil': row['neuropil'], 'strict_source_nt_eligible': six[str(s)]['nt_output_eligible']})
    csv_graph = sparse.csr_matrix((np.ones(len(source), dtype=bool),
                                 (np.frombuffer(source, dtype=np.int32), np.frombuffer(target, dtype=np.int32))), shape=raw.shape)
    csv_difference = int((csv_graph != raw).nnz) if skipped is None else None
    for r, entry in six.items():
        for direction in ['incoming_from_tier', 'outgoing_to_tier']:
            ee = entry['raw_csv_' + direction]
            entry[direction + '_summary'] = {'rows': len(ee), 'synapses': sum(e['synapses'] for e in ee),
                'strict_nt_eligible_rows': sum(e['strict_source_nt_eligible'] and e['synapses'] > 0 for e in ee),
                'strict_nt_eligible_synapses': sum(e['synapses'] for e in ee if e['strict_source_nt_eligible'])}
        g = entry['global_index']
        for direction, mask in [('incoming_from_tier', (a['post'] == g) & built['tier']['keep_mask'][a['pre']]),
                                ('outgoing_to_tier', (a['pre'] == g) & built['tier']['keep_mask'][a['post']])]:
            entry['cached_' + direction] = {'rows': int(mask.sum()), 'synapses': float(a['syn'][mask].sum()),
                                             'node_present': g is not None}
        entry['boundary_complete'] = skipped is None
    for entry in six.values():
        entry['original_annotation_present'] = 'original_annotation' in entry
        entry.setdefault('original_annotation', None)
        entry.setdefault('original_verified_nt', None)
        entry.setdefault('original_predicted_nt', None)
        entry.setdefault('original_super_class', None)
        entry.setdefault('original_class', None)
    inputs = [p1file, args.p1 / 'traces.npz', args.p3, Path(a['path']), rawpath,
              ROOT / 'data/flywire/banc_neurons.csv.gz']
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values())
               if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(ROOT / 'engine')
               and str(m.__file__).endswith('.py')}
    sources.update([Path(__file__), ROOT / 'test_p3_path_eligibility.py'])
    out = ROOT / 'outputs/p3_path_eligibility' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(sources): z.write(f, str(f.relative_to(ROOT)))
    source_hashes = {str(f): sha(f) for f in sorted(sources)}
    p1_comparison = {f: {'current': h, 'p1': p1['source_hashes'].get(f),
                         'matches': h == p1['source_hashes'].get(f)}
                     for f, h in source_hashes.items() if f in p1['source_hashes']}
    if any(not v['matches'] for v in p1_comparison.values()):
        raise ValueError('live imported construction source differs from P1 snapshot')
    result = {'status': 'BOUNDED_READONLY_STRUCTURE_NOT_ACTIVATION', 'p1_ordered_roots_match': True,
        'strict_support_equals_actual_EI_nonzero': True, 'network_dynamics_run': False,
        'graphs': {'strict_nnz': int(strict.nnz), 'raw_cached_nnz': int(raw.nnz),
                   'raw_positive_rows': int(inside.sum()), 'csv_scan_rows': scanned,
                   'csv_relevant_positive_rows': len(source), 'csv_cached_support_symmetric_difference': csv_difference,
                   'csv_complete': skipped is None},
        'raw_definition': 'Unsigned cached positive within-tier connectivity before verified presynaptic NT filter. Includes biologically unknown edges; NOT a permissive repaired signed model.',
        'target_counts_per_leg': {leg: len(rr) for leg, rr in motors.items()},
        'normal_active_groups': 12, 'normal_zero_input_groups': 30, 'max_hops': args.max_hops,
        'searches': searches, 'six_cropped_roots': six, 'selection': built['coverage']['growth_history'],
        'input_sha256': {str(f): sha(f) for f in inputs}, 'source_sha256': source_hashes,
        'source_snapshot_sha256': sha(out / 'source_snapshot.zip'), 'p1_source_comparison': p1_comparison,
        'skipped_scope': [skipped] if skipped else [],
        'limitations': ['No all-brain path closure, no outside-tier intermediate paths, no edge replacement or sign prediction.',
                       'Unsigned directed reachability does not imply excitation, activation, muscle control, or biological validation.',
                       'Descending pooled reachability does not prove every descending neuron reaches each motor.',
                       'Verified NT is an annotation; target receptors and physiological signs remain unverified.'],
        'output_directory': str(out), 'memory_guard': memory_report()}
    # Compact new plot: source group x target leg, split by graph.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    legs = list(motors)
    fig, axes = plt.subplots(1, 2, figsize=(12, 13))
    for ax, (name, rr) in zip(axes, searches.items()):
        mat = np.array([[r['per_leg_reachable'][leg] for leg in legs] for r in rr])
        ax.imshow(mat, aspect='auto', vmin=0, vmax=max(map(len, motors.values())), cmap='viridis')
        ax.set_xticks(range(len(legs)), legs)
        ax.set_yticks(range(len(rr)), [r['group'] + (' *' if r['normal_p1_nonzero_injection'] else '') for r in rr], fontsize=7)
        for y in range(len(rr)):
            for x in range(len(legs)): ax.text(x, y, str(mat[y,x]), ha='center', va='center', fontsize=7, color='white')
        ax.set_title(name)
    fig.suptitle('P3 unsigned directed motor reachability (* = nonzero P1 current)\nStructure is NOT activation; raw includes unknown NT, NOT repaired signed edges')
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(out / 'reachability.png', dpi=160); plt.close(fig)
    result['memory_guard'] = memory_report()
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['# P3 有界有向路径资格审计', '', '手动实施/测试的研究原型，不是成品。未运行神经/物理动态、未改模型、P1或历史源。',
             '', '严格图精确取当前实际构建 We/Wi 非零支撑，W[target,source] 转为 source→target；独立缓存资格投影精确一致。P1 12000根顺序和导入构建源哈希核对通过。',
             f'严格图 {strict.nnz} 条唯一边；原始缓存tier图 {raw.nnz} 条唯一边。原始图包含递质生物学未知边，不是允许未知符号的修复模型。',
             f'原始CSV仅一次流式扫描 {scanned} 行，tier相关正边行 {len(source)}；完整={skipped is None}，与缓存支撑差={csv_difference}。',
             '', '42感觉群：12群P1正常臂实际非零注流、30群全零。每群及实际下降readout/全部tier下降集合分别搜索202目标，逐目标最短见证含根、class、super_class与NT。',
             '', '| 来源 | 正常注流 | 严格可达 | 原始可达 |', '|---|---|---:|---:|']
    for s, r in zip(searches['strict_actual_nonzero_EI'], searches['raw_cached_within_tier']):
        lines.append(f"|{s['group']}|{s['normal_p1_nonzero_injection']}|{s['reachable_motor_count']}|{r['reachable_motor_count']}|")
    lines += ['', '逐腿目标分母：' + json.dumps(result['target_counts_per_leg']), '', '## 六裁剪根（原始CSV，非新增模型）',
              '| 根 | verified NT | 第二次选择inflow/rank/seats | tier入边行/突触/资格行 | tier出边行/突触/资格行 |', '|---|---|---|---|---|']
    for r, e in six.items():
        inc, exc = e['incoming_from_tier_summary'], e['outgoing_to_tier_summary']
        lines.append(f"|{r}|{e['verified_nt']}|{e['second_iteration_seed_inflow_synapses']}/{e['candidate_rank_1based']}/{room}|{inc['rows']}/{inc['synapses']}/{inc['strict_nt_eligible_rows']}|{exc['rows']}/{exc['synapses']}/{exc['strict_nt_eligible_rows']}|")
    lines += ['', '重要边界修正：四根存在于缓存但第二次rank超过3394个partner席位；720575941434056663和720575941500775137不在缓存节点全集，且原始注释CSV未找到，完整原始连接CSV中出入行均零。后二者是上游身份/摄取边界，不可称为rank裁剪，更不可据此断言动物无细胞/无连接。', '', '完整原始注释（预测NT与verified分开；未找到者显式null）、逐边清单和缓存计数在JSON。边资格是突触前NT资格；缺根自身NT不决定其接收输入。选择规则先保留原seed，再按阈值10与(-inflow,global index)截取；没有把六根插入网络。',
              '', '结构可达≠激活，更不证明兴奋传播、肌肉招募或站立。全脑/跨tier外部中间节点闭包未执行；下降集合有路径不意味着每根都有路径，也不独立验证脑→下降→VNC链路。',
              '资源跳过范围：' + str(result['skipped_scope']),
              '', '下一最小动作：优先根据见证中未知NT/受体和六根输入资格核对真实证据；不为表现替换边或扩大网络。',
              '', f'项目证据目录：`{out}`', f'JSON：`{out / "result.json"}`', f'新图：`{out / "reachability.png"}`', f'源码快照：`{out / "source_snapshot.zip"}`']
    (out / 'report.zh-CN.md').write_text('\n'.join(lines) + '\n')
    DESKTOP.mkdir(parents=True, exist_ok=True)
    for src, dst in [('result.json', 'P3-path-eligibility.json'), ('report.zh-CN.md', 'P3-路径资格审计报告.md'), ('reachability.png', 'P3-path-eligibility.png')]:
        shutil.copyfile(out / src, DESKTOP / dst)
    print(json.dumps({'output': str(out), 'graphs': result['graphs'], 'memory': result['memory_guard']}, ensure_ascii=False), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--p1', type=Path, default=DEFAULT_P1)
    p.add_argument('--p3', type=Path, default=DESKTOP / 'P3-control-chain-gaps.json')
    p.add_argument('--max-hops', type=int, default=None)
    p.add_argument('--max-raw-rows', type=int, default=5000000)
    p.add_argument('--max-relevant-rows', type=int, default=1500000)
    p.add_argument('--raw-seconds', type=float, default=120)
    args = p.parse_args()
    if min(args.max_raw_rows, args.max_relevant_rows, args.raw_seconds) <= 0:
        p.error('positive raw resource limits required')
    from engine.memory_guard import guarded_run
    with guarded_run('p3_path_eligibility', need_mb=1600, cap_mb=4608) as report:
        run(args, report)

if __name__ == '__main__':
    main()
