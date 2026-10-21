"""Offline event localization only. No MjData/forward/step or physical-cause claim."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/sensen/Desktop/engine修复验收/S2-contact-clock-protocol-complete-20261014-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(coarse, fine):
    """Missing pairs zero-filled for arithmetic, presence retained separately."""
    rows = []
    for pair in sorted(set(coarse) | set(fine)):
        a = np.asarray(coarse.get(pair, [0., 0., 0.]), dtype=float)
        b = np.asarray(fine.get(pair, [0., 0., 0.]), dtype=float)
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
            raise ValueError('invalid world force')
        an, bn, dn = map(float, (np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(a-b)))
        denom = max(an, bn)
        rows.append(dict(pair=pair, coarse_present=pair in coarse, fine_present=pair in fine,
                         coarse=a.tolist(), fine=b.tolist(), delta=(a-b).tolist(),
                         coarse_norm=an, fine_norm=bn, delta_norm=dn,
                         relative_denominator=denom, norm_relative=None if denom == 0 else dn/denom))
    rows.sort(key=lambda x: (-x['delta_norm'], x['pair']))
    ca = np.sum([r['coarse'] for r in rows], axis=0) if rows else np.zeros(3)
    fb = np.sum([r['fine'] for r in rows], axis=0) if rows else np.zeros(3)
    return dict(pairs=rows, pairmax=rows[0]['delta_norm'] if rows else 0.,
                sum_pair_delta_norms=sum(r['delta_norm'] for r in rows),
                coarse_sum_vector=ca.tolist(), fine_sum_vector=fb.tolist(),
                sum_vector_delta=(ca-fb).tolist(), sum_vector_delta_norm=float(np.linalg.norm(ca-fb)))


def select_peak(times, values, lower, upper):
    ids = np.flatnonzero((times >= lower) & (times <= upper))
    if not len(ids):
        raise ValueError('empty event window')
    return int(ids[np.argmax(np.asarray(values)[ids])])


def state_row(a, b, i, labels, field):
    x, y = a[field][i], b[field][i]
    delta = x-y
    top = sorted(range(len(delta)), key=lambda k: (-abs(delta[k]), k))[:5]
    return dict(coarse=x.tolist(), fine=y.tolist(), delta=delta.tolist(),
                top_absolute_differences=[dict(labels[k], coarse=float(x[k]), fine=float(y[k]),
                                               delta=float(delta[k])) for k in top])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=Path('/home/sensen/Desktop/engine修复验收') /
                        ('S2-contact-peak-audit-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]))
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    report = json.loads((source/'report.json').read_text())
    files = [source/'report.json'] + [source/r[k] for r in report['records'] for k in ('npz', 'contacts')]
    if len(files) != 17 or len(set(files)) != 17:
        raise ValueError('expected one report and eight archive/ledger pairs')
    before = {str(p): sha(p) for p in files}
    provenance_before = {p: sha(p) for p in report['source_hashes']}
    assert provenance_before == report['source_hashes']
    import mujoco
    import engine.canonical_physics as cp
    model = cp.load_model()  # Names only; never create MjData or evaluate dynamics.
    geom_names = {}
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        geom_names[g] = dict(geom_id=g, geom=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
                             body_id=body, body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body))
    output = dict(source_directory=str(source), output_directory=str(out), physical_cause='unknown',
                  force_unit='model force units', body_convergence_proven=False, biological_validation=False,
                  sum_vector_caveat='Cancellation across pairs; second geoms differ; not net external/support force.',
                  environment=dict(python=sys.executable, mujoco=mujoco.__version__), comparisons=[])
    lines = ['# S2/P2 接触力差峰离线定位报告', '',
             '手动研究实施，非成品。仅加载规范模型命名；没有身体重跑/forward/step。',
             'delta=coarse-fine；力全部为模型力单位，relative分母=max(两侧范数)，零分母为undefined。',
             'pairmax与sum-vector不可互换：后者跨pair可相消，且排序第二geom不同，不是系统净外力或支撑力。', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(13, 12))
    for row_index, comparison in enumerate(report['comparisons']):
        branch, case = comparison['branch'], comparison['case']
        recs = sorted([r for r in report['records'] if (r['branch'], r['case']) == (branch, case)], key=lambda r: -r['dt_s'])
        if [r['dt_s'] for r in recs] != [2.5e-5, 1.25e-5]:
            raise ValueError('unexpected clock pair')
        arrays, ledgers = [], []
        for r in recs:
            with np.load(source/r['npz'], allow_pickle=False) as archive:
                arrays.append({k: archive[k] for k in ('time_s', 'expected_time_s', 'ncon', 'qvel', 'qfrc_constraint')})
            ledgers.append(json.loads((source/r['contacts']).read_text()))
        a, b = arrays
        ac, bc = ledgers
        times = a['expected_time_s']
        assert len(times) == len(ac) == len(bc) == len(a['ncon']) == len(b['ncon']) == 400
        assert np.array_equal(times, b['expected_time_s'])
        for arr in arrays:
            assert np.allclose(arr['time_s'], times, rtol=0, atol=1e-14)
            assert arr['qvel'].shape == arr['qfrc_constraint'].shape == (400, len(comparison['dofs']))
        samples = [metrics(x, y) for x, y in zip(ac, bc)]
        values = [x['pairmax'] for x in samples]
        assert np.allclose(values, comparison['force_difference_series'], rtol=0, atol=1e-18)
        c = dict(branch=branch, case=case, archives=[r['npz'] for r in recs],
                 ledger_lengths=[len(ac), len(bc)], time_s=times.tolist(), pairmax_series=values,
                 sum_vector_delta_norm_series=[s['sum_vector_delta_norm'] for s in samples], events=[])
        windows = [('early', .0015, .0023), ('late', .0077, .0087), ('global', 0., .01)]
        lines += [f'## {branch} / {case}', '', '|事件|端点时间 ms|top pair：geom/body|粗力范数|细力范数|欧氏差|相对差|sum-vector差|', '|---|---:|---|---:|---:|---:|---:|---:|']
        for event_name, lower, upper in windows:
            i = select_peak(times, values, lower, upper)
            event = dict(event=event_name, window_s=[lower, upper], index=i, time_s=float(times[i]), neighbors=[])
            for j in range(max(0, i-2), min(len(times), i+3)):
                s = samples[j]
                top = []
                for pair in s['pairs'][:5]:
                    ids = list(map(int, pair['pair'].split(',')))
                    top.append(dict(pair, names=[geom_names[k] for k in ids]))
                event['neighbors'].append(dict(index=j, expected_time_s=float(times[j]),
                    coarse_time_s=float(a['time_s'][j]), fine_time_s=float(b['time_s'][j]),
                    coarse_ncon=int(a['ncon'][j]), fine_ncon=int(b['ncon'][j]),
                    coarse_pairs=sorted(ac[j]), fine_pairs=sorted(bc[j]),
                    presence_symmetric_difference=sorted(set(ac[j]) ^ set(bc[j])),
                    top_pairs=top, **{k:v for k,v in s.items() if k != 'pairs'},
                    qvel=state_row(a,b,j,comparison['dofs'],'qvel'),
                    qfrc_constraint=state_row(a,b,j,comparison['dofs'],'qfrc_constraint')))
            c['events'].append(event)
            selected = next(n for n in event['neighbors'] if n['index'] == i)
            p = selected['top_pairs'][0]
            names = ' ↔ '.join(f"{n['geom_id']}:{n['geom']} / {n['body']}" for n in p['names'])
            lines += [f"|{event_name}|{times[i]*1000:.6f}|{names}|{p['coarse_norm']:.12g}|{p['fine_norm']:.12g}|{p['delta_norm']:.12g}|{p['norm_relative']:.12g}|{selected['sum_vector_delta_norm']:.12g}|"]
            ax = axes[row_index, 0 if event_name == 'early' else 1]
            if event_name != 'global':
                ids = (times >= lower) & (times <= upper)
                ax.plot(times[ids]*1000, np.asarray(values)[ids], label='pairmax')
                ax.plot(times[ids]*1000, np.array(c['sum_vector_delta_norm_series'])[ids], label='sum-vector (cancellation)')
                ax.axvline(times[i]*1000, color='k', alpha=.3)
                ax.set_title(f'{branch} {case}: {event_name}'); ax.legend(fontsize=7); ax.grid(alpha=.2)
                ax.set_xlabel('Time (ms)'); ax.set_ylabel('Force difference (model units)')
        lines += ['', '### 所选端点及±2共同端点（每步25µs）', '', '|事件|时间 ms|ncon粗/细|presence异集|top pair|pair差|qvel最大差DOF/值|qfrc最大差DOF/值|', '|---|---:|---|---|---|---:|---|---|']
        for ev in c['events'][:2]:
            for n in ev['neighbors']:
                v=n['qvel']['top_absolute_differences'][0]; f=n['qfrc_constraint']['top_absolute_differences'][0]
                lines += [f"|{ev['event']}|{n['expected_time_s']*1000:.6f}|{n['coarse_ncon']}/{n['fine_ncon']}|{n['presence_symmetric_difference']}|{n['top_pairs'][0]['pair']}|{n['pairmax']:.10g}|{v['dof']}:{v['delta']:.10g} {v['velocity_unit']}|{f['dof']}:{f['delta']:.10g} {f['constraint_force_unit']}|"]
        lines += ['', 'JSON保存每事件前后全部qvel/qfrc_constraint向量、逐DOF差及top5标签，完整pair集合和top5力向量/范数/相对差。', '']
        output['comparisons'].append(c)
    after = {str(p): sha(p) for p in files}
    provenance_after = {p: sha(p) for p in report['source_hashes']}
    assert before == after and provenance_before == provenance_after
    output.update(input_hashes_before=before, input_hashes_after=after, hashes_unchanged=True,
                  original_source_hashes_current=provenance_after,
                  analysis_source_hashes={str(p):sha(p) for p in (Path(__file__),ROOT/'S2_CONTACT_PEAK_AUDIT_PROTOCOL.zh-CN.md',ROOT/'test_s2_contact_peak_audit.py')})
    lines += ['## 结论边界与下一步', '', '这是事件定位，不是物理原因证明。物理原因未知：保存数据不足以分辨接触点漂移、约束活跃集、求解器/积分器内部阶段等机制，不把相关状态变化当因果。',
              '没有新增通过/收敛/生物阈值；ncon或pair presence相同不证明力/支撑收敛。未重建原始contact frame/local force，未计算冲量/功平衡，未全量回归。',
              '下一最小动作：由父审阅已定位的pair和邻域证据；若必须问物理原因，另行预声明最小原始接触/约束观测协议并获得新运行授权，不无限细化步长或修改生产积分器。', '',
              f'输入：`{source}`；机器报告：`{out / "report.json"}`；图：`{out / "contact-peak-audit.png"}`。输入17文件及原source_hashes前后哈希一致。']
    out.mkdir(parents=True, exist_ok=False)
    (out/'report.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))
    (out/'接触差峰离线审计报告.zh-CN.md').write_text('\n'.join(lines)+'\n')
    fig.suptitle('OFFLINE contact force events: localization, NOT physical cause / convergence')
    fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(out/'contact-peak-audit.png',dpi=150); plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
