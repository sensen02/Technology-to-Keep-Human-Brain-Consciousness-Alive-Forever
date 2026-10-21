"""Offline same-endpoint force decoding audit; never loads or advances a model."""
from pathlib import Path
import json
import hashlib
import uuid
from datetime import datetime, timezone
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('/home/sensen/Desktop/engine修复验收/S2-raw-contact-20261014T204339Z-7ba64625')
TOL = 1e-10
STATES = {0: 'mjCNSTRSTATE_SATISFIED', 1: 'mjCNSTRSTATE_QUADRATIC', 2: 'mjCNSTRSTATE_LINEARNEG', 3: 'mjCNSTRSTATE_LINEARPOS', 4: 'mjCNSTRSTATE_CONE'}
MISSING = ['efc_AR', 'efc_b', 'efc_vel', 'efc_aref', 'efc_R', 'efc_D', 'efc_J', 'qacc', 'qacc_smooth', 'contact_solref', 'contact_solreffriction', 'contact_solimp', 'contact_includemargin']
URLS = ['https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_util_misc.c', 'https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_core_util.c', 'https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/include/mujoco/mjtype.h']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decode_local6(efc, address, dim, friction, cone=0):
    """Independent scalar implementation of documented MuJoCo contact extraction."""
    if cone not in (0, 1):
        raise ValueError('unknown cone: frozen identity/load-only verification required')
    if dim not in (1, 3, 4, 6):
        raise ValueError('invalid contact dimension')
    out = np.zeros(6)
    if address == -1:
        return out
    if address < 0:
        raise ValueError('invalid negative address')
    f = np.asarray(efc, dtype=float)
    mu = np.asarray(friction, dtype=float)
    if f.ndim != 1 or mu.ndim != 1 or len(mu) < dim - 1:
        raise ValueError('force/friction dimensions')
    if not np.isfinite(mu).all() or np.any(mu < 0):
        raise ValueError('invalid friction')
    n = 1 if dim == 1 else (2 * (dim - 1) if cone == 0 else dim)
    if address + n > len(f) or not np.isfinite(f[address:address+n]).all():
        raise ValueError('force bounds/finite')
    p = f[address:address+n]
    if cone == 1 or dim == 1:
        out[:dim] = p
    else:
        for value in p:
            out[0] += float(value)
        for i in range(dim - 1):
            out[i+1] = (p[2*i] - p[2*i+1]) * mu[i]
    return out


def audit_case(a, cone):
    if cone != 0:
        raise ValueError('this preregistered audit requires saved pyramidal cone')
    co, eo = a['contact_offsets'], a['efc_offsets']
    for off, count, total in ((co, a['ncon'], len(a['contact_local6'])), (eo, a['nefc'], len(a['efc_force']))):
        assert len(off) == len(a['time_s'])+1 and off[0] == 0 and off[-1] == total
        assert np.array_equal(np.diff(off), count) and np.all(np.diff(off) >= 0)
    assert a['contact_local6'].shape == (co[-1], 6)
    assert a['contact_friction'].shape == (co[-1], 5)
    assert all(np.isfinite(a[k]).all() for k in a.files)
    residuals, norms, torques, rows, histogram = [], [], [], [], Counter()
    worst = dict(residual=0., endpoint=0, contact=0, component=0)
    inactive = failed = 0
    dims = Counter()
    for t in range(len(a['time_s'])):
        c0, c1, e0, e1 = map(int, (co[t], co[t+1], eo[t], eo[t+1]))
        ef = a['efc_force'][e0:e1]
        rr, nn, tt = [], [], []
        for j in range(c0, c1):
            address, dim = int(a['contact_efc_address'][j]), int(a['contact_dim'][j])
            dims[dim] += 1
            decoded = decode_local6(ef, address, dim, a['contact_friction'][j], cone)
            delta = np.abs(decoded-a['contact_local6'][j])
            r = float(delta.max()); rr.append(r)
            nn.append(float(np.linalg.norm(decoded[:3]))); tt.append(float(np.linalg.norm(decoded[3:])))
            failed += r > TOL
            if r > worst['residual']:
                worst = dict(residual=r, endpoint=t, time_s=float(a['time_s'][t]), contact=j-c0, component=int(delta.argmax()))
            if address == -1:
                inactive += 1
            else:
                n = 1 if dim == 1 else 2*(dim-1)
                ix = slice(e0+address, e0+address+n)
                assert np.all(a['efc_type'][ix] == (5 if dim == 1 else 6))
                assert np.all(a['efc_id'][ix] == j-c0)
                rows.extend(ef[address:address+n].tolist())
                for state in a['efc_state'][ix]:
                    histogram[STATES[int(state)]] += 1
        residuals.append(max(rr, default=0.)); norms.append(max(nn, default=0.)); torques.append(max(tt, default=0.))
    row = np.asarray(rows)
    return dict(endpoints=len(a['time_s']), contacts=int(co[-1]), active_contacts=int(co[-1])-inactive, inactive_contacts=inactive, active_rows=len(rows), dimensions=dict(dims), max_residual=max(residuals), worst=worst, failed_contacts=int(failed), decode_pass=failed == 0, active_row_min=float(row.min()) if len(row) else None, strict_negative_rows=int((row < 0).sum()), below_tolerance_rows=int((row < -TOL).sum()), pyramidal_nonnegative_engineering_pass=bool(np.all(row >= -TOL)), state_row_histogram=dict(histogram), state_histogram_is_tracked_transition=False, endpoint_residual=residuals, endpoint_max_linear_force_norm=norms, endpoint_max_torque_norm=torques)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mujoco
    assert mujoco.__version__ == '3.9.0'
    m = json.loads((ARCHIVE/'manifest.json').read_text())
    out = Path('/home/sensen/Desktop/engine修复验收') / ('S2-contact-law-decode-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    inputs = list(ARCHIVE.iterdir()) + [Path(p) for p in m['source_hashes_before']] + [Path(__file__), ROOT/'test_s2_contact_law_decode.py', ROOT/'S2_CONTACT_LAW_DECODE_PROTOCOL.zh-CN.md']
    before = {str(p): sha(p) for p in inputs if p.is_file()}
    assert m['source_hashes_before'] == m['source_hashes_after']
    assert all(before[p] == h for p,h in m['source_hashes_before'].items())
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    cases = []
    for rec in m['records']:
        with np.load(ARCHIVE/rec['npz'], allow_pickle=False) as a:
            result = audit_case(a, rec['cone'])
            result.update(npz=str(ARCHIVE/rec['npz']), cone=rec['cone'], cone_evidence='manifest record; runner line69 saves int(m.opt.cone)', schema={k:dict(shape=list(a[k].shape), dtype=str(a[k].dtype)) for k in a.files}, missing_dynamic_fields=[k for k in MISSING if k not in a.files])
            cases.append(result)
            label=f"dt={rec['dt_s']*1e6:g} us"
            axes[0].plot(a['time_s']*1000, result['endpoint_residual'], label=label)
            axes[1].plot(a['time_s']*1000, result['endpoint_max_linear_force_norm'], label=label)
    axes[0].axhline(TOL, color='red', linestyle='--', label='engineering abs tolerance 1e-10')
    axes[0].set_ylabel('max abs decode residual'); axes[0].set_ylim(-TOL*.05,TOL*1.15)
    axes[1].set_ylabel('max contact linear force norm\n(model force units, not support sum)'); axes[1].set_xlabel('saved endpoint time (ms)')
    for ax in axes: ax.grid(True, alpha=.3); ax.legend()
    fig.suptitle('Same-endpoint pyramidal force extraction only; NOT dynamic law residual')
    fig.tight_layout(); fig.savefig(out/'decode-residual-localnorm.png', dpi=160); plt.close(fig)
    after = {p: sha(p) for p in before}; assert before == after
    report = dict(archive=str(ARCHIVE), protocol=str(ROOT/'S2_CONTACT_LAW_DECODE_PROTOCOL.zh-CN.md'), tolerance=TOL, official_sources=URLS, environment=dict(python=__import__('sys').executable, mujoco=mujoco.__version__), cases=cases, hashes_before=before, hashes_after=after, original_source_identity_verified=True, model_loaded=False, forward_or_step=False, full_dynamic_law_residual_computable=False, physical_cause='unknown', body_convergence_proven=False, biological_validation=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    text = '# S2/P2 同端点接触力解码审计\n\n手动离线研究实施，非成品。无新动力学/生产编辑/模型加载/forward/step。\n\n预注册：'+report['protocol']+'\n输入：'+str(ARCHIVE)+'\n\n'
    for c in cases:
        text += f"## {c['npz']}\n- cone=0 pyramidal（manifest编译时record）；{c['endpoints']}端点、{c['contacts']}contacts，active {c['active_contacts']}，inactive {c['inactive_contacts']}；active rows {c['active_rows']}。\n- 六维绝对最大解码误差 {c['max_residual']:.17g}，失败contact {c['failed_contacts']}；固定1e-10工程通过={c['decode_pass']}。\n- active row minimum {c['active_row_min']}；严格负行 {c['strict_negative_rows']}，<-1e-10行 {c['below_tolerance_rows']}；金字塔工程非负通过={c['pyramidal_nonnegative_engineering_pass']}。\n- state行分布 {c['state_row_histogram']}。QUADRATIC是约束二次cost，不是跟踪接触点状态变化；地址不跨端点/时钟匹配。\n- 缺失：{', '.join(c['missing_dynamic_fields'])}。\n\n"
    text += '## 结论边界\n解码仅证明saved efc_force按官方算法与saved contact_local6自洽；非独立物理真值。缺失AR/b/速度/参考加速度/正则化/J与solref等，不能重建完整动态接触律/KKT残差，不能说明差峰原因，physical_cause=unknown。身体收敛/生物验收/站立行走未证明。所有源文件与输入前后SHA一致并匹配历史十source身份。图上残差为零时明确画零线，局部力范数不是支撑总力；力矩范数单列JSON不混单位。\n\n官方源码：\n' + '\n'.join('- '+u for u in URLS) + '\n\n下一最小动作：先起草最小动态律观测字段/同端点语义协议；本轮不执行新动态。独立recheck另存independent-check.json；父审阅状态由tracker记录。\n'
    (out/'同端点接触力解码报告.zh-CN.md').write_text(text)
    print(out)


if __name__ == '__main__':
    main()
