"""Post-run additive notes: shared-CPU timing caveat, P3 boundary caveat, amplification statistics.

This file is NOT part of the frozen computation: SOURCE_FREEZE.json covers exactly
run.py / verify.py / report.py / finalize.py and those hashes must stay unchanged.
This script (a) regenerates the base report with the frozen report.py so that it is
idempotent, (b) appends clearly labelled notes to the generated outputs, and (c) records
the pre/post hashes of everything it touches in POSTRUN_ADDENDUM.json.
"""
import pickle
from pathlib import Path
import numpy as np
import run as r
import report

NOTE_ENV = ('本轮 4 次神经运行的 wall_s（%s s/次）是在同机另一重任务（阶段二感觉对照）串行占用 CPU 时测得的实际墙钟时间，'
            '不代表典型耗时。神经计算是确定性的：同一封存输入必得同一结果，计时不影响任何数值，故结果与结论不受共享 CPU 影响。')

NOTE_P3 = ('P3 的 exp(+δ/τ) 幅值等价只在“到达所在区间之后”的所有区间边界上与真实晚到 δ 一致（相对误差 <1e-14）；'
           '在到达所在区间自身的边界处，它与“把该事件整体挪到下一个整数槽”的对照相差约 δ/τ'
           '（兴奋 3.33e-7、抑制 1.25e-7 相对），这一差值与该微扰本身同阶，属于本微扰定义自带的一阶项，'
           '因此 **不应写成“完全无损”**。本轮实际施加量为：抑制电导最大增量 %s µS（目标细胞无兴奋出向连接），'
           '到达区间内 %d 个突触后细胞膜电位出现差异（该区间最大 %s mV），与解析值最大偏差 %s µS；'
           '其后所有区间边界与解析一致（见 VERIFICATION.json 的 P3_conductance_injection_exact）。')

LABEL = {'P1_current': 'P1', 'P2_voltage': 'P2', 'P3_event_phase': 'P3'}


def main():
    # (a) idempotent base report from the frozen generator
    report.main()
    rep = r.OUT / 'REPORT.zh-CN.md'
    summ = r.OUT / 'SUMMARY.json'
    base_report = rep.read_text()
    before = {str(summ): r.sha(summ), str(rep): r.sha(rep)}

    s = r.js(summ)
    walls = [x['wall_s'] for x in s['results']]

    zc = r.arrays(r.OUT / 'RUNS/control/NEURAL.npz')
    ref_rms = {k: float(np.sqrt(np.mean(np.asarray(zc[k]) ** 2))) for k in ['v', 'ge', 'gi']}
    tc = r.arrays(r.OUT / 'RUNS/control/TARGET_TRACE.npz')
    obs = {}
    for label in LABEL:
        z = r.arrays(r.OUT / 'RUNS' / label / 'NEURAL.npz')
        tp = r.arrays(r.OUT / 'RUNS' / label / 'TARGET_TRACE.npz')
        dv = np.abs(np.asarray(z['v']) - np.asarray(zc['v']))
        d = np.abs(tp['v'] - tc['v'])
        nz = np.flatnonzero(d > 0)
        gmax = {k: float(np.abs(np.asarray(z[k]) - np.asarray(zc[k])).max()) for k in ['v', 'ge', 'gi']}
        obs[label] = dict(
            global_max_abs_state_difference=gmax,
            cells_with_any_v_difference=int((dv > 0).any(axis=0).sum()),
            target_max_abs_v_difference_mV=float(d.max()),
            target_first_nonzero_step=(int(nz[0]) if nz.size else None),
            target_first_nonzero_time_ms=(float((nz[0] + 1) * r.DT_MS) if nz.size else None),
            target_first_nonzero_value_mV=(float(d[nz[0]]) if nz.size else None),
            target_max_abs_difference_per_window_mV={str(w): float(d[:w * r.SUBSTEPS].max()) for w in r.WINDOWS_MS},
            target_growth_after_first_ms=(None if not nz.size else float(d.max() / max(d[:r.SUBSTEPS].max(), 1e-300))),
            threshold_flips_total=int(s['comparisons'][label]['windows']['200']['differing_spikes_cumulative']))
        del z, dv

    a = np.load(r.FOL / 'RUNS/backward_euler_0p03125ms/NEURAL.npz')
    b = np.load(r.FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz')
    dtstate = {}
    for k in ['v', 'ge', 'gi']:
        x = a[k][-1].astype(float)
        y = b[k][-1].astype(float)
        dd = x - y
        dtstate[k] = dict(rms=float(np.sqrt(np.mean(dd * dd))), max_abs=float(np.abs(dd).max()),
                          reference_rms=float(np.sqrt(np.mean(y * y))),
                          normalized_rms=float(np.sqrt(np.mean(dd * dd)) / max(np.sqrt(np.mean(y * y)), 1e-9)))
    a.close()
    b.close()
    for label in LABEL:
        obs[label]['dt_v_rms_over_perturbation_max_abs_dv'] = float(
            dtstate['v']['rms'] / max(obs[label]['global_max_abs_state_difference']['v'], 1e-300))

    ini = pickle.load((r.OUT / 'RUNS/control/INITIAL_STATE.pkl').open('rb'))
    ee, ei, meta = r.phase_injection(ini['net'], r.TARGET)
    site_c = r.arrays(r.OUT / 'RUNS/control/SITE_STATE.npz')
    site_p = r.arrays(r.OUT / 'RUNS/P3_event_phase/SITE_STATE.npz')
    dgi = np.asarray(site_p['gi']) - np.asarray(site_c['gi'])
    dv_site = np.abs(np.asarray(site_p['v']) - np.asarray(site_c['v']))
    p3_numbers = dict(applied_max_inh_increment_uS=float(np.abs(dgi).max()),
                      applied_max_exc_increment_uS=float(np.abs(ee).max()),
                      posts_with_v_difference_at_arrival=int((dv_site > 0).sum()),
                      max_post_v_difference_at_arrival_mV=float(dv_site.max()),
                      analytic_max_abs_error_uS=float(np.abs(dgi - ei * np.exp(-r.DT_MS / r.INH_TAU_MS)).max()),
                      factor_exc=meta['factor_exc'], factor_inh=meta['factor_inh'],
                      delta_over_tau_exc=r.P3_MS / r.EXC_TAU_MS, delta_over_tau_inh=r.P3_MS / r.INH_TAU_MS,
                      site_step=int(site_p['step']))

    s['environment_note'] = NOTE_ENV % ('/'.join('%.2f' % w for w in walls))
    s['wall_time_environment'] = 'shared_cpu'
    s['p3_boundary_note'] = NOTE_P3 % ('%.3g' % p3_numbers['applied_max_inh_increment_uS'],
                                       p3_numbers['posts_with_v_difference_at_arrival'],
                                       '%.3g' % p3_numbers['max_post_v_difference_at_arrival_mV'],
                                       '%.3g' % p3_numbers['analytic_max_abs_error_uS'])
    s['amplification_observations'] = dict(
        observations=obs, p3_applied=p3_numbers, dt_induced_state_difference_200ms=dtstate,
        control_state_rms=ref_rms,
        interpretation='三个病例在 12800 个神经步内没有任何阈值翻转：P1 仅 1 个细胞、P2 仅 1 个细胞、P3 仅 165 个突触后细胞的'
                       '膜电位出现非零差异，突触电导与全部尖峰在 P1、P2 中完全不变。微扰在全网络的最大状态位移比 dt 档间在'
                       ' 200 ms 的状态差小 7–9 个数量级，且没有随时间增长。')
    s['report_text_correction'] = ('REPORT.zh-CN.md 第三节 P2 行的 “ulp≈-1.421e-14” 是 np.spacing 对负值的符号显示，'
                                   '量级正确（1.421e-14 mV），报告中改为正号显示；数值内容未变。')
    s['postrun_addendum'] = str(r.OUT / 'POSTRUN_ADDENDUM.json')
    r.dump(summ, s)

    # (b) report surgery on a freshly regenerated base report
    anchor = '\n## 四、微扰何时变成离散事件'
    assert anchor in base_report and 'P3 的边界说明' not in base_report
    text = base_report
    text = text.replace('ulp≈-1.421e-14', 'ulp≈1.421e-14')
    text = text.replace('该因子在每个后续区间边界上精确等于“晚到 δ”的衰减值，已在合成两细胞网络端到端验证',
                        '该因子在每个后续区间边界上精确等于“晚到 δ”的衰减值（相对误差 <1e-14，已在合成两细胞网络端到端验证）；'
                        '到达所在区间自身边界处与整数槽对照相差约 δ/τ 的同阶项，见下方“P3 的边界说明”')
    assert anchor in text and '**P3 的边界说明（必读）**' not in text
    text = text.replace(anchor, '\n**P3 的边界说明（必读）**：' + s['p3_boundary_note'] + '\n' + anchor)

    L = ['\n## 十、放大 vs 偏差：本轮定量观测\n',
         '下列数值全部由保存的 1 ms 全状态数组与逐步目标轨迹重算（不重新仿真）：\n',
         '|病例|全网络最大 |Δv| mV|最大 |Δge| µS|最大 |Δgi| µS|膜电位有差异的细胞数|200 ms 内不同尖峰|'
         'dt 的 v rms 与该病例最大 |Δv| 之比（倍）|',
         '|---|---|---|---|---|---|---|']
    for label in LABEL:
        o = obs[label]
        g = o['global_max_abs_state_difference']
        L.append('|%s|%.4g|%.4g|%.4g|%d|%d|%.3g|' % (LABEL[label], g['v'], g['ge'], g['gi'],
                                                      o['cells_with_any_v_difference'], o['threshold_flips_total'],
                                                      o['dt_v_rms_over_perturbation_max_abs_dv']))
    L.append('')
    L.append('目标细胞自身轨迹：P1 首次分歧在 %s ms（%s mV），第 1 个 1 ms 内即达 %s mV，全程最大 %s mV（其余 199 ms 只再变化到首毫秒的 %s 倍）；'
             'P2 首次分歧在 %s ms（%s mV），全程最大 %s mV（此后单调衰减）；'
             'P3 注入作用于该细胞的突触后细胞而非其自身，故其自身轨迹逐位不变，而 %d 个突触后细胞出现差异。'
             % ('%.6g' % obs['P1_current']['target_first_nonzero_time_ms'],
                '%.4g' % obs['P1_current']['target_first_nonzero_value_mV'],
                '%.4g' % obs['P1_current']['target_max_abs_difference_per_window_mV']['1'],
                '%.4g' % obs['P1_current']['target_max_abs_v_difference_mV'],
                '%.3g' % obs['P1_current']['target_growth_after_first_ms'],
                '%.6g' % obs['P2_voltage']['target_first_nonzero_time_ms'],
                '%.4g' % obs['P2_voltage']['target_first_nonzero_value_mV'],
                '%.4g' % obs['P2_voltage']['target_max_abs_v_difference_mV'],
                obs['P3_event_phase']['cells_with_any_v_difference']))
    L.append('')
    L.append('对照量（封存最细两档 dt，0.03125 vs 0.015625 ms，200 ms 的**状态**差异）：v rms %s mV / 最大 %s mV / nRMS %s；'
             'ge rms %s µS / nRMS %s；gi nRMS %s。同一时刻池率/原始/限幅命令的 nRMS 为 %s / %s / %s。'
             % ('%.6g' % dtstate['v']['rms'], '%.6g' % dtstate['v']['max_abs'], '%.6g' % dtstate['v']['normalized_rms'],
                '%.6g' % dtstate['ge']['rms'], '%.6g' % dtstate['ge']['normalized_rms'],
                '%.6g' % dtstate['gi']['normalized_rms'],
                '%.6g' % r.DT_BASELINE_200['pool_rates']['fine'],
                '%.6g' % r.DT_BASELINE_200['raw_command']['fine'],
                '%.6g' % r.DT_BASELINE_200['clipped_command']['fine']))
    L.append('')
    L.append('## 十一、9 个比值、判定与适用边界\n')
    L.append('按预先登记规则（9 个 perturbation/dt 的 200 ms nRMS 比值中 ≥6 个 ≥0.1 为放大主导；全部 ≤1e-3 为偏差主导；否则混合），'
             '本轮判定为 **%s**：9 个比值全部恰为 0——不是“接近阈值”，而是三个病例在 12800 步内一个阈值翻转都没有发生。'
             % s['verdict']['reading'])
    L.append('')
    L.append('|病例|pool 比值|raw 比值|clip 比值|')
    L.append('|---|---|---|---|')
    for label in LABEL:
        pr = s['verdict']['per_run'][label]
        L.append('|%s|%.6g|%.6g|%.6g|' % (LABEL[label], pr['pool_rates']['ratio'], pr['raw_command']['ratio'],
                                          pr['clipped_command']['ratio']))
    L.append('')
    L.append('适用边界（必须同时写清）：本轮微扰量级（1e-9 nA、1e-6 mV、1e-6 ms）比 dt 档间在 200 ms 的状态差小约 7–9 个数量级。'
             '因此本轮证据确立的是“在这个量级、这个 200 ms 窗口内，微小差异没有被放大到 dt 差异量级”，'
             '而不是“该网络在任何尺度、任何时长上都没有放大”——后者需要更大微扰或更长窗口才能检验，本轮未做，也不据此断言。'
             '同理，本轮不判定脑模型有效或无效，不选择时间步长，不给出百分比。')
    L.append('')
    L.append('由登记规则得到的验收含义：既然 50–200 ms 的 dt 分歧无法被此量级的任意小差异复现，分歧与系统性截断/量化偏差'
             '（步内电导变化、事件相位与阈值时刻的离散处理）一致，改进积分器在原理上仍有意义；'
             '反过来，本轮也没有取得“任何细化都不可能收敛”的证据，不能据此把验收目标降级为只看统计量。')
    L.append('')
    L.append('## 十二、环境与计时\n')
    L.append(s['environment_note'])
    L.append('')
    L.append('本附录由 postrun_addendum.py 在冻结管线（run.py/verify.py/report.py/finalize.py，四个哈希均未变）之后追加；'
             '该脚本只重新生成并追加文本、新增 SUMMARY 字段，不重新仿真、不改任何数值结果。'
             '脚本哈希与被修改文件的前后哈希见 POSTRUN_ADDENDUM.json；SUMMARY.json 新增字段为 environment_note / '
             'wall_time_environment / p3_boundary_note / amplification_observations / report_text_correction / postrun_addendum。')
    rep.write_text(text.rstrip('\n') + '\n' + '\n'.join(L) + '\n')

    frozen = {str(p): r.sha(p) for p in sorted(r.SRC.glob('*.py'))}
    freeze = r.js(r.OUT / 'SOURCE_FREEZE.json')
    r.dump(r.OUT / 'POSTRUN_ADDENDUM.json', dict(
        script=str(Path(__file__).resolve()), script_sha256=r.sha(Path(__file__).resolve()),
        frozen_source_hashes=frozen, frozen_source_freeze=freeze,
        frozen_source_unchanged=all(frozen.get(p) == h for p, h in freeze.items()),
        before_sha256=before,
        after_sha256={str(summ): r.sha(summ), str(rep): r.sha(rep)},
        summary_keys_added=['environment_note', 'wall_time_environment', 'p3_boundary_note',
                            'amplification_observations', 'report_text_correction', 'postrun_addendum'],
        report_changes=['第三节 P2 行 ulp 符号显示修正（-1.421e-14 -> 1.421e-14）',
                        '第三节 P3 说明列追加“到达区间自身边界处约 δ/τ 同阶项”提示',
                        '第三节表后插入“P3 的边界说明（必读）”',
                        '末尾追加第十、十一、十二节'],
        wall_s_seconds=walls, wall_time_environment='shared_cpu',
        reason='父代理要求：wall_s 需注明为共享 CPU 下测得；P3 需如实写明到达区间边界处约 δ/τ 的一阶项差额，不得写成完全无损'))
    print('ADDENDUM_WRITTEN', len(rep.read_text().splitlines()), flush=True)


def inventory_notes():
    """Append post-run artifact rows to the frozen finalize.py inventory, record real deletions."""
    import finalize
    inv = r.js(r.OUT / 'FILE_INVENTORY.json')
    have = {x['path'] for x in inv['files']}
    extra = [(r.SRC / 'postrun_addendum.py', '重要辅助',
              '追加说明脚本：共享 CPU 计时说明、P3 边界如实说明、放大统计；不重新仿真、不改任何数值结果'),
             (r.OUT / 'POSTRUN_ADDENDUM.json', '重要辅助',
              '追加脚本哈希、被修改文件前后哈希、新增字段与清单追加行'),
             (r.OUT / 'VERIFY.log', '可再生临时', 'verify.py 运行日志')]
    added = []
    for path, imp, purpose in extra:
        if str(path) not in have:
            fp = Path(path)
            inv['files'].append(dict(path=str(fp), importance=imp, purpose=purpose, exists=fp.exists(),
                                     bytes=(fp.stat().st_size if fp.exists() else None)))
            added.append(str(fp))
    inv['postrun_note'] = 'FILE_INVENTORY.json 由冻结 finalize.py 生成后，由 postrun_addendum.py 追加上述行，其余内容未改。'
    inv['postrun_added_rows'] = added
    r.dump(r.OUT / 'FILE_INVENTORY.json', inv)

    led = r.js(r.OUT / 'CLEANUP_LEDGER.json')
    led['deleted'] = [
        dict(path=str(r.SRC / '__pycache__'), importance='可再生临时',
             reason='py_compile 语法检查产生的字节码目录，已删除；此后所有调用均带 -B / PYTHONDONTWRITEBYTECODE=1'),
        dict(path=str(r.OUT / 'PREREGISTRATION.json'), importance='核心必需',
             reason='冻结前一次失败的 prereg 尝试残留（自身脚本内的源路径字面量写错，未进行任何仿真）；删除后由冻结 prereg 重新生成'),
        dict(path=str(r.OUT / 'INPUT_HASHES_BEFORE.json'), importance='核心必需',
             reason='同上，为失败尝试的残留版本；删除后由冻结 prereg 重新生成'),
        dict(path=str(r.OUT / 'CHANGE_LEDGER.json'), importance='核心必需',
             reason='同上，为失败尝试的残留版本；删除后由冻结 prereg 重新生成'),
        dict(path=str(r.OUT / 'MECHANISM_TESTS.json'), importance='核心必需',
             reason='同上，为失败尝试的残留版本；删除后由冻结 prereg 重新生成'),
        dict(path=str(r.OUT / 'SOURCE_FREEZE.json'), importance='核心必需',
             reason='同上，为失败尝试的残留版本；删除后由冻结 prereg 重新生成'),
        dict(path='/tmp/chaos_probe', importance='可再生临时',
             reason='本轮开发期的临时探针脚本与假数据（probe1..5.py、fake_out/），位于交付目录之外，已整体删除')]
    led['kept_unsure'] = [dict(path=str(x), reason='冻结前一次失败的 prereg 尝试（脚本路径笔误，未仿真、未产生数据），按项目规则保留失败轨迹')
                          for x in sorted(r.OUT.glob('FAILURE_*.json'))]
    led['note'] = ('本阶段删除的只有：字节码目录、失败 prereg 尝试的残留冻结文件（随后重新生成）、以及交付目录之外的临时探针目录；'
                   '任何运行结果、失败轨迹与封存输入均未删除')
    r.dump(r.OUT / 'CLEANUP_LEDGER.json', led)

    ledger = r.js(r.OUT / 'POSTRUN_ADDENDUM.json')
    ledger['inventory_rows_added'] = added
    ledger['deleted_paths'] = [x['path'] for x in led['deleted']]
    r.dump(r.OUT / 'POSTRUN_ADDENDUM.json', ledger)
    count = len([x for x in r.OUT.rglob('*') if x.is_file()])
    audit = r.js(r.OUT / 'FINAL_AUDIT.json')
    audit['manifest_file_count'] = count
    audit['manifest_self_hash_excluded'] = True
    audit['postrun_inventory_rows_added'] = added
    audit['postrun_addendum'] = str(r.OUT / 'POSTRUN_ADDENDUM.json')
    audit['cleanup'] = led
    r.dump(r.OUT / 'FINAL_AUDIT.json', audit)
    files = finalize.manifest()
    man = r.js(r.OUT / 'MANIFEST_SHA256.json')
    selfkey = str(r.OUT / 'MANIFEST_SHA256.json')
    if selfkey in man['files']:
        del man['files'][selfkey]
    man['self_excluded'] = True
    man['self_excluded_reason'] = '清单文件无法包含自身哈希；其余每个文件均有有效 sha256'
    man['file_count'] = len(man['files'])
    r.dump(r.OUT / 'MANIFEST_SHA256.json', man)
    print('INVENTORY_NOTES', len(added), len(man['files']), flush=True)


if __name__ == '__main__':
    import sys
    {'notes': main, 'inventory': inventory_notes}[sys.argv[1] if len(sys.argv) > 1 else 'notes']()
