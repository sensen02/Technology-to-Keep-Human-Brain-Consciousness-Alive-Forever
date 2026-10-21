"""Turn 31: the single in-place errata round.

Lands the owner's five items AND every finding of the FAIL verification (B1, M1-M4, m1-m6) into the canonical batch and the
record batch, with one errata ledger carrying before/after sha256 for every edit, and with the frozen manifest corrected so
that it really covers every delivered file.

No integration is re-run except one short targeted run used to measure the thorax-contact facts that M1 requires.
"""

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
REC = DESK / 'S2-correction-record-v3-20261015T172848'
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015/appended_events.json'
VOID_A = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
VOID_B = DESK / 'S2-accepted-equilibrium-localization-20261015T170058'
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
OLD_ART = DESK / 'S2-true-equilibrium-openloop-20261015T154720'
EXT_A = DESK / 'S2-dynamic-from-s1-pose-20261015T133124'
STATE = V1 / 'equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DT, STEPS = 2.5e-5, 20000
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
ms_of = lambda s: round(s * DT * 1000.0, 6)
ledger = []


def edit(file, fn, reason):
    p = pathlib.Path(file)
    b = sha(p)
    t = p.read_text()
    nt = fn(t)
    if nt != t:
        p.write_text(nt)
    a = sha(p)
    ledger.append(dict(file=str(p), sha256_before=b, sha256_after=a, changed=bool(a != b), changed_at=now(),
                       reason=reason, changed_by='the model on the owner\'s instruction'))
    return p


def write_frozen(d):
    d = pathlib.Path(d)
    files = {p.name: sha(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json'}
    files['delivery-hashes.json'] = sha(d / 'delivery-hashes.json')
    (d / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(
        status='FROZEN_AFTER_ERRATA', frozen_at=now(),
        covers='every delivered file, including delivery-hashes.json; this manifest excludes only itself and that is the only exclusion',
        files=files), ensure_ascii=False, indent=1))


def thorax_facts():
    """M1: measure the thorax-floor contact facts in the accepted-equilibrium run instead of asserting they do not happen."""
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp
    m = cp.load_model()
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    st = np.load(STATE)
    ctrl = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)
    d = mujoco.MjData(m)
    d.qpos[:] = st['qpos']
    d.qvel[:] = 0.0
    d.act[:] = st['act']
    d.ctrl[:] = ctrl[0]
    mujoco.mj_forward(m, d)
    f6 = np.zeros(6)
    first_instance = first_penetration = first_load = None
    max_fn = 0.0
    instances = 0
    for step in range(STEPS):
        d.ctrl[:] = ctrl[min(step // 200, ctrl.shape[0] - 1)]
        mujoco.mj_step(m, d)
        i = step + 1
        for ci, c in enumerate(d.contact):
            if c.efc_address < 0:
                continue
            n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
            if 'Thorax_geom' in (n1 + n2) and 'floor' in (n1 + n2):
                instances += 1
                if first_instance is None:
                    first_instance = i
                if first_penetration is None and float(c.dist) < 0.0:
                    first_penetration = i
                mujoco.mj_contactForce(m, d, ci, f6)
                fn = float(f6[0])
                max_fn = max(max_fn, fn)
                if first_load is None and fn > 0.0:
                    first_load = i
    return dict(first_contact_instance_step=first_instance, first_contact_instance_ms=ms_of(first_instance) if first_instance else None,
                first_true_penetration_step=first_penetration, first_true_penetration_ms=ms_of(first_penetration) if first_penetration else None,
                first_load_bearing_step=first_load, first_load_bearing_ms=ms_of(first_load) if first_load else None,
                max_normal_force_model_units=max_fn, contact_instances=instances)


def main():
    out = DESK / ('S2-errata-2-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    facts = thorax_facts()

    # ---------------- canonical batch ----------------
    # (a) README: replace the false "no thorax contact" claim and the F-A line with the batch's own numbers
    own_line = ('- 本批自身实测：首个足外接触 **步 2811 = 70.275000 ms**（`A6_geom | floor`，`A6_geom` 属 body A6）；'
                '`legs_in_contact < 6` **步 538 = 13.450000 ms**；首次零跗节接触 **步 5082 = 127.050000 ms**；'
                '首个 `Thorax_geom | floor` 接触 **步 3134 = 78.350000 ms**（步 %s 起 `dist < 0` 真穿透，步 %s = %s ms 起 `fn > 0` 真正承载，'
                '最大 `fn = %.4f` 模型单位，共 %d 次实例）。'
                % (facts['first_true_penetration_step'], facts['first_load_bearing_step'], facts['first_load_bearing_ms'],
                   facts['max_normal_force_model_units'], facts['contact_instances']))
    src_line = ('- **出处说明（旧值只作历史）**：`2739 / 537 / 5111 / 3055` 属于**另一件** `%s` 的 '
                '`true_equilibrium_pose__qualified_25us__*` 数组（该件 `qacc_first_step = 4252.326930873624`，与本批 `2.5865734771464476e-11` '
                '相距 14 个数量级，起点不同）；`76.375 ms`（= 3055 × 25 µs）与 `76.400 ms`（= 3056 × 25 µs）**只作算式示例**，'
                '不指代本批任何事件时刻。' % OLD_ART)

    def fix_readme(t):
        lines = t.split('\n')
        for i, ln in enumerate(lines):
            if '未出现' in ln and 'Thorax' in ln:
                lines[i] = own_line + '\n' + src_line
            elif '不是"失去足接触"' in ln and '3134' not in ln:
                pass
        t2 = '\n'.join(lines)
        if '本批**未出现**' in t2:
            t2 = t2.replace('本批**未出现** `Thorax_geom|floor` 接触', '本批首个 `Thorax_geom|floor` 接触出现在步 3134 = 78.350000 ms')
        return t2

    edit(CANON / 'README.zh-CN.md', fix_readme, 'M1: the false "no thorax contact" claim replaced by the measured thorax facts; '
                                                'the F-A line now carries this batch own numbers plus the source explanation')

    # (b) report.json: thorax note, pose-A comparison correction, self-check note, u_diamond wording, hard-coded constant label
    def fix_report(t):
        r = json.loads(t)
        th = r['this_run']['first_thorax_geom_floor_contact']
        th['note'] = ('measured, not assumed: the first Thorax_geom|floor contact instance is at step %s (%s ms); true penetration (dist < 0) '
                      'from step %s; load bearing (fn > 0) from step %s = %s ms; maximum fn %s model units'
                      % (facts['first_contact_instance_step'], facts['first_contact_instance_ms'], facts['first_true_penetration_step'],
                         facts['first_load_bearing_step'], facts['first_load_bearing_ms'], facts['max_normal_force_model_units']))
        cmp_ = r['comparison_with_pose_A']
        cmp_['pose_A_run']['convention_note'] = ('index convention: the external artifact stores executed-step indices while this batch counts the same '
                                                'samples one-based, and the verification showed mine[i] == external[i-1] for all 19999 overlapping samples '
                                                '(max |delta| = 0) with the same pose npz, the same drive, the same 25 us step and the same act=0 '
                                                'initialisation, i.e. it is the SAME physical event')
        cmp_['pose_A_run']['same_physical_event'] = True
        cmp_['caveat'] = ('the earlier caveat "the two runs differ in their initial actuator state" is WITHDRAWN: both runs initialise act to zeros, so the '
                          'only difference was the index convention. In this batch convention pose A first reaches a non-foot contact at step 2399 = '
                          '59.975000 ms, which equals the external artifact time[2398] = 0.059974999999997475 s')
        cmp_['external_label_note'] = ('the external artifact label "2398 = 59.950 ms" is a hard-coded constant inside the older runner, quoting the '
                                       'independent verification S2-batch-check4-20261015T210312Z; its own recorded time array gives 59.975 ms for that index')
        r['declared']['self_check_strength_note'] = ('the bare assert cannot fail on its own and a cadence comparison written with the declared dt would be '
                                                     'self-referential; the decisive evidence is the measured d.time increment (0.0 for the first step), the '
                                                     'bit-identical reproduction of every array, and the labelled 100 us side branch whose measured cadence is '
                                                     '0.019999999999999934 s instead of 0.005 s')
        r['this_run']['max_u_diamond_over_the_whole_run']['note'] = (
            'both readings are disclosed: over tarsus contacts %s with %d of 20001 steps at or above 1, and over all contacts %s with %d steps at or above 1'
            % (r['this_run']['max_u_diamond_over_the_whole_run']['over_tarsus_contacts_only']['value'],
               r['this_run']['max_u_diamond_over_the_whole_run']['over_tarsus_contacts_only']['steps_at_or_above_1'],
               r['this_run']['max_u_diamond_over_the_whole_run']['over_all_contacts']['value'],
               r['this_run']['max_u_diamond_over_the_whole_run']['over_all_contacts']['steps_at_or_above_1']))
        r['errata'] = dict(round='errata-2', applied_at=now(),
                           items=['M1 thorax facts measured', 'M2 pose-A convention corrected and the wrong caveat withdrawn',
                                  'self-check strength note', 'u_diamond both readings restated',
                                  'frozen manifest corrected to cover every file including delivery-hashes.json'],
                           note='edited in place on the owner\'s instruction; before/after sha256 in the errata ledger')
        return json.dumps(r, ensure_ascii=False, indent=1)

    edit(CANON / 'report.json', fix_report, 'M1/M2/M4 findings and the owner items 4-5 landed in the canonical report')

    # (c) contact_identity_sentence.json with own numbers + restored data_sources
    sentence = dict(
        status='UNIQUE_WORDING_SOURCE_CORRECTED',
        this_batch=dict(artifact=str(CANON),
                        first_non_foot_contact=dict(step=2811, ms=ms_of(2811), pair=['A6_geom', 'floor'], owning_body='A6'),
                        first_step_legs_in_contact_below_6=dict(step=538, ms=ms_of(538)),
                        first_step_with_zero_tarsus_contacts=dict(step=5082, ms=ms_of(5082)),
                        first_thorax_geom_floor_contact=dict(step=facts['first_contact_instance_step'],
                                                             ms=facts['first_contact_instance_ms'],
                                                             first_true_penetration_step=facts['first_true_penetration_step'],
                                                             first_load_bearing_step=facts['first_load_bearing_step'],
                                                             first_load_bearing_ms=facts['first_load_bearing_ms'],
                                                             max_normal_force_model_units=facts['max_normal_force_model_units'])),
        historical_numbers_only=[dict(values='2739 (first non-foot), 537 (legs<6), 5111 (zero tarsus), 3055 (first Thorax_geom|floor)',
                                      belong_to=str(OLD_ART),
                                      arrays=['true_equilibrium_pose__qualified_25us__nonfoot_contacts',
                                              'true_equilibrium_pose__qualified_25us__legs_in_contact',
                                              'true_equilibrium_pose__qualified_25us__tarsus_contacts',
                                              'true_equilibrium_pose__qualified_25us__leg_tarsus_counts'],
                                      source_npz_sha256=sha(OLD_ART / 'time_series.npz'),
                                      why='that artifact started from a different state (qacc at step 0 = 4252.326930873624 vs 2.5865734771464476e-11), so its '
                                          'step indices are not this batch\'s')],
        arithmetic_example_only=dict(step_3055_ms=ms_of(3055), step_3056_ms=ms_of(3056),
                                     note='76.375 ms belongs to step 3055 and 76.400 ms to step 3056 in the other artifact convention; shown as arithmetic only and '
                                          'never used as an event time of this batch'),
        data_sources=dict(npz=str(STATE), npz_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                          arrays_used=['legs_in_contact (derived per step)', 'tarsus contact counts', 'non-foot contact pairs',
                                       'root_z = qpos[2]', 'qacc = mj_data.qacc', 'contact forces via mj_contactForce'],
                          rule='every number quoted for this batch is computed as step x DT from the measured d.time axis'),
        recorded_at=now())
    (CANON / 'contact_identity_sentence.json').write_text(json.dumps(sentence, ensure_ascii=False, indent=1))
    ledger.append(dict(file=str(CANON / 'contact_identity_sentence.json'), sha256_before=None,
                       sha256_after=sha(CANON / 'contact_identity_sentence.json'), changed=True, changed_at=now(),
                       reason='this batch own numbers with the restored data_sources block; the old numbers are kept only as source history',
                       changed_by='the model on the owner\'s instruction'))

    # (d) re-render the localization figure with the dt fields and the lookback axis label
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rep = json.loads((CANON / 'report.json').read_text())
    ts = np.load(CANON / 'time_series_25us.npz')
    t = np.asarray(ts['physical__time_s']) * 1000.0
    lok = rep['localization']['lookback_before_first_nonfoot']
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    lt = np.asarray(ts['physical__leg_tar'])
    for k, L in enumerate(LEGS):
        a.plot(t, np.where(lt[:, k] > 0, lt[:, k], np.nan), lw=1.2, label=L)
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('tarsus contacts (gap = no contact)')
    a.set_title('per-leg tarsus contacts; first loss %s at step %d = %.6f ms'
                % (rep['localization']['first_leg_to_lose_tarsus_contact']['leg'],
                   rep['localization']['first_leg_to_lose_tarsus_contact']['step'],
                   rep['localization']['first_leg_to_lose_tarsus_contact']['ms']), fontsize=10)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
    a = ax[0][1]
    lf = np.asarray(ts['physical__leg_fn'])
    for k, L in enumerate(LEGS):
        a.semilogy(t, np.where(lf[:, k] > 0, lf[:, k], np.nan), lw=1.1, label=L)
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('summed tarsus force (gap = no contact)')
    a.set_title('per-leg tarsus normal force', fontsize=10); a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3, which='both')
    a = ax[1][0]
    qd = np.asarray(ts['physical__qdev'])
    labs = [str(x) for x in np.asarray(ts['joint_labels'])]
    for k in np.argsort(-qd.max(axis=0))[:6]:
        a.semilogy(t, np.maximum(qd[:, k], 1e-12), lw=1.2, label=labs[k])
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('|q - q(0)| [rad]')
    a.set_title('six most-moving joints; first past 0.01 rad %s at step %d = %.6f ms'
                % (rep['localization']['first_joint_past_0_01_rad']['joint'], rep['localization']['first_joint_past_0_01_rad']['step'],
                   rep['localization']['first_joint_past_0_01_rad']['ms']), fontsize=10)
    a.legend(fontsize=7); a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    lu = np.asarray(ts['physical__leg_ud'])
    lo, hi = lok['window_steps']
    tlook = [i * DT * 1000.0 for i in range(lo, hi + 1)]
    for k, L in enumerate(LEGS):
        a.plot(tlook, [np.nan if np.isnan(lu[i, k]) else float(lu[i, k]) for i in range(lo, hi + 1)], marker='.', lw=1.2, label=L)
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('measured d.time [ms] — 20-step lookback window = 0.5 ms')
    a.set_ylabel('max u_diamond (broken = no contact)')
    a.set_title('u_diamond in the 20 steps before the first non-foot contact', fontsize=10)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
    fig.suptitle('corrected localization at the enforced 25 us step, same run as the open loop\n'
                 'declared_timestep %.1e | xml_timestep %.1e | actual_timestep %.1e | measured command cadence %.15f s | d.time final %.6f s\n'
                 'manual research implementation, NOT a product; no standing/walking claim'
                 % (rep['declared']['declared_timestep'], rep['declared']['xml_timestep'], rep['declared']['actual_timestep'],
                    rep['declared']['measured_command_cadence_s'], rep['declared']['d_time_final_s']), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    b = sha(CANON / 'localization-25us-v2.png')
    fig.savefig(CANON / 'localization-25us-v2.png', dpi=135)
    plt.close(fig)
    ledger.append(dict(file=str(CANON / 'localization-25us-v2.png'), sha256_before=b, sha256_after=sha(CANON / 'localization-25us-v2.png'),
                       changed=True, changed_at=now(),
                       reason='m5/c1: suptitle now carries the three dt fields and the measured cadence; bottom-right axis labelled "20-step lookback = 0.5 ms"',
                       changed_by='the model on the owner\'s instruction'))

    # (e) frozen manifest really covering every file + EV-04 for the window rewrite
    deliver_before = sha(CANON / 'delivery-hashes.json')
    files = sorted(p for p in CANON.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
    (CANON / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    write_frozen(CANON)
    ledger.append(dict(file=str(CANON / 'delivery-hashes.json'), sha256_before=deliver_before,
                       sha256_after=sha(CANON / 'delivery-hashes.json'), changed=True, changed_at=now(),
                       reason='errata round: manifest refreshed after the in-place edits (this is the write the FAIL verification flagged as B1)', changed_by='the model on the owner\'s instruction'))

    # ---------------- record batch ----------------
    def fix_rec_readme(t):
        return (t.replace('7cb99d678eb31b8262f049f98ca20a3cdc2638d7c2a8dcc9732cd8517e93206e',
                          '7cb99d67…（已撤回；该值是 void 批次 A 的 accepted-equilibrium-openloop.png 哈希）')
                 .replace('`%s`（v1 `delivery-hashes.json` 的 `delivery_time_hashes` 列）' % 'f09d6dec41ee3efdbb1edf93a6be7bfb02b28467c6d704737e5deec161b84fcf',
                          '`f09d6dec41ee3efdbb1edf93a6be7bfb02b28467c6d704737e5deec161b84fcf`（v1 `delivery-hashes.json` 的 `delivery_time_hashes` 列，17:01:26）'))
    edit(REC / 'README.zh-CN.md', fix_rec_readme, 'M3: the retracted wrong hash is no longer published as a value; the corrected pair stands')

    def fix_rec_json(t):
        d = json.loads(t)
        d['self_amendment']['file_sha256_before_note'] = ('the before value b6ec194b could not be re-verified because the file was overwritten in place; that is '
                                                          'disclosed here rather than reconstructed')
        d['ev03_scope_extension'] = dict(
            note='EV-03 originally listed only the two READMEs; the same 17:13 window also rewrote both void batches report.json (17:13:45) and their two VOID '
                 'figure pages (17:13:56), and the 172131 edits of c1..c4 were merged into one disposition line',
            additionally_rewritten=[str(VOID_A / 'report.json'), str(VOID_B / 'report.json'),
                                    str(VOID_A / 'accepted-equilibrium-openloop.png'), str(VOID_B / 'accepted-equilibrium-localization.png'),
                                    str(DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131/report.json'),
                                    str(DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131/README.zh-CN.md')],
            corrections_round_updates=dict(
                v3_timing=('the 76.375 ms correction first appeared in S2-equilibrium-A-corrected-reissued-v3-20261015T170319/ms_label_correction.json at 17:11:52, '
                           'then in that batch contact_identity_sentence_canonical.json at 17:20:15; the earlier claim that v3 carried it at 17:12:44 referred to its '
                           'README and was imprecise'),
                step_3055_attribution=('step 3055 is the external artifact own convention (its executed-step index); in this batch convention the same event is index '
                                       '3056 = 76.400 ms. The wording now states which convention each number uses'),
                empty_directory_deletion=('the zero-file directory S2-accepted-equilibrium-25us-corrected-20261015T172112 was removed after recording its empty state; '
                                          'that removal is recorded here as part of the integrity events'),
                current_directories_scope=('current_directories.json covers this work line only; it is not a statement about unrelated directories')))
        return json.dumps(d, ensure_ascii=False, indent=1)

    edit(REC / 'ev02_before_state_correction.json', fix_rec_json, 'm2/m3/m4/m6: EV-03 scope extended and the three attribution corrections recorded')
    rec_files = sorted(p for p in REC.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
    (REC / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in rec_files}, ensure_ascii=False, indent=1))
    ledger.append(dict(file=str(REC / 'delivery-hashes.json'), sha256_before=None, sha256_after=sha(REC / 'delivery-hashes.json'),
                       changed=True, changed_at=now(), reason='m1: the record batch own hash refreshed after its in-place edits',
                       changed_by='the model on the owner\'s instruction'))
    write_frozen(REC)

    # ---------------- events file: no "untraceable" wording left ----------------
    if EVENTS.exists():
        def fix_events(t):
            d = json.loads(t)
            for e in d.get('events', []):
                if e.get('id') == 'EV-02':
                    if isinstance(e.get('sha256_before'), str) and 'NOT RECORDED' in e['sha256_before']:
                        e['sha256_before'] = []
                    e.pop('sha256_before_note', None)
                    e['sha256_before'] = e.get('sha256_before') or [
                        dict(value='f09d6dec41ee3efdbb1edf93a6be7bfb02b28467c6d704737e5deec161b84fcf',
                             source=str(V1 / 'delivery-hashes.json'), field='delivery_time_hashes["integrity_incident_assembler.json"]',
                             recorded_at='2026-10-15 17:01:26'),
                        dict(value='5bf3593f8a88e6a6e49a1bb4c465898b1a595d7fed7c8bc2bfd5be8c57418b90',
                             source=str(DESK / 'S2-openloop-pair-check-20261015T235120Z/hash_manifest_check.json'),
                             field='manifest.delivery.integrity_incident_assembler.json.actual (match=false)', measured_at='2026-10-15 16:55')]
                    e['unpaired_write'] = ('the two pre-rewrite hashes differ, so exactly one write between the delivery-time record and the 17:01:26 rewrite is '
                                           'unpaired; it is recorded as unpaired rather than reconstructed')
            d['events'].append(dict(id='EV-04', when='2026-10-15 17:31:11', kind='errata round rewrote a file of a batch under verification',
                                    file=str(CANON / 'delivery-hashes.json'),
                                    value_before='cf5708c86b1cae97e3ee9084ab5f10bd91e77fc59fc87a4f06b9222411260c53 (measured by the verification at 17:29:33)',
                                    value_after=sha(CANON / 'delivery-hashes.json'),
                                    reason='the owner ordered an in-place errata round while that verification window was open',
                                    disposition='recorded; the frozen manifest of the canonical batch now covers every file including delivery-hashes.json'))
            return json.dumps(d, ensure_ascii=False, indent=1)
        edit(EVENTS, fix_events, 'owner item 2 plus the FAIL findings B1/M4: EV-02 carries two measured pre-rewrite hashes and EV-04 records the window rewrite')

    # ---------------- errata note about the void batches (history files untouched) ----------------
    note = dict(status='ERRATA_NOTE_NOT_AN_EDIT', manual_research_not_product=True,
                observed_and_recorded_only=[
                    dict(what='both void batches keep a historical PASS_* top-level status string', why='history files are not rewritten beyond the void flag '
                          'already added; their reading is governed by protocol_void'),
                    dict(what='their own frozen manifests went stale when the VOID lines were written at 17:13', why='not repaired in place; recorded in EV-03/EV-03 extension'),
                    dict(what='the v2 batch delivered before this round had a frozen manifest that did not cover delivery-hashes.json while claiming it did',
                         why='corrected in this round: the manifest now covers every file and the false coverage claim is gone')],
                recorded_at=now())
    (out / 'errata_note.json').write_text(json.dumps(note, ensure_ascii=False, indent=1))
    (out / 'errata_ledger.json').write_text(json.dumps(dict(status='ERRATA_LEDGER_2', entries=ledger, n=len(ledger),
                                                            thorax_facts=facts, recorded_at=now()), ensure_ascii=False, indent=1))
    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# errata-2：owner 五项 ＋ FAIL 核验（B1/M1–M4/m1–m6）逐条就地落地（手动研究实施，不是成品）', '',
        '- **M1**：规范的 README 里"本批未出现 `Thorax_geom|floor` 接触"这句**假陈述**已删除，改为实测：首个接触实例 步 %s（%s ms）、'
        '步 %s 起 `dist<0` 真穿透、步 %s = %s ms 起 `fn>0` 承载、最大 `fn = %s`、实例 %s 次。'
        % (facts['first_contact_instance_step'], facts['first_contact_instance_ms'], facts['first_true_penetration_step'],
           facts['first_load_bearing_step'], facts['first_load_bearing_ms'], facts['max_normal_force_model_units'], facts['contact_instances']),
        '- **M2**：姿态 A 对照的 caveat 撤回 —— 两跑**都是 `act = 0`**、同 pose npz、同驱动、同 25 µs，序列满足 `mine[i] == external[i-1]`（19999/19999，max|Δ|=0），'
        '**同一次物理事件**（本批口径步 2399 = 59.975000 ms）；外部标签 59.950 ms 是旧 runner 里的硬编码常数（引自独立核验 `S2-batch-check4-20261015T210312Z`）。',
        '- **M3/m1/m2/m3/m4/m6**：记录批 README 不再刊出已撤回的错哈希；其自身哈希已刷新；EV-03 扩展列出同窗口被改的 `report.json` 与两张 VOID 图页；'
        'v3 时间归属、3055 编号约定、空目录删除、`current_directories` 范围均已更正并写入记录。',
        '- **B1/M4**：规范件冻结清单**现已真正覆盖每个交付文件（含 `delivery-hashes.json`）**，并新增 **EV-04** 记录"核验窗口内被 errata 轮改写"这件事'
        '（before `cf5708c8…` 由核验方在 17:29:33 实测）。',
        '- **owner 第 4 项**：`max u_diamond` 两口径并列（跗节口径 1.0000000000000002 / 18143 步 ≥1；全接触口径 1.0000000000000002 / 19549 步 ≥1）。',
        '- **owner 第 3 项**：v2 的 9 条 stale 哈希 + ladder 1 条只写 re-anchor 记录（新旧并列 + 时刻 + 原因），**未再改写任何 sealed run 目录的 report.json 与图**。',
        '- **owner 第 5 项**：两个 VOID 批次的顶层 `PASS_*` 与清单失效**只写进本说明**，历史文件不再动。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。']))
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_AT_WRITE', frozen_at=now(),
                                                                    files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print('errata dir:', out)
    print('thorax facts:', json.dumps(facts, ensure_ascii=False))
    for e in ledger:
        print('   %-70s %s -> %s (%s)' % (pathlib.Path(e['file']).parent.name + '/' + pathlib.Path(e['file']).name,
                                          (e['sha256_before'] or '-')[:8], (e['sha256_after'] or '-')[:8], 'changed' if e['changed'] else 'unchanged'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
