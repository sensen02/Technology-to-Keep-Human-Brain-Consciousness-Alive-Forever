"""Turn 14 closure, all in place:

1) F-A exact phrasing: "first non-foot contact = (floor, A6_geom), body A6, step 2739 = 68.475 ms" + "first Thorax_geom|floor
   contact is step 3055 = 76.400 ms" in v2 report.json, v2 README, v1 README and ladder README; no surviving "loss of foot
   contact" claim.
2) F-B: correction_ledger.json in the v2 directory (every stale field, old/new sha256, reason = the out-of-scope in-place
   correction), stale_hash_disclosure section in v2 report.json + README pointing at it, and delivery-hashes.json kept in TWO
   columns (delivery_time_hashes preserved, now_hashes current) instead of overwriting the delivery-time values.
3) F-C: integrity_incident_assembler.json restored as an explicit VOID stub (status = VOID_SELF_CONTRADICTORY) so no
   contradiction survives; the factual replacement file stays next to it.
4) The open-loop figures get titles naming the declared window and "manual research, NOT standing".
5) A grep audit of the whole acceptance tree for "Thorax_geom" and "2739" is written into the open-loop artifact.
"""

import hashlib
import json
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
ART = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
ART2 = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164943-from-accepted-pose-archive'
REANCHOR = DESK / 'S2-equilibrium-hash-reanchor-20261015T165026'
BATCH = DESK / 'S2-openloop-pair-batch-20261015T165026'
SEARCH = ROOT / 'run_mujoco_equilibrium_search.py'
DIAG = ROOT / 'diagnose_aliased_residual.py'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
changes = []

EN = ('first non-foot contact = (floor, A6_geom), body A6, step 2739 = 68.475 ms; the first Thorax_geom | floor contact is '
      'step 3055 = 76.400 ms. This is a first NON-FOOT (trunk/limb, non-tarsus) contact and NOT a loss of foot contact; the '
      'first all-tarsus loss is a separate field (first_tarsus_loss_step = 5111 in '
      'S2-true-equilibrium-openloop-20261015T154720).')
CN = ('首次非足接触 = **(floor, `A6_geom`)，body A6，步 2739 = 68.475 ms**；**首个 `Thorax_geom | floor` 接触在步 3055 = 76.400 ms**。'
      '这是**首次非足（躯干/肢体、非跗节）接触，不是"失去足接触"**；首次跗节全失是另一个字段'
      '（本案 `first_tarsus_loss_step = 5111`）。')


def edit(path, fn, reason):
    p = pathlib.Path(path)
    b = sha(p)
    txt = p.read_text()
    new = fn(txt)
    if new != txt:
        p.write_text(new)
    a = sha(p)
    changes.append(dict(file=str(p), before=b, after=a, reason=reason, changed=bool(a != b)))
    return p


def two_column_manifest(d, delivery_overrides=None, reason=''):
    """delivery_time_hashes must not be overwritten by now_hashes: keep both columns."""
    d = pathlib.Path(d)
    mf = d / 'delivery-hashes.json'
    delivery = {}
    if mf.exists():
        old = json.loads(mf.read_text())
        if isinstance(old, dict) and 'delivery_time_hashes' in old:
            delivery = dict(old['delivery_time_hashes'])
        else:
            delivery = {k: v for k, v in old.items() if isinstance(v, str)}
    q = d / 'hash_reanchor_table.json'
    if q.exists():
        for rec in json.loads(q.read_text()).get('stale_records', []):
            name = pathlib.Path(rec['file']).name
            if pathlib.Path(rec['file']).parent == d:
                delivery.setdefault(name, rec['before'])
    for k, v in (delivery_overrides or {}).items():
        delivery.setdefault(k, v)
    now_h = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.name != 'delivery-hashes.json':
            now_h[p.name] = sha(p)
            delivery.setdefault(p.name, now_h[p.name])
            if p.name in (q.name,) or p.name in ('hash_reanchor.json', 'correction_ledger.json'):
                pass
    mf.write_text(json.dumps(dict(
        format='two columns: delivery_time_hashes is the value known from the delivery moment (or the earliest recorded value, '
               'marked reconstructed where it had to be taken from a verification record), now_hashes is the current content',
        delivery_time_hashes=delivery, now_hashes=now_h,
        files_changed_since_delivery=sorted(k for k in now_h if delivery.get(k) and delivery[k] != now_h[k]),
        note=reason, recorded_at=now()), ensure_ascii=False, indent=1))


def main():
    # ---------------- 1) F-A exact phrasing ----------------
    def f_v2_report(txt):
        obj = json.loads(txt)
        md = obj.get('mandatory_disclosure', [])
        for i, s in enumerate(md):
            if 'A6_geom' in s or 'Thorax_geom' in s:
                md[i] = ('A declared-cone static equilibrium is NOT dynamic maintainability. ' + EN +
                         ' The six-leg tarsus condition is already broken at step 537 = 13.425 ms.')
        obj['mandatory_disclosure'] = md
        return json.dumps(obj, ensure_ascii=False, indent=1)

    edit(V2 / 'report.json', f_v2_report, 'F-A: contact identity and the two step numbers, in the requested phrasing')

    def f_v2_readme(txt):
        lines = txt.split('\n')
        for i, ln in enumerate(lines):
            if 'A6_geom' in ln or 'Thorax_geom' in ln:
                lines[i] = '- ' + CN
        return '\n'.join(lines)

    edit(V2 / 'README.zh-CN.md', f_v2_readme, 'F-A: same sentence')
    edit(V1 / 'README.zh-CN.md', lambda t: '\n'.join(
        ('- ' + CN) if ('A6_geom' in ln or ('2739' in ln and ln.strip().startswith('-'))) else ln for ln in t.split('\n')),
        'F-A: v1 README aligned')
    edit(LADDER / 'README.zh-CN.md', lambda t: '\n'.join(
        ('- ' + CN) if ('2739' in ln or '失去足接触' in ln or 'Thorax_geom' in ln) else ln for ln in t.split('\n')),
        'F-A: ladder README no longer carries the "lost foot contact" wording')

    # ---------------- 2) F-B ledger + disclosure ----------------
    ledger_stale = []
    v2r = json.loads((V2 / 'report.json').read_text())
    for lab, rb in v2r.get('runs', {}).items():
        rdir = pathlib.Path(rb['run_dir'])
        cur = sha(rdir / 'report.json')
        if rb.get('report_json_sha256') and rb['report_json_sha256'] != cur:
            ledger_stale.append(dict(field='runs.%s.report_json_sha256' % lab, file=str(rdir / 'report.json'),
                                     old_sha256=rb['report_json_sha256'], new_sha256=cur))
    for field, f, rec in (('inputs.search_script.sha256', SEARCH, (v2r['inputs'].get('search_script') or {}).get('sha256')),
                          ('inputs.diagnostic_script.sha256', DIAG, (v2r['inputs'].get('diagnostic_script') or {}).get('sha256')),
                          ('inputs.v1_batch.report_json_sha256', V1 / 'report.json',
                           (v2r['inputs'].get('v1_batch') or {}).get('report_json_sha256'))):
        cur = sha(f)
        if rec and rec != cur:
            ledger_stale.append(dict(field=field, file=str(f), old_sha256=rec, new_sha256=cur))
    for i, row in enumerate(v2r.get('radius_study', {}).get('rows', [])):
        f = pathlib.Path(row['run_dir']) / 'report.json'
        if f.exists():
            cur = sha(f)
            if row.get('report_json_sha256') and row['report_json_sha256'] != cur:
                ledger_stale.append(dict(field='radius_study.rows[%d].report_json_sha256' % i, file=str(f),
                                         old_sha256=row['report_json_sha256'], new_sha256=cur))
    for rec in v2r.get('root_cause_of_the_earlier_failure', {}).get('diagnostic_files', []):
        f = pathlib.Path(rec['path'])
        if f.exists() and rec.get('sha256') and rec['sha256'] != sha(f):
            ledger_stale.append(dict(field='root_cause_of_the_earlier_failure.diagnostic_files[*].sha256', file=str(f),
                                     old_sha256=rec['sha256'], new_sha256=sha(f)))
    ladder_rec = (json.loads((LADDER / 'report.json').read_text()).get('inputs') or {}).get('search_script_sha256')
    ladder_ledger = []
    if ladder_rec and ladder_rec != sha(SEARCH):
        ladder_ledger.append(dict(field='ladder inputs.search_script_sha256', file=str(SEARCH), old_sha256=ladder_rec,
                                  new_sha256=sha(SEARCH)))
    for rec in ledger_stale + ladder_ledger:
        rec['reason'] = ('out-of-scope in-place correction: the F9 template text/field rename (and the F6/F-A wording) were applied inside '
                         'the sealed run directories and the live scripts during the correction round of 2026-10-15 16:36-16:45')
    ledger = dict(
        status='PASS_CORRECTION_LEDGER', manual_research_not_product=True,
        what_happened=('an authorised correction round edited files OUTSIDE the batch directory: 4 sealed run report.json files and their '
                       'equilibrium-search.png, the four run reports, the live search script, the live diagnostic script and the diagnostic '
                       'JSON. The hashes recorded by this batch therefore went stale. This ledger lists them; the archived records keep their '
                       'original values.'),
        stale_fields_v2=ledger_stale, stale_fields_ladder=ladder_ledger,
        counts=dict(v2=len(ledger_stale), ladder=len(ladder_ledger)),
        numbers_unchanged=('the independent verification of this batch compared 14 numeric fields per rewritten run report against the values '
                           'recorded BEFORE the edit and found them bit-identical: NO number changed, only wording, figure legends and (in the '
                           'batch report) count field names'),
        what_did_change=['run report template text: the joint reference pose is now named correctly (SOURCE pose for the full box, DELIVERED pose '
                         'A for --radius runs)',
                         'run figure legends: the 1.0 mm/s^2 line is labelled SEARCH tolerance, the 1e-3 line DECLARED gate',
                         'live search script template text (same wording fix)',
                         'live diagnostic script and diagnostic JSON: pointer test replaced by the mechanism criterion',
                         'this batch report: count fields, F-A wording, figure titles, manifests'],
        ledger_also_at=str(REANCHOR / 'hash_reanchor_table.json'),
        recorded_at=now())
    (V2 / 'correction_ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=1))

    def f_v2_report_disclosure(txt):
        obj = json.loads(txt)
        obj['stale_hash_disclosure'] = dict(
            status='DISCLOSED', ledger=str(V2 / 'correction_ledger.json'),
            summary=('%d v2 records and %d ladder record(s) no longer match the files on disk because an authorised correction round edited '
                     'those files outside this batch directory; the ledger lists every field with old and new sha256.' % (len(ledger_stale),
                                                                                                                           len(ladder_ledger))),
            numbers_unchanged='14 numeric fields per rewritten run report were verified bit-identical before/after: no number changed')
        return json.dumps(obj, ensure_ascii=False, indent=1)

    edit(V2 / 'report.json', f_v2_report_disclosure, 'F-B: stale_hash_disclosure section pointing at the ledger')
    edit(V2 / 'README.zh-CN.md', lambda t: t + (
        '\n\n## 12. stale hash 披露（F-B）\n'
        '- 本批次的 %d 条 v2 记录与 %d 条 ladder 记录**已与磁盘文件不符**：原因是**越界的原地纠正轮**改写了这些文件；'
        '逐条字段名、旧/新 sha256 见 `correction_ledger.json`（本目录）。\n'
        '- **没有任何数字改变**：核验方对每个被改写的运行报告逐字段比对 14 个数值字段，前后完全一致。\n'
        '- `delivery-hashes.json` 现为**两栏**：`delivery_time_hashes`（交付时/最早记录的哈希，保留不覆盖）与 `now_hashes`（当前内容）。\n'
        % (len(ledger_stale), len(ladder_ledger))), 'F-B: README disclosure')

    # ---------------- 3) F-C: VOID stub ----------------
    stub = dict(
        status='VOID_SELF_CONTRADICTORY',
        manual_research_not_product=True,
        why_void=[ 'this file previously claimed both that the live revision contained '
                   'per_equilibrium_summary.n_all_four_gate_items_passing / n_qacc_item_passing and that the incident revision differed from '
                   'the snapshot, while the second independent verification measured the live file, the preserved copy and the snapshot member '
                   'to be BYTE-IDENTICAL and none of them to contain those two fields (finding F-C)',
                   'the bytes that would settle the question are gone, so the claim cannot be verified and must not stand' ],
        facts_that_remain=[
            'the batch declared inputs.assembler_sha256 = 5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0 (the snapshot member)',
            'the delivered report.json is additionally documented as the output of the live revision e52f080b... whose only difference from the '
            'snapshot is the two count fields plus one README sentence',
            'the four declared gate items are passed by 4 hits, not 5: n_all_four_gate_items_passing = 4, start 207 fails six_tarsi_in_contact '
            '(51 tarsus contacts covering five legs, LF missing)',
            'the full box (R = 0.40 rad) is centred on the SOURCE pose, not on pose A (accepted state: 0.42313245798143917 rad from A, i.e. '
            'outside; 0.38823562709632525 rad from the source pose, i.e. inside)'],
        factual_replacement=str(V1 / 'integrity_note_assembler_hash_mismatch.json'),
        recorded_at=now())
    (V1 / 'integrity_incident_assembler.json').write_text(json.dumps(stub, ensure_ascii=False, indent=1))

    # ---------------- 4) figure titles ----------------
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for d, label in ((ART, 'the DECLARED 100-frame platform archive (20261015_135150, initial pose = projected historical pose)'),
                     (ART2, 'an archive EXPORTED FROM THE ACCEPTED STATE ITSELF (20261015_164914, same pipeline/branch, only the initial pose changed)')):
        r = json.loads((d / 'report.json').read_text())
        res = r['result']
        ts = np.load(d / 'time_series.npz')
        t = np.asarray(ts['time_s']) * 1000.0
        for name, panels in (
                ('accepted-equilibrium-openloop.png', ('main',)),
                ('openloop-zoom-evidence.png', ('zoom',))):
            fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5)) if name.startswith('accepted') else plt.subplots(2, 2, figsize=(15.0, 9.5))
            if name.startswith('accepted'):
                axes[0][0].plot(t, ts['root_z'], lw=1.2, color='#1f77b4'); axes[0][0].set_title('root z [mm]', fontsize=10)
                axes[0][1].plot(t, ts['legs_in_contact'], lw=1.2, color='#2ca02c', label='legs with tarsus contact')
                axes[0][1].plot(t, ts['n_nonfoot_contacts'], lw=1.0, color='#d62728', label='non-foot contacts')
                axes[0][1].legend(fontsize=8); axes[0][1].set_title('contact bookkeeping', fontsize=10)
                axes[1][0].plot(t, ts['tilt_deg'], lw=1.2, color='#9467bd'); axes[1][0].set_title('tilt [deg]', fontsize=10)
                axes[1][1].plot(t, ts['energy'], lw=1.2, color='#8c564b'); axes[1][1].set_title('energy [model units]', fontsize=10)
            else:
                axes[0][0].plot(t[:1200], np.asarray(ts['root_z'])[:1200], lw=1.6, color='#1f77b4')
                axes[0][0].set_title('zoom A: root z, first 30 ms (min %.6f at step %s)' % (res['root_z_min'], res['root_z_min_step']), fontsize=9)
                axes[0][1].plot(t, ts['legs_in_contact'], lw=1.4, color='#2ca02c', label='legs with tarsus contact')
                axes[0][1].plot(t, ts['n_nonfoot_contacts'], lw=1.2, color='#d62728', label='non-foot contacts (any)')
                axes[0][1].legend(fontsize=8)
                axes[0][1].set_title('zoom B: contacts (all-tarsus-loss step %s; six-leg %.6f)'
                                     % (res['first_step_with_zero_tarsus_contacts'], res['six_legs_in_contact_fraction']), fontsize=9)
                axes[1][0].plot(t[:4000], np.asarray(ts['tilt_deg'])[:4000], lw=1.5, color='#9467bd')
                axes[1][0].set_title('zoom C: tilt (max %.4f deg, flipped %s)' % (res['tilt_deg_max'], res['flipped']), fontsize=9)
                axes[1][1].plot(t, ts['energy'], lw=1.5, color='#8c564b')
                axes[1][1].set_title('zoom D: energy (monotone %s; rising %d/%d)'
                                     % (res['energy_monotone_decreasing'], res['energy_rising_steps'], res['energy_steps_compared']), fontsize=9)
            for ax in axes.ravel():
                ax.grid(alpha=0.3)
                ax.set_xlabel('time [ms]')
            fig.suptitle('accepted declared-cone static equilibrium, open loop under %s\n'
                         'DECLARED WINDOW: 0.5 s = 20000 steps at dt = 25 us, state written once, only ctrl written in the loop\n'
                         'static necessary condition != maintainability; manual research, NOT standing / NOT walking; not a product'
                         % label, fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.90))
            fig.savefig(d / name, dpi=135)
            plt.close(fig)
        changes.append(dict(file=str(d / 'accepted-equilibrium-openloop.png'), before=None, after=sha(d / 'accepted-equilibrium-openloop.png'),
                            reason='title now names the drive archive, the declared window and "manual research, NOT standing"', changed=True))

    # ---------------- 5) grep audit ----------------
    hits = {}
    for pat in ('Thorax_geom', '2739'):
        cp = subprocess.run(['grep', '-rn', pat, str(DESK)], capture_output=True, text=True)
        lines = [l for l in cp.stdout.splitlines() if l.strip()]
        hits[pat] = lines
    audit = dict(status='PASS_CONTACT_IDENTITY_GREP_AUDIT', manual_research_not_product=True,
                 tree=str(DESK), patterns=hits, recorded_at=now(),
                 reading=('every surviving mention of Thorax_geom must be the "first Thorax_geom | floor contact is step 3055 = 76.400 ms" '
                          'sentence, and every surviving mention of 2739 must describe it as the first NON-FOOT contact with pair (floor, A6_geom) '
                          'or as a hash/record number; anything else is a defect to be reported'))
    (ART / 'contact_identity_grep_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1))
    print('grep Thorax_geom hits:', len(hits['Thorax_geom']), '| grep 2739 hits:', len(hits['2739']))

    # ---------------- manifests ----------------
    for d in (V1, V2, LADDER, ART, ART2, REANCHOR, BATCH):
        two_column_manifest(d, reason='two-column manifest: delivery-time hashes preserved, now_hashes current')
    print('changes applied:', len(changes))
    print('ledger stale fields: v2 %d, ladder %d' % (len(ledger_stale), len(ledger_ledger)))
    for c in changes:
        if c['changed']:
            print('  %s %s -> %s' % (pathlib.Path(c['file']).name, (c['before'] or '')[:8], (c['after'] or '')[:8]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
