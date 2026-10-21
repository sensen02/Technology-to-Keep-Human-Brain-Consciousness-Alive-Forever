"""Turn 22, part A: land the protocol-void statement (report + README + figure titles) in the two voided batches, add the
historical marker to the old S2-batch series, and build ONE consolidated, traceable correction ledger.

Single writer: this script is the only writer in this tree while it runs.
"""

import hashlib
import json
import pathlib
import time

import numpy as np

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
FAIL_A = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
FAIL_B = DESK / 'S2-accepted-equilibrium-localization-20261015T170058'
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
NEW = DESK / 'S2-accepted-equilibrium-openloop-CORRECTED-REISSUED-25us-20261015T172000'
BATCH_DIRS = sorted(DESK.glob('S2-batch-*'))
VOID = ('PROTOCOL VOID: 本批次的物理口径全部作废 —— 脚本未设 m.opt.timestep，实际按 canonical XML 的 timestep=1e-04 积分，'
        '声明的 25 µs / 0.5 s 不成立；所有 ms 标注、时间轴、事件竖线、窗口长度一律不得引用；step 号与数值仍可作历史记录。')
VOID_EN = ('PROTOCOL VOID: every physical reading of this batch is void — the script never set m.opt.timestep, so the model integrated at the '
           'canonical XML value 1e-04 s and the declared 25 us / 0.5 s does not hold; no millisecond label, time axis, event line or window '
           'length may be quoted; step indices and raw values remain usable as history.')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
ledger_rows = []


def void_report(d, label):
    p = d / 'report.json'
    before = sha(p)
    r = json.loads(p.read_text())
    r['protocol_void'] = dict(cn=VOID, en=VOID_EN, voided_at=now(),
                              reason='script never set m.opt.timestep; the model integrated at the XML timestep 1e-04 s',
                              may_still_be_used=['step indices', 'raw values as history'],
                              must_not_be_used=['every millisecond label', 'time axes', 'event vertical lines', 'window lengths'])
    p.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    ledger_rows.append(dict(record_class='protocol_void', batch=label, file=str(p), sha256_before=before, sha256_after=sha(p),
                            changed_at=now(), changed_by='the model on the owner\'s instruction',
                            reason='declare the protocol void inside the batch report'))
    rmp = d / 'README.zh-CN.md'
    b = sha(rmp)
    t = rmp.read_text()
    if 'PROTOCOL VOID' not in t:
        rmp.write_text('**' + VOID + '**\n\n' + t)
        ledger_rows.append(dict(record_class='protocol_void', batch=label, file=str(rmp), sha256_before=b, sha256_after=sha(rmp),
                                changed_at=now(), changed_by='the model on the owner\'s instruction',
                                reason='void statement at the top of the README'))


def void_figures(d, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = json.loads((d / 'report.json').read_text())
    ts = np.load(d / 'time_series.npz')
    edges = list(ts.keys())
    tmax = float(np.max(ts[edges[0]])) if edges else 0.0
    for name in ('accepted-equilibrium-openloop.png', 'openloop-zoom-evidence.png', 'accepted-equilibrium-localization.png'):
        f = d / name
        if not f.exists():
            continue
        before = sha(f)
        fig, ax = plt.subplots(figsize=(11.0, 6.2))
        ax.axis('off')
        ax.text(0.5, 0.5, ('PROTOCOL VOID — 本批次物理口径全部作废\n\n'
                           '脚本未设 m.opt.timestep，实际按 canonical XML 的 timestep=1e-04 积分；\n'
                           '声明的 25 µs / 0.5 s 不成立，所有 ms 标注 / 时间轴 / 事件线 / 窗口长度一律不得引用。\n'
                           '（step 号与数值仍可作历史记录）\n\n'
                           '原图已按本行要求替换为说明页；原始数值仍在 time_series.npz 与 report.json。\n'
                           'manual research, not a product.'),
                  ha='center', va='center', fontsize=13, wrap=True)
        fig.suptitle('%s — voided figure (%s)' % (name, label), fontsize=10)
        fig.savefig(f, dpi=120)
        plt.close(fig)
        ledger_rows.append(dict(record_class='protocol_void', batch=label, file=str(f), sha256_before=before, sha256_after=sha(f),
                                changed_at=now(), changed_by='the model on the owner\'s instruction',
                                reason='figure superseded by a void page: its event lines and axis used the false 25 us clock'))


def historical_marker(d):
    rmp = d / 'README.zh-CN.md'
    if not rmp.exists():
        return None
    b = sha(rmp)
    t = rmp.read_text()
    line = '**含已证伪结论，仅作历史（historical only: contains refuted conclusions）**\n\n'
    if t.startswith('**含已证伪结论'):
        return None
    rmp.write_text(line + t)
    ledger_rows.append(dict(record_class='historical_marker', batch=d.name, file=str(rmp), sha256_before=b,
                            sha256_after=sha(rmp), changed_at=now(), changed_by='the model on the owner\'s instruction',
                            reason='one line at the top so the directory cannot be mis-cited'))
    return str(rmp)


def main():
    NEW.mkdir(exist_ok=True)
    void_report(FAIL_A, 'FAIL_batch_A')
    void_report(FAIL_B, 'FAIL_batch_B')
    void_figures(FAIL_A, 'FAIL_batch_A')
    void_figures(FAIL_B, 'FAIL_batch_B')
    marked = [m for m in (historical_marker(d) for d in BATCH_DIRS) if m]

    # consolidated correction ledger (unified field names)
    sources = [V2 / 'correction_ledger.json', V2 / 'hash_realignment.json', V2 / 'post_seal_amendment.json',
               V2 / 'scope_incident_inplace_outside_v1.json', V1 / 'CHANGELOG-inplace.md']
    consolidated = []
    cl = json.loads((V2 / 'correction_ledger.json').read_text())
    for r in cl['stale_fields_v2'] + cl['stale_fields_ladder']:
        consolidated.append(dict(record_class='post_correction_hash', file=r['file'], field=r['field'],
                                 sha256_before=r['old_sha256'], sha256_after=sha(r['file']),
                                 changed_at='2026-10-15 16:36-16:43 (and 16:45 / 17:0x)', changed_by='the model',
                                 reason='authorised in-place correction of run template text / diagnostic wording / F-A wording',
                                 anchor_status='STALE — needs re-anchoring'))
    for r in json.loads((V2 / 'scope_incident_inplace_outside_v1.json').read_text())['what_was_touched']:
        consolidated.append(dict(record_class='scope_incident', file=r['path'], field='whole file',
                                 sha256_before=r['before_sha256'], sha256_after=r['after_sha256'],
                                 changed_at=r['changed_at'], changed_by=r['changed_by'],
                                 reason='edit reached outside the batch directory during a verification window',
                                 anchor_status='STALE — needs re-anchoring'))
    consolidated.extend(ledger_rows)
    all_rows = []
    anchor_files = [V1 / 'report.json', V1 / 'README.zh-CN.md', V2 / 'report.json', V2 / 'README.zh-CN.md',
                    LADDER / 'README.zh-CN.md', V1 / 'equilibrium_best_state.npz', FAIL_A / 'report.json', FAIL_B / 'report.json']
    for f in anchor_files:
        all_rows.append(dict(record_class='run_time_anchor', file=str(f), field='—', sha256_before=None, sha256_after=sha(f),
                             changed_at=now(), changed_by='nobody (recorded for re-anchoring)', reason='current value, recorded so the anchors '
                             'can be re-established after all writing stops', anchor_status='CURRENT (recorded post-write)'))
    ledger = dict(status='CONSOLIDATED_CORRECTION_LEDGER', manual_research_not_product=True,
                  single_writer_rule='one writer per tree at a time; no background agent may write the project root or a delivery directory',
                  field_names=dict(file='full path', field='json field or "whole file"', sha256_before='value before the change (null if none)',
                                   sha256_after='value after the change', changed_at='local time', changed_by='who changed it',
                                   reason='one sentence', anchor_status='STALE / CURRENT / VOID'),
                  correction_records=consolidated, run_time_anchors=all_rows,
                  stale_anchor_summary=dict(n=sum(1 for r in consolidated if str(r.get('anchor_status', '')).startswith('STALE')),
                                            note='every stale anchor has its before/after pair here; the CURRENT rows in run_time_anchors are '
                                                 'the values to re-anchor against once all writing has stopped'),
                  superseded_records=[str(s) for s in sources], recorded_at=now())
    (NEW / 'correction_ledger_consolidated.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=1))
    print('voided report+figures:', FAIL_A.name, FAIL_B.name)
    print('historical markers added:', len(marked))
    for m in marked:
        print('   ', m)
    print('ledger records:', len(consolidated), '| run-time anchors:', len(all_rows))
    print('new batch dir (created, empty):', NEW)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
