"""Turn 22, part A2: re-render the void pages in ASCII (the Chinese text lacked glyphs) and add the historical marker to the
old S2-batch series directories that have no README at all.
"""

import hashlib
import json
import pathlib
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
FAIL_A = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
FAIL_B = DESK / 'S2-accepted-equilibrium-localization-20261015T170058'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
rows = []

VOID_PAGE = ('PROTOCOL VOID - every physical reading of this batch is void.\n\n'
             'The script never set m.opt.timestep, so the model integrated at the canonical XML value 1e-04 s:\n'
             'the declared "25 us / 0.5 s" does not hold, and no millisecond label, time axis, event line or\n'
             'window length may be quoted. Step indices and raw values remain usable as history.\n\n'
             'The original figure was replaced by this page on the owner\'s instruction; the raw arrays are still\n'
             'in time_series.npz and report.json.\n\n'
             'manual research implementation, not a product.')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for d, label in ((FAIL_A, 'FAIL_batch_A'), (FAIL_B, 'FAIL_batch_B')):
        for name in ('accepted-equilibrium-openloop.png', 'openloop-zoom-evidence.png', 'accepted-equilibrium-localization.png'):
            f = d / name
            if not f.exists():
                continue
            before = sha(f)
            fig, ax = plt.subplots(figsize=(11.0, 6.2))
            ax.axis('off')
            ax.text(0.5, 0.5, VOID_PAGE, ha='center', va='center', fontsize=12)
            fig.suptitle('%s - VOID page (%s)' % (name, label), fontsize=10)
            fig.tight_layout()
            fig.savefig(f, dpi=120)
            plt.close(fig)
            rows.append(dict(record_class='protocol_void_figure', file=str(f), sha256_before=before, sha256_after=sha(f),
                             changed_at=now(), changed_by='the model on the owner\'s instruction',
                             reason='void page replacing a figure whose axis/event lines used the false 25 us clock'))

    marked = []
    for d in sorted(DESK.glob('S2-batch-*')):
        target = d / 'README.md'
        if target.exists():
            t = target.read_text()
            if t.startswith('**含已证伪结论'):
                continue
            before = sha(target)
            target.write_text('**含已证伪结论，仅作历史（historical only: contains refuted conclusions）**\n\n' + t)
        else:
            before = None
            target.write_text('**含已证伪结论，仅作历史（historical only: contains refuted conclusions）**\n\n'
                              '本目录内容未删改；此行由 2026-10-15 的收口轮加上，防止被误引为当前结论。\n')
        marked.append(dict(record_class='historical_marker', file=str(target), sha256_before=before, sha256_after=sha(target),
                           changed_at=now(), changed_by='the model on the owner\'s instruction',
                           reason='mark the refuted-conclusion series as historical only'))
    p = DESK / 'S2-accepted-equilibrium-openloop-CORRECTED-REISSUED-25us-20261015T172000/correction_ledger_consolidated.json'
    if p.exists():
        doc = json.loads(p.read_text())
        doc['correction_records'].extend(rows + marked)
        doc['recorded_at'] = now()
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print('void pages rewritten:', len(rows), '| historical markers:', len(marked))
    for m in marked:
        print('   ', m['file'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
