"""Turn-32 in-place fix (errata-3) for the drive-sensitivity probe batch and the errata-2 ledger.

Two objects, one round:

A) /home/sensen/Desktop/engine修复验收/S2-drive-sensitivity-probe-20261015T174405/
   1. README.zh-CN.md   : the blanket "not a delicate amplitude or phase condition" reading is replaced by the
                          measured contrast (x0.95 / x1.05 move the first event by 20-21 steps, while a single
                          frame = 5 ms misalignment moves it by 328-557 steps) and by the step-attribution source
                          (report.json max_abs_qacc_step; the reference value is independently confirmed by the
                          canonical npz physical__qacc argmax = 3011).
   2. FROZEN_DELIVERY_HASHES.json : the `covers` claim said "every delivered file including delivery-hashes.json"
                          while its `files` map omitted delivery-hashes.json (and the README). The claim is now
                          explicit and enumerated, the map is the plain truth, and the errata ledger is declared
                          as not covered.
   3. delivery-hashes.json refreshed AFTER the in-place edits (exactly one write, at the end of the round).

B) /home/sensen/Desktop/engine修复验收/S2-errata-2-20261015T174214/errata_ledger.json
   Append the missing canonical-README entry: errata-2's first run edited the canonical README's M1 text and then
   crashed before writing its ledger, so that file's pre-run bytes were overwritten within the failed run and its
   before-hash is NOT recoverable. The entry recorded today says sha256_before == sha256_after, which is true of
   the surviving content but must not be read as proof that no write happened inside the failed run.

Every edit records before/after sha256. Nothing is re-anchored: no sealed run directory is touched.
"""

import hashlib
import json
import pathlib
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
PROBE = DESK / 'S2-drive-sensitivity-probe-20261015T174405'
ERRATA2 = DESK / 'S2-errata-2-20261015T174214'
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')

def fix(doc, old, new):
    assert doc.count(old) == 1, 'expected exactly one occurrence, found %d:\n%s' % (doc.count(old), old[:200])
    return doc.replace(old, new, 1)


def main():
    if (PROBE / 'errata_ledger.json').exists():
        raise SystemExit('refusing to run twice: %s already exists' % (PROBE / 'errata_ledger.json'))
    ledger = []

    def record(p, before, after, reason):
        ledger.append(dict(file=str(p), sha256_before=before, sha256_after=after,
                           changed=before != after, changed_at=now(), reason=reason,
                           changed_by='the model on the owner\'s instruction'))

    # ---------------- A1: probe README reading ----------------
    p = PROBE / 'README.zh-CN.md'
    b = sha(p)
    t = p.read_text(encoding='utf-8')
    old = ('- the first non-foot contact happens in every group inside a narrow band around the reference step, so the near-term collapse is decided by '
           'the archive-plus-state pair rather than by a delicate amplitude or phase condition; the numbers above state the band explicitly. The '
           'root z minimum and the u_diamond peak are reported per group as well.')
    new = ('- scaling the whole archive by 0.95 / 1.05 moves the first non-foot contact by only 20 / 21 steps (0.50 / 0.525 ms), while misaligning it by a '
           'single frame (5 ms, the archive cadence) moves it by 557 / 328 steps: the ~70 ms collapse is fixed by the archive-plus-state pair, and within '
           'that pair it is far more sensitive to frame alignment than to fine drive amplitude. The band is stated by the numbers above; the root z minimum '
           'and the u_diamond peak are reported per group as well.\n'
           '- `max|qacc|` 的步号出处：`report.json` 的逐组 `max_abs_qacc_step`（本批 npz 只存幅值、不存步号）。参考组 `24303.06890766346`（步 3011，75.275 ms）\n'
           '  由既有规范件 `%s/time_series_25us.npz` 的 `physical__qacc` 独立确认（argmax 步号同为 3011）；\n'
           '  错位组步 3208 / 2813 目前只有本批 `report.json` 一处来源，尚未有第二次独立复算。'
           % str(CANON))
    t = fix(t, old, new)
    assert 'max_abs_qacc_step' in t
    p.write_text(t, encoding='utf-8')
    record(p, b, sha(p), 'the blanket "not a delicate phase condition" reading contradicted this batch own numbers; replaced by the measured '
                         'amplitude-vs-phase contrast, and the max|qacc| step attribution now names its source and its one independent confirmation')

    # ---------------- A2: frozen manifest truth ----------------
    p = PROBE / 'FROZEN_DELIVERY_HASHES.json'
    b = sha(p)
    fz = json.loads(p.read_text(encoding='utf-8'))
    files = dict(fz.get('files', {}))
    covered = ['README.zh-CN.md', 'delivery-hashes.json', 'drive-sensitivity.png', 'drive_sensitivity.npz', 'report.json', 'source_snapshot.zip']
    missing = [f for f in covered if f not in files]
    files['delivery-hashes.json'] = sha(PROBE / 'delivery-hashes.json')
    fz['files'] = {k: files[k] for k in covered}
    fz['covers'] = 'the 6 files listed in `files` (README.zh-CN.md, delivery-hashes.json, drive-sensitivity.png, ' \
                   'drive_sensitivity.npz, report.json, source_snapshot.zip); declared NOT covered: this manifest itself ' \
                   'and errata_ledger.json (the ledger is written after the freeze, by construction)'
    fz['covers_count'] = len(fz['files'])
    fz['manifest_corrected_at'] = now()
    fz['manifest_correction'] = 'the previous `covers` string claimed "every delivered file including delivery-hashes.json" while the ' \
                                '`files` map omitted delivery-hashes.json: claim and map now agree, and the excluded files are named. ' \
                                'Files that were already listed keep their original hashes (missing from the old map: %s).' % (', '.join(missing) or 'none')
    p.write_text(json.dumps(fz, ensure_ascii=False, indent=1), encoding='utf-8')
    record(p, b, sha(p), 'false coverage claim in the frozen manifest (files map omitted delivery-hashes.json while covers claimed it); '
                         'claim and map reconciled, exclusions named')

    # ---------------- B: errata-2 ledger, missing README entry ----------------
    p = ERRATA2 / 'errata_ledger.json'
    b = sha(p)
    led = json.loads(p.read_text(encoding='utf-8'))
    entry = dict(
        file=str(CANON / 'README.zh-CN.md'),
        sha256_before=None,
        sha256_before_recoverable=False,
        sha256_of_surviving_content='8a2702105a299707d9303795d5159ed198b6874040326853a4b924c3b0a93769',
        sha256_after='8a2702105a299707d9303795d5159ed198b6874040326853a4b924c3b0a93769',
        changed=None,
        changed_at='2026-10-15 17:41-17:42 window (errata-2 first run); recorded retroactively at ' + now(),
        reason='M1: the canonical README\'s false "no thorax contact" sentence was replaced by the measured thorax facts in errata-2\'s FIRST run, '
               'which then died on an unrelated KeyError before writing any ledger. The pre-run bytes of this file were therefore overwritten '
               'without a paired before-hash and the true before-state is NOT reconstructible.',
        read_this_as='sha256_before is unknown, not equal to sha256_after. The entry deliberately does not claim "unchanged": the file was '
                     'written inside a failed run; only the surviving content (identical to sha256_after, which equals the post-errata-1 '
                     'content) is established. No backup of the pre-errata-2 README exists on disk.',
        changed_by='the model on the owner\'s instruction')
    led['entries'].append(entry)
    led['n'] = len(led['entries'])
    led['unpaired_write'] = dict(
        what='one write to the canonical README inside errata-2\'s failed first run',
        file=str(CANON / 'README.zh-CN.md'),
        before_hash='NOT RECOVERABLE (the run overwrote it, then crashed before recording)',
        after_hash='8a2702105a299707d9303795d5159ed198b6874040326853a4b924c3b0a93769',
        why_unrecoverable='errata-2 run 1 mutated the file in place and exited with KeyError before the ledger write; no copy, no backup and no '
                          'snapshot of the pre-run bytes exists in this session',
        consequence='the M1 correction to the canonical README cannot be point-for-point audited; it is disclosed here instead of being hidden '
                    'behind an "unchanged" flag')
    led['ledger_corrected_at'] = now()
    p.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding='utf-8')
    record(p, b, sha(p), 'appended the missing canonical-README entry and an explicit unpaired-write disclosure; the previous entry showed '
                         'sha256_before == sha256_after, which could be misread as "nothing was written"')

    # ---------------- A3: refresh the probe manifest (one write, last) ----------------
    p = PROBE / 'delivery-hashes.json'
    b = sha(p)
    files = sorted(q for q in PROBE.iterdir() if q.is_file() and q.name != 'FROZEN_DELIVERY_HASHES.json')
    p.write_text(json.dumps({q.name: sha(q) for q in files}, ensure_ascii=False, indent=1), encoding='utf-8')
    record(p, b, sha(p), 'refreshed after the in-place errata edits of this round (the only write to this file in the round)')

    # keep the frozen manifest in step with the refreshed manifest, then freeze the ledger as not-covered
    p = PROBE / 'FROZEN_DELIVERY_HASHES.json'
    b = sha(p)
    fz = json.loads(p.read_text(encoding='utf-8'))
    fz['files']['delivery-hashes.json'] = sha(PROBE / 'delivery-hashes.json')
    fz['not_covered'] = ['FROZEN_DELIVERY_HASHES.json (this file)', 'errata_ledger.json (written after the freeze)']
    fz['refreshed_after_manifest_write_at'] = now()
    fz['files'] = {k: fz['files'][k] for k in fz['files']}   # keep the insertion order of the first pass
    p.write_text(json.dumps(fz, ensure_ascii=False, indent=1), encoding='utf-8')
    record(p, b, sha(p), 'delivery-hashes.json refreshed after it was itself listed; the frozen digest follows the refreshed manifest')

    out = PROBE / 'errata_ledger.json'
    doc = dict(status='ERRATA_LEDGER_3', manual_research_not_product=True,
               scope='drive-sensitivity probe batch + errata-2 ledger (objects of turn 32)',
               entries=ledger, n=len(ledger), recorded_at=now(),
               note='the frozen manifest and this ledger are the two files excluded from the frozen `files` map, by construction; '
                    'the probe README and the errata-2 ledger are corrected in place, the errata-2 ledger is the other object of this round')
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')

    print('ERRATA_LEDGER_3 entries:', len(ledger))
    for e in ledger:
        print('  %-70s %s -> %s' % (pathlib.Path(e['file']).name, (e['sha256_before'] or 'None')[:12], e['sha256_after'][:12]))
    print('ledger:', out)
    print('final delivery-hashes.json:')
    print((PROBE / 'delivery-hashes.json').read_text(encoding='utf-8'))
    print('final FROZEN files map:')
    print(json.dumps(json.loads((PROBE / 'FROZEN_DELIVERY_HASHES.json').read_text(encoding='utf-8'))['files'], ensure_ascii=False, indent=1))
    return doc


if __name__ == '__main__':
    main()
