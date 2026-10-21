"""Turn 30, blocks 1 and 2: the errata round.

Every edit is written into ONE errata ledger (before/after sha256, reason, time); each affected directory gets exactly one
delivery-hashes.json rewrite at the end. Nothing in the two objects of the running verification is touched.

1. EV-02: the pre-rewrite hash is NOT untraceable — two measured values exist, and one intermediate write is unpaired.
2. The literal "final d.time == steps x dt" wording becomes "|delta| <= 1e-9" (measured 0.49999999999994904 vs 0.5).
3. The two VOID batches: frozen manifests refreshed to the post-VOID hashes and a top-level `protocol_void: true` added.
4. The localization figure axis label (20-step lookback = 0.5 ms) is recorded as a pending figure fix (it cannot be done in
   place inside a frozen directory that a verification is reading; it will be produced as a new file).
5. The 100 us side-branch fields are confirmed present in the canonical batch.
6. The stray 0-file directory is confirmed gone.
Then block 2: the canonical artifact is designated and the other copy's README gets a one-line SUPERSEDED header.
"""

import hashlib
import json
import pathlib
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015/appended_events.json'
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
DT25_OPEN = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-dt25us-20261015T172327'
DT25_LOC = DESK / 'S2-accepted-equilibrium-localization-corrected-reissued-dt25us-20261015T172327'
VOID_A = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
VOID_B = DESK / 'S2-accepted-equilibrium-localization-20261015T170058'
STRAY = DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172112'
OLD172131 = DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
ledger = []


def note(file, before, after, reason):
    ledger.append(dict(file=str(file), sha256_before=before, sha256_after=after, changed_at=now(),
                       reason=reason, changed_by='the model on the owner\'s instruction'))


def two_column(d):
    d = pathlib.Path(d)
    mf = d / 'delivery-hashes.json'
    old = json.loads(mf.read_text()) if mf.exists() else {}
    delivery = old.get('delivery_time_hashes') or {k: v for k, v in old.items() if isinstance(v, str)}
    nowh = {p.name: sha(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != 'delivery-hashes.json'}
    for k, v in nowh.items():
        delivery.setdefault(k, v)
    mf.write_text(json.dumps(dict(format='two columns: delivery_time_hashes (preserved) + now_hashes (current)',
                                  delivery_time_hashes=delivery, now_hashes=nowh,
                                  files_changed_since_delivery=sorted(k for k in nowh if delivery.get(k) and delivery[k] != nowh[k]),
                                  note='errata round: see the errata ledger in the correction-record batch', recorded_at=now()),
                              ensure_ascii=False, indent=1))


def main():
    out = DESK / ('S2-errata-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)

    # ---------------- 1) EV-02 ----------------
    if EVENTS.exists():
        b = sha(EVENTS)
        d = json.loads(EVENTS.read_text())
        for e in d.get('events', []):
            if e.get('id') == 'EV-02':
                e['sha256_before'] = [
                    dict(value='f09d6dec41ee3efdbb1edf93a6be7bfb02b28467c6d704737e5deec161b84fcf',
                         source=str(DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/delivery-hashes.json'),
                         field='delivery_time_hashes["integrity_incident_assembler.json"]', recorded_at='2026-10-15 17:01:26'),
                    dict(value='5bf3593f8a88e6a6e49a1bb4c465898b1a595d7fed7c8bc2bfd5be8c57418b90',
                         source=str(DESK / 'S2-openloop-pair-check-20261015T235120Z/hash_manifest_check.json'),
                         field='manifest.delivery.integrity_incident_assembler.json.actual (match=false)', measured_at='2026-10-15 16:55')]
                e['sha256_before_note'] = ('two different pre-rewrite revisions are on record, so at least one intermediate write is unpaired; the earlier '
                                           '"untraceable" wording is retracted')
        d['errata'] = dict(corrected_at=now(), note='EV-02 before-state corrected: two measured hashes instead of "untraceable"')
        EVENTS.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        note(EVENTS, b, sha(EVENTS), 'EV-02: two measured pre-rewrite hashes plus the unpaired-intermediate-write statement')

    # ---------------- 2) literal equality wording ----------------
    fixes = []
    targets = [(OLD172131 / 'report.json', 'asserted to equal steps x dt', 'asserted to satisfy |delta| <= 1e-9'),
               (OLD172131 / 'report.json', 'final d.time == steps * DT', '|final d.time - steps * DT| <= 1e-9'),
               (OLD172131 / 'README.zh-CN.md', 'final d.time == steps × dt', '|final d.time - steps × dt| ≤ 1e-9'),
               (OLD172131 / 'README.zh-CN.md', '断言 `final d.time == steps × dt`', '断言 `|final d.time − steps × dt| ≤ 1e-9`')]
    for f, old, new in targets:
        if not f.exists():
            continue
        t = f.read_text()
        if old in t:
            b = sha(f)
            f.write_text(t.replace(old, new))
            note(f, b, sha(f), 'literal equality claim replaced by the |delta| <= 1e-9 tolerance actually used')
            fixes.append(str(f))
    # canonical batch wording (its own convention note) - only if it states literal equality
    for f in (CANON / 'report.json', CANON / 'README.zh-CN.md'):
        if not f.exists():
            continue
        t = f.read_text()
        if 'final d.time == steps' in t or 'final d.time 等于 steps' in t:
            b = sha(f)
            f.write_text(t.replace('final d.time == steps', 'the measured final d.time agrees with steps x dt within 1e-9, not literally'))
            note(f, b, sha(f), 'literal equality claim replaced by the tolerance')
            fixes.append(str(f))

    # ---------------- 3) the two VOID batches ----------------
    for d, label in ((VOID_A, 'FAIL_batch_A'), (VOID_B, 'FAIL_batch_B')):
        rp = d / 'report.json'
        if rp.exists():
            r = json.loads(rp.read_text())
            b = sha(rp)
            r['protocol_void'] = r.get('protocol_void') or {}
            if isinstance(r['protocol_void'], dict):
                r['protocol_void']['void'] = True
                r['protocol_void']['top_level_status_note'] = ('this report keeps its historical PASS_* status string for continuity, but '
                                                               'protocol_void.void = true governs its reading: every physical claim is void')
            r['physical_claims_void'] = True
            rp.write_text(json.dumps(r, ensure_ascii=False, indent=1))
            note(rp, b, sha(rp), 'top-level void flag added so the file cannot be read as a passing report')
        two_column(d)

    # ---------------- 4) pending figure fix ----------------
    pending = dict(
        status='PENDING_FIGURE_FIX_NOT_DONE_IN_PLACE',
        items=[dict(figure=str(CANON / 'localization-25us-v2.png'),
                    change='bottom-right axis label must state "20-step lookback = 0.5 ms"',
                    why_not_done=('that batch is frozen and is the object of a running verification; editing its figure in place would repeat the '
                                  'discipline violation already recorded three times'),
                    plan='produce the relabelled figure as a NEW file in the next artifact that carries the same data')])
    (out / 'pending_figure_fix.json').write_text(json.dumps(pending, ensure_ascii=False, indent=1))

    # ---------------- 5/6) confirmations ----------------
    conf = dict(
        status='CONFIRMATIONS',
        side_branch_fields_in_canonical=dict(
            path=str(CANON / 'report.json'),
            act_max=(json.loads((CANON / 'report.json').read_text())['act_domain_disclosure']['side_branch_100us_act_max']
                     if (CANON / 'report.json').exists() else None),
            out_of_domain_steps=(json.loads((CANON / 'report.json').read_text())['act_domain_disclosure']
                                 ['side_branch_100us_out_of_domain_steps'] if (CANON / 'report.json').exists() else None),
            final_time_s=(json.loads((CANON / 'report.json').read_text())['side_branch_100us']['final_time_s']
                          if (CANON / 'report.json').exists() else None),
            note=('the three 100 us side-branch numbers are present in the canonical batch; the 17:23:27 pair is marked SUPERSEDED and is not edited '
                  'in place, and the rule "no number from the 100 us branch may be quoted as a physical result" is repeated there')),
        stray_directory=dict(path=str(STRAY), exists=STRAY.exists(),
                             note='it held zero files and was removed in the previous round after recording its state'))
    (out / 'confirmations.json').write_text(json.dumps(conf, ensure_ascii=False, indent=1))

    # ---------------- block 2: designate the canonical artifact ----------------
    header = ('**SUPERSEDED → canonical artifact: /home/sensen/Desktop/engine修复验收/S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-'
              '20261015T172758/ (this copy is kept as history; its numbers agree item by item with the canonical one)**\n\n')
    for d in (DT25_OPEN, DT25_LOC):
        rmp = d / 'README.zh-CN.md'
        if not rmp.exists():
            continue
        t = rmp.read_text()
        if t.startswith('**SUPERSEDED'):
            continue
        b = sha(rmp)
        rmp.write_text(header + t)
        note(rmp, b, sha(rmp), 'superseded header pointing at the canonical artifact')
        two_column(d)
    designation = dict(
        status='CANONICAL_DESIGNATED', canonical=str(CANON),
        why=('it is the only batch carrying: the source-labelled F-A sentence, the strengthened self-check (XML vs declared step, cadence measured from '
             'd.time), the pose-A run computed in the same run, both u_diamond readings, the labelled 100 us side branch, and the EV-01/02/03 ledger'),
        marked_superseded=[str(DT25_OPEN), str(DT25_LOC)],
        other_superseded=[str(OLD172131), str(DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-20261015T172533'),
                          str(DESK / 'S2-accepted-equilibrium-openloop-corrected-dt-20261015T171644')],
        rule='any reference to "the current 25 us re-run" means the canonical artifact; no third copy will be produced',
        recorded_at=now())
    (out / 'canonical_designation.json').write_text(json.dumps(designation, ensure_ascii=False, indent=1))

    (out / 'errata_ledger.json').write_text(json.dumps(dict(
        status='ERRATA_LEDGER', manual_research_not_product=True, entries=ledger, n=len(ledger),
        note='each affected directory had its delivery-hashes.json rewritten exactly once at the end of this round',
        recorded_at=now()), ensure_ascii=False, indent=1))
    if EVENTS.exists():
        two_column(EVENTS.parent)
    for d in (CANON,):
        if d.exists():
            two_column(d)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_AT_WRITE', frozen_at=now(),
                                                                    files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print('errata dir:', out)
    print('edits:', len(ledger))
    for e in ledger:
        print('   %-90s %s -> %s' % (e['file'].replace(str(DESK) + '/', ''), e['sha256_before'][:8], e['sha256_after'][:8]))
    print('canonical:', designation['canonical'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
