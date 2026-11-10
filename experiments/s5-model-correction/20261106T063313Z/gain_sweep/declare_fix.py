"""Declare a source change made after SOURCE_FREEZE.json, with its reason and both hashes.

The freeze was taken before the first simulation. Two later changes exist, both in common.py
and both made before any physical release of this task ran (no physics number existed):

  1. per-leg vector order.  `leg_cells_and_rh` first grouped legs in the MAPPING file order
     (LF, RF, LM, RM, LH, RH).  The sealed runtime's canonical order is rt.b.LEGS
     (LF, LM, LH, RF, RM, RH).  The gain-0.3 bitwise reproduction check failed on exactly one
     key, `legs_exact`; every other key (packed spikes, 1200x12000 filtered rates, 66 pool
     rates, all 17 network/history/integration/assist states) was already array_equal.
  2. `anatomy()` materialised the annotation columns once instead of once per tier cell
     (the first version re-converted the whole 188508-row feather column per cell; that made
     the static checks take 8m40s instead of 3s).  Same values, no behaviour change.

The whole offline part was then re-run from scratch for all four gains.
"""
import json
import time
from pathlib import Path

import common as C

js, sha, dump = C.js, C.sha, C.dump

FREEZE = C.O / 'SOURCE_FREEZE.json'
LEDGER = C.O / 'CHANGE_LEDGER.json'

REASONS = {
    'common.py': dict(
        changes=['leg_cells_and_rh: leg order is now the sealed runtime order '
                 'rt.b.LEGS=(LF,LM,LH,RF,RM,RH) instead of the MAPPING file order '
                 '(LF,RF,LM,RM,LH,RH)',
                 'anatomy: annotation columns are materialised once instead of once per tier '
                 'cell (same values, large speed-up)'],
        evidence='the first offline run failed only the leg-vector bitwise check '
                 '(FAILURE_offline_1793956477129314650.json keeps the record); after the fix the '
                 'gain-0.3 warm replay reproduces every sealed array bitwise',
        physics_number_existed_when_made=False,
        rerun='the offline part (4 warms + diagnostics + information) was re-run in full; no '
              'result from the pre-fix run is used'),
    'static.py': dict(
        changes=['the sensory-channel lookup path was corrected to the sealed snapshot on the '
                 'desktop (s3-banc888-runtime-admission) and the missing `from pathlib import '
                 'Path` import was added'],
        evidence='the first two static runs stopped on a NameError / missing path before any '
                 'check ran; the third run passed all 15 checks in 3.2 s',
        physics_number_existed_when_made=False,
        rerun='static.py was re-run from scratch; STATIC_CHECKS.json comes from the passing run'),
}


def main():
    freeze = js(FREEZE)['files']
    ledger = js(LEDGER)
    fixes = {f['path']: f for f in ledger.get('tool_fixes', [])}
    changed, missing = {}, []
    for p, h in freeze.items():
        now = sha(p)
        if now == h:
            continue
        name = Path(p).name
        if name not in REASONS:
            missing.append(p)
            continue
        r = REASONS[name]
        fixes[p] = dict(path=p, old_sha256=h, new_sha256=now,
                        reason='; '.join(r['changes']) + ' | evidence: ' + r['evidence'] +
                               ' | no physics number existed when the change was made | rerun: ' +
                               r['rerun'],
                        declared_at_unix=time.time(),
                        frozen_mtime=str(Path(p).stat().st_mtime),
                        physics_number_existed_when_made=r['physics_number_existed_when_made'])
        changed[p] = fixes[p]
    assert not missing, 'changed frozen source without a declared reason: %r' % (missing,)
    ledger['tool_fixes'] = list(fixes.values())
    dump(LEDGER, ledger)
    dump(C.O / 'FIX_DECLARATION.json', dict(
        freeze=str(FREEZE), freeze_sha256=sha(FREEZE), declared=list(changed.values()),
        note='declared BEFORE the heavy release job started; the sealed inputs themselves are '
             'byte-identical (INPUT_HASHES_BEFORE == INPUT_HASHES_AFTER)',
        importance='核心必需'))
    print('DECLARED', list(changed), flush=True)


if __name__ == '__main__':
    main()
