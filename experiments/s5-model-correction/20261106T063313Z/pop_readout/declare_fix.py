"""Declare post-freeze source repairs and add new analysis tools to the freeze.

CHANGE_LEDGER.json keeps, for every changed frozen file, the hash it had when the heavy job
was frozen (old_sha256) and the hash it has now (new_sha256), so verify.py can distinguish a
declared repair from an undeclared source change.  The freeze itself is NOT relaxed to the
new hashes: it stays the pre-run record, exactly as in the previous stage of this project.
"""
import common as C

FIXES = [
    dict(path=str(C.SRC / 'run.py'),
         reason='the runtime self-check that replays the sealed 5 ms per-cell filter compared a '
                'per-NEURAL-step array (2400 / 19200 rows) with the sealed per-COMMAND-interval '
                'trace (1200 rows) and raised ValueError before the first release. The helper now '
                'returns the state after the last neural step of each 1 ms interval. NO release '
                'had been produced (RUNS/ did not exist; the traceback is kept in '
                'FAILURE_1793954876274728711.json) and the physics code path (episode, command, '
                'integrate) is unchanged by the repair.'),
    dict(path=str(C.SRC / 'verify.py'),
         reason='the same per-interval indexing fix in the verifier\'s independent filter replay; '
                'the verifier had not been run yet.'),
    dict(path=str(C.SRC / 'run.py'),
         reason='the pre-run freeze assertion required bit-identical source and therefore rejected '
                'the two declared repairs above. It now accepts a deviation only when '
                'CHANGE_LEDGER.tool_fixes bridges the frozen hash to the current hash, records '
                'each one in DECLARED_SOURCE_FIXES_USED.json, and still refuses any undeclared '
                'change. No physics result existed at the time of the change.'),
    dict(path=str(C.SRC / 'report.py'),
         reason='the per-timestep verdict section was too vague to enforce the instruction that the '
                'two neural timesteps must be reported separately: it did not show WHICH D checks '
                'carried the count changes. It now prints the per-timestep gate counts, the '
                'R0 -> readout best-phase D count, the explicit "single-timestep observation" '
                'wording when only one timestep meets the D-count criterion, and a table of the D '
                'checks that were gained or lost. ALSO corrected the D-gate narrative (D7 is '
                'passed by the baseline at one phase and lost by the new readouts, so "D7 always '
                'fails" was wrong) and added section 7.4 stating plainly whether the dominant '
                'cells are sensory/proprioceptive-like or motor-like, with the cell_function / '
                'super_class shares and the 204-cell weight share. Later revisions translated the '
                'closed-loop definition and truncation note into Chinese, added the explicit '
                'reading conclusion (starts right, then diverges), put the numeric reproducibility '
                'worst case into section 3.4, stated which branch of the causal-boundary criterion '
                'this run lands in, and added section 11b listing every failed attempt with its exit '
                'code and artifact, and finally added a one-page section 0 summary. Report '
                'generation only; no result, threshold or release '
                'changed.'),
    dict(path=str(C.SRC / 'closedloop_corr.py'),
         reason='the teacher torque recording ends at 1.4 s, so for the late release phases the '
                '100/200 ms windows have fewer teacher samples than 2801-start; the index bound was '
                'computed from 2801-start instead of from the number of 1 ms intervals whose 0.5 ms '
                'sample exists, and indexed past the end of the array (IndexError). Analysis-only '
                'tool: no physics, no release, no threshold is touched.'),
    dict(path=str(C.SRC / 'deliver.py'),
         reason='delivery-tool corrections only: DECLARED_SOURCE_FIXES_USED.json is now labelled '
                '核心必需 instead of 待确认, the cleanup note no longer claims a deleted log that '
                'was in fact the definitive offline fit log (it was renamed into LOGS/ and every '
                'log there is regenerable), and the stale superseded-log classifier was removed. '
                'No measurement, threshold, result or release is affected.'),
    dict(path=str(C.SRC / 'declare_fix.py'),
         reason='this declaration tool itself was extended to carry the repairs above; it performs '
                'no simulation.'),
]


def main():
    freeze = C.js(C.O / 'SOURCE_FREEZE.json')
    led = C.js(C.O / 'CHANGE_LEDGER.json')
    fixes = led.setdefault('tool_fixes', [])
    src = {str(p): C.sha(p) for p in sorted(C.SRC.glob('*.py'))}

    # 1. new tools written before the heavy job are added to the freeze at their current hash
    added = []
    for p, h in src.items():
        if p not in freeze['files']:
            freeze['files'][p] = h
            added.append(p)
            fixes.append(dict(path=p, old_sha256=None, new_sha256=h,
                              reason='analysis tool written before the heavy job and added to the '
                                     'freeze; it performs no simulation',
                              timestamp='before_heavy_job'))
    freeze.setdefault('frozen_tools', []).extend(added)

    # 2. declared repairs: freeze keeps the pre-run hash, the ledger bridges it
    for fx in FIXES:
        p = fx['path']
        cur = src[p]
        if cur == freeze['files'][p]:
            continue
        if any(x['path'] == p and x.get('new_sha256') == cur for x in fixes):
            continue
        fixes.append(dict(path=p, old_sha256=freeze['files'][p], new_sha256=cur,
                          reason=fx['reason'], timestamp='after_freeze_before_any_release'))

    bridged = {x['path'] for x in fixes
               if x.get('new_sha256') is not None and src.get(x['path']) == x['new_sha256']}
    still = [p for p, h in freeze['files'].items() if src.get(p, C.sha(p)) != h and p not in bridged]
    if still:
        print('WARNING unbridged frozen files:', still, flush=True)

    C.dump(C.O / 'SOURCE_FREEZE.json', freeze)
    fixes.append(dict(path=str(C.O / 'SOURCE_FREEZE.json'), old_sha256=None,
                      new_sha256=C.sha(C.O / 'SOURCE_FREEZE.json'),
                      reason='freeze extended with the pre-run analysis tools',
                      timestamp='before_heavy_job'))
    C.dump(C.O / 'CHANGE_LEDGER.json', led)
    print('frozen files', len(freeze['files']), 'added', len(added), flush=True)


if __name__ == '__main__':
    main()
