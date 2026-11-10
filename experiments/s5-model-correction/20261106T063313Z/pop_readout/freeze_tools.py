"""Freeze every analysis tool before the single heavy job.

Adds any *.py in the source directory that is not already in SOURCE_FREEZE.json, and
records the delta in CHANGE_LEDGER.json.  Refuses to change the hash of a file that was
already frozen (that would be an undeclared source change).
"""
import common as C


def main():
    freeze = C.js(C.O / 'SOURCE_FREEZE.json')
    led = C.js(C.O / 'CHANGE_LEDGER.json')
    cur = {str(p): C.sha(p) for p in sorted(C.SRC.glob('*.py'))}
    added, changed = [], []
    for p, h in cur.items():
        if p in freeze['files']:
            if freeze['files'][p] != h:
                changed.append(p)
        else:
            added.append(p)
            freeze['files'][p] = h
    if changed:
        raise SystemExit('a previously frozen source file changed: %r' % changed)
    freeze.setdefault('frozen_tools', []).extend(added)
    C.dump(C.O / 'SOURCE_FREEZE.json', freeze)
    fixes = led.setdefault('tool_fixes', [])
    for p in added:
        fixes.append(dict(path=p, old_sha256=None, new_sha256=cur[p],
                          reason='analysis tool written before the heavy job and added to the '
                                 'freeze; it performs no simulation',
                          timestamp='before_heavy_job'))
    fixes.append(dict(path=str(C.O / 'SOURCE_FREEZE.json'), old_sha256=None,
                      new_sha256=C.sha(C.O / 'SOURCE_FREEZE.json'),
                      reason='freeze extended to the pre-run analysis tools',
                      timestamp='before_heavy_job'))
    C.dump(C.O / 'CHANGE_LEDGER.json', led)
    assert {p: C.sha(p) for p in freeze['files']} == freeze['files']
    print('FROZEN', len(freeze['files']), 'source files; added', len(added), flush=True)


if __name__ == '__main__':
    main()
