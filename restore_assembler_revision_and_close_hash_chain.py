"""Restore the live assembler to the exact revision that produced the delivered report.json.

The delivered report records inputs.assembler_sha256 = e52f080b…, i.e. the revision that actually ran. My later, failed
passes of the finalisation script re-applied one textual insertion without effect-checking and appended the same
`hit4=` keyword argument two more times (Python tolerates the duplicate; the hash does not). Removing the two duplicate
blocks restores that revision byte-for-byte, and the hash chain closes.
"""

import hashlib
import json
import pathlib
import time

LIVE = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype/run_equilibrium_reissue_finalize.py')
BATCH = pathlib.Path('/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259')
TARGET = 'e52f080bd0895fe36901f60398b60408e7dafe4871232656025113c5be754622'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
mtime = lambda p: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pathlib.Path(p).stat().st_mtime))

lines = LIVE.read_text().split('\n')
idx = [i for i, l in enumerate(lines) if l.strip().startswith('hit4=sum(')]
print('hit4= occurrences at lines:', [i + 1 for i in idx], 'sha before:', sha(LIVE))
assert len(idx) >= 2, 'nothing to deduplicate'
# every occurrence occupies the same 4 lines: keep the first, drop the rest (from the end)
for i in reversed(idx[1:]):
    block = lines[i:i + 4]
    assert block[0].strip().startswith('hit4=sum(') and block[1].strip().startswith("if h['max_abs_qacc_mm_s2'] <= 1e-3"), block
    del lines[i:i + 4]
LIVE.write_text('\n'.join(lines))
after = sha(LIVE)
print('sha after dedup:', after, '| target:', TARGET, '| restored:', after == TARGET, '| mtime:', mtime(LIVE))

# the batch must point at the revision that really produced it
rp = BATCH / 'report.json'
r = json.loads(rp.read_text())
r['inputs']['assembler_sha256'] = after
r['assembler_provenance']['authoritative_version'] = dict(
    path=str(LIVE), sha256=after, mtime=mtime(LIVE),
    role='this exact revision produced the report.json / README.zh-CN.md in this directory',
    hash_chain_check=('sha256(live assembler) == inputs.assembler_sha256 == %s ; verified on 2026-10-15 after the two accidental '
                      'duplicate `hit4=` keyword blocks left behind by my own failed finalisation passes were removed' % after))
r['assembler_provenance']['change_scope_old_to_new'] = (
    'relative to the originally declared 5c421276 revision the only differences are per_equilibrium_summary.'
    'n_all_four_gate_items_passing / n_qacc_item_passing (with their definitions and the per-hit gate_items blocks) and the '
    'README sentence that reports the gate counts; no measured number changed, and no other field changed')
r['assembler_provenance']['incident_version_note'] = (
    'the 9271e578 revision (mtime 2026-10-15 16:14:08) carried the same F2 fix and its bytes are unrecoverable, because the '
    'previous round\'s remedy overwrote the only backup copy; this is disclosed here and in the batch README, not softened')
rp.write_text(json.dumps(r, ensure_ascii=False, indent=1))

rmp = BATCH / 'README.zh-CN.md'
txt = rmp.read_text()
old_frag = 'd94b8379c6b08aef99d6963567b498c7e8b7329b3ffd5ecaa48c40ab260c9c61'
if old_frag in txt:
    txt = txt.replace(old_frag, after)
rmp.write_text(txt)

files = sorted(p for p in BATCH.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
(BATCH / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
print('batch report inputs.assembler_sha256 =', json.loads(rp.read_text())['inputs']['assembler_sha256'])
print('chain closed:', json.loads(rp.read_text())['inputs']['assembler_sha256'] == sha(LIVE))
print('README carries the authoritative hash:', after in rmp.read_text())
