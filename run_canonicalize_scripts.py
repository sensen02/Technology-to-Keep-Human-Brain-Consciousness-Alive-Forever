"""CANONICALISE THE DELIVERY SCRIPT (one-time version freeze).

Owner-directed. It does NOT delete anything and does NOT edit any frozen delivery directory.
  * the CURRENT content of run_closed_loop_first.py (an early draft, 27199 bytes) is preserved into a new
    timestamped directory, because that name is about to become the canonical delivery script;
  * run_closed_loop_first.py is then set to the content of the last multi-arm version, byte-for-byte;
  * every other run_closed_loop_first*.py file gets a one-line SUPERSEDED header at the top, naming this one
    as the canonical script and listing its own sha256 BEFORE the edit, so no historical hash is lost.
"""
import hashlib, json, os, pathlib, time

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CANON = ROOT / 'run_closed_loop_first.py'
SRC = ROOT / 'run_closed_loop_first_v10.py'      # the last multi-arm revision (gain sweep reference)
HEAD = ("# SUPERSEDED HISTORICAL DRAFT - not the delivery script.\n"
        "# Canonical delivery script: run_closed_loop_first.py (see the ledger entry for this round).\n"
        "# sha256 of THIS file before the header was prepended: {sha}\n")


def main():
    out = DESK / ('S2-script-canonicalisation-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    old = CANON.read_text(encoding='utf-8')
    old_sha = sha(CANON)
    (out / 'run_closed_loop_first.py.pre_canonical_draft').write_text(old, encoding='utf-8')
    src_sha = sha(SRC)
    CANON.write_text(SRC.read_text(encoding='utf-8'), encoding='utf-8')
    canon_sha = sha(CANON)
    assert canon_sha == src_sha, 'the canonical script is not byte-identical to its source revision'
    records = [dict(file=str(CANON), action='BECAME THE CANONICAL DELIVERY SCRIPT',
                    content_source=str(SRC), sha256_before=old_sha, sha256_after=canon_sha,
                    pre_canonical_content_kept_at=str(out / 'run_closed_loop_first.py.pre_canonical_draft')),
               dict(file=str(SRC), action='LEFT UNTOUCHED, marked historical via its own header below',
                    sha256_before=src_sha)]
    marked = []
    for p in sorted(ROOT.glob('run_closed_loop_first*.py')):
        if p.resolve() == CANON.resolve():
            continue
        t = p.read_text(encoding='utf-8')
        if t.startswith('# SUPERSEDED HISTORICAL DRAFT'):
            continue
        before = sha(p)
        p.write_text(HEAD.format(sha=before) + t, encoding='utf-8')
        marked.append(dict(file=str(p), sha256_before=before, sha256_after=sha(p),
                           action='one-line SUPERSEDED header prepended; content otherwise unchanged'))
    doc = dict(status='SCRIPT_VERSION_FREEZE', manual_research_not_product=True,
        canonical_script=str(CANON), canonical_script_sha256=sha(CANON),
        rule='exactly ONE delivery script; the others are historical drafts, marked once at the top of the file '
             'with their pre-edit sha256 so nothing is lost. Nothing was deleted.',
        records=records, drafts_marked=marked, n_drafts_marked=len(marked),
        note='a sealed delivery batch keeps the filename it was produced with; the canonical name applies from this '
             'round forward, and the frozen batches are NOT rewritten.',
        written_at=time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)'))
    (out / 'SCRIPT_FREEZE.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
    os.chmod(out / 'SCRIPT_FREEZE.json', 0o444)
    for f in out.iterdir():
        if f.is_file(): os.chmod(f, 0o444)
    os.chmod(out, 0o555)
    print('FREEZE DIR:', out)
    print('canonical :', CANON, sha(CANON))
    print('pre-draft :', old_sha, '-> kept at', out / 'run_closed_loop_first.py.pre_canonical_draft')
    print('drafts marked:', len(marked))
    for m in marked: print('   ', pathlib.Path(m['file']).name, m['sha256_before'][:16], '->', m['sha256_after'][:16])
    return out


if __name__ == '__main__':
    main()
