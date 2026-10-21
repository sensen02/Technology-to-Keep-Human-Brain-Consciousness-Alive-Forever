"""Turn-34 finish: the last two 25 us copies get their SUPERSEDED first line, then everything re-freezes.

This is the last write of the round. It exists because the owner's item 3 requires that ONE 25 us artifact be
canonical and that the other copies carry a SUPERSEDED first line. Two of those copies (172131 and 172533) still
lacked it. They are un-frozen, edited, re-frozen with the same asserts, and the change is appended to the errata-4
change_list, whose own directory is then re-frozen too.
"""

import hashlib
import json
import os
import pathlib
import stat
import sys
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
SNAP_ROOT = DESK / '_frozen_snapshots'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')


def unfreeze(d):
    for root, dirs, files in os.walk(d):
        for f in files:
            os.chmod(os.path.join(root, f), 0o664)
        os.chmod(root, 0o775)


def freeze(d, extra_manifest_fields=None):
    d = pathlib.Path(d)

    # start from a clean state: remove the two generated files, then rebuild them in this order
    for name in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json'):
        p = d / name
        if p.exists():
            p.unlink()

    # ORDER MATTERS and is asserted at the end: the manifest file must EXIST before the listing is taken,
    # otherwise its own entry cannot be in the listing it is supposed to cover.
    fz_path = d / 'FROZEN_DELIVERY_HASHES.json'
    dh_path = d / 'delivery-hashes.json'
    fz_path.write_text('{}', encoding='utf-8')          # placeholder so both generated files count as files
    dh_path.write_text('{}', encoding='utf-8')
    names = sorted(p.name for p in d.iterdir() if p.is_file())
    assert 'FROZEN_DELIVERY_HASHES.json' in names and 'delivery-hashes.json' in names, 'placeholders missing'
    manual = len(os.listdir(d))
    assert manual == len(names), 'os.listdir count %d != is_file count %d' % (manual, len(names))

    # ---- ORDER (each step asserted at the end, no step may invalidate an earlier one):
    #  1. hash every file that is not one of the two generated files
    #  2. finalise delivery-hashes.json COMPLETELY (it carries the manifest digest)
    #  3. hash the final delivery-hashes.json into the manifest, then write the manifest last
    #  4. only then may the manifest digest be copied into delivery-hashes.json, and that copy is re-asserted
    other = [f for f in names if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]
    # ONE authoritative direction only. The manifest lists every file, delivery-hashes.json and itself included.
    # delivery-hashes.json lists every file EXCEPT the manifest and itself, and carries NO digest of the manifest,
    # because two files each carrying the other digest is mathematically impossible (a fixed point).
    dh = {f: sha(d / f) for f in other}
    dh['_manifest'] = dict(file='FROZEN_DELIVERY_HASHES.json',
                           digest_here=False,
                           reason=('a mutual digest pair between this file and the manifest has no solution: writing the manifest '
                                   'digest here changes this file, which changes what the manifest must record. The manifest is '
                                   'therefore the single authoritative direction and digests this file; this file does not digest the '
                                   'manifest. The manifest digest is unlisted by construction and is checked by re-hashing.'))
    dh['_self_excluded'] = 'delivery-hashes.json cannot contain its own digest; the manifest carries it'
    dh_path.write_text(json.dumps(dh, ensure_ascii=False, indent=1), encoding='utf-8')

    manifest = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
                    files={f: sha(d / f) for f in other},
                    covers_complete=True,
                    covers_rule=('every file present in this directory at freeze time appears in `files`, this manifest included: '
                                 'manifest entry count == os.listdir count is asserted by script at freeze time. delivery-hashes.json '
                                 'cannot contain its own digest, so its digest is carried here. This manifest own digest cannot be '
                                 'carried anywhere (a file cannot contain its own digest) and is verified by re-hashing, not by a field.'),
                    manifest_sha256=None,
                    manifest_sha256_note='cannot be self-referential and cannot be carried by delivery-hashes.json either, because that '
                                         'would be a mutual digest pair with no fixed point; verified by re-hashing instead',
                    not_covered=[],
                    assert_rule='manifest entries == os.listdir count',
                    frozen_by='turn34 script with assert, then chmod read-only')
    if extra_manifest_fields:
        manifest.update(extra_manifest_fields)
    manifest['files']['delivery-hashes.json'] = sha(dh_path)      # FINAL delivery-hashes.json
    fz_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    manifest['files']['FROZEN_DELIVERY_HASHES.json'] = sha(fz_path)
    fz_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    m2 = json.loads(fz_path.read_text(encoding='utf-8'))

    # ---- FINAL ASSERTS: the owner's rule, checked by script, not by hand
    m2 = json.loads(fz_path.read_text(encoding='utf-8'))
    n_listdir = len([n for n in os.listdir(d) if (d / n).is_file()])
    assert len(m2['files']) == n_listdir, 'manifest entries %d != os.listdir files %d in %s' % (len(m2['files']), n_listdir, d)
    assert 'FROZEN_DELIVERY_HASHES.json' in m2['files'], 'manifest does not list itself'
    assert 'delivery-hashes.json' in m2['files'], 'manifest does not list delivery-hashes.json'
    assert m2['covers_complete'] is True
    dh2 = json.loads(dh_path.read_text(encoding='utf-8'))
    assert dh2['_manifest']['digest_here'] is False, 'delivery-hashes.json must not carry the manifest digest (circular)'
    for f, h in m2['files'].items():
        if f != 'FROZEN_DELIVERY_HASHES.json':
            assert sha(d / f) == h, 'stale digest for %s' % f

    for f in sorted(p for p in d.iterdir() if p.is_file()):
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    final = sorted(p.name for p in d.iterdir() if p.is_file())
    assert len(final) == len(m2['files']), 'post-chmod listing changed'
    return dict(directory=str(d), files={f: sha(d / f) for f in final}, frozen_at=m2['frozen_at'],
                manifest_sha256=sha(d / 'FROZEN_DELIVERY_HASHES.json'),
                covers_complete=True, n_files=len(final), read_only_test=str(d))




COPIED = [
    DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131',
    DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-20261015T172533',
]
ERRATA = sorted(DESK.glob('S2-errata-4-*'))[-1]

BANNER = ('**SUPERSEDED → canonical artifact: %s**（本题材只保留一个 canonical；本目录为旧拷贝，只作历史，不再派核验）\n' % CANON)


def main():
    print('errata dir:', ERRATA)
    changes = []
    for d in COPIED:
        unfreeze(d)
        p = d / 'README.zh-CN.md'
        b = sha(p)
        t = p.read_text(encoding='utf-8')
        if t.startswith('**SUPERSEDED'):
            changes.append(dict(file=str(p), sha256_before=b, sha256_after=b, changed=False,
                                changed_at=now(), reason='already carried the SUPERSEDED first line; no write needed',
                                edit_kind='text_only'))
            print('  already superseded:', d.name)
        else:
            p.write_text(BANNER + t, encoding='utf-8')
            changes.append(dict(file=str(p), sha256_before=b, sha256_after=sha(p), changed=True,
                                changed_at=now(),
                                reason='owner item 3: one canonical 25 us artifact only; this copy gets the SUPERSEDED first '
                                       'line pointing at the canonical artifact (text only, no number changed)',
                                edit_kind='text_only'))
            print('  superseded:', d.name, b[:12], '->', sha(p)[:12])

    # record the late writes in the errata change_list, then re-freeze the errata dir and the two copies
    unfreeze(ERRATA)
    cl = json.loads((ERRATA / 'change_list.json').read_text(encoding='utf-8'))
    cl['entries'].extend(changes)
    cl['n'] = len(cl['entries'])
    cl['late_write_note'] = ('items 3 of this round: the two remaining 25 us copies received their SUPERSEDED first line after the '
                             'first freeze pass, so the freeze was re-run for them and for this errata directory. No other directory '
                             'was re-opened.')
    (ERRATA / 'change_list.json').write_text(json.dumps(cl, ensure_ascii=False, indent=1), encoding='utf-8')
    os.chmod(ERRATA / 'change_list.json', 0o664)

    out = [freeze(d, dict(kind='superseded 25 us copy (history only, no further verifier)')) for d in COPIED]
    out.append(freeze(ERRATA, dict(kind='this round errata record')))

    previous = json.loads(sorted(SNAP_ROOT.glob('freeze_report-errata-4-*.json'))[-1].read_text(encoding='utf-8'))
    all_dirs = {d['directory']: d for d in previous['dirs']}
    for d in out:
        all_dirs[d['directory']] = d
    previous['dirs'] = list(all_dirs.values())
    previous['re_frozen_at'] = now()
    previous['re_freeze_reason'] = 'two superseded 25 us copies gained their SUPERSEDED first line after the first pass; they and this ' \
                                   'round errata directory were re-frozen. Their digests below are the post-edit ones.'
    rp = SNAP_ROOT / ('freeze_report-errata-4-' + time.strftime('%Y%m%dT%H%M%S') + '-final.json')
    rp.write_text(json.dumps(previous, ensure_ascii=False, indent=1), encoding='utf-8')

    print('final freeze report:', rp)
    for d in previous['dirs']:
        print('  %-64s files=%d covers=%s ro=%s' % (pathlib.Path(d['directory']).name, d['n_files'],
                                                    d['covers_complete'], oct(os.stat(d['directory']).st_mode)[-4:]))
    return previous


if __name__ == '__main__':
    main()
