"""Recompute hash_reanchor.json properly (a generic walk instead of three hand-written lookups that silently failed)."""

import hashlib
import json
import pathlib
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
NEW = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def walk(node, path=''):
    if isinstance(node, dict):
        if 'sha256' in node and isinstance(node['sha256'], str) and len(node['sha256']) == 64:
            f = node.get('path') or node.get('report_json') or node.get('time_series')
            yield path, node['sha256'], f
        for k, v in node.items():
            if k.endswith('_sha256') and isinstance(v, str) and len(v) == 64:
                ref = node.get(k[:-len('_sha256')])
                if not isinstance(ref, str) and isinstance(node.get('run_dir'), str):
                    name = {'report_json_sha256': 'report.json', 'stage2_best_state_sha256': 'stage2_best_state.npz'}.get(k)
                    if name:
                        ref = str(pathlib.Path(node['run_dir']) / name)
                if isinstance(ref, str):
                    yield path + '.' + k, v, ref
        for k, v in node.items():
            yield from walk(v, path + '.' + k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, path + '[%d]' % i)


def collect(report_path, extra=()):
    rep = json.loads(pathlib.Path(report_path).read_text())
    stale = []
    for where, recorded, ref in walk(rep):
        if not ref:
            continue
        f = pathlib.Path(ref)
        if not f.exists() or not f.is_file():
            continue
        cur = sha(f)
        if cur != recorded:
            stale.append(dict(where=where.lstrip('.'), file=str(f), recorded=recorded, current=cur))
    for where, ref in extra:
        node = rep
        for part in where.split('.'):
            node = node.get(part) if isinstance(node, dict) else None
        f = pathlib.Path(ref)
        if isinstance(node, str) and f.exists() and node != sha(f):
            stale.append(dict(where=where, file=str(f), recorded=node, current=sha(f)))
    return stale


def main():
    v2_stale = collect(V2 / 'report.json',
                       extra=[('inputs.search_script.sha256', '/run/media/sensen/Data2/cell_wound_prototype/run_mujoco_equilibrium_search.py'),
                              ('inputs.diagnostic_script.sha256', '/run/media/sensen/Data2/cell_wound_prototype/diagnose_aliased_residual.py'),
                              ('inputs.ladder_script.sha256', '/run/media/sensen/Data2/cell_wound_prototype/run_equilibrium_radius_ladder.py'),
                              ('inputs.v1_batch.report_json_sha256', str(DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/report.json'))])
    ladder_stale = collect(LADDER / 'report.json',
                           extra=[('inputs.search_script_sha256', '/run/media/sensen/Data2/cell_wound_prototype/run_mujoco_equilibrium_search.py'),
                                  ('inputs.runner_sha256', '/run/media/sensen/Data2/cell_wound_prototype/run_equilibrium_radius_ladder.py')])
    print('stale v2 records:', len(v2_stale))
    for s in v2_stale:
        print('   ', s['where'], s['recorded'][:8], '->', s['current'][:8])
    seen = set()
    ladder_stale = [s for s in ladder_stale if not (s['where'] in seen or seen.add(s['where']))]
    seen = set()
    v2_stale = [s for s in v2_stale if not (s['where'] in seen or seen.add(s['where']))]
    print('stale ladder records:', len(ladder_stale))
    for s in ladder_stale:
        print('   ', s['where'], s['recorded'][:8], '->', s['current'][:8])

    p = V2 / 'hash_reanchor.json'
    reanchor = json.loads(p.read_text())
    reanchor['stale_records_v2'] = v2_stale
    reanchor['stale_records_ladder'] = ladder_stale
    reanchor['counts'] = dict(v2=len(v2_stale), ladder=len(ladder_stale))
    reanchor['recorded_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    p.write_text(json.dumps(reanchor, ensure_ascii=False, indent=1))
    (NEW / 'hash_reanchor.json').write_text(json.dumps(reanchor, ensure_ascii=False, indent=1))
    for d in (V2, NEW):
        files = sorted(x for x in d.iterdir() if x.is_file() and x.name != 'delivery-hashes.json')
        (d / 'delivery-hashes.json').write_text(json.dumps({x.name: sha(x) for x in files}, ensure_ascii=False, indent=1))
    print('rewritten; manifests refreshed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
