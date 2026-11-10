"""Synthetic false-pass checks and read-only saved-native diagnostic. No simulation.

Run with this directory's env/bin/python -B. Writes ONLY the selected checks JSON;
temporary synthetic chunks are deleted automatically. Existing history is read-only.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import native_acceptance as N

HERE = Path(__file__).resolve().parent
REAL = HERE.parent / '20261110T051002Z/bc_gru238_native'
G = 9801.
W = 24.446354755878005
M = W / G
DT = N.NATIVE_DT


def fixture(n=1600, *, az=None, external=None, nonfoot_fraction=0., speed=.01):
    t = np.arange(n + 1) * DT
    az = np.zeros(n) if az is None else np.broadcast_to(az, (n,)).copy()
    ext = np.zeros(n) if external is None else np.broadcast_to(external, (n,)).copy()
    vz = np.r_[0., np.cumsum(az * DT)]
    z = 1.3 + np.r_[0., np.cumsum((vz[:-1] + vz[1:]) * .5 * DT)]
    com = np.column_stack((.1 * t, np.zeros(n + 1), z))
    velocity = np.column_stack((np.full(n + 1, .1), np.zeros(n + 1), vz))
    total = W + M * az - ext
    loads = np.repeat((total * (1 - nonfoot_fraction) / 6)[:, None], 6, axis=1)
    nf = total * nonfoot_fraction
    return dict(time_before_s=t[:-1], time_after_s=t[1:], com_before=com[:-1], com_after=com[1:],
                com_velocity_before=velocity[:-1], com_velocity_after=velocity[1:],
                loads_native=loads, nonfoot_force_native=nf,
                contact_speed_native_per_s=np.full((n, 6), speed),
                ground_force_native=np.column_stack((np.zeros(n), np.zeros(n), total)),
                external_total_force_native=np.column_stack((np.zeros(n), np.zeros(n), ext)))


def split(data, size=137):
    n = len(data['time_before_s'])
    return ({k: v[i:i + size] for k, v in data.items()} for i in range(0, n, size))


def evaluate(data, **kw):
    defaults = dict(autonomous_window=(0., len(data['time_before_s']) * DT), mass_native=M)
    defaults.update(kw)
    return N.evaluate_chunks(split(data), **defaults)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(HERE / 'NATIVE_ACCEPTANCE_CHECKS.json'))
    p.add_argument('--skip-real', action='store_true')
    p.add_argument('--real-dir', default=str(REAL))
    args = p.parse_args(argv)
    started = time.time()
    tests = []
    def check(name, condition, details=None):
        tests.append(dict(name=name, passed=bool(condition), details=details))
        if not condition:
            raise AssertionError(name + ': ' + str(details))
    base = fixture()
    r = evaluate(base)
    check('healthy physical signal is processed but not full-goal PASS', r['status'] == 'OK' and r['final_acceptance']['status'] == 'UNKNOWN')
    check('missing auxiliary means D3 and D5 UNKNOWN', all(r['criteria'][k]['status'] == 'UNKNOWN' for k in ('D3', 'D5')))
    check('D7 threshold remains provisional', r['criteria']['D7']['status'] == 'UNKNOWN' and r['criteria']['D7']['provisional_indicator_pass'])
    check('D1/D2/D4/D4b/D6/D8 healthy indicators', all(r['criteria'][k]['status'] == 'PASS' for k in ('D1','D2','D4','D4b','D6','D8')))
    check('native endpoint forward and weighted slip exact', abs(r['windows']['autonomous']['forward_signed_native_per_s'] - .1) < 1e-12 and abs(r['windows']['autonomous']['slip_ratio_max_forward'] - .1) < 1e-12)
    noadh = r['windows']['autonomous']
    check('no adhesion original and external-accounted residual agree', noadh['original_D6_absolute_p95_W'] == noadh['external_accounted_absolute_p95_W'])
    nf = fixture(nonfoot_fraction=1e-6)
    rn = evaluate(nf)
    check('any positive nonfoot fails D1 despite legacy engineering tolerance', rn['criteria']['D1']['status'] == 'FAIL' and rn['engineering_only']['nonfoot']['status'] == 'PASS' and rn['final_acceptance']['status'] == 'FAIL')
    rf = evaluate(fixture(az=-G))
    check('freefall cannot pass via zero nonfoot', rf['criteria']['D1']['status'] == 'PASS' and rf['criteria']['D2']['status'] == 'FAIL' and rf['windows']['autonomous']['freefall_duration_s'] > 0)
    rb = evaluate(fixture(nonfoot_fraction=1.))
    check('belly support never counted as foot support', all(rb['criteria'][k]['status'] == 'FAIL' for k in ('D1','D4','D4b','D7')) and rb['criteria']['D2']['status'] == 'PASS')
    low = fixture()
    low['com_before'][:, 2] -= .5
    low['com_after'][:, 2] -= .5
    check('D8 true height floor', evaluate(low)['criteria']['D8']['status'] == 'FAIL')
    # Time-weighted endpoint Newton signal; old recorder acceleration field is deliberately ignored.
    signal = .12 * G * np.sin(np.arange(1600) * .07)
    ext = .30 * W * (1. + .1 * np.cos(np.arange(1600) * .11))
    known = fixture(az=signal, external=ext)
    known['com_acceleration_fd_native'] = np.full((1600, 3), 1e20)
    rk = evaluate(known)
    k = rk['windows']['autonomous']
    check('filtered Newton known signal includes applied forces only in diagnostic', k['external_accounted_absolute_p95_W'] < 1e-10 and k['original_D6_signed_p95_W'] > .15 and rk['criteria']['D6']['status'] == 'FAIL', k)
    # Compare same filter to independently computed signed external force residual.
    alpha = -np.expm1(-DT / .001)
    y = float(ext[0]); history = []
    for x in ext:
        y += alpha * (float(x) - y)
        history.append(y / W)
    check('one identical causal filter applied to both sides of Newton', abs(k['original_D6_signed_p95_W'] - np.quantile(history, .95)) < 1e-10)
    downward = evaluate(fixture(external=-.5 * W))
    check('literal signed D6 and absolute diagnostic are not conflated', downward['criteria']['D6']['status'] == 'PASS' and downward['windows']['autonomous']['original_D6_absolute_p95_W'] > .49 and downward['external_force_review']['status'] == 'REQUIRES_REVIEW' and downward['final_acceptance']['status'] != 'PASS')
    # Chunking must not reset filter or alter quantiles/endpoints.
    whole = N.evaluate_chunks([known], autonomous_window=(0., .02), mass_native=M)
    check('chunk boundaries preserve causal filter and aggregates', whole['windows'] == rk['windows'])
    with tempfile.TemporaryDirectory(prefix='native-acceptance-check-') as temp:
        temp = Path(temp)
        proof = temp / 'auxiliary_evidence.json'
        proof.write_text('{"auxiliary_control_valid":true}\n')
        meta = dict(run_id='synthetic-same-run', evidence=[dict(path=str(proof), sha256=N.sha256(proof), claims=['auxiliary_control_valid'])])
        aux_kwargs = dict(autonomous_window=(.01, .02), auxiliary_window=(0., .01), metadata=meta)
        ra = evaluate(base, **aux_kwargs)
        check('same-run evidenced healthy auxiliary unlocks D3 and D5', ra['auxiliary_reference_valid'] and all(ra['criteria'][key]['status'] == 'PASS' for key in ('D3','D5')))
        dropload = copy.deepcopy(base)
        dropload['loads_native'][:800] *= 2.
        dropload['ground_force_native'][:800,2] *= 2.
        check('D3 compares against actual same-run auxiliary load', evaluate(dropload, **aux_kwargs)['criteria']['D3']['status'] == 'FAIL')
        dropheight = copy.deepcopy(base)
        dropheight['com_before'][:,2] = 1.4 - 15 * dropheight['time_before_s']
        dropheight['com_after'][:,2] = 1.4 - 15 * dropheight['time_after_s']
        dropheight['com_velocity_before'][:,2] = -15.
        dropheight['com_velocity_after'][:,2] = -15.
        check('D5 uses auxiliary mean height not global minimum', evaluate(dropheight, **aux_kwargs)['criteria']['D5']['status'] == 'FAIL')
        invalid = copy.deepcopy(base)
        invalid['nonfoot_force_native'][799] = .01 * W
        invalid['loads_native'][799] *= .99
        boundary = evaluate(invalid, **aux_kwargs)
        check('auxiliary last sample does not leak raw D1 into autonomous window', boundary['criteria']['D1']['status'] == 'PASS' and boundary['criteria']['D3']['status'] == 'UNKNOWN')
        invalid = copy.deepcopy(base)
        invalid['nonfoot_force_native'][800] = .01 * W
        invalid['loads_native'][800] *= .99
        check('first autonomous sample belongs to autonomous D1', evaluate(invalid, **aux_kwargs)['criteria']['D1']['status'] == 'FAIL')
        check('auxiliary time alone without control evidence is insufficient', evaluate(base, autonomous_window=(.01,.02), auxiliary_window=(0.,.01))['criteria']['D3']['status'] == 'UNKNOWN')
        badmeta = copy.deepcopy(meta)
        badmeta['evidence'][0]['sha256'] = '0' * 64
        check('bad external evidence hash is ERROR', evaluate(base, **(aux_kwargs | {'metadata': badmeta}))['status'] == 'ERROR')
        check('overlapping auxiliary and autonomous ERROR', evaluate(base, autonomous_window=(.009,.02), auxiliary_window=(0.,.01))['status'] == 'ERROR')
        check('partial native boundary remains UNKNOWN', evaluate(base, autonomous_window=(DT/2,.02))['criteria']['D1']['status'] == 'UNKNOWN')
        check('outside recording window remains UNKNOWN', evaluate(base, autonomous_window=(0.,.03))['criteria']['D4']['status'] == 'UNKNOWN')
        for name, key in [('missing field', 'com_velocity_before'), ('missing applied force', 'external_total_force_native')]:
            bad = copy.deepcopy(base); del bad[key]
            check(name + ' prevents autoPASS', evaluate(bad)['status'] == 'ERROR')
        bad = copy.deepcopy(base); bad['loads_native'][10, 1] = np.nan
        check('NaN prevents partial PASS', evaluate(bad)['status'] == 'ERROR' and evaluate(bad)['criteria']['D1']['status'] == 'UNKNOWN')
        bad = copy.deepcopy(base); bad['loads_native'][10, 1] = -1.
        check('negative contact load ERROR', evaluate(bad)['status'] == 'ERROR')
        bad = copy.deepcopy(base); bad['ground_force_native'][10,2] += 1.
        check('contact sum cross-check catches wrong force accounting', evaluate(bad)['status'] == 'ERROR')
        for name, shift in [('gap', DT), ('duplicate overlap', -DT)]:
            bad = copy.deepcopy(base)
            bad['time_before_s'][800:] += shift; bad['time_after_s'][800:] += shift
            check(name + ' time ERROR', evaluate(bad)['status'] == 'ERROR')
        bad = copy.deepcopy(base); bad['com_velocity_before'][1,2] += 1.
        check('actual endpoint velocity continuity checked', evaluate(bad)['status'] == 'ERROR')
        bad = copy.deepcopy(base); bad['contact_speed_native_per_s'] = np.zeros((1600,5))
        check('malformed shape ERROR', evaluate(bad)['status'] == 'ERROR')
        check('bounded scalar retention cap enforced', evaluate(base, max_scalar_samples=100)['status'] == 'ERROR')
        no_load = fixture()
        no_load['loads_native'][:,0] = 0.
        no_load['loads_native'][:,1] *= 2.
        check('zero-load leg slip UNKNOWN instead of fake zero', evaluate(no_load)['engineering_only']['slip']['status'] == 'UNKNOWN')
        sideways = fixture()
        sideways['com_before'][:,1] = 10 * sideways['time_before_s']
        sideways['com_after'][:,1] = 10 * sideways['time_after_s']
        side = evaluate(sideways)['windows']['autonomous']
        check('side drift cannot lower signed-forward slip denominator', side['slip_ratio_max_forward'] > 50 * side['slip_ratio_max_legacy'])
        backwards = fixture()
        backwards['com_before'][:,0] *= -1; backwards['com_after'][:,0] *= -1
        back = evaluate(backwards)
        check('backwards cannot pass engineering forward/slip', back['engineering_only']['forward']['status'] == 'FAIL' and back['engineering_only']['slip']['status'] == 'UNKNOWN')
        # Hash-backed directory loader, count/trailing completion and CLI integration.
        chunks = []
        for i, block in enumerate(split(base, 400)):
            path = temp / f'native_{i:06d}.npz'
            np.savez_compressed(path, **block)
            chunks.append(dict(path=str(path), rows=400, sha256=N.sha256(path)))
        manifest = dict(chunks=chunks, native_samples=1600, pending_rows=0)
        rd = N.evaluate_directory(temp, autonomous_window=(0.,.02), mass_native=M, source_manifest=manifest)
        check('streamed directory loader and recorded hashes verified', rd['status'] == 'OK' and rd['integrity']['manifest_count_verified'] and len(rd['source']['chunks']) == 4)
        wrong = copy.deepcopy(manifest); wrong['native_samples'] += 1
        check('missing trailing rows manifest ERROR', N.evaluate_directory(temp, autonomous_window=(0.,.02), mass_native=M, source_manifest=wrong)['status'] == 'ERROR')
        wrong = copy.deepcopy(manifest); wrong['chunks'][0]['sha256'] = '0'*64
        check('chunk hash tampering ERROR', N.evaluate_directory(temp, autonomous_window=(0.,.02), mass_native=M, source_manifest=wrong)['status'] == 'ERROR')
        cli_out = temp / 'cli.json'
        command = [sys.executable, '-B', str(HERE/'native_acceptance.py'), '--input-dir', str(temp), '--out', str(cli_out), '--autonomous', '0', '.02', '--weight', str(W), '--g', str(G)]
        proc = subprocess.run(command, capture_output=True, text=True)
        check('CLI uses explicit window/weight/g and strict JSON', proc.returncode == 0 and json.loads(cli_out.read_text())['status'] == 'OK', proc.stderr)
        proc = subprocess.run(command, capture_output=True, text=True)
        check('CLI refuses to overwrite history', proc.returncode != 0)
        (temp/'native_000001.npz').unlink()
        check('missing chunk sequence ERROR', N.evaluate_directory(temp, autonomous_window=(0.,.02), mass_native=M)['status'] == 'ERROR')
    report = dict(schema='native-acceptance-checks-v1', synthetic_checks=tests, synthetic_all_passed=all(x['passed'] for x in tests),
                  command=[str(Path(sys.executable).absolute()), '-B', str(Path(__file__).resolve()), *sys.argv[1:]],
                  executable=str(Path(sys.executable).absolute()), code={str(p):N.sha256(p) for p in [HERE/'native_acceptance.py', Path(__file__).resolve()]},
                  roadmap=dict(path=str(N.ROADMAP), sha256=N.sha256(N.ROADMAP), lines=[47,77]),
                  real_saved_recording=None, no_simulation_or_training=True,
                  memory_contract='required fields one chunk at a time; selected scalar quantiles only, sample and chunk allocation caps',
                  filter_status='proposed 1ms causal EMA, not fully frozen original processing',
                  D7_status='provisional .1W load threshold, not sourced by roadmap',
                  full_goal_acceptance='requires external evidence and processing review; no automatic PASS')
    if not args.skip_real:
        root = Path(args.real_dir).resolve()
        evaluation = root/'EVALUATION.json'
        info = json.loads(evaluation.read_text())
        summary = info['metrics']['native_peak_summary'] if 'metrics' in info else None
        if summary is None:
            # Current saved EVALUATION structure names the main summary under evaluation.
            def find_native(node):
                if isinstance(node,dict):
                    if 'native_peak_summary' in node:
                        return node['native_peak_summary']
                    for value in node.values():
                        found=find_native(value)
                        if found is not None: return found
                return None
            summary = find_native(info)
        if summary is None:
            raise AssertionError('saved EVALUATION lacks native recorder summary')
        with np.load(root/'native/native_000000.npz', allow_pickle=False) as chunk:
            start = float(chunk['time_before_s'][0])
        with np.load(root/f"native/native_{len(summary['chunks'])-1:06d}.npz", allow_pickle=False) as chunk:
            end = float(chunk['time_after_s'][-1])
        real = N.evaluate_directory(root/'native', autonomous_window=(start,end), mass_native=M,
                                    source_manifest=summary, metadata=dict(run_id='saved-bc-diagnostic-not-autonomous-proof'))
        real.setdefault('source', {})['parent_evaluation'] = dict(path=str(evaluation), sha256=N.sha256(evaluation))
        real['window_label'] = 'diagnostic only; no same-run auxiliary; physical data cannot establish autonomous control source'
        check('saved BC recording reads without creating historical artifacts', real['status'] == 'OK', real.get('errors'))
        check('saved BC without auxiliary stays D3/D5 UNKNOWN and no final PASS', all(real['criteria'][k]['status']=='UNKNOWN' for k in ('D3','D5')) and real['final_acceptance']['status'] != 'PASS')
        report['real_saved_recording'] = real
    report['wall_s'] = time.time() - started
    report['synthetic_all_passed'] = all(t['passed'] for t in tests)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(checks=len(tests), all_passed=report['synthetic_all_passed'], wall_s=round(report['wall_s'],3), out=str(out),
                         real_status=report['real_saved_recording']['status'] if report['real_saved_recording'] else None)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
