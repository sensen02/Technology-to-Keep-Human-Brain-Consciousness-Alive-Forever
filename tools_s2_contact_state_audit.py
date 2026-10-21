"""Offline, dense frozen archive only; no MuJoCo import or dynamics."""
from pathlib import Path
import hashlib
import json
import csv
import numpy as np

ARCHIVE = Path('/home/sensen/Desktop/engine修复验收/S2-dynamic-residual-20261014T211552Z-7fbf1130')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def classify(J, qacc, aref, D, force, saved_state):
    J, qacc, aref, D, force, saved_state = map(np.asarray, (J, qacc, aref, D, force, saved_state))
    n = len(aref)
    if J.ndim != 2 or qacc.ndim != 1 or J.shape != (n, len(qacc)):
        raise ValueError('dense layout')
    if any(x.shape != (n,) for x in (aref, D, force, saved_state)):
        raise ValueError('row layout')
    if not all(np.isfinite(x).all() for x in (J, qacc, aref, D, force, saved_state)) or np.any(D <= 0):
        raise ValueError('nonfinite or nonpositive D')
    r = J @ qacc - aref
    state = np.where(r < 0, 1, 0)
    expected = np.maximum(0, -D*r)
    return r, state, force-expected, force+D*r


def main(out):
    out = Path(out); out.mkdir(exist_ok=False)
    inputs = {str(p): sha(p) for p in ARCHIVE.iterdir() if p.is_file()}
    manifest = json.loads((ARCHIVE/'manifest.json').read_text())
    sources = manifest['source_hashes_before']
    assert all(sha(p) == h for p, h in sources.items())
    reports = []; curves = []; all_bad = []
    for rec in manifest['records']:
        assert sha(ARCHIVE/rec['npz']) == rec['sha256']
        opt = rec['options']
        assert opt['solver'] == 2 and opt['cone'] == 0 and opt['noslip_iterations'] == 0
        with np.load(ARCHIVE/rec['npz'], allow_pickle=False) as loaded:
            a = {k: loaded[k] for k in loaded.files}
            assert not np.any(a['sparse'])
            table = {}; errors = []; old_e = []; fresh_e = []; series = []; worst = None; rd = []
            for i in range(len(a['time_s'])):
                lo, hi = map(int, a['efc_offsets'][i:i+2]); jl, jh = map(int, a['J_offsets'][i:i+2])
                J = a['J_values'][jl:jh].reshape(hi-lo, rec['nv'])
                r, state, error, e = classify(J, a['qacc'][i], a['efc_aref'][lo:hi], a['efc_D'][lo:hi], a['efc_force'][lo:hi], a['efc_state'][lo:hi])
                saved = a['efc_state'][lo:hi]; mask = a['efc_type'][lo:hi] == 6
                errors.extend(error[mask]); old_e.extend(e[mask & (saved == 1)]); fresh_e.extend(e[mask & (state == 1)])
                rd.extend((a['efc_R'][lo:hi]*a['efc_D'][lo:hi]-1)[mask])
                series.append([float(np.max(np.abs(e[mask & (saved == 1)]))), float(np.max(np.abs(error[mask])))])
                cl, ch = map(int, a['contact_offsets'][i:i+2])
                for row in np.flatnonzero(mask):
                    key = f'{int(saved[row])}->{int(state[row])}'
                    table[key] = table.get(key, 0)+1
                    cid = int(a['efc_id'][lo+row]); c = cl+cid
                    assert cl <= c < ch
                    edge = int(row-a['contact_efc_address'][c]); dim = int(a['contact_dim'][c])
                    assert 0 <= edge < 2*(dim-1)
                    item = dict(case=rec['npz'], endpoint=i, time_s=float(a['time_s'][i]), row=int(row), contact=cid, geom1=int(a['contact_geom'][c,0]), geom2=int(a['contact_geom'][c,1]), edge=edge, saved_state=int(saved[row]), derived_state=int(state[row]), r=float(r[row]), force=float(a['efc_force'][lo+row]), D=float(a['efc_D'][lo+row]), old_e=float(e[row]), piecewise_error=float(error[row]))
                    if saved[row] != state[row]: all_bad.append(item)
                    if saved[row] == 1 and (worst is None or abs(item['old_e']) > abs(worst['old_e'])): worst = item
            maxabs = lambda v: float(np.max(np.abs(v))) if len(v) else None
            reports.append(dict(case=rec['npz'], rows=len(errors), cross_table=table, saved_quadratic_e_absmax=maxabs(old_e), derived_quadratic_e_absmax=maxabs(fresh_e), piecewise_error_absmax=maxabs(errors), RD_minus_one_absmax=maxabs(rd), worst_old_e=worst))
            curves.append((a['time_s'].copy(), np.asarray(series), rec['dt_s']))
    assert inputs == {str(p): sha(p) for p in ARCHIVE.iterdir() if p.is_file()}
    assert all(sha(p) == h for p,h in sources.items())
    report = dict(policy='REPORT_ONLY', cases=reports, mismatched_rows=len(all_bad), frozen_inputs_unchanged=True, source_hash_count=len(sources), island_final_state='not captured; derived state is not an observed iefc_state', limitations='not stationarity, full KKT, mass balance, convergence, physical cause or biological validation')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'input-hashes.json').write_text(json.dumps(inputs, indent=2)+'\n')
    with (out/'state-mismatch-rows.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(all_bad[0])); w.writeheader(); w.writerows(all_bad)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10,7))
    for time, values, dt in curves:
        for k, ax in enumerate(axes): ax.plot(time*1000, values[:,k], label=f'{dt*1e6:g} us')
    axes[0].set_ylabel('Saved-label quadratic e absmax')
    axes[1].set_ylabel('Piecewise force error absmax')
    axes[1].set_xlabel('Time (ms)')
    for ax in axes: ax.legend(); ax.grid(alpha=.3)
    fig.suptitle('Offline state-label audit / REPORT ONLY; model row-force units')
    fig.tight_layout(); fig.savefig(out/'state-audit.png'); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
