"""Compare the stock matrix profile and a passive, damage-free voltage control."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OMNI = Path('/run/media/sensen/Data1/omnisim')
ARCHIVE = Path('/run/media/sensen/Data1/omnisim-results/implant-site/banc-neck/array64-45deg/array-audit.json')
sys.path[:0] = [str(BASE/'20261120T051324Z-neck-spike-sorting'), str(OMNI)]
from evaluate import reference, grade
from run import PROJECT, lead_vectors, voltages, load_fixed_neck
from sorting import learn
from matrix_sorting import detect


def main():
    profile = json.loads((HERE/'matrix.json').read_text())
    a = np.array(profile['array']['axes_um'])
    assert a.shape == (64, 2, 3)
    delta = a[:,1] - a[:,0]
    tilt = np.rad2deg(np.arctan2(np.linalg.norm(delta[:,:2], axis=1), delta[:,2]))
    np.testing.assert_allclose(tilt, 45.)
    assert np.ptp(a[:,1,2]) < 1e-10 and len(np.unique(a[:,1], axis=0)) == 64
    assert min(profile['array']['solid_voxels_per_needle']) > 0
    normal = delta[0]/np.linalg.norm(delta[0])
    step = a[[1,8],1] - a[0,1]
    np.testing.assert_allclose(np.linalg.norm(step - (step@normal)[:,None]*normal, axis=1), 15.)
    assert ARCHIVE.read_bytes() == (HERE/'matrix.json').read_bytes()
    # Partition the coplanar tips by head-to-thorax coordinate; no access to root identity for decoding.
    order = np.lexsort((a[:,1,0], a[:,1,1]))
    contacts = a[order,1] * 1e-6
    assert contacts[31,1] < contacts[32,1]
    neck, _, _, identity = load_fixed_neck(PROJECT)
    paths = {p.root:p for p in neck}
    rec = BASE/'20261120T043500Z-neck-recording/recordings'
    arm_uv = []
    for seed in (10901,11311,11701):
        with np.load(rec/f'{seed}_intact_reference.npz') as f:
            roots = f['roots'].tolist()
            leads = lead_vectors(paths, roots, contacts)
            noise = np.random.default_rng(seed).normal(0., .08, (len(f['entry']),64))
            arm_uv.append(voltages(f['entry'],f['arrival'],leads,noise))
    rows = {r['root']:r for r in profile['routes']}
    ground = Counter((rows[r]['direction'],rows[r]['available']) for r in roots)
    templates = 0
    try:
        bank, lags, scale, _ = learn(arm_uv[:2])
    except ValueError as exc:
        if str(exc) != 'no recurring waveform supported in both calibration runs':
            raise
        status = str(exc)
        calls = np.zeros((0,3), int)
    else:
        templates = len(bank)
        status = 'calibration completed'
        calls = detect(arm_uv[-1],bank,lags,scale,multichannel=True,subtract=True)
    np.savez_compressed(HERE/'passive_predictions.npz', predictions=calls, contact_order=order)
    ref = reference(11701,'intact')
    result=dict(archive_equal=True, sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                axes=a.shape[0], tip_plane_spread_um=float(np.ptp(a[:,1,2])),
                tilt_deg=[float(tilt.min()),float(tilt.max())], applied_ground={str(k):v for k,v in ground.items()},
                remaining=[r for r in profile['routes'] if r['available']],
                healthy_dn_arrivals=sum(bool(r['arrival_ticks']) for r in profile['frozen_runtime']['intact'].values()),
                damaged_dn_arrivals=sum(bool(r['arrival_ticks']) for r in profile['frozen_runtime']['injured'].values()),
                damaged_sweep=[dict(Counter(e['kind'] for e in row['events'])) for row in profile['selected']['electrode']['sampling']],
                passive_control=dict(note='ideal non-damaging field on exported 64 tips, reduced end dipoles only',
                                     calibration_status=status, templates=templates, score=grade(calls,ref,{})))
    (HERE/'VERDICT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
