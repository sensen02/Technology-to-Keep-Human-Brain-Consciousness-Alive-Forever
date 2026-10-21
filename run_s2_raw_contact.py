"""Two fixed clocks; numeric ragged raw contact/constraint endpoint archive."""
from pathlib import Path
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
import numpy as np
import run_s2_contact_clock as base
from run_s2_fixed_stimulus import compiled_metadata
ROOT = Path(__file__).resolve().parent


def pack_ragged(rows, width=None, dtype=float):
    offsets = np.concatenate(([0], np.cumsum([len(r) for r in rows]))).astype(np.int64)
    shape = (-1,) if width is None else (-1, width)
    data = np.concatenate([np.asarray(r, dtype=dtype).reshape(shape) for r in rows], axis=0)
    return offsets, data


def main():
    import mujoco
    import engine.canonical_physics as cp
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('/home/sensen/Desktop/engine修复验收') / ('S2-raw-contact-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]))
    out = parser.parse_args().out.resolve(); out.mkdir(exist_ok=False)
    sources = [ROOT / n for n in ('run_s2_raw_contact.py', 'tools_s2_raw_contact.py', 'tools_verify_s2_raw_contact.py', 'test_s2_raw_contact.py', 'S2_RAW_CONTACT_PROTOCOL.zh-CN.md', 'run_s2_contact_clock.py', 'run_s2_fixed_stimulus.py', 'engine/canonical_physics.py')]
    sources += [base.POSE, Path(cp.CANONICAL_XML)]
    before = {str(p): base.sha(p) for p in sources}
    assert base.sha(cp.CANONICAL_XML) == cp.CANONICAL_XML_SHA256
    pose = np.load(base.POSE, allow_pickle=False)['projected_seated_qpos']
    records = []
    for dt, ratio in ((25e-6, 1), (12.5e-6, 2)):
        m = cp.load_model(); m.opt.timestep = dt; m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        d = mujoco.MjData(m); d.qpos[:] = pose; d.qvel[:] = 0; d.act[:] = 0; d.ctrl[:] = 0
        a = {'initial_' + k: getattr(d, k).copy() for k in ('qpos', 'qvel', 'act', 'ctrl')}
        rows = {k: [] for k in ('time_s', 'qpos', 'qvel', 'act', 'ctrl', 'ncon', 'nefc', 'solver_niter', 'step_ctrl', 'step_time_s', 'warning_counts')}
        fields = {'geom': (2, int), 'pos': (3, float), 'dist': (None, float), 'frame': (9, float), 'local6': (6, float), 'efc_address': (None, int), 'dim': (None, int), 'exclude': (None, int), 'friction': (5, float)}
        raw = {k: [] for k in fields}; efc = {k: [] for k in ('force', 'state', 'type', 'id')}; pairs = []
        for step in range(round(.01 / dt)):
            d.ctrl[:] = 0; d.ctrl[0] = .1 if step < round(.001 / dt) else 0
            rows['step_ctrl'].append(d.ctrl.copy()); rows['step_time_s'].append(float(d.time))
            mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
            if np.any(d.warning.number): raise RuntimeError('MuJoCo warning')
            if (step + 1) % ratio: continue
            assert d.ncon <= 512 and d.nefc <= 4096
            for k in ('qpos', 'qvel', 'act', 'ctrl', 'solver_niter'): rows[k].append(getattr(d, k).copy())
            for k in ('ncon', 'nefc'): rows[k].append(int(getattr(d, k)))
            rows['warning_counts'].append(d.warning.number.copy()); rows['time_s'].append(float(d.time))
            sample = {k: [] for k in fields}; summed = {}
            for i in range(d.ncon):
                c = d.contact[i]; force = np.zeros(6); mujoco.mj_contactForce(m, d, i, force)
                values = dict(geom=[int(c.geom1), int(c.geom2)], pos=c.pos.copy(), dist=float(c.dist), frame=c.frame.copy(), local6=force, efc_address=int(c.efc_address), dim=int(c.dim), exclude=int(c.exclude), friction=c.friction.copy())
                for k, v in values.items(): sample[k].append(v)
                p, world = base.pair_world(c.frame, force[:3], int(c.geom1), int(c.geom2))
                summed[p] = summed.get(p, np.zeros(3)) + world
            for k in raw: raw[k].append(sample[k])
            pairs.append({f'{p[0]},{p[1]}': v.tolist() for p, v in summed.items()})
            for k in efc: efc[k].append(getattr(d, 'efc_' + k)[:d.nefc].copy())
        a.update({k: np.asarray(v) for k, v in rows.items()})
        for k, (width, dtype) in fields.items():
            offset, data = pack_ragged(raw[k], width, dtype); a['contact_offsets'] = offset; a['contact_' + k] = data
        for k, rr in efc.items():
            offset, data = pack_ragged(rr, dtype=float if k == 'force' else int); a['efc_offsets'] = offset; a['efc_' + k] = data
        assert all(np.isfinite(v).all() for v in a.values())
        assert np.allclose(a['time_s'], np.arange(1,401)*25e-6, rtol=0, atol=1e-14)
        name = 'native_floor_pulse_' + str(dt) + '.npz'; np.savez_compressed(out / name, **a)
        (out / (name + '.pairs.json')).write_text(json.dumps(pairs))
        records.append(dict(npz=name, dt_s=dt, pair_json=name+'.pairs.json', cone=int(m.opt.cone), compiled=compiled_metadata(m,mujoco)))
        del a, raw, efc, rows, d, m
    after = {str(p): base.sha(p) for p in sources}; assert before == after
    manifest = dict(records=records, source_hashes_before=before, source_hashes_after=after, environment=dict(python=sys.executable,mujoco=mujoco.__version__), fixed_cases=2, physical_cause='unknown', biological_validation=False, body_convergence_proven=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from tools_s2_raw_contact import analyze
    analyze(out)
    print(str(out))


if __name__ == '__main__': main()
