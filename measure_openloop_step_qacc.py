"""Per-step max|qacc| along the two accepted-equilibrium open-loop trajectories (the "capture gap" first half), written
INTO the existing artifacts (no new directory): the 0.5 s loop is re-run with qacc recorded every step, and the result is
added as per_step_max_qacc.npz + a report block + a README line, with before/after hashes appended to the reanchor table.
"""

import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
sys.path.insert(0, str(ROOT))
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
ARTS = [DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559',
        DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164943-from-accepted-pose-archive']
REANCHOR = DESK / 'S2-equilibrium-hash-reanchor-20261015T165026'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


def run(drive_path, qpos, act0, model, mujoco, dt=25e-6, steps=20000):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(model, d)
    out = np.zeros(steps + 1)
    out[0] = float(np.max(np.abs(np.asarray(d.qacc))))
    ctrl_seq = np.asarray(np.load(drive_path)['normal_controls'], dtype=float)
    time_s_measured = np.zeros(steps + 1)
    for step in range(steps):
        idx = min(step // int(round(0.005 / dt)), ctrl_seq.shape[0] - 1)
        d.ctrl[:] = ctrl_seq[idx]
        mujoco.mj_step(model, d)
        time_s_measured[step + 1] = float(d.time)
        out[step + 1] = float(np.max(np.abs(np.asarray(d.qacc))))
    return out


def main():
    import mujoco
    from engine import canonical_physics as cp
    m = cp.load_model()
    m.opt.timestep = 25e-6  # these scripts did not set it before turn 24; their ms labels were 4x too small
    assert float(m.opt.timestep) == 25e-6
    st = np.load(STATE)
    qpos, act0 = st['qpos'].copy(), st['act'].copy()
    dt = 25e-6
    edits = []
    for art in ARTS:
        rep_path = art / 'report.json'
        rep = json.loads(rep_path.read_text())
        drive = rep['inputs']['drive']
        assert rep['inputs']['state_npz_sha256'] == sha(STATE), 'state hash mismatch'
        qacc = run(drive, qpos, act0, m, mujoco)
        first_gate = int(np.argmax(qacc > 1e-3)) if (qacc > 1e-3).any() else None
        first_tol = int(np.argmax(qacc > 1.0)) if (qacc > 1.0).any() else None
        block = dict(
            definition='max over the 48 DOF of |qacc| after each mj_step, qvel carried, state written once, only ctrl in the loop',
            step0_mm_s2=float(qacc[0]), final_step_mm_s2=float(qacc[-1]),
            window_max_mm_s2=float(qacc.max()), window_max_step=int(np.argmax(qacc)),
            window_min_mm_s2=float(qacc.min()),
            first_step_above_declared_gate_1e_minus_3=first_gate,
            first_step_above_declared_gate_1e_minus_3_ms=(None if first_gate is None else round(first_gate * dt * 1000, 3)),
            first_step_above_1_mm_s2_tolerance=first_tol,
            first_step_above_1_mm_s2_tolerance_ms=(None if first_tol is None else round(first_tol * dt * 1000, 3)),
            steps_above_declared_gate=int((qacc > 1e-3).sum()), steps_total=int(qacc.size),
            note='this block was added after the fact by re-running the same 0.5 s loop with qacc recorded per step; the '
                 'trajectory and every other number of this artifact are unchanged')
        rep['post_step_max_qacc'] = block
        before = sha(rep_path)
        rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        np.savez_compressed(art / 'per_step_max_qacc.npz', max_abs_qacc=qacc,
                            time_s=time_s_measured,
                            note='per-step max|qacc| of the accepted equilibrium open loop under this artifact drive')
        rmp = art / 'README.zh-CN.md'
        txt = rmp.read_text()
        if 'post_step_max_qacc' not in txt:
            txt += ('\n\n## 逐步 max|qacc|（后加，捕获缺口前半）\n'
                    '- 首步（状态写入后第一次 `mj_forward`）`max|qacc| = %.6g mm/s²`；**末步 `%.6g mm/s²`**；'
                    '全窗最大 `%.6g mm/s²`（步 %d）。\n'
                    '- **首次超过声明门限 1e-3 mm/s²：步 %s（%s ms）**；**首次超过 1.0 mm/s² 容差：步 %s（%s ms）**；'
                    '全程超过 1e-3 的步数 %d/%d。\n'
                    '- 明细见 `per_step_max_qacc.npz`；本块为**同一条 0.5 s 轨迹**补记，其他数值未变。\n'
                    % (block['step0_mm_s2'], block['final_step_mm_s2'], block['window_max_mm_s2'], block['window_max_step'],
                       block['first_step_above_declared_gate_1e_minus_3'], block['first_step_above_declared_gate_1e_minus_3_ms'],
                       block['first_step_above_1_mm_s2_tolerance'], block['first_step_above_1_mm_s2_tolerance_ms'],
                       block['steps_above_declared_gate'], block['steps_total']))
            rmp.write_text(txt)
        edits.append(dict(file=str(rep_path), before=before, after=sha(rep_path),
                          reason='added the per-step max|qacc| block (post-hoc re-run of the same trajectory)'))
        edits.append(dict(file=str(rmp), before=None, after=sha(rmp), reason='README paragraph for the per-step max|qacc| block'))
        files = sorted(p for p in art.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
        mf = art / 'delivery-hashes.json'
        old = json.loads(mf.read_text()) if mf.exists() else {}
        delivery = old.get('delivery_time_hashes') or {k: v for k, v in old.items() if isinstance(v, str)}
        nowh = {p.name: sha(p) for p in files}
        for k, v in nowh.items():
            delivery.setdefault(k, v)
        mf.write_text(json.dumps(dict(format='two columns: delivery_time_hashes (preserved) + now_hashes (current)',
                                      delivery_time_hashes=delivery, now_hashes=nowh,
                                      files_changed_since_delivery=sorted(k for k in nowh if delivery.get(k) and delivery[k] != nowh[k]),
                                      recorded_at=now()), ensure_ascii=False, indent=1))
        print(art.name)
        print('  step0 %.6g | final %.6g | window max %.6g at step %d' % (block['step0_mm_s2'], block['final_step_mm_s2'],
                                                                         block['window_max_mm_s2'], block['window_max_step']))
        print('  first > 1e-3 mm/s^2: step %s (%s ms) | first > 1.0: step %s (%s ms)'
              % (block['first_step_above_declared_gate_1e_minus_3'], block['first_step_above_declared_gate_1e_minus_3_ms'],
                 block['first_step_above_1_mm_s2_tolerance'], block['first_step_above_1_mm_s2_tolerance_ms']))
    table = REANCHOR / 'hash_reanchor_table.json'
    doc = json.loads(table.read_text())
    doc.setdefault('this_pass_edits', []).extend(edits)
    doc['recorded_at'] = now()
    table.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    (DESK / 'S2-openloop-pair-batch-20261015T165026/hash_reanchor_table.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
