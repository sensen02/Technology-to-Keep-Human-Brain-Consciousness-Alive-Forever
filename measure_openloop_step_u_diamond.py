"""Per-step max u_diamond, the first step that reaches the declared diamond limit, and the root z displacement relative to
pose A, along the accepted-equilibrium open loop. Written INTO the existing artifact (no new directory); the artifact is
frozen afterwards and only then is a verification dispatched.
"""

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
sys.path.insert(0, str(ROOT))
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
ARTS = [DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559',
        DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164943-from-accepted-pose-archive']
REANCHOR = DESK / 'S2-equilibrium-hash-reanchor-20261015T165026'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


def main():
    import mujoco
    from engine import canonical_physics as cp
    m = cp.load_model()
    m.opt.timestep = 25e-6  # these scripts did not set it before turn 24; their ms labels were 4x too small
    assert float(m.opt.timestep) == 25e-6
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    st = np.load(STATE)
    qpos0, act0 = st['qpos'].copy(), st['act'].copy()
    pose_a_z = float(np.load(POSE_A)['projected_seated_qpos'][2])
    dt, steps = 25e-6, 20000
    edits = []
    for art in ARTS:
        rep_path = art / 'report.json'
        rep = json.loads(rep_path.read_text())
        drive = rep['inputs']['drive']
        ctrl_seq = np.asarray(np.load(drive)['normal_controls'], dtype=float)
        d = mujoco.MjData(m)
        d.qpos[:] = qpos0
        d.qvel[:] = 0.0
        d.act[:] = act0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        u_max = np.zeros(steps + 1)
        u_contact_max = np.zeros(steps + 1)
        rz = np.zeros(steps + 1)
        f6 = np.zeros(6)

        def sample(i):
            rz[i] = float(d.qpos[2])
            worst = 0.0
            n_tar = 0
            for ci, c in enumerate(d.contact):
                if c.efc_address < 0:
                    continue
                n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                if 'Tarsus' not in (n1 + n2):
                    continue
                n_tar += 1
                mujoco.mj_contactForce(m, d, ci, f6)
                fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                fr = np.asarray(c.friction)[:3]
                if fn > 0:
                    worst = max(worst, abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn))
            u_contact_max[i] = worst
            u_max[i] = worst if n_tar else 0.0

        sample(0)
        time_s_measured = np.zeros(steps + 1)
    for step in range(steps):
            idx = min(step // int(round(0.005 / dt)), ctrl_seq.shape[0] - 1)
            d.ctrl[:] = ctrl_seq[idx]
            mujoco.mj_step(m, d)
            sample(step + 1)

        first_ge_1 = int(np.argmax(u_contact_max >= 1.0)) if (u_contact_max >= 1.0).any() else None
        first_ge_095 = int(np.argmax(u_contact_max >= 0.95)) if (u_contact_max >= 0.95).any() else None
        disp = rz - pose_a_z
        block = dict(
            definition=('max over the tarsus-floor contacts of the declared axis-ray diamond '
                        'u_diamond = |ft1|/(mu1*fn) + |ft2|/(mu1*fn), evaluated after each mj_step; zero recorded when a step has no '
                        'tarsus contact'),
            max_u_diamond_window=float(u_contact_max.max()), max_u_diamond_step=int(np.argmax(u_contact_max)),
            max_u_diamond_step0=float(u_contact_max[0]),
            first_step_u_diamond_ge_1_0=first_ge_1,
            first_step_u_diamond_ge_1_0_ms=(None if first_ge_1 is None else round(first_ge_1 * dt * 1000, 3)),
            first_step_u_diamond_ge_0_95=first_ge_095,
            first_step_u_diamond_ge_0_95_ms=(None if first_ge_095 is None else round(first_ge_095 * dt * 1000, 3)),
            steps_with_u_diamond_ge_1_0=int((u_contact_max >= 1.0).sum()),
            root_z_displacement_vs_pose_A=dict(
                pose_A_root_z=pose_a_z, initial_mm=float(disp[0]), min_mm=float(disp.min()),
                min_step=int(np.argmin(disp)), final_mm=float(disp[-1]),
                note='signed: root_z(t) minus the root z of the delivered pose A (mm)'),
            note='added after the fact by re-running the same 0.5 s loop with u_diamond recorded per step; no other number changes')
        rep['post_step_u_diamond'] = block
        before_rep = sha(rep_path)
        rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        np.savez_compressed(art / 'per_step_u_diamond.npz', max_u_diamond=u_contact_max, root_z=rz,
                            root_z_displacement_vs_pose_A=disp, time_s=time_s_measured,
                            note='per-step max u_diamond over tarsus-floor contacts and the root z displacement relative to pose A')
        rmp = art / 'README.zh-CN.md'
        txt = rmp.read_text()
        if 'post_step_u_diamond' not in txt:
            txt += ('\n\n## 逐步 max u_diamond 与相对姿态 A 的位移（后加）\n'
                    '- `max u_diamond`（全窗最大）**`%.6g`**（步 %d）；首步 `%.6g`。\n'
                    '- **首次达到菱形极限 1.0：步 %s（%s ms）**；首次 ≥ 0.95：步 %s（%s ms）；全程 ≥1.0 的步数 %d。\n'
                    '- **root z 相对交付姿态 A 的位移**：姿态 A 的 root z `%.6f`；首步 `%+.6f mm`、最小 `%+.6f mm`（步 %d）、末值 `%+.6f mm`。\n'
                    '- 明细 `per_step_u_diamond.npz`；本块为同一条 0.5 s 轨迹补记，其他数值未变。\n'
                    % (block['max_u_diamond_window'], block['max_u_diamond_step'], block['max_u_diamond_step0'],
                       block['first_step_u_diamond_ge_1_0'], block['first_step_u_diamond_ge_1_0_ms'],
                       block['first_step_u_diamond_ge_0_95'], block['first_step_u_diamond_ge_0_95_ms'],
                       block['steps_with_u_diamond_ge_1_0'], pose_a_z,
                       block['root_z_displacement_vs_pose_A']['initial_mm'], block['root_z_displacement_vs_pose_A']['min_mm'],
                       block['root_z_displacement_vs_pose_A']['min_step'], block['root_z_displacement_vs_pose_A']['final_mm']))
            rmp.write_text(txt)
        edits.append(dict(file=str(rep_path), before=before_rep, after=sha(rep_path),
                          reason='added the per-step u_diamond / root-z-displacement block'))
        edits.append(dict(file=str(rmp), before=None, after=sha(rmp), reason='README paragraph for that block'))
        print(art.name)
        print('  max u_diamond %.6g at step %d | first >=1.0 step %s (%s ms) | first >=0.95 step %s'
              % (block['max_u_diamond_window'], block['max_u_diamond_step'], block['first_step_u_diamond_ge_1_0'],
                 block['first_step_u_diamond_ge_1_0_ms'], block['first_step_u_diamond_ge_0_95']))
        print('  root z - pose A: init %+.6f | min %+.6f at step %d | final %+.6f'
              % (block['root_z_displacement_vs_pose_A']['initial_mm'], block['root_z_displacement_vs_pose_A']['min_mm'],
                 block['root_z_displacement_vs_pose_A']['min_step'], block['root_z_displacement_vs_pose_A']['final_mm']))

    table = REANCHOR / 'hash_reanchor_table.json'
    doc = json.loads(table.read_text())
    doc.setdefault('this_pass_edits', []).extend(edits)
    doc['recorded_at'] = now()
    table.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    (DESK / 'S2-openloop-pair-batch-20261015T165026/hash_reanchor_table.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    for art in ARTS:
        files = sorted(p for p in art.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
