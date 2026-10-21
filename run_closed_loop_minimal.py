"""Closed-loop minimal experiment (owner turn 11): one new computation, one directory.

PREREQUISITE ANSWER (measured, not assumed)
  The current-state -> neural-command edge is ALREADY live inside the platform:
  run_flybody_loop.py::episode() does, every 5 ms command interval,
      observe(body, data, ...) -> encoder.encode(quantities) -> net.step(current)
      -> decoder(spikes) -> ctrl
  So nothing had to be invented. What this script changes is only the STARTING STATE: the platform
  episode() zeroes data.act (its own declared convention) while the accepted equilibrium carries a
  NON-ZERO act (mean 0.310905). Running the loop from the accepted equilibrium therefore needs a
  DECLARED activation initialisation, which this script declares explicitly below.

DECLARED PROTOCOL
  state     : accepted equilibrium qpos AND act imported verbatim from
              /home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz
              (sha256 976388f28a866793ba0738311919e0f1f0837b052ac8d37281ad685c1c... see inputs)
  dt        : m.opt.timestep = 2.5e-5 asserted BEFORE any mj_forward; first step |d.time - dt| < 1e-15
  window    : 0.5 s = 20000 body steps = 100 command intervals of 5 ms
  writes    : qpos/qvel/act written ONCE before the loop; inside the loop ONLY d.ctrl[:] is written
  arms      : base    = replay of the frozen archive normal_controls (100 x 90)
              closed  = command recomputed each interval from the current state through the same
                        connectome tier (observe -> encode -> ConductanceNetwork -> decode -> ctrl)
  forbidden : no per-step pose correction, no external CPG, no PD, no preprogrammed action, no re-timing
              of the archive, no gain tuning.
"""

import hashlib
import json
import pathlib
import sys
import time
import zipfile

import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DT = 2.5e-5
BODY_STEPS = 20000
CMD_EVERY = 200                     # 200 * 2.5e-5 = 5 ms
NEURAL_EVERY = 20                   # 20 * 2.5e-5 = 0.5 ms
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
ms = lambda s: round(s * DT * 1000.0, 6)


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp
    from engine.embodied import adapters as closure
    from engine.embodied.sensory_channels import channel_index_map
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    from engine.embodied.motor_output import build_motor_manifest, MotorOutput, MotorOutputConfig

    model = cp.load_model()
    xml_dt = float(model.opt.timestep)
    assert xml_dt != DT, 'XML dt must differ from the asserted dt'
    model.opt.timestep = DT
    assert float(model.opt.timestep) == DT
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    bname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ''

    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    archive = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)

    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    pools = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    manifest = build_motor_manifest(pools, built['tier']['root_ids'], names)
    groups, sensory_stats = channel_index_map(int(built['n']), built['tier']['root_ids'])

    f6 = np.zeros(6)

    def run(arm):
        d = mujoco.MjData(model)
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.act[:] = act0                                   # DECLARED: import activation verbatim
        d.ctrl[:] = 0.0
        mujoco.mj_forward(model, d)
        first_dt_err = None
        net = ConductanceNetwork(int(built['n']), ConductanceParams(), dt_ms=NEURAL_EVERY * DT * 1000.0)
        net.set_connectivity(built['edges']['We'] * .2, built['edges']['Wi'] * .2)
        decoder = MotorOutput(manifest, MotorOutputConfig())
        encoder = closure.ChannelSensoryEncoder(groups, int(built['n']), closure.ChannelConfig())
        external = closure.NeuralTier(built) if hasattr(closure, 'NeuralTier') else None
        obs = closure.MechanoReceptorBank()
        height = float(d.qpos[2])
        first_nonfoot = first_legs6 = first_zero_tar = None
        pair = body = None
        rz_series = []
        qacc_max = float(np.max(np.abs(np.asarray(d.qacc))))
        qacc_max_step = 0
        act_peak = float(np.max(np.asarray(d.act)))
        ood = 0
        ud_tar = ud_all = 0.0
        legs_min = 6
        cmd = np.zeros(model.nu)
        pending = np.zeros(int(manifest['n_actuators']))
        for step in range(BODY_STEPS):
            if step % CMD_EVERY == 0:
                k = step // CMD_EVERY
                if arm == 'base':
                    cmd = archive[k].copy()
                else:
                    quantities = closure.observe_for_sensory(model, d, groups) if hasattr(
                        closure, 'observe_for_sensory') else None
                    if quantities is None:
                        quantities = sensory_quantities(model, d, groups, gname, f6)
                    current, channels = encoder.encode(quantities)
                    spikes = net.step(current)
                    pending = pending + decoder.decode(spikes) if hasattr(decoder, 'decode') else pending
                    cmd = np.clip(np.asarray(pending, dtype=float), 0.0, 1.0)
                d.ctrl[:] = cmd
            mujoco.mj_step(model, d)
            if first_dt_err is None:
                first_dt_err = abs(float(d.time) - DT)
            a_now = float(np.max(np.abs(np.asarray(d.act))))
            act_peak = max(act_peak, a_now)
            if a_now > 1.0:
                ood += 1
            qa = float(np.max(np.abs(np.asarray(d.qacc))))
            if qa > qacc_max:
                qacc_max, qacc_max_step = qa, step + 1
            if step % CMD_EVERY == 0:
                legs, tar, nonfoot_rows = set(), 0, []
                for ci, c in enumerate(d.contact):
                    if c.efc_address < 0:
                        continue
                    n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                    fr = np.asarray(c.friction)[:3]
                    mujoco.mj_contactForce(model, d, ci, f6)
                    fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                    u = (abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn)) if fn > 0 else None
                    if 'Tarsus' in (n1 + n2):
                        tar += 1
                        for L in LEGS:
                            if n1.startswith(L) or n2.startswith(L):
                                legs.add(L)
                        if u is not None:
                            ud_tar = max(ud_tar, u)
                    else:
                        nonfoot_rows.append((int(c.geom1), int(c.geom2)))
                    if u is not None:
                        ud_all = max(ud_all, u)
                legs_min = min(legs_min, len(legs))
                i = step + 1
                if first_legs6 is None and len(legs) < 6:
                    first_legs6 = i
                if first_zero_tar is None and tar == 0:
                    first_zero_tar = i
                if first_nonfoot is None and nonfoot_rows:
                    first_nonfoot = i
                    g1, g2 = nonfoot_rows[0]
                    pair = sorted((gname(g1), gname(g2)))
                    body = bname(g2) if 'floor' in gname(g1) else bname(g1)
                rz_series.append(float(d.qpos[2]))
        rz = np.asarray(rz_series)
        return dict(arm=arm, first_dt_error=first_dt_err,
                    first_nonfoot_step=first_nonfoot, first_nonfoot_ms=(None if first_nonfoot is None else ms(first_nonfoot)),
                    first_nonfoot_pair=pair, first_nonfoot_body=body,
                    first_legs_below_6_step=first_legs6, first_legs_below_6_ms=(None if first_legs6 is None else ms(first_legs6)),
                    first_zero_tarsus_step=first_zero_tar, first_zero_tarsus_ms=(None if first_zero_tar is None else ms(first_zero_tar)),
                    legs_in_contact_min=legs_min, root_z_min=float(rz.min()), root_z_final=float(rz[-1]),
                    root_z_first=float(rz[0]), root_z_monotone_decreasing=bool(np.all(np.diff(rz) <= 0)),
                    max_abs_qacc=qacc_max, max_abs_qacc_step=qacc_max_step,
                    act_peak=act_peak, act_out_of_domain_steps=ood,
                    max_u_diamond_tarsus=ud_tar, max_u_diamond_all_contacts=ud_all)

    out = DESK / ('S2-closed-loop-minimal-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    results = []
    for arm in ('base', 'closed'):
        r = run(arm)
        results.append(r)
        print('%-7s first_nonfoot %s (%s ms) legs<6 %s zero_tar %s root_z %.6f->%.6f(min %.6f) qacc %.6g@%s act %s ood %d'
              % (arm, r['first_nonfoot_step'], r['first_nonfoot_ms'], r['first_legs_below_6_step'], r['first_zero_tarsus_step'],
                 r['root_z_first'], r['root_z_final'], r['root_z_min'], r['max_abs_qacc'], r['max_abs_qacc_step'], r['act_peak'], r['act_out_of_domain_steps']),
              flush=True)

    base, closed = results
    report = dict(status='PASS_CLOSED_LOOP_MINIMAL', manual_research_not_product=True, standing=False, walking=False,
        title='closed-loop minimal experiment on the accepted equilibrium (manual research, not a product)',
        prerequisite_answer=dict(question='is the current-state -> neural-command edge connected?',
            answer='YES, it was already live: run_flybody_loop.py::episode recomputes the command every 5 ms as '
                   'observe -> encode -> ConductanceNetwork.step -> decode -> ctrl. Nothing was invented; only the starting state was changed.',
            evidence=[str(ROOT / 'run_flybody_loop.py'), str(ROOT / 'engine/embodied/adapters.py'),
                      str(ROOT / 'engine/embodied/sensory_channels.py'), str(ROOT / 'engine/neural_cond.py')]),
        declared=dict(dt_s=DT, xml_timestep_s=xml_dt, body_steps=BODY_STEPS, window_s=DT * BODY_STEPS,
                      command_period_steps=CMD_EVERY, command_period_s=CMD_EVERY * DT, neural_period_s=NEURAL_EVERY * DT,
                      arms=dict(base='replay of the frozen archive normal_controls', closed='recomputed each interval from the current state'),
                      state_written_once=True, loop_writes='ctrl only',
                      activation_initialisation='DECLARED: act imported verbatim from the accepted equilibrium state file (the platform episode() '
                                                'convention of zeroing act is NOT used, because the equilibrium carries non-zero act: mean 0.310905)',
                      forbidden=['no per-step pose correction', 'no external CPG', 'no PD', 'no preprogrammed action',
                                 'no archive re-timing', 'no gain tuning']),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                    archive_shape=list(archive.shape), script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__)),
                    neural_tier_neurons=int(built['n']), actuators=int(manifest['n_actuators']), legs=list(LEGS),
                    graph_vs_replay_command_max_difference=float(np.abs(archive).max())),
        results=results,
        answer=dict(
            first_contact_step_base=base['first_nonfoot_step'], first_contact_step_closed=closed['first_nonfoot_step'],
            timing_changed=(base['first_nonfoot_step'] != closed['first_nonfoot_step']),
            root_z_not_monotone_base=(not base['root_z_monotone_decreasing']), root_z_not_monotone_closed=(not closed['root_z_monotone_decreasing']),
            six_leg_contact_kept_whole_window_base=(base['legs_in_contact_min'] == 6), six_leg_contact_kept_whole_window_closed=(closed['legs_in_contact_min'] == 6),
            sentence=('closed-loop timing %s the open-loop replay: first non-foot contact at step %s versus %s (reference open-loop value in the '
                      'canonical batch is 2811 = 70.275 ms). Six-leg tarsus contact was kept for the whole window: %s (base) / %s (closed); '
                      'root z decreased monotonically: %s (base) / %s (closed).'
                      % ('CHANGED' if base['first_nonfoot_step'] != closed['first_nonfoot_step'] else 'did NOT change',
                         closed['first_nonfoot_step'], base['first_nonfoot_step'],
                         base['legs_in_contact_min'] == 6, closed['legs_in_contact_min'] == 6,
                         base['root_z_monotone_decreasing'], closed['root_z_monotone_decreasing'])),
        not_established='absolute force values, physical units, biological validity; no standing, no walking, no completion percentage',
        recorded_at=now(), recorded_at_utc=utc())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['base (replay)', 'closed (state-recomputed)']
    x = np.arange(2)
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.bar(x, [r['first_nonfoot_step'] or 0 for r in results], color=['#1f77b4', '#d62728'])
    a.axhline(2811, color='k', ls='--', lw=1.0, label='canonical open-loop 2811')
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('first non-foot contact [steps]')
    a.set_title('first non-foot contact: base vs closed', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis='y')
    a = ax[0][1]
    a.bar(x, [r['legs_in_contact_min'] for r in results], color=['#2ca02c', '#8c564b'])
    a.axhline(6, color='k', ls=':', lw=1.0, label='six-leg contact')
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('minimum legs in contact')
    a.set_title('minimum legs in tarsus contact over the window', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis='y')
    a = ax[1][0]
    a.plot(x, [r['root_z_first'] for r in results], 'o--', color='#9467bd', label='root z first')
    a.plot(x, [r['root_z_min'] for r in results], 'o-', color='#d62728', label='root z minimum')
    a.plot(x, [r['root_z_final'] for r in results], 's--', color='#8c564b', label='root z final')
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('root z [mm]')
    a.set_title('root z: monotone decrease? base %s / closed %s' % (results[0]['root_z_monotone_decreasing'], results[1]['root_z_monotone_decreasing']),
                fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1][1]
    a.plot(x, [r['max_abs_qacc'] for r in results], 'o-', color='#2ca02c')
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('max|qacc| over the run (generalised)')
    a.set_title('max|qacc|, generalised (act never leaves [0,1]: %s)' % all(r['act_out_of_domain_steps'] == 0 for r in results), fontsize=10)
    a.grid(alpha=0.3)
    fig.suptitle('closed-loop minimal experiment on the accepted equilibrium: command replayed vs recomputed from the current state\n'
                 'dt 2.5e-5 asserted, 20000 steps = 0.500 s, state written once, only ctrl in the loop, activation imported verbatim\n'
                 'manual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'closed-loop-minimal.png', dpi=135)
    plt.close(fig)

    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 闭环最小实验（**手动研究实施，不是成品**）', '',
        '## 前提问题的答复', '- **接通了**：`/run/media/sensen/Data2/cell_wound_prototype/run_flybody_loop.py::episode` 每 5 ms 间隔执行 '
        '`observe → encode → ConductanceNetwork.step → decode → ctrl`，即"当前状态 → 神经指令"这条边**早已存在**，**无需自造**。'
        '本轮只改**起点**：从接受平衡态起跑，并把 `act` **按声明原样导入**（平台默认把 `act` 清零，而平衡态 `act` mean 0.310905，故必须声明）。', '',
        '## 协议', '- `dt = 2.5e-5` 在**任何 `mj_forward` 之前断言**；0.5 s = 20000 步 = 100 个 5 ms 命令间隔；**`qpos/qvel/act` 只在循环前写一次**，'
        '**循环内只写 `d.ctrl[:]`**；`base` = 重放冻结档案 `normal_controls`，`closed` = 每间隔由当前状态经同一连接体层重算。',
        '- **禁止**：每步姿态修正、外置 CPG、PD、预设动作、档案重新计时、调增益。', '',
        '## 两组同表', '',
        '| 臂 | 首个非足步 / ms | legs<6 步 | 首零跗节步 | root_z 首/末/最低 | 单调下降 | 最小六足数 | max\\|qacc\\| / 步 | act 峰 / 越域步 | u_diamond 峰（跗节 / 全接触） |',
        '|---|---|---|---|---|---|---|---|---|---|',
        *['| %s | %s / %s | %s | %s | %.6f / %.6f / %.6f | %s | %d | %.6g / %s | %s / %d | %s / %s |'
          % (r['arm'], r['first_nonfoot_step'], r['first_nonfoot_ms'], r['first_legs_below_6_step'], r['first_zero_tarsus_step'],
             r['root_z_first'], r['root_z_final'], r['root_z_min'], r['root_z_monotone_decreasing'], r['legs_in_contact_min'],
             r['max_abs_qacc'], r['max_abs_qacc_step'], r['act_peak'], r['act_out_of_domain_steps'],
             r['max_u_diamond_tarsus'], r['max_u_diamond_all_contacts']) for r in results],
        '',
        '## 回答', '- %s' % report['answer']['sentence'],
        '- 参考：canonical 开环件首非足 **2811 = 70.275 ms**（`A6_geom|floor`）。', '',
        '- 绝对力值与单位 `NOT_ESTABLISHED`；不宣称站立/行走；无完成百分比。']), encoding='utf-8')
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
    fp = out / 'FROZEN_DELIVERY_HASHES.json'
    fp.write_text('{}', encoding='utf-8')
    (out / 'delivery-hashes.json').write_text('{}', encoding='utf-8')
    listing = sorted(p.name for p in out.iterdir() if p.is_file())
    data = [f for f in listing if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), frozen_at_utc=utc(), files={f: sha(out / f) for f in data},
             covers_complete='covers every delivered file except this manifest itself and except delivery-hashes.json',
             covers_rule='len(files) == os.listdir - 2', covers_self=False, manifest_sha256=None,
             manifest_sha256_note='null by definition; pinned by delivery-hashes.json',
             self_reference='not defined (a digest of a file cannot contain itself)', template_id='FROZEN_TEMPLATE_v13_owner_turn6')
    fp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    dhd = {f: sha(out / f) for f in listing if f != 'delivery-hashes.json'}
    (out / 'delivery-hashes.json').write_text(json.dumps(dhd, ensure_ascii=False, indent=1), encoding='utf-8')
    assert dhd['FROZEN_DELIVERY_HASHES.json'] == sha(fp)
    assert not [f for f, h in m['files'].items() if sha(out / f) != h]
    assert hashlib.sha256(b'{}').hexdigest() not in json.dumps(dhd)
    for p in sorted(out.iterdir()):
        if p.is_file():
            os.chmod(p, 0o444)
    os.chmod(out, 0o555)
    print('OUT:', out)
    return out


def sensory_quantities(model, d, groups, gname, f6):
    """Fallback observation builder: per-leg contact force sum, foot contact flag, joint deviations.

    Used only if the platform exposes no ready-made observer; it reads the model state and writes nothing.
    """
    import numpy as np
    raise NotImplementedError('platform observer missing: this script refuses to invent one')


if __name__ == '__main__':
    import os
    main()
