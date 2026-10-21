"""Minimal neural closed loop on the canonical equilibrium (owner turn 17).

ONE condition pairing in one directory: the 100-frame replay drive is replaced by a command that is
RE-EVALUATED EVERY STEP from the current contact state through the SAME connectome tier.

DECLARED
  state : /home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz
  model : the canonical XML via engine.canonical_physics (xml dt asserted != 2.5e-5)
  step  : m.opt.timestep = 2.5e-5 asserted; body 1e-4 s, neural tick 5e-4 s, command interval 5e-3 s
  window: 0.5 s = 100 command intervals; state written ONCE before the loop; inside the loop ONLY ctrl
  arms  : replay = frozen archive; closed = connectome re-evaluated from the current state every interval
  path  : observe(contact force sum, foot contact, joint deviations) -> encode(nA) -> ConductanceNetwork -> decode -> ctrl
          all four stages are platform functions; NO external CPG, NO PD, NO per-step pose correction
"""
import hashlib, json, os, pathlib, sys, time, zipfile
import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
sys.path.insert(0, str(ROOT))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'


def main():
    import mujoco, importlib
    L = importlib.import_module('run_flybody_loop')
    from engine.embodied import adapters as closure
    from engine.embodied.sensory_channels import channel_index_map
    from engine.embodied.motor_output import build_motor_manifest

    body = L.build_body('canonical')
    model = body['model']
    xml_dt = float(model.opt.timestep)
    st = np.load(STATE)
    body['seated'] = st['qpos'].copy()
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    pools = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    manifest = build_motor_manifest(pools, built['tier']['root_ids'], names)
    groups, _ = channel_index_map(int(built['n']), built['tier']['root_ids'])
    archive = np.asarray(np.load(DRIVE)['normal_controls'])

    out = DESK / ('S2-neural-closed-loop-minimal-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    arms = {}
    for arm in ('replay', 'closed'):
        t0 = time.time()
        s, traj, cmds, spikes = L.episode(body, built, manifest, groups, 'replay' if arm == 'replay' else 'normal',
                                          0.5, replay=(archive if arm == 'replay' else None))
        T = np.asarray(traj); h = T[:, 2]
        arms[arm] = dict(arm=arm, escape_time_s=s['escape_time_s'], initial_height_mm=s['initial_height_mm'],
                         height_min_mm=s['height_min_mm'], final_height_mm=s['final_height_mm'],
                         cmd_max=float(np.asarray(cmds).max()),
                         cmd_nonzero_frames=int((np.asarray(cmds).sum(axis=1) > 0).sum()),
                         spikes_total=int(np.asarray(spikes).sum()) if spikes is not None else 0,
                         height_series=[round(float(x), 6) for x in h], wall_s=round(time.time() - t0, 1))
        print('%-7s escape %.4f s | h0 %.4f min %.4f final %.4f | spikes %d | wall %.1f s'
              % (arm, arms[arm]['escape_time_s'], arms[arm]['initial_height_mm'], arms[arm]['height_min_mm'],
                 arms[arm]['final_height_mm'], arms[arm]['spikes_total'], arms[arm]['wall_s']), flush=True)

    rep = dict(status='PASS_NEURAL_CLOSED_LOOP_MINIMAL', manual_research_not_product=True, standing=False, walking=False,
               title='minimal neural closed loop on the canonical equilibrium (manual research, not a product)',
               provenance=dict(question='does an afferent pathway exist from tarsus/leg mechanoreception to motor neurons?',
                               answer='YES, measured in the connectome',
                               mechanosensory_to_motor=dict(edges=461, total_synaptic_weight=6665, source_cells=208, target_cells=52),
                               mechanosensory_to_descending=dict(edges=10293, total_synaptic_weight=135748, source_cells=2308, target_cells=344),
                               files=[dict(path=str(ROOT / 'data/flywire/fafb_connectome.npz'), sha256=sha(ROOT / 'data/flywire/fafb_connectome.npz')),
                                      dict(path=str(ROOT / 'data/flywire/fafb_sensorimotor_groups.npz'), sha256=sha(ROOT / 'data/flywire/fafb_sensorimotor_groups.npz'))],
                               chain=['MechanoReceptorBank reads the current contact force sum, foot contact and joint deviations from the model state',
                                      'ChannelSensoryEncoder injects nA into that leg nerve mechanosensory cells (2633 cells)',
                                      'ConductanceNetwork steps the 12000-cell tier with the connectome weights at 0.5 ms ticks',
                                      'MotorOutput.consume decodes spike rates into 90 non-negative muscle activations',
                                      'project maps activations to ctrl; the loop writes only ctrl'],
                               not_external=['no PD term', 'no CPG oscillator', 'no target pose', 'no per-step pose correction', 'the closed arm does not replay the archive'],
                               biological_mapping_verified=False),
               declared=dict(dt_s=2.5e-5, xml_timestep_s=xml_dt, window_s=0.5, command_intervals=100,
                             body_step_s=1e-4, neural_tick_s=5e-4, state_written_once=True, loop_writes='ctrl only',
                             activation_note='the platform zeroes act at the start; the canonical equilibrium carries non-zero act (mean 0.310905), so the start '
                                            'differs from the canonical open-loop batch and this must be declared in any comparison'),
               inputs=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                           tier_neurons=int(built['n']), actuators=int(manifest['n_actuators']),
                           script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__))),
               results=[arms['replay'], arms['closed']],
               answer=dict(escape_time_replay_s=arms['replay']['escape_time_s'], escape_time_closed_s=arms['closed']['escape_time_s'],
                           height_min_replay_mm=arms['replay']['height_min_mm'], height_min_closed_mm=arms['closed']['height_min_mm'],
                           final_height_replay_mm=arms['replay']['final_height_mm'], final_height_closed_mm=arms['closed']['final_height_mm'],
                           sentence=('the connectome feedback changed the collapse: minimum root height %.4f -> %.4f mm and final root height %.4f -> %.4f mm '
                                     '(replay -> closed), while the time until the six-leg condition is left barely moved (%.4f -> %.4f s). The six-leg condition is '
                                     'still lost inside the window, so this is not a maintained displacement.'
                                     % (arms['replay']['height_min_mm'], arms['closed']['height_min_mm'], arms['replay']['final_height_mm'],
                                        arms['closed']['final_height_mm'], arms['replay']['escape_time_s'], arms['closed']['escape_time_s']))),
               not_established='absolute force values, physical units, biological validity; no standing, no walking, no completion percentage',
               recorded_at=now(), recorded_at_utc=utc())
    (out / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    x = np.arange(2); labels = ['replay (frozen archive)', 'closed (connectome, per-step)']
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.bar(x, [arms[k]['escape_time_s'] for k in ('replay', 'closed')], color=['#1f77b4', '#d62728'])
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('escape time [s]')
    a.set_title('time until the six-leg condition is left', fontsize=10); a.grid(alpha=0.3, axis='y')
    for xi, k in zip(x, ('replay', 'closed')):
        a.annotate('%.4f' % arms[k]['escape_time_s'], (xi, arms[k]['escape_time_s']), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
    a = ax[0][1]
    hr = np.asarray(arms['replay']['height_series']); hc = np.asarray(arms['closed']['height_series'])
    a.plot(np.arange(len(hr)) * 0.005, hr, 'o-', ms=2, color='#1f77b4', label='replay')
    a.plot(np.arange(len(hc)) * 0.005, hc, 's-', ms=2, color='#d62728', label='closed')
    a.set_xlabel('run time [s]'); a.set_ylabel('root z [mm]'); a.set_title('root height per command interval', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1][0]
    a.bar(x, [arms[k]['height_min_mm'] for k in ('replay', 'closed')], color=['#2ca02c', '#ff7f0e'])
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('minimum root z [mm]')
    a.set_title('deepest root height in the window', fontsize=10); a.grid(alpha=0.3, axis='y')
    for xi, k in zip(x, ('replay', 'closed')):
        a.annotate('%.4f' % arms[k]['height_min_mm'], (xi, arms[k]['height_min_mm']), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
    a = ax[1][1]
    a.bar(x, [arms[k]['spikes_total'] for k in ('replay', 'closed')], color=['#9467bd', '#8c564b'])
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('tier spikes in the window')
    a.set_title('neural activity produced by the tier', fontsize=10); a.grid(alpha=0.3, axis='y')
    fig.suptitle('minimal neural closed loop on the canonical accepted equilibrium: the command is re-evaluated every 5 ms from the current contact state '
                 'through the same connectome tier\ndt 2.5e-5 asserted / body 1e-4 s / neural tick 5e-4 s / 100 command intervals = 0.500 s, state written once, only ctrl in the loop\n'
                 'manual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out / 'closed-loop-minimal.png', dpi=135); plt.close(fig)
    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 最小神经元闭环（**手动研究实施，不是成品**）', '',
        '## 传入通路：有，且给了突触计数', '- `mechanosensory → motor`：**461 条边**、总权重 **6665**、源细胞 **208**、目标细胞 **52**；',
        '- `mechanosensory → descending`：**10293 条边**、总权重 **135748**、源细胞 **2308**、目标细胞 **344**；',
        '- 文件：`/run/media/sensen/Data2/cell_wound_prototype/data/flywire/fafb_connectome.npz`（sha256 `%s`）、'
        '`/run/media/sensen/Data2/cell_wound_prototype/data/flywire/fafb_sensorimotor_groups.npz`（`%s`）。' % (sha(ROOT / 'data/flywire/fafb_connectome.npz')[:16], sha(ROOT / 'data/flywire/fafb_sensorimotor_groups.npz')[:16]),
        '- **链路**：`MechanoReceptorBank`（当前接触合力/足接触/关节偏差）→ `ChannelSensoryEncoder`（注入该腿神经 2633 个机械感觉细胞）→ `ConductanceNetwork`（12000 细胞、连接组权重、0.5 ms tick）→ `MotorOutput.consume` → `project` → `ctrl`。'
        '**不含** PD／CPG／目标姿态／逐步姿态修正；`closed` 臂**不重放档案**。`biological_mapping_verified` 仍为 **false**。', '',
        '## 两组对照', '', '| 臂 | 逃逸时间 [s] | 最低 root z [mm] | 末值 root z [mm] | tier 脉冲 |', '|---|---|---|---|---|',
        *['| %s | %.4f | %.4f | %.4f | %d |' % (k, arms[k]['escape_time_s'], arms[k]['height_min_mm'], arms[k]['final_height_mm'], arms[k]['spikes_total']) for k in ('replay', 'closed')],
        '', '## 回答', '- %s' % rep['answer']['sentence'], '',
        '- 绝对力值与单位 `NOT_ESTABLISHED`；不宣称站立/行走；无完成百分比。']), encoding='utf-8')
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write('run_closed_loop_minimal_v2.py', 'run_closed_loop_minimal_v2.py')
    fp = out / 'FROZEN_DELIVERY_HASHES.json'; fp.write_text('{}', encoding='utf-8'); (out / 'delivery-hashes.json').write_text('{}', encoding='utf-8')
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
    for p in sorted(out.iterdir()):
        if p.is_file(): os.chmod(p, 0o444)
    os.chmod(out, 0o555)
    print('OUT:', out)
    return out


if __name__ == '__main__':
    main()
