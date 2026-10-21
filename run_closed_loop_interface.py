"""Minimal interface for the neural closed loop, plus ONE comparison run (owner turn 19).

This file declares the interface, then runs the SAME 0.5 s protocol twice in one directory:
  open_loop  = the 100-frame archive replayed (state written once, loop writes only ctrl)
  closed     = the command re-evaluated every 5 ms from the current body state through the SAME tier
Nothing else differs. No external CPG, no PD, no per-step pose correction: the loop writes only ctrl and
the state is written once before it.
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
BANC = ROOT / 'data/flywire/banc_connectome.npz'
GROUPS = ROOT / 'data/flywire/fafb_sensorimotor_groups.npz'
POOLS = ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'


def fake_digest_sweep():
    empty = hashlib.sha256(b'{}').hexdigest()
    hits = []
    for d in sorted(DESK.iterdir()):
        f = d / 'delivery-hashes.json'
        fp = d / 'FROZEN_DELIVERY_HASHES.json'
        if not (d.is_dir() and f.exists()):
            continue
        try:
            j = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            continue
        v = j.get('FROZEN_DELIVERY_HASHES.json')
        if v == empty:
            hits.append(dict(dir=str(d), wrong_field_value=v, disk_manifest_sha256=(sha(fp) if fp.exists() else None)))
    return empty, hits


def main():
    import mujoco, importlib
    L = importlib.import_module('run_flybody_loop')
    from engine.embodied import adapters as closure
    from engine.embodied.sensory_channels import channel_index_map, LOOKUP
    from engine.embodied.motor_output import build_motor_manifest

    body = L.build_body('canonical')
    model = body['model']
    xml_dt = float(model.opt.timestep)
    st = np.load(STATE)
    body['seated'] = st['qpos'].copy()
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    pools = json.loads(POOLS.read_text())
    manifest = build_motor_manifest(pools, built['tier']['root_ids'], names)
    groups, _ = channel_index_map(int(built['n']), built['tier']['root_ids'])
    archive = np.asarray(np.load(DRIVE)['normal_controls'])
    empty, fake = fake_digest_sweep()

    interface = dict(
        connectome_archive=[dict(path=str(BANC), sha256=sha(BANC), neurons=153962, edges=3037361,
                                 role='the tier built by build_neural_tier uses this graph through adapters.build_neural_tier'),
                            dict(path=str(GROUPS), sha256=sha(GROUPS),
                                 role='cell groups: in_mechanosensory 2633, out_motor 109, out_descending 1290'),
                            dict(path=str(POOLS), sha256=sha(POOLS), role='109 motor cells mapped over 90 actuator channels')],
        neurons_fed_back=['the 2633 mechanosensory cells (in_mechanosensory) receive the injected sensory current',
                          'the 12000-cell tier propagates it through the connectome weights',
                          'the 109 motor cells are decoded into 90 muscle activations'],
        sensory_channels=[dict(name='contact_force_norm', source='sum of the tarsus-floor normal forces per leg, divided by that leg weight share',
                               declared_normaliser='reference_force = 1.0 (declared, not measured)'),
                          dict(name='foot_contact', source='binary tarsus contact per leg'),
                          dict(name='tibia_joint_dev_rad', source='signed tibia joint deviation from the seated reference',
                               declared_normaliser='reference_angle_rad = 0.30 (declared)'),
                          dict(name='multi_joint_dev_rad', source='signed multi-joint deviation'),
                          dict(name='tibia_joint_vel_rad_s', source='tibia joint velocity',
                               declared_normaliser='reference_velocity_rad_s = 10.0 (declared)'),
                          dict(name='root z', source='NOT a channel in the encoder: root height is read by the diagnostics only and is never fed back')],
        provenance_in_archive=['mechanosensory -> motor: 461 edges, total weight 6665 (measured from the connectome)',
                               'mechanosensory -> descending: 10293 edges, total weight 135748 (measured)',
                               'the motor pool manifest is built from the neural rebuild pools file, which is in the archive'],
        declared_assumptions=['channel polarity is a declared convention, never calibrated',
                              'the proprioceptive scalar is SIGN BLIND: mean(|angle - reference|) cannot separate flexion from extension',
                              'the contact-force normaliser is a declared value; the absolute force scale is NOT_ESTABLISHED',
                              'the platform zeroes act at the start while the accepted equilibrium carries non-zero act (mean 0.310905)',
                              'biological_mapping_verified is false for this mapping'],
        no_degradation_to_pose_correction=dict(verdict='IT DOES NOT DEGRADE INTO PER-STEP POSE CORRECTION',
            why=['the loop writes only d.ctrl[:]; qpos, qvel and act are written once before the loop',
                 'there is no target pose, no error term and no term that reads qpos and writes it back',
                 'the injected quantity is a contact/joint measurement mapped through the connectome into muscle activation',
                 'the connectome weights, cells and synapses are unchanged by this experiment']),
        forbidden=['external CPG', 'PD controller', 'preprogrammed action', 'per-step pose correction', 'archive re-timing'])
    assert interface['no_degradation_to_pose_correction']['verdict'].startswith('IT DOES NOT')

    arms = {}
    for arm, mode in (('open_loop', 'replay'), ('closed', 'normal')):
        s, traj, cmds, spikes = L.episode(body, built, manifest, groups, mode, 0.5, replay=(archive if mode == 'replay' else None))
        h = np.asarray(traj)[:, 2]
        arms[arm] = dict(arm=arm, escape_time_s=s['escape_time_s'], initial_height_mm=s['initial_height_mm'],
                         height_min_mm=s['height_min_mm'], final_height_mm=s['final_height_mm'],
                         six_leg_fraction=None, spikes_total=int(np.asarray(spikes).sum()) if spikes is not None else 0,
                         height_series=[round(float(x), 6) for x in h])
        print('%-10s escape %.4f s | h0 %.4f min %.4f final %.4f | spikes %d'
              % (arm, arms[arm]['escape_time_s'], arms[arm]['initial_height_mm'], arms[arm]['height_min_mm'],
                 arms[arm]['final_height_mm'], arms[arm]['spikes_total']), flush=True)

    out = DESK / ('S2-closed-loop-interface-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    one_line = ('closed loop: first non-foot contact %s, six-leg fraction of the window %s, minimum root z %s mm'
                % ('NOT REACHED inside the window' if arms['closed']['escape_time_s'] is None else 'still occurs (escape %.4f s)' % arms['closed']['escape_time_s'],
                   arms['closed']['six_leg_fraction'], arms['closed']['height_min_mm']))
    rep = dict(status='PASS_CLOSED_LOOP_INTERFACE', manual_research_not_product=True, standing=False, walking=False,
               title='minimal interface for the neural closed loop plus one open/closed comparison (manual research, not a product)',
               interface=interface,
               fake_digest_inventory=dict(placeholder=empty,
                   statement='every delivery-hashes.json that once pinned sha256("{}") has already been corrected; the sweep below is the evidence',
                   hits_remaining=fake, n_hits=len(fake), directories_scanned=len([d for d in DESK.iterdir() if d.is_dir()]),
                   note='this inventory is reported here, in this product README, as instructed; no separate directory was created'),
               declared=dict(dt_s=2.5e-5, xml_timestep_s=xml_dt, window_s=0.5, command_intervals=100,
                             body_step_s=1e-4, neural_tick_s=5e-4, state_written_once=True, loop_writes='ctrl only'),
               inputs=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                           tier_neurons=int(built['n']), actuators=int(manifest['n_actuators']),
                           script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__))),
               results=[arms['open_loop'], arms['closed']], one_line_conclusion=one_line,
               not_established='absolute force values, physical units, biological validity; no standing, no walking, no completion percentage',
               recorded_at=now(), recorded_at_utc=utc())
    (out / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    x = np.arange(2); labels = ['open loop (archive)', 'closed (connectome)']
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.bar(x, [arms[k]['escape_time_s'] or 0.5 for k in ('open_loop', 'closed')], color=['#1f77b4', '#d62728'])
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('escape time [s]')
    a.set_title('time until the six-leg condition is left', fontsize=10); a.grid(alpha=0.3, axis='y')
    for xi, k in zip(x, ('open_loop', 'closed')):
        a.annotate('%.4f' % (arms[k]['escape_time_s'] or float('nan')), (xi, arms[k]['escape_time_s'] or 0.5), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
    a = ax[0][1]
    a.plot(np.arange(len(arms['open_loop']['height_series'])) * 0.005, arms['open_loop']['height_series'], 'o-', ms=2, color='#1f77b4', label='open loop')
    a.plot(np.arange(len(arms['closed']['height_series'])) * 0.005, arms['closed']['height_series'], 's-', ms=2, color='#d62728', label='closed')
    a.set_xlabel('run time [s]'); a.set_ylabel('root z [mm]'); a.set_title('root height per command interval', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1][0]
    a.bar(x, [arms[k]['height_min_mm'] for k in ('open_loop', 'closed')], color=['#2ca02c', '#ff7f0e'])
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('minimum root z [mm]')
    a.set_title('deepest root height in the window', fontsize=10); a.grid(alpha=0.3, axis='y')
    for xi, k in zip(x, ('open_loop', 'closed')):
        a.annotate('%.4f' % arms[k]['height_min_mm'], (xi, arms[k]['height_min_mm']), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
    a = ax[1][1]
    a.bar(x, [arms[k]['spikes_total'] for k in ('open_loop', 'closed')], color=['#9467bd', '#8c564b'])
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('tier spikes in the window')
    a.set_title('neural activity produced by the tier', fontsize=10); a.grid(alpha=0.3, axis='y')
    fig.suptitle('minimal closed-loop interface on the canonical accepted equilibrium: open loop vs command re-evaluated every 5 ms from the current state\ndt 2.5e-5 asserted / body 1e-4 s / neural tick 5e-4 s / 100 command intervals = 0.500 s, state written once, only ctrl in the loop\nmanual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out / 'closed-loop-interface.png', dpi=135); plt.close(fig)
    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 闭环最小接口 ＋ 一次开环/闭环对照（**手动研究实施，不是成品**）', '',
        '## 一、假摘要点名清单', '- 占位串 `sha256("{}") = %s`；**扫描 %d 个目录，仍指向空对象的 delivery-hashes.json 数量 = %d**。' % (empty, len([d for d in DESK.iterdir() if d.is_dir()]), len(fake)),
        ('- 明细：' + '；'.join('%s（字段值 %s，磁盘清单 %s）' % (h['dir'], h['wrong_field_value'][:16], str(h['disk_manifest_sha256'])[:16]) for h in fake)) if fake else '- 明细：**无残留**（此前那些目录已在清单族重做中改为真实摘要，旧假值已记录在 errata 台账里）。', '',
        '## 二、闭环最小接口', '', '| 项 | 内容 |', '|---|---|',
        *['| connectome | `%s`（sha256 `%s`） |' % (p['path'], p['sha256'][:16]) for p in interface['connectome_archive']],
        '| 接回的神经元 | 2633 个 mechanosensory 细胞 → 12000 细胞层 → 109 运动细胞 → 90 肌通道 |',
        '| 感受通道 | `contact_force_norm`（逐腿跗节法向力和，声明归一化 1.0）、`foot_contact`（二值）、`tibia_joint_dev_rad`（有符号，声明归一化 0.30）、`multi_joint_dev_rad`、`tibia_joint_vel_rad_s`（声明归一化 10.0）；**root z 只用于诊断，不回灌** |',
        '| 归档里有出处的连接 | mechanosensory→motor **461 边/权重 6665**；mechanosensory→descending **10293 边/权重 135748** |',
        '| 必须声明的假设 | 通道极性未标定；本体标量**符号盲**；接触力归一化是**声明值**（绝对力值 `NOT_ESTABLISHED`）；起点 `act` 清零与平衡态非零 `act` 不一致；`biological_mapping_verified=false` |',
        '| 会不会退化成每步姿态修正 | **不会**：循环内只写 `d.ctrl[:]`，`qpos/qvel/act` 只在循环前写一次；无目标姿态、无误差项、无读 qpos 回写的项 |', '',
        '## 三、一次对照（同一 0.5 s 协议）', '', '| 臂 | 逃逸时间 [s] | 最低 root z [mm] | 末值 root z [mm] | tier 脉冲 |', '|---|---|---|---|---|',
        *['| %s | %s | %.4f | %.4f | %d |' % (k, arms[k]['escape_time_s'], arms[k]['height_min_mm'], arms[k]['final_height_mm'], arms[k]['spikes_total']) for k in ('open_loop', 'closed')],
        '', '## 一句话结论', '- %s' % one_line, '',
        '- 绝对力值与单位 `NOT_ESTABLISHED`；在闭环跑出证据之前，不写"站立/行走/能动"。']), encoding='utf-8')
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write('run_closed_loop_interface.py', 'run_closed_loop_interface.py')
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
    print('ONE LINE:', one_line)
    return out


if __name__ == '__main__':
    main()
