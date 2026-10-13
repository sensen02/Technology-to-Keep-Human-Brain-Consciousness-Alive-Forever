"""Independent mechanics audit; hand-built research prototype, not a controller.
Never modifies canonical assets or historical reports. Default outputs are exclusive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import mujoco
import numpy as np
import scipy
from scipy.optimize import lsq_linear
import engine.canonical_physics as cp
import run_fix_round39 as r39
import engine.memory_guard as mg

HERE = Path(__file__).resolve().parent
POOLS = HERE / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
HISTORY = HERE / 'outputs/neural_rebuild/neural_rebuild_15_zeroing_cost.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded_delta(J, residual, a):
    """Linearize r(a+delta)=r(a)+J delta, with -a<=delta<=1-a."""
    return lsq_linear(J, -residual, bounds=(-a, 1-a), tol=1e-10,
                      max_iter=300, method='bvls')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--seeds', type=float, nargs='+', default=[0., .5, 1.])
    parser.add_argument('--seconds', type=float, default=.05)
    parser.add_argument('--output-dir', type=Path, default=HERE/'outputs/neural_muscle_closure')
    parser.add_argument('--log', type=Path, default=HERE/'MECHANICS_CLOSURE_AUDIT.zh-CN.md')
    args = parser.parse_args()
    assert args.rounds > 0 and 0 < args.seconds <= .2
    assert args.seeds and all(0 <= s <= 1 for s in args.seeds)
    out = args.output_dir.resolve() / 'mechanics_closure_audit.json'
    log = args.log.resolve()
    manifest = out.parent / 'mechanics_closure_audit.sha256'
    for p in (out, log, manifest):
        if p.exists():
            raise SystemExit(f'Refusing to overwrite existing file: {p}')
    protected = [cp.CANONICAL_XML, HERE/'engine/canonical_physics.py',
                 HERE/'run_fix_round39.py', HERE/'run_neural_rebuild_15_zeroing_cost.py',
                 POOLS, HISTORY, r39.ROUND37_POSE]
    before = {str(p): digest(p) for p in protected}
    assert before[str(cp.CANONICAL_XML)] == cp.CANONICAL_XML_SHA256
    model = cp.load_model()
    data = mujoco.MjData(model)
    pools = json.loads(POOLS.read_text())['measured']
    resolved = [r['actuator'] for r in pools['resolved']]
    unresolved = [r['actuator'] for r in pools['unresolved']]
    names = [model.actuator(i).name for i in range(model.nu)]
    assert len(set(names)) == model.nu and all(names)
    assert len(set(resolved)) == len(resolved) and len(set(unresolved)) == len(unresolved)
    assert not set(resolved) & set(unresolved)
    muscles = [i for i in range(model.nu)
               if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE
               and model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE]
    assert set(resolved + unresolved) == {names[i] for i in muscles}
    assert len(muscles) == 90 and len(resolved) == 74 and len(unresolved) == 16
    actadr = np.asarray(model.actuator_actadr, dtype=int)
    assert all(model.actuator_actnum[i] == 1 and actadr[i] >= 0 for i in muscles)
    records = [{'name': n, 'id': i, 'activation_address': int(actadr[i]),
                'muscle_identity_confirmed': i in muscles,
                'pool_coverage': 'resolved' if n in resolved else 'unresolved',
                'gain_type': int(model.actuator_gaintype[i]),
                'dynamics_type': int(model.actuator_dyntype[i]),
                'control_limited': bool(model.actuator_ctrllimited[i]),
                'control_range': model.actuator_ctrlrange[i].tolist()}
               for i, n in enumerate(names)]
    pose = json.loads(r39.ROUND37_POSE.read_text())
    joint_names = list(pose['joint_offsets'])
    joint_ids = [model.joint(n).id for n in joint_names]
    assert len(joint_ids) == 42 and len(set(joint_ids)) == 42
    assert all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in joint_ids)
    leg = np.array([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
    root = model.joint('thorax_free').id
    assert model.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE
    rd = int(model.jnt_dofadr[root])
    r39.apply_round37_pose(model, data, pose)
    cp.seat_root(model, data, data.qpos.copy())
    gate = cp.stance_facts(model, data).as_dict()
    seated = data.qpos.copy()
    M = np.zeros((model.nv, model.nv))

    def initialise(a):
        # Clear warmstarts/time/applied forces for reproducible independent probes.
        mujoco.mj_resetData(model, data)
        data.qpos[:] = seated
        for i in muscles:
            data.act[actadr[i]] = a[i]
            data.ctrl[i] = a[i]
        mujoco.mj_forward(model, data)

    def evaluate(a):
        initialise(a)
        mujoco.mj_fullM(model, M, data.qM)
        return M @ data.qacc.copy(), data.qacc.copy()

    def metrics(a):
        u, acc = evaluate(a)
        return {'leg_torque_l2_model': float(np.linalg.norm(u[leg])),
                'leg_torque_linf_model': float(np.max(np.abs(u[leg]))),
                'leg_generalized_torque_model': dict(zip(joint_names, u[leg].tolist())),
                'root_generalized_force_model': u[rd:rd+3].tolist(),
                'root_generalized_torque_model': u[rd+3:rd+6].tolist(),
                'root_linear_acceleration_model': acc[rd:rd+3].tolist(),
                'root_angular_acceleration_rad_s2': acc[rd+3:rd+6].tolist(),
                'joint_acceleration_rad_s2': dict(zip(joint_names, acc[leg].tolist())),
                'joint_acceleration_linf_rad_s2': float(np.max(np.abs(acc[leg])))}

    def jacobian(a, free, eps=.002):
        # Actual signed denominator; at upper bound use a backward probe.
        u = evaluate(a)[0][leg]
        J = np.empty((len(leg), len(free)))
        for k, i in enumerate(free):
            b = a.copy()
            b[i] += eps if a[i] <= 1-eps else -eps
            J[:, k] = (evaluate(b)[0][leg]-u)/(b[i]-a[i])
        return u, J

    # Regression: a=.8, target=.2 requires NEGATIVE delta, not absolute .0.
    toy = bounded_delta(np.eye(1), np.array([.6]), np.array([.8]))
    assert abs((.8+toy.x[0])-.2) < 1e-9
    boundary = np.zeros(model.nu); boundary[muscles] = 1
    _, Jbound = jacobian(boundary, muscles)
    assert np.isfinite(Jbound).all()
    test_a = np.zeros(model.nu); test_a[muscles] = .5
    u0, J = jacobian(test_a, muscles)
    direction = np.random.default_rng(39015).uniform(-1, 1, len(muscles))
    da = 1e-4 * direction
    test_b = test_a.copy(); test_b[muscles] += da
    actual = evaluate(test_b)[0][leg]-u0
    local_error = float(np.linalg.norm(actual-J@da)/max(np.linalg.norm(actual), 1e-15))
    assert local_error < .1, f'Local derivative check failed: {local_error}'

    def integrate(a, fixed):
        initialise(a)
        h0 = float(data.xpos[model.body('Thorax').id, 2])
        trace = []
        escape = None
        max_leak = 0.
        count = int(np.ceil(args.seconds/model.opt.timestep))
        initial_warnings = [w.number for w in data.warning]
        for k in range(count):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)  # synchronise qpos-derived observations
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            assert abs(data.time-(k+1)*model.opt.timestep) < 1e-9
            h = float(data.xpos[model.body('Thorax').id, 2])
            if escape is None and h < .9*h0:
                escape = float(data.time)
            if len(fixed):
                max_leak = max(max_leak, float(np.max(np.abs(data.act[actadr[fixed]]))))
            trace.append({'time_s': float(data.time), 'thorax_height_model': h,
                          'joint_velocity_linf_rad_s': float(np.max(np.abs(data.qvel[leg]))),
                          'root_translation_model': data.qpos[:3].tolist()})
        return {'command': 'constant control, initial activation preloaded; no state reset during integration',
                'zero_control_caveat': 'Compiled ctrl lower bound is 0.0001: excluded controls request zero but can develop activation; not exact sustained ablation.',
                'requested_seconds': args.seconds, 'actual_seconds': float(data.time),
                'initial_thorax_height_model': h0,
                'min_thorax_height_model': min(t['thorax_height_model'] for t in trace),
                'escape_below_90pct_time_s': escape,
                'excluded_activation_max': max_leak,
                'warning_count_delta': [int(w.number-initial_warnings[i]) for i,w in enumerate(data.warning)],
                'final_gate': cp.stance_facts(model, data, raise_on_fail=False).as_dict(),
                'trace': trace[::max(1, count//50)] + [trace[-1]]}

    def solve(label, free):
        free = np.array(free, dtype=int)
        fixed = np.array(sorted(set(muscles)-set(free)), dtype=int)
        runs = []
        best = None
        for seed in args.seeds:
            a = np.zeros(model.nu); a[free] = seed
            history = [float(np.linalg.norm(evaluate(a)[0][leg]))]
            solver_status = []
            for _ in range(args.rounds):
                residual, J = jacobian(a, free)
                sol = bounded_delta(J, residual, a[free])
                solver_status.append({'status': int(sol.status), 'success': bool(sol.success),
                                      'optimality': float(sol.optimality)})
                accepted = False
                for scale in (1., .5, .25, .125, .0625, .03125):
                    candidate = a.copy()
                    candidate[free] = np.clip(a[free]+scale*sol.x, 0, 1)
                    score = float(np.linalg.norm(evaluate(candidate)[0][leg]))
                    if score < history[-1]-1e-12:
                        a = candidate; history.append(score); accepted = True
                        break
                if not accepted:
                    break
            score = float(np.linalg.norm(evaluate(a)[0][leg]))
            assert np.all(a >= 0) and np.all(a <= 1) and np.all(a[fixed] == 0)
            assert all(y <= x for x,y in zip(history, history[1:]))
            runs.append({'seed': seed, 'leg_torque_l2_history_model': history,
                         'least_squares_solver_status': solver_status})
            if best is None or score < best[0]:
                best = (score, a.copy())
        a = best[1]
        result = {'label': label, 'n_free_actuators': len(free),
                  'free_actuator_names': [names[i] for i in free],
                  'best_activation_by_name': dict(zip(names, a.tolist())),
                  'per_seed': runs, 'fixed_pose': metrics(a),
                  'short_free_integration': integrate(a, fixed)}
        print(label, result['fixed_pose']['leg_torque_l2_model'],
              result['short_free_integration']['escape_below_90pct_time_s'], flush=True)
        return result

    conditions = [solve('ALL90_BY_IDENTITY', muscles),
                  solve('COVERED74_ONLY', [names.index(n) for n in resolved]),
                  solve('UNCOVERED16_ONLY', [names.index(n) for n in unresolved])]
    after = {str(p): digest(p) for p in protected}
    assert before == after, 'Protected input changed during audit'
    ratio = conditions[1]['fixed_pose']['leg_torque_l2_model']/conditions[0]['fixed_pose']['leg_torque_l2_model']
    command = shlex.join([str(Path(sys.executable).absolute()), str(Path(__file__).resolve()), *sys.argv[1:]])
    units = {'length_model': 'L_model; canonical module declares cm (not biological calibration)',
             'mass_model': 'M_model; SI mapping not established here', 'time': 's',
             'force_model': 'M_model*L_model/s^2', 'torque_model': 'M_model*L_model^2/s^2',
             'linear_acceleration_model': 'L_model/s^2', 'joint_acceleration': 'rad/s^2',
             'torque_divided_by_weight': 'L_model, NOT dimensionless',
             'gate_mm_caveat': 'gate *_mm follows canonical 10 mm/model-length convention only',
             'mass_caveat': 'canonical docstring and mass conversion constant conflict; no SI mass/force/torque claim',
             'gravity_model': model.opt.gravity.tolist()}
    doc = {'schema_version': 'mechanics-closure-audit-1', 'status': 'PASS',
           'status_meaning': 'audit execution and internal consistency checks passed; NOT standing/closure acceptance',
           'question': 'Re-evaluate step15 muscle identity, bounded optimization and instantaneous/dynamic evidence independently.',
           'measured': {'actuator_identity': records, 'n_muscles': len(muscles),
                        'n_nonmuscle_actuators': model.nu-len(muscles),
                        'historically_wrongly_excluded': names[-2:],
                        'seated_gate': gate, 'conditions': conditions,
                        'covered_vs_all_observed_l2_ratio': ratio,
                        'self_checks': {'negative_increment_regression_passed': True,
                                        'upper_bound_backward_difference_passed': True,
                                        'directional_derivative_relative_error': local_error,
                                        'nonincreasing_accepted_objectives': True}},
           'method': {'objective': 'L2 of 42 named hinge components of M(q) qacc; not contact force or direct muscle torque',
                      'update': 'argmin_delta ||r(a)+J delta||2, -a<=delta<=1-a; a_new=a+alpha*delta, actual forward-score line search',
                      'derivative': 'finite difference 0.002 with actual signed denominator, backward at upper bound',
                      'seeds': args.seeds, 'rounds': args.rounds,
                      'root_note': 'Root force and torque are separate generalized inertial residual components; joint-only objective does not enforce root equilibrium.',
                      'root_coordinate_frame': 'MuJoCo free joint: translation world axes, angular DOFs local body axes',
                      'physics': {'timestep_s': float(model.opt.timestep), 'integrator': int(model.opt.integrator),
                                  'iterations': int(model.opt.iterations), 'tolerance': float(model.opt.tolerance),
                                  'cone': int(model.opt.cone), 'noslip_iterations': int(model.opt.noslip_iterations)},
                      'model_mutation': 'none; canonical model and control bounds retained'},
           'units': units,
           'provenance': {'command': command, 'code': str(Path(__file__).resolve()),
                          'code_sha256': digest(__file__), 'interpreter': sys.executable,
                          'versions': {'mujoco': mujoco.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__},
                          'input_sha256_before': before, 'input_sha256_after': after,
                          'protected_inputs_unchanged': True},
           'illustrative_vs_measured': 'manual research prototype; simulator measurements, not biological validation',
           'what_this_does_not_support': [
               'No global optimum or lower bound: only best sampled local L2 descent at one seated pose.',
               'A small covered/all ratio does not establish that unresolved muscles are not load-bearing or that no pose search is needed.',
               'Joint torque residual does not imply whole-body equilibrium; root residual and all accelerations must be considered separately.',
               'Short open-loop integration is not stable standing, a neural controller, or an autonomous closed loop.',
               'Unresolved muscles remain physically present including passive effects. Zero activation is not deletion.',
               'SI scaling and biological geometry are not independently calibrated; no dimensionless torque/BW claim.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2, allow_nan=False)
    rows = []
    for c in conditions:
        m = c['fixed_pose']; t = c['short_free_integration']
        rows.append(f"| {c['label']} | {c['n_free_actuators']} | {m['leg_torque_l2_model']:.8g} | {m['leg_torque_linf_model']:.8g} | {m['joint_acceleration_linf_rad_s2']:.8g} | {t['escape_below_90pct_time_s']} |")
    text = '# 独立力学闭合证据审计\n\n这是手工实施和测试的研究原型，不是成品、神经控制器或稳定站立验收。\n\n'
    text += '## 纠正内容\n\n- 编译模型90个执行器全部按名称及muscle gain/dynamics核对；没有adhesion执行器。历史末两项实际是右后腿胫节屈/伸肌，不能减2。\n- 正确有界增量：`-a <= delta <= 1-a`，更新`a+alpha*delta`并用实际forward残差接受步长；上界采用真实分母的后向差分。\n- 目标为42个具名铰链的广义惯性力矩L2，额外报告L∞，非全局最优。根部力、力矩、线/角加速度及逐关节加速度分别记录。\n- 力矩除体重有长度量纲，不是无量纲。只报告model力/力矩；canonical质量文档与换算常数存在矛盾，不宣称SI质量或真实果蝇尺度。\n- 初态激活可为0，但模型控制下限0.0001；短时自由积分保留原生激活动力学，报告排除肌激活泄漏，不伪称持续严格置零。\n\n'
    text += '## 小规模实测\n\n| 条件 | 自由肌数 | 腿力矩L2(model) | 腿力矩L∞(model) | 关节加速度L∞(rad/s²) | 高度跌破90%时间(s) |\n|---|---:|---:|---:|---:|---:|\n' + '\n'.join(rows)
    text += f'\n\n覆盖/全部实测L2比={ratio:.8g}。仅为当前求解器、初姿和接触设置结果，不能推导未覆盖肌不承重。方向导数相对误差={local_error:.6g}；负增量回归、上界差分、接受步单调性、输入hash不变均通过。\n'
    text += '\n## 运行命令\n\n```bash\n'+command+'\n```\n\n## 独立交付与校验\n\n'
    text += '\n'.join('- `'+str(p)+'`' for p in [Path(__file__).resolve(), out, log, manifest])
    text += '\n\n契约JSON SHA256: `'+digest(out)+'`\n\n受保护输入运行前后hash完全一致：\n\n'
    text += '\n'.join('- `'+p+'` — `'+h+'`' for p,h in before.items())
    text += '\n\n契约中的PASS只说明审计执行成功，不表示力学闭合；历史代码/日志/XML未修改。未使用pandas，不混用虚拟环境。脚本拒绝覆盖已有同名产物；复跑须指定新的绝对输出目录与日志路径。\n'
    with log.open('x') as fh:
        fh.write(text)
    with manifest.open('x') as fh:
        for p in [Path(__file__).resolve(), out, log, *protected]:
            fh.write(f'{digest(p)}  {p}\n')
    print('Wrote', out, log, manifest, sep='\n')


if __name__ == '__main__':
    with mg.guarded_run('mechanics_audit', **mg.BUDGETS['mechanics_audit']):
        main()
