"""Manual DIRECTCTRL NOT AUTONOMOUS numerical diagnostic; no controller imports."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
import numpy as np
import mujoco
from scipy.integrate import solve_ivp
from engine import memory_guard as mg

HERE = Path(__file__).resolve().parent
BASE = HERE/'outputs/model_branches/zero_floor'
OUT = HERE/'outputs/model_branches/activation_timestep'
DTS = (.0001, .00005, .000025)
DURATION = .005

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rhs(act, ctrl, prm):
    # Matches mju_muscleDynamics: clamp ONLY local timescale argument, not state.
    assert prm[2] == 0
    scale = .5 + 1.5 * min(1., max(0., act))
    delta = min(1., max(0., ctrl)) - act
    tau = prm[0]*scale if delta > 0 else prm[1]/scale
    return delta/max(1e-15, tau)

def reference(prm, ctrl=1.):
    sol = solve_ivp(lambda t, y: [rhs(y[0], ctrl, prm)], (0, DURATION), [0.],
                    method='DOP853', rtol=1e-12, atol=1e-14, max_step=1e-6, dense_output=True)
    assert sol.success
    return sol

def run_case(model, pose, dt, sol):
    model.opt.timestep = dt  # Only alternative model parameter; XML never rewritten.
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    data.qvel[:] = 0
    data.act[:] = 0  # Initialization only. No subsequent activation clamp/reset.
    data.ctrl[:] = 1  # Uniform vector step excites all 90, NOT behavioral evidence.
    mujoco.mj_forward(model, data)
    n = round(DURATION/dt)
    assert abs(n*dt-DURATION) < 1e-15
    records = {k: [] for k in ('qpos', 'qvel', 'qacc', 'act', 'act_dot', 'ctrl', 'actuator_force')}
    times, warnings = [], []
    def record():
        times.append(float(data.time))
        warnings.append(data.warning.number.copy())
        for k in records:
            records[k].append(getattr(data, k).copy())
    record()
    for _ in range(n):
        mujoco.mj_step(model, data)
        record()
    arrays = {k: np.asarray(v) for k, v in records.items()}
    arrays['time_s'] = np.asarray(times)
    arrays['warning_counts'] = np.asarray(warnings)
    expected = sol.sol(np.linspace(0, DURATION, n+1))[0]
    arrays['reference_activation'] = expected
    err = arrays['act'] - expected[:, None]
    result = dict(dt_s=dt, steps=n, requested_duration_s=DURATION, actual_final_time_s=float(data.time),
                  activation_max=float(arrays['act'].max()), activation_min=float(arrays['act'].min()),
                  activation_first_step=float(arrays['act'][1,0]), max_abs_error=float(abs(err).max()),
                  rms_error=float(np.sqrt(np.mean(err**2))), final_abs_error=float(abs(err[-1]).max()),
                  exceeds_ctrl_continuum_bound=bool(np.any(arrays['act'] > 1+1e-12)),
                  finite=all(bool(np.isfinite(a).all()) for a in arrays.values()),
                  warning_free=not bool(arrays['warning_counts'].any()),
                  all_90_activation_traces_identical=bool(np.all(arrays['act']==arrays['act'][:, :1])))
    assert abs(data.time-DURATION) < 1e-14
    return result, arrays

def main(report):
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    xml = BASE/'fruitfly_zero_floor.xml'
    manifest_path = BASE/'model_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = [xml, manifest_path, BASE/'zero_floor_traces.npz', BASE/'zero_floor_results.json',
               Path(manifest['canonical_xml']), Path(__file__), HERE/'test_activation_timestep_audit.py', Path(mg.__file__)]
    frozen = {str(p): sha(p) for p in sources}
    assert sha(xml) == manifest['branch_sha256']
    assert sha(manifest['canonical_xml']) == manifest['canonical_sha256']
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.nu == model.na == 90 and model.opt.timestep == DTS[0]
    assert int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER)
    assert np.all(model.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_MUSCLE))
    assert np.all(model.actuator_actlimited == 0)
    assert np.all(model.actuator_ctrllimited == 1)
    assert np.all(model.actuator_ctrlrange == [0., 1.])
    prm = model.actuator_dynprm[0, :3].copy()
    assert np.all(model.actuator_dynprm[:, :3] == prm)
    assert np.array_equal(prm, [.0001, .0004, 0.])
    with np.load(BASE/'zero_floor_traces.npz') as z:
        pose = z['seated_qpos'].copy()
    previous = json.loads((BASE/'zero_floor_results.json').read_text())
    sol = reference(prm)
    cases, trajectories = [], {}
    baseline_arrays = {name: getattr(model, name).copy() for name in dir(model)
                       if isinstance(getattr(model, name), np.ndarray)}
    for i, dt in enumerate(DTS):
        case, arrays = run_case(model, pose, dt, sol)
        case['alternative_in_memory_only'] = dt != DTS[0]
        case['declared_model_changes'] = {} if dt == DTS[0] else {'opt.timestep': {'from': DTS[0], 'to': dt}}
        case['model_identity_sha256'] = hashlib.sha256(json.dumps(dict(xml_sha256=sha(xml), timestep_s=dt), sort_keys=True).encode()).hexdigest()
        cases.append(case)
        trajectories.update({f'case{i}_{k}': v for k,v in arrays.items()})
        print(json.dumps(case), flush=True)
    assert all(np.array_equal(getattr(model,k),v) for k,v in baseline_arrays.items())
    assert all(sha(p)==h for p,h in frozen.items())
    assert all(c['finite'] and c['warning_free'] for c in cases)
    assert cases[0]['activation_max'] == 2.
    assert cases[0]['max_abs_error'] > cases[1]['max_abs_error'] > cases[2]['max_abs_error']
    trace = OUT/'activation_timestep_traces.npz'
    np.savez_compressed(trace, **trajectories)
    archived = {}
    for p in sources:
        if p.suffix not in ('.py', '.xml'): continue
        dest = OUT/'source_archive'/p.relative_to(HERE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        archived[str(dest)] = sha(dest)
        assert sha(dest)==frozen[str(p)]
    guard = report('completed')
    if not np.isfinite(guard['previous_soft_limit_mb']): guard['previous_soft_limit_mb']='unlimited'
    result = dict(status='PASS_NUMERICAL_DIAGNOSTIC_NATIVE_OVERSHOOT_CONFIRMED',
        label='DIRECTCTRL NOT AUTONOMOUS', implementation='manual experimental audit, not finished behavior',
        protocol=dict(duration_s=DURATION, cases=3, control_step=1., control_vector='all 90 uniformly one',
                      initialization='same frozen seated_qpos, zero velocity/activation; body freely integrates thereafter',
                      no_controller=True, no_activation_clamp=True, no_neural_network=True,
                      scalar_isolation='activation RHS independent of body state; compare one scalar ODE to all 90 native trajectories'),
        model=dict(mujoco_version=mujoco.__version__, integrator='Euler', dynprm=prm.tolist(),
                   initial_activation_timescale_s=float(prm[0]*.5), actuator_actlimited=model.actuator_actlimited.tolist(),
                   actuator_ctrllimited=model.actuator_ctrllimited.tolist(), actuator_ctrlrange=model.actuator_ctrlrange.tolist()),
        prior_zero_floor_normal_force_summary=previous['episodes']['normal']['force_summary'],
        reference=dict(method='scipy solve_ivp DOP853 scalar activation only', rtol=1e-12, atol=1e-14,
                       max_step_s=1e-6, continuum_bound=[0,1], sampled_max=float(sol.y.max())),
        cases=cases, provenance=dict(source_hashes=frozen, source_archive_hashes=archived, trace_sha256=sha(trace)),
        checks=dict(source_files_unchanged=True, compiled_model_arrays_unchanged=True),
        interpretation='Finite warning-free native dt produces activation 2 despite ctrl 1. Smaller dt reduces trajectory error; bound compliance alone is not convergence.',
        next_step='Separate explicit neural scheduler branch only: dt 25us candidate, 20 body steps per 0.5ms neural tick, 200 per 5ms command; preserve strict clock. Main runner unchanged; no mismatch-check bypass. Additional refinement needed before claiming converged behavior.',
        references=['https://mujoco.readthedocs.io/en/3.9.0/APIreference/APIfunctions.html#mju-muscledynamics',
                    'https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_util_misc.c'],
        clamp_policy='mju_muscleDynamics clamps ctrl locally to [0,1] and act only for timescale evaluation; excess excitation uses raw act. This model has no integrated activation limit.',
        memory_guard=guard, wall_s=time.monotonic()-start)
    (OUT/'activation_timestep_results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    # Numerical plot, not a standing/behavior visualization.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4))
    t=np.linspace(0,.001,1001)
    ax.plot(t*1000,sol.sol(t)[0],color='black',label='scalar ODE reference')
    for i, c in enumerate(cases):
        ax.plot(trajectories[f'case{i}_time_s']*1000,trajectories[f'case{i}_act'][:,0],'.-',label=f"dt={c['dt_s']*1e6:g} us")
    ax.axhline(1,color='gray',linestyle='--'); ax.set(xlim=(0,1),xlabel='Time (ms)',ylabel='Activation',title='DIRECTCTRL NOT AUTONOMOUS — no activation clamp')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/'activation_timestep.png',dpi=150); plt.close(fig)
    return result

if __name__ == '__main__':
    def deadline(*_): raise TimeoutError('60 second activation audit deadline')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(60)
    with mg.guarded_run('activation_timestep_audit', **mg.BUDGETS['mechanics_audit']) as report:
        main(report)
    signal.alarm(0)
