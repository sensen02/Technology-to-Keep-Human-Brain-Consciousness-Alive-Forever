"""Parent-only finite complete-body ALGORITHM TEACHER, no ANN/optimizer."""
import argparse
import random
import sys
import time
from common import *


def main():
    p = argparse.ArgumentParser(); p.add_argument('--condition', choices=('stop','left_pivot','right_pivot'), required=True); p.add_argument('--execute-parent-physical', action='store_true'); a = p.parse_args()
    if not a.execute_parent_physical: p.error('Parent physical flag required')
    cpu1(); sources = frozen(); verify(sources)
    preflight = read(ROOT/'PREFLIGHT.json')
    if not preflight['all_passed'] or preflight['physical_integration_steps'] != 0 or preflight['source_hashes'] != sources:
        raise ValueError('Same-source actual zero-integration preflight required')
    sw, physics, identity, static, Peak, ast_ids = physical_stack()
    import numpy as np
    import mujoco as mj
    import teacher
    seed = 20261113; random.seed(seed); np.random.seed(seed)
    out = ROOT/'trials'/a.condition; out.mkdir(parents=True, exist_ok=False)
    c = None; tr = None; recorder = Peak(); decisions = []; geometry = []; error = None; result = None; unsafe = 0; early_stop = None; wall = time.monotonic()
    diag_stream = (out/'teacher_diagnostics.jsonl').open('x')
    try:
        c = physics.Core(adhesion=False, recorder=recorder)
        mesh = c.plant.geom_type == mj.mjtGeom.mjGEOM_MESH
        if mesh.sum() != 71 or not np.all(c.plant.geom_rgba[mesh, 3] == 1.): raise ValueError('Full71opaque required')
        ident = identity.capture(c, dict(kind='ALGORITHM_TEACHER_STOP_PIVOT_FEASIBILITY', condition=a.condition, seed=seed, source_hashes=sources, selected_legacy_ast=ast_ids, noANN=True, no_optimizer=True), extra_paths=list(sources)+[ROOT/'PREFLIGHT.json'])
        save(out/'IDENTITY.json', ident)
        save(out/'RNG_INITIAL.json', dict(python=random.getstate(), numpy=np.random.get_state(), seed=seed))
        save(out/'INITIAL_STATE.json', dict(time_s=float(c.d.time), **snapshot(c)))
        np.savez_compressed(out/'runtime_geometry.npz', geom_type=c.plant.geom_type.copy(), geom_rgba=c.plant.geom_rgba.copy())
        tr = sw.Trace(full=True)
        # Exact selected legacy settle source, no Torch import from student_wave.
        sw.settle(c, tr)
        bodylength = teacher.body_length_from_compiled_mesh(c)
        save(out/'BODY_LENGTH.json', bodylength)
        warm = static.StaticTeacher(c, sw.teacher_config(3)); previous = np.zeros(42)
        candidate = None; boundary = None; cycles = PivotCycles(physics.W)
        for k in range(2880):
            physical_ms = 1000+k*5
            if abs(float(c.d.time)-physical_ms*.001) > 1e-8: raise ValueError('Actual 5ms clock drift')
            post = k >= 960
            if k == 960:
                before = snapshot(c)
                candidate = teacher.make_teacher(c, a.condition, order_indices=warm.reference.order_indices)
                assert_same(before, snapshot(c))
                boundary = dict(time=float(c.d.time), com=c.d.subtree_com[c.th].copy(), yaw=yaw(c), no_reset=True)
                save(out/'HANDOVER.json', dict(**boundary, actual_state_exact=True, before={key: hashlib.sha256(value.tobytes()).hexdigest() for key,value in before.items()}, teacher_switch='old walking warm -> new training-only candidate', no_neural_state_exists=True))
                for row in tr.rows[-10:]: cycles.update(row['time'], row['loads'], row['padz'])
            x = c.obs238(previous, target_speed=.075).copy()
            elapsed = (k-960)*.005 if post else k*.005
            control = candidate if post else warm
            action, diagnostic = control.action(c, elapsed)
            action = sw.bounded_action(action)
            source = SOURCE_CANDIDATE if post else sw.SOURCE['teacher']
            decisions.append(dict(time=float(c.d.time), physical_ms=physical_ms, candidate_phase=post, elapsed_teacher_s=elapsed, X238=x, Y42=action.copy(), control_source=source, requested_speed=0. if post else .075, requested_yaw_rate=teacher.CONDITIONS[a.condition] if post else 0.))
            diag_stream.write(json.dumps(jsonable(dict(decision=k, physical_ms=physical_ms, condition=a.condition, candidate_phase=post, diagnostic=diagnostic)), allow_nan=False)+'\n')
            for _ in range(5):
                measured = tr.tick(c, action, source)
                rot = c.d.xmat[c.th].reshape(3,3).copy(); head = c.plant.body('Head').id
                geometry.append(dict(time=float(c.d.time), thorax_yaw=yaw(c), thorax_rotation=rot, thorax_xyz=c.d.xpos[c.th].copy(), head_xyz=c.d.xpos[head].copy(), head_rotation=c.d.xmat[head].reshape(3,3).copy(), foot_world=measured['foot'] @ rot.T+c.d.xpos[c.th], control_source=source))
                if post: cycles.update(float(c.d.time), measured['loads'], measured['padz'])
                unsafe = unsafe+1 if measured['com'] < .8 or measured['nonfoot'] > .5*physics.W else 0
                if unsafe >= 100:
                    early_stop = dict(time_s=float(c.d.time), reason='100ms COM<.8 or nonfoot>.5W', during_candidate=post)
                    break
            previous = action.copy()
            if k % 240 == 0:
                diag_stream.flush(); print(json.dumps(dict(condition=a.condition, teacher_decision=k, actual_s=float(c.d.time), wall_s=time.monotonic()-wall, label='ALGORITHM_TEACHER_NOT_ANN')), flush=True)
            if early_stop is not None: break
        diag_stream.flush()
        native = recorder.close()
        if native['post_native_steps'] == 0:
            native['post_COM_min'] = None
        auto = tr.summary(c, SOURCE_CANDIDATE)
        candidate_rows = [r for r in tr.rows if r['control_source'] == SOURCE_CANDIDATE]
        candidate_geometry = [g for g in geometry if g['control_source'] == SOURCE_CANDIDATE]
        metrics = candidate_metrics(candidate_rows, candidate_geometry, boundary, bodylength['length_native'], physics.W, a.condition, cycles) if candidate_rows else None
        if metrics is not None:
            native_gates = dict(native_nonfoot=native['post_nonfoot_peak_native'] == 0., native_extra_force=native['extra_force_max'] == 0., native_COM=native['post_COM_min'] >= 1.2, full_window=native['post_native_steps'] == 768000 and native['actual_native_steps'] == 1232000 and early_stop is None)
            metrics['gates'].update(native_gates)
            metrics['task_pass_candidate'] = None if a.condition == 'stop' else bool(all(metrics['gates'].values()))
        identity.verify_runtime(ident, c); verify(sources)
        if 'torch' in sys.modules: raise ValueError('Teacher-only run imported Torch')
        result = dict(kind='FULL_BODY_ALGORITHM_TEACHER_PIVOT_FEASIBILITY_V1', condition=a.condition, control='ALGORITHM_TEACHER_ONLY_NOT_ANN', candidate_metrics=metrics, original_five=auto, original_forward_gate_retained=True, original_five_not_pivot_success=True, native=native, early_stop=early_stop, actual_ms=len(tr.rows), actual_native_steps=len(tr.rows)*80, full_duration=early_stop is None and len(tr.rows)==15400, all71opaque=True, body_length=bodylength, weight_native=physics.W, candidate_start=boundary, candidate_cycles_applicable=a.condition!='stop', teacher_decisions=len(decisions), no_neural_models=True, no_optimizer=True, source_hashes=sources, selected_legacy_ast=ast_ids, identity_sha256=ident['identity_sha256'], science_scope='Engineering teacher feasibility only; no neck deployment, neural pivot, native fly validity, oldwalking qualification inheritance or originalD1-D8 claim', wall_s=time.monotonic()-wall)
    except BaseException as exc:
        error = exc
    finally:
        diag_stream.close(); artifacts = {}
        if tr is not None and tr.rows: artifacts['trace'] = tr.save(out/'trace_1ms.npz')
        for filename, records, key in (('teacher_samples_5ms.npz', decisions, 'teacher_samples'), ('geometry_1ms.npz', geometry, 'geometry')):
            if records:
                path = out/filename
                np.savez_compressed(path, **{name: np.asarray([row[name] for row in records]) for name in records[0]})
                artifacts[key] = dict(path=str(path), sha256=sha(path), rows=len(records))
        artifacts['diagnostics'] = dict(path=str(out/'teacher_diagnostics.jsonl'), sha256=sha(out/'teacher_diagnostics.jsonl'), rows=len(decisions))
        if error is not None:
            partial = recorder.close() if recorder.count else dict(actual_native_steps=0)
            if partial.get('post_native_steps', 0) == 0 and 'post_COM_min' in partial:
                partial['post_COM_min'] = None
            save(out/'FAILURE.json', dict(error=repr(error), artifacts=artifacts, native=partial, physical_ms=0 if tr is None else len(tr.rows), no_retry_overwrite=True))
        elif result is not None:
            result.update(artifacts); save(out/'RESULT.json', result)
    if error is not None: raise error
    print(json.dumps(dict(condition=a.condition, result=str(out/'RESULT.json'), candidate_metrics=result['candidate_metrics'], original_five=result['original_five'].get('gates'), early_stop=early_stop)), flush=True)


if __name__ == '__main__': main()
