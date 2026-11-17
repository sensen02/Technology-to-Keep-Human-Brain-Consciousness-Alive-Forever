"""Parent-only zero-Core fourNN/camera-queue/336/source-neck preflight.

No physical backend import, no render/model compile, no integration. Camera
fixtures test optical geometry and pending timing; actual terrain RGB is first
captured and saved by the full physical runner, never replaced by this fixture.
"""
import argparse
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-resource-released',action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('SOURCE_ONLY no model loads, arrays or physical backend')
        return
    import runtime
    runtime.verify()
    old = runtime.upstream()
    old.cpu1()
    checks = {}

    def check(name,flag):
        checks[name] = bool(flag)
        if not checks[name]:
            raise ValueError('Terrain neural preflight '+name)

    check('original_policy_loader_ROOT_globals_preserved',Path(old.load_policies.__globals__['ROOT']).resolve()==runtime.UPSTREAM_ROOT)
    check('no_physical_backend_before_policies',not any(name in sys.modules for name in ('physics','mujoco','student_wave','flygym')))
    sentinel = object()
    prior = {name:sys.modules.get(name,sentinel) for name in ('model','common','steering_behavior')}
    visual,odor,bridge,motor,vp,op,bp,mp = old.load_policies()
    import numpy as np
    import torch
    check('temporary_policy_aliases_restored',all(sys.modules.get(name,sentinel) is value for name,value in prior.items()))
    check('all_frozen_weights',all(not parameter.requires_grad for model in (visual,odor,bridge,motor) for parameter in model.parameters()))
    check('336_component_sizes',old.flat(motor.initial_state(1),visual.initial_state(1),odor.initial_state(1),torch.zeros((1,16),dtype=torch.float64)).shape == (1,336))
    check('latest_fusion400_odor250',bp['step']==400 and bp['format_version']==old.FORMAT and op['step']==250 and op['format_version']==old.ODOR_FORMAT)
    check('brain_only_AN_visual_odor',bridge.brain.input_size==3 and bridge.brain.hidden_size==16 and bridge.allowed_raw_indices.tolist()==list(range(42))+list(range(84,126))+list(range(168,195)))
    with np.load(old.NEUTRAL/'neural_decisions.npz',allow_pickle=False) as archive:
        x = archive['X238'][0].copy()
    loop = old.BrainNeckLoop(bridge,'V_empty')
    zero = torch.zeros(1,dtype=torch.float64)
    first = loop.decision(x,zero,zero,0)
    for _ in range(5):
        loop.advance()
    second = loop.decision(x,zero,zero,5)
    check('source_AN_capture0_release2_consume5',second['AN_capture_step']==0 and second['AN_release_step']==2 and np.array_equal(second['AN_held'],first['AN_encoded']))
    check('warm_brain_command_zero',not np.any(second['brain_hidden_after']) and not np.any(second['DN_encoded']) and not np.any(second['command']))
    camera = old.module('_terrain_nn_camera_contract',old.VIS/'camera.py')
    synthetic = np.zeros((2,32,32,3),dtype=np.uint8)
    queue = camera.PendingVision()
    queue.capture(0.,synthetic)
    queue_rows = []
    for tick in range(9):
        now = tick*.005
        row = queue.query(now)
        queue_rows.append(row)
        if tick == 4:
            queue.capture(.020,synthetic)
    check('camera_pending20ms',not any(row['valid'] for row in queue_rows[:4]) and all(row['valid'] for row in queue_rows[4:]) and queue_rows[4]['capture_time_s']==0. and queue_rows[8]['capture_time_s']==.020)
    frames = camera.eye_frames(np.zeros(3),np.eye(3),old.read(old.HEAD_AUDIT),old.read(old.VIS/'PREREG_CAMERA_PROBE.json')['eyes'])
    check('actual_source_eye_frames_geometry',len(frames)==2 and all(abs(np.linalg.norm(frame['forward'])-1.)<1e-10 and abs(np.linalg.norm(frame['up'])-1.)<1e-10 and abs(frame['forward']@frame['up'])<1e-10 for frame in frames))
    camera_text = (old.VIS/'camera.py').read_text()
    check('camera_all_geom_groups_visible',"option.geomgroup[:]=1" in camera_text)
    base_camera = old.VIS/'visual_body_v1/runtime.py'
    check('camera_head_resolved_by_name',"c.plant.body('Head').id" in base_camera.read_text())
    field = old.OdorField('V_empty')
    field.anchor(np.zeros(3),np.eye(3))
    sample = field.sample(np.zeros(3),np.eye(3),6600)
    check('Vempty_actual_concentration_zero',np.array_equal(sample['concentration'],np.zeros(2)) and np.array_equal(sample['actor_receptors'],np.zeros(2)))
    with torch.no_grad():
        held = visual.encode(torch.from_numpy(synthetic[None,None]))[0]
        vr,vl,vh = visual.step(held,visual.initial_state(1))
        om,oh = odor.mean_step(torch.zeros((1,2),dtype=torch.float64),odor.initial_state(1))
        dn,bh = bridge.brain_step(torch.zeros((1,1),dtype=torch.float64),torch.cat((vr.reshape(1,1),om[:,0:1].tanh()),-1),torch.zeros((1,16),dtype=torch.float64))
        command = old.project(bridge.decoder(dn)[:,0])
        means,mh = motor.step(torch.tensor(x[None],dtype=torch.float64),command,motor.initial_state(1))
    check('full_fourNN_forward_shapes',held.shape==(1,16) and vh.shape==(1,32) and oh.shape==(1,32) and dn.shape==(1,2) and bh.shape==(1,16) and means.shape==(1,42))
    check('natural_visual_and_odor_outputs_finite_no_forced_zero',bool(torch.isfinite(vr).all() and torch.isfinite(om).all() and torch.isfinite(means).all()))
    check('source_files_unchanged',loop.transport.verify_final()['source_files_unchanged'])
    check('no_Core_or_physical_backend_after_checks',not any(name in sys.modules for name in ('physics','mujoco','student_wave','flygym')))
    from adapter import tree
    for name in ('runner.py','audit.py'):
        transformed,changes = tree(name)
        compile(transformed,str(HERE/name)+'::preflight_only','exec')
        check('exact_single_namespace_AST_'+name,changes['route']==1 and changes['runtime_import']==1)
    runtime.verify()
    old.save(HERE/'PREFLIGHT.json',dict(kind='ZEROCORE_FOURNN_CAMERAQUEUE_TERRAIN_PREFLIGHT_V1',all_passed=True,
        checks=checks,state_total=336,Core_constructions=0,new_physics_steps=0,physical_model_compiles=0,
        actual_render_tested=False,camera_test_scope='source eye geometry + all-group visibility contract + synthetic pending fixture; actual terrain RGB mandatory in trial',
        synthetic_fixture_visual_raw=float(vr[0]),natural_zero_receptor_odor_raw=float(om[0,0].tanh()),
        old_load_policies_globals_unchanged=True,sourcefreeze_sha256=runtime.sha(HERE/'FROZEN.json')))
    print('ZEROCORE_4NN_CAMERAQUEUE_336_SOURCE_NECK_PREFLIGHT_PASS '+str(len(checks)))


if __name__ == '__main__':
    main()
