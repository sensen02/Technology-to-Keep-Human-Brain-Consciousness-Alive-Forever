"""Frozen-state full-body diagnostic. No simulation, action generation or training."""
from pathlib import Path
import hashlib
import json
import traceback
import numpy as np
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import imageio_ffmpeg
from physics import Core, W, LEGS
from identity import HERE, sha256, atomic_json, verify_runtime

OUT = HERE / 'VIDEO_DIAGNOSTIC'
SOURCE = HERE / 'EVAL_FF_TARGET'


def main():
    OUT.mkdir(exist_ok=True)
    report_path = SOURCE / 'RESULT.json'
    identity_path = SOURCE / 'IDENTITY.json'
    trace_path = SOURCE / 'trace_1ms.npz'
    report = json.loads(report_path.read_text())
    identity = json.loads(identity_path.read_text())
    frozen = {str(p): sha256(p) for p in (report_path, identity_path, trace_path,
              SOURCE / 'decisions_5ms.npz', Path(__file__).resolve())}
    assert frozen[str(trace_path)] == report['trace']['sha256']
    assert frozen[str(SOURCE / 'decisions_5ms.npz')] == report['decisions']['sha256']
    # Fail explicitly if any imported loader ever tries to integrate physics.
    def forbidden_step(*args, **kwargs):
        raise RuntimeError('Rendering may only call mj_forward; mj_step forbidden')
    mj.mj_step = forbidden_step
    mj.mj_step1 = forbidden_step
    mj.mj_step2 = forbidden_step
    c = Core(adhesion=False)
    verify_runtime(identity, c)
    model = c.plant
    assert (model.nq, model.nv, model.nu) == (49, 48, 42)
    mesh = model.geom_type == mj.mjtGeom.mjGEOM_MESH
    assert int(mesh.sum()) == 71
    assert np.all(model.geom_rgba[mesh, 3] == 1.)
    compiled_before = {name: hashlib.sha256(getattr(model, name).tobytes()).hexdigest()
                       for name in identity['compiled_arrays']}
    mesh_details = [dict(id=int(i), name=model.geom(int(i)).name,
                        alpha=float(model.geom_rgba[i, 3])) for i in np.flatnonzero(mesh)]
    with np.load(trace_path, allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    n = len(z['time'])
    assert n == 15400 and z['qpos'].shape == (n, 49) and z['qvel'].shape == (n, 48)
    assert all(np.isfinite(a).all() for a in z.values())
    assert np.allclose(z['time'], np.arange(1, n + 1) * .001, atol=1e-8, rtol=0)
    expected = np.r_[np.zeros(1000, int), np.full(4800, 2), np.full(9600, 3)]
    assert np.array_equal(z['control_source'], expected)
    assert np.max(np.abs(z['actual_requested_action'][:1000])) == 0.
    external = {key: float(np.max(np.abs(z[key]))) for key in
                ('external_body_force_max', 'external_generalized_force_max')}
    assert all(value == 0. for value in external.values())
    assert report['transition']['all_exact']
    boundary = float(report['transition']['time_s'])
    assert abs(boundary - 5.8) < 1e-8
    with np.load(SOURCE / 'decisions_5ms.npz', allow_pickle=False) as decisions:
        assert decisions['action'].shape == (2880, 42)
        assert decisions['X238'].shape == (2880, 238)
        assert np.array_equal(decisions['source'], np.r_[np.full(960, 2), np.full(1920, 3)])
    ann = report['autonomous']
    assert ann['gates'] == dict(support=True, nonfoot=True, com=True, forward=False, slip=False)
    assert not report['autonomous_cycles']['all_legs_completed']
    renderer = mj.Renderer(model, height=480, width=640)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance = 8.
    camera.azimuth = 140.
    camera.elevation = -25.
    camera.lookat[:] = z['com'][0]
    option = mj.MjvOption()
    option.flags[mj.mjtVisFlag.mjVIS_TENDON] = False
    # All geometry groups remain enabled; no mesh/body removal or model changes.
    option.geomgroup[:] = 1
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 12) if font_path.exists() else ImageFont.load_default()
    title_font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else font
    fps, stride = 30, 20
    indexes = list(range(0, n, stride))
    assert len(indexes) == 770
    video = OUT / 'FULL_BODY_ANN_TAKEOVER_NOT_WALKING.mp4'
    mapping = []
    chunks = []
    for start, stop, code, label in [(0, 1000, 0, 'ZERO RATE SETTLE'),
                                    (1000, 5800, 2, 'ALGORITHM TEACHER'),
                                    (5800, 15400, 3, 'PURE SUBSTITUTE ANN')]:
        chunks.append(dict(trace_start=start, trace_stop_exclusive=stop,
                           physical_interval_s=[start / 1000., stop / 1000.],
                           control_source=code, label=label, samples=stop-start,
                           external_body_force_max=float(z['external_body_force_max'][start:stop].max()),
                           external_generalized_force_max=float(z['external_generalized_force_max'][start:stop].max())))
    writer = imageio.get_writer(video, fps=fps, macro_block_size=1, codec='libx264',
                               quality=8, ffmpeg_log_level='error')
    try:
        for frame, i in enumerate(indexes):
            t = float(z['time'][i])
            code = int(z['control_source'][i])
            stage_start = {0: 0., 2: 1., 3: boundary}[code]
            phase = {0: 'ZERO RATE SETTLE', 2: 'ALGORITHM TEACHER', 3: 'PURE ANN TAKEOVER'}[code]
            c.d.qpos[:] = z['qpos'][i]
            c.d.qvel[:] = z['qvel'][i]
            c.d.time = t
            mj.mj_forward(model, c.d)
            assert np.array_equal(c.d.qpos, z['qpos'][i]) and np.array_equal(c.d.qvel, z['qvel'][i])
            assert np.allclose(c.d.subtree_com[c.th], z['com'][i], atol=1e-10, rtol=0)
            renderer.update_scene(c.d, camera=camera, scene_option=option)
            image = Image.new('RGB', (1000, 480), (246, 246, 246))
            image.paste(Image.fromarray(renderer.render()), (0, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 639, 59), fill=(246, 246, 246))
            draw.text((7, 3), 'FULL BODY ANN TAKEOVER DIAGNOSTIC / NOT WALKING', font=title_font, fill=(140, 20, 20))
            draw.text((7, 25), f'{phase} | actual t={t:.3f}s | stage elapsed={t-stage_start:.3f}s', font=font, fill=(15, 15, 15))
            draw.text((7, 43), '1.667x SLOW: actual state every 20ms, 30fps | boundary 5.800s', font=font, fill=(15, 15, 15))
            com = z['com'][i]
            lines = ['ACTUAL SAVED MEASUREMENTS',
                     f'COM xyz {com[0]:+.4f}, {com[1]:+.4f}, {com[2]:.4f}',
                     f'Foot load/W = {z["loads"][i].sum()/W:.5f}',
                     f'Nonfoot = {z["nonfoot"][i]:.5f} native',
                     'PAD bottom z (native) / actual load/W']
            for leg, height, load in zip(LEGS, z['padz'][i], z['loads'][i]/W):
                lines.append(f'{leg}: z {height:+.5f}  load/W {load:.4f}')
            lines.extend(['', 'WHOLE PURE ANN: 9.600s',
                          f'Signed forward {ann["forward_signed_native_per_s"]:+.6f} /s FAIL',
                          f'Min COM {ann["com_height_min"]:.5f} PASS',
                          f'Max nonfoot {ann["nonfoot_force_max_native"]:.5f} PASS',
                          f'Mean support/W {ann["foot_support_mean"]:.5f} PASS',
                          f'Max slip ratio {ann["slip_ratio_max"]:.4f} FAIL',
                          'All six complete cycles: FAIL',
                          'LF2 LH0 LM0 RF0 RH0 RM18',
                          'No external force; adhesion OFF',
                          'ANN output: 42 joint rates only',
                          'Substitute ANN; NOT original connectome',
                          'Standing; forward/slip fail. Goal NOT passed.'])
            for line, text in enumerate(lines):
                color = (150, 20, 20) if 'FAIL' in text or 'NOT passed' in text else (20, 20, 20)
                draw.text((647, 8 + line*19), text, font=font, fill=color)
            writer.append_data(np.asarray(image))
            mapping.append(dict(video_frame=frame, playback_time_s=frame/fps,
                                trace_index=i, actual_time_s=t, control_source=code,
                                phase=phase, stage_elapsed_s=t-stage_start))
    finally:
        writer.close()
        renderer.close()
    # Decode EVERY encoded frame sequentially, saving review images from VIDEO pixels.
    targets = [0., 1., 5.78, 5.8, 7.3, 15.38]
    selected = {min(range(len(mapping)), key=lambda j: abs(mapping[j]['actual_time_s']-target)): target for target in targets}
    keyframes = []
    decoded_images = []
    count = 0
    reader = imageio.get_reader(video)
    try:
        for frame, pixels in enumerate(reader):
            assert pixels.shape == (480, 1000, 3)
            count += 1
            if frame in selected:
                target = selected[frame]
                path = OUT / f'decoded_t{target:06.2f}_frame{frame:04d}.png'
                decoded = Image.fromarray(pixels)
                decoded.save(path)
                decoded_images.append(decoded)
                keyframes.append(dict(target_time_s=target, path=str(path), sha256=sha256(path), **mapping[frame]))
    finally:
        reader.close()
    assert count == len(indexes) == 770
    montage = Image.new('RGB', (2000, 1440), (246, 246, 246))
    for j, image in enumerate(decoded_images):
        montage.paste(image, ((j % 2)*1000, (j // 2)*480))
    montage_path = OUT / 'DECODED_MONTAGE.png'
    montage.save(montage_path)
    for path, digest in frozen.items():
        assert sha256(path) == digest, 'Frozen input changed: ' + path
    for name, digest in compiled_before.items():
        assert hashlib.sha256(getattr(model, name).tobytes()).hexdigest() == digest
    verify_runtime(identity, c)
    map_path = OUT / 'FRAME_INDEX_MAPPING.json'
    atomic_json(map_path, mapping)
    result = dict(video=str(video), video_sha256=sha256(video), frames=770, decoded_frames=count,
                  resolution=[1000, 480], body_view=[640, 480], measurement_panel=[360, 480],
                  fps=fps, physical_duration_s=15.4, playback_duration_s=770/fps,
                  physical_frame_dt_s=.020, slow_factor=1/(.020*fps),
                  first_source_time_s=float(z['time'][0]), last_source_time_s=float(z['time'][-1]),
                  last_video_source_time_s=mapping[-1]['actual_time_s'],
                  sampling_note='Post-step trace begins .001s; frames at .001+.020*k; targets use nearest decoded frame. No invented t=0 pose. Last frame represents final 20ms interval.',
                  source_trace=str(trace_path), source_trace_sha256=frozen[str(trace_path)],
                  source_report=str(report_path), source_report_sha256=frozen[str(report_path)],
                  source_identity=str(identity_path), source_identity_file_sha256=frozen[str(identity_path)],
                  source_identity_sha256=identity['identity_sha256'], identity_runtime_verified=True,
                  renderer_sha256=frozen[str(Path(__file__).resolve())], frozen_inputs=frozen,
                  frame_index_mapping=str(map_path), frame_index_mapping_sha256=sha256(map_path),
                  control_chunks=chunks, control_sources=report['control_sources'],
                  handover_time_s=boundary, transition_all_exact=True,
                  external_forces=external, no_external_force=True, adhesion_enabled=False,
                  qpos_dimension=49, qvel_dimension=48, all_qpos_qvel_saved=True,
                  pose_reconstruction='All49 qpos and48 qvel assigned exactly from same-run trace; mj_forward only. No physics stepping, controller or action generator.',
                  visible_mesh_geoms=71, all_71_mesh_alpha_one=True, mesh_alpha_checks=mesh_details,
                  no_mesh_or_body_deleted=True, all_geometry_groups_enabled=True,
                  compiled_model_arrays_unchanged=True, tendon_overlay_hidden_only_via_view_option=True,
                  camera=dict(type='fixed world free camera', lookat=z['com'][0].tolist(), distance=8.,
                              azimuth=140., elevation=-25., follow=False, visual_only=True, root_written=False),
                  ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe(), sequential_decode_verified=True,
                  keyframes=keyframes, montage=str(montage_path), montage_sha256=sha256(montage_path),
                  autonomous_metrics={k: ann[k] for k in ('foot_support_mean','com_height_min','nonfoot_force_max_native','forward_signed_native_per_s','slip_ratio_max','gates','all_gates')},
                  autonomous_complete_per_leg=report['autonomous_cycles']['complete_per_leg'],
                  interpretation='Continuous same-run 1s zero settle +4.8s algorithm teacher +9.6s substitute ANN. ANN standing support/COM/nonfoot engineering gates pass; forward/slip/all-six-cycles fail. NOT walking, NOT original connectome, NOT full original acceptance.',
                  visual_review='pending parent opening decoded keyframes and montage')
    atomic_json(OUT / 'VIDEO.json', result)
    print(json.dumps(dict(video=str(video), frames=770, decoded_frames=count,
                          slow_factor=result['slow_factor'], video_sha256=result['video_sha256'],
                          manifest=str(OUT/'VIDEO.json'), visual_review='pending parent')), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        OUT.mkdir(exist_ok=True)
        atomic_json(OUT / 'FAILURE.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
