"""Diagnose the detail camera's pose log against the frames actually rendered."""
import sys, importlib.util
sys.path.insert(0, '.')
import numpy as np
import mujoco
from engine.embodied import BodyBackend, BodyConfig

spec = importlib.util.spec_from_file_location("rp", "run_recording_pipeline.py")
rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)

cfg = BodyConfig(seed=0, add_world_camera=True, world_camera_name='worldcam',
                 world_camera_pos_mm=rp.WORLD_CAM_POS_MM,
                 world_camera_fovy=rp.WORLD_CAM_FOVY_DEG,
                 world_camera_xyaxes=rp.WORLD_CAM_XYAXES,
                 extra_cameras=({"name": rp.DETAIL_CAM_NAME,
                                 "pos": [0., 0., rp.DETAIL_CAM_HEIGHT_MM],
                                 "xyaxes": list(rp.DETAIL_CAM_XYAXES),
                                 "fovy": rp.DETAIL_CAM_FOVY_DEG},))
be = BodyBackend(cfg, gl_backend='egl').attach_cpg_baseline()
cid = mujoco.mj_name2id(be.model, mujoco.mjtObj.mjOBJ_CAMERA, rp.DETAIL_CAM_NAME)
print('detail cam id', cid, 'order', [str(be.model.camera(i).name) for i in range(be.model.ncam)])
r = be.sim.set_renderer([rp.WORLD_CAM_NAME, rp.DETAIL_CAM_NAME], camera_res=(240, 320),
                        playback_speed=1.0, output_fps=rp.RENDERING_OPEN_FPS,
                        buffer_frames=True)
dt = cfg.timestep_s; fps = 60.0
fstep = max(1, int(round(1.0 / (fps * dt))))
logged = []; fly_at_log = []; times = []; cam_after = []
pan = None
for k in range(4000):
    be.step()
    if k % fstep == 0:
        th = np.asarray(be.data.xpos[1], dtype=float).copy()
        vel = np.asarray(be.data.qvel[0:3], dtype=float).copy()
        des = th + rp.DETAIL_PAN_LOOKAHEAD_S * vel
        pan = des if pan is None else pan + (1 - np.exp(-fstep * dt / rp.PAN_TAU_S)) * (des - pan)
        be.model.cam_pos[cid] = [float(pan[0]), float(pan[1]), rp.DETAIL_CAM_HEIGHT_MM]
        if be.sim.render_as_needed():
            logged.append(float(be.model.cam_pos[cid][0]))
            fly_at_log.append(float(th[0]))
            times.append(float(be.data.time))
            cam_after.append(float(be.model.cam_pos[cid][0]))
logged = np.array(logged); fly = np.array(fly_at_log); times = np.array(times)
print('n frames', len(logged))
print('logged cam x   %.3f .. %.3f' % (logged.min(), logged.max()))
print('fly x at log   %.3f .. %.3f' % (fly.min(), fly.max()))
print('gap fly-cam    %+.3f .. %+.3f  median %+.3f' % (
    (fly - logged).min(), (fly - logged).max(), np.median(fly - logged)))
bad = np.nonzero(np.abs(fly - logged) > 6.0)[0]
print('frames with |gap| > 6 mm:', len(bad), bad[:10])
print('first 8 (t, fly, cam):')
for i in range(8):
    print('   %.3f  %.3f  %.3f' % (times[i], fly[i], logged[i]))
be.close()
