"""Render the single-leg step for the legs that pass, from the same simulation and the same
network that produced REPLAY_CHECK.json. Nothing here is scripted motion: the frames are the
actual MuJoCo state, the labels state exactly what the result is and is not.
"""
from pathlib import Path
import json, os, sys
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco as mj
import imageio.v2 as imageio
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DEST = HERE   # write locally; verify_and_mirror.py is the only thing that copies to the Desktop
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

LEGS = ['LF', 'LM', 'RF']

HEADER = ['single-leg step task: 3 of 6 legs pass (LF, LM, RF)',
          'DERIVED world (triad pads, releasable adhesion) / REPLACEMENT ANN initialised from a',
          'declared timed demonstration - this is NOT walking and NOT frozen-body acceptance',
          'time is slowed 25x: 300 policy ticks = 0.300 s of simulated time']


def load_net(path):
    Z = np.load(path)
    net = T.Net(np.random.default_rng(11))
    for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2'):
        net.p[k] = Z['actor_' + k].astype(np.float64)
    net.p['wc'] = Z['critic_w'].astype(np.float64)
    net.p['bc'] = Z['critic_b'].astype(np.float64)
    net.p['logstd'] = Z['logstd'].astype(np.float64)
    return net


def frames_for_leg(net, leg_idx, leg_name, every=2):
    core = rc.Core()
    renderer = mj.Renderer(core.plant, 640, 640)
    cam = mj.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 95., -14., 6.2
    cam.lookat[:] = np.array([0., 0., 1.35])
    core.reset()
    prompt = T.onehot(leg_idx)
    out = []
    for k in range(T.T_EP):
        obs = core.obs(prompt)
        h0, h1, mean, _ = net.forward(obs[None])
        a = np.tanh(mean[0])
        m = core.tick(a)
        if k % every == 0:
            renderer.update_scene(core.d, camera=cam)
            img = renderer.render()
            out.append((img, k, float(m['padz'][leg_idx]), float(m['loads'][leg_idx] / rc.WEIGHT),
                        float(m['loads'].sum() / rc.WEIGHT), m['com'], float(m['nonfoot'])))
    renderer.close()
    return out


def label(img, k, padz, load, sup, com, nonfoot, leg):
    im = Image.fromarray(img).convert('RGB')
    W, H = im.size
    canvas = Image.new('RGB', (W + 430, H), (245, 245, 245))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle([4, 6, W - 4, 6 + 16 * len(HEADER) + 8], fill=(245, 245, 245))
    y = 12
    for line in HEADER:
        d.text((8, y), line, fill=(20, 20, 20)); y += 16
    y += 10
    d.text((W + 12, y), 'leg being stepped: %s' % leg, fill=(0, 0, 0)); y += 22
    d.text((W + 12, y), 't = %.3f s   tick %d' % (k * rc.PDT, k), fill=(0, 0, 0)); y += 22
    y += 8
    rows = [('pad height (native)', '%+.4f' % padz, padz >= T.THRESH['lift']),
            ('leg load / body weight', '%.3f' % load, False),
            ('support (all legs)', '%.3f' % sup, sup >= T.THRESH['support']),
            ('body height (COM)', '%.3f' % com, com >= T.THRESH['com']),
            ('non-foot contact (W)', '%.2f' % nonfoot, nonfoot <= 0.20 * rc.WEIGHT)]
    for name, val, ok in rows:
        d.text((W + 12, y), name, fill=(0, 0, 0))
        d.text((W + 300, y), val, fill=(0, 110, 0) if ok else (20, 20, 20))
        y += 18
    y += 12
    d.text((W + 12, y), 'acceptance: unload .10 / lift .015 /', fill=(0, 0, 0)); y += 15
    d.text((W + 12, y), 'advance .03 / place .10 and .004 /', fill=(0, 0, 0)); y += 15
    d.text((W + 12, y), 'support .90 / non-foot 4.889 W /', fill=(0, 0, 0)); y += 15
    d.text((W + 12, y), 'body height 1.00', fill=(0, 0, 0))
    return np.asarray(canvas)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    net = load_net(HERE / 'MODEL_singleleg_demo_timed.npz')
    written = []
    for leg in LEGS:
        idx = rc.LEG_ORDER.index(leg)
        frames = frames_for_leg(net, idx, leg)
        path = DEST / ('singleleg_%s.mp4' % leg)
        with imageio.get_writer(path, fps=30, macro_block_size=1) as w:
            for img, k, padz, load, sup, com, nf in frames:
                w.append_data(label(img, k, padz, load, sup, com, nf, leg))
        still = frames[len(frames) // 2]
        Image.fromarray(label(still[0], still[1], still[2], still[3], still[4],
                              still[5], still[6], leg)).save(
            DEST / ('singleleg_%s_frame.png' % leg))
        written.append(dict(leg=leg, mp4=str(path), frames=len(frames),
                            ticks=T.T_EP, slow_motion='25x'))
        print(leg, 'frames', len(frames), flush=True)
    (HERE / 'VIDEO_MANIFEST.json').write_text(json.dumps(
        dict(label='visual samples of the passing legs, same simulation as REPLAY_CHECK.json',
             files=written), indent=2) + '\n')
    print('done')


if __name__ == '__main__':
    main()
