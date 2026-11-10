"""Render the assembled five-of-six model: one passing leg and the failing one, same simulation.

The frames are the actual MuJoCo state of the same deterministic evaluation that produced
ASSEMBLE_SOLO.json; the labels say exactly what this is and is not. LH is included because it is the
leg whose pass is marginal, and RH because it is the one leg that does not advance at all.
"""
from pathlib import Path
import json, os, sys
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco as mj
import imageio.v2 as imageio
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DEST = HERE
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

LEGS = ['LH', 'RH']
HEADER = ['assembled replacement ANN: 5 of 6 legs pass this single-leg step task',
          'DERIVED world (triad pads, releasable adhesion) / REPLACEMENT ANN trained per leg',
          'labelled per leg: LH passes with a low clearance shuffle, RH never advances',
          'this is NOT walking and NOT frozen-body acceptance; time slowed 25x (300 ticks = 0.3 s)']


def load_net():
    Z = np.load(HERE / 'MODEL_singleleg_demo_assembled.npz')
    net = T.Net(np.random.default_rng(11))
    for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2', 'wc0', 'bc0', 'wc', 'bc', 'logstd'):
        net.p[k] = Z['actor_' + k].astype(np.float64)
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
        a = np.tanh(net.forward(obs[None])[2][0])
        m = core.tick(a)
        if k % every == 0:
            renderer.update_scene(core.d, camera=cam)
            out.append((renderer.render(), k, float(m['padz'][leg_idx]),
                        float(m['loads'][leg_idx] / rc.WEIGHT),
                        float(m['loads'].sum() / rc.WEIGHT), float(m['com']),
                        float(m['nonfoot'])))
    renderer.close()
    return out


def label(img, k, padz, load, sup, com, nonfoot, leg, verdict):
    im = Image.fromarray(img).convert('RGB')
    W, H = im.size
    canvas = Image.new('RGB', (W + 440, H), (245, 245, 245))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle([4, 6, W - 4, 6 + 16 * len(HEADER) + 8], fill=(245, 245, 245))
    y = 12
    for line in HEADER:
        d.text((8, y), line, fill=(20, 20, 20)); y += 16
    y += 10
    d.text((W + 12, y), 'leg being stepped: %s' % leg, fill=(0, 0, 0)); y += 20
    d.text((W + 12, y), verdict, fill=(150, 0, 0) if 'does not' in verdict else (0, 100, 0)); y += 22
    y += 6
    d.text((W + 12, y), 't = %.3f s   tick %d' % (k * rc.PDT, k), fill=(0, 0, 0)); y += 22
    rows = [('pad height (native)', '%+.4f' % padz, padz >= T.THRESH['lift']),
            ('leg load / body weight', '%.3f' % load, False),
            ('support (all legs)', '%.3f' % sup, sup >= T.THRESH['support']),
            ('body height (COM)', '%.3f' % com, com >= T.THRESH['com']),
            ('non-foot contact (W)', '%.2f' % nonfoot, nonfoot <= 0.20 * rc.WEIGHT)]
    for name, val, ok in rows:
        d.text((W + 12, y), name, fill=(0, 0, 0))
        d.text((W + 310, y), val, fill=(0, 110, 0) if ok else (20, 20, 20))
        y += 18
    y += 10
    d.text((W + 12, y), 'acceptance: unload .10 / lift .015 /', fill=(0, 0, 0)); y += 15
    d.text((W + 12, y), 'advance .03 / place .10 and .004 /', fill=(0, 0, 0)); y += 15
    d.text((W + 12, y), 'support .90 / non-foot 4.889 W /', fill=(0, 0, 0)); y += 15
    d.text((W + 12, y), 'body height 1.00', fill=(0, 0, 0))
    return np.asarray(canvas)


def main():
    res = json.loads((HERE / 'ASSEMBLE_SOLO.json').read_text())
    net = load_net()
    written = []
    for leg in LEGS:
        idx = rc.LEG_ORDER.index(leg)
        v = res['per_leg'][leg]
        verdict = ('PASSES (advance %.3f, low clearance)' % v['best_advance']
                   if v['passed'] else
                   ('does not pass: airborne advance %.4f' % v['best_advance']))
        frames = frames_for_leg(net, idx, leg)
        path = DEST / ('assembled_%s.mp4' % leg)
        with imageio.get_writer(path, fps=30, macro_block_size=1) as w:
            for img, k, padz, load, sup, com, nf in frames:
                w.append_data(label(img, k, padz, load, sup, com, nf, leg, verdict))
        mid = frames[len(frames) // 2]
        Image.fromarray(label(mid[0], mid[1], mid[2], mid[3], mid[4], mid[5], mid[6], leg,
                              verdict)).save(DEST / ('assembled_%s_frame.png' % leg))
        written.append(dict(leg=leg, passed=bool(v['passed']), mp4=str(path),
                            frames=len(frames), ticks=T.T_EP, slow_motion='25x'))
        print(leg, 'frames', len(frames), verdict, flush=True)
    (HERE / 'VIDEO_MANIFEST_ASSEMBLED.json').write_text(json.dumps(
        dict(label='visual samples from the assembled model, same simulation as ASSEMBLE_SOLO.json',
             files=written), indent=2) + '\n')


if __name__ == '__main__':
    main()
