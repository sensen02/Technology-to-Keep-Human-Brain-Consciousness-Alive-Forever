"""Evaluate and render the trained whole-body walking policy. Replacement ANN, NOT connectome ability.

The policy is a plain MLP driven only by the observation; there is no gait controller, no phase input,
no leg prompt sequence and no teacher in the loop. The video is rendered from the same rollout that
produces the numbers, and the labels state what the result is and is not.
"""
from pathlib import Path
import json, os, sys
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco as mj
import imageio.v2 as imageio
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import walk_ppo as P
from walk_ppo import Core, Net, GATES, T_EP
from gait_teacher import summarise, W, LEGS, NL

TICKS = int(os.environ.get('EVAL_TICKS', str(T_EP)))


def load_net(path=None):
    Z = np.load(path or (HERE / 'WALK_MODEL_best.npz'))
    net = Net(np.random.default_rng(0))
    for k in net.p:
        if k in Z:
            net.p[k] = Z[k].astype(np.float64)
    return net


def rollout(net, ticks=TICKS, render=False, every=2):
    core = Core()
    renderer = None
    if render:
        renderer = mj.Renderer(core.plant, 640, 640)
        cam = mj.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 95., -14., 6.2
        cam.lookat[:] = np.array([0., 0., 1.35])
    core.reset()
    from walk_ppo import onehot
    rec, frames = [], []
    for k in range(ticks):
        x = core.obs()
        mean = net.forward(x[None])[2]
        m = core.tick(np.tanh(mean[0]))
        rec.append((float(core.d.time), np.array(core.d.subtree_com[core.th], float),
                    m['loads'].copy(), float(m['nonfoot']), np.array(m['speed'], float)))
        if render and k % every == 0:
            renderer.update_scene(core.d, camera=cam)
            frames.append((renderer.render(), k, m, float(core.d.subtree_com[core.th][2])))
    if renderer:
        renderer.close()
    return summarise(rec, ticks * 80), frames


def label(img, k, m, com, ev):
    im = Image.fromarray(img).convert('RGB')
    Wd, H = im.size
    canvas = Image.new('RGB', (Wd + 470, H), (245, 245, 245))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    head = ['replacement ANN, single whole-body network (42 joint rate outputs)',
            'DERIVED world (triad pads, releasable adhesion); no teacher or gait controller in the loop',
            'this run: ' + ('ALL FIVE FROZEN GATES PASS' if ev['all_gates'] else 'gates NOT all met') +
            ' - it is not walking acceptance until every gate holds',
            'time slowed 25x: %d policy ticks = %.3f s of simulated time' % (TICKS, TICKS * 0.001)]
    d.rectangle([4, 6, Wd - 4, 6 + 16 * len(head) + 8], fill=(245, 245, 245))
    y = 12
    for line in head:
        d.text((8, y), line, fill=(20, 20, 20)); y += 16
    y += 8
    d.text((Wd + 12, y), 't = %.3f s   tick %d' % (k * 0.001, k), fill=(0, 0, 0)); y += 22
    rows = [('support (all legs)', '%.3f' % (m['loads'].sum() / W), m['loads'].sum() / W >= GATES['foot_support_mean_min']),
            ('body height (COM)', '%.3f' % com, com >= GATES['com_height_min']),
            ('non-foot contact (W)', '%.2f' % m['nonfoot'], m['nonfoot'] <= GATES['nonfoot_force_max_W'])]
    for nm, val, ok in rows:
        d.text((Wd + 12, y), nm, fill=(0, 0, 0))
        d.text((Wd + 330, y), val, fill=(0, 110, 0) if ok else (170, 0, 0)); y += 18
    y += 10
    d.text((Wd + 12, y), 'run summary (' + str(TICKS) + ' ticks):', fill=(0, 0, 0)); y += 18
    for nm, val in [('forward (signed)', '%+.4f' % ev['forward_signed_native_per_s']),
                    ('support mean / min', '%.3f / %.3f' % (ev['foot_support_mean'], ev['foot_support_min'])),
                    ('com height min', '%.3f' % ev['com_height_min']),
                    ('non-foot max (W)', '%.2f' % ev['nonfoot_force_max_W']),
                    ('slip ratio max', '%.2f' % ev['slip_ratio_max'])]:
        d.text((Wd + 12, y), nm, fill=(0, 0, 0))
        d.text((Wd + 330, y), val, fill=(20, 20, 20)); y += 17
    y += 8
    for k2, v2 in ev['gates'].items():
        d.text((Wd + 12, y), '%-9s %s' % (k2, 'PASS' if v2 else 'fail'),
               fill=(0, 110, 0) if v2 else (170, 0, 0)); y += 16
    return np.asarray(canvas)


def main():
    net = load_net()
    ev, frames = rollout(net, render=True)
    ev['label'] = ('replacement ANN whole-body walking attempt; algorithm warm-up was used only to '
                   'produce imitation data; no teacher in the loop')
    ev['gates_source'] = 'banc-strong-actuator/20261109T230000Z/train_strong.py GATES and summarise()'
    (HERE / 'WALK_EVAL.json').write_text(json.dumps(ev, indent=2) + '\n')
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in ev.items() if k != 'per_leg_slip_ratio'}, indent=1))
    path = HERE / 'WALK_POLICY.mp4'
    with imageio.get_writer(path, fps=30, macro_block_size=1) as w:
        for img, k, m, com in frames:
            w.append_data(label(img, k, m, com, ev))
    mid = frames[len(frames) // 2]
    Image.fromarray(label(mid[0], mid[1], mid[2], mid[3], ev)).save(HERE / 'WALK_POLICY_frame.png')
    print('frames', len(frames))


if __name__ == '__main__':
    main()
