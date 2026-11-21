"""Train a visual reachability readout from paired low and elevated MuJoCo fruit."""
import argparse
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / '20261118T231627Z-open-fruit-loop'
sys.path.insert(0, str(ORIGINAL))
from sensors import BODY_DIR, vision_features, build_scene, place

STEPS = 55


def features(view, pitch):
    # Rotation changes elevation on the retinas, so include the head's gravity reference.
    mean = np.mean(view[5:7])
    return np.r_[view[1:], pitch, pitch * mean, pitch**2]


class Reachability:
    def __init__(self, config):
        self.mean = np.array(config['mean'])
        self.scale = np.array(config['scale'])
        self.weight = np.array(config['weight'])
        self.bias = config['bias']

    def predict(self, view, pitch, visible):
        if not visible:
            return 0.
        row = (features(view, pitch) - self.mean) / self.scale
        return float(expit(row @ self.weight + self.bias))


def collect(body, init, seed, count, held):
    place(body, init, (0., 0.), 0.)
    body.motor.reset(seed=seed)
    rng = np.random.default_rng(seed)
    geom = body.m.geom('fruit').id
    stock_size = body.m.geom_size[geom].copy()
    traces, imgs = [], []
    for frame in range(count):
        yaw_command = np.sin(frame*.55 + seed) * .27
        command = np.array([.53 - yaw_command, .53 + yaw_command])
        for _ in range(STEPS):
            control, pads = body.teacher(command)
            body.apply(control, pads)
        mj.mj_kinematics(body.m, body.d)
        thorax = body.d.xpos[body.thorax]
        axes = body.d.xmat[body.thorax].reshape(3, 3)
        heading = np.arctan2(axes[1, 0], axes[0, 0])
        head_basis = body.d.body('nmf/c_head').xmat.reshape(3, 3)
        pitch = float(np.arcsin(np.clip(head_basis[2,0], -1., 1.)))
        target_range = rng.uniform(2.2, 8.2)
        phi = heading + rng.uniform(-.65, .65)
        xy = thorax[:2] + target_range * np.array([np.cos(phi), np.sin(phi)])
        scale = rng.uniform(.85, 1.18) if not held else rng.uniform(.78, 1.31)
        body.m.geom_size[geom] = stock_size * scale
        for air in (0,1):
            low, high = ((.68, 1.00), (2.75, 4.2))[air]
            if held:
                low, high = ((.72, 1.12), (2.50, 4.48))[air]
            z = rng.uniform(low, high)
            body.m.geom_pos[geom] = (*xy, z)
            mj.mj_forward(body.m, body.d)
            rgb = body.sim.get_raw_vision(body.name)
            vf, seen = vision_features(rgb)
            if seen:
                traces.append(np.r_[air, target_range, z, pitch, vf])
                if len(imgs) < 8:
                    imgs.append(rgb[:,::3,::3].copy())
    return np.array(traces), np.array(imgs)


def model_fit(traces):
    ys, poses = traces[:,0], traces[:,3]
    xs = np.array([features(row,pitch) for row,pitch in zip(traces[:,4:],poses,strict=True)])
    base = np.mean(xs,axis=0)
    scale = np.maximum(np.std(xs,axis=0),.001)
    x = np.column_stack((np.ones(len(xs)), (xs-base)/scale))

    def loss(w):
        logits=x@w
        return np.mean(np.logaddexp(0.,logits)-ys*logits)+.004*np.sum(w[1:]**2), \
               x.T@(expit(logits)-ys)/len(x)+np.r_[0.,.008*w[1:]]

    f=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    assert f.success
    return {'mean':base.tolist(),'scale':scale.tolist(),'bias':float(f.x[0]),'weight':f.x[1:].tolist(),
            'loss':float(f.fun),'iterations':int(f.nit)}


def main(out):
    out.mkdir()
    body=build_scene(None)
    with np.load(BODY_DIR/'CANONICAL_INIT.npz',allow_pickle=False) as src:
        init={k:src[k] for k in ('qpos','qvel','ctrl_joint')}
    train, train_images=collect(body,init,401,240,False)
    holdout, held_images=collect(body,init,2627,150,True)
    body.sim.eye_renderer.close()
    config=model_fit(train)
    model=Reachability(config)
    for name, data in (('fit',train),('heldout',holdout)):
        p=np.array([model.predict(row[4:],row[3],True) for row in data])
        labels=data[:,0].astype(bool)
        run=dict(samples=len(data),air_recall=float(np.mean(p[labels]>=.8)),
                 ground_false_launch=float(np.mean(p[~labels]>=.8)),accuracy=float(np.mean((p>=.5)==labels)))
        config[name]=run
        print(name,run,flush=True)
    assert config['heldout']['samples'] >= 120
    assert config['heldout']['air_recall'] >= .70 and config['heldout']['ground_false_launch'] <= .06
    (out/'FIT.json').write_text(json.dumps(config,indent=2)+'\n')
    np.savez_compressed(out/'PAIRED.npz',train=train,heldout=holdout,train_images=train_images,held_images=held_images)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    main(p.parse_args().output)
