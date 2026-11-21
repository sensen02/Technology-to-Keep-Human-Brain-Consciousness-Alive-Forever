"""Train the height cue on the same FlyBody fixture used by both motor modes."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from dm_control.mujoco import wrapper

from walk_adapter import land
from perching import build_scene, TOP_CM, RADIUS_CM

HERE = Path(__file__).resolve().parent
LOW_CM = .014
FRUIT_HALF_THICKNESS = .0275
RESOLUTION = (256,144)


def scene(x, raised):
    env=build_scene(x)
    fruit=env.task._arena.mjcf_model.find('geom','fruit')
    top=TOP_CM if raised else LOW_CM
    fruit.pos=(x,0.,top-FRUIT_HALF_THICKNESS)
    fruit.size=(RADIUS_CM,FRUIT_HALF_THICKNESS)
    if not raised:
        # The low fruit is a pulp patch on the load-bearing floor.
        fruit.contype=0
        fruit.conaffinity=0
    return env


class Retina:
    def __init__(self, phy):
        self.phy=phy
        self.cam=phy.model.name2id('fruit_sensor','camera')
        self.head=phy.model.name2id('walker/head','body')
        self.opt=wrapper.MjvOption()
        self.opt.geomgroup[1:]=0
        self.opt.sitegroup[:]=0

    def image(self, yaw):
        f=np.array([np.cos(yaw),np.sin(yaw),0.])
        r=np.array([f[1],-f[0],0.])
        up=np.array([0.,0.,1.])
        tilt=np.deg2rad(2.)
        self.phy.data.cam_xpos[self.cam]=self.phy.data.xpos[self.head]+.06*f
        self.phy.data.cam_xmat[self.cam]=np.column_stack((r,np.sin(tilt)*f+np.cos(tilt)*up,
                                                        -np.cos(tilt)*f+np.sin(tilt)*up)).ravel()
        return self.phy.render(width=RESOLUTION[0],height=RESOLUTION[1],camera_id=self.cam,
                               scene_option=self.opt)

    def features(self, yaw, require_uncropped=True):
        image=self.image(yaw).astype(np.int16)
        red,green,blue=image.transpose(2,0,1)
        y,x=np.nonzero((red>=75)&(red-green>=32)&(red-blue>=32))
        if len(x)<12:
            return None,0.,0.
        width=max(3,int(np.ptp(x)+1))
        center=(float(x.mean())-128.)/128.
        # A clipped red patch can still guide walking, but its midpoint
        # cannot tell the head whether the target is above its reach.
        clipped = x.min() <= 1 or x.max() >= RESOLUTION[0]-2 or y.min() <= 1 or y.max() >= RESOLUTION[1]-2
        if require_uncropped and clipped:
            return None,center,width
        elevation=(72.-float(y.mean()))/width
        size=len(y)/width**2
        return np.array([elevation,size,1./width,center**2]),center,width


def train(out):
    env=scene(8.,True)
    env.reset()
    land.set_ground_pose(env)
    phy=env.physics
    eye=Retina(phy)
    g=phy.model.name2id('fruit','geom')
    root=phy.bind(env.task._root_joint)
    initial=root.qpos.copy()
    datasets=[]
    for seed, n in ((37,210),(293,120)):
        rng=np.random.default_rng(seed)
        rows=[]
        for i in range(n):
            distance=rng.uniform(1.8,10.4)
            bearing=rng.uniform(-.15,.15)
            y=rng.uniform(-.12,.12)
            radius=rng.uniform(.77,1.04)
            placement=initial.copy()
            placement[2]+=rng.uniform(-.018,.018)
            root.qpos=placement
            for label in (0,1):
                if label==1:
                    h=rng.uniform(.25,.38)
                else:
                    h=rng.uniform(.0005,.033)
                phy.model.geom_pos[g]=(distance,y,h-FRUIT_HALF_THICKNESS)
                phy.model.geom_size[g,0]=radius
                phy.forward()
                value,_,_=eye.features(bearing,require_uncropped=False)
                if value is not None:
                    rows.append(np.r_[label,value])
        datasets.append(np.asarray(rows))
    env.close()
    train,held=datasets
    mean=train[:,1:].mean(axis=0)
    scale=np.maximum(train[:,1:].std(axis=0),.001)

    def design(rows):
        return np.column_stack((np.ones(len(rows)),(rows[:,1:]-mean)/scale))

    X,y=design(train),train[:,0]
    def loss(w):
        z=X@w
        return np.mean(np.logaddexp(0.,z)-y*z)+.002*np.sum(w[1:]**2), \
               X.T@(expit(z)-y)/len(X)+np.r_[0.,.004*w[1:]]

    f=minimize(loss,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B')
    assert f.success
    p=expit(design(held)@f.x)
    high=held[:,0]>0.
    facts=dict(mean=mean.tolist(),scale=scale.tolist(),weight=f.x.tolist(),iterations=int(f.nit),
               train_rows=len(train),held_rows=len(held),
               elevated_recall=float(np.mean(p[high]>=.8)),low_false_flight=float(np.mean(p[~high]>=.8)))
    out.write_text(json.dumps(facts,indent=2)+'\n')
    print(facts,flush=True)
    assert facts['elevated_recall'] >= .8 and facts['low_false_flight'] <= .05


class HeightClassifier:
    def __init__(self, path):
        f=json.loads(path.read_text())
        self.mean=np.array(f['mean'])
        self.scale=np.array(f['scale'])
        self.coefficients=np.array(f['weight'])

    def predict(self, features):
        if features is None:
            return 0.
        return float(expit(np.r_[1.,(features-self.mean)/self.scale]@self.coefficients))


if __name__=='__main__':
    train(HERE/'COMMON_EYE_FIT.json')
