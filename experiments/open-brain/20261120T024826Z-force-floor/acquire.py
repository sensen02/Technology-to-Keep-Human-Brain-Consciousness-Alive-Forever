"""Two telecentric oblique bottom views. Store digitized 13-pixel line ROIs per dot."""
from pathlib import Path
import sys
import numpy as np
from floor import Floor, HERE, SHAPE, CX, CY, X, Y, ORIGIN, DT, mj

def image(z,folder,seed):
    rng=np.random.default_rng(seed)
    folder.mkdir(exist_ok=True)
    offsets=np.random.default_rng(94).uniform(-.32,.32,(2,*SHAPE))
    pixels=np.arange(-6,7)
    def profiles(depth):
        # Opposing 45-degree views, 2 µm/pixel in their imaging planes.
        centers=offsets+np.array([-1,1])[:,None,None]*np.sin(np.pi/4)*depth/2
        intensity=150+3250*np.exp(-.5*((pixels-centers[...,None])/.95)**2)
        noise=rng.normal(0,np.sqrt(intensity)/2)
        return np.clip(np.rint(intensity+noise),0,4095).astype(np.uint16)
    base=np.mean([profiles(np.zeros(SHAPE)) for _ in range(12)],axis=0).round().astype(np.uint16)
    stack=np.stack([profiles(a) for a in z])
    np.savez_compressed(folder/'bottom.npz',blank=base,profiles=stack,
                        time_ms=2*np.arange(1,len(stack)+1),x=CX[0],y=CY[:,0])

def bench(case):
    plate=Floor('bench')
    series=[];truth=[]
    if case=='cal':
        schedule=[(10,[])]
        for j,i,f in [(4,5,1),(8,16,3),(13,27,6),(0,0,2),(9,8,4)]:
            schedule.extend([(8,[(CX[j,i],CY[j,i],f)]),(6,[])])
    else:
        a=(CX[7,17]+.07,CY[7,17]-.045)
        b=(a[0]+.50,a[1])
        schedule=[(12,[]),(12,[(*a,1.4)]),(15,[(*a,1.4),(*b,4.8)]),
                  (14,[(*a,1.4)]),(10,[]),(12,[(*b,3.2)]),(10,[])]
    for count,points in schedule:
        force=np.zeros(SHAPE)
        for x,y,value in points:
            force[np.searchsorted(Y,y)-1,np.searchsorted(X,x)-1]+=value
        for _ in range(count):
            for _ in range(20): plate.step(force)
            series.append(plate.depth().copy())
            truth.append(points)
    folder=HERE/case
    image(series,folder,101 if case=='cal' else 303)
    import json
    (folder/('known.json' if case=='cal' else 'private.json')).write_text(json.dumps(truth))

def hover():
    f=Floor('hover')
    src=np.load(HERE/'exterior_keyframes.npz')
    pos=(src['pos']-ORIGIN)*1000
    quat=src['quat'];sizes=src['scale']*1000
    z=[];touch=0
    for k in range(len(pos)):
        for s in range(1,21):
            prev=max(0,k-1);t=s/20
            f.d.mocap_pos[:]=(1-t)*pos[prev]+t*pos[k]
            q1=quat[prev];q2=np.where((q1*quat[k]).sum(1,keepdims=True)<0,-quat[k],quat[k])
            q=(1-t)*q1+t*q2
            f.d.mocap_quat[:]=q/np.linalg.norm(q,axis=1,keepdims=True)
            for i in range(18):
                geom=f.m.body(f'probe_{i}').geomadr[0]
                rr=(1-t)*sizes[prev,i]+t*sizes[k,i]
                f.m.geom_size[geom,:2]=rr[[0,2]]
                f.m.geom_rbound[geom]=np.hypot(rr[0],rr[2])
            f.step()
            touch+=f.d.ncon
        z.append(f.depth().copy())
    image(z,HERE/'hover',404)
    print('hover',touch,float(np.max(np.abs(z))),flush=True)

if __name__=='__main__':
    for case in ('cal','blind'): bench(case)
    hover()
    image(np.load(HERE/'insert/tile_motion.npz')['depth_um'],HERE/'insert',505)
