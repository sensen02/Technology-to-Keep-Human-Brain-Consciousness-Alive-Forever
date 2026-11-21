"""Observer. Reads optical recordings and external calibration only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label,find_objects
from scipy.optimize import linear_sum_assignment
from itertools import combinations

def laplacian(data):
    out=np.zeros_like(data)
    for axis in (-1,-2):
        diff=np.diff(data,axis=axis)
        start=[slice(None)]*data.ndim;end=start.copy()
        start[axis]=slice(0,-1);end[axis]=slice(1,None)
        out[tuple(start)]-=diff;out[tuple(end)]+=diff
    return out

def shift(profiles):
    # Shot/read noise about the pedestal is retained, including negative excursions.
    weights=profiles.astype(np.float32)-profiles[...,[0,-1]].mean(axis=-1,keepdims=True)
    return (weights@np.arange(-6,7,dtype=np.float32))/np.maximum(1,weights.sum(axis=-1))

def read_bottom(path):
    with np.load(path/'bottom.npz') as a:
        change=shift(a['profiles'])-shift(a['blank'])
        return change[:,1]-change[:,0],a['x'],a['y']

def calibrate(folder):
    signal,x,y=read_bottom(folder)
    known=json.loads((folder/'known.json').read_text())
    maps=np.zeros_like(signal); stable=[]
    for i,points in enumerate(known):
        for px,py,f in points:
            maps[i,np.argmin(abs(y-py)),np.argmin(abs(x-px))]+=f
        if i>=3 and known[i-3:i]==[points]*3: stable.append(i)
    gradients=laplacian(signal)
    features=np.stack([signal,gradients],axis=-1)
    # Include the loaded taxel and its nearest neighbors from all stable known holds.
    from scipy.ndimage import maximum_filter
    fit=maximum_filter(maps[stable],size=(1,3,3))>0
    coef=np.linalg.lstsq(features[stable][fit],maps[stable][fit],rcond=None)[0]
    (folder/'response.json').write_text(json.dumps({'gain':coef.tolist(),'min_patch_uN':.25,
        'pixel_model':'force = a * differential optical shift + b * spatial Laplacian'}))
    return coef

def decode(folder,config):
    signal,x,y=read_bottom(folder)
    a,b=config['gain'];force=a*signal+b*laplacian(signal)
    records=[]
    for t,frame in enumerate(force):
        for sign in (1,-1):
            groups,n=label(sign*frame>config['min_patch_uN'])
            for idx,sl in enumerate(find_objects(groups),1):
                hit=groups[sl]==idx
                weights=frame[sl][hit]
                iy,ix=np.where(hit);iy+=sl[0].start;ix+=sl[1].start
                records.append(dict(frame=t,time_ms=2*(1+t),x_mm=np.average(x[ix],weights=weights),
                                    y_mm=np.average(y[iy],weights=weights),fz_uN=weights.sum(),
                                    taxels=len(ix)))
    return records,force.sum(axis=(1,2))

def triangulate(ps,xy):
    good=np.isfinite(xy).all(1)
    ps,xy=ps[good],xy[good]
    if len(ps)<2:return np.full(3,np.nan)
    def fit(p,q):
        rows=np.stack([q[:,i,None]*p[:,2]-p[:,i] for i in range(2)],axis=1).reshape(-1,4)
        v=np.linalg.svd(rows/np.linalg.norm(rows[:,:3],axis=1)[:,None],full_matrices=False)[2][-1]
        return v[:3]/v[3]
    best=None
    for ab in combinations(range(len(ps)),2):
        idx=list(ab);v=fit(ps[idx],xy[idx])
        proj=ps@np.r_[v,1]
        err=np.linalg.norm(proj[:,:2]/proj[:,2:]-xy,axis=1)
        ok=err<1.8
        score=(ok.sum(),-err[ok].sum())
        if ok.sum()>=2 and (best is None or score>best[0]):best=(score,ok)
    return fit(ps[best[1]],xy[best[1]]) if best else np.full(3,np.nan)

def track(folder,origin):
    info=json.loads((folder/'rig.json').read_text())
    all_xy=[]
    for cam in info['projection']:
        with np.load(folder/(cam+'.npz')) as z:
            image=z['background'].copy();palette=z['palette'].astype(float)
            dist=np.linalg.norm(palette[:,None]-np.array(info['colors'])[None],axis=-1)
            lookup=np.where(dist.min(1)<15,dist.argmin(1)+1,0)
            trace=[]
            for delta in z['delta']:
                image^=delta
                colors=lookup[image]
                xy=np.full((6,2),np.nan)
                yy,xx=np.nonzero(colors)
                labels=colors[yy,xx]
                for k in range(6):
                    ys,xs=yy[labels==k+1],xx[labels==k+1]
                    if len(xs)>=5 and np.ptp(xs)<19 and np.ptp(ys)<19:
                        xy[k]=[xs.mean()+.5,ys.mean()+.5]
                trace.append(xy)
            all_xy.append(trace)
    ps=np.asarray(list(info['projection'].values()))
    xy=np.array(all_xy)
    return np.asarray([[triangulate(ps,xy[:,t,k]) for k in range(6)] for t in range(xy.shape[1])])*1000-origin*1000

def write_rows(path,fields,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('cal',type=Path)
    p.add_argument('--rig-origin',type=Path)
    args=p.parse_args()
    if args.folder==args.cal:calibrate(args.cal)
    config=json.loads((args.cal/'response.json').read_text())
    external=args.folder/'external'
    feet=None
    if external.is_dir():
        feet=track(external,np.array(json.loads(args.rig_origin.read_text())))
        np.save(args.folder/'observed_feet_mm.npy',feet)
    if not (args.folder/'bottom.npz').exists():return
    records,total=decode(args.folder,config)
    for r in records:
        r['legs']='';r['attribution']='unassigned'
        if feet is not None:
            distances=np.linalg.norm(feet[r['frame'],:,:2]-[r['x_mm'],r['y_mm']],axis=1)
            ids=np.flatnonzero(distances<.45)
            r['legs']='|'.join(['LF','LM','LH','RF','RM','RH'][k] for k in ids)
            r['attribution']={0:'unassigned',1:'single'}.get(len(ids),'shared')
    write_rows(args.folder/'patches.csv',['frame','time_ms','x_mm','y_mm','fz_uN','taxels','legs','attribution'],records)
    np.savetxt(args.folder/'total_uN.csv',total,delimiter=',',header='signed_total_uN',comments='')

if __name__=='__main__':main()
