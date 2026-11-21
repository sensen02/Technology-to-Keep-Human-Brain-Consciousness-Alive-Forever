"""Estimate rigid head and eye pose from exterior RGB, stationary camera and marker calibration."""
import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def detect(rgb, colors):
    pixel=rgb.astype(float)
    corr=np.linalg.norm(pixel[:,:,None,:]-colors[None,None,:,:],axis=-1)
    out=np.full((len(colors),2),np.nan)
    for i in range(len(colors)):
        labels,n=ndimage.label(corr[:,:,i]<65.)
        if not n:continue
        count=np.bincount(labels.ravel())[1:]
        best=np.argmax(count)+1
        y,x=np.where(labels==best)
        if 3<=len(x)<=125 and max(np.ptp(x),np.ptp(y))<=13:
            weight=(66-corr[y,x,i])**.7
            out[i]=[np.average(x+.5,weights=weight),np.average(y+.5,weights=weight)]
    return out

def project(p,xyz):
    h=p@np.r_[xyz,1.]
    return h[...,:2]/h[...,2,None]

def intersect(p,x):
    rr=np.stack([p[:,2]*x[:,j,None]-p[:,j] for j in range(2)],axis=1).reshape(-1,4)
    v=np.linalg.svd(rr/np.linalg.norm(rr[:,:3],axis=1,keepdims=True),full_matrices=False)[2][-1]
    return v[:3]/v[3]

def solve(points, cameras, obs):
    xyz=[];ids=[]
    for i in range(len(points)):
        keep=np.isfinite(obs[:,i]).all(1)
        p,x=cameras[keep],obs[keep,i]
        best=None
        for pair in combinations(range(len(x)),2):
            pos=intersect(p[list(pair)],x[list(pair)])
            delta=np.linalg.norm(project(p,pos)-x,axis=1)
            seen=delta<2.5
            score=(seen.sum(),-np.sum(delta[seen]))
            if seen.sum()>=2 and (best is None or score>best[0]):
                best=(score,seen)
        if best:
            xyz.append(intersect(p[best[1]],x[best[1]]));ids.append(i)
    if len(ids)<3:return None
    origin=np.mean(xyz,axis=0)
    centered=points[ids]-points[ids].mean(axis=0)
    if np.linalg.matrix_rank(centered,tol=.02)<2:return None
    u,_,vh=np.linalg.svd(centered.T@(xyz-origin))
    r=vh.T@np.diag([1,1,np.linalg.det(vh.T@u.T)])@u.T
    mk=points[ids]
    pairs=np.argwhere(np.isfinite(obs).all(axis=2))
    cam,tag=pairs.T
    image=obs[cam,tag]
    p=cameras[cam]
    def residual(params):
        vertices=Rotation.from_rotvec(params[:3]).apply(points[tag])+params[3:]
        uv=(p@np.c_[vertices,np.ones(len(vertices))][:,:,None]).squeeze(-1)
        return uv[:,:2]/uv[:,2:]-image
    x0=np.r_[Rotation.from_matrix(r).as_rotvec(),origin-r@mk.mean(axis=0)]
    fitted=least_squares(lambda x:residual(x).ravel(),x0,loss='soft_l1',f_scale=1.,max_nfev=80)
    delta=np.linalg.norm(residual(fitted.x),axis=1)
    valid=delta<2.5
    if len(set(cam[valid]))<2 or len(set(tag[valid]))<4 or valid.sum()<8:
        return None
    return dict(rot=Rotation.from_rotvec(fitted.x[:3]).as_matrix().tolist(),pos_mm=fitted.x[3:].tolist(),
                visible_cameras=len(set(cam[valid])),visible_tags=len(set(tag[valid])),
                median_error_px=float(np.median(delta[valid])),used_observations=int(valid.sum()))

def main(folder):
    cal=json.loads((folder/'observations.json').read_text())
    rig=json.loads((folder/'rig.json').read_text())
    cams=list(cal['projections'])
    p=np.array(list(cal['projections'].values()))
    p[:,:,:3]/=1000
    out={}
    points=np.array(rig['tags_mm'])
    colors=np.array(rig['colors'])
    for frame in cal['frames']:
        xy=np.full((len(cams),len(colors),2),np.nan)
        for k,c in enumerate(cams):
            path=folder/'external'/f'{frame}_{c}.png'
            if path.exists():
                with Image.open(path) as im:xy[k]=detect(np.asarray(im.convert('RGB')),colors)
        fit=solve(points,p,xy)
        if fit is not None:
            r=np.array(fit['rot']);t=np.array(fit['pos_mm'])
            fit['forward']=(r@rig['head_forward']).tolist()
            fit['up']=(r@rig['head_up']).tolist()
            fit['eyes']={e['name']:dict(pos_mm=(t+r@e['pos_mm']).tolist(),
                         rot=(r@e['rot']).tolist(),optical_axis=(-r@np.array(e['rot'])[:,2]).tolist()) for e in rig['eyes']}
        out[frame]=fit
    (folder/'estimated.json').write_text(json.dumps(out,indent=2)+'\n')
    print('resolved',sum(p is not None for p in out.values()),'/',len(out))

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('folder',type=Path)
    main(cli.parse_args().folder)
