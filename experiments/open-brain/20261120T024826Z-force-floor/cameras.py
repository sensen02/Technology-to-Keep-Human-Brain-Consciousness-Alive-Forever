"""Virtual external camera capture. Files encode lossless indexed RGB frames by XOR."""
import argparse
import json
from pathlib import Path
import numpy as np
from numba import njit
from PIL import Image
from scipy.spatial import ConvexHull

HERE=Path(__file__).resolve().parent
SRC=Path('/home/sensen/Desktop/engine修复验收/八机位果蝇观测实验室-小箱与关节标记')
MARK=np.array([[225,28,33],[239,111,18],[205,190,13],[15,184,127],[21,97,216],[195,23,208]],np.uint8)

@njit
def paint(image,depth,uv,faces,colors):
    height,width=image.shape
    for idx in range(len(faces)):
        a,b,c=uv[faces[idx,0]],uv[faces[idx,1]],uv[faces[idx,2]]
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-9:continue
        for y in range(max(0,int(np.floor(min(a[1],b[1],c[1])))),min(height,int(np.ceil(max(a[1],b[1],c[1]))))):
            for x in range(max(0,int(np.floor(min(a[0],b[0],c[0])))),min(width,int(np.ceil(max(a[0],b[0],c[0]))))):
                u=((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/den
                v=((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/den
                if min(u,v,1-u-v)<0:continue
                z=a[2]*u+b[2]*v+c[2]*(1-u-v)
                if z<depth[y,x]:
                    depth[y,x]=z;image[y,x]=colors[idx]

def surface(folder):
    mesh=np.load(folder/'render_mesh.npz')
    scene=np.load(folder/'optical_scene.npz')
    lookup={g:i for i,g in enumerate(scene['gids'])}
    angles=np.arange(72)*np.pi*(3-5**.5)
    z=1-(2*np.arange(72)+1)/72
    dirs=np.c_[np.sqrt(1-z*z)*np.cos(angles),np.sqrt(1-z*z)*np.sin(angles),z]
    parts=[];faces=[];colors=[];offset=0
    def add(v,g,color):
        nonlocal offset
        tri=ConvexHull(v).simplices
        parts.append((v,lookup[g]))
        faces.extend(tri+offset);colors.extend([color]*len(tri))
        offset+=len(v)
    for g,v in zip(mesh['gids'],np.split(mesh['vertices'],np.cumsum(mesh['vcounts'])[:-1])):
        vv=v[np.unique((v@dirs.T).argmax(axis=0))]
        add(vv,g,240)
    for i,g in enumerate(mesh['foot_ids']):
        add(.066*dirs,g,241+i)
    return parts,np.array(faces),np.uint8(colors),scene['transforms']

def main(folder):
    parts,faces,colors,poses=surface(folder)
    origin=np.load(HERE/'geometry.npz')['world_origin_m']
    coords=np.stack([np.vstack([v@tr[g,3:].reshape(3,3).T+tr[g,:3] for v,g in parts])/1000+origin for tr in poses])
    homogeneous=np.concatenate([coords,np.ones((*coords.shape[:2],1))],axis=-1)
    info=json.loads((SRC/'input.json').read_text())
    out=folder/'external';out.mkdir(exist_ok=True)
    for name,p in info['projection'].items():
        bg=Image.open(HERE/('room_original' if folder.name=='natural' else 'room_insert')/(name+'.png')).convert('RGB').quantize(colors=232)
        palette=np.zeros((256,3),np.uint8)
        raw=np.array(bg.getpalette(),np.uint8).reshape(-1,3);palette[:len(raw)]=raw
        palette[240]=[85,51,36];palette[241:247]=MARK
        ref=np.asarray(bg);previous=ref.copy()
        # Sparse RGB difference is lossless and avoids duplicating the static room in every frame.
        stack=np.empty((len(poses),*ref.shape),np.uint8)
        for i,pts in enumerate(homogeneous):
            uv=pts@np.asarray(p).T;uv[:,:2]/=uv[:,2:3]
            current=ref.copy()
            paint(current,np.full(ref.shape,np.inf),uv,faces,colors)
            stack[i]=current^previous
            previous=current
        np.savez_compressed(out/(name+'.npz'),background=ref,palette=palette,delta=stack)
    (out/'rig.json').write_text(json.dumps(dict(projection=info['projection'],colors=MARK.tolist(),dt_ms=2)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    main(p.parse_args().folder)
