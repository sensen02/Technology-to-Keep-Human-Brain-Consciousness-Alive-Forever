"""Novel-view projection of Brush 3D Gaussians. No geometry or reference RGB is loaded."""
import argparse,json
from pathlib import Path
from math import radians,tan
import numpy as np
from numba import njit
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.special import expit

class GaussianField:
    def __init__(self,path):
        with path.open('rb') as f:
            assert f.readline().strip()==b'ply'
            assert f.readline().strip()==b'format binary_little_endian 1.0'
            columns=[]
            while True:
                row=f.readline().decode().split()
                if row[:2]==['element','vertex']:n=int(row[2])
                elif row[:2]==['property','float']:columns.append(row[2])
                elif row==['end_header']:break
                elif not row:raise ValueError('Truncated PLY header')
            v=np.fromfile(f,dtype=np.dtype([(c,'<f4') for c in columns]),count=n)
        assert len(v)==n
        def vec(keys):return np.stack([v[k] for k in keys],axis=1).astype(float)
        self.center=vec('xyz')
        self.opacity=expit(v['opacity'])
        q=vec([f'rot_{k}' for k in [1,2,3,0]])
        r=Rotation.from_quat(q).as_matrix()
        scale=np.exp(vec([f'scale_{k}' for k in range(3)]))
        cov_factor=r*scale[:,None,:]
        self.cov=cov_factor@cov_factor.transpose(0,2,1)
        self.dc=vec([f'f_dc_{k}' for k in range(3)])*.2820947917738781+.5
        coeff=[k for k in columns if k.startswith('f_rest_')]
        if len(coeff) not in (0,9):
            raise ValueError('This renderer supports SH degree 0 or 1.')
        self.sh=vec([f'f_rest_{k}' for k in range(9)]).reshape(n,3,3) if coeff else None
        assert np.isfinite(self.center).all() and np.isfinite(self.cov).all()
    def project(self,cam,near=.01,correct_near=False):
        c2w=np.array(cam['transform_matrix'])
        world_to_cv=np.diag([1,-1,-1])@c2w[:3,:3].T
        d=self.center-c2w[:3,3]
        local=d@world_to_cv.T
        active=(local[:,2]>near)&(self.opacity>=1/255)
        ids=np.flatnonzero(active)
        xyz=local[active];z=xyz[:,2]
        hf=np.array([cam['fl_x'],cam['fl_y']])
        center=np.array([cam['cx'],cam['cy']])
        size=np.array([cam['w'],cam['h']])
        uv=xyz[:,:2]/z[:,None]
        pix=uv*hf+center
        uv=np.clip(uv,(-.15*size-center)/hf,(1.15*size-center)/hf)
        j=np.zeros((len(z),2,3))
        j[:,0,0]=hf[0]/z;j[:,1,1]=hf[1]/z
        j[:,:,2]=-uv*hf/z[:,None]
        jw=j@world_to_cv
        v2=jw@self.cov[active]@jw.transpose(0,2,1)
        v2[:,0,0]+=.3;v2[:,1,1]+=.3
        rad=np.sqrt(np.maximum(0.,2*np.log(255*self.opacity[active]))[:,None]*v2[:,[0,1],[0,1]])
        exact=(z<.42) if correct_near else np.zeros_like(z,dtype=bool)
        original=pix.copy()
        idx=np.flatnonzero(exact)
        if len(idx):
            cov=world_to_cv@self.cov[ids[idx]]@world_to_cv.T
            q=np.sqrt(np.maximum(0.,2*np.log(255*self.opacity[ids[idx]])))[:,None]
            margin=q*np.sqrt(cov[:,[0,1,2],[0,1,2]])
            low=xyz[idx]-margin;high=xyz[idx]+margin
            z0=np.maximum(low[:,2],near)
            z1=high[:,2]
            ratios=np.stack([low[:,:2]/z0[:,None],low[:,:2]/z1[:,None],
                             high[:,:2]/z0[:,None],high[:,:2]/z1[:,None]])
            minp=np.maximum(ratios.min(axis=0)*hf+center,-1.)
            maxp=np.minimum(ratios.max(axis=0)*hf+center,size+1.)
            pix[idx]=.5*(minp+maxp)
            rad[idx]=.5*(maxp-minp)
        outside=(pix+rad<0)|(pix-rad>=size)|(rad<0)
        keep=~np.any(outside,axis=1)
        rad,pix,v2,z,ids=rad[keep],pix[keep],v2[keep],z[keep],ids[keep]
        colors=self.dc[ids]
        if self.sh is not None:
            direction=d[ids]/np.linalg.norm(d[ids],axis=1,keepdims=True)
            basis=direction[:,[1,2,0]]*np.array([-.4886025119,.4886025119,-.4886025119])
            colors+=np.einsum('nij,nj->ni',self.sh[ids],basis)
        exact=exact[keep]
        parameters=np.zeros((len(ids),10))
        ix=np.flatnonzero(exact)
        if len(ix):
            cov=world_to_cv@self.cov[ids[ix]]@world_to_cv.T
            cam_xyz=local[ids[ix]]
            inv=np.linalg.inv(cov)
            b=(inv@cam_xyz[:,:,None]).squeeze(-1)
            parameters[ix]=np.column_stack([inv[:,0,0],inv[:,0,1],inv[:,0,2],inv[:,1,1],
                                            inv[:,1,2],inv[:,2,2],b,np.sum(b*cam_xyz,axis=1)])
        return z,pix,rad,np.linalg.inv(v2),np.maximum(colors,0),self.opacity[ids],original[keep],parameters

@njit
def raster(width,height,order,pix,rad,inv,colors,opacity,original,params,fx,fy,cx,cy):
    rgb=np.zeros((height,width,3),np.float32)
    trans=np.ones((height,width),np.float32)
    for i in order:
        u,v=pix[i];rx,ry=rad[i]
        xmin=max(0,int(np.floor(u-rx)));xmax=min(width,int(np.ceil(u+rx)))
        ymin=max(0,int(np.floor(v-ry)));ymax=min(height,int(np.ceil(v+ry)))
        for y in range(ymin,ymax):
            dy=y+.5-original[i,1]
            for x in range(xmin,xmax):
                old=trans[y,x]
                if old<=1e-4:continue
                dx=x+.5-original[i,0]
                if params[i,0]>0:
                    tx=(x+.5-cx)/fx
                    ty=(y+.5-cy)/fy
                    h=params[i]
                    a=h[0]*tx*tx+2*h[1]*tx*ty+2*h[2]*tx+h[3]*ty*ty+2*h[4]*ty+h[5]
                    b=h[6]*tx+h[7]*ty+h[8]
                    if b<=0:continue
                    sigma=.5*max(0.,h[9]-b*b/a)
                else:
                    sigma=.5*(inv[i,0,0]*dx*dx+inv[i,1,1]*dy*dy)+inv[i,0,1]*dx*dy
                alpha=min(.999,opacity[i]*np.exp(-sigma))
                if alpha<1/255:continue
                for c in range(3):rgb[y,x,c]+=old*alpha*colors[i,c]
                trans[y,x]=old*(1-alpha)
    return np.rint(255*np.clip(rgb,0,1)).astype(np.uint8),trans

def camera_for_eye(pose,origin,unit):
    a=np.eye(4)
    a[:3,:3]=pose['rot']
    a[:3,3]=(np.array(pose['pos_mm'])/1000-origin)/unit
    f=512/(2*tan(radians(157)/2))
    return dict(transform_matrix=a.tolist(),w=450,h=512,fl_x=f,fl_y=f,cx=225,cy=256)

def main(model,cal,poses,out,foreground=None,correct_near=False):
    g=GaussianField(model)
    cfg=json.loads(cal.read_text())
    query=json.loads(poses.read_text())
    out.mkdir(exist_ok=True)
    cameras={Path(v['file_path']).stem:v for v in cfg['frames'] if v['split']=='holdout'}
    for frame,pose in query.items():
        if pose is None:continue
        for side,eye in pose['eyes'].items():
            cameras[f'eye_{frame}_{side}']=camera_for_eye(eye,np.array(cfg['origin_m']),cfg['unit_m'])
    for key,c in cameras.items():
        arrays=g.project(c,correct_near=correct_near)
        if foreground and key.startswith('eye_'):
            frame=key.split('_')[1]
            fg=GaussianField(foreground/(frame+'.ply')).project(c,near=.003,correct_near=True)
            arrays=[np.concatenate([array,other]) for array,other in zip(arrays,fg)]
        z,*projected=arrays
        image,trans=raster(c['w'],c['h'],np.argsort(z,kind='stable'),*projected,c['fl_x'],c['fl_y'],c['cx'],c['cy'])
        Image.fromarray(image).save(out/(key+'.png'))
        if key.startswith(('hold_','eye_211')):
            Image.fromarray(np.uint8(np.rint(255*(1-trans)))).save(out/(key+'_alpha.png'))
    print('Model splats',len(g.center),'renders',len(cameras))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('model','calibration','eye_poses','output'):
        p.add_argument(name,type=Path)
    p.add_argument('--foreground',type=Path)
    p.add_argument('--correct-near',action='store_true')
    a=p.parse_args()
    main(a.model,a.calibration,a.eye_poses,a.output,a.foreground,a.correct_near)
