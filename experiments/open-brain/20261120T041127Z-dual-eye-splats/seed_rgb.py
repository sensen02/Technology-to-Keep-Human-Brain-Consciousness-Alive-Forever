"""Initialize a Gaussian field from calibrated RGB feature matches and a surveyed floor prior."""
import argparse,csv,json,os
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree

def cameras(*datasets):
    frames=[]
    for root in datasets:
        cfg=json.loads((root/'calibration.json').read_text())
        for f in cfg['frames']:
            if f['split']!='train':continue
            c2w=np.array(f['transform_matrix']);r=np.diag([1.,-1.,-1.])@c2w[:3,:3].T;t=c2w[:3,3]
            k=np.array([[f['fl_x'],0,f['cx']],[0,f['fl_y'],f['cy']],[0,0,1]])
            p=k@np.c_[r,-r@t]
            rgb=cv2.cvtColor(cv2.imread(str(root/f['file_path'])),cv2.COLOR_BGR2RGB)
            frames.append(dict(row=f,root=root,c=t,p=p,r=r,k=k,rgb=rgb))
    return frames,cfg

def pair_points(a,b,matcher):
    ka,da=a['feat'];kb,db=b['feat']
    if len(ka)<8 or len(kb)<8:return np.empty((0,3)),np.empty((0,3))
    xy=[]
    for pair in matcher.knnMatch(da,db,k=2):
        if len(pair)==2:
            m,n=pair
            if m.distance<.83*n.distance:xy.append([ka[m.queryIdx],kb[m.trainIdx]])
    if not xy:return np.empty((0,3)),np.empty((0,3))
    l,r=np.array(xy).transpose(1,0,2)
    c1,c2,p1,p2=a['c'],b['c'],a['p'],b['p']
    x=cv2.triangulatePoints(p1,p2,l.T,r.T)
    x=(x[:3]/x[3:4]).T
    def project(p):
        v=x@p[:,:3].T+p[:,3];return v[:,:2]/v[:,2,None],v[:,2]
    u,d1=project(p1);v,d2=project(p2)
    w1=x-c1;w2=x-c2
    f1=np.linalg.norm(w1,axis=1);f2=np.linalg.norm(w2,axis=1)
    cos=np.sum(w1*w2,axis=1)/np.maximum(f1*f2,1e-9)
    keep=(d1>.055)&(d2>.055)&(np.maximum(f1,f2)<105)&(cos<.99978)
    keep&=(np.linalg.norm(u-l,axis=1)<.8)&(np.linalg.norm(v-r,axis=1)<.8)
    def color(cam,uv):
        ux,uy=np.rint(uv).astype(int).T
        image=cam['rgb']
        return image[np.clip(uy,0,image.shape[0]-1),np.clip(ux,0,image.shape[1]-1)].astype(float)
    s,t=color(a,l),color(b,r)
    keep&=(np.linalg.norm(s-t,axis=1)<49)
    return x[keep],(.5*(s+t))[keep]/255.

def write_gaussians(path,xyz,colors,scales):
    n=len(xyz)
    names=['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    v=np.column_stack([xyz,(colors-.5)/.2820947917738781,np.full(n,-1.4),np.log(scales),
                       np.ones(n),np.zeros((n,3))]).astype('<f4')
    h='ply\nformat binary_little_endian 1.0\n'
    h+=f'element vertex {n}\n'
    h+=''.join(f'property float {k}\n' for k in names)+'end_header\n'
    with path.open('wb') as f:f.write(h.encode());v.tofile(f)

def main(scan,detail,survey,out):
    cv2.setNumThreads(5)
    frames,cfg=cameras(scan,detail)
    sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.006,edgeThreshold=15)
    for f in frames:
        kp,d=sift.detectAndCompute(cv2.cvtColor(f['rgb'],cv2.COLOR_RGB2GRAY),None)
        f['feat']=np.array([i.pt for i in kp]),d
    matcher=cv2.BFMatcher()
    points=[];colors=[]
    for i,a in enumerate(frames):
        best=[]
        for j,b in enumerate(frames):
            if j==i:continue
            f1,f2=a['r'][2],b['r'][2]
            baseline=np.linalg.norm(b['c']-a['c'])
            if f1@f2>.84 and baseline>.14:
                best.append((j,float(f1@f2)+min(baseline,.8)*.045))
        for j,_ in sorted(best,key=lambda x:-x[1])[:12]:
            p,c=pair_points(a,frames[j],matcher)
            points.extend(p);colors.extend(c)
    points=np.array(points);colors=np.array(colors)
    ground_rows=list(csv.DictReader(survey.open()))
    ground=np.array([[float(r[k]) for k in ['toe_x_mm','toe_y_mm','ground_z_mm']] for r in ground_rows])
    ground=(ground/1000-np.array(cfg['origin_m']))/cfg['unit_m']
    abc=np.linalg.lstsq(np.c_[ground[:,:2],np.ones(len(ground))],ground[:,2],rcond=None)[0]
    # Use the measured local plane only in the central area, before photometric fitting.
    def local_plane(x):return x[:,:2]@abc[:2]+abc[2]
    ok=(abs(points[:,0])<85)&(abs(points[:,1])<85)&(points[:,2]>-.6)&(points[:,2]<85)
    points,colors=points[ok],colors[ok]
    _,ix=np.unique(np.rint(points/.037).astype(np.int32),axis=0,return_index=True)
    points,colors=points[ix],colors[ix]
    assert len(points)>1500,len(points)
    d=cKDTree(points).query(points,k=4)[0][:,1:]
    s=np.clip(np.mean(d,axis=1)*.48,.02,2.)
    g=np.arange(-1.45,1.46,.032)
    x,y=np.meshgrid(g,g)
    xy=np.c_[x.ravel(),y.ravel()]
    ground_pts=np.c_[xy,local_plane(xy)]
    # Sand color is estimated from bottom rows of the normal RGB scan.
    rgb=np.median(np.concatenate([a['rgb'][-30:,:,:].reshape(-1,3) for a in frames[:80]]),axis=0)/255.
    floor_scale=np.broadcast_to([.027,.027,.004],(len(ground_pts),3))
    zz=np.vstack([points,ground_pts])
    cc=np.vstack([colors,np.broadcast_to(rgb,(len(ground_pts),3))])
    ss=np.vstack([np.repeat(s[:,None],3,axis=1),floor_scale])
    out.mkdir(exist_ok=False);(out/'images').mkdir()
    write_gaussians(out/'init.ply',zz,cc,ss)
    cameras_out=[]
    for a in frames:
        row=a['row'];key=row['file_path'];os.link(a['root']/key,out/key);cameras_out.append(row)
    (out/'transforms.json').write_text(json.dumps(dict(frames=cameras_out,ply_file_path='init.ply'),indent=2))
    audit=dict(image_feature_points=len(points),survey_plane_points=len(ground_pts),floor_rgb=rgb.tolist(),
               plane=abc.tolist(),rgb_views=len(frames))
    (out/'INITIALIZATION.json').write_text(json.dumps(audit,indent=2))
    print(audit,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('scan','detail','survey','output'):p.add_argument(name,type=Path)
    a=p.parse_args()
    main(a.scan,a.detail,a.survey,a.output)
