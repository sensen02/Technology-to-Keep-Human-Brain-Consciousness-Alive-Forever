"""Calibrated orthographic fluorescence cameras. Decoding depends on RGB and calibration only."""
import json,io,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
W,H=256,180
pitch=.022 # mm per pixel
angles=np.deg2rad(np.arange(8)*45+17)
elev=np.deg2rad([54,24]*4)
view=np.stack((np.cos(angles)*np.cos(elev),np.sin(angles)*np.cos(elev),np.sin(elev)),1)
right=np.stack((-np.sin(angles),np.cos(angles),np.zeros(8)),1)
down=np.cross(right,view)
P=np.stack((right,down),1)/pitch
palette=np.array([[24+9*i,40+(71*i)%205,45+(59*i)%201] for i in range(22)],dtype='uint8')
def reconstruct(rgb,offset):
    observations=np.full((8,len(palette),2),np.nan)
    for v in range(8):
        pixels=rgb[(v//4)*H:(v//4+1)*H,(v%4)*W:(v%4+1)*W]
        y,x=np.nonzero(pixels[:,:,0])
        if not len(x):continue
        a=pixels[y,x]; ids=(a[:,0].astype(int)-24)//9
        valid=(ids>=0)&(ids<len(palette))
        ids,x,y,a=ids[valid],x[valid],y[valid],a[valid]
        ok=np.all(a==palette[ids],axis=1)
        ids,x,y=ids[ok],x[ok],y[ok]
        count=np.bincount(ids,minlength=len(palette))
        for axis,coordinate in enumerate((x,y)):
            sums=np.bincount(ids,weights=coordinate,minlength=len(palette))
            valid=count>=8
            observations[v,valid,axis]=sums[valid]/count[valid]-offset[v,axis]
    valid=np.isfinite(observations[:,:,0])
    points=np.full((len(palette),3),np.nan)
    for i,mask in enumerate(valid.T):
        if mask.sum()>=2:
            pts=P[mask].reshape(-1,3); b=observations[mask,i].ravel()
            q=np.linalg.lstsq(pts,b,rcond=None)[0]
            residual=np.linalg.norm((pts@q-b).reshape(-1,2),axis=1)
            good=residual<2.5
            if good.sum()>=2:
                points[i]=np.linalg.lstsq(pts.reshape(-1,2,3)[good].reshape(-1,3),b.reshape(-1,2)[good].ravel(),rcond=None)[0]
    return points,valid.sum(0)
def project_frame(points,offset,occluders):
    # Fluorescent spheres are opaque to one another. Approximate cuticle occlusion by
    # ellipsoids on the trunk and head; other surfaces are dark in this camera model.
    canvas=Image.new('RGB',(W*4,H*2))
    for k,direction in enumerate(view):
        im=Image.new('RGB',(W,H));draw=ImageDraw.Draw(im)
        xy=points@P[k].T+offset[k]
        seen=np.ones(len(points),bool)
        for center,rot,radius in occluders:
            delta=(points+.066*direction-center)@rot/radius
            ray=direction@rot/radius
            b=delta@ray;a=ray@ray
            t=np.maximum(-b/a,0.)
            seen&=np.sum((delta+t[:,None]*ray)**2,axis=1)>=1
        for i in np.argsort(points@direction):
            if not seen[i]:continue
            x,y=np.rint(xy[i]).astype(int)
            draw.ellipse((x-3,y-3,x+3,y+3),fill=tuple(palette[i]))
        canvas.paste(im,((k%4)*W,(k//4)*H))
    return canvas
def capture(folder):
    import mujoco
    m=mujoco.MjModel.from_binary_path(str(folder/'scene.mjb'));d=mujoco.MjData(m)
    ids=[m.body('walker/'+s+'_'+leg+'_'+side).id for side in ['left','right'] for leg in ['T1','T2','T3'] for s in ['tibia','tarsus','claw']]
    th=m.body('walker/thorax').id;head=m.body('walker/head').id
    mouths=[m.geom('walker/labrum_'+s+'_lower_collision').id for s in ['left','right']]
    camera_center=np.array([0.,0.,1.3]);offsets=[]
    with np.load(folder/'BODY.npz') as z: poses=z['poses']
    with zipfile.ZipFile(folder/'cameras.zip','w') as out:
        for t,q in enumerate(poses):
            d.qpos[:]=q;mujoco.mj_kinematics(m,d)
            rot=d.xmat[th].reshape(3,3)
            tags=d.xpos[th]*10+np.array([[0,0,.72],[.4,0,.72],[0,.25,.72]])@rot.T
            feet=d.xpos[ids].copy()*10
            for i in (2,5,8,11,14,17):
                geom_id=m.geom(m.body(ids[i]).name.replace('/claw_','/tarsal_claw_')+'_collision').id
                feet[i]=10*d.geom_xpos[geom_id]
            points=np.vstack((tags,feet,10*d.geom_xpos[mouths].mean(0)))
            off=np.array([W/2,H/2])-P@camera_center
            im=project_frame(points,off,[(10*d.xpos[th],rot,np.array([.48,.39,.4])),(10*d.xpos[head],d.xmat[head].reshape(3,3),np.array([.31,.34,.29]))])
            pix=np.asarray(im)
            # Track the actual rendered chest dot to aim the next exposure.
            chest=[]
            for k in range(8):
                tile=pix[k//4*H:(k//4+1)*H,k%4*W:(k%4+1)*W]
                y,x=np.nonzero(tile[:,:,0]==palette[0,0])
                if len(x)>=8:chest.append((k,np.array([x.mean(),y.mean()])-off[k]))
            if len(chest)>=2:
                ks=[k for k,_ in chest];camera_center=np.linalg.lstsq(P[ks].reshape(-1,3),np.ravel([a for _,a in chest]),rcond=None)[0]
            offsets.append(off)
            buf=io.BytesIO();im.save(buf,format='PNG',compress_level=1);out.writestr(f'{t:04d}.png',buf.getvalue())
    np.savez_compressed(folder/'camera_cal.npz',projection=P,offsets=offsets,palette=palette)
def decode(folder):
    with np.load(folder/'camera_cal.npz') as z:
        assert np.array_equal(P,z['projection']) and np.array_equal(palette,z['palette']);offsets=z['offsets']
    with np.load(folder/'floor.npz') as z:
        cells=z['cells'];cuts=z['offsets'];pitch_mm=z['pitch_mm']
    assert len(cuts)==len(offsets)+1
    pos=[];counts=[];fo=[];mouths=[];valids=[];matched=total=0.
    with zipfile.ZipFile(folder/'cameras.zip') as archive:
        for t,off in enumerate(offsets):
            im=np.asarray(Image.open(io.BytesIO(archive.read(f'{t:04d}.png'))))
            pts,n=reconstruct(im,off);mouths.append(pts[-1].copy());pts=pts[:-1].copy()
            # Convert externally triangulated raised chest dots to the parent frame.
            a,b,c=pts[:3];axis=np.cross(b-a,c-a)
            pts[:3]-=.54*axis/max(np.linalg.norm(axis),1e-8)
            feet=pts[5::3,:2];force=np.zeros(6)
            for x,y,f in cells[cuts[t]:cuts[t+1]]:
                total+=f
                distances=np.linalg.norm(feet-[x*pitch_mm,y*pitch_mm],axis=1)
                if np.isfinite(distances).any() and np.nanmin(distances)<.25:
                    force[np.nanargmin(distances)]+=f;matched+=f
            pos.append(pts);valids.append(np.isfinite(pts).all());counts.append(n);fo.append(force)
    np.savez_compressed(folder/'external.npz',markers_mm=np.array(pos,'f'),mouth_mm=np.array(mouths,'f'),force_uN=np.array(fo,'f'),valid=valids,views=counts)
    print(folder.name,'frames',len(pos),'complete',sum(valids),'feet pressure fraction',matched/max(total,1e-9),flush=True)
    (folder/'optical.json').write_text(json.dumps(dict(frames=len(pos),complete=int(sum(valids)),assigned_force_fraction=matched/max(total,1e-9),mean_assigned_force_uN=float(np.mean(np.sum(fo,axis=1))),total_force_uN_samples=total),indent=2))
if __name__=='__main__':
    import sys
    for arg in sys.argv[2:]:
        (capture if sys.argv[1]=='capture' else decode)(Path(arg))
