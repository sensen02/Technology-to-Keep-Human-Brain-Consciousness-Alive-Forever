"""Render only saved physics states; explicit teacher/replacement labels and source hashes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import mujoco as mj
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z')
from incremental_actuator import load_full_model


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder');ap.add_argument('--label',default='Replacement ANN')
    args=ap.parse_args();folder=Path(args.folder)
    source=folder/'BODY.npz'; before=sha(source)
    with np.load(source) as z:b={k:z[k].copy() for k in z.files}
    with np.load(folder/'NATIVE.npz') as z:native=z['rows']
    result=json.loads((folder/'RESULT.json').read_text())
    m=load_full_model();mask=m.geom_type==mj.mjtGeom.mjGEOM_MESH
    assert int(mask.sum())==71 and np.all(m.geom_rgba[mask,3]==1.)
    m.vis.global_.offwidth=1024;m.vis.global_.offheight=720
    r=mj.Renderer(m,height=720,width=1024);d=mj.MjData(m);cam=mj.MjvCamera();mj.mjv_defaultCamera(cam)
    cam.distance=7.5;cam.azimuth=125;cam.elevation=-26
    opt=mj.MjvOption();mj.mjv_defaultOption(opt);opt.sitegroup[:]=0
    for name in ['mjVIS_TENDON','mjVIS_ACTUATOR','mjVIS_JOINT']:opt.flags[getattr(mj.mjtVisFlag,name)]=0
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    fps=30;slow=4;mapping=[];err=0.;tiles=[];gif=[]
    video=folder/'WALK_4x_slow.mp4'
    sample_times=[.5,1.,2.,4.,6.,8.,10.,float(b['time'][-1])-.01]
    nextsample=0
    with imageio.get_writer(video,fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
        for fi,t in enumerate(np.arange(b['time'][0],b['time'][-1]+1e-9,1/(fps*slow))):
            ix=int(np.argmin(np.abs(b['time']-t)));actual=float(b['time'][ix])
            d.qpos[:]=b['qpos'][ix];d.qvel[:]=b['qvel'][ix];d.ctrl[:]=b['ctrl'][ix];mj.mj_forward(m,d)
            err=max(err,float(np.abs(d.subtree_com[m.body('Thorax').id]-b['com'][ix]).max()))
            cam.lookat[:]=[d.qpos[0],d.qpos[1],1.0]
            r.update_scene(d,camera=cam,scene_option=opt)
            im=Image.fromarray(r.render());dr=ImageDraw.Draw(im)
            dr.rectangle((0,0,1024,112),fill=(18,20,26))
            phase='ALGORITHM TEACHER WARMUP' if actual<1 else ('ALGORITHM TEACHER CONTROL' if result['controller']=='teacher' else 'REPLACEMENT ANN ONLY')
            dr.text((12,7),f'{args.label} | t={actual:.3f}s | 4x slow',font=font,fill='white')
            dr.text((12,34),phase,font=font,fill=(255,220,95))
            ni=min(int(np.searchsorted(native[:,0],actual)),len(native)-1)
            lo=max(ni-200,0);hi=min(ni+200,len(native))
            foot=float(native[lo:hi,1:7].sum(1).mean()/24.446354755878005)
            nf=float(native[lo:hi,7].max());com=float(b['com'][ix,2])
            dr.text((12,61),f'Foot load={foot:.3f} W | non-foot={nf:.3g} native | COM height={com:.3f}',font=small,fill='white')
            checks=result['validation']['checks'];failed=[k.split('_',1)[0] for k,v in checks.items() if not v]
            dr.text((12,85),'Full 71-mesh body | no root support | '+('failed support gates: '+','.join(failed) if failed else 'support gates pass'),font=small,fill=(255,180,155) if failed else (170,240,170))
            writer.append_data(np.asarray(im));mapping.append(dict(frame=fi,source_row=ix,time=actual))
            if fi%12==0:gif.append(im.resize((640,450)))
            if nextsample<len(sample_times) and actual>=sample_times[nextsample]:
                tiles.append(im.resize((512,360)));nextsample+=1
    r.close();assert sha(source)==before and err<1e-12
    rd=imageio.get_reader(video);decoded=rd.count_frames();rd.get_data(decoded//2);rd.close();assert decoded==len(mapping)
    if tiles:
        sheet=Image.new('RGB',(1024,360*((len(tiles)+1)//2)),(20,20,25))
        for i,im in enumerate(tiles):sheet.paste(im,((i%2)*512,(i//2)*360))
        sheet.save(folder/'PREVIEW.png')
    if gif:gif[0].save(folder/'PLAY.gif',save_all=True,append_images=gif[1:],duration=400,loop=0,optimize=False)
    manifest=dict(source=str(source),source_sha256=before,source_unchanged=True,
        visible_meshes=71,material_alpha='opaque',video=str(video),frames=decoded,fps=fps,slow_factor=slow,
        replay_com_max_error=err,mapping=mapping,video_sha256=sha(video),
        provenance='rendered saved BODY only; no generated or corrected motion',visual_inspection='pending independent image read')
    (folder/'VIDEO_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='mapping'}),flush=True)


if __name__=='__main__':main()
