"""Render saved official physics trajectory; no rollout, correction or generated motion."""
from pathlib import Path
import argparse,json,hashlib,subprocess
import numpy as np
import mujoco
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--suffix',default='_ground');a=ap.parse_args();p=a.folder
    video=p/('TEACHER_4x_slow'+a.suffix+'.mp4');assert not video.exists()
    with np.load(p/'BODY.npz') as saved:z={k:saved[k] for k in saved.files}
    m=mujoco.MjModel.from_binary_path(str(p/'MODEL.mjb'));d=mujoco.MjData(m)
    ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'groundplane')
    assert m.geom_type[ground]==mujoco.mjtGeom.mjGEOM_PLANE
    m.geom_size[ground,:2]=100. # Display extent only: physical plane was already infinite.
    # Render-only: hide separate noncolliding reference ghost, preserve physical source model.
    for i in range(m.ngeom):
        b=int(m.geom_bodyid[i]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or ''
        if name.startswith('ghost/'):m.geom_rgba[i,3]=0.
    thorax=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'walker/thorax')
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=.8;cam.azimuth=135;cam.elevation=-25
    fps=30;slow=4
    desired=np.arange(0,float(z['time'][-1]),1/(fps*slow));indices=np.minimum(np.searchsorted(z['time'],desired),len(z['time'])-1)
    width,height=960,720
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p',str(video)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    renderer=mujoco.Renderer(m,height=height,width=width)
    selected=np.linspace(0,len(indices)-1,8).round().astype(int);shots=[];maxcom=0.
    try:
        for frame,i in enumerate(indices):
            d.time=float(z['time'][i]);d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.act[:]=z['act'][i];d.ctrl[:]=z['ctrl'][i];d.qacc_warmstart[:]=z['qacc_warmstart'][i]
            mujoco.mj_forward(m,d)
            maxcom=max(maxcom,float(np.abs(d.subtree_com[thorax]-z['com'][i]).max()))
            cam.lookat[:]=d.xpos[thorax];renderer.update_scene(d,camera=cam)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,width,62),fill=(10,12,15))
            draw.text((10,7),f'OFFICIAL FLYBODY ANN TEACHER | reference-conditioned | t={d.time:.3f}s | 4x slow',fill='white')
            draw.text((10,25),'Original adhesive feet enabled | Synthetic reference | NOT connectome brain or surgical validation',fill=(255,220,90))
            draw.text((10,43),'Saved physical body trajectory only; noncolliding reference ghost hidden for display',fill=(180,220,190))
            proc.stdin.write(np.asarray(im).tobytes())
            if frame in selected:shots.append(im.resize((480,360)))
        proc.stdin.close();code=proc.wait();assert code==0,code
    finally:
        renderer.close()
        if proc.poll() is None:proc.kill();proc.wait()
    sheet=Image.new('RGB',(960,1440))
    for i,im in enumerate(shots):sheet.paste(im,((i%2)*480,(i//2)*360))
    sheet.save(p/('PREVIEW'+a.suffix+'.png'))
    report=dict(video=str(video),frames=len(indices),fps=fps,slow_factor=slow,physical_seconds=float(z['time'][-1]),body_sha256=hashlib.sha256((p/'BODY.npz').read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),render_com_max_error=maxcom,ghost_hidden_render_only=True,infinite_plane_visible_extent_only=100,visual_review='pending',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (p/('VIDEO_MANIFEST'+a.suffix+'.json')).open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
