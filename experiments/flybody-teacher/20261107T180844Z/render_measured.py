"""Display saved measured-reference teacher physics; no new motion simulation."""
from pathlib import Path
import argparse,json,hashlib,subprocess
import numpy as np
import mujoco
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);p=ap.parse_args().folder
    video=p/'MEASURED_TEACHER_4x_slow.mp4';assert not video.exists()
    with np.load(p/'BODY.npz') as f:z={k:f[k] for k in f.files}
    protocol=json.loads((p/'PROTOCOL.json').read_text());result=json.loads((p/'RESULT.json').read_text())
    m=mujoco.MjModel.from_binary_path(str(p/'MODEL.mjb'));d=mujoco.MjData(m)
    for i in range(m.ngeom):
        n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])) or ''
        if n.startswith('ghost/'):m.geom_rgba[i,3]=0.
    ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'groundplane');m.geom_size[ground,:2]=100
    body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'walker/thorax')
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=.8;cam.azimuth=135;cam.elevation=-25
    framecount=int(np.ceil((z['time'][-1]-1e-10)*120))
    indices=np.minimum(np.searchsorted(z['time'],np.arange(framecount)/120),len(z['time'])-1)
    selected=np.linspace(0,len(indices)-1,8).round().astype(int);shots=[];comerr=0.
    w,h=960,720
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    render=mujoco.Renderer(m,height=h,width=w)
    try:
        for frame,i in enumerate(indices):
            d.time=float(z['time'][i])
            for key in ['qpos','qvel','act','ctrl','qacc_warmstart']:getattr(d,key)[:]=z[key][i]
            mujoco.mj_forward(m,d);comerr=max(comerr,float(np.abs(d.subtree_com[body]-z['com'][i]).max()))
            cam.lookat[:]=d.xpos[body];render.update_scene(d,camera=cam);im=Image.fromarray(render.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,w,64),fill=(10,12,15))
            draw.text((10,7),f'OFFICIAL ANN TEACHER | MEASURED REFERENCE {protocol["trajectory"]} | t={d.time:.3f}s | 4x slow',fill='white')
            draw.text((10,25),'Original adhesive feet | Actual physics response, not reference pose playback | NOT connectome brain',fill=(255,220,90))
            draw.text((10,43),f'Non-claw contact max={result["nonfoot_max"]:.4g} native | All outcomes retained | Ghost hidden for display',fill=(255,180,160))
            proc.stdin.write(np.asarray(im).tobytes())
            if frame in selected:shots.append(im.resize((480,360)))
        proc.stdin.close();assert proc.wait()==0
    finally:
        render.close()
        if proc.poll() is None:proc.kill();proc.wait()
    sheet=Image.new('RGB',(960,1440))
    for i,im in enumerate(shots):sheet.paste(im,((i%2)*480,(i//2)*360))
    sheet.save(p/'PREVIEW.png')
    report=dict(video=str(video),frames=len(indices),fps=30,slow_factor=4,physical_s=float(z['time'][-1]),render_com_max_error=comerr,
        body_sha256=hashlib.sha256((p/'BODY.npz').read_bytes()).hexdigest(),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        modifications='Render-only ghost transparency and infinite-plane display extent; BODY and original MODEL untouched',visual_review='pending')
    with (p/'VIDEO_MANIFEST.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
