"""Render recorded retained-body trajectory only; all71meshes retained."""
from pathlib import Path
import argparse,json,subprocess,hashlib
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);p=ap.parse_args().folder
    with np.load(p/'BODY.npz') as f:z={k:f[k] for k in f.files}
    m=mj.MjModel.from_binary_path(str(p/'MODEL.mjb'));d=mj.MjData(m);thorax=m.body('Thorax').id
    assert m.nmesh==71 and np.all(m.geom_rgba[m.geom_type==mj.mjtGeom.mjGEOM_MESH,3]==1)
    for g in np.flatnonzero(m.geom_type==mj.mjtGeom.mjGEOM_PLANE):m.geom_size[g,:2]=100
    result=json.loads((p/'RESULT.json').read_text());protocol=json.loads((p/'PROTOCOL.json').read_text());scale=protocol.get('time_scale',1.);video=p/'RETARGET_PHYSICS_4x_slow.mp4';assert not video.exists()
    cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=8.;cam.azimuth=135;cam.elevation=-25
    width,height=960,720
    m.vis.global_.offwidth=width;m.vis.global_.offheight=height # Render-only framebuffer; original model artifact unchanged.
    renderer=mj.Renderer(m,height=height,width=width)
    samples=np.arange(int(np.ceil((z['time'][-1]-1e-9)*120)))/120;indices=np.minimum(np.searchsorted(z['time'],samples),len(z['time'])-1);selected=np.linspace(0,len(indices)-1,8).round().astype(int);shots=[];error=0.
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','30','-i','-','-an','-c:v','libx264','-crf','21','-preset','fast','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    try:
        for frame,i in enumerate(indices):
            d.time=float(z['time'][i])
            for key in ['qpos','qvel','ctrl','qacc_warmstart']:getattr(d,key)[:]=z[key][i]
            mj.mj_forward(m,d);error=max(error,float(np.abs(d.subtree_com[thorax]-z['com'][i]).max()));cam.lookat[:]=d.xpos[thorax];renderer.update_scene(d,camera=cam)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,width,65),fill=(8,10,12))
            phase='INITIAL HOLD' if d.time<.2 else 'RETARGETED REFERENCE FORCING'
            draw.text((10,7),f'CURRENT FULL BODY | {phase} | t={d.time:.3f}s | 4x slow',fill='white')
            draw.text((10,25),'Original gravity / contacts / finite torque | NO added adhesion | NO brain or learned policy in this diagnostic',fill=(255,220,90))
            draw.text((10,43),f'Source-time scale={scale:g}; support passed={result["physical_gate"]["passed"]} | Saved physical failure, no added motion',fill=(255,180,150))
            proc.stdin.write(np.asarray(im).tobytes())
            if frame in selected:shots.append(im.resize((480,360)))
        proc.stdin.close();assert proc.wait()==0
    finally:
        renderer.close()
        if proc.poll() is None:proc.kill();proc.wait()
    sheet=Image.new('RGB',(960,1440))
    for i,im in enumerate(shots):sheet.paste(im,((i%2)*480,(i//2)*360))
    sheet.save(p/'PREVIEW.png')
    report=dict(video=str(video),source_time_scale=scale,render_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frames=len(indices),fps=30,slow_factor=4,mesh_count=71,render_COM_error=error,body_sha256=hashlib.sha256((p/'BODY.npz').read_bytes()).hexdigest(),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),visual_review='pending')
    with (p/'VIDEO_MANIFEST.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
