"""Render same-run full body, labeled neural handover and matched hold."""
from pathlib import Path
import argparse,json,subprocess,hashlib
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);p=ap.parse_args().folder
    with np.load(p/'BODY.npz') as f:z={k:f[k] for k in f.files}
    r=json.loads((p/'RESULT.json').read_text());m=mj.MjModel.from_binary_path(str(p/'MODEL.mjb'));d=mj.MjData(m);thorax=m.body('Thorax').id
    assert m.nmesh==71 and np.all(m.geom_rgba[m.geom_type==mj.mjtGeom.mjGEOM_MESH,3]==1)
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    m.geom_size[m.geom_type==mj.mjtGeom.mjGEOM_PLANE,:2]=100
    renderer=mj.Renderer(m,height=720,width=960);cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=8;cam.azimuth=135;cam.elevation=-25
    video=p/'BANC_LOCAL_4x_slow.mp4';assert not video.exists()
    times=np.arange(360)/120;indices=np.minimum(np.searchsorted(z['time'],times),len(z['time'])-1);selected=np.linspace(0,len(indices)-1,8).round().astype(int);shots=[];error=0.
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','30','-i','-','-an','-c:v','libx264','-crf','21','-preset','fast','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    try:
        for f,i in enumerate(indices):
            d.time=float(z['time'][i])
            for key in ['qpos','qvel','ctrl','qacc_warmstart']:getattr(d,key)[:]=z[key][i]
            mj.mj_forward(m,d);error=max(error,float(np.abs(d.subtree_com[thorax]-z['com'][i]).max()));cam.lookat[:]=d.xpos[thorax];renderer.update_scene(d,camera=cam)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,68),fill=(8,10,12))
            phase='TEACHER WARMUP' if d.time<1 else ('LOCAL NEURONS + LEARNED MN READOUT' if r['condition']=='learned' else 'ZERO RATE: PD TARGET HOLD BASELINE')
            draw.text((10,7),f'{phase} | t={d.time:.3f}s | display4xslow',fill='white')
            draw.text((10,25),'20k real IDs, strict source edges; zero DN drive | No brain/feeding yet | No teacher after1s',fill=(255,220,90))
            draw.text((10,43),f'Illustrative neural parameters: voltage-floor check FAILED | support={r["physical_gate"]["passed"]} | pilot only',fill=(255,180,150))
            proc.stdin.write(np.asarray(im).tobytes())
            if f in selected:shots.append(im.resize((480,360)))
        proc.stdin.close();assert proc.wait()==0
    finally:
        renderer.close()
        if proc.poll() is None:proc.kill();proc.wait()
    sheet=Image.new('RGB',(960,1440))
    for i,im in enumerate(shots):sheet.paste(im,((i%2)*480,(i//2)*360))
    sheet.save(p/'PREVIEW.png')
    out=dict(video=str(video),frames=len(indices),fps=30,display_slow_factor=4,mesh_count=71,render_COM_error=error,body_sha256=hashlib.sha256((p/'BODY.npz').read_bytes()).hexdigest(),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),render_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),visual_review='pending')
    with (p/'VIDEO_MANIFEST.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
