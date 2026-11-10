"""Same-run whole-body visualization of the new conductance/history pilot."""
from pathlib import Path
import json,subprocess,hashlib
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw
P=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/closed_loop')
with np.load(P/'BODY.npz') as f:z={k:f[k] for k in f.files}
r=json.loads((P/'RESULT.json').read_text());m=mj.MjModel.from_binary_path(str(P/'MODEL.mjb'));d=mj.MjData(m);thorax=m.body('Thorax').id
assert m.nmesh==71 and np.all(m.geom_rgba[m.geom_type==mj.mjtGeom.mjGEOM_MESH,3]==1)
m.vis.global_.offwidth=960;m.vis.global_.offheight=720;m.geom_size[m.geom_type==mj.mjtGeom.mjGEOM_PLANE,:2]=100
renderer=mj.Renderer(m,height=720,width=960);cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=8;cam.azimuth=135;cam.elevation=-25
video=P/'CONDUCTANCE_HISTORY_4x_slow.mp4';assert not video.exists()
times=np.arange(360)/120;indices=np.minimum(np.searchsorted(z['time'],times),len(z['time'])-1);selected=np.linspace(0,359,8).round().astype(int);shots=[];error=0.
proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','30','-i','-','-an','-c:v','libx264','-crf','21','-preset','fast','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
try:
    for f,i in enumerate(indices):
        d.time=float(z['time'][i])
        for key in ['qpos','qvel','ctrl','qacc_warmstart']:getattr(d,key)[:]=z[key][i]
        mj.mj_forward(m,d);error=max(error,float(np.abs(d.subtree_com[thorax]-z['com'][i]).max()));cam.lookat[:]=d.xpos[thorax];renderer.update_scene(d,camera=cam)
        im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,68),fill=(8,10,12))
        phase='REFERENCE-RATE WARMUP' if d.time<1 else 'LOCAL NEURONS + MN HISTORY READOUT'
        draw.text((10,7),f'{phase} | t={d.time:.3f}s | display4xslow',fill='white')
        draw.text((10,25),'20k source identities | adaptation CONDUCTANCE | past MN lags 0/5/20/50ms | zero DN drive',fill=(255,220,90))
        draw.text((10,43),f'Uncalibrated physiology | support screen={r["physical_gate"]["passed"]} | short pilot; NOT walking acceptance',fill=(255,180,150))
        proc.stdin.write(np.asarray(im).tobytes())
        if f in selected:shots.append(im.resize((480,360)))
    proc.stdin.close();assert proc.wait()==0
finally:
    renderer.close()
    if proc.poll() is None:proc.kill();proc.wait()
sheet=Image.new('RGB',(960,1440))
for i,im in enumerate(shots):sheet.paste(im,((i%2)*480,(i//2)*360))
sheet.save(P/'PREVIEW.png')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=dict(video=str(video),frames=360,fps=30,display_slow_factor=4,mesh_count=71,render_COM_error=error,body_sha256=sha(P/'BODY.npz'),video_sha256=sha(video),render_source_sha256=sha(__file__),visual_review='pending')
with (P/'VIDEO_MANIFEST.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
