"""Render verified shared warmup plus same-run checkpoint release, clear provenance."""
from pathlib import Path
import argparse,json,hashlib,subprocess
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw
OLD=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/closed_loop')
def load(p):
    with np.load(p) as f:return {k:f[k] for k in f.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);p=ap.parse_args().folder
    new=load(p/'BODY.npz');old=load(OLD/'BODY.npz');r=json.loads((p/'RESULT.json').read_text());is_zero=not np.any(load(p/'ACTOR.npz')['theta'])
    assert abs(new['time'][0]-1.)<1e-8
    seam={k:float(np.abs(old[k][1000]-new[k][0]).max()) for k in ['time','qpos','qvel','com']};assert max(seam.values())<1e-9,seam
    z={k:np.concatenate([old[k][:1000],new[k]],axis=0) for k in ['time','qpos','qvel','ctrl','qacc_warmstart','com']}
    m=mj.MjModel.from_binary_path(str(p/'MODEL.mjb'));d=mj.MjData(m);thorax=m.body('Thorax').id
    assert m.nmesh==71 and np.all(m.geom_rgba[m.geom_type==mj.mjtGeom.mjGEOM_MESH,3]==1)
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720;m.geom_size[m.geom_type==mj.mjtGeom.mjGEOM_PLANE,:2]=100
    renderer=mj.Renderer(m,height=720,width=960);cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=8;cam.azimuth=135;cam.elevation=-25
    video=p/'PHYSICAL_LEARNING_4x_slow.mp4';assert not video.exists();nframes=int(round(z['time'][-1]*120))
    indices=np.minimum(np.searchsorted(z['time'],np.arange(nframes)/120),len(z['time'])-1);selected=np.linspace(0,len(indices)-1,8).round().astype(int);shots=[];error=0.
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','30','-i','-','-an','-c:v','libx264','-crf','21','-preset','fast','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    try:
        for fi,i in enumerate(indices):
            d.time=float(z['time'][i])
            for key in ['qpos','qvel','ctrl','qacc_warmstart']:getattr(d,key)[:]=z[key][i]
            mj.mj_forward(m,d);error=max(error,float(np.abs(d.subtree_com[thorax]-z['com'][i]).max()));cam.lookat[:]=d.xpos[thorax];renderer.update_scene(d,camera=cam)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,76),fill=(8,10,12))
            phase='SHARED VERIFIED REFERENCE WARMUP (ARCHIVE)' if d.time<1 else ('SELECTED ZERO-RATE INCUMBENT: NO WALK LEARNED' if is_zero and p.name=='selected' else ('FROZEN TRAINED MN READOUT' if p.name=='selected' else 'ZERO RATE HOLD BASELINE'))
            draw.text((10,7),f'{phase} | t={d.time:.3f}s | display4xslow',fill='white')
            draw.text((10,25),'Release initialized from exact same physical/neural checkpoint at1s; no new warmup physics',fill=(255,220,90))
            draw.text((10,43),'20k source neuron IDs | same-leg MN-only readout | zero DN | original complete body',fill=(230,230,230))
            draw.text((10,59),f'Support screen={r["physical_gate"]["passed"]} | 5s release is not stable walking/platform acceptance',fill=(255,180,150))
            proc.stdin.write(np.asarray(im).tobytes())
            if fi in selected:shots.append(im.resize((480,360)))
        proc.stdin.close();assert proc.wait()==0
    finally:
        renderer.close()
        if proc.poll() is None:proc.kill();proc.wait()
    sheet=Image.new('RGB',(960,1440))
    for i,im in enumerate(shots):sheet.paste(im,((i%2)*480,(i//2)*360))
    sheet.save(p/'PREVIEW.png')
    result=dict(video=str(video),frames=nframes,fps=30,display_slow_factor=4,mesh_count=71,render_COM_error=error,seam_max_errors=seam,shared_warm_BODY=str(OLD/'BODY.npz'),shared_warm_BODY_sha256=sha(OLD/'BODY.npz'),release_BODY_sha256=sha(p/'BODY.npz'),video_sha256=sha(video),source_sha256=sha(__file__),visual_review='pending',provenance='Archived exact common1s warmup prefix + actual new5s release; matching seam checked, no new warmup simulation claimed.')
    with (p/'VIDEO_MANIFEST.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
