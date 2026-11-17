"""Save synchronized arc exits from actual retained physical trajectories."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from rollout import stan,read

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
W,H,PIC,FPS=620,382,308,30


def load_case(p):
    obj=read(p/'RESULT.json')
    path=p/'TRACE.npz'
    assert obj['trace_sha256']==stan.sha(path)
    with np.load(path,allow_pickle=False) as f:
        trace={k:f[k].copy() for k in ('qpos','qvel','yaw','com')}
    title=('左弧' if obj['direction']=='left' else '右弧')+(' · 直接切换' if obj['kind'].startswith('direct') else ' · 渐消偏航')
    return obj,trace,path,title


def render(output,a,b):
    output.mkdir()
    data=(load_case(a),load_case(b))
    end=data[0][0]['actual_ms']
    assert all(t[0]['actual_ms']==end for t in data)
    core=stan.physics.Core(adhesion=False)
    def forbidden(*a,**kw):
        raise RuntimeError('Rendering cannot advance physical simulation')
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    points=np.concatenate([d[1]['com'][:,:2] for d in data],axis=0)
    cam=mj.MjvCamera()
    cam.type=mj.mjtCamera.mjCAMERA_FREE
    cam.azimuth=131
    cam.elevation=-25
    cam.distance=6.8+float(np.ptp(points,axis=0).max())*.29
    cam.lookat[:]=[*points.mean(0),.88]
    f1=ImageFont.truetype(FONT,23)
    f2=ImageFont.truetype(FONT,17)
    finish=data[0][0]['bounds'][1]
    settle=data[0][0]['straight_begin_ms']
    keyframes=[('静立',150),('直走入弯',data[0][0]['bounds'][0]-20),('进入出弯',finish),
               ('偏航渐消过程',finish+(settle-finish)//2),('已接直走',settle+300),
               ('末端复验',end)]
    frames=sorted(set(range(0,end+1,20))|{ms for title,ms in keyframes}|
                  set(range(finish-50,min(end,settle+380),10)))
    keynotes={k:v for v,k in keyframes}
    images={}
    video=output/'弧线出弯到直走配对.mp4'
    grid=output/'弧线衔接关键帧.png'
    with mj.Renderer(core.plant,height=PIC,width=W) as renderer,imageio.get_writer(
            video,fps=FPS,codec='libx264',quality=8,macro_block_size=2,ffmpeg_log_level='error') as sink:
        for idx in frames:
            pic=Image.new('RGB',(2*W,H),(244,246,245))
            draw=ImageDraw.Draw(pic)
            for i,(result,tr,p,title) in enumerate(data):
                core.d.qpos[:]=tr['qpos'][idx]
                core.d.qvel[:]=tr['qvel'][idx]
                mj.mj_forward(core.plant,core.d)
                np.testing.assert_array_equal(core.d.qpos,tr['qpos'][idx])
                renderer.update_scene(core.d,camera=cam)
                x=i*W
                pic.paste(Image.fromarray(renderer.render()),(x,H-PIC))
                start,arc,coast,end_=result['bounds']
                phase='直走入弯' if 500<=idx<start else '弧行' if start<=idx<arc else '出弯' if arc<=idx<coast else '站稳'
                yaw=np.rad2deg(np.arctan2(np.sin(tr['yaw'][idx]-tr['yaw'][start]),np.cos(tr['yaw'][idx]-tr['yaw'][start])))
                slide=result['legacy_exit_window']['max_normalized_slip']
                draw.text((x+11,5),f'{title} · {keynotes.get(idx,phase)}',font=f1,fill='#263936')
                draw.text((x+11,39),f'{idx/1000:.3f} s · 相对转角 {yaw:+.2f}° · 4.8秒出弯滑移 {slide:.3f}',
                          font=f2,fill='#39504a')
            sink.append_data(np.asarray(pic))
            if idx in keynotes:
                images[idx]=pic
    montage=Image.new('RGB',(2*W,H*len(keyframes)))
    for i,(_,idx) in enumerate(keyframes):
        montage.paste(images[idx],(0,i*H))
    montage.save(grid)
    out=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                      '-show_entries','stream=nb_read_frames','-of','csv=p=0',str(video)],
                      capture_output=True,text=True,check=True)
    decoded=int(out.stdout.strip())
    assert decoded==len(frames)
    for r,_,p,_ in data:
        assert r['trace_sha256']==stan.sha(p)
    stan.put(output/'VIDEO.json',dict(video=str(video),video_sha256=stan.sha(video),montage=str(grid),
        montage_sha256=stan.sha(grid),decoded_frames=decoded,physical_steps=0,
        milliseconds=frames,paired_runs=[str(a),str(b)],moments=keyframes,renderer_sha256=stan.sha(Path(__file__))))
    print('Decoded',decoded)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('a',type=Path)
    p.add_argument('b',type=Path)
    x=p.parse_args()
    render(x.output.resolve(),x.a.resolve(),x.b.resolve())
