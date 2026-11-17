"""Render only the saved slope positions and actual COM elevation."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from slope import STAND, Terrain, WARM, bench

WIDTH, HEIGHT, CONTENT, FPS = 620, 426, 338, 30
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def run(output, folder):
    result=bench.read(folder/'RESULT.json')
    trace=folder/'TRACE.npz'
    actual=STAND.sha(trace)
    if actual!=result['files']['TRACE.npz']:
        raise ValueError('Retained slope metrics and trajectory identity disagree')
    with np.load(trace,allow_pickle=False) as z:
        q,v,com=(z[name] for name in ('qpos','qvel','com'))
    final=len(q)-1
    if final!=result['actual_ms']:
        raise ValueError('Video and evidence endpoints do not agree')
    output.mkdir()
    corridor=result.get('plan')
    heightfield=result.get('surface_type','').startswith('heightfield')
    if corridor:
        from ramp_course import RAMP, scene, surface

        if heightfield:
            from ramp_hfield import scene as compile_heightfield

            model,_,_,_=compile_heightfield(corridor,result['surface_type']=='heightfield_overlay')
        else:
            model,_,_,_=scene(corridor)
        terrain=None
        h=surface(com[:,0],corridor)
        plane_height=h-h[0]
    else:
        model=STAND.physics.Core(adhesion=False)
        terrain=Terrain(model,result['angle_deg'])
        plane_height=-result['plane_normal'][0]/result['plane_normal'][2]*(com[:,0]-com[0,0])
    d=model.d
    plant=model.plant
    def never(*_args,**_kwargs):
        raise RuntimeError('Only recorded states may be rendered')
    mj.mj_step=mj.mj_step1=mj.mj_step2=never
    elevation=com[:,2]-com[0,2]
    upper=max(float(elevation.max()),float(plane_height.max()))
    lower=min(float(elevation.min()),float(plane_height.min()))
    span=max(upper-lower,1e-6)
    font=[ImageFont.truetype(FONT,size) for size in (23,16,12)]
    w0=result.get('warm_ms',WARM)
    w1=w0+result['walk_ms']
    moments={0,min(250,final),min(w0+1600,final),min(w0+4200,final),min(w0+7600,final),min(w1+600,final),final}
    if corridor:
        entry=result.get('first_com_x0_ms')
        coast=result.get('first_com_x1_ms')
        moments.update(t for t in (entry,coast) if t is not None)
        if entry is not None and coast is not None and coast-entry>100:
            moments.add((entry+coast)//2)
    moments=sorted(moments)
    if len(moments)%2 and final>=2:
        moments=sorted({*moments,max(final-200,1)})
    sample_ms=sorted(set(range(0,final+1,35))|set(moments))
    sample_line=sorted({*range(0,final+1,max(1,final//190)),final})
    def chart_point(timestamp, height):
        return 14+int(145*timestamp/final),179-int(74*(height-lower)/span)

    view=mj.MjvCamera()
    view.type=mj.mjtCamera.mjCAMERA_FREE
    view.azimuth=93.
    view.elevation=-14.
    view.distance=7.2 if corridor else 5.1
    view.lookat[:]=[*q[:,:2].mean(0),.80]

    def frame(tick, renderer):
        d.qpos[:]=q[tick]
        d.qvel[:]=v[tick]
        mj.mj_forward(plant,d)
        np.testing.assert_array_equal(d.qpos,q[tick])
        if terrain is not None:
            terrain.confirm()
        elif not heightfield:
            normal=d.geom_xmat[plant.geom(RAMP).id].reshape(3,3)[:,2]
            np.testing.assert_allclose(normal,result['plane_normal'],rtol=0.,atol=1e-14)
        if corridor:
            view.lookat[0]=.65*q[tick,0]+.35*q[:,0].mean()
        renderer.update_scene(d,camera=view)
        out=Image.new('RGB',(WIDTH,HEIGHT),'#edf1f1')
        out.paste(Image.fromarray(renderer.render()),(0,HEIGHT-CONTENT))
        painter=ImageDraw.Draw(out)
        stage='起始站立' if tick<w0 else '沿坡前行' if tick<w1 else '撤令后'
        if corridor and w0<=tick<w1:
            x=com[tick,0]
            stage='平地接近' if x<corridor['lower_x'] else '通过有限坡面' if x<corridor['upper_x'] else '坡后平地续走'
        if result['angle_deg']==0 and tick>=w0 and tick<w1:
            stage='平地前行'
        label='物理验收通过' if result['passed'] else '本次有验收项未通过'
        painter.text((10,8),f"坡面 {result['angle_deg']:+.1f}°  {stage}  {tick*.001:.3f} s",font=font[0],fill='#20383a')
        painter.text((11,45),f"{label}  前进量 {com[tick,0]-com[0,0]:+.3f}  实际抬升 {elevation[tick]:+.4f} 原生长度",
                     font=font[1],fill='#315055')
        painter.rounded_rectangle((7,94,169,202),radius=7,fill='#e7eeee',outline='#8aaaae')
        painter.line([chart_point(i,plane_height[i]) for i in sample_line], fill='#7d9295',width=2)
        painter.line([chart_point(i,elevation[i]) for i in sample_line], fill='#35848a',width=2)
        x,y=chart_point(tick,elevation[tick])
        painter.ellipse((x-3,y-3,x+3,y+3),fill='#ff694e')
        painter.text((11,182),'质心抬升  ·  灰线为坡面增高',font=font[2],fill='#284347')
        return out

    video=output/'坡面物理回放.mp4'
    montage=output/'坡面关键帧.png'
    sheet=Image.new('RGB',(2*WIDTH,HEIGHT*((len(moments)+1)//2)),'#d1dfdf')
    with mj.Renderer(plant,width=WIDTH,height=CONTENT) as renderer:
        with imageio.get_writer(video,fps=FPS,codec='libx264',quality=8,macro_block_size=2,ffmpeg_log_level='error') as sink:
            for tick in sample_ms:
                picture=frame(tick,renderer)
                sink.append_data(np.asarray(picture))
                if tick in moments:
                    placement=moments.index(tick)
                    sheet.paste(picture,(placement%2*WIDTH,placement//2*HEIGHT))
    sheet.save(montage)
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries',
                          'stream=nb_read_frames','-of','csv=p=0',str(video)],check=True,capture_output=True,text=True)
    assert int(probe.stdout.strip())==len(sample_ms) and STAND.sha(trace)==actual
    STAND.put(output/'VIDEO.json',dict(source=str(folder),trace_sha256=actual,
              video_sha256=STAND.sha(video),montage_sha256=STAND.sha(montage),decoded_frames=len(sample_ms),
              actual_integrations=0,angle_deg=result['angle_deg'],source_ms=sample_ms))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('record',type=Path)
    args=parser.parse_args()
    run(args.output.resolve(),args.record.resolve())
