"""Render same-time physical restart and hold controls without integration."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from run_restart import ORIGINAL,RELEASE,WALK_MS,read,stan,W,RF
from geometry import make_gap_model

WIDTH,HEIGHT,VISUAL,FPS=620,378,308,30
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def load(path,caption):
    info=read(path/'RESULT.json')
    p=path/'TRACE.npz'
    assert info['trace_sha256']==stan.sha(p)
    with np.load(p,allow_pickle=False) as z:
        saved={k:z[k] for k in ('qpos','qvel','loads','rf_pad_xyz','com','renewed_propulsion')}
    model=make_gap_model(info['scenario'])
    model.geom_friction[:,0]=stan.physics.rc.MU
    return dict(info=info,saved=saved,trace=p,caption=caption,model=model,data=mj.MjData(model),path=path)


def camera(x,y,z,zoom=False):
    c=mj.MjvCamera()
    c.type=mj.mjtCamera.mjCAMERA_FREE
    c.azimuth=245 if zoom else 125
    c.elevation=-37 if zoom else -11
    c.distance=1.52 if zoom else 5.7
    c.lookat[:]=[x,y,z]
    return c


def render(output,held,active):
    output.mkdir()
    def forbidden(*a,**kw):
        raise RuntimeError('Retained-frame rendering may not integrate dynamics')
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    trials=(held,active)
    final=active['info']['actual_ms']
    shape=active['info']['scenario']
    ax,bx,ay,by=shape['opening']
    zoom=camera((ax+bx)*.5,(ay+by)*.5+.11,.17,zoom=True)
    common_center=np.concatenate([t['saved']['qpos'][:,:2] for t in trials],axis=0).mean(0)
    wide=camera(*common_center,.82)
    loss=active['info']['local_stop_original']['first_pad_drop_ms']
    reload=active['info']['local_stop_original']['first_upper_floor_reload_ms']
    complete_step=active['info']['restart']['first_resume_latency_ms']
    first=RELEASE+int(round(complete_step)) if complete_step is not None else RELEASE+400
    moments=[('同一回合右前足失载',loss),('同一回合重新承重',reload),
             ('同一回合站稳',RELEASE-125),('右侧续行意图已交付',RELEASE+26),
             ('右侧首个完整换步时刻',first),('两侧末段复验',final)]
    indices=sorted(set(range(0,final+1,20)) | set(range(loss-55,reload+80,5)) |
                   set(range(RELEASE-80,min(first+90,final),10)) | {v for _,v in moments})
    fonts=(ImageFont.truetype(FONT,21),ImageFont.truetype(FONT,16))
    def frame(ms,note=None,z_detail=False):
        img=Image.new('RGB',(WIDTH*2,HEIGHT),(241,244,243))
        draw=ImageDraw.Draw(img)
        for col,trial in enumerate(trials):
            data=trial['data']
            saved=trial['saved']
            data.qpos[:]=saved['qpos'][ms]
            data.qvel[:]=saved['qvel'][ms]
            mj.mj_forward(trial['model'],data)
            assert np.array_equal(data.qpos,saved['qpos'][ms])
            ren=trial['renderer']
            view=zoom if z_detail else wide
            ren.update_scene(data,camera=view)
            pic=Image.fromarray(ren.render())
            x=col*WIDTH
            img.paste(pic,(x,HEIGHT-VISUAL))
            phase=('初始站立' if ms<500 else '走向缺口' if ms<loss else '踏空制动' if ms<reload else '复踏及站稳'
                   if ms<RELEASE else '维持站立' if col==0 else '重新行走' if ms<RELEASE+WALK_MS else '再次停稳')
            fz=saved['loads'][ms,RF]/W
            dx=saved['com'][ms,0]-saved['com'][RELEASE,0]
            draw.text((x+10,7),f"{trial['caption']} · {note or phase}",font=fonts[0],fill='#243633')
            draw.text((x+10,41),f'{ms/1000:.3f} s · 右前足支撑 {fz:.3f} W · 续行窗口相对 x {dx:+.3f}',
                      font=fonts[1],fill='#40504d')
        return img
    video=output/'踏空后再起步配对.mp4'
    montage=output/'再起步关键帧.png'
    with ExitStack() as stack:
        for t in trials:
            t['renderer']=stack.enter_context(mj.Renderer(t['model'],height=VISUAL,width=WIDTH))
        sink=stack.enter_context(imageio.get_writer(video, fps=FPS,codec='libx264',quality=8,
                                                    macro_block_size=2,ffmpeg_log_level='error'))
        for ms in indices:
            sink.append_data(np.asarray(frame(ms,z_detail=loss-60<=ms<=reload+120)))
        sheet=Image.new('RGB',(2*WIDTH,len(moments)*HEIGHT))
        for i,(note,ms) in enumerate(moments):
            sheet.paste(frame(ms,note,z_detail=(ms in (loss,reload))),(0,i*HEIGHT))
        sheet.save(montage)
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                           '-show_entries','stream=nb_read_frames','-of','csv=p=0',str(video)],
                            check=True,capture_output=True,text=True)
    decoded=int(probe.stdout.strip())
    assert decoded==len(indices)
    for t in trials:
        assert stan.sha(t['trace'])==t['info']['trace_sha256']
    stan.put(output/'VIDEO.json',dict(scope='same-event AS gap recovery and direct renewed propulsion',
             actual_integrations=0,video=str(video),video_sha256=stan.sha(video),montage=str(montage),
             montage_sha256=stan.sha(montage),renderer_sha256=stan.sha(Path(__file__)),decoded_frames=decoded,
             milliseconds=indices,moments=moments,
             runs=[dict(path=str(t['path']),physical_sha256=t['info']['trace_sha256']) for t in trials]))
    print('decoded',decoded)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('output',type=Path)
    a.add_argument('hold',type=Path)
    a.add_argument('release',type=Path)
    p=a.parse_args()
    render(p.output.resolve(),load(p.hold.resolve(),'停稳后保持'),load(p.release.resolve(),'停稳后续行'))
