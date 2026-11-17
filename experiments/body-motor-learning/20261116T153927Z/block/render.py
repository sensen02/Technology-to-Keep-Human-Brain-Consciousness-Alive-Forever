"""Render the saved native-run poses without advancing the simulation."""
import argparse
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rollout import build_scene, stan, W, LEGS

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
WIDTH, PICTURE, HEIGHT, FPS = 630, 322, 394, 30
TITLES = {'aware':'实心低障碍 · 教师规划落足', 'blind':'实心低障碍 · 无地形规划',
          'flat':'同一波步 · 平地对照', 'stock':'实心低障碍 · 既有步态',
          'reflex':'实心低障碍 · 既有收腿模型', 'tactile':'实体碰撞 · 触觉收腿教师'}


def load(path):
    result = json.loads((path/'RESULT.json').read_text())
    file = path/'TRACE.npz'
    if result['files']['TRACE.npz'] != stan.sha(file):
        raise RuntimeError('Trace artifact changed')
    with np.load(file,allow_pickle=False) as data:
        trace = {key:data[key].copy() for key in ('qpos','qvel','com','block_native','pad_min_x')}
    return result, trace, file


def event_frames(result, until):
    event = result['block_passage']
    times = {150:'初始稳站', until:'末端姿态'}
    for label, values, offset in (('侧面触碰', event['first_box_face_native_step'], 1),
                                  ('踏实块面', event['first_box_top_native_step'], 5)):
        for leg, step in values.items():
            if step is not None and step//80+offset < until:
                times[step//80+offset] = leg + ' ' + label
    for trigger in result.get('triggered',[]):
        moment = int(round(trigger['relative_s'] * 1000)) + result['plan']['warm_ms']
        if moment+120 < until:
            times[moment+120] = trigger['leg']+' 收腿加抬'
    hind = [event['first_floor_reload_beyond_ms'][leg] for leg in ('RH','LH')]
    for label, tick in (('后足回地', max(hind) if all(x is not None for x in hind) else None),
                        ('全部足端过尾', event['continued_ground_only_ms'])):
        if tick is not None and tick < until:
            times[tick] = label
    return times


def render(output, paths, trunk_ms=None):
    output.mkdir()
    reports = [load(path) for path in paths]
    until = min(row[0]['actual_ms'] for row in reports)
    scenes = [build_scene(None if r['mode']=='flat' else r['plan'])[0] for r,_,_ in reports]
    notes = event_frames(reports[0][0], until)
    if trunk_ms is not None and 0 < trunk_ms < until:
        notes[trunk_ms] = "头胸腹过尾缘"
    f1,f2 = [ImageFont.truetype(FONT,size) for size in (22,17)]
    frames=sorted(set(range(0,until+1, max(20,(until//850//5)*5))) | set(notes))
    stored = {}
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 128.
    cam.elevation = -18.
    cam.distance = 5.15
    file = output/'实心低障碍物理复演.mp4'
    collage = output/'实心低障碍关键帧.png'
    def forbid(*args, **kwargs):
        raise RuntimeError('Video renderer must never advance native physics')
    mj.mj_step = mj.mj_step1 = mj.mj_step2 = forbid
    renderers = [mj.Renderer(c.plant,height=PICTURE,width=WIDTH) for c in scenes]
    try:
        with imageio.get_writer(file,fps=FPS,codec='libx264',quality=8,
                macro_block_size=2,ffmpeg_log_level='error') as sink:
            for t in frames:
                pic=Image.new('RGB',(WIDTH*len(scenes),HEIGHT),'#ecf0ee')
                d=ImageDraw.Draw(pic)
                for i,(c,renderer,(r,z,_)) in enumerate(zip(scenes,renderers,reports,strict=True)):
                    c.d.qpos[:] = z['qpos'][t]
                    c.d.qvel[:] = z['qvel'][t]
                    mj.mj_forward(c.plant,c.d)
                    np.testing.assert_array_equal(c.d.qpos,z['qpos'][t])
                    block = r['plan']['center']
                    torso = reports[0][1]['com'][t,0] + .89
                    blend = np.clip((torso-2.) / 1.8,0.,1.)
                    cam.lookat[:] = [torso-blend*.37*(torso-block[0]), -0.17, .69]
                    cam.distance = 5.15 + 1.25*blend
                    renderer.update_scene(c.d,camera=cam)
                    x = i * WIDTH
                    pic.paste(Image.fromarray(renderer.render()),(x,HEIGHT-PICTURE))
                    d.text((x+10,4),TITLES[r['mode']] + ((' · '+notes[t]) if i==0 and t in notes else ''),font=f1,fill='#20362d')
                    loaded = ' '.join(leg for leg,v in zip(LEGS,z['block_native'][t,:6],strict=True) if v>=.02*W)
                    rear = z['pad_min_x'][t].min()-block[0]-r['plan']['half_size'][0]
                    d.text((x+10,39),f'{t*.001:.3f} s  ·  块面承重 {loaded or "—"}  ·  最后足端至块尾 {rear:+.2f}',
                           font=f2,fill='#395046')
                sink.append_data(np.asarray(pic))
                if t in notes:
                    stored[t] = pic
    finally:
        for renderer in renderers:
            renderer.close()
    montage = Image.new('RGB',(WIDTH*len(scenes),HEIGHT*len(stored)))
    for i,idx in enumerate(sorted(stored)):
        montage.paste(stored[idx],(0,HEIGHT*i))
    montage.save(collage)
    p=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                      '-show_entries','stream=nb_read_frames','-of','csv=p=0',str(file)],
                      check=True,capture_output=True,text=True)
    decoded=int(p.stdout.strip())
    if decoded != len(frames):
        raise RuntimeError('Decoded video length changed')
    for result,_,trace in reports:
        assert result['files']['TRACE.npz'] == stan.sha(trace)
    stan.put(output/'VIDEO.json',dict(native_steps=0,decoded_frames=decoded,ms=frames,
             video_path=str(file),video_sha256=stan.sha(file),grid_path=str(collage),grid_sha256=stan.sha(collage),
             replay_paths=list(map(str,paths)),trace_sha256=[r['files']['TRACE.npz'] for r,_,_ in reports]))
    print('decoded',decoded,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('traces',type=Path,nargs='+')
    parser.add_argument('--trunk-ms',type=int)
    a=parser.parse_args()
    render(a.output.resolve(),[x.resolve() for x in a.traces],a.trunk_ms)
