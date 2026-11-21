"""Audit saved stands and render representative physical frames."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from body_righting_stow import RightingController
from fixture import HERE,Plant
sys.path.insert(0,str(HERE.parent/'20261119T012530Z-survival-orchard'))
from validate import camera_frame


def check(folder,p,ctrl):
    expected=np.load(folder/'TRACE.npy')
    qpos=np.load(folder/'QPOS.npy')
    initial=np.load(folder/'INITIAL_STATE.npy')
    a=np.load(folder/'ACTIONS.npz')
    meta=json.loads((folder/'RESULT.json').read_text())
    p.put(initial)
    ctrl.trace,ctrl.poses,ctrl.actions=[],[],[]
    ctrl.t0=p.data.time
    ctrl.observe(0)
    for hold,phase,command in zip(a['hold'],a['phase'],a['commands'],strict=True):
        if phase==4:
            ctrl.stow(hold)
        else:
            ctrl.execute(command,hold,int(phase))
    count=len(ctrl.trace)
    same_t=bool(np.array_equal(ctrl.trace,expected[:count]))
    same_q=bool(np.array_equal(ctrl.poses,qpos[:count]))
    assert same_t and same_q,folder
    assert ctrl.steady(ctrl.trace)==meta['passed']
    t=np.array(ctrl.trace)
    return dict(same_trace=same_t,same_qpos=same_q,stand=meta['passed'],wing_error_deg=float(
                 np.rad2deg(t[-100:,10].max())),initial_tilt=meta['initial_tilt_deg'],
                 time_s=meta['upright_confirmed_s'],loaded_stance_feet=int(t[-1,7]),
                 handoff=meta['handoff'])


def render(banks,out,p):
    examples=('test_shallow_left','test_shallow_right','test_dorsal','orchard_fall')
    labels={'test_shallow_left':'左侧卧','test_shallow_right':'右侧卧',
            'test_dorsal':'近180°仰卧','orchard_fall':'果园真实倒地姿态'}
    w,h=400,225
    image=Image.new('RGB',(w*3,(h+30)*len(examples)),'#17272a')
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    for row,name in enumerate(examples):
        folder=next(path/name for path in banks if (path/name/'RESULT.json').exists())
        meta=json.loads((folder/'RESULT.json').read_text())
        qpos=np.load(folder/'QPOS.npy')
        stand=meta['upright_confirmed_s']
        times=(0.,.036,stand if stand else .42)
        for index,t in enumerate(times):
            p.data.qpos[:]=qpos[round(t/.002)]
            p.phy.forward()
            center=p.data.subtree_com[p.th].copy()
            im=Image.fromarray(camera_frame(p.phy,(center,1.20,139,-22),w,h))
            x,y=index*w,row*(h+30)
            image.paste(im,(x,y+29))
            draw.text((x+10,y+5),f'{labels[name]}  ·  {t:.3f}s',font=font,fill='white')
    image.save(out)


def main(banks, output):
    p=Plant()
    c=RightingController(p,2)
    audits={}
    for bank in banks:
        for f in sorted(bank.iterdir()):
            if (f/'RESULT.json').is_file():
                audits[f.name]=check(f,p,c)
    (output/'STANCE_AUDIT.json').write_text(json.dumps(audits,indent=2)+'\n')
    render(banks,output/'RECOVERY_FRAMES.png',p)
    c.pool.shutdown()
    p.env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('banks',nargs='+')
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    main([HERE/name for name in args.banks],args.output)
