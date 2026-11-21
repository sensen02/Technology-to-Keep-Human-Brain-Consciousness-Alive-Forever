"""Audit actual neck-spike timing and sensory transfer on the matched live courses."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from forage import Course, MotorTable, Terrain
from policy_runtime import decode_policy


def load_policy(path):
    with np.load(path,allow_pickle=False) as canonical:
        return {k:canonical[k] for k in canonical.files}


def transfer(folder,cal):
    result=json.loads((folder/'RESULT.json').read_text())
    with np.load(folder/'LIVE.npz',allow_pickle=False) as saved:
        trace,obs,meals,frames=(saved[k] for k in ('trace','observation','meals','visual'))
    with np.load(folder/'NECK.npz',allow_pickle=False) as saved:
        emitted,arrived,delays=(saved[k] for k in ('emitted','arrived','delays'))
        assert saved['weights'].item()==result['weights']
        for i,d in enumerate(delays):
            assert not arrived[:d,i].any()
            assert np.array_equal(arrived[d:,i],emitted[:-d,i])
        drive=saved['drive']
        ascend=int(saved['ascending'].sum())
    assert result['intake']==meals[-1,1] and result['extra_force']==0.
    assert np.isfinite(trace).all() and np.isfinite(obs).all()
    gp=Path(result['checkpoint'])
    saved=load_policy(gp)
    info=json.loads(saved['identity'].item())
    assert info['brain_sha256']==result['weights']
    motor=MotorTable(cal['trials'][str(info['gain'])]['rows'])
    env=Course(motor)
    tf=Terrain(**result['terrain'])
    env.reset(result['seed'],result['level'],side=tf.side)
    env.terrain=tf
    yaw_errors=[]
    onset=[]
    approximate=[]
    for row in obs:
        t=row[0]
        idx=round(t/.005)-1
        xyz=trace[idx,:3]
        w,x,y,z=trace[idx,3:7]
        heading=np.arctan2(2.*(w*z+x*y),1.-2.*(y*y+z*z))
        env.position=xyz[:2].copy()
        env.heading=heading
        approx=env.observe()[4:10]
        if xyz[0]<tf.near:
            onset.append(float(row[5:11].max()))
            approximate.append(float(approx.max()))
        if t+.13>len(drive)*.0001:
            continue
        _,(p,y)=decode_policy(saved,row[1:15].astype(np.float32))
        _,expected_yaw=motor.drive(np.array([2.*p/.76-1.,y/.22]))
        a,b=np.mean(drive[round((t+.075)/.0001):round((t+.13)/.0001)],axis=0)
        actual_yaw=np.clip(3.25*(a-b),-2.6,2.6)
        yaw_errors.append(float(actual_yaw-expected_yaw))
    facts=result | dict(neural_ticks=len(drive),dn_emitted=int(emitted.sum()),dn_arrived=int(arrived.sum()),
                        an_arrived=ascend,all_dn_delays_exact=True,
                        max_advance_mm=float(10*trace[:,0].max()),
                        live_vs_table_yaw_abs_mean_rad_s=float(np.abs(yaw_errors).mean()),
                        pre_gap_live_hazard_max=max(onset,default=0.),
                        pre_gap_proxy_hazard_max=max(approximate,default=0.))
    return facts,frames


def table_image(frames,path):
    names=list(frames)
    w,h=270,152
    columns=4
    image=Image.new('RGB',(columns*w,len(names)*(h+26)),'#19252b')
    draw=ImageDraw.Draw(image)
    for i,key in enumerate(names):
        views=frames[key]
        for j,index in enumerate(np.linspace(0,len(views)-1,columns,dtype=int)):
            base=(j*w,i*(h+26))
            img=Image.fromarray(views[index]).resize((w,h))
            image.paste(img,(base[0],base[1]+25))
            draw.text((base[0]+7,base[1]+6),key if j==0 else f'camera sample {index+1}',fill='white')
    image.save(path)


def main(root):
    cal=json.loads((root/'NEURAL_CAL/CALIBRATION.json').read_text())
    entries={}
    images={}
    for name in ('LIVE_FROZEN_FLAT','LIVE_TUNED_FLAT','LIVE_FROZEN_GAP','LIVE_TUNED_GAP'):
        if (root/name/'RESULT.json').is_file():
            entries[name],images[name]=transfer(root/name,cal)
    (root/'TRANSFER_AUDIT.json').write_text(json.dumps(entries,indent=2,allow_nan=False)+'\n')
    table_image(images,root/'LIVE_EYES.png')
    print(json.dumps(entries,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    main(p.parse_args().output)
