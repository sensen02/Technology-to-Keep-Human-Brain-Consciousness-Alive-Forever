"""Label the actual post-meal eye views from the saved trajectory."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'20261119T012530Z-survival-orchard'))
from orchard import RESOURCES, build_orchard
from orchard_eye import OrchardEyes


def audit(trial, out):
    env, _ = build_orchard()
    env.random_state.seed(29)
    env.reset()
    phy = env.physics
    eye = OrchardEyes(phy)
    poses = np.load(trial/'QPOS.npy', allow_pickle=False)
    trace = np.load(trial/'TRACE.npy', allow_pickle=False)
    meal = np.load(trial/'MEAL.npy', allow_pickle=False)
    rows = [json.loads(line) for line in (trial/'EYES.jsonl').read_text().splitlines()]
    last_meal = float(meal[meal[:,2]>0.,0][-1])
    rows = [row for row in rows if row['time_s'] >= last_meal]
    geomsets = {res.name:[] for res in RESOURCES}
    for index in range(phy.model.ngeom):
        name = phy.model.id2name(index,'geom') or ''
        if name.startswith(('food_','rind_')) and name.split('_',1)[1] in geomsets:
            geomsets[name.split('_',1)[1]].append(index)
        if name.startswith('fruit_dimple_'):
            geomsets[RESOURCES[int(name.split('_')[2])].name].append(index)
        if name.startswith('calyx_'):
            geomsets[name.removeprefix('calyx_').rsplit('_',1)[0]].append(index)
    visible = {res.name:[0,0] for res in RESOURCES}
    max_pixels = {res.name:[0,0] for res in RESOURCES}
    choices, candidates = Counter(), Counter()
    head_yaw = []
    selected_samples = {0,len(rows)//2,len(rows)-1}
    frames = []
    for i,row in enumerate(rows):
        j = round(row['time_s']/.005)-1
        phy.data.qpos[:] = poses[j]
        phy.forward()
        spot,forward,up = eye.head_pose()
        head_yaw.append(float(np.arctan2(forward[1],forward[0])))
        eye.pose(spot,forward,up)
        lids,rids = (im[...,0] for im in eye.render(segmentation=True))
        masks = {name:np.isin(lids,ids) for name,ids in geomsets.items()}
        for name,ids in geomsets.items():
            counts = [int(masks[name].sum()),int(np.isin(rids,ids).sum())]
            for side,pixels in enumerate(counts):
                visible[name][side] += pixels>0
                max_pixels[name][side] = max(max_pixels[name][side],pixels)

        def associate(item):
            x0,y0,x1,y1 = item['box']
            fractions = {name:float(mask[y0:y1,x0:x1].mean()) for name,mask in masks.items()}
            best = max(fractions,key=fractions.get)
            return best if fractions[best]>=.28 else 'unmatched'

        chosen = row['selection']
        choices.update([associate(chosen) if chosen else 'none'])
        candidates.update(associate(item) for item in row['candidates'])
        if i in selected_samples:
            left,right,items,_ = eye.sample()
            assert sorted(tuple(item['box']) for item in row['candidates']) == sorted(item.box for item in items)
            second = [item for item in row['candidates'] if associate(item)=='ground_far']
            frames.append((row['time_s'],left,right,chosen,second))
    active = trace[trace[:,0]>=last_meal]
    yaw = np.rad2deg(np.unwrap(active[:,11]))
    head = np.rad2deg(np.unwrap(head_yaw))
    n = np.load(trial/'NEURAL.npz',allow_pickle=False)
    closed = n['head'][n['head'][:,0]>=last_meal,4].astype(bool)
    walk = n['ground'][round(last_meal/.0001):]
    audit = dict(samples=len(rows), first_sample_s=rows[0]['time_s'], last_sample_s=rows[-1]['time_s'],
                 visible_pixel_frames_left_right=visible,max_visible_pixel_count_left_right=max_pixels,
                 selected=dict(choices),all_left_eye_candidates=dict(candidates),
                 body_yaw_range_deg=[float(yaw.min()),float(yaw.max())],
                 body_yaw_net_deg=float(yaw[-1]-yaw[0]),head_yaw_swing_deg=float(np.ptp(head)),
                 max_planar_movement_mm=float(10*np.linalg.norm(active[:,1:3]-active[0,1:3],axis=1).max()),
                 postmeal_walk_gate_open_packets=int(closed.sum()),
                 post_neck_walk_peak=float(walk.max()))
    (out/'VISIBILITY.json').write_text(json.dumps(audit,indent=2)+'\n')
    montage=Image.new('RGB',(768,249*len(frames)),'#172427')
    draw=ImageDraw.Draw(montage)
    for i,(t,left,right,chosen,second) in enumerate(frames):
        top=249*i
        draw.text((7,top+8),f'{t:.2f}s  |  Left: empty fruit (green), second ground fruit (blue)     Right',
                  fill='#edf1ed')
        montage.paste(Image.fromarray(left),(0,top+27))
        montage.paste(Image.fromarray(right),(384,top+27))
        if chosen:
            x0,y0,x1,y1 = chosen['box']
            draw.rectangle((x0,y0+top+27,x1-1,y1+top+26),outline='#80f6b0',width=2)
        for item in second:
            x0,y0,x1,y1=item['box']
            draw.rectangle((x0,y0+top+27,x1-1,y1+top+26),outline='#6be3ff',width=2)
    montage.save(out/'POSTMEAL_EYES.png')
    print(json.dumps(audit,indent=2),flush=True)
    env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    args=parser.parse_args()
    audit(args.trial,HERE)
