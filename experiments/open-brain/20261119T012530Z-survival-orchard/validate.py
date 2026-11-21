"""Audit rendered pixels, compiled contacts, and independent patch stocks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dm_control.mujoco import engine, wrapper

from orchard import (BED, CENTER_X, HALF_X, HALF_Y, RESOURCES, build_orchard, ground_height,
                     FeedingBody, EMPTY)
from orchard_eye import OrchardEyes, FOCAL_PX, WIDTH, HEIGHT
from engine.embodied.feeding import FeedingState

SCANS = (
    ('start_head', None),
    ('start_left', ((.14,0,.24),(1,.32,.02))),
    ('start_right', ((.14,0,.24),(1,-.38,.027))),
    ('tree_near', ((6.95,2.40,.35),(1.4,1.15,.38))),
    ('table_near', ((9.1,-1.55,1.32),(4,-2.8,-.24))),
    ('table_overhead', ((9.95,-4.30,2.78),(1,.01,-.36))),
    ('ground_near', ((2.5,-.73,.30),(1,-.04,-.06))),
    ('sky_only', ((1,0,3.),(1,-.3,1.2))),
    ('under_table', ((13.36,-4.50,.20),(1,.32,0.))),
    ('tree_crossview', ((14.25,8.15,.55),(-1,-.38,.02))),
)


def ray(phy,x,y, z=6.):
    idx=np.full(1,-1,dtype=np.int32)
    distance=mujoco.mj_ray(phy.model.ptr,phy.data.ptr,np.array([x,y,z]),np.array([0.,0.,-1.]),
                           np.array([1,0,0,0,0,0],dtype=np.uint8),True,-1,idx)
    return z-distance, phy.model.id2name(idx[0],'geom') if idx[0]>=0 else ''


def terrain_audit(phy):
    h=ground_height()
    # Compare asymmetric grid nodes with physics rays in a clear region.
    measure=[]
    for row,col in [(64,58),(79,76),(102,90),(124,75),(151,49),(160,108),(192,65),(207,102)]:
        x=CENTER_X-HALF_X+2*HALF_X*col/256
        y=-HALF_Y+2*HALF_Y*row/256
        z,hit=ray(phy,x,y)
        measure.append(dict(x_cm=x,y_cm=y,z_cm=z,geom=hit,expected_cm=float(BED+h[row,col])))
    error=max(abs(row['z_cm']-row['expected_cm']) for row in measure)
    return dict(height_range_mm=((h.max()-h.min())*10).item(), rms_relief_mm=(h.std()*10).item(),
                ray_error_mm=error*10, samples=measure, outside=ray(phy,-7.,0.),
                start=ray(phy,.02,.02), table=ray(phy,15.65,-5.50))


def support(phy,names):
    contacts={name: set() for name in names}
    for c in phy.data.contact:
        a,b = (phy.model.id2name(g,'geom') for g in (c.geom1,c.geom2))
        if a in contacts:
            contacts[a].add(b)
        if b in contacts:
            contacts[b].add(a)
    return contacts


def drop_audit(out):
    env,_=build_orchard()
    world=env.task._arena.mjcf_model.worldbody
    sites=[(r.name,r.x+.26*r.radius,r.y,r.top) for r in RESOURCES]
    sites += [('table_surface',15.65,-5.50,.875), ('away',5.1,1.65,.11)]
    for name,x,y,z in sites:
        probe=world.add('body',name='drop_'+name,pos=(x,y,z+.16))
        probe.add('freejoint')
        probe.add('geom',name='probe_'+name,type='sphere',mass=.000001,size=(.030,),solref=(.0012,1.),
                  group=3,contype=2,conaffinity=1,friction=(.9,.005,.0001),rgba=(.3,.8,.9,1))
    env.reset()
    phy=env.physics
    sites[-1]=(*sites[-1][:3],ray(phy,*sites[-1][1:3])[0])
    probe_names=['probe_'+name for name,_,_,_ in sites]
    watch=['orchard_terrain','groundplane',*probe_names]
    trace=[]
    for tick in range(2800):
        phy.step()
        if tick%10==0:
            hit=support(phy,watch)
            legs={'_'.join(item.removesuffix('_collision').rsplit('_',2)[-2:])
                  for item in hit['orchard_terrain'] | hit['groundplane']
                  if item.startswith('walker/') and ('tarsus' in item or 'claw' in item)}
            nonfeet={item for item in hit['orchard_terrain'] | hit['groundplane']
                     if item.startswith('walker/') and 'tarsus' not in item and 'claw' not in item}
            trace.append([phy.time(),len(legs),len(nonfeet),float(phy.named.data.subtree_com['walker/'][2])])
    hits=support(phy,watch)
    results={}
    for name,x,y,surface in sites:
        geom='probe_'+name
        expected='food_'+name
        if name=='away':
            expected='orchard_terrain'
        if name=='table_surface':
            expected='table_top'
        z=float(phy.named.data.geom_xpos[geom][2])
        results[name]=dict(contact=sorted(hits[geom]),expected=expected,z_cm=z,
                           expected_z_cm=surface+.030,passed=expected in hits[geom] and abs(z-surface-.030)<.004)
    standing=np.asarray(trace)
    np.save(out/'STANDING.npy',standing,allow_pickle=False)
    env.close()
    return dict(probes=results,standing_end=standing[-1].tolist(),
                minimum_feet_last_40ms=float(standing[standing[:,0]>.1,1].min()),
                maximum_nonfoot_last_40ms=float(standing[standing[:,0]>.1,2].max()),
                frames=len(trace),total_native_steps=2800)


def food_audit(bank):
    feeding=FeedingBody()
    state=FeedingState(EMPTY)
    initial={k:f.volume for k,f in bank.stocks.items()}
    moved=0.
    touched=[]
    for r in RESOURCES:
        select=bank.contact((r.x,r.y,r.top+.04),True)
        assert bank.contact((r.x,r.y,r.top+.8),True) is None
        assert bank.contact((r.x,r.y,r.top+.04),False) is None
        assert select == r.name
        result=bank.advance(feeding,state,.2,feed=True,contact=select)
        moved+=result.ingested.volume
        assert result.ingested.volume>0
        touched.append(select)
        state=result.state
    closed=bank.advance(feeding,state,.2,feed=True,contact=None)
    removed=sum(initial[k]-f.volume for k,f in bank.stocks.items())
    return dict(sources=touched,transferred=moved,removed=removed,no_contact_transfer=closed.ingested.volume,
                resource_conservation_error=abs(removed-moved),
                withdrawals={k:initial[k]-f.volume for k,f in bank.stocks.items()})


def inspect_candidates(phy, eye, key, pose, out):
    left,right,items,sectors=eye.sample(pose)
    labels=eye.render(segmentation=True)[0][...,0]
    ids={}
    for i in np.unique(labels):
        if i < 0:
            continue
        name=phy.model.id2name(i,'geom') if i<phy.model.ngeom else ''
        if name.startswith(('food_','rind_')):
            ids[i]=name.split('_',1)[1]
        elif name.startswith('fruit_dimple_'):
            ids[i]=RESOURCES[int(name.split('_')[2])].name
    masks={name:np.isin(labels,[i for i,n in ids.items() if n==name]) for name in sorted(set(ids.values()))}
    ground_truth={}
    for name,mask in sorted(masks.items()):
        y,x=np.nonzero(mask)
        if len(x)<20 or np.ptp(x)<4 or np.ptp(y)<3:
            continue
        ground_truth[name]=int(len(x))
    pretty=Image.new('RGB',(768,496),'#222820')
    pretty.paste(Image.fromarray(left).resize((768,432),Image.Resampling.NEAREST))
    draw=ImageDraw.Draw(pretty)
    colors=['#e4ff9d','#f0a0ff','#a7eff8','#eaffff','#ffb3af','#cad9ff','#ffd982']
    palette={r.name:color for r,color in zip(RESOURCES,colors,strict=True)}
    reports=[]
    associated=set()
    for item in items:
        x0,y0,x1,y1=item.box
        overlap={name:float(mask[y0:y1,x0:x1].sum()/((x1-x0)*(y1-y0))) for name,mask in masks.items()}
        name=max(overlap,key=overlap.get) if overlap and max(overlap.values())>=.28 else None
        color=palette.get(name,'#bdbdbd')
        draw.rectangle((2*x0,2*y0,2*x1,2*y1),outline=color,width=2)
        reports.append(dict(**asdict(item),oracle_match=name,overlap=overlap.get(name)))
        if name is not None:
            associated.add(name)
    font=ImageFont.load_default(size=17)
    legend=sorted(associated)
    if any(r['oracle_match'] is None for r in reports):
        legend.append(None)
    for i,name in enumerate(legend):
        x,y=14+(i%4)*190,442+(i//4)*25
        draw.rectangle((x,y+3,x+11,y+14),fill=palette.get(name,'#bdbdbd'))
        draw.text((x+18,y),(name or 'Unmatched').replace('_',' '),font=font,fill='#f2f2e8')
    pretty.save(out/(key+'_annotated.png'))
    Image.fromarray(left).save(out/(key+'_left.png'))
    Image.fromarray(right).save(out/(key+'_right.png'))
    return dict(visible=ground_truth, detected=reports, missed=sorted(set(ground_truth)-associated),
                duplicates=sum(r['oracle_match'] is not None for r in reports)-len(associated),
                false_positives=sum(r['oracle_match'] is None for r in reports),
                contrast_sectors=sectors, mean_luma=float(left.mean()))


def camera_frame(phy, pose, width=1360, height=880):
    view=engine.MovableCamera(phy,height=height,width=width)
    view.set_pose(*pose)
    opt=wrapper.MjvOption()
    opt.geomgroup[2:]=0
    opt.sitegroup[:]=0
    return view.render(scene_option=opt)


def run(out, tour):
    out.mkdir(parents=True)
    env,bank=build_orchard()
    env.reset()
    phy=env.physics
    eye=OrchardEyes(phy)
    observations={k:inspect_candidates(phy,eye,k,None if p is None else (*p,(0,0,1)),out) for k,p in SCANS}
    camera=dict(head_origin_cm=eye.head_pose()[0].tolist(),head_forward=eye.head_pose()[1].tolist(),
              focal_px=FOCAL_PX, eye_width=WIDTH, eye_height=HEIGHT)
    geology=terrain_audit(phy)
    food=food_audit(bank)
    print('terrain',geology,'food',food,flush=True)
    for name,obs in observations.items():
        print(name,obs['visible'],[i['oracle_match'] for i in obs['detected']],obs['missed'],flush=True)
    for name,pose in {
        'orchard_overview':((8,0,.8),23.,150.,-26.),
        'tree_detail':((10.3,5.25,1.12),8.5,-92.,-17.),
        'table_detail':((13.7,-4.62,.63),8.0,-83.,-35.),
        'fly_start':((.04,0,.08),1.10,-142.,-24.),
    }.items():
        Image.fromarray(camera_frame(phy,pose)).save(out/(name+'.png'))
    if tour:
        frames=out/'tour'
        frames.mkdir()
        # A presentation camera moving around the single scene snapshot.
        cam=engine.MovableCamera(phy,height=720,width=1120)
        opt=wrapper.MjvOption()
        opt.geomgroup[2:]=0
        opt.sitegroup[:]=0
        for i,t in enumerate(np.linspace(0,1,120)):
            cam.set_pose((7.5+.7*t,.30,.75),23-2*np.sin(np.pi*t),147+57*t,-23-10*np.sin(np.pi*t))
            Image.fromarray(cam.render(scene_option=opt)).save(frames/f'{i:04}.jpg',quality=92)
    env.reset()
    food['new_episode_restores_stocks']=all(bank.stocks[r.name].volume==r.volume for r in RESOURCES)
    env.close()
    collisions=drop_audit(out)
    names={r['oracle_match'] for obs in observations.values() for r in obs['detected']}
    gates={
        'terrain_rays_and_nonzero_relief': geology['height_range_mm']>1. and geology['ray_error_mm']<.045
            and all(row['geom']=='orchard_terrain' for row in geology['samples']),
        'support_surfaces_collide':all(r['passed'] for r in collisions['probes'].values()),
        'fly_spawn_stands_in_final_window':collisions['minimum_feet_last_40ms']==6
            and collisions['maximum_nonfoot_last_40ms']==0,
        'all_fruits_detected_by_rgb_across_scripted_views':names.issuperset(r.name for r in RESOURCES),
        'same_view_separates_all_three_table_fruits':any(
            {r['oracle_match'] for r in observations[k]['detected']}.issuperset(
                r.name for r in RESOURCES if r.support=='table') for k in ('table_near','table_overhead')),
        'background_no_false_target_in_sky_and_under_table': all(not observations[k]['detected']
                                                                 for k in ('sky_only','under_table')),
        'distinct_food_stocks_and_contact_gate': len(food['sources'])==7 and food['resource_conservation_error']<1e-12
            and food['no_contact_transfer']==0. and food['new_episode_restores_stocks'],
    }
    total_visible=sum(len(obs['visible']) for obs in observations.values())
    metrics=dict(eligible_instances=total_visible,
                 recalled_instances=total_visible-sum(len(obs['missed']) for obs in observations.values()),
                 false_positive_candidates=sum(obs['false_positives'] for obs in observations.values()),
                 duplicate_candidates=sum(obs['duplicates'] for obs in observations.values()),
                 stereo_candidates=sum(item['depth_cm'] is not None for obs in observations.values()
                                       for item in obs['detected']))
    facts=dict(status='SCENE_FIXTURE_VERIFIED' if all(gates.values()) else 'SCENE_FIXTURE_PARTIAL',
               gates=gates, terrain=geology, food=food, camera=camera, views=observations, vision_metrics=metrics, contact=collisions)
    (out/'VALIDATION.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
    print(facts['status'],gates,flush=True)
    return all(gates.values())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--tour',action='store_true')
    args=parser.parse_args()
    raise SystemExit(0 if run(args.output,args.tour) else 1)
