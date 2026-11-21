"""Held wall challenge for the already-frozen open-field navigator."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

import live
from fruit_field import FRUIT, OdorField, Wall


def run(output, mode, variant):
    wall=Wall()
    field=OdorField(wall)
    apply_scene=live.build_scene
    metrics=dict(wall_contact_steps=0, solid_antenna_read_steps=0,
                 out_of_field_steps=0, thorax_inside_wall_steps=0, first_wall_contact_s=None,
                 max_wall_force_native=0., min_abs_y_at_wall_plane=None)
    row=live.SCENES['left']

    class WalledArena:
        def __init__(self,fruit):
            assert np.array_equal(fruit,FRUIT)

        def smell(self,points):
            try:
                return field.sample(points)
            except ValueError as exc:
                if 'inside a solid wall' in str(exc):
                    metrics['solid_antenna_read_steps']+=1
                elif 'outside modeled arena' in str(exc):
                    metrics['out_of_field_steps']+=1
                else:
                    raise
                return np.zeros(2)

    def build(_):
        body=apply_scene(wall)
        original=body.apply
        wall_id=body.m.geom('fruit_wall').id
        wrench=np.empty(6)

        def apply(*args):
            original(*args)
            contacts=[idx for idx,c in enumerate(body.d.contact[:body.d.ncon])
                      if wall_id in (c.geom1,c.geom2) and not c.exclude]
            metrics['wall_contact_steps']+=bool(contacts)
            if contacts:
                if metrics['first_wall_contact_s'] is None:
                    metrics['first_wall_contact_s']=float(body.d.time)
                for idx in contacts:
                    mj.mj_contactForce(body.m,body.d,idx,wrench)
                    metrics['max_wall_force_native']=max(metrics['max_wall_force_native'],float(wrench[0]))
            x,y,z=body.d.xpos[body.thorax]
            if abs(x-wall.x)<=wall.half_width:
                old=metrics['min_abs_y_at_wall_plane']
                metrics['min_abs_y_at_wall_plane']=float(abs(y)) if old is None else min(old,float(abs(y)))
                metrics['thorax_inside_wall_steps']+=int(abs(y)<=wall.half_length and z<=wall.height)

        body.apply=apply
        return body

    # The same open-field controller operates on physical observations. Only
    # the environment changes. No wall waypoints or wall label enter the head.
    live.SCENES['left']=(2579,tuple(FRUIT[:2]),.025) if variant=='alternate' else (
        row[0],tuple(FRUIT[:2]),row[2])
    live.Arena=WalledArena
    live.build_scene=build
    live.run('left',mode,output,3.5)
    result=json.loads((output/'RESULT.json').read_text())
    with (output/'WALL_RESULT.json').open('x') as stream:
        json.dump(dict(**metrics, body_steps=result['native_body_steps'],
                      actual_intake=result['ingested'], min_oral_distance=result['min_oral_distance'],
                      first_intake_s=result['first_intake_s'], mode=mode, variant=variant,
                      ate_by_detour=bool(result['navigation_accepted']
                           and not metrics['thorax_inside_wall_steps']
                           and (metrics['min_abs_y_at_wall_plane'] or 0.)>wall.half_length),
                      qualified=False, purpose='environment transfer baseline without wall training'),
                  stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--mode',default='intact',choices=('intact','vision_only','odor_only'))
    p.add_argument('--variant',default='primary',choices=('primary','alternate'))
    cfg=p.parse_args()
    run(cfg.output,cfg.mode,cfg.variant)
