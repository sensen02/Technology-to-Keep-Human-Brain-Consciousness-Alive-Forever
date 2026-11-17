"""Check matched block, flat, and repeat evidence without taking simulation steps."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from rollout import build_scene, consecutive, event, stan, W, LEGS, OBSTACLE


def read(path):
    return json.loads(path.read_text())


def verified_run(path):
    result = read(path/'RESULT.json')
    if not result['complete'] or result['probe']:
        raise ValueError('Full course evidence required')
    for file, checksum in result['files'].items():
        if stan.sha(path/file) != checksum:
            raise ValueError('Output digest mismatch: '+str(path/file))
    with np.load(path/'TRACE.npz',allow_pickle=False) as z:
        trace = {k:z[k].copy() for k in ('qpos','qvel','com','padz','pad_xyz','pad_min_x','loads','block_native','time')}
    with np.load(path/'DECISIONS.npz',allow_pickle=False) as z:
        action, command = z['action'].copy(), z['command'].copy()
        np.testing.assert_array_equal(z['ms'], np.arange(0,result['actual_ms'],5))
        assert action.shape == (len(command),42) and np.isfinite(action).all() and np.abs(action).max() <= 1.
    with np.load(path/'NATIVE.npz',allow_pickle=False) as z:
        n_step = result['native_integrations']
        force = z['forces']
        b = z['block']
        pose = z['poses']
        assert len(force) == len(b) == len(pose) == n_step == 80*result['actual_ms']
        assert not np.any(pose[:,7])
        reclassified=event(trace,b,result['plan'],result['mode']!='flat',result['plan']['warm_ms']+result['plan']['course_ms'])
        assert reclassified == result['block_passage']
        raw_stand = {}
        warm = result['plan']['warm_ms']
        for phase, stop in (('initial',warm),('terminal',result['actual_ms'])):
            rows=force[(stop-200)*80:stop*80]
            post=pose[(stop-200)*80:stop*80]
            raw_stand[phase]=dict(
                support_W=float(rows[:,:6].sum(1).mean()/W),
                nonfoot_peak_W=float(rows[:,12].max()/W),
                stable_fraction_3_or_more_feet=float(((rows[:,:6]>=.02*W).sum(1)>=3).mean()),
                upright_at_least=float(post[:,5].min()), min_com=float(post[:,2].min()),
            )
        joint=result['native_engineering']
        native_begin=warm*80
        native_end=(warm+result['plan']['course_ms'])*80
        selected=force[native_begin:native_end]
        means=selected[:,6:12].sum(0)/selected[:,:6].sum(0)
        np.testing.assert_allclose(means,list(joint['absolute_slide_native_s'].values()),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(means.max()/joint['world_native_s'],joint['raw_slip_ratio'],rtol=1e-12,atol=1e-12)
    return result,trace,action,raw_stand


def actual_trunk_mesh_clearance(c, trace, xmax):
    names=('Head','Thorax','A1A2','A3','A4','A5','A6')
    vertices=[]
    for name in names:
        geom=c.plant.geom(name+'_geom')
        mesh_id=int(geom.dataid[0])
        start=int(c.plant.mesh_vertadr[mesh_id])
        size=int(c.plant.mesh_vertnum[mesh_id])
        vertices.append((geom.id,np.asarray(c.plant.mesh_vert[start:start+size],dtype=float).copy()))
    ms=np.arange(0,len(trace['qpos']),5)
    rear_x, min_z = np.empty(len(ms)), np.empty(len(ms))
    for i,t in enumerate(ms):
        c.d.qpos[:]=trace['qpos'][t]
        mj.mj_kinematics(c.plant,c.d)
        bounds = np.asarray([(mesh @ c.d.geom_xmat[g].reshape(3,3)[[0,2]].T
                              + c.d.geom_xpos[g,[0,2]]).min(0) for g,mesh in vertices])
        rear_x[i], min_z[i] = bounds.min(0)
    past=consecutive(rear_x>xmax+.01,40)
    return dict(physical_mesh_sections=list(names), every_5_ms=len(ms), first_stayed_past_ms=None if past is None else int(ms[past]),
                clearance_at_last_ms_native=float(rear_x[-1]-xmax),
                minimum_trunk_mesh_z_native=float(min_z.min()))


def whole_mesh_footprint(c, trace, plan):
    lower=np.array(plan['center'][:2])-plan['half_size'][:2]
    upper=np.array(plan['center'][:2])+plan['half_size'][:2]
    vertices=[]
    for g in range(c.plant.ngeom):
        mesh_id=c.plant.geom_dataid[g]
        if mesh_id<0 or c.plant.geom_type[g]!=mj.mjtGeom.mjGEOM_MESH:
            continue
        start=c.plant.mesh_vertadr[mesh_id]
        size=c.plant.mesh_vertnum[mesh_id]
        vertices.append((g,np.asarray(c.plant.mesh_vert[start:start+size],dtype=float)))
    assert len(vertices)==71
    separation=[]
    wings={}
    for q in trace['qpos'][-201::5]:
        c.d.qpos[:]=q
        mj.mj_kinematics(c.plant,c.d)
        for g,mesh in vertices:
            xy=mesh@c.d.geom_xmat[g].reshape(3,3)[:2].T+c.d.geom_xpos[g,:2]
            lo,hi=xy.min(0),xy.max(0)
            separation.append(float(np.r_[lower-hi,lo-upper].max()))
            name=c.plant.geom(g).name
            if name in ('LWing_geom','RWing_geom'):
                wings[name]={'rear_x':float(lo[0]),'lower_y':float(lo[1])}
    return dict(mesh_count=len(vertices), final_window_ms=200, stride_ms=5,
                minimum_disjoint_AABB_axis_margin_native=float(min(separation)),
                far_plane_x=float(upper[0]), near_body_side_y=float(upper[1]),
                wings_final_pose=wings)

def qualify(output, aware, blind, flat, repeat):
    a, at, aa, ast=verified_run(aware)
    b, bt, ba, bst=verified_run(blind)
    f, ft, fa, fst=verified_run(flat)
    r, rt, ra, rst=verified_run(repeat)
    assert a['mode']==r['mode']=='aware' and b['mode']=='blind' and f['mode']=='flat'
    assert a['plan']==b['plan']==f['plan']==r['plan']
    warm=a['plan']['warm_ms']
    for pair in (bt,ft,rt):
        for name in ('qpos','qvel','com','loads','pad_xyz'):
            np.testing.assert_array_equal(at[name][:warm+1],pair[name][:warm+1])
    for data in (ba,fa,ra):
        np.testing.assert_array_equal(aa[:warm//5],data[:warm//5])
    assert r['independent_repeat_bitwise'] and r['files']==a['files']
    assert not ft['block_native'].any()
    matched_prefix=np.flatnonzero(np.any(aa!=ba,axis=1))
    earliest=int(matched_prefix[0]*5) if len(matched_prefix) else None
    c,_,_=build_scene(a['plan'])
    g=c.plant.geom(OBSTACLE)
    np.testing.assert_array_equal(c.plant.geom_size[g.id],a['plan']['half_size'])
    np.testing.assert_array_equal(c.plant.geom_pos[g.id],a['plan']['center'])
    assert int(g.bodyid[0])==0 and int(g.type[0])==int(mj.mjtGeom.mjGEOM_BOX) and g.contype[0] and g.conaffinity[0]
    assert c.plant.nmesh==71 and c.plant.nu==42 and c.plant.opt.timestep==12.5e-6 and not c.adhesion_on
    xmax=a['plan']['center'][0]+a['plan']['half_size'][0]
    mesh=actual_trunk_mesh_clearance(c,at,xmax)
    block=a['block_passage']
    onset=block['continued_ground_only_ms']
    walk_end=a['plan']['warm_ms']+a['plan']['course_ms']
    through=None if onset is None else float(at['com'][walk_end,0]-at['com'][onset,0])
    def good_stand(s):
        return (s['support_W']>=.9 and s['nonfoot_peak_W']<.01 and s['stable_fraction_3_or_more_feet']>=.95
                and s['upright_at_least']>=np.cos(np.deg2rad(20.)) and s['min_com']>=1.2)
    mesh_passed = bool(mesh['clearance_at_last_ms_native']>.01 and mesh['first_stayed_past_ms'] is not None)
    checks=dict(reference_course_passed=a['passed'], independent_repeat_bitwise=True, flat_no_box_contact=True,
                all_arms_same_initial_standing=True,
                continued_measured_ground_forward_gt_0_03=bool(through is not None and through>.03),
                whole_native_initial_terminal_standing=all(good_stand(s) for s in ast.values()))
    def compact(value, raw_stands, path):
        return dict(run=str(path),native_integrations=value['native_integrations'],passed=value['passed'],
                    native_gate=value['native_engineering'],block_passage=value['block_passage'],raw_stands=raw_stands)
    reports=[compact(value,raw_stands,path) for value,raw_stands,path in
             ((a,ast,aware),(b,bst,blind),(f,fst,flat),(r,rst,repeat))]
    result=dict(passed=all(checks.values()),checks=checks,body_trunk=mesh,
                additional_named_trunk_mesh_passed=mesh_passed,continued_floor_advance_native=through,
                first_aware_blind_difference_ms=earliest, reports=reports,
                contrast_boundary='A learned collision response requires measured current side load, causal correction replay and generalization. These arms compare known-geometry planning only.',
                blind_already_passes=bool(b['passed']),run_ids=[stan.sha(p/'RUN_IDENTITY.json') for p in (aware,blind,flat,repeat)])
    stan.put(output,result)
    print(json.dumps(dict(checks=checks, through=through, mesh=mesh, blind_pass=b['passed']),indent=2),flush=True)


def extension(output, earlier, extended, repeat):
    primary, pt, pa, _ = verified_run(earlier)
    main, trace, action, raw_stand = verified_run(extended)
    assurance, _, _, _ = verified_run(repeat)
    assert main['files']==assurance['files'] and assurance['independent_repeat_bitwise']
    p, plan = primary['plan'].copy(), main['plan'].copy()
    earlier_duration, extended_duration = p.pop('course_ms'), plan.pop('course_ms')
    assert p==plan and extended_duration>earlier_duration
    base_change=np.flatnonzero(np.any(pa!=action[:len(pa)],axis=1))
    first_changed=int(base_change[0]*5) if len(base_change) else len(pa)*5
    for field in ('qpos','qvel','loads','pad_xyz','block_native'):
        np.testing.assert_array_equal(pt[field][:first_changed+1],trace[field][:first_changed+1])
    c,_,_ = build_scene(main['plan'])
    xmax=main['plan']['center'][0]+main['plan']['half_size'][0]
    mesh=actual_trunk_mesh_clearance(c,trace,xmax)
    footprint=whole_mesh_footprint(c,trace,main['plan'])
    finish_walk=main['plan']['warm_ms']+main['plan']['course_ms']
    onset=mesh['first_stayed_past_ms']
    onward=None if onset is None else float(trace['com'][finish_walk,0]-trace['com'][onset,0])
    checks=dict(primary_engineering_and_course=bool(main['passed']), independent_repeat_bitwise=True,
                original_exact_prefix_through_30_s=first_changed>=30000,
                actual_trunk_held_beyond_far_plane=bool(onset is not None and mesh['clearance_at_last_ms_native']>.01),
                additional_forward_after_trunk_clearance=bool(onward is not None and onward>.03),
                all_71_meshes_outside_box_footprint_in_terminal_window=footprint['minimum_disjoint_AABB_axis_margin_native']>0.,
                terminal_ground_only=bool(trace['block_native'][-200:].max() < .002*W))
    result=dict(passed=all(checks.values()),checks=checks,first_duration_change_effect_in_action_ms=first_changed,
                effective_earlier_and_extended_walk_ms=[earlier_duration,extended_duration],
                trunk=mesh,full_mesh_terminal_footprint=footprint,after_trunk_advance_native=onward,main=main,raw_stands=raw_stand,
                run_ids=[stan.sha(p/'RUN_IDENTITY.json') for p in (earlier,extended,repeat)])
    stan.put(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='main'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='method',required=True)
    for name, fields, func in (
            ('matched',('aware','blind','flat','repeat'),qualify),
            ('extended',('earlier','extended','repeat'),extension)):
        p=subs.add_parser(name)
        p.add_argument('output',type=Path)
        for field in fields:
            p.add_argument(field,type=Path)
        p.set_defaults(func=func,fields=fields)
    a=parser.parse_args()
    a.func(a.output.resolve(),*(getattr(a,key).resolve() for key in a.fields))
