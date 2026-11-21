"""Verify physical contact bound, handoff causality, repeatability, support geometry and held controls."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from solve import HERE, BASE, body, BodyObservations, load_pose
from mesh_clearance import cross_leg_clearance
from warm_release import SETTLE, HANDOFF, MODES


def floor_bound(pose):
    native=body()
    obs=BodyObservations(native)
    feet=list(obs.foot_by_geom)
    assert native.npair==0 and native.nflex==0 and len(feet)==30
    assert all(native.geom_type[g]==mj.mjtGeom.mjGEOM_MESH and native.geom_condim[g]==3 for g in feet)
    assert native.opt.cone==mj.mjtCone.mjCONE_PYRAMIDAL and native.opt.impratio==1
    assert np.all(native.geom_margin==.001) and np.all(native.geom_gap==0)
    np.testing.assert_array_equal(native.geom_solref,np.tile([.02,1.],(native.ngeom,1)))
    np.testing.assert_array_equal(native.geom_solimp,np.tile([.9,.95,.001,.5,2],(native.ngeom,1)))
    np.testing.assert_array_equal(native.geom_friction,np.tile([1.,.005,.0001],(native.ngeom,1)))
    # At zero velocity and acceleration, 4 equal pyramid rows sum to the normal force.
    # i is contact impedance, 0.9 <= i <= 0.95. All 30 meshes get the maximum three
    # plane contacts here, deliberately overestimating the load achievable in one pose.
    total=sum(3*(.012+.002)/(.02**2*(1-.95)*native.body_invweight0[native.geom_bodyid[g],0])
              for g in feet)
    samples=0
    for q in (pose,native.key_qpos[0],np.load(BASE/'body/archived_pose.npz')['qpos']):
        for dz in (-.009,0,.002):
            d=load_pose(native,q)
            d.qpos[2]+=dz
            d.qacc[:]=0
            mj.mj_inverse(native,d)
            for ci,c in enumerate(d.contact):
                if not (obs.floor in c.geom and int(c.geom[1]) in obs.foot_by_geom):
                    continue
                address=c.efc_address
                assert address>=0 and c.dim==3 and c.includemargin==.002
                i=d.efc_KBIP[address,2]
                inv=native.body_invweight0[native.geom_bodyid[c.geom[1]],0]
                calculated=(c.includemargin-c.dist)*i**2/(.95**2*.02**2*(1-i)*inv)
                f=np.zeros(6)
                mj.mj_contactForce(native,d,ci,f)
                np.testing.assert_allclose(f[0],calculated,rtol=1e-10,atol=1e-12)
                samples+=1
    return dict(native_static_foot_weight_ceiling=total/obs.weight,contact_samples_checked=samples)


def verify():
    results=HERE/'results'
    geometry=json.loads((HERE/'stance.json').read_text())
    q=np.array(geometry['qpos'])
    u=np.array(geometry['tonic_activation'])
    m=body(1.)
    obs=BodyObservations(m)
    d=load_pose(m,q)
    d.act[:]=u
    d.ctrl[:]=u
    mj.mj_forward(m,d)
    assert np.all(q[7:]>=m.jnt_range[1:,0]) and np.all(q[7:]<=m.jnt_range[1:,1])
    assert np.all(u>=m.actuator_ctrlrange[:,0]) and np.all(u<=m.actuator_ctrlrange[:,1])
    assert all(obs.floor in c.geom and c.geom[1] in obs.foot_by_geom for c in d.contact)
    assert all(d.geom_xpos[g,1]*(1 if m.geom(g).name.startswith('L') else -1)>0
               for g in obs.foot_by_geom)
    assert max(abs(d.qacc))<1e-7
    cross_leg=cross_leg_clearance(m,d)
    trunk=cross_leg_clearance(m,d,trunk=True)
    assert not cross_leg['unresolved'] and not trunk['unresolved']
    native=body()
    alt=body(1.)
    changed=[k for k in dir(native) if isinstance(getattr(native,k),np.ndarray)
             and not np.array_equal(getattr(native,k),getattr(alt,k))]
    assert set(changed)=={'geom_priority','geom_solref'}
    f=native.geom('floor').id
    for k in changed:
        a,b=getattr(native,k),getattr(alt,k)
        np.testing.assert_array_equal(np.delete(a,f,axis=0),np.delete(b,f,axis=0))
    nr=load_pose(native,q)
    assert nr.ncon==d.ncon
    for c,other in zip(nr.contact,d.contact,strict=True):
        np.testing.assert_array_equal(c.geom,other.geom)
        for attr in ('dim','friction','solimp','dist','includemargin','solreffriction','pos'):
            np.testing.assert_array_equal(getattr(c,attr),getattr(other,attr))
        np.testing.assert_equal(c.solref,[.02,1.])
        np.testing.assert_equal(other.solref,[.001,1.])
    reports={k:json.loads((results/(k+'.json')).read_text()) for k in MODES}
    replay_dir=HERE/'replay'
    assert reports['warm_release']==json.loads((replay_dir/'warm_release.json').read_text())
    with np.load(results/'warm_release.npz') as main,np.load(replay_dir/'warm_release.npz') as replay:
        for key in main.files:
            np.testing.assert_array_equal(main[key],replay[key])
        replay_fields=len(main.files)
        for tag in MODES:
            with np.load(results/(tag+'.npz')) as tr:
                for k in ('qpos','qvel','activation','loads','torso','depth_mm'):
                    np.testing.assert_array_equal(main[k][:HANDOFF+1],tr[k][:HANDOFF+1])
                if tag!='cold_release':
                    for k in ('input_nA','receptor_spikes','motor_spikes','neural_command'):
                        np.testing.assert_array_equal(main[k][:HANDOFF//5],tr[k][:HANDOFF//5])
                else:
                    assert not np.any(tr['input_nA'][:HANDOFF//5]) and not np.any(tr['motor_spikes'][:HANDOFF//5])
                if tag in ('warm_release','cold_release'):
                    np.testing.assert_array_equal(tr['command'][HANDOFF:],
                        np.repeat(tr['neural_command'][HANDOFF//5-1:-1],5,axis=0))
                if tag=='motor_cut':
                    assert not np.any(tr['command'][HANDOFF:])
                assert reports[tag]['warm_all_six']
        first_load=main['loads'][HANDOFF+1,:6].sum()
    perturb={}
    for folder in ('perturbed','perturbed_negative'):
        r=json.loads((HERE/folder/'teacher_hold.json').read_text())
        assert r['warm_all_six'] and r['post_entire_stand']
        perturb[folder]=r['initial_vx_mm_s']
    assert reports['teacher_hold']['post_entire_stand'] and not reports['warm_release']['post_tail_stand']
    result=dict(**floor_bound(q), changed_model_fields=changed, replay_arrays=replay_fields,
                cross_leg_meshes=cross_leg, distal_leg_to_trunk_meshes=trunk,
                perturbed_teacher_initial_vx_mm_s=perturb, first_released_foot_weight=float(first_load),
                equilibrium_max_accel=float(max(abs(d.qacc))), cases=reports)
    (HERE/'ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print('replayed',replay_fields,'native upper bound',result['native_static_foot_weight_ceiling'])


if __name__=='__main__':
    verify()
