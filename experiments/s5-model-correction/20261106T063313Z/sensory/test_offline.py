"""Independent offline checks: extracted OLD observer + native OLD encoder oracle.
No physics/neural imports, model load, forward or stepping. All comparisons deterministic.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from adapter import (Layout, RawObservation, SensoryAdapter, capture_raw, joint_quantities,
                     save_stream, load_stream, ARMS, LEGS, ChannelSensoryEncoder)
from engine.embodied.sensory_channels import ChannelQuantities
from export_identity import META, IDENT, LOOKUP, OUT, build

R=Path('/run/media/sensen/Data2/cell_wound_prototype')
O=OUT.parent
XML=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml')
COXA={'720575941577076072','720575941432016009','720575941442197595','720575941563170984','720575941618732757','720575941491974604','720575941411400815','720575941487662989','720575941607583341','720575941579875317','720575941480796835','720575941414263953'}
QUAR={'720575941624613116','720575941611480650','720575941595199525','720575941525349634','720575941565115655','720575941556298323','720575941506863013','720575941668780209','720575941532457091'}
TROCH={'720575941491131854','720575941542576413','720575941507236165','720575941431833519','720575941557956772','720575941538267107','720575941478415681','720575941463174544','720575941598954908','720575941539438741','720575941565519255','720575941509287553','720575941689339510','720575941557736356','720575941615009945','720575941592005160','720575941393503382'}


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(a,b):
    assert np.asarray(a).dtype==np.asarray(b).dtype
    assert np.asarray(a).shape==np.asarray(b).shape
    assert np.asarray(a).tobytes()==np.asarray(b).tobytes(), 'bitwise mismatch'


def main():
    source_paths=[META,IDENT,LOOKUP,XML,R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py']
    before={str(p):digest(p) for p in source_paths}
    identity=json.loads(OUT.read_text()); groups=identity['channel_groups']
    assert set(identity['quarantine_ids'])==QUAR
    cb=set(r for b in identity['consensus_blocks'] if b['block']=='coxa' for r in b['banc_888_ids'])
    tb=set(r for b in identity['consensus_blocks'] if b['block']=='trochanter' for r in b['banc_888_ids'])
    assert cb==COXA and tb==TROCH and not (cb|tb)&QUAR
    # A separate membership oracle checks exact source location/nerve/type, not exporter output fields.
    meta=pd.read_feather(META).fillna('').set_index('banc_888_id',drop=False)
    for root in COXA:
        row=meta.loc[root]; assert row.body_part_sensory=='coxa,front_leg'
        assert row.peripheral_target_type in ['CoHP8,hair_plate','CoHP3,hair_plate','CoHP4,hair_plate']
    for root in TROCH:
        row=meta.loc[root]; assert row.body_part_sensory=='front_leg,trochanter' and row.cell_type!='CoHP8'
    assert sum(len(v) for leg in groups.values() for v in leg.values())==3814
    # Joint names are verified against real XML without compilation. Addresses below are synthetic.
    names=[j.get('name') for j in ET.parse(XML).getroot().iter('joint') if (j.get('name') or '').startswith('joint_')]
    assert len(names)==42
    names=tuple(reversed(names))  # Deliberately not canonical/source order; mapping must use names.
    rng=np.random.default_rng(981); ref=rng.normal(size=42)
    layout=Layout(names,np.arange(1,43),np.arange(7,49),np.arange(6,48),ref,
        ('LFTarsus1','floor','box','Thorax','RFTarsus1'),np.array([1,0,0,3,2]),
        ('world','LFTarsus1','RFTarsus1','Thorax'),np.array([7,0,6,7,7]),{0:'lf',4:'rf'},0,
        dict(path=str(XML),sha256=digest(XML),fixture=True))
    legacy=ChannelSensoryEncoder(groups,12000)
    adapters={arm:SensoryAdapter(copy.deepcopy(legacy),layout,identity,arm) for arm in ARMS}
    # Extract ONLY the previous pure observation function, not the heavyweight runner module.
    node=next(x for x in ast.parse((R/'run_neural_muscle_closure.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='observe_quantities')
    ns={'np':np,'LEGS':LEGS,'ChannelQuantities':ChannelQuantities}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<OLD observe_quantities>','exec'),ns)
    fake=types.SimpleNamespace(mjtGeom=types.SimpleNamespace(mjGEOM_PLANE=0),mj_contactForce=lambda m,d,i,f:f.__setitem__(slice(None),d.force[i]))
    sys.modules['mujoco']=fake
    model=types.SimpleNamespace(geom_type=layout.geom_types)
    qref=np.r_[np.zeros(7),ref]
    jmap={leg:dict(q=layout.qadr[layout.rows[leg]].tolist(),tib_q=int(layout.qadr[layout.tibia[leg]]),tib_v=int(layout.vadr[layout.tibia[leg]])) for leg in LEGS}
    def data(delta=None,velocity=None,pairs=(),forces=(),time=0.):
        q=qref.copy();q[7:]+=np.zeros(42) if delta is None else delta
        v=np.r_[np.zeros(6),np.zeros(42) if velocity is None else velocity]
        contacts=[types.SimpleNamespace(geom1=a,geom2=b,frame=np.eye(3).ravel(),pos=np.array([i,.2,.3]),dist=-.001,efc_address=i) for i,(a,b) in enumerate(pairs)]
        f=np.array(forces,dtype=float).reshape(-1,6)
        return types.SimpleNamespace(time=time,qpos=q,qvel=v,ncon=len(contacts),contact=contacts,force=f)
    tested=0; fixture=[]
    for trial in range(100):
        n=int(rng.integers(0,15));pairs=rng.integers(0,5,size=(n,2));forces=rng.normal(size=(n,6));forces[:,0]=abs(forces[:,0])
        if n:forces[0]=0
        d=data(rng.normal(size=42),rng.normal(size=42)*10,pairs,forces,trial*.001)
        raw=capture_raw(model,d,layout,fake.mj_contactForce)
        oldq=ns['observe_quantities'](model,d,qref,jmap,layout.foot_leg)
        oldcurrent,_=legacy.encode(oldq)
        newcurrent,_=adapters['baseline'].encode(raw);equal(oldcurrent,newcurrent)
        # Independent original .quantities() context: reference subtraction before subset.
        ev=ChannelSensoryEncoder(groups,12000);ev.set_reference_pose(ref)
        for leg in LEGS:
            ev.assign(leg,layout.tibia[leg],int(layout.blocks[(leg,'trochanter')][0]),layout.rows[leg])
            q=ev.quantities(leg,raw.joint_position,raw.joint_velocity,oldq[leg].contact_force_norm,oldq[leg].foot_contact)
            common=joint_quantities(raw.joint_position,raw.joint_velocity,layout.reference,layout.rows[leg],layout.tibia[leg],oldq[leg].contact_force_norm,oldq[leg].foot_contact)
            equal(np.array(list(vars(q).values())),np.array(list(vars(common).values())))
        for arm in ARMS[1:]:
            current,record=adapters[arm].encode(raw)
            allowed=set()
            if arm=='A_positive_force':allowed={i for vals in groups.values() for i in vals['bristle_touch']}
            if arm=='B_club_motion':allowed={i for vals in groups.values() for i in vals['club_vibration']}
            if arm=='C_local_amplitude':allowed={i for g in identity['consensus_blocks'] for i in g['local_indices']}
            assert set(np.flatnonzero(current!=oldcurrent)) <= allowed
            mask=np.ones(12000,bool);mask[list(allowed)]=False;equal(current[mask],oldcurrent[mask])
        if trial<3:fixture.append((raw,newcurrent))
        tested+=1
    # Contact expectations use exact constants, with all joint deviations zero.
    for pairs,forces,old_touch,new_touch,force in [([(0,1)],[[0]*6],1,0,0),
        ([(0,1)],[[3,4,0,0,0,0]],1,1,5), ([(1,0)],[[3,4,0,0,0,0]],1,1,5),
        ([(0,2)],[[3,4,0,0,0,0]],0,0,0), ([(3,1)],[[3,4,0,0,0,0]],0,0,0)]:
        raw=capture_raw(model,data(pairs=pairs,forces=forces),layout,fake.mj_contactForce)
        base,_=adapters['baseline'].encode(raw);a,_=adapters['A_positive_force'].encode(raw)
        np.testing.assert_array_equal(base[groups['lf']['bristle_touch']],.3*old_touch)
        np.testing.assert_array_equal(a[groups['lf']['bristle_touch']],.3*new_touch)
        np.testing.assert_array_equal(a[groups['lf']['campaniform_strain']],.3*force)
    raw=capture_raw(model,data(pairs=[(0,1),(0,1)],forces=[[3,4,0,0,0,0],[2,-4,0,0,0,0]]),layout,fake.mj_contactForce)
    q,_=adapters['baseline'].quantities(raw);assert q['lf'].contact_force_norm==5+np.sqrt(20)
    # Club stream, both signs and clamp; no dynamics, membrane or calcium tau introduced.
    club_ids=groups['lf']['club_vibration'];v=np.zeros(42)
    for speed,expected in zip([-20.,-10.,-5.,0.,5.,10.,20.],[.3,.3,.15,0.,.15,.3,.3]):
        v[layout.tibia['lf']]=speed
        raw=capture_raw(model,data(velocity=v),layout,fake.mj_contactForce)
        cur,record=adapters['B_club_motion'].encode(raw)
        np.testing.assert_array_equal(cur[club_ids],expected)
        assert record['adapter']['vibration_frequency_tuning']=='NOT_MODELLED'
    # C only29 IDs; local block specificity, sign ambiguity explicitly proven.
    delta=np.zeros(42);block=layout.blocks[('lf','coxa')];delta[block[0]]=3.;delta[block[1]]=-3.
    raw=capture_raw(model,data(delta),layout,fake.mj_contactForce)
    c,_=adapters['C_local_amplitude'].encode(raw)
    expected_ids=next(g['local_indices'] for g in identity['consensus_blocks'] if g['leg']=='lf' and g['block']=='coxa')
    np.testing.assert_array_equal(c[expected_ids],.3)
    other_ids=[i for g in identity['consensus_blocks'] if (g['leg'],g['block'])!=('lf','coxa') for i in g['local_indices']]
    np.testing.assert_array_equal(c[other_ids],0.)
    reverse=capture_raw(model,data(-delta),layout,fake.mj_contactForce)
    equal(c[expected_ids],adapters['C_local_amplitude'].encode(reverse)[0][expected_ids])
    # Identity permutation equivariance: ID -> current, independent of storage order.
    permutation=rng.permutation(12000);roots=np.array(identity['roots'])[permutation]
    reid=build(meta.reset_index(drop=True),roots,json.loads(LOOKUP.read_text()))
    reenc=ChannelSensoryEncoder(reid['channel_groups'],12000)
    for arm in ARMS:
        equal(SensoryAdapter(reenc,layout,reid,arm).encode(raw)[0],adapters[arm].encode(raw)[0][permutation])
    # Invalid data must fail, never silently turn malformed input into a valid channel.
    rejected=0
    valid_contact=capture_raw(model,data(pairs=[(0,1)],forces=[[3,4,0,0,0,0]]),layout,fake.mj_contactForce)
    for field in ['joint_position','local_force6','frames']:
        bad=copy.deepcopy(valid_contact);getattr(bad,field).flat[0]=np.nan
        try: adapters['baseline'].encode(bad)
        except ValueError:rejected+=1
        else:raise AssertionError('accepted NaN')
    bad=copy.deepcopy(valid_contact);bad.local_force6[0,0]=-1
    try:adapters['baseline'].encode(bad)
    except ValueError:rejected+=1
    else:raise AssertionError('accepted negative normal')
    # The raw .npz round-trip is bit-exact, including all six force components.
    stream=O/'TEST_RAW_STREAM.npz'
    save_stream(stream,[x[0] for x in fixture],[x[1] for x in fixture],layout)
    ll,rr,cc=load_stream(stream)
    for original,restored,current in zip(fixture,rr,cc):
        for field in RawObservation.__dataclass_fields__:
            equal(getattr(original[0],field),getattr(restored,field))
        equal(SensoryAdapter(copy.deepcopy(legacy),ll,identity).encode(restored)[0],current)
    assert all(digest(p)==h for p,h in before.items())
    report=dict(passed=True,random_raw_cases=tested,baseline_current_bitwise=True,legacy_vector_quantities_bitwise=True,
        target_cells_only_all_arms=True,identity_permutation_equivariant=True,consensus_ids=sorted(COXA|TROCH),quarantine_ids=sorted(QUAR),
        physics_model_load=0,physics_forward=0,physics_step=0,neural_step=0,raw_stream_roundtrip_bitwise=True,
        actual_physical_raw_equivalence='PENDING_PARENT_HEAVY_GO',new_C_direction_restoration=False,
        tests='100 random raw frames; old source AST/current oracle; exact ID sets; sign/clamp/contact cases; raw roundtrip; row permutation; malformed input',
        source_hashes=before,source_hashes_unchanged=True,importance='核心必需')
    with (O/'OFFLINE_TESTS.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
