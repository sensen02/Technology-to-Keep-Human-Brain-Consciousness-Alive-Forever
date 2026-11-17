"""Full body paired floor-recess experiment with per-native-step ground evidence."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np
from scipy.spatial import cKDTree
import torch

from geometry import HERE,BASE,LEG,make_gap_model,terrain_masks
import walking_bench as bench
from prototype import policy as frozen_policy
sys.path.insert(0,str(BASE.parent/'20261116T004235Z'))
from verify_native import Record

W=bench.W
stan=bench.stand
read=bench.read
LEGS=bench.LEGS
RF=LEGS.index(LEG)
HELPER='RM'
RM=LEGS.index(HELPER)
LF=LEGS.index('LF')
RF_JOINTS=slice(RF*7,(RF+1)*7)
RM_JOINTS=slice(RM*7,(RM+1)*7)
OTHER=np.setdiff1d(np.arange(42),np.r_[RF*7:(RF+1)*7,RM*7:(RM+1)*7])
OTHER_FEET=np.arange(6)!=RF
WARM,STOP=500,1200
ARCHIVE=BASE/'PROTO_nominal'
NATIVE_LIB=HERE/'floor_contacts.so'
LOCAL_FIELDS=('phase_expected_RF_load_W','RF_load_W','RF_z_minus_expected_body_native',
              'RF_vz_relative_body_native_s','other_support_W',
              *(f'RF_actual_body_{a}' for a in 'xyz'),
              *(f'RF_target_departure_{j}_rad' for j in range(7)),'LF_load_W','RM_load_W')


class FloorRecord(Record):
    def __init__(self, body, duration):
        super().__init__(body,duration)
        self.terrain=np.ascontiguousarray(terrain_masks(body.plant),dtype=np.int32)
        self.load=np.zeros(17)
        self.rf_contact=np.zeros((duration*80,4))
        self.fk_z=np.zeros(duration*80)
        self.pad_ids=np.asarray(body.pads[LEG])
        self.pad_radii=body.plant.geom_size[self.pad_ids,0].copy()
        self.lib=ctypes.CDLL(str(NATIVE_LIB))
        self.function=self.lib.gap_totals
        self.function.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]+[ctypes.c_void_p]*2
        self.function.restype=None

    def before(self,c):
        super().before(c)
        self.fk_z[self.count]=np.min(c.d.geom_xpos[self.pad_ids,2]-self.pad_radii)

    def solved(self,c,before):
        self.function(c.plant._address,c.d._address,self.foot_map.ctypes.data,
                      self.terrain.ctypes.data,RF,self.velocity.ctypes.data,self.load.ctypes.data)
        self.forces[self.count]=self.load[:13]
        self.rf_contact[self.count]=self.load[13:17]
        if self.count%4031==0:
            original=stan.physics.NativeRecorder.solved(None,c,{'qvel_before':self.velocity})
            want=np.r_[original['loads_native'],
                       original['loads_native']*original['contact_speed_native_per_s'],original['nonfoot_force_native']]
            np.testing.assert_allclose(self.load[:13],want,atol=1e-9,rtol=1e-12)
            self.cross_checks+=1


class Local:
    def __init__(self,c,walk_ms):
        self.body=c
        self.pads=c.pads[LEG]
        self.pad_ids=np.array(self.pads)
        self.pad_body=int(c.plant.geom_bodyid[self.pads[0]])
        self.jac=np.zeros((3,c.plant.nv))
        with np.load(ARCHIVE/'TRACE.npz',allow_pickle=False) as z:
            self.nominal=z['target'].copy()
            self.nominal_pad=z['padz'][:,RF].copy()
            self.nominal_qpos=z['qpos'].copy()
            self.nominal_pos=z['foot'][:,RF].copy()
            self.nominal_load=z['loads'][:,RF].copy()
        self.ref_ms=np.arange(WARM,WARM+min(walk_ms,9600),5)
        self.tree=cKDTree(self.nominal[self.ref_ms][:,OTHER])

    def sense(self,m):
        c=self.body
        _,i=self.tree.query(c.actuator.target_rad[OTHER])
        ref=self.ref_ms[i]
        p=c.d.geom_xpos[self.pad_ids].mean(0)
        mj.mj_jac(c.plant,c.d,self.jac,None,p,self.pad_body)
        rotation=c.d.xmat[c.th].reshape(3,3)
        # Chest site velocity is subtracted before projecting onto current chest z.
        velocity=rotation.T @ (self.jac @ c.d.qvel-c.d.qvel[:3])
        departure=c.actuator.target_rad[RF_JOINTS]-self.nominal[ref,RF_JOINTS]
        return np.r_[self.nominal_load[ref]/W,m['loads'][RF]/W,
                     m['foot'][RF,2]-self.nominal_pos[ref,2],velocity[2],
                     m['loads'][OTHER_FEET].sum()/W,m['foot'][RF],departure,
                     m['loads'][LF]/W,m['loads'][RM]/W]


def build_scene(shape, initial, initial_sha):
    loader=stan.physics.act_mod.load_full_model
    if shape is not None:
        model=make_gap_model(shape)
        stan.physics.act_mod.load_full_model=lambda *a,**kw:model
    try:
        c=stan.physics.Core(adhesion=False)
    finally:
        stan.physics.act_mod.load_full_model=loader
    c.planes=terrain_masks(c.plant)>0
    c.physics_order=stan.physics.rc.SENS_IDX
    np.testing.assert_array_equal(c.physics_order,bench.SENSOR_BY_LEG)
    if stan.sha(initial)!=initial_sha:
        raise ValueError('Frozen standing snapshot identity mismatch')
    with np.load(initial,allow_pickle=False) as z:
        mj.mj_setState(c.plant,c.d,z['state'],mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(c.plant,c.d)
        c.t=int(z['t'])
        c.actuator.command_ticks=int(z['ticks'])
        for key in z.files:
            if key.startswith('core_'):
                setattr(c,key[5:],z[key].copy())
            if key.startswith('act_'):
                setattr(c.actuator,key[4:],z[key].copy())
        np.testing.assert_array_equal(c.d.qpos,z['qpos'])
        np.testing.assert_array_equal(c.d.qvel,z['qvel'])
        delta=stan.observe(c)-z['observation']
    return c,float(np.abs(delta).max())


def event_metrics(trace,shape):
    if shape is None or shape['filled']:
        return dict(gap=bool(shape),filled=bool(shape and shape['filled']))
    a,b,c,d=shape['opening']
    p=trace['rf_pad_xyz']
    inside=(p[:,0]>a)&(p[:,0]<b)&(p[:,1]>c)&(p[:,1]<d)
    depressed=inside&(trace['padz'][:,RF]<-.012)&(trace['loads'][:,RF]<.004*W)
    time_ids=np.flatnonzero(depressed)
    out=dict(gap=True,filled=False,real_pad_below_opening_ms=time_ids.tolist(),
             observed_gap_event=bool(len(time_ids)),peak_cavity_penetration_native=float(-trace['padz'][inside,RF].min()) if inside.any() else None)
    if not len(time_ids):
        return out
    first=time_ids[0]
    reload=(np.arange(len(p))>first)&~inside&(trace['loads'][:,RF]>=.02*W)
    steady=np.flatnonzero(np.convolve(reload,np.ones(30,dtype=np.int32),'valid')==30)
    out.update(first_pad_drop_ms=int(first),first_upper_floor_reload_ms=int(steady[0]) if len(steady) else None,
               lowest_pad_ms=int(time_ids[np.argmin(trace['padz'][time_ids,RF])]),
               other_support_min_W=float(trace['loads'][first:first+200,OTHER_FEET].sum(1).min()/W),
               helper_peak_W=float(trace['loads'][first:first+250,RM].max()/W),
               native_step_length=sta_length())
    return out


def sta_length():
    return float(stan.LENGTH)


def native_metrics(rec,output,result,trace):
    n=rec.count
    np.testing.assert_array_equal(rec.endpoints[:len(trace['qpos'])],trace['qpos'])
    np.testing.assert_allclose(rec.poses[:n,6],trace['time'][0]+np.arange(n)*12.5e-6,atol=5e-8,rtol=0.)
    assert not rec.poses[:n,7].any()
    end=min(WARM+result['walk_ms'],len(trace['qpos'])-1)
    walk=rec.forces[WARM*80:end*80]
    totals=walk[:,:6].sum(0)
    absolute=walk[:,6:12].sum(0)/np.maximum(1e-10,totals)
    dx=trace['com'][end,:2]-trace['com'][WARM,:2]
    duration=(end-WARM)*.001
    slip=float(absolute.max()/(np.linalg.norm(dx)/duration))
    com_min=float(min(rec.poses[WARM*80:end*80,2].min(),trace['com'][end,2]))
    nonfoot=float(walk[:,12].max()/W)
    support=float(totals.sum()/len(walk)/W)
    repose={}
    for key,a,b in (('initial',WARM-200,WARM),('terminal',end+STOP-200,end+STOP)):
        if b*80>n:
            repose[key]=dict(passed=False)
            continue
        ws=rec.forces[a*80:b*80]
        h=rec.poses[a*80:b*80]
        g=dict(support=ws[:,:6].sum()/len(ws)/W>=.9,nonfoot=ws[:,12].max()/W<.01,
               com=min(h[:,2].min(),trace['com'][b,2])>=1.2,upright=h[:,5].min()>=np.cos(np.deg2rad(20.)),
               three=np.mean((ws[:,:6]>=.02*W).sum(1)>=3)>=.95,
               sampled=bool(result['initial_stand' if key=='initial' else 'terminal_stand']['passed']))
        g={name:bool(v) for name,v in g.items()}
        repose[key]=dict(passed=all(g.values()),gates=g)
    forward=float(dx[0]/duration)
    gates=dict(support=support>=.9,plane_nonfoot=nonfoot<=.20,com=com_min>=1.2,
               forward=forward>=.05,slip=slip<=.5,cycles=all(result['cycles'].values()),
               initial=repose['initial']['passed'],terminal=repose['terminal']['passed'])
    np.savez_compressed(output/'NATIVE.npz',forces=rec.forces[:n],pose=rec.poses[:n],
                        rf_floor=rec.rf_contact[:n],rf_pad_bottom_z=rec.fk_z[:n])
    out=dict(native_integrations=n,zero_extra_force=True,endpoint_qpos_exact=True,
             contact_cross_checks=rec.cross_checks,native_sha256=stan.sha(output/'NATIVE.npz'),
             rf_floor_fields=['bottom_up_W_native','upper_up_W_native','edge_support_native','edge_normal_native'],
             ground_foot_support_mean_W=support,ground_nonfoot_peak_W=nonfoot,minimum_com=com_min,
             signed_forward_native_s=forward,max_ground_normalized_slip=slip,standing=repose,
             absolute_slide_native_s=dict(zip(LEGS,absolute.tolist(),strict=True)),
             native_gates=gates,native_joint_pass=all(gates.values()))
    stan.put(output/'NATIVE_AUDIT.json',out)
    return out


def run(output,mode,scenario,policy=None,native=False,walk_ms=9600,*,probe_ms=None,
        teacher_config=None,policy_source=None):
    output=Path(output).resolve()
    scenario=Path(scenario).resolve() if scenario is not None else None
    shape=read(scenario) if scenario is not None else None
    output.mkdir()
    torch.set_num_threads(1)
    initial=bench.STARTS/'nominal.npz'
    initial_sha=read(bench.STARTS/'BANK.json')['nominal']['sha256']
    c,initial_obs_delta=build_scene(shape,initial,initial_sha)
    local=Local(c,walk_ms)
    mode=mode.strip()
    if mode=='teacher':
        from gap_demo import Teacher
        policy=Teacher(local,read(Path(teacher_config)))
    elif mode not in ('control','student','replay'):
        raise ValueError(f'Unsupported mode {mode}')
    if mode in ('student','replay') and policy is None:
        raise ValueError('Mode requires loaded observation policy')
    root=Path(__file__).resolve()
    base=frozen_policy()
    base_file=BASE/'prototype.py'
    extra=[root,HERE/'geometry.py',base_file,BASE/'walking_bench.py',ARCHIVE/'RESULT.json',
           ARCHIVE/'TRACE.npz',initial,bench.STOP_SCRIPT]
    if scenario is not None:
        extra.append(scenario)
    if teacher_config is not None:
        extra.extend((HERE/'gap_demo.py',Path(teacher_config)))
    if policy_source is not None:
        extra.append(Path(policy_source))
    if native:
        extra.extend((NATIVE_LIB,HERE/'floor_contacts.cpp'))
    info=dict(scope='single missed foothold and support transfer',mode=mode,initial_sha256=initial_sha,
              initial_observation_max_abs_delta=initial_obs_delta,scenario=shape,
              actuator_limit_native_s=stan.physics.MAX_RATE)
    ident=stan.identity.capture(c,info,extra_paths=extra)
    stan.put(output/'RUN_IDENTITY.json',ident)
    walk_end=WARM+walk_ms
    duration=walk_end+STOP
    if probe_ms is not None:
        duration=min(probe_ms,duration)
    rec=FloorRecord(c,duration) if native else None
    c.recorder=rec
    zero=np.zeros(42)
    trace=[]
    decisions=[]
    start=time.monotonic()
    def snap(m,applied,correction):
        pad=c.d.geom_xpos[local.pad_ids].mean(0).copy()
        trace.append(dict(time=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),
             com=c.d.subtree_com[c.th].copy(),loads=m['loads'].copy(),nonfoot=float(m['nonfoot']),
             padz=m['padz'].copy(),speed=m['speed'].copy(),foot=m['foot'].copy(),upright=float(c.d.xmat[c.th,8]),
             target=c.actuator.target_rad.copy(),ctrl=c.d.ctrl.copy(),applied=applied.copy(),
             correction=correction.copy(),rf_pad_xyz=pad))
    m=c.measure()
    snap(m,zero,zero)
    unsafe=0
    for t in range(duration):
        if t%5==0:
            intent=int(WARM<=t<walk_end)
            x=bench.features(c,m)
            s=local.sense(m)
            b=np.asarray(base(x,intent),dtype=np.float64)
            if mode=='teacher':
                a=policy(t,x,intent,s,b,m)
            elif mode in ('student','replay'):
                a=policy(x,intent,s)
            else:
                a=b.copy()
            a=bench.legacy.bounded_action(a)
            correction=a-b
            decisions.append((t,x.copy(),intent,s.copy(),b.copy(),a.copy()))
        m=c.tick(a)
        if np.any(c.d.xfrc_applied) or np.any(c.d.qfrc_applied):
            raise RuntimeError('Unexpected external force or hidden assistance')
        snap(m,a,correction)
        unsafe=unsafe+1 if m['com']<.8 or m['nonfoot']>.5*W else 0
        if unsafe>=100:
            break
    tr={key:np.asarray([s[key] for s in trace]) for key in trace[0]}
    np.savez_compressed(output/'TRACE.npz',**tr)
    names=('ms','x','intent','local','base','action')
    np.savez_compressed(output/'DECISIONS.npz',
                        **{name:np.asarray(value) for name,value in zip(names,zip(*decisions),strict=True)})
    result=bench.metrics(tr,WARM,walk_ms,walk_end+STOP)
    result.update(start='nominal',scenario=shape,scenario_source=str(scenario) if scenario else None,
                  mode=mode,walk_ms=walk_ms,probe=probe_ms is not None,recipient_leg=LEG,helper_leg=HELPER,
                  sensor_fields=LOCAL_FIELDS,initial_snapshot_sha256=initial_sha,
                  initial_observation_max_abs_delta=initial_obs_delta,elapsed_wall_s=time.monotonic()-start,
                  trace_sha256=stan.sha(output/'TRACE.npz'),decision_sha256=stan.sha(output/'DECISIONS.npz'),
                  physical_native_iterations=(len(tr['time'])-1)*80,raw_native_contacts=bool(native),
                  event=event_metrics(tr,shape),teacher_engagement_ms=policy.onset if mode=='teacher' else None)
    if native:
        report=native_metrics(rec,output,result,tr)
        result['native_audit_sha256']=stan.sha(output/'NATIVE_AUDIT.json')
        result['native_joint_pass']=report['native_joint_pass']
    stan.identity.verify_runtime(ident,c)
    stan.put(output/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('mode','actual_ms','joint_sampled_pass','teacher_engagement_ms')},indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['control','teacher'])
    ap.add_argument('output',type=Path)
    ap.add_argument('--scenario',type=Path)
    ap.add_argument('--plan',type=Path)
    ap.add_argument('--native',action='store_true')
    ap.add_argument('--probe-ms',type=int)
    args=ap.parse_args()
    run(args.output,args.mode,args.scenario,native=args.native,probe_ms=args.probe_ms,teacher_config=args.plan)
