"""Independently replay a turn with compact contact readings on every native step."""
import argparse
import ctypes
from pathlib import Path

import numpy as np
import torch

import turning as run
from turning import HERE, W, read, stand

SO = HERE/"native_contact.so"


class Record:
    def __init__(self,c,duration):
        if c.plant.nv!=48 or c.plant.opt.timestep!=12.5e-6:
            raise ValueError("Unexpected frozen plant or native timestep")
        lib=ctypes.CDLL(str(SO))
        self.contact=lib.contact_totals
        self.contact.argtypes=[ctypes.c_void_p]*5
        self.contact.restype=None
        self.c=c
        self.velocity=np.zeros(48)
        self.foot_map=np.ascontiguousarray(c.geom_leg,dtype=np.int32)
        self.forces=np.zeros((duration*80,13))
        self.poses=np.zeros((duration*80,8))
        self.endpoints=np.zeros((duration+1,c.plant.nq))
        self.endpoints[0]=c.d.qpos
        self.count=0
        self.cross_checks=0

    def before(self,c):
        self.velocity[:]=c.d.qvel
        self.poses[self.count,:3]=c.d.subtree_com[c.th]
        rot=c.d.xmat[c.th]
        self.poses[self.count,3:5]=[rot[0],rot[3]]
        self.poses[self.count,5]=rot[8]
        self.poses[self.count,6]=c.d.time
        self.poses[self.count,7]=float(np.max(np.abs(c.d.xfrc_applied))+np.max(np.abs(c.d.qfrc_applied)))

    def solved(self,c,before):
        row=self.forces[self.count]
        self.contact(c.plant._address,c.d._address,self.foot_map.ctypes.data,self.velocity.ctypes.data,row.ctypes.data)
        if self.count%4031==0:
            original=stand.physics.NativeRecorder.solved(None,c,{"qvel_before":self.velocity})
            expected=np.r_[original["loads_native"],
                  original["loads_native"]*original["contact_speed_native_per_s"],original["nonfoot_force_native"]]
            np.testing.assert_allclose(row,expected,atol=1e-9,rtol=1e-12)
            self.cross_checks+=1

    def after(self,c,before,solved):
        self.count+=1
        if self.count%80==0:
            self.endpoints[self.count//80]=c.d.qpos


def measure(rec,ms,result):
    begin,end,after,finish=result["bounds"]
    if rec.count != finish*80:
        raise ValueError("Incomplete trajectory for native classification")
    np.testing.assert_array_equal(rec.endpoints,ms["qpos"])
    np.testing.assert_allclose(rec.poses[:,6],ms["time"][0]+np.arange(rec.count)*12.5e-6,atol=3e-8,rtol=0.)
    assert not rec.poses[:,7].any()
    native=rec.forces[begin*80:end*80]
    pose=rec.poses[begin*80:end*80]
    drift=float(np.linalg.norm(pose[:,:2]-pose[0,:2],axis=1).max()/stand.LENGTH)
    x=rec.poses[begin*80+1:end*80,:2]-pose[:-1,:2]
    axis=pose[:-1,3:5]
    axis=axis/np.linalg.norm(axis,axis=1,keepdims=True)
    move=(x*axis).sum()+((ms["com"][end,:2]-pose[-1,:2])*axis[-1]).sum()
    speed=float(move/((end-begin)*.001))
    slip=native[:,6:12].sum(0)/native[:,:6].sum(0)
    pivot=result["mode"].startswith("pivot")
    slip_ratio=float((slip/speed).max()) if speed>0 and not pivot else None
    inner=dict(support=float(native[:,:6].sum()/len(native)/W),com=float(min(pose[:,2].min(),ms["com"][end,2])),
               nonfoot=float(native[:,12].max()),forward=speed,drift=drift,slip_absolute=slip.tolist(),
               arc_slip=slip_ratio,angle=result["actual_turn_deg"])
    gates=dict(support=inner["support"]>=.9,com=inner["com"]>=1.2,nonfoot=inner["nonfoot"]==0.,
               turn=bool(result["turn_gates"]["turn"]),cycles=bool(result["turn_gates"]["six_cycles"]))
    gates["drift" if pivot else "forward"]=drift<=.5 if pivot else speed>=.05
    tail0=(finish-200)*80
    tail=rec.forces[tail0:finish*80]
    angle=np.unwrap(ms["yaw"])
    tail_yaw=float(abs(np.rad2deg(angle[finish]-angle[finish-200])))
    standing=dict(before=bool(result["initial_stand"]["passed"]),
        tail=bool(result["tail"]["passed"] and tail[:,12].max()<.01*W
            and tail[:,:6].sum(1).mean()>=.9*W and tail_yaw<=.5),
        tail_yaw=tail_yaw,tail_nonfoot=float(tail[:,12].max()))
    low=bool(pivot or slip_ratio is not None and slip_ratio<=.5)
    continued=result["continue_walk"]
    return dict(turn=inner,gates=gates,standing=standing,low_slip_pass=low,
                function_pass=all(gates.values()),
                joint=bool(all(gates.values()) and low and standing["before"] and standing["tail"]
                           and (continued is None or continued["passed"])),
                native_samples=rec.count,native_force_cross_checks=rec.cross_checks,
                all_endpoints_bitwise=True,extra_force_zero=True)


def execute(args):
    duration=run.WARM+args.entry+run.TURN+args.exit_walk+run.STOP
    loader=run.load_initial
    recorder=[]

    def instrument(bench,*rest):
        loader(bench,*rest)
        r=Record(bench.body,duration)
        bench.body.recorder=r
        recorder.append(r)

    run.load_initial=instrument
    pinned=stand.sha(SO)
    try:
        result=run.run(args.output,args.start,args.mode,"model",.13,.115,args.entry,args.exit_walk)
    finally:
        run.load_initial=loader
    assert len(recorder)==1 and stand.sha(SO)==pinned
    rec=recorder[0]
    with np.load(args.output/"TRACE.npz",allow_pickle=False) as z:
        metrics=measure(rec,z,result)
    if args.compare:
        reference=read(args.compare/"RESULT.json")
        for key in ("trace_sha256","decisions_sha256"):
            assert reference[key]==result[key], f"Fresh physics replay changed: {key}"
        metrics["independent_repeat_bitwise"]=True
    np.savez_compressed(args.output/"NATIVE.npz",forces=rec.forces,pose=rec.poses)
    metrics["native_arrays_sha256"]=stand.sha(args.output/"NATIVE.npz")
    metrics["native_library_sha256"]=pinned
    stand.put(args.output/"NATIVE.json",metrics)
    print(metrics,flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("mode",choices=run.MODES)
    a.add_argument("output",type=Path)
    a.add_argument("--start",default="nominal")
    a.add_argument("--entry",type=int,default=0)
    a.add_argument("--exit-walk",type=int,default=0)
    a.add_argument("--compare",type=Path)
    torch.set_num_threads(1)
    execute(a.parse_args())
