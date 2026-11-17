"""Numerical and anatomical qualification before the new neck reaches the body."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from fixed_neck import (CAPACITANCE_PF, CHARGE_NS_MS, DECAY_MS, DT_MS, FixedCircuit, HERE,
                        LEAK_NS, PARENT, REFRACTORY_MS, REST_MV, REVERSAL_MV, RISE_MS,
                        full_catalog, parameters, write)
from build_catalog import hash_file


def kernel_checks():
    results=[]
    for rise,decay,charge,rev in zip(RISE_MS,DECAY_MS,CHARGE_NS_MS,REVERSAL_MV,strict=True):
        amplitude=charge/(decay-rise)
        continuous=lambda t:amplitude*(np.exp(-t/decay)-np.exp(-t/rise))
        integral=quad(continuous,0,300)[0]
        assert abs(integral-charge)<1e-9
        ar=ad=amplitude
        integrated=0.
        difference=0.
        for i in range(3000):
            left,right=i*DT_MS,(i+1)*DT_MS
            gr=ar*rise/DT_MS*(1-np.exp(-DT_MS/rise))
            gd=ad*decay/DT_MS*(1-np.exp(-DT_MS/decay))
            avg=gd-gr
            exact=quad(continuous,left,right)[0]/DT_MS
            difference=max(difference,abs(exact-avg))
            integrated+=avg*DT_MS
            ar*=np.exp(-DT_MS/rise)
            ad*=np.exp(-DT_MS/decay)
        assert difference<1e-10 and abs(integrated-charge)<1e-9
        results.append(dict(rise_ms=rise,decay_ms=decay,reversal_mv=rev,charge_ns_ms=charge,
                            integrated_ns_ms=integrated,max_bin_error=difference))
    return results


def hand_transit(events,cuts,delays):
    target=np.zeros_like(events)
    for time,lane,port in np.argwhere(events):
        due=time+int(delays[port])
        # A cut deletes all in-flight events of the named axon.
        if due<len(events) and not np.any(cuts[time:due+1,port]):
            target[due,lane,port]=True
    return target


def all_ports():
    catalog=full_catalog()
    model=FixedCircuit(catalog)
    nports=len(model.port_roots)
    total=nports*160+1400
    emitted=np.zeros((total,1,nports),dtype=bool)
    arrivals=np.zeros_like(emitted)
    cuts=np.zeros((total,nports),dtype=bool)
    drive=np.zeros((1,len(model.roots)),dtype=np.float32)
    metrics={}
    observed=np.empty((total,6))
    for tick in range(total):
        drive.fill(0.)
        if tick<nports*160:
            p=tick//160
            within=tick%160
            root=model.port_roots[p]
            port=model.ports[root]
            receptor=[root] if port['direction'] in ('DN','SA') else [pre for pre,_ in port['incoming_receptors']]
            indices=model.validate_inputs(receptor,'brain' if port['direction']=='DN' else 'body')
            if within in (0,30,80,120):
                drive[0,indices]=68.75
            cuts[tick,p]=20<=within<65
        emitted[tick],arrivals[tick],_,v=model.step(drive,cuts[tick])
        observed[tick]=[v.min(),v.max(),model.last_conductance.min(),model.last_conductance.max(),
                        model.last_current.min(),model.last_current.max()]
    expected=hand_transit(emitted,cuts,model.neck_delay)
    assert np.array_equal(arrivals,expected)
    assert not np.any(arrivals[:model.neck_delay.min()])
    assert np.isfinite(observed).all() and observed[:,2].min()>=0.
    assert abs(model._voltage-REST_MV).max()<.05
    for j,root in enumerate(model.port_roots):
        a=j*160
        e=emitted[a:a+160,0,j]
        r=arrivals[a:a+160,0,j]
        if model.ports[root]['direction'] in ('DN','SA'):
            assert e.any() and r.any(),(root,e.sum(),r.sum())
        assert not arrivals[a+20:a+65,0,j].any()
        if e[80:].any():
            assert arrivals[a+80:a+160,0,j].any(),root
        metrics[root]=dict(direction=model.ports[root]['direction'],cell_type=model.ports[root]['cell_type'],
                           delay_ticks=int(model.neck_delay[j]),emitted=int(e.sum()),arrived=int(r.sum()),
                           cut_suppressed=True,restored=bool(arrivals[a+80:a+160,0,j].any()),
                           fires_under_test_stimulus=bool(e.any()))
    np.savez_compressed(HERE/'NEURAL.npz',emitted=emitted,arrivals=arrivals,cuts=cuts,extrema=observed)
    return dict(ports=metrics,all_delay_exact=True,trace_sha256=hash_file(HERE/'NEURAL.npz'),
                recovery_max_rest_error_mv=float(abs(model._voltage-REST_MV).max()),native_steps=total)


def pulse_experiment(dt,sign):
    catalog=full_catalog()['ports']
    group='forward' if sign>0 else 'halt_web_a'
    selected={root:port for root,port in catalog.items()
              if port['group'] in (group,'front_touch','front_load')}
    model=FixedCircuit({'ports':selected},dt_ms=dt)
    root=next(root for root,port in selected.items() if port['group']==group and port['side']=='left')
    receiver=max(selected[root]['outgoing'],key=lambda pair:pair[1])[0]
    column=model.index[receiver]
    drive=np.zeros((1,len(model.roots)),dtype=np.float32)
    cut=np.zeros(len(model.ports),dtype=bool)
    times=np.arange(round(100/dt))*dt
    data=[]
    spikes=[]
    for tick,t in enumerate(times):
        drive.fill(0.)
        if abs(t-5.)<dt*.1:
            drive[0,model.index[root]]=68.75
        emitted,arrived,_,v=model.step(drive,cut)
        data.append([v[0,column],*model.last_current[:,0,column],*model.last_conductance[:,0,column]])
        if emitted[0,model.port_index[root]]:
            spikes.append(t)
    values=np.asarray(data)
    if sign<0:
        assert values[:,0].min()>REVERSAL_MV[1]-1e-9
        assert values[:,0].min()<REST_MV
    else:
        assert values[:,0].max()>REST_MV and values[:,4:].min()>=0.
    assert abs(values[-1,0]-REST_MV)<.1
    return times,values,root


def membrane_and_steps():
    table={}
    for sign in (1,-1):
        coarse_t,coarse,root=pulse_experiment(.1,sign)
        fine_t,fine,_=pulse_experiment(.05,sign)
        aligned=fine[1::2]
        err=float(abs(aligned[:,0]-coarse[:,0]).max())
        assert err<1.5
        table[root]=dict(max_voltage_error_mv_dt_halving=err,
                        low_mv=float(coarse[:,0].min()),high_mv=float(coarse[:,0].max()))
        np.savez_compressed(HERE/f'PULSE_{sign}.npz',time_ms=coarse_t,values=coarse,
                            fine_time_ms=fine_t,fine_values=fine)
    model=FixedCircuit(full_catalog())
    drive=np.zeros((1,len(model.roots)),dtype=np.float32)
    root=next(r for r,p in model.ports.items() if p['direction']=='DN')
    drive[0,model.index[root]]=68.75
    times=[]
    mask=np.zeros(len(model.ports),dtype=bool)
    for tick in range(200):
        e,*_=model.step(drive,mask)
        if e[0,model.port_index[root]]:
            times.append(tick*DT_MS)
    assert min(np.diff(times))>=REFRACTORY_MS-1e-9
    table['refractory']={'intervals_ms':np.diff(times).tolist(),'minimum_ms':min(np.diff(times))}
    return table


if __name__=='__main__':
    result=dict(kernel=kernel_checks(),pulse_and_membrane=membrane_and_steps(),all_ports=all_ports(),
                parameters_sha256=hash_file(HERE/'PARAMETERS.json'),
                runtime_sha256=hash_file(HERE/'fixed_neck.py'),plasticity=False)
    write(HERE/'DYNAMICS_VERIFIED.json',result)
    print('52 anatomical delays checked, every emitted spike has exact transit; weak AN responses reported separately',flush=True)
