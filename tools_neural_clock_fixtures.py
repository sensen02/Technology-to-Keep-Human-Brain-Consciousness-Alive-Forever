"""Isolated numerical fixtures, NOT BANC or biological experiment.
Synthetic test edges never enter production network. 手工研究原型，不是成品.
"""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from engine.neural_cond import ConductanceNetwork, ConductanceParams
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/model_branches/neural_clock_fixtures'
DTS=(.5,.25,.125,.0625)


def refractory(dt):
    net=ConductanceNetwork(1,dt_ms=dt)
    times=[]
    for _ in range(round(20/dt)):
        if net.step(current_nA=100.)[0]: times.append(net.t_ms)
    return dict(dt_ms=dt,spike_times_ms=times,min_isi_ms=float(min(np.diff(times))),
                expected_min_isi_ms=net.p.refractory_ms+dt,
                meaning='Extreme synthetic current forces immediate eligible-step firing; not physiological drive.')


def constant(dt,method):
    p=replace(ConductanceParams(),threshold_mV=1e6)
    net=ConductanceNetwork(1,p,dt_ms=dt,voltage_method=method)
    current=.01
    for _ in range(round(10/dt)):net.step(current_nA=current)
    equilibrium=p.rest_mV+current/p.leak_uS
    exact=equilibrium+(p.rest_mV-equilibrium)*np.exp(-10*p.leak_uS/p.capacitance_nF)
    return dict(dt_ms=dt,method=method,final_mV=float(net.v[0]),analytic_mV=float(exact),error_mV=float(net.v[0]-exact))


def pulse(dt,method,kind):
    p=replace(ConductanceParams(),threshold_mV=1e6)
    net=ConductanceNetwork(1,p,dt_ms=dt,voltage_method=method)
    times=[];volts=[];conductance=[]
    for k in range(round(12/dt)):
        net.step(**{('exc_events_uS' if kind=='exc' else 'inh_events_uS'):.002 if k==0 else 0.})
        times.append(net.t_ms);volts.append(float(net.v[0]));conductance.append(float(net.ge[0] if kind=='exc' else net.gi[0]))
    tau=p.exc_tau_ms if kind=='exc' else p.inh_tau_ms
    return dict(dt_ms=dt,method=method,kind=kind,time_end_ms=times,voltage_mV=volts,
                conductance_end_uS=conductance,left_hold_area_ratio=float((dt/tau)/(-np.expm1(-dt/tau))))


def delay(dt):
    net=ConductanceNetwork(2,dt_ms=dt)
    # Deliberate synthetic fixture edge, not evidence/extension of BANC.
    net.set_connectivity([[0.,0.],[.001,0.]],np.zeros((2,2)))
    emissions=[];arrivals=[]
    for k in range(round(6/dt)):
        pending=net._ring[net._ring_i].copy()
        if pending[0]:arrivals.append(net.t_ms)
        s=net.step(current_nA=[100. if k==0 else 0.,0.])
        if s[0]:emissions.append(net.t_ms)
    return dict(dt_ms=dt,source_emission_end_ms=emissions,target_arrival_start_ms=arrivals,
                observed_delay_ms=arrivals[0]-emissions[0],declared_delay_ms=net.p.delay_ms)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[pulse(dt,m,k) for dt in DTS for m in ('backward_euler','frozen_exact') for k in ('exc','inh')]
    # Fine reference uses unchanged params and existing frozen_exact method.
    # It is a numerical reference, not exact continuous-time solution.
    refs={k:pulse(.00390625,'frozen_exact',k) for k in ('exc','inh')}
    for row in rows:
        ref=refs[row['kind']]
        sampled=np.interp(row['time_end_ms'],ref['time_end_ms'],ref['voltage_mV'])
        row['max_voltage_difference_fine_reference_mV']=float(np.max(np.abs(np.asarray(row['voltage_mV'])-sampled)))
    result=dict(label='手工研究原型，不是成品',scope='ISOLATED_SYNTHETIC_NUMERICAL_FIXTURES_NOT_BANC',
                refractory=[refractory(dt) for dt in DTS],constant=[constant(dt,m) for dt in DTS for m in ('backward_euler','frozen_exact')],
                delays=[delay(dt) for dt in DTS],pulses=rows,
                limits=['High current and synthetic edge are unit fixtures only, never production drive or connectivity.',
                        'Threshold raised only for subthreshold diagnostic fixtures; no runtime parameters altered.',
                        'Fine frozen_exact still has conductance splitting; not physiological ground truth.'])
    (OUT/'fixtures.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for col,kind in enumerate(('exc','inh')):
        for row in rows:
            if row['kind']==kind:axes[col].plot(row['time_end_ms'],row['voltage_mV'],label=f"{row['method']} dt={row['dt_ms']}")
        axes[col].plot(refs[kind]['time_end_ms'],refs[kind]['voltage_mV'],'k--',label='fine split reference')
        axes[col].set_title(kind+' single conductance pulse');axes[col].set_xlabel('Time (ms)');axes[col].set_ylabel('Voltage (mV)');axes[col].legend(fontsize=7)
    fig.suptitle('Manual numerical fixtures — no production network changes')
    fig.tight_layout();fig.savefig(OUT/'pulse_clock_comparison.png',dpi=150);plt.close(fig)
    print('OUTPUT',OUT)
    for r in result['refractory']:print('refractory',r['dt_ms'],r['min_isi_ms'])
    for r in result['delays']:print('delay',r['dt_ms'],r['observed_delay_ms'])
    return result

if __name__=='__main__':main()
