"""Illustrative local passive neck-bundle cut/electrode comparisons.
Not real fly geometry, active spike propagation, or viable isolated brain.
"""
from pathlib import Path
import json
import time
import resource
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.cable import CableNeuron, Morphology, DN_PASSIVE
from engine.electrode import Contact, BipolarField, VoltageRecorder

OUT=Path(__file__).resolve().parent/'outputs'/'brain_isolation'

def main():
    start=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    contacts=[Contact((25,10,0),5),Contact((80,80,0),5)]
    field=BipolarField(contacts,conductivity_S_m=.3,reference_um=(500,500,0))
    morph=Morphology.cylinder(300,1,nseg=120)
    scenarios={}
    dt=.025; count=800; t=np.arange(1,count+1)*dt
    for name in ('intact','cut_sealed','cut_leaky','sham'):
        c=CableNeuron(morph,dt_ms=dt)
        if name.startswith('cut'):
            c.cut_edges([61])
            if name=='cut_leaky': c.set_end_leak([60,61],.001,-20)
        rec=VoltageRecorder((120,8,0),(200,50,0),noise_sd_mV=.0001,seed=8)
        V=[]; artifact=[]; measured=[]
        for k in range(count):
            pulse=10 if 2<=k*dt<3 else 0
            if name=='sham': pulse=0
            I=field.inward_axial_drive_nA(c,[pulse,-pulse])
            V.append(c.step(i_inject=I).copy())
            obs=rec.step(dt,stimulus_field=field,stimulus_currents_nA=[pulse,-pulse])
            artifact.append(obs['artifact_unfiltered_mV']); measured.append(obs['measured_mV'])
        scenarios[name]={'V':np.array(V),'artifact':np.array(artifact),'measured':np.array(measured)}
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    x=np.linspace(-30,330,150); y=np.linspace(-40,110,90)
    X,Y=np.meshgrid(x,y)
    pot=field.potential_mV(np.c_[X.ravel(),Y.ravel(),np.zeros(X.size)],[10,-10]).reshape(X.shape)
    im=ax[0,0].pcolormesh(X,Y,pot,cmap='coolwarm',shading='auto')
    ax[0,0].plot(morph.x,morph.y,'k-',lw=1,label='ideal cable')
    ax[0,0].scatter([25,80],[10,80],marker='o',c=['red','blue'])
    ax[0,0].axvline(151.25,color='k',ls='--',label='cut plane')
    ax[0,0].set(title='Finite volume source + return: extracellular field',xlabel='um',ylabel='um')
    ax[0,0].legend(); fig.colorbar(im,ax=ax[0,0],label='mV, 10 nA source')
    for name,r in scenarios.items():
        ax[0,1].plot(t,r['V'][:,60],label=name)
        ax[1,0].plot(t,r['V'][:,100],label=name)
    ax[0,1].set(title='Near cut: sealed vs leak are different hypotheses',xlabel='ms',ylabel='Vm (mV)')
    ax[1,0].set(title='Distal Vm: field can polarize BOTH cut pieces',xlabel='ms',ylabel='Vm (mV)')
    ax[0,1].legend();ax[1,0].legend(loc='upper left',fontsize=8)
    inset=ax[1,0].inset_axes([.42,.18,.53,.38])
    for name in ('intact','cut_sealed','sham'):
        inset.plot(t,1000*(scenarios[name]['V'][:,100]-DN_PASSIVE['E_leak_mV'][0]),label=name)
    inset.set(xlim=(1,8),title='Zoom: distal delta Vm (uV)',xlabel='ms')
    inset.tick_params(labelsize=7)
    r=scenarios['intact']
    ax[1,1].plot(t,r['artifact'],label='unfiltered stimulus artifact')
    ax[1,1].plot(t,r['measured'],label='bandlimited + noise')
    ax[1,1].set(title='Recorder artifact only; no neuronal source inferred',xlabel='ms',ylabel='mV')
    ax[1,1].legend()
    fig.suptitle('LOCAL NUMERICAL DEMO — ideal geometry; passive only; NOT isolated-brain validation')
    fig.savefig(OUT/'local_electrode_demo.png',dpi=150);plt.close(fig)
    np.savez_compressed(OUT/'local_electrode_traces.npz',times_ms=t,**{k:v['V'] for k,v in scenarios.items()})
    metrics={'scope':'illustrative passive local cable, not fly cut anatomy or biological isolation',
        'dt_ms':dt,'duration_ms':count*dt,'geometry':'ideal cylinder 300 um x 1 um; not measured',
        'electrode':'finite uniform spherical volume source, infinite homogeneous conductor, explicit return; no metal interface',
        'parameters':{'passive_defaults':DN_PASSIVE,'conductivity_S_m':{'value':.3,'status':'illustrative'},
                      'contact_radius_um':{'value':5,'status':'illustrative'},
                      'pulse_nA':{'value':10,'status':'illustrative, not an experimental recommendation'},
                      'end_leak_uS':{'value':.001,'status':'illustrative'},
                      'end_reversal_mV':{'value':-20,'status':'illustrative'}},
        'scenario_ranges_mV':{k:[float(v['V'].min()),float(v['V'].max())] for k,v in scenarios.items()},
        'sham_max_rest_deviation_mV':float(abs(scenarios['sham']['V']-DN_PASSIVE['E_leak_mV'][0]).max()),
        'wall_seconds':time.perf_counter()-start,'process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'recording_scope':'stimulus artifact only; membrane source currents not supplied'}
    assert metrics['sham_max_rest_deviation_mV']<1e-8
    (OUT/'local_electrode_metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
