"""Active cable machinery demonstration, classical squid HH NOT fly physiology."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_hybrid_selftest import trace
from engine.neural_active import IdealCableSpec
OUT=Path(__file__).resolve().parent/'outputs'/'brain_isolation'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    t,a,sa=trace();_,b,sb=trace(cut=True);_,c,sc=trace(drive=False)
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    xyz=(np.arange(a.shape[1])+.5)*3000/a.shape[1]
    for axis,v,name in zip(ax[:2],[a,b],['Connected ideal cable','Axial cut at 1000 um (sealed ends)']):
        im=axis.pcolormesh(t,xyz,v.T,cmap='coolwarm',shading='auto',vmin=-80,vmax=40)
        axis.set(title=name,xlabel='ms',ylabel='axial position (um)')
        fig.colorbar(im,ax=axis,label='Vm (mV)')
    ax[1].axhline(1000,color='black',ls='--')
    ax[2].plot(t,a[:,0],label='intact proximal')
    ax[2].plot(t,a[:,-1],label='intact distal')
    ax[2].plot(t,b[:,-1],label='cut distal')
    ax[2].plot(t,c[:,-1],label='no-input distal',ls='--')
    ax[2].set(title='Cut blocks distal propagation',xlabel='ms',ylabel='Vm (mV)');ax[2].legend(fontsize=8)
    fig.suptitle('ACTIVE ENGINE TEST — ideal cylinder + squid HH at 6.3 C; NOT a calibrated fruit-fly axon')
    fig.savefig(OUT/'active_cut_demo.png',dpi=150);plt.close(fig)
    metrics={'scope':'NEURON classical squid HH machinery test, NOT fly neural bundle anatomy or viability',
             'provenance':IdealCableSpec().provenance(),
             'stimulus':'illustrative focal inward IClamp 20 nA over [1,2] ms; NOT extracellular electrode',
             'distal_spikes':{'intact':sa,'cut':sb,'no_input':sc},
             'first_upward_0mV_ms':{'proximal':float(t[np.flatnonzero(a[:,0]>=0)[0]]),'distal':float(t[np.flatnonzero(a[:,-1]>=0)[0]])},
             'cut_distal_max_mV':float(b[:,-1].max()),'dt_ms':.025}
    (OUT/'active_cut_metrics.json').write_text(json.dumps(metrics,indent=2))
    np.savez_compressed(OUT/'active_cut_traces.npz',times_ms=t,intact=a,cut=b,no_input=c)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
