"""Passive membrane-current forward recording with explicit current closure.
Lumped finite spherical volume sources approximate compartment membrane currents.
Not real electrode surface integration or ephaptic self-consistent coupling.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.cable import Morphology,CableNeuron
from engine.electrode import Contact,BipolarField,VoltageRecorder
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'/'brain_isolation'


def run(reference=(500,500,0),stimulus=10.):
    morph=Morphology.cylinder(300,1,nseg=60)
    net=CableNeuron(morph,dt_ms=.025)
    sources=[Contact(tuple(x),.5) for x in np.c_[morph.x,morph.y,morph.z]]
    field=BipolarField([Contact((20,10,0),5),Contact((200,80,0),5)],reference_um=reference)
    rec=VoltageRecorder((100,8,0),(280,60,0),bandwidth_Hz=1000)
    trace=[];balance=[]
    for k in range(600):
        current=stimulus if 2<=k*net.dt<3 else 0.
        old=net.v.copy();drive=field.inward_axial_drive_nA(net,[current,-current])
        net.step(i_inject=drive)
        # Positive OUTWARD membrane current includes capacitive displacement.
        imem=net.C*(net.v-old)/net.dt+net.g_mem*(net.v-net.E_leak)
        balance.append(imem.sum())
        # Allow floating-point solve residual (record it, do NOT subtract mean).
        if abs(imem.sum())>1e-12:raise AssertionError('membrane source closure failed')
        signal=rec.step(net.dt,sources,imem,field,[current,-current])
        trace.append([net.t,signal['neural_unfiltered_mV'],signal['artifact_unfiltered_mV'],signal['measured_mV']])
    return np.array(trace),np.array(balance)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a,balance=run();b,_=run(reference=(900,10,100));dark,db=run(stimulus=0)
    gauge=float(abs(a-b).max());zero=float(abs(dark[:,1:]).max())
    assert gauge<1e-9 and zero<1e-9
    assert np.max(abs(a[:,1]))>1e-8
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    axes[0].plot(a[:,0],a[:,1]*1e6);axes[0].set(title='Membrane-derived forward signal',xlabel='ms',ylabel='nV (NOT mean Vm)')
    axes[1].plot(a[:,0],a[:,2]*1000,label='artifact raw');axes[1].plot(a[:,0],a[:,3]*1000,label='total filtered')
    axes[1].set(title='Artifact dominates this geometry',xlabel='ms',ylabel='uV');axes[1].legend(fontsize=8)
    axes[2].plot(a[:,0],balance);axes[2].set(title='Outward current sum, no mean subtraction',xlabel='ms',ylabel='nA residual')
    fig.suptitle('PASSIVE RECORDING PROTOTYPE — balanced membrane sources, homogeneous volume conductor')
    fig.savefig(OUT/'neural_recording_demo.png',dpi=150);plt.close(fig)
    report={'scope':'hand-built passive membrane-current forward recording, not realistic fly extracellular electrode',
       'current_formula':'Iout[nA]=C[nF]*(Vm_new-Vm_old)/dt[ms]+g_leak[uS]*(Vm_new-E_leak)[mV]',
       'closure':'no intracellular injection; extracellular axial forcing sums zero; NO numerical mean-current repair',
       'max_balance_residual_nA':float(abs(balance).max()),'reference_shift_max_difference':gauge,
       'zero_input_max_mV':zero,'neural_peak_mV':float(abs(a[:,1]).max()),'artifact_peak_mV':float(abs(a[:,2]).max()),
       'assumptions':['passive ideal cylinder','finite spherical volume source radius .5 um per compartment, illustrative',
        'DN passive constants borrowed','no extracellular feedback/ephaptic solve','no metal interface','no tissue anisotropy','capacitive current included'],
       'not_claimed':['true neural spike recording','consciousness','biological injury validation']}
    (OUT/'neural_recording_metrics.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(OUT/'neural_recording_traces.npz',trace=a,balance_nA=balance)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
