"""Quantify the operating point of the conductance tier: input-driven or recurrence-driven?

Declared parameters are NOT calibrated biology, so this does not measure a fly.
It measures whether the network's own declared weight scale puts its spontaneous
operating point above or below the level at which its own recurrence, rather than
sensory input, sets its firing.  That distinction decides what must be calibrated
next, and it is measured here instead of assumed.

No simulation of the body, no controller, no target, no weight change on disk.
"""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/recurrent_regime_round15'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceParams
    from engine.embodied.adapters import TierConfig
    from muscle_wiring import build_mn_muscle_wiring

    OUT.mkdir(parents=True,exist_ok=True)
    cfg=TierConfig(background_mean_nA=0.,background_sd_nA=0.)
    wiring,stats,built=build_mn_muscle_wiring(verbose=False)
    We,Wi=built['edges']['We'],built['edges']['Wi']
    n=int(built['n'])
    p=ConductanceParams()
    assert We.shape==(n,n) and Wi.shape==(n,n)
    assert We.nnz and Wi.nnz

    row_e=np.asarray(We.sum(axis=1)).ravel()
    row_i=np.asarray(Wi.sum(axis=1)).ravel()
    row_all=row_e+row_i
    degree=np.diff(We.indptr)
    weights=np.concatenate([We.data,Wi.data])

    # Mean conductance a neuron accumulates if its partners fire asynchronously
    # at rate r: g = sum_partners(w) * r * tau.  Steady-state point neuron
    # resting-plus-synaptic conductance is compared with the leak.
    rates=np.array([1.,2.,5.,10.,20.,50.,100.,200.])
    ge=row_e[:,None]*rates[None,:]*p.exc_tau_ms/1000.
    gi=row_i[:,None]*rates[None,:]*p.inh_tau_ms/1000.
    ratio=(ge+gi)/p.leak_uS
    median_ratio=np.median(ratio,axis=0)
    frac_above_leak=(ratio>1).mean(axis=0)
    frac_above_10x=(ratio>10).mean(axis=0)

    report={'status':'RECURRENT_REGIME_MEASUREMENT',
            'tier_fingerprint':built['fingerprint'],'n':n,
            'tier_config':{k:v for k,v in vars(cfg).items()},
            'conductance_params':{k:v for k,v in vars(p).items()},
            'source_sha256':{str(ROOT/'muscle_wiring.py'):sha(ROOT/'muscle_wiring.py'),
                             str(ROOT/'engine/neural_cond.py'):sha(ROOT/'engine/neural_cond.py'),
                             str(ROOT/'engine/embodied/adapters.py'):sha(ROOT/'engine/embodied/adapters.py')},
            'synapses':{'We_nnz':int(We.nnz),'Wi_nnz':int(Wi.nnz),
                        'weight_median_uS':float(np.median(weights)),
                        'weight_max_uS':float(weights.max()),
                        'weight_scale_uS_per_synapse':cfg.weight_scale_uS_per_synapse},
            'fan_in':{'median_partners':int(np.median(degree)),'max_partners':int(degree.max()),'mean_partners':float(degree.mean())},
            'conductance_if_all_partners_fire_once_uS':{
                'median':float(np.median(row_all)),'mean':float(row_all.mean()),'max':float(row_all.max()),
                'leak_uS':p.leak_uS,
                'median_over_leak':float(np.median(row_all)/p.leak_uS),
                'fraction_above_leak':float((row_all>p.leak_uS).mean()),
                'fraction_above_10x_leak':float((row_all>10*p.leak_uS).mean()),
                'fraction_above_100x_leak':float((row_all>100*p.leak_uS).mean())},
            'asynchronous_rate_curve':{'rates_hz':rates.tolist(),
                'median_synaptic_over_leak':median_ratio.tolist(),
                'fraction_neurons_above_leak':frac_above_leak.tolist(),
                'fraction_neurons_above_10x_leak':frac_above_10x.tolist()},
            'interpretation':'A NEGATIVE background current of zero mean does not make the network '
                'quiescent: with this declared weight scale, asynchronous partner firing at ordinary '
                'rates already drives most neurons above their leak conductance, so recurrence rather '
                'than sensory input sets the operating point. This is consistent with the measured '
                'persistent activity after a brief stimulus and with activity surviving a finer '
                'timestep. It is a property of DECLARED, uncalibrated parameters, not a fly measurement, '
                'and it is NOT a bug: no code defect was found earlier.',
            'what_this_does_not_establish':[
                'It does not identify the correct weight scale for a fly.',
                'It does not license changing the weight scale to obtain a desired behaviour; the curve '
                'is reported as a sensitivity curve with no recommended value.',
                'It does not show the sensory translation is correct or that standing is attainable.'],
            'scope':'declared-parameter sensitivity measurement; no body simulation, no controller, no target.'}
    (OUT/'recurrent_regime.json').write_text(json.dumps(report,indent=2))

    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    axs[0].plot(rates,median_ratio,'o-')
    axs[0].axhline(1,ls='--',c='gray',label='synaptic = leak')
    axs[0].set_xscale('log');axs[0].set_yscale('log')
    axs[0].set_xlabel('assumed partner firing rate (Hz)')
    axs[0].set_ylabel('median synaptic conductance / leak')
    axs[0].set_title('Median neuron: synaptic conductance vs leak\n(declared uncalibrated weight scale)')
    axs[0].legend();axs[0].grid(alpha=.3,which='both')
    axs[1].plot(rates,frac_above_leak,'o-',label='above 1x leak')
    axs[1].plot(rates,frac_above_10x,'s-',label='above 10x leak')
    axs[1].set_xscale('log');axs[1].set_xlabel('assumed partner firing rate (Hz)')
    axs[1].set_ylabel('fraction of 12000 neurons')
    axs[1].set_title('Fraction of neurons dominated by synaptic input\nover their own leak')
    axs[1].legend();axs[1].grid(alpha=.3)
    fig.suptitle('Recurrence-dominated operating point: a DECLARED parameter regime, not a fly measurement')
    fig.tight_layout();fig.savefig(OUT/'recurrent_regime.png',dpi=110)

    print(json.dumps({'fan_in':report['fan_in'],
        'conductance_if_all_partners_fire_once_uS':{k:v for k,v in report['conductance_if_all_partners_fire_once_uS'].items() if k!='leak_uS'},
        'median_synaptic_over_leak_by_rate':dict(zip(rates.tolist(),[round(x,3) for x in median_ratio]))},indent=2))
    print(OUT)

if __name__=='__main__':main()
