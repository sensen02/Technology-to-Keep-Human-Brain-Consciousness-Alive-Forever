"""Structural identifiability + hypothesis sensitivity, no measured DA fitting.
A fixed free-ligand clamp deliberately isolates receptor kinetics. Real tissue
finite-pool transport is tested separately by local_tissue. All values illustrative.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from engine.neural_cond import ConductanceNetwork
OUT=Path(__file__).resolve().parent/'outputs'/'brain_isolation'


def occupancy(t,c,kon,koff):
    rate=kon*c+koff
    return kon*c/rate*(-np.expm1(-rate*np.asarray(t)))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Synthetic train and held-out concentrations clearly NOT biological data.
    t=np.linspace(0,15,61); true=(.2,.1)
    y=occupancy(t,1,*true)
    fit=least_squares(lambda z:occupancy(t,1,*np.exp(z))-y,np.log([.08,.3]),
                      xtol=1e-13,ftol=1e-13,gtol=1e-13)
    estimated=np.exp(fit.x)
    train_error=float(np.max(abs(occupancy(t,1,*estimated)-y)))
    hold_error=float(np.max(abs(occupancy(t,3,*estimated)-occupancy(t,3,*true))))
    # At equilibrium, scaling both rates preserves Kd: cannot identify kinetics.
    rates=[(.02,.01),(.2,.1),(2,1)]
    doses=np.logspace(-2,2,40)
    equilibria=np.array([doses/(doses+koff/kon) for kon,koff in rates])
    assert np.max(abs(equilibria-equilibria[0]))<1e-12
    assert train_error<1e-6 and hold_error<1e-6
    # Short neural trial starts at specified receptor state, not hidden time scaling.
    # Signed sensitivity means selecting one of two explicit receptor conductances.
    gains=np.array([-.002,-.001,0,.001,.002])
    fractions=np.array([0,.1,.3,.6,1.])
    hz=np.empty((len(gains),len(fractions)))
    for i,g in enumerate(gains):
        for j,r in enumerate(fractions):
            net=ConductanceNetwork(1,dt_ms=.1)
            count=0
            for _ in range(3000):
                # Added tonic receptor conductance held during each interval.
                # New conductance offsets its exact decay, not event frequency.
                ge=max(g,0)*r;gi=max(-g,0)*r
                e=ge-net.ge[0]; inh=gi-net.gi[0]
                count+=int(net.step(current_nA=.022,
                    exc_events_uS=max(0,e),inh_events_uS=max(0,inh))[0])
            hz[i,j]=count/.3
    assert np.all(hz[2]==hz[2,0]) # target gain=0 -> occupancy has no neural effect
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for (kon,koff),eq in zip(rates,equilibria):
        label=f'kon={kon:g}, koff={koff:g}'
        ax[0].semilogx(doses,eq,label=label)
        ax[1].plot(t,occupancy(t,1,kon,koff),label=label)
    ax[0].set(title='Same equilibrium: rates NOT identifiable',xlabel='clamped free ligand (nM)',ylabel='occupancy')
    ax[1].set(title='Time course distinguishes rate scale',xlabel='s',ylabel='occupancy')
    ax[0].legend(fontsize=7);ax[1].legend(fontsize=7)
    im=ax[2].imshow(hz,origin='lower',aspect='auto',cmap='viridis')
    ax[2].set_xticks(range(len(fractions)),[str(x) for x in fractions])
    ax[2].set_yticks(range(len(gains)),[str(x) for x in gains])
    ax[2].set(title='Neural effect depends on ASSUMED target',xlabel='prescribed initial receptor fraction',ylabel='hypothesis signed g_max (uS)')
    fig.colorbar(im,ax=ax[2],label='example point neuron Hz')
    fig.suptitle('UNCERTAINTY DEMO — synthetic data and assumed targets; NOT measured hormone effects')
    fig.savefig(OUT/'modulation_uncertainty.png',dpi=150);plt.close(fig)
    report={'scope':'synthetic identifiability and target sensitivity; NOT Drosophila data fit',
            'clamp':'prescribed infinite free ligand reservoir; no finite pool depletion in this experiment',
            'equilibrium_ambiguity':{'rates_kon_nM_inv_s_koff_s':rates,'same_Kd_nM':.5},
            'synthetic_fit':{'true':true,'estimated':estimated.tolist(),'train_max_abs_error':train_error,'heldout_concentration_max_abs_error':hold_error},
            'hypothesis_scan':{'signed_gmax_uS':gains.tolist(),'occupancy':fractions.tolist(),'Hz':hz.tolist(),
                'sign_semantics':'positive uses E_exc=0mV, negative uses E_inh=-70mV; BOTH unverified target hypotheses',
                'default_gain':0,'default_off_check_passed':True,'neuron_parameters':'all ConductanceParams illustrative'},
            'biological_evidence':{'candidate':'https://pmc.ncbi.nlm.nih.gov/articles/PMC9897283/',
                'available_claim':'adult dissected brain regional stimulated dopamine release/clearance measured in article',
                'not_done':'no raw experimental concentration-time values extracted or fitted here',
                'cannot_infer':['global endocrine replacement','receptor-specific channel efficacy','long-term viability','consciousness']}}
    (OUT/'modulation_uncertainty.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
