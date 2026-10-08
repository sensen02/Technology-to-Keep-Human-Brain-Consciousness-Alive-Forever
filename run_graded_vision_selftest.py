"""Analytical and sparse-tier tests for graded vision interface."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from engine.graded_vision import GradedVision
from engine.neural_cond import ConductanceNetwork, ConductanceParams
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'/'brain_isolation'; OUT.mkdir(parents=True,exist_ok=True)
def main():
    n=8; visual=np.array([0,1]); pre=np.array([0,1,0,1,2]); post=np.array([2,3,4,5,6]); syn=np.array([2.,1.,3.,2.,9.]); pure=np.array([1,1,0,0,1],bool)
    g=GradedVision(n,visual,pre,post,syn,pure,dt_ms=.2,scale_uS_per_synapse=1e-3)
    assert g.describe()['pure_histamine_edges']==2
    dark=np.array([g.step(0).copy() for _ in range(10)]); assert np.all(dark==0)
    on=np.array([g.step(1).copy() for _ in range(30)]); assert np.all(on>=0); target_on=float(g.last_target_uS[2])
    off=np.array([g.step(0).copy() for _ in range(30)]); assert np.all(off>=0)
    # exact steady target and bounded activation
    assert np.all((g.last_activation>=0)&(g.last_activation<=1)); assert target_on > 0
    # blocked visual receptor cannot inject through a removed edge
    g2=GradedVision(n,visual,pre,post,syn,np.zeros_like(pure),dt_ms=.2,scale_uS_per_synapse=1e-3)
    assert np.all(g2.step(1)==0) and g2.describe()['reachable_posts']==0
    # tonic release has no duplicate additive bookkeeping: target is fixed each step
    x=g.step(1); y=g.step(1); assert np.all(x>=0) and np.all(y>=0)
    # dt convergence of the bounded adaptation trajectory
    gdt=GradedVision(n,visual,pre,post,syn,pure,dt_ms=.1,scale_uS_per_synapse=1e-3)
    for _ in range(60): gdt.step(1)
    assert np.all(np.isfinite(gdt.last_activation)) and np.all((gdt.last_activation>=0)&(gdt.last_activation<=1))
    net=ConductanceNetwork(n,ConductanceParams(inh_reversal_mV=-70.0,inh_tau_ms=8.0),dt_ms=.2); g.apply_to(net,1); assert np.all(net.gi>=0)
    try: g.apply_to(ConductanceNetwork(n,ConductanceParams(inh_reversal_mV=-65.0),dt_ms=.2),1); raise AssertionError('contract mismatch accepted')
    except ValueError: pass
    # Fixed adapted activation permits an independent conductance and Vm solution.
    steady=GradedVision(n,visual,pre,post,syn,pure,dt_ms=.2,scale_uS_per_synapse=1e-3)
    steady.receptor.A[:]=1/(1+steady.receptor.p['i50'])
    snet=ConductanceNetwork(n,ConductanceParams(),dt_ms=.2)
    for _ in range(3000): steady.apply_to(snet,1)
    expected_g=.002*float(steady.receptor.steady_state(1))
    expected_v=(.001*(-65)+expected_g*(-70))/(.001+expected_g)
    assert abs(snet.v[2]-expected_v)<1e-8
    assert abs(snet.gi[2]-expected_g*np.exp(-.2/8))<1e-12
    # Identical pulse, exact end times: error must fall on dt refinement.
    def trajectory(dt):
        gg=GradedVision(n,visual,pre,post,syn,pure,dt_ms=dt,scale_uS_per_synapse=1e-3)
        nn=ConductanceNetwork(n,ConductanceParams(),dt_ms=dt)
        for _ in range(round(20/dt)): gg.apply_to(nn,1)
        return nn.v[2]
    vv=[trajectory(dt) for dt in (.4,.2,.1,.025)]
    errors=[abs(v-vv[-1]) for v in vv[:-1]]
    assert errors[2]<errors[1]<errors[0]
    assert np.all(snet.spike_count==0)
    for bad in (np.nan,np.inf,-1):
        try: g.step(bad); raise AssertionError('invalid light accepted')
        except ValueError: pass
    positive=GradedVision(n,visual,pre,post,syn,pure,dt_ms=.2,scale_uS_per_synapse=1e-3,e_hist_mV=-55)
    pnet=ConductanceNetwork(n,ConductanceParams(inh_reversal_mV=-55),dt_ms=.2)
    for _ in range(100): positive.apply_to(pnet,1)
    assert -65 < pnet.v[2] < -55
    assert snet.v[2] < -65
    invalid_kwargs=[{'visual_indices':np.array([0,0])},{'pre':pre.astype(float)}, {'post':np.full_like(post,n)}, {'pure_histamine':pure.astype(int)}, {'synapses':syn*-1}, {'synapses':syn*np.nan}, {'e_hist_mV':np.nan}, {'tau_release_ms':np.inf}, {'dt_ms':np.nan}, {'scale_uS_per_synapse':np.inf}]
    base=dict(n_neurons=n,visual_indices=visual,pre=pre,post=post,synapses=syn,pure_histamine=pure)
    for change in invalid_kwargs:
        try: GradedVision(**(base|change)); raise AssertionError('invalid constructor accepted')
        except ValueError: pass
    huge=GradedVision(**base); huge.step(1,dt_ms=1000)
    assert np.all((huge.receptor.A>=0)&(huge.receptor.A<=1))
    assert np.all((huge.last_activation>=0)&(huge.last_activation<=1))
    results={'reversal_changed_target_Vm_mV':float(pnet.v[2]),'steady_Vm_mV':expected_v,'steady_conductance_during_integration_uS':expected_g,'dt_refinement_errors_mV':errors,'status':'passed','dark_max_uS':float(dark.max()),'on_max_uS':float(on.max()),'washout_last_uS':float(off[-1].max()),'description':g.describe(),'dt_ms':.2,'analytical_target_uS':target_on}
    (OUT/'graded_vision_selftest.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
