"""Numerical tests only; not validation of Drosophila physiology."""
from dataclasses import replace
import numpy as np
from engine.neural_cond import ConductanceNetwork, ConductanceParams


def main():
    p = ConductanceParams(threshold_mV=1000, adaptation_increment_nA=0)
    a = ConductanceNetwork(1, p)
    a.run(200, current_nA=0.01)
    assert abs(a.v[0] - (p.rest_mV+10)) < 0.001
    print('PASS passive steady state: nA/uS=mV')
    b = ConductanceNetwork(1, p)
    b.step(exc_events_uS=0.001)
    assert abs(b.ge[0]-0.001*np.exp(-0.1/p.exc_tau_ms)) < 1e-15
    print('PASS exact event decay')
    # Spike at t=1; delay=2 -> arrival at t=3 -> voltage changes during [3,4].
    c = ConductanceNetwork(2, ConductanceParams(adaptation_increment_nA=0), dt_ms=1)
    c.set_connectivity([[0,0],[0.01,0]], np.zeros((2,2)))
    s = c.step([1,0]); assert s.tolist() == [True,False]
    c.step(); assert c.ge[1] == 0
    c.step(); assert c.ge[1] == 0
    c.step(); assert c.ge[1] > 0
    print('PASS timestamped end-to-start synaptic delay')
    d = ConductanceNetwork(2)
    d.step([1,1], enabled=[True,False])
    assert d.spike_count[1] == 0 and d.v[1] == d.p.rest_mV
    print('PASS disabled point owner stays inactive')
    no = ConductanceNetwork(1, ConductanceParams(adaptation_increment_nA=0))
    yes = ConductanceNetwork(1)
    nr = no.run(1000, 0.05)['mean_rate_hz']
    yr = yes.run(1000, 0.05)['mean_rate_hz']
    assert 0 < yr < nr
    print('PASS adaptation reduces sustained rate', nr, yr)
    # Shunting at rest: identical reversal, different input resistance.
    def shunt(g):
        n=ConductanceNetwork(1, replace(p, inh_reversal_mV=p.rest_mV))
        for _ in range(3000):
            # Hold conductance at g for each integration interval.
            n.gi[:] = g
            n.step(0.01)
        return n.v[0]-p.rest_mV
    x,y=shunt(0),shunt(0.003)
    assert abs(x/4-y)<1e-4
    print('PASS shunting input-resistance ratio',x/y)
    for bad in (-1, float('nan')):
        try: ConductanceNetwork(1).step(exc_events_uS=bad)
        except ValueError: pass
        else: raise AssertionError('invalid conductance accepted')
    for bad in ([float('nan')], [2], [1], [True,False]):
        try: ConductanceNetwork(1).step(enabled=bad)
        except ValueError: pass
        else: raise AssertionError('invalid ownership mask accepted')
    print('PASS invalid conductances and nonboolean ownership rejected')
    from scipy.integrate import solve_ivp
    ge,gi,I = .002,.001,.01
    def rhs(t,v):
        e=ge*np.exp(-t/p.exc_tau_ms); i=gi*np.exp(-t/p.inh_tau_ms)
        return (p.leak_uS*(p.rest_mV-v)+e*(p.exc_reversal_mV-v)
                +i*(p.inh_reversal_mV-v)+I)/p.capacitance_nF
    ref=solve_ivp(rhs,(0,10),[p.rest_mV],rtol=1e-11,atol=1e-12).y[0,-1]
    errors=[]
    for dt in (.1,.05,.025):
        n=ConductanceNetwork(1,p,dt_ms=dt)
        n.step(I,ge,gi)
        for _ in range(round(10/dt)-1): n.step(I)
        errors.append(abs(n.v[0]-ref))
    assert errors[2]<errors[1]<errors[0] and errors[2]<.05
    print('PASS independent RK45 reference and dt convergence',errors)
    print('8/8 numerical checks; all physiological parameters illustrative')

if __name__ == '__main__':
    main()
