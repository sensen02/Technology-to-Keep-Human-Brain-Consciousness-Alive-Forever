"""Independent scalar muscle ODE reference for archived clock cases."""
import json
from pathlib import Path
import math
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path('/home/sensen/Desktop/engine修复验收/S2-activation-clock')
OUT=Path('/home/sensen/Desktop/engine修复验收/S2-activation-reference')
SOURCE='https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_util_misc.c'


def rhs(act,u,ta,td):
    a=min(1.,max(0.,act));delta=min(1.,max(0.,u))-act
    return delta/(ta*(.5+1.5*a) if delta>0 else td/(.5+1.5*a))


def implicit_reference(t,u,ta):
    if u==0 or t==0:return 0.
    # Monotone activation from0 to u<=1; analytic separated ODE root.
    lo=0.;hi=np.nextafter(u,0.)
    def elapsed(a):return ta*((.5+1.5*u)*math.log(u/(u-a))-1.5*a)
    if t>=elapsed(hi):return u
    for _ in range(80):
        mid=(lo+hi)/2
        if elapsed(mid)<t:lo=mid
        else:hi=mid
    return (lo+hi)/2


def main():
    OUT.mkdir(exist_ok=False);r=json.loads((ROOT/'report.json').read_text());results=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for c in r['cases']:
        z=np.load(ROOT/c['npz'],allow_pickle=False);prm=c['dynprm'];assert prm[2]==0 and c['dyntype']==4
        u=max(c['ctrl'],.0001) if c['branch']=='native_floor' else c['ctrl']
        sol=solve_ivp(lambda t,y:[rhs(y[0],u,prm[0],prm[1])],(0,.01000000000001),[0.],method='DOP853',rtol=1e-12,atol=1e-14,max_step=1e-5,dense_output=True)
        assert sol.success
        ref=np.array([implicit_reference(t,u,prm[0]) for t in z['time_s']]);numeric=sol.sol(z['time_s'])[0]
        results.append(dict(branch=c['branch'],ctrl=c['ctrl'],dt_s=c['dt_s'],reference_solver_max_difference=float(np.max(np.abs(ref-numeric))),activation_max_error=float(np.max(np.abs(z['activation']-ref))),first_error=float(z['activation'][0]-ref[0]),final_error=float(z['activation'][-1]-ref[-1])))
        np.savez_compressed(OUT/c['npz'],time_s=z['time_s'],implicit_reference=ref,dop853_reference=numeric,archived_activation=z['activation'])
        if c['branch']=='native_floor' and c['ctrl']==.1:ax.plot(z['time_s']*1000,z['activation']-ref,label=f"{c['dt_s']*1e6:g} us")
    ax.set_xlabel('Time (ms)');ax.set_ylabel('Activation minus scalar reference');ax.legend();ax.grid(alpha=.2)
    ax.set_title('Independent scalar ODE reference — NOT body convergence')
    fig.tight_layout();fig.savefig(OUT/'reference-errors.png',dpi=150)
    with (OUT/'report.json').open('x') as f:json.dump(dict(cases=results,source=SOURCE,scope='hard-switch muscle dynamics; constant control/zero initial activation',solver=dict(method='DOP853',rtol=1e-12,atol=1e-14,max_step=1e-5),biological_validation=False),f,indent=2);f.write('\n')
    print(json.dumps([v for v in results if v['branch']=='native_floor' and v['ctrl']==.1],indent=2))


if __name__=='__main__':main()
