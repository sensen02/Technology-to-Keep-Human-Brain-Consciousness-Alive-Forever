"""Fixed scalar decay/pulse diagnostic, no body or production modifications."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from tools_s2_activation_reference import rhs

OUT=Path('/home/sensen/Desktop/engine修复验收/S2-switched-activation')
DTS=(1e-4,5e-5,2.5e-5,1.25e-5)


def control(t,case):
    if case not in ('decay','pulse'):raise ValueError('unknown case')
    return .1 if case=='pulse' and t<.001 else 0.


def euler(dt,case,floor):
    if not math.isfinite(dt) or dt<=0 or abs(.001/dt-round(.001/dt))>1e-10:raise ValueError('event not aligned')
    steps=round(.01/dt);a=.1 if case=='decay' else 0.;act=[]
    for k in range(steps):
        # Integer event index avoids rounding the left control boundary.
        u=.1 if case=='pulse' and k<round(.001/dt) else 0.
        a+=dt*rhs(a,max(u,floor),.0001,.0004);act.append(a)
    return np.arange(1,steps+1)*dt,np.array(act)


def reference(times,case,floor,strict=False):
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or not np.isfinite(times).all() or np.any(times<=0) or np.any(times>.01):
        raise ValueError('reference times outside positive 10ms interval')
    a=.1 if case=='decay' else 0.;answer=np.full_like(times,np.nan)
    for start,end,u in ((0.,.001,.1 if case=='pulse' else 0.),(.001,.01,0.)):
        sol=solve_ivp(lambda t,y:[rhs(y[0],max(u,floor),.0001,.0004)],(start,end),[a],method='DOP853',rtol=2e-13 if strict else 1e-12,atol=2e-15 if strict else 1e-14,max_step=5e-6 if strict else 1e-5,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        mask=(times>start)&(times<=end);answer[mask]=sol.sol(times[mask])[0];a=float(sol.y[0,-1])
    if not np.isfinite(answer).all():raise RuntimeError('unfilled reference time')
    return answer


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=False);records=[];fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,case in enumerate(('decay','pulse')):
        for j,floor in enumerate((.0001,0.)):
            ax=axes[i,j]
            for dt in DTS:
                t,a=euler(dt,case,floor);r=reference(t,case,floor);strict=reference(t,case,floor,True)
                err=float(np.max(np.abs(a-r)));name=f'{case}_floor{floor}_dt{dt}.npz'
                np.savez_compressed(OUT/name,time_s=t,euler=a,reference=r,stricter_reference=strict)
                records.append(dict(case=case,floor=floor,dt_s=dt,max_error=err,early_error=float(np.max(np.abs(a[t<=.001]-r[t<=.001]))),late_error=float(np.max(np.abs(a[t>.001]-r[t>.001]))),reference_crosscheck=float(np.max(np.abs(r-strict))),engineering_screen_pass=err<=.001,npz=name))
                ax.plot(t*1000,a-r,label=f'{dt*1e6:g}us')
            ax.set_title(case+' floor='+str(floor));ax.set_xlabel('Time(ms)');ax.set_ylabel('Activation error');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Scalar switching errors — engineering screen, NOT body convergence')
    fig.tight_layout();fig.savefig(OUT/'switch-errors.png',dpi=150)
    with (OUT/'report.json').open('x') as f:json.dump(dict(records=records,tolerance=.001,tolerance_scope='declared scalar engineering target only',biological_validation=False,production_changed=False),f,indent=2);f.write('\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
