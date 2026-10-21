"""Fixed scalar method comparison; not MuJoCo production integration."""
import json
from pathlib import Path
import numpy as np
from tools_s2_activation_reference import rhs
from tools_s2_switched_activation import reference

OUT=Path('/home/sensen/Desktop/engine修复验收/S2-scalar-methods')
GRIDS={'euler':(6.25e-6,3.125e-6,1.5625e-6),'rk4':(1e-4,5e-5,2.5e-5,1.25e-5)}


def step(a,dt,f,method):
    if method=='euler':return a+dt*f(a)
    if method!='rk4':raise ValueError('unknown method')
    k1=f(a);k2=f(a+dt*k1/2);k3=f(a+dt*k2/2);k4=f(a+dt*k3)
    return a+dt*(k1+2*k2+2*k3+k4)/6


def simulate(dt,case,floor,method):
    if not np.isfinite(dt) or dt<=0 or abs(.001/dt-round(.001/dt))>1e-9:raise ValueError('unaligned grid')
    if case not in ('decay','pulse'):raise ValueError('unknown case')
    a=.1 if case=='decay' else 0.;values=[]
    for k in range(round(.01/dt)):
        u=max(.1 if case=='pulse' and k<round(.001/dt) else 0.,floor)
        a=step(a,dt,lambda x:rhs(x,u,.0001,.0004),method);values.append(a)
    return np.arange(1,len(values)+1)*dt,np.array(values)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=False);records=[];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method,dts in GRIDS.items():
        for case in ('decay','pulse'):
            for floor in (.0001,0.):
                for dt in dts:
                    t,a=simulate(dt,case,floor,method);r=reference(t,case,floor);err=float(np.max(np.abs(a-r)))
                    name=f'{method}_{case}_{floor}_{dt}.npz';np.savez_compressed(OUT/name,time_s=t,activation=a,reference=r)
                    records.append(dict(method=method,case=case,floor=floor,dt_s=dt,max_error=err,screen_pass=err<=.001,rhs_calls=len(t)*(4 if method=='rk4' else 1),npz=name))
                    if floor==0:
                        ax=axes[0 if case=='decay' else 1];ax.scatter(dt*1e6,err,label=method if dt==dts[0] else None,marker='o' if method=='euler' else 'x')
    for ax,case in zip(axes,('decay','pulse')):
        ax.set_xscale('log');ax.set_yscale('log');ax.axhline(.001,color='gray',linestyle='--',label='engineering target');ax.set_title(case+' zero floor');ax.set_xlabel('Step (us)');ax.set_ylabel('Maximum activation error');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Fixed scalar integration comparison — NOT body or biological validation')
    fig.tight_layout();fig.savefig(OUT/'method-errors.png',dpi=150)
    with (OUT/'report.json').open('x') as f:json.dump(dict(records=records,tolerance=.001,production_changed=False,biological_validation=False),f,indent=2);f.write('\n')
    print(json.dumps([c for c in records if c['case']=='pulse' and c['floor']==0],indent=2))


if __name__=='__main__':main()
