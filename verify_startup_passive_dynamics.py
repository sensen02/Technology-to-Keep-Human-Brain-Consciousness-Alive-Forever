"""Independent NumPy-only reconstruction of startup diagnostic gates and plot."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output
    results=json.loads((out/'RESULTS.json').read_text())
    ledger=json.loads((out/'CALL_LEDGER.json').read_text())
    assert json.loads((out/'COMPLETE.json').read_text())['status']=='COMPLETE'
    assert ledger==dict(model_load=6,forward=2406,step=2400,completed=[r['condition']['name'] for r in results],active=None)
    prereg=json.loads((out/'PREREG.json').read_text())
    for name,h in prereg['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    checks=[]; baseline=None
    for r in results:
        path=Path(r['trajectory']); assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        with np.load(path,allow_pickle=False) as z: a={k:z[k] for k in z.files}
        assert a['qpos'].shape==(401,49) and a['qvel'].shape==(401,48)
        assert np.allclose(a['time'],np.arange(401)*.0001,rtol=0,atol=1e-12)
        assert all(np.isfinite(x).all() for x in a.values())
        assert not any(a[k].any() for k in ('ctrl','act','qfrc_applied','xfrc_applied','warnings'))
        if baseline is None: baseline=a
        assert np.array_equal(a['qpos'][0],baseline['qpos'][0]) and not a['qvel'][0].any()
        h=a['qpos'][:,2]/a['qpos'][0,2]
        quat=a['qpos'][:,3:7]; q0=quat[0]
        ang=2*np.arccos(np.clip(abs(quat@q0)/(np.linalg.norm(quat,axis=1)*np.linalg.norm(q0)),0,1))
        load=np.sum(a['foot_force'][:,:,2],axis=1)/r['weight_native']
        gates=np.array([h>=.85,ang<=.35,np.count_nonzero(a['contacts']>0,axis=1)>=3,load>=.8,a['nonfoot_depth']<=1e-12,a['limit_excess']<=1e-6]).T
        assert np.array_equal(gates,a['gates'])
        assert np.allclose(load,a['load'],rtol=0,atol=1e-14)
        total=float(np.trapezoid(gates.all(1).astype(float),a['time'])*1000)
        assert abs(total-r['all_gates_support_ms'])<1e-12
        first={k:(float(a['time'][np.flatnonzero(~gates[:,i])[0]]*1000) if (~gates[:,i]).any() else None) for i,k in enumerate(('height','angle','legs','load','nonfoot','limit'))}
        assert first==r['first_failures_ms']
        cfg=r['condition']
        assert np.array_equal(a['jnt_stiffness'],baseline['jnt_stiffness']*cfg['stiffness'])
        assert np.array_equal(a['dof_damping'],baseline['dof_damping']*cfg['damping'])
        if cfg['bias']==0: assert np.all(a['actuator_biastype']==0) and not a['actuator_force'].any()
        else: assert np.array_equal(a['actuator_biasprm'][:,2],baseline['actuator_biasprm'][:,2]*cfg['bias'])
        for ax,y in zip(axes,(load,h,gates.all(1).astype(int))): ax.plot(a['time']*1000,y,label=cfg['name'])
        checks.append(dict(condition=cfg['name'],passed=True,all_gates_support_ms=total))
    for ax,label in zip(axes,('Foot load / native weight','Root height / initial height','All six original gates')):
        ax.set_ylabel(label); ax.grid(alpha=.3)
    axes[0].axhline(.8,color='black',ls='--'); axes[1].axhline(.85,color='black',ls='--')
    axes[0].legend(fontsize=8,ncol=2); axes[-1].set_xlabel('Physical time (ms)')
    fig.suptitle('Full-body zero-control startup diagnostics — NOT neural support or walking')
    fig.tight_layout(); fig.savefig(out/'STARTUP_COMPARISON.png',dpi=150); plt.close(fig)
    (out/'INDEPENDENT_CHECK.json').write_text(json.dumps(dict(passed=True,checks=checks,ledger=ledger,new_physics_calls=0),indent=2)+'\n')
    print(json.dumps(checks))


if __name__=='__main__': main()
