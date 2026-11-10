"""Independently verify causal interventions and edge-weight inference, then replay."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T153224Z')
D=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T152908Z')
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T062800Z/verify_and_render.py')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def diagnose_verify():
    cover=js(D/'COVERAGE.json');limit=np.array([r['limit'] for r in cover['joints']]);warm=np.load(D/'WARM_P0/BODY.npz');checked=[]
    for r in js(D/'DIAGNOSIS.json')['results']:
        f=Path(r['folder']);p=js(f/'PARAMETERS.json');cfg=js(f/'READOUT_MATRIX.json');z=np.array(p['z']);S=np.array(cfg['matrix']);n=np.load(f/'NEURAL.npz');b=np.load(f/'BODY.npz');a=np.load(f/'NATIVE.npz')['rows']
        proposal=np.array([np.clip(limit/100*np.exp(z[66:])*(S@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])
        if cfg['cut']:proposal[:]=0
        assert np.allclose(proposal,n['applied'],atol=1e-13,rtol=0)
        assert np.array_equal(n['rates'][:-1],n['source_rates'][1:])
        assert np.array_equal(b['qpos'][0],warm['qpos'][-1]) and np.array_equal(b['qvel'][0],warm['qvel'][-1])
        assert a.shape==(16001,20) and np.isfinite(a).all()
        if cfg['sensory_cut']:assert not np.any(n['currents'])
        if cfg['cut']:assert not np.any(b['ctrl'])
        assert np.array_equal(b['ctrl'][1::2],n['applied'])
        checked.append(dict(label=r['label'],command_error=float(np.max(abs(proposal-n['applied']))),finite=True))
    before=js(D/'INPUT_HASHES_BEFORE.json');assert before==js(D/'INPUT_HASHES_AFTER.json')
    assert all(sha(p)==h for p,h in before.items())
    dump(D/'VERIFICATION.json',dict(checked=checked,count=len(checked),input_hashes_unchanged=True,exact_baseline=js(D/'DIAGNOSIS.json')['baseline_exact']))

def main():
    diagnose_verify()
    text=SOURCE.read_text()
    old="proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(S@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])"
    new="identity=js(folder/'READOUT_IDENTITY.json'); assert sha(identity['file'])==identity['sha256']\n        config=js(identity['file']); matrix=np.array(config['matrix'])\n        proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(matrix@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])\n        if config['sensory_cut']: assert not np.any(n['currents'])"
    assert text.count(old)==1
    text=text.replace(old,new).replace('assert len(checked)==38','assert len(checked)==50').replace('total=38','total=50').replace('/s5-neural-joint-torque/20261105T062800Z','/s5-neural-joint-torque/20261105T153224Z')
    namespace={'__name__':'edge_readout_verifier','__file__':str(SOURCE)};exec(compile(text,str(Path(__file__)),'exec'),namespace);namespace['main']()
    dump(O/'VERIFIER_ADAPTER.json',dict(source=str(SOURCE),sha256=sha(SOURCE),changes=['candidate-specific matrix identity checked','50 evaluations','sensory cut current zero assertion'],importance='核心必需'))
    # Compact causal diagnostic figure uses the same saved result values.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res=js(D/'DIAGNOSIS.json')['results'];fig,axs=plt.subplots(1,2,figsize=(13,6))
    labels=[r['label'] for r in res];y=np.arange(len(res))
    for ax,vals,label in [(axs[0],[r['J'] for r in res],'Loss J (lower is better)'),(axs[1],[r['parts']['foot_support_mean'] for r in res],'Mean foot load / weight')]:
        ax.barh(y,vals);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel(label);ax.grid(axis='x',alpha=.3)
    fig.suptitle('Frozen best causal interventions: same physical and neural release state');fig.tight_layout();fig.savefig(D/'CAUSAL_DIAGNOSIS.png',dpi=140);plt.close(fig)
    print('CAUSAL + ESCALATION VERIFIED',flush=True)
if __name__=='__main__':main()
