"""Independently verify saved audit and all RH intervention trajectories."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T181843Z')
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T062800Z/verify_and_render.py')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
# The original audit failed only while printing; verify its persisted claims independently.
audit=js(O/'CHAIN_AUDIT.json');f=Path(audit['baseline']['folder']);n=np.load(f/'NEURAL.npz');native=np.load(f/'NATIVE.npz')['rows'];body=np.load(f/'BODY.npz')
for li,r in enumerate(audit['per_leg']):
    counts=n['spikes'][:,r['cells']].sum(0)
    assert int(counts.sum())==r['spikes_full'] and int(np.count_nonzero(counts))==r['active_cells']
    assert float(native[:,1+li].mean()/24.446354755878005)==r['foot_load_mean']
    assert float(body['feet'][120,li,2])==r['foot_origin_z_at_60ms']
for r in audit['per_pool']:assert r['spikes']==int(n['spikes'][:,r['cells']].sum())
old=js(O/'AUDIT_PREREGISTRATION.json')['source_hashes'];assert old==js(O/'AUDIT_INPUT_HASHES_AFTER.json');assert all(sha(p)==h for p,h in old.items())
dump(O/'AUDIT_RECOVERY_VERIFICATION.json',dict(original_exit_code=1,failed_only_console_serialization=True,persisted_spike_load_and_foot_trajectory_recomputed=True,all_pool_spikes_recomputed=True,original_input_hashes_unchanged=True,importance='核心必需'))
# Readout identity and neural intervention identities must match preregistered files.
for fp in O.glob('*/RESULT.json'):
    folder=fp.parent;ident=js(folder/'NEURAL_INTERVENTION.json');assert sha(ident['file'])==ident['sha256']
    intervention=js(ident['file']);old=np.array(intervention['old_threshold']);new=np.array(intervention['new_threshold'])
    assert np.allclose(new-old,intervention['config'].get('threshold_shift_mV',0.),atol=0,rtol=0)
    assert intervention['membrane_conductance_state_preserved']
    assert all(v['sparsity_unchanged'] and v['untargeted_weights_unchanged'] for v in intervention['synapses'].values())
text=SOURCE.read_text()
needle="proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(S@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])"
replacement="identity=js(folder/'READOUT_IDENTITY.json'); assert sha(identity['file'])==identity['sha256']\n        config=js(identity['file']); matrix=np.array(config['matrix'])\n        proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(matrix@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])\n        if config['sensory_cut']: assert not np.any(n['currents'])"
selection="selected=[('P0 original',summary['P0']['best'],'P0'),('P1 readout cut',summary['P0']['cut'],'P0'),('P2 random',summary['P2']['best'],'P2')]"
new_selection="selected=[('P0 frozen baseline',summary['baseline'],'P0'),('P0 best intervention',summary['best'],'P0'),('P1 readout cut',summary['cut'],'P0')]"
assert text.count(needle)==1 and text.count(selection)==1
text=text.replace(needle,replacement).replace(selection,new_selection).replace('assert len(checked)==38','assert len(checked)==15').replace('total=38','total=15').replace('/s5-neural-joint-torque/20261105T062800Z','/s5-neural-joint-torque/20261105T181843Z').replace('P0_P1_P2_8x_slow.mp4','BASELINE_BEST_CUT_8x_slow.mp4')
ns={'__name__':'RH_probe_verifier','__file__':str(SOURCE)};exec(compile(text,str(Path(__file__)),'exec'),ns);ns['main']()
dump(O/'VERIFIER_ADAPTER.json',dict(source=str(SOURCE),sha256=sha(SOURCE),changes=['15 probe runs','candidate-specific matrix','neural parameter identities','baseline/best/cut panels','audit recovery checks'],importance='核心必需'))
# Neural-output/contact diagnostics from the same recorded runs.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary=js(O/'SUMMARY.json');fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
cells=next(r['cells'] for r in audit['per_leg'] if r['leg']=='RH')
for label,r in [('baseline',summary['baseline']),('best',summary['best']),('readout cut',summary['cut'])]:
    path=Path(r['folder']);a=np.load(path/'NATIVE.npz')['rows'];bb=np.load(path/'BODY.npz');nn=np.load(path/'NEURAL.npz')
    axs[0].plot(bb['time']-1.3,bb['feet'][:,5,2],label=label)
    axs[1].plot(a[:,0]-1.3,a[:,6]/24.446354755878005,label=label)
    axs[2].plot(nn['time']-1.3,np.cumsum(nn['spikes'][:,cells].sum(1)),label=label)
axs[0].set_ylabel('RH tarsus body origin z');axs[1].set_ylabel('RH foot load / weight');axs[2].set_ylabel('Cumulative RH motor spikes');axs[2].set_xlabel('Time after release (s)')
for ax in axs:ax.legend();ax.grid(alpha=.3)
fig.suptitle('RH neural activity, foot trajectory and real contact force');fig.tight_layout();fig.savefig(O/'RH_CHAIN.png',dpi=140);plt.close(fig)
print('AUDIT RECOVERED; 15 NEURAL PROBES VERIFIED',flush=True)
