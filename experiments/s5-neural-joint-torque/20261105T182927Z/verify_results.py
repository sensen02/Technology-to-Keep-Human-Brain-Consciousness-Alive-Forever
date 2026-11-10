"""Independently validate relative-time scoring and neural torque for all phases."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path[:0]=[str(R),str(R/'tools')]
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T182927Z')
SOURCE=R/'experiments/s5-neural-joint-torque/20261105T062800Z/verify_and_render.py'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
import train_dynamic_support as dst
import dynamic_support_metrics as dsm
summary=js(O/'SUMMARY.json');records=js(O/'PROGRESS.json');cover=js(O/'COVERAGE.json');limits=np.array([r['limit'] for r in cover['joints']]);checks=[]
assert len(records)==19
for r in records:
    folder=Path(r['folder']);phase=folder.parent;cfg=js(folder/'PHASE_IDENTITY.json');assert sha(cfg['intervention_file'])==cfg['intervention_sha256'];intervention=js(cfg['intervention_file'])
    assert intervention['synapses']['We']['scale']==r['dose']
    assert all(v['sparsity_unchanged'] and v['untargeted_weights_unchanged'] for v in intervention['synapses'].values())
    mid=js(folder/'READOUT_IDENTITY.json');assert sha(mid['file'])==mid['sha256'];mcfg=js(mid['file']);matrix=np.array(mcfg['matrix']);n=np.load(folder/'NEURAL.npz');body=np.load(folder/'BODY.npz');a=np.load(folder/'NATIVE.npz')['rows'];warm=np.load(phase/'WARM_P0/BODY.npz');warmnative=np.load(phase/'WARM_P0/NATIVE.npz')['rows']
    predicted=np.array([np.clip(limits/100*(matrix@rr),-limits,limits) for rr in n['source_rates']])
    if mcfg['cut']:predicted[:]=0
    assert np.allclose(predicted,n['applied'],rtol=0,atol=1e-13)
    assert np.array_equal(n['rates'][:-1],n['source_rates'][1:])
    if mcfg['sensory_cut']:assert not np.any(n['currents'])
    assert np.array_equal(body['qpos'][0],warm['qpos'][-1]) and np.array_equal(body['qvel'][0],warm['qvel'][-1])
    assert np.array_equal(body['ctrl'][1::2],n['applied']) and np.isfinite(body['qpos']).all()
    assert a.shape==(16001,20) and np.allclose(np.diff(a[:,0]),.0000125,atol=1e-12,rtol=0)
    assert abs(a[0,0]-r['release_time_s'])<1e-9 and abs(a[-1,0]-r['release_time_s']-.2)<1e-9
    assist=dict(foot_support_mean=float(warmnative[:,1:7].sum(1).mean()/dsm.WEIGHT),com_z_mean=float(warmnative[:,10].mean()))
    shifted=a.copy();shifted[:,0]-=r['release_time_s']-1.3
    J,parts=dst.objective(shifted,assist);assert abs(J-r['J'])<1e-12
    full=dsm.evaluate_native(assist,a);full['checks']['D1_no_nonfoot_ground_force']=bool(np.max(a[:,7])==0)
    assert full['checks']==r['validation']['checks']
    checks.append(dict(folder=str(folder),release_time_s=r['release_time_s'],J_recomputed=J,command_error=float(np.max(abs(predicted-n['applied']))),continuity_exact=True))
inputs=js(O/'INPUT_HASHES_BEFORE.json');assert inputs==js(O/'INPUT_HASHES_AFTER.json');assert all(sha(p)==h for p,h in inputs.items())
dump(O/'VERIFICATION.json',dict(total=19,checks=checks,input_hashes_unchanged=True,relative_time_scoring_verified=True,all_phase_duration_s=.2))
# Reuse the checked renderer for the selection phase, without writing into historical dirs.
text=SOURCE.read_text()
old="proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(S@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])"
new="identity=js(folder/'READOUT_IDENTITY.json'); assert sha(identity['file'])==identity['sha256']\n        config=js(identity['file']); matrix=np.array(config['matrix'])\n        proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(matrix@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])\n        if config['sensory_cut']: assert not np.any(n['currents'])"
selection="selected=[('P0 original',summary['P0']['best'],'P0'),('P1 readout cut',summary['P0']['cut'],'P0'),('P2 random',summary['P2']['best'],'P2')]"
newselection="selected=[('P0 dose 1 baseline',summary['baseline'],'P0'),('P0 selected RH dose',summary['best'],'P0'),('P1 readout cut',summary['cut'],'P0')]"
assert text.count(old)==1 and text.count(selection)==1
text=text.replace(old,new).replace(selection,newselection).replace('assert len(checked)==38','assert len(checked)==11').replace('total=38','total=11').replace('/s5-neural-joint-torque/20261105T062800Z','/s5-neural-joint-torque/20261105T182927Z/PHASE_1.300').replace("O/'INPUT_HASHES_BEFORE.json'","O.parent/'INPUT_HASHES_BEFORE.json'").replace("O/'INPUT_HASHES_AFTER.json'","O.parent/'INPUT_HASHES_AFTER.json'").replace('P0_P1_P2_8x_slow.mp4','BASELINE_DOSE_CUT_8x_slow.mp4')
ns={'__name__':'dose_phase_renderer','__file__':str(SOURCE)};exec(compile(text,str(Path(__file__)),'exec'),ns);ns['main']()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(12,5));doses=sorted(summary['dose_results'],key=lambda r:r['dose']);axs[0].plot([r['dose'] for r in doses],[r['J'] for r in doses],'o-');axs[0].set_xlabel('RH incoming excitation multiplier');axs[0].set_ylabel('Loss J (lower is better)');axs[0].set_title('Selection phase: release at 1.300s')
pairs=summary['phase_pairs'];times=[r['release_time_s'] for r in pairs]
axs[1].plot(times,[r['baseline']['J'] for r in pairs],'o-',label='dose1');axs[1].plot(times,[r['treated']['J'] for r in pairs],'o-',label='frozen selected dose');axs[1].set_xlabel('Held-out release time (s)');axs[1].set_ylabel('Loss J');axs[1].legend();axs[1].set_title('No retuning across release phases')
for ax in axs:ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(O/'DOSE_PHASE.png',dpi=150);plt.close(fig)
dump(O/'VERIFIER_ADAPTER.json',dict(source=str(SOURCE),source_sha256=sha(SOURCE),changes=['selection phase only11 runs','candidate matrix identities','baseline/selected/cut panels','parent hash records'],importance='核心必需'))
print('ALL19 DOSE/PHASE RUNS VERIFIED',flush=True)
