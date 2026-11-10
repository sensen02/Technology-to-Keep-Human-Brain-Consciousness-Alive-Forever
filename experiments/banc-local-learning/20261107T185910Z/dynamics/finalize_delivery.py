"""Make a diagnostic figure and byte-verified scoped Desktop delivery; no neural steps."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
DEST=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/dynamics')
os.environ['MPLCONFIGDIR']=str(HERE/'.plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface')
    with (source/'LOCAL_MOTOR_READOUT_PORTS.csv').open() as f:legs=np.array([r['nerve_matched_legs'] for r in csv.DictReader(f) if r['readout_eligible_single_leg']=='True'])
    with np.load(HERE/'recorded_sample_nominal.npz') as z:rates=z['MN_filtered_Hz'];volts=z['voltage_range_mV']
    with np.load(HERE/'recorded_sample_refined.npz') as z:fine_rates=z['MN_filtered_Hz'];fine_volts=z['voltage_range_mV']
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    fig.suptitle('BANC local engineering probe | fixed .02 uS fallback | calibration NOT PASSED',fontsize=13)
    t=np.arange(1,len(rates)+1)*.5;tf=np.arange(1,len(fine_rates)+1)*.25
    for leg in ('lf','lm','lh','rf','rm','rh'):axes[0,0].plot(t,rates[:,legs==leg].mean(axis=1),label=leg.upper())
    axes[0,0].set(title='One recorded physical input held constant; zero DN',xlabel='Neural time (ms)',ylabel='Leg mean filtered MN rate (Hz)');axes[0,0].legend(ncol=3,fontsize=8)
    axes[0,1].plot(t,volts[:,0],label='min, dt .5 ms');axes[0,1].plot(tf,fine_volts[:,0],label='min, dt .25 ms');axes[0,1].axhline(-90,color='red',ls='--',label='original floor: FAILED');axes[0,1].set(title='All-neuron minima recorded every step',xlabel='Neural time (ms)',ylabel='Voltage (mV)');axes[0,1].legend(fontsize=8)
    axes[1,0].plot(t,rates.mean(axis=1),label='dt .5 ms');axes[1,0].plot(tf,fine_rates.mean(axis=1),label='dt .25 ms');axes[1,0].set(title='All 323 MN: aggregate sensitivity, not convergence proof',xlabel='Neural time (ms)',ylabel='Mean filtered rate (Hz)');axes[1,0].legend(fontsize=8)
    summary=json.loads((HERE/'RECORDED_SAMPLE_RESULT.json').read_text())
    r=summary['results'][0];leg_order=('lf','lm','lh','rf','rm','rh')
    axes[1,1].bar([s.upper() for s in leg_order],[r['per_leg'][s]['recruited'] for s in leg_order]);axes[1,1].set(title='144 of 323 MN recruited through original strict edges',xlabel='MN leg label',ylabel='Distinct MN with spikes')
    fig.savefig(HERE/'DYNAMICS_DIAGNOSTICS.png',dpi=160);plt.close(fig)
    shutil.rmtree(HERE/'.plot_cache')
    # No task-owned temporary directories or bytecode need keeping.
    status=subprocess.run(['git','status','--short','--untracked-files=no'],cwd='/run/media/sensen/Data2/cell_wound_prototype',capture_output=True,text=True,check=True)
    (HERE/'TRACKED_STATUS.txt').write_text(status.stdout)
    report={'status':'NUMERICAL_IDENTITY_AND_PROPAGATION_VERIFIED_CALIBRATION_FAILED','neural_steps':9422,'manual_reference_iterations':212,'manual_scalar_row_checks':128,'physical_steps':0,'training_updates':0,'configuration':{'module_sha256':sha(HERE/'module.py'),'chosen_config_sha256':sha(HERE/'CHOSEN_CONFIG.json'),'immutable_after_parent_handoff':True},'source_prereg_sha256':sha(HERE/'PREREGISTRATION.json'),'independent_verification_sha256':sha(HERE/'INDEPENDENT_VERIFICATION.json'),'module_own_paths_only':True,'visual_review':'read_image opened Desktop DYNAMICS_DIAGNOSTICS.png: all6leg traces/recruitment, timestep comparison and failed -90mV floor visibly present; neural diagnostic only, not a behavior visualization'}
    (HERE/'EXECUTION_RECORD.json').write_text(json.dumps(report,indent=2)+'\n')
    artifacts={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='DELIVERY_MANIFEST.json'}
    manifest={'workspace':str(HERE),'desktop':str(DEST),'files':artifacts,'physical_steps_this_finalizer':0,'neural_steps_this_finalizer':0}
    (HERE/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    DEST.mkdir(parents=True,exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    assert all(sha(HERE/name)==sha(DEST/name) for name in list(artifacts)+['DELIVERY_MANIFEST.json'])
    print(json.dumps({'files_byte_verified':len(artifacts)+1,'total_artifact_bytes':sum(x['bytes'] for x in artifacts.values()),'desktop':str(DEST),'neural_steps':0,'physical_steps':0,'module_sha256':sha(HERE/'module.py'),'config_sha256':sha(HERE/'CHOSEN_CONFIG.json')},indent=2))

if __name__=='__main__':main()
