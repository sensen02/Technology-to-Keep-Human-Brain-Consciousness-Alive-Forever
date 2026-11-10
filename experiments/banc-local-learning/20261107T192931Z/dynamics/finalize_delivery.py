"""Render hypothesis diagnostics and mirror scoped evidence; zero model steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
DEST=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/dynamics')
OLD=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout/TRAIN_TRACE.npz')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def draw():
    os.environ['MPLCONFIGDIR']=str(HERE/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(HERE/'SOURCE_TRACE.npz') as z:
        t=z['time_s'];rates=z['mn_rates_hz'];step_t=z['every_step_time_ms']/1000;vmin=z['voltage_min_mV']
    with np.load(OLD) as z:oldrates=z['mn_rates_hz']
    with np.load(HERE/'held_nominal.npz') as z:held=z['mn_rates_hz']
    with np.load(HERE/'held_refined.npz') as z:fine=z['mn_rates_hz']
    result=json.loads((HERE/'PROBE_RESULTS.json').read_text())['results']['full_source_trace']
    fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    fig.suptitle('Conductance-adaptation hypothesis | fixed .02 uS | engineering checks passed, physiology uncalibrated',fontsize=12)
    ax[0,0].plot(step_t,vmin,label='All-neuron minimum, every .5ms step');ax[0,0].axhline(-80,c='orange',ls='--',label='Derived lower bound');ax[0,0].axhline(-90,c='red',ls=':',label='Unchanged engineering floor');ax[0,0].set(title='Exact prior 556-port sensory input; zero DN',xlabel='Neural time (s)',ylabel='Minimum voltage (mV)');ax[0,0].legend(fontsize=8)
    ax[0,1].plot(t,oldrates.mean(axis=1),label='Prior current adaptation',alpha=.8);ax[0,1].plot(t,rates.mean(axis=1),label='New conductance adaptation',alpha=.8);ax[0,1].set(title='Identical input, start-of-bin MN features',xlabel='Source time (s)',ylabel='Mean MN filtered rate (Hz)');ax[0,1].legend(fontsize=8)
    legs=('lf','lm','lh','rf','rm','rh');ax[1,0].bar([x.upper() for x in legs],[result['per_leg'][x]['recruited'] for x in legs]);ax[1,0].set(title='209 / 323 MN recruited over 9.085 s',xlabel='Source MN leg label',ylabel='Distinct neurons with spikes')
    ax[1,1].plot((np.arange(len(held))+1)*.5,held.mean(axis=1),label='.5ms');ax[1,1].plot((np.arange(len(fine))+1)*.25,fine.mean(axis=1),label='.25ms');ax[1,1].set(title='Held physical-input sample: timestep sensitivity',xlabel='Neural time (ms)',ylabel='Mean MN filtered rate (Hz)');ax[1,1].legend(fontsize=8)
    fig.savefig(HERE/'HYPOTHESIS_DIAGNOSTICS.png',dpi=160);plt.close(fig);shutil.rmtree(HERE/'.plot_cache')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reviewed',action='store_true');args=parser.parse_args()
    if not args.reviewed:draw()
    else:assert (HERE/'HYPOTHESIS_DIAGNOSTICS.png').is_file()
    independent=json.loads((HERE/'INDEPENDENT_VERIFICATION.json').read_text());assert independent['status']=='PASS'
    expected='c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250';assert sha(HERE/'module.py')==expected
    assert sha(HERE/'SOURCE_TRACE.npz')=='8a73c8a221b0f752e7c53f9e129fcf9747c42dc3ead6209a7ce912f430e70847'
    status=subprocess.run(['git','status','--short','--untracked-files=no'],cwd='/run/media/sensen/Data2/cell_wound_prototype',text=True,capture_output=True,check=True)
    (HERE/'TRACKED_STATUS.txt').write_text(status.stdout)
    record={'status':'FIXED_HYPOTHESIS_ENGINEERING_VERIFIED_NOT_PHYSIOLOGY','completed_neural_steps':21212,'net_step_invocations':21236,'expected_rejected_net_calls':24,'manual_full_vector_reference_iterations':632,'manual_scalar_row_checks':256,'saved_event_reconstruction_iterations':18170,'physical_steps':0,'training_updates':0,'jobs':{'initial_main_runner_parse_failure':'bash206 exit1;0steps;FAILED_FIRST_RUN.json','main':'bash207 exit0;20570steps','independent':'bash208 exit0;632steps+24rejectedcalls','saved_trace':'bash211 exit0;0networksteps','API_smoke':'10steps;exit0;absolute cwd /'},'frozen_module_sha256':sha(HERE/'module.py'),'frozen_config_sha256':sha(HERE/'FIXED_CONFIG.json'),'source_trace_sha256':sha(HERE/'SOURCE_TRACE.npz'),'full_final_state_sha256':sha(HERE/'FULL_FINAL_STATE.npz'),'visual_review':'read_image opened HYPOTHESIS_DIAGNOSTICS.png: allstep voltage bounds, matched prior/new features,6leg recruitment,held timestep plots inspected; no behavior visualization' if args.reviewed else 'pending read_image; figure rendered','prior_artifacts_engine_unchanged':True,'scope':'writes only new workspace/Desktop dynamics; no biology or walking claim'}
    (HERE/'EXECUTION_RECORD.json').write_text(json.dumps(record,indent=2)+'\n')
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='DELIVERY_MANIFEST.json'}
    manifest={'workspace':str(HERE),'desktop':str(DEST),'files':files,'completed_neural_steps':21212,'visual_review_complete':args.reviewed}
    (HERE/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    DEST.mkdir(parents=True,exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    assert all(sha(HERE/name)==sha(DEST/name) for name in list(files)+['DELIVERY_MANIFEST.json'])
    print(json.dumps({'files_verified':len(files)+1,'bytes':sum(v['bytes'] for v in files.values()),'visual_review_complete':args.reviewed,'module_sha256':sha(HERE/'module.py'),'config_sha256':sha(HERE/'FIXED_CONFIG.json'),'physical_steps':0,'neural_steps_this_command':0,'desktop':str(DEST)},indent=2))

if __name__=='__main__':main()
