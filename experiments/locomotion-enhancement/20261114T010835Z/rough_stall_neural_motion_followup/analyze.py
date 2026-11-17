"""Read only recorded neural outputs and joint excursions; no models or simulation."""
from pathlib import Path
import argparse, hashlib, json, resource, signal, sys, time
ROOT=Path(__file__).resolve().parent
BIND=ROOT.parent/'rough_stall_saved_diagnostic/SOURCE_SCHEMA_BINDINGS.json'
PYTHON='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-saved-only',action='store_true');a=p.parse_args()
    require(a.execute_parent_saved_only and sys.executable==PYTHON,'Explicit qualified-parent execution required')
    out=ROOT/'RESULT.json';require(not out.exists(),'No overwrite')
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('Saved signal followup90s exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(90)
    resource.setrlimit(resource.RLIMIT_AS,(1500*1024**2,1500*1024**2))
    import numpy as np
    own={str(x):sha(x) for x in (ROOT/'PREREG.json',Path(__file__),BIND)}
    binding=read(BIND);case=binding['cases'][0];require(Path(case['root']).name=='mild_blocks_seed17_start_repair_v1_long38p4','Exact recorded long case')
    ledgers={};arrays={}
    needed={'neural_decisions.npz':('physical_step_ms','mean','policy_action','actually_applied_action','hidden_after','brain_hidden_after','command','control_source'),'trace_1ms.npz':('target','qpos')}
    for name,keys in needed.items():
        path=Path(case['root'])/name;spec=case['archives'][name];require(sha(path)==spec['recorded_sha256'],'Recorded archive SHA drift');ledgers[str(path)]=spec['recorded_sha256']
        with np.load(path,allow_pickle=False) as z:
            for key in keys:
                value=z[key];header=spec['headers'][key]
                require(list(value.shape)==header['shape'] and value.dtype.str==header['descr'] and bool(np.isfinite(value).all()),'Recorded field integrity '+key)
                arrays[key]=value
    require(np.array_equal(arrays['physical_step_ms'],np.arange(8840)*5),'Actual5ms clock')
    require(bool(np.all(arrays['control_source'][1160:]==3)),'Unassisted source3 only')
    require(bool(np.array_equal(arrays['policy_action'][1160:],arrays['actually_applied_action'][1160:])),'Recorded policy/applied actions differ')
    qadr=[a['qpos_address'] for a in case['xml_schema']['actuators']]
    def rms(x):return float(np.sqrt(np.mean(np.square(x))))
    def variation(x):return rms(x-x.mean(0,keepdims=True))
    rows=[]
    for lo,hi in read(ROOT/'PREREG.json')['windows_ms']:
        mask=(arrays['physical_step_ms']>=lo)&(arrays['physical_step_ms']<hi);require(int(mask.sum())==1920,'Window mass')
        action=arrays['actually_applied_action'][mask];h=arrays['hidden_after'][mask];brain=arrays['brain_hidden_after'][mask];cmd=arrays['command'][mask]
        target=arrays['target'][lo:hi];q=arrays['qpos'][lo:hi][:,qadr]
        rows.append(dict(edges_s=[lo/1000,hi/1000],samples=1920,recorded_mean_RMS=rms(arrays['mean'][mask]),applied_action_RMS=rms(action),applied_action_variation_RMS=variation(action),applied_action_absmax=float(np.max(np.abs(action))),motor256_variation_RMS=variation(h),motor256_per_step_change_RMS=rms(np.diff(h,axis=0)),motor_base128_per_step_change_RMS=rms(np.diff(h[:,:128],axis=0)),motor_RIGHT64_per_step_change_RMS=rms(np.diff(h[:,128:192],axis=0)),motor_LEFT64_per_step_change_RMS=rms(np.diff(h[:,192:256],axis=0)),brain16_variation_RMS=variation(brain),brain16_per_step_change_RMS=rms(np.diff(brain,axis=0)),command_mean=cmd.mean(0).tolist(),command_std=cmd.std(0).tolist(),target_peak_to_peak_rad=np.ptp(target,axis=0).tolist(),q_peak_to_peak_rad=np.ptp(q,axis=0).tolist(),target_peak_to_peak_max_rad=float(np.ptp(target,axis=0).max()),q_peak_to_peak_max_rad=float(np.ptp(q,axis=0).max())))
    for path,digest in {**own,**ledgers}.items():require(sha(path)==digest,'Input changed during analysis')
    require(not any(x in sys.modules for x in ('torch','mujoco','physics','flygym','static_teacher')),'No model or physical backend')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;require(rss<=1500 and time.monotonic()-started<90,'Resource gate')
    result=dict(kind='SAVED_NEURAL_MOTOR_OUTPUT_FOLLOWUP_V1',rows=rows,source_hashes=own,archive_hashes=ledgers,archive_hashes_unchanged=True,recorded_policy_equals_applied_source3=True,model_loads=0,neural_forwards=0,training_updates=0,physical_steps=0,Core_constructions=0,wall_s=time.monotonic()-started,RSS_MiB=rss,interpretation='Recorded motor-output and memory variation only. Immobility or small action is not proof of neural silence, a unique mechanical cause, or consciousness. No new inference or simulation.')
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    signal.alarm(0);print(json.dumps(dict(output=str(out),wall_s=result['wall_s'],RSS_MiB=rss,neural_forwards=0,physical_steps=0)))
if __name__=='__main__':main()
