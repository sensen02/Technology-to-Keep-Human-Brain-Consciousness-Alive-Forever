"""Parent entry for frozen one-trial orchestrator plus richer late-window descriptions."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    for path,digest in json.loads((HERE/'FROZEN.json').read_text())['source_hashes'].items():
        if sha(path)!=digest: raise ValueError('Appendix drift '+path)
    sys.path.insert(0,str(BASE))
    r=load('runtime',BASE/'runtime.py')
    diagnostic=load('diagnostics',BASE/'diagnostics.py')
    original=diagnostic.summarize
    def summarize(folder,result):
        value=original(folder,result)
        import numpy as np
        with np.load(folder/'trace_1ms.npz',allow_pickle=False) as t, np.load(folder/'neural_decisions.npz',allow_pickle=False) as d, np.load(folder/'geometry_1ms.npz',allow_pickle=False) as g:
            for window in value['windows']:
                lo,hi=window['absolute_interval_ms'];part=slice(lo,hi)
                root=t['qpos'][part,:3];com=t['com'][part];yaw=np.unwrap(g['thorax_yaw'][part]);target=t['target'][part]
                window.update(root_net_xyz=(root[-1]-root[0]).tolist(),COM_net_xyz=(com[-1]-com[0]).tolist(),root_XY_path_length=float(np.linalg.norm(np.diff(root[:,:2],axis=0),axis=1).sum()),COM_XY_path_length=float(np.linalg.norm(np.diff(com[:,:2],axis=0),axis=1).sum()),yaw_net_rad=float(yaw[-1]-yaw[0]),yaw_path_rad=float(np.abs(np.diff(yaw)).sum()),target_max_joint_range_rad=float(np.ptp(target,axis=0).max()),policy_action_RMS=float(np.sqrt(np.mean(d['policy_action'][lo//5:hi//5]**2))),load_mean_per_leg=t['loads'][part].mean(axis=0).tolist(),loaded_fraction_per_leg_at_original_point1W=(t['loads'][part]>=.1*24.446354755878005).mean(axis=0).tolist(),loaded_threshold_scope='source STANCE_LOAD=.1*WEIGHT; descriptive samples, not new cycle acceptance',heading_projected_COM_path=float((np.diff(com[:,:2],axis=0)*np.column_stack((np.cos(yaw[:-1]),np.sin(yaw[:-1])))).sum()),original_forward_gate_reference_speed_x=float((com[-1,0]-com[0,0])/(t['time'][hi-1]-t['time'][lo])))
            value['last9p6']=value['windows'][-1]
        value.update(description='Whole-run five gates/cycles can hide late arrest. Late-window raw values are separate; no added threshold or claim of sustained progress from whole PASS.',appendix_freeze_sha256=sha(HERE/'FROZEN.json'))
        return value
    diagnostic.summarize=summarize
    guard=r.read(BASE/'PARENT_ZERO_STEP_GUARD.json') if '--parent-resource-released' in sys.argv else None
    if guard is not None and (guard.get('guard_appendix_freeze_sha256')!=sha(HERE/'FROZEN.json') or guard.get('base_Core_constructors')!=1):
        raise ValueError('Exact appendix compiled guard required')
    orchestrator=load('_frozen_matched_flat_orchestrator',BASE/'orchestrate.py')
    orchestrator.main()
if __name__=='__main__': main()
