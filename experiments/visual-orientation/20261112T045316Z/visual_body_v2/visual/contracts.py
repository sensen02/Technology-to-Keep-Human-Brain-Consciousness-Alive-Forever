"""Exact savedactualvisualbody2 reader. Stdlib-only until explicitarrayload."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;VIS=ROOT.parent
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd';VISUAL_SHA='79dbcdcd38c1c3de94588bff9923b3758f82b1abbc63f017e499ece464340fee'
NORMAL=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left');CONTROLS=('V_left_occluded','V_flip_fixedframe');IDS=NORMAL+CONTROLS

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,message):
 if not v:raise ValueError(message)
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def unchanged(hashes):
 for p,h in hashes.items():require(sha(p)==h,'visual input drift '+p)
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def load(condition):
 import numpy as np
 require(condition in IDS,'registered visualbody2condition required');folder=ROOT/'trials'/condition;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');ident=read(folder/'IDENTITY.json');hand=read(folder/'HANDOVER.json')
 require(a['integrity_all_pass'] is True and bool(a['checks']) and all(v is True for v in a['checks'].values()),'independent actualRGB andbody audit required')
 for k in ('full288chain','actual_RGB_online_CNNcache_exact','actual_RGB_online_visionmemory_exact','actual_RGB_online_motorhidden_exact','actual_RGB_online_motormeans_exact','actual_RGB_online_torqueactions_exact','original_metrics_exact','cycles_recomputed_exact'):require(a['checks'].get(k) is True,'missing actual audit '+k)
 require(a['condition']==r['condition']==condition and Path(a['run']).resolve()==folder and a['result_sha256']==sha(folder/'RESULT.json'),'audit/condition/result binding')
 require(r['checkpoint_sha256']==a['checkpoint_sha256']==MOTOR_SHA and r['visual_checkpoint_sha256']==a['visual_checkpoint_sha256']==VISUAL_SHA,'single accepted motor+actual visual2SHA required')
 binding=read(ROOT/'INPUT_BINDINGS.json');require(binding['visual_sha256']==VISUAL_SHA and sha(binding['visual_checkpoint'])==VISUAL_SHA,'actual visualweight byte binding')
 require(r['state_total']==288 and r['state_layout']==[128,64,64,32] and r['identity_sha256']==ident['identity_sha256'],'complete actual identity288')
 require(hand==r['transition'] and hand['all_exact'] is True and hand['no_reset'] is True and hand['camera_queue_retained'] is True,'continuous body/neural/optical handover')
 require(r['native']['actual_native_steps']==1232000 and r['native']['post_native_steps']==768000,'complete realphysics window')
 require(r['teacher_calls_after_handover']==0 and r['extra_forces'] is False,'no hidden control or forces')
 paths=[folder/n for n in ('RESULT.json','VERIFY.json','IDENTITY.json','HANDOVER.json','INITIAL_STATE.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz','runtime_geometry.npz','capture_rgb.npz','CAPTURES.json')]+[ROOT/'INPUT_BINDINGS.json',ROOT/'FROZEN.json',ROOT/'PREFLIGHT.json',HERE/'contracts.py',HERE/'PREREG.json',Path(binding['visual_checkpoint'])]
 hashes={str(p):sha(p) for p in paths};unchanged(r['source_hashes']);hashes.update(r['source_hashes'])
 for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry'),('capture_rgb.npz','capture_rgb'),('CAPTURES.json','captures_metadata'),('INITIAL_STATE.json','initial_state')]:require(sha(folder/name)==r[key]['sha256'],'actualartifact drift '+name)
 t=arrays(folder/'trace_1ms.npz');d=arrays(folder/'neural_decisions.npz');g=arrays(folder/'geometry_1ms.npz');rgb=arrays(folder/'capture_rgb.npz')['rgb'];cap=read(folder/'CAPTURES.json')
 require(t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and rgb.shape==(770,2,32,32,3) and rgb.dtype==np.uint8 and len(cap)==770,'fullactualbody+RGB stream required')
 require(np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(abs(d['time']-(1+np.arange(2880)*.005)))<1e-8 and np.array_equal(g['time'],t['time']),'actual synchronized clocks')
 require(np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]),'complete neuralchain')
 for i,c in enumerate(cap):require(c['capture_id']==i and hashlib.sha256(rgb[i].tobytes()).hexdigest()==c['rgb_sha256'] and c['physical_before_sha256']==c['physical_after_sha256'] and c['render_does_not_modify_physical_data'] is True,'actualRGB/read-only identity')
 normal=read(ROOT/'NORMAL6_SUMMARY.json');summary=read(ROOT/'VISUAL_BODY_SUMMARY.json');require(summary['kind']=='VISUAL_BODY_OCCLUSION_FIXEDFRAME_V2','exactactualsummary kind')
 require([row['condition'] for row in summary['normal6']]==list(NORMAL) and [row['condition'] for row in summary['control2']]==list(CONTROLS),'actual8summary order')
 require(normal['all_functional_pass']==summary['normal6_all_functional_pass'] and normal['rows']==summary['normal6'],'NORMAL6 proof correspondence')
 unchanged(summary['source_hashes']);hashes.update(summary['source_hashes']);hashes[str(ROOT/'NORMAL6_SUMMARY.json')]=sha(ROOT/'NORMAL6_SUMMARY.json');hashes[str(ROOT/'VISUAL_BODY_SUMMARY.json')]=sha(ROOT/'VISUAL_BODY_SUMMARY.json')
 row=next(v for v in summary['normal6']+summary['control2'] if v['condition']==condition);iscontrol=condition in CONTROLS;functional=row['control_pass'] if iscontrol else row['functional_pass'];flags=row['flags'] if iscontrol else row['functional_flags']
 require(type(functional) is bool and functional==all(flags.values()),'summary functional flags consistent; false renderable')
 yaw=np.unwrap(g['thorax_yaw']);yaw=yaw-yaw[5799]
 return dict(condition=condition,run=folder,result=r,audit=a,identity=ident,trace=t,decisions=d,geometry=g,rgb=rgb,captures=cap,yaw=yaw,source_hashes=hashes,functional_pass=functional,functional_flags=flags,functional_metrics=row['metrics'],iscontrol=iscontrol)
