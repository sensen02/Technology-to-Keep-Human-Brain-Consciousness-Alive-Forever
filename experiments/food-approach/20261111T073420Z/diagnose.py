from pathlib import Path
import json, hashlib, shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
RUN=PARENT/'20261111T072012Z/EVAL_right'
DESKTOP=Path('/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p,keys):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in keys}
files=[RUN/n for n in ('RESULT.json','IDENTITY.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz')]
r=json.loads((RUN/'RESULT.json').read_text());ident=json.loads((RUN/'IDENTITY.json').read_text())
checkpoint=Path(r['checkpoint']);files.append(checkpoint)
teacher_path=PARENT/'20261110T191449Z/teacher/right_full/RESULT.json';files.append(teacher_path)
hashes={str(p):sha(p) for p in files}
assert sha(checkpoint)==r['checkpoint_sha256']
t=load(RUN/'trace_1ms.npz',['time','action','actual_requested_action','target','actualctrl','actuator_force','com','control_source'])
d=load(RUN/'neural_decisions.npz',['time','command','hidden_after','mean','policy_action','actually_applied_action','control_source'])
g=load(RUN/'geometry_1ms.npz',['thorax_yaw'])
a=t['control_source']==3;b=d['control_source']==3
assert a.sum()==9600 and b.sum()==1920
assert np.array_equal(d['policy_action'][b],d['actually_applied_action'][b])
assert np.array_equal(t['action'][a],t['actual_requested_action'][a])
assert np.max(np.abs(np.tanh(d['mean'])-d['policy_action']))<1e-14
u=d['command'][b,1];elapsed=d['time'][b]-5.8
yaw=np.unwrap(g['thorax_yaw']);yaw0=yaw[5799]
assert abs(yaw[-1]-yaw0-r['yaw_change_rad'])<1e-12
bound=np.asarray(ident['actuator']['target_bounds_rad']);limit=np.asarray(ident['actuator']['torque_limit'])
target=t['target'][a];torque=t['actualctrl'][a];action=t['action'][a]
margin=np.minimum(target-bound[:,0],bound[:,1]-target)
ratio=np.abs(torque)/limit
p=torch.load(checkpoint,map_location='cpu',weights_only=False);sd=p['state_dict']
W=sd['parent.head.weight'].numpy();bias=sd['parent.head.bias'].numpy()
latent=d['hidden_after'][b,128:]@W.T+bias
gate=np.tanh(np.sum(d['command'][b]**2,axis=1))[:,None]
correction=.5*gate*np.tanh(latent)
base_mean=d['mean'][b]-correction
same_state_action_difference=np.tanh(d['mean'][b])-np.tanh(base_mean)
bysecond=[]
for sec in (0,1,2,4,6,8,9.6):
    i=int(np.argmin(np.abs(t['time']-(5.8+sec))))
    bysecond.append({'after_handover_s':sec,'actual_time_s':float(t['time'][i]),'yaw_change_deg':float(np.rad2deg(yaw[i]-yaw0))})
summary={
 'scope':'Read-only saved-data diagnosis; no physics, training or parameter changes',
 'source_hashes':hashes,
 'right':{'yaw_change_rad':r['yaw_change_rad'],'yaw_change_deg':float(np.rad2deg(r['yaw_change_rad'])),'window_s':9.6,'mean_yaw_deg_per_s':float(np.rad2deg(r['yaw_change_rad'])/9.6),'yaw_by_time':bysecond,'slip':r['autonomous']['slip_ratio_max'],'forward_per_s':r['autonomous']['forward_signed_native_per_s']},
 'command':{'min':float(u.min()),'max':float(u.max()),'full_minus1_decisions':int(np.sum(u==-1)),'ANN_decisions':len(u),'full_minus1_seconds':float(np.sum(u==-1)*.005),'command_at_1s_after_handover':float(u[np.argmin(np.abs(elapsed-1))]),'no_request_application_difference':True},
 'endpoint_limits':{'sampling_s':.001,'max_abs_normalized_action':float(np.abs(action).max()),'abs_action_ge_099_count':int(np.sum(np.abs(action)>=.99)),'target_at_bound_count_tol1e10':int(np.sum(margin<=1e-10)),'minimum_target_margin_rad':float(margin.min()),'max_abs_torque_to_limit':float(ratio.max()),'torque_ge_099limit_count':int(np.sum(ratio>=.99)),'joint_samples':int(action.size),'limits':'1ms saved endpoints only; no full native torque/force certification'},
 'residual_on_saved_states':{'max_abs_pre_tanh_correction':float(np.abs(correction).max()),'theoretical_fullcommand_bound':float(.5*np.tanh(1)),'max_abs_head_tanh':float(np.abs(np.tanh(latent)).max()),'action_delta_RMS':float(np.sqrt(np.mean(same_state_action_difference**2))),'interpretation':'same saved-state arithmetic, NOT a physical zero-residual control'},
 'source_policy':{'positive_readout_in_right':0,'old_yaw_gate_rad':.01,'old_yaw_gate_deg':float(np.rad2deg(.01)),'turn_command_units':'normalized modulation, not radians or rad/s'},
 'teacher_result_path':str(teacher_path)
}
assert all(sha(path)==h for path,h in hashes.items())
(ROOT/'DIAGNOSIS.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
x=t['time'][a]-5.8
axes[0].plot(x,np.rad2deg(yaw[a]-yaw0),label='Actual right turn, same saved run',lw=2)
axes[0].axhline(-np.rad2deg(.01),color='gray',ls=':',label='Old direction gate: -0.573 deg')
axes[0].set_ylabel('Heading change (degrees)');axes[0].legend();axes[0].grid(alpha=.25)
axes[1].plot(elapsed,u,color='#b75d24',lw=2);axes[1].set_xlabel('Seconds after ANN handover');axes[1].set_ylabel('Neural turn command');axes[1].set_ylim(-1.08,.08);axes[1].grid(alpha=.25)
fig.suptitle('Right turn is physically small: 4.379 degrees in 9.6 seconds\nCommand reaches -1; fixed world camera does not exaggerate yaw')
fig.tight_layout();plot=ROOT/'RIGHT_TURN_DIAGNOSIS.png';fig.savefig(plot,dpi=150);plt.close(fig)
copy=DESKTOP/'右转幅度诊断-实际转角与命令.png';shutil.copyfile(plot,copy);assert sha(plot)==sha(copy)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('PLOT',str(copy))
