"""Corrected S119 saved-array constraints; no simulation, weights or historical edits."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from verify_single_weight_attenuation import decision
B=Path('/home/sensen/Desktop/engine修复验收');OUT=B/'s119-training-constraints/20261024T070600Z'
REFS=dict(S113=B/'s113-com-timestep/20261024T060500Z/dt25us',S115=B/'s115-extended-initialization/20261024T062500Z/init80ms',S117=B/'s117-single-weight-attenuation/20261024T064500Z/w36b4_down',S118=B/'s118-secant-weight-step/20261024T065500Z/w36b4_secant')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvsave(n,r):
 with (OUT/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def main():
 hashes={str(Path(__file__).resolve()):sha(__file__)};samples=[];limits=[];coupling=[];metrics={}
 for label,p in REFS.items():
  for mapping in (js(p.parent/'INPUT_HASHES.json'),js(p/'RESULT.json')['hashes']):
   for name,h in mapping.items():
    assert Path(name).is_file() and sha(name)==h,name
    hashes[str(Path(name).resolve())]=h
  for f in (p/'RESULT.json',p.parent/'PREREG.json',p.parent/'COMPLETE.json'):hashes[str(f.resolve())]=sha(f)
  with np.load(p/'TRAJECTORY.npz') as z:a={k:z[k] for k in z.files}
  m=js(p/'MOMENT_METADATA.json');t=a['time'];series=[];lr=[]
  for j,limited in enumerate(m['jnt_limited']):
   if not limited:continue
   q=a['qpos'][:,m['jnt_qposadr'][j]];lo,hi=m['jnt_range'][j];ex=np.maximum(np.maximum(lo-q,q-hi),0);series.append(ex)
   r=dict(sample=label,joint=m['joint_names'][j],dof=m['jnt_dofadr'][j],min_q=float(q.min()),max_q=float(q.max()),max_excess=float(ex.max()),peak_ms=float(t[ex.argmax()]*1000));limits.append(r);lr.append(r)
  assert len(series)==42;globalex=np.stack(series).max(0);assert np.array_equal(globalex,a['limit_excess'])
  g=a['six_gates'].astype(bool);good=g.all(1);longest=0;start=None
  for i,value in enumerate(good):
   if value and start is None:start=i
   if start is not None and (not value or i==len(good)-1):
    end=i if value else i-1;longest=max(longest,float((t[end]-t[start])*1000));start=None
  first={n:(float(t[np.flatnonzero(~g[:,i])[0]]*1000) if (~g[:,i]).any() else None) for i,n in enumerate(('height','angle','legs','load','nonfoot','limit'))};ix=t<=.0017+1e-12
  ext=dict(limit_excess=dict(max=float(globalex.max())),angle_rad=dict(max=float(a['root_angle_rad'].max())),nonfoot_depth=dict(max=float(a['nonfoot_depth'].max())),height_ratio=dict(min=float((a['qpos'][:,2]/a['qpos'][0,2]).min())))
  metric=dict(extrema=ext,first_failures_ms=first,initial_load_ratio=float(a['load_ratio'][0]),load_deficit_integral0_1p7_ratio_ms=float(np.trapezoid(np.maximum(.8-a['load_ratio'][ix],0),t[ix])*1000),all_gates_support_ms=float(np.trapezoid(good.astype(float),t)*1000),longest_all_gates_true_endpoint_span_ms=longest,max_downward_root_velocity=float(np.maximum(-a['qvel'][:,2],0).max()),max_downward_COM_velocity=float(np.maximum(-a['com_velocity_world'][:,2],0).max()))
  metrics[label]=metric;worst=max(lr,key=lambda r:r['max_excess'])
  samples.append(dict(sample=label,role='original_reference_not_behavior_success' if label=='S113' else 'rejected_safety_boundary',W36b4=float(a['weights'][36,4]),weight_bytes_sha256=hashlib.sha256(np.ascontiguousarray(a['weights'],dtype=np.float64).tobytes()).hexdigest(),max_limit=worst['max_excess'],peak_joint=worst['joint'],peak_ms=worst['peak_ms'],deficit=metric['load_deficit_integral0_1p7_ratio_ms'],first_load_ms=first['load'],support_ms=metric['all_gates_support_ms'],longest_ms=longest,first_limit_ms=first['limit']))
  for i,time in enumerate(t):
   for dof in (6,7,8):
    torque=a['actuator_force'][i]*a['actuator_moment'][i,:,dof];assert abs(torque.sum()-a['qfrc_actuator'][i,dof])<1e-10
    coupling.append(dict(sample=label,time_ms=float(time*1000),dof=dof,act2_force=float(a['actuator_force'][i,2]),act2_moment=float(a['actuator_moment'][i,2,dof]),act2_torque=float(torque[2]),actuator_total=float(torque.sum()),passive=float(a['qfrc_passive'][i,dof]),bias=float(a['qfrc_bias'][i,dof]),constraint=float(a['qfrc_constraint'][i,dof]),Ma=float(a['Ma'][i,dof])))
 pre=js(REFS['S113'].parent/'PREREG.json');source=Path(pre['same_weights']);assert sha(source)==hashes[str(source.resolve())]
 checkpoint=B/'s109-initialized-translation-training/20261024T045944Z'
 for n in ('CHECKPOINT.json','CHECKPOINT.npz'):assert sha(checkpoint/n)==hashes[str((checkpoint/n).resolve())]
 active=dict(source_weights=str(source),source_weights_sha256=sha(source),checkpoint={str(checkpoint/n):sha(checkpoint/n) for n in ('CHECKPOINT.json','CHECKPOINT.npz')},selection='retain original S113 W and checkpoint; no overwrite or new checkpoint',W36b4=.18265069824315935)
 constraints=dict(benefit_reference='S115',safety_reference='S113',benefit_exact_bounds=metrics['S115'],safety_exact_bounds=metrics['S113'],rules='S115 initial load >=, deficit <=, first load failure not earlier, support sum >=, longest >=; S113 all42 maximum excess/angle/nonfoot/downroot/downCOM <=, height >=, safety gate failures not earlier. Existing six gates/native COM criteria unchanged.',evaluations={label:decision(metrics['S113'],metrics['S115'],metrics[label],True) for label in REFS},all_sample_metrics=metrics,coupling='DOF6/7/8 actual force times moment; no DOF-isolated intervention; state confounding retained',failed_channel='No further W[36,4] scan; S117 to S118 less weight but higher maximum excess; no monotone behavioral model',units='NOT_ESTABLISHED',physics_calls=0,neural_ticks=0,checkpoint_replacement=False)
 assert all(constraints['evaluations'][k]['reject'] for k in ('S115','S117','S118'))
 OUT.mkdir(parents=True,exist_ok=False)
 save('INPUT_HASHES.json',hashes);save('ACTIVE_BASELINE.json',active);save('TRAINING_CONSTRAINTS.json',constraints)
 save('SOURCE_RESOLUTION.json',dict(requested='/run/media/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z/',requested_exists=False,actual=str(REFS['S113'].parent),actual_resolved=str(REFS['S113'].parent.resolve()),note='Requested absent; committed historical path used, no copying'))
 save('CORRECTION_NOTICE.json',dict(rejected_registration=str(B/'s119-training-constraints/20261024T070500Z'),reason='Prior draft mislabeled absolute-q coordinate as peak joint, moment norm as torque and omitted numeric gate bounds. Not accepted. This fresh corrected registration uses actual limits and signed force times moment. Prior draft preserved.'))
 csvsave('TRAINING_SAMPLES.csv',samples);csvsave('ALL42_FAILURE_BOUNDARIES.csv',limits);csvsave('DOF678_COUPLING.csv',coupling)
 assert all(sha(name)==h for name,h in hashes.items())
 save('ARTIFACT_HASHES.json',{str(f):sha(f) for f in OUT.iterdir() if f.is_file()})
 print(json.dumps(dict(passed=True,samples=len(samples),joints=len(limits),coupling_rows=len(coupling),hashes=len(hashes),output=str(OUT))))
if __name__=='__main__':main()
