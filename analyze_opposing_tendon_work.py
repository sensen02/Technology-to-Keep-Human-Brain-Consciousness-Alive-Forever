"""S152/P1 bounded saved-array audit. No simulator/neural imports or calls."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, shutil, xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_muscle_reconstruction import evaluate
P=Path('/run/media/sensen/Data2/cell_wound_prototype')
B=Path('/home/sensen/Desktop/engine修复验收')
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
MAP=B/'s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json'
SOURCES={k:B/v/'init80ms' for k,v in {'S149':'s149-half-gain-physical/20261024T213108429114Z','S151':'s151-lf-channel-gain/20261024T214136681607Z','S146':'s146-secant-pose-free-physical/20261024T203645678947Z'}.items()}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(p,rows):
 with p.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def cum(y,t):
 return np.concatenate([np.zeros((1,)+y.shape[1:]),np.cumsum((y[1:]+y[:-1])*np.diff(t).reshape((-1,)+(1,)*(y.ndim-1))/2,axis=0)],axis=0)
def event(y,t):
 peak=int(np.argmax(y));start=next((i for i in range(len(y)-1) if np.all(np.diff(y[i:]) < -1e-10)),None)
 return dict(peak_sample=peak,peak_ms=float(t[peak]*1000),peak=float(y[peak]),sustained_sample=start,sustained_ms=None if start is None else float(t[start]*1000),final=float(y[-1]))
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path);out=pa.parse_args().output or B/'s152-opposing-tendon-work'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
 inputs=[Path(__file__),P/'verify_muscle_reconstruction.py',XML,MAP]+[p/f for p in SOURCES.values() for f in ['TRAJECTORY.npz','NATIVE_MODEL_ARRAYS.npz','MOMENT_METADATA.json','COM_CONTACT_METADATA.json','MODEL_PROVENANCE.json','INITIALIZATION.npz','SCHEDULE.json']]
 hashes={str(p):sha(p) for p in inputs};dump(out/'INPUT_HASHES.json',hashes)
 dump(out/'PREANALYSIS.json',dict(task='S152/P1 bounded OFFLINE descriptive audit',sources={k:str(v) for k,v in SOURCES.items()},windows={'S149/S151':'80 native samples 0..1.975ms including terminal failed gate','S146':'78 native samples 0..1.925ms separate reference, never padded'},event='first i<last with EVERY remaining diff(Fz)<-1e-10 to declared window end; first argmax peak; endpoint-conditioned, not causal onset',event_windows='LM/RM and total sustained starts, +/-0.1ms inclusive native25us records clipped at endpoints; no interpolation. Rank per-actuator opposing allocation integral and signed negative active work in each window.',opposing='positive torque=sum max(M*active,0); negative torque=sum min(M*active,0); net=sum; mechanically opposing magnitude=min(positive,-negative). Allocate magnitude proportionally within each torque sign, half to each sign (sums to magnitude). Mechanical opposition NOT harmfulness.',velocity='assert native actuator_velocity == CSR moment @ native qvel; per-actuator active/passive/total force*velocity == sumDOF M*force*qvel; atol1e-10 rtol1e-12',near_zero='abs(qvel)<=1e-9 has NO_FUNCTIONAL_POWER_ROLE; no flex/ext role inferred',work='trapezoid native seconds; integrate max(power,0), min(power,0) separately and signed total, cumulative arrays included; negative_absorbed_magnitude=-negative_signed. Not energy waste.',map='frozen S150 actual tendon-crossed DOF mapping, names/IDs only; verify every abs(CSR)>1e-12 DOF included and native names match; smaller roundoff entries retained in full closure and CSV. First failed run used exact !=0 map assertion and stopped before reconstruction complete due native ~1e-16 off-map roundoff, no physical calls.',comparison='S149/S151 spikes/features/decoded independently of actual command. Applied and ctrl only channel4 differs; other89 act expected bit-identical, geometry-dependent force/power need not match.',physical_units='NOT_ESTABLISHED',calls=dict(model=0,forward=0,step=0,neural=0),implementation_failures='Prior offline runs: exact !=0 mapping assertion hit retained ~1e-16 roundoff; second run boolean spike subtraction TypeError after arrays reconstructed. Corrected membership tolerance and bool->float comparison only; no physical calls or changed power tolerances.',next_hypothesis='At most ONE final minimal falsifiable hypothesis; no intervention or parameter writes'))
 freeze=out/'SOURCE_FREEZE';freeze.mkdir()
 for p in [Path(__file__),P/'verify_muscle_reconstruction.py']:shutil.copyfile(p,freeze/p.name)
 maps=json.loads(MAP.read_text());nodes=list(ET.parse(XML).getroot().find('actuator'));assert len(nodes)==90
 arrays={};summary={};dofrows=[];actrows=[];contrib=[];legrows=[];eventrows=[];ranks=[]
 for label,p in SOURCES.items():
  a=load(p/'TRAJECTORY.npz');model=load(p/'NATIVE_MODEL_ARRAYS.npz');mm=json.loads((p/'MOMENT_METADATA.json').read_text());meta=json.loads((p/'COM_CONTACT_METADATA.json').read_text());n=78 if label=='S146' else 80;t=a['time'];assert len(t)==n and np.allclose(t,np.arange(n)*25e-6,atol=1e-14,rtol=0)
  assert [r['name'] for r in maps]==mm['actuator_names'];assert [x.get('name') for x in nodes]==mm['actuator_names'];assert np.array_equal(model['actuator_actadr'],np.arange(90))
  assert json.loads((p/'MODEL_PROVENANCE.json').read_text())['xml_sha256']==sha(XML)
  for key in ['gainprm','biasprm','dynprm']:
   v=[list(map(float,node.get(key).split())) for node in nodes];assert np.array_equal(np.array([x+[0.]*(10-len(x)) for x in v]),model['actuator_'+key])
  M=np.zeros((n,90,48));active=np.zeros((n,90));bias=active.copy();gain=active.copy()
  for i in range(n):
   for u,node in enumerate(nodes):
    f,g,b=evaluate(a['actuator_length'][i,u],a['actuator_velocity'][i,u],a['act'][i,u],node);gain[i,u]=g;active[i,u]=g*a['act'][i,u];bias[i,u]=b
    adr=int(a['moment_rowadr'][i,u]);nnz=int(a['moment_rownnz'][i,u]);cols=a['moment_colind'][i,adr:adr+nnz];M[i,u,cols]=a['actuator_moment_raw'][i,adr:adr+nnz]
    assert set(cols[np.abs(M[i,u,cols])>1e-12]).issubset(maps[u]['dofs']) # roundoff entries retained in full power closure/CSV
  vel=np.einsum('tud,td->tu',M,a['qvel']);forces={'active':active,'passive':bias,'total':active+bias};powers={k:f*a['actuator_velocity'] for k,f in forces.items()};torques={k:M*f[:,:,None] for k,f in forces.items()};dofpowers={k:tau*a['qvel'][:,None,:] for k,tau in torques.items()}
  errors={'velocity':float(abs(vel-a['actuator_velocity']).max()),'force':float(abs(active+bias-a['actuator_force']).max()),'moment':float(abs(M-a['actuator_moment']).max()),'all48_projection':float(abs(torques['total'].sum(1)-a['qfrc_actuator']).max())}
  for k in forces:
   errors[k+'_per_actuator_power']=float(abs(powers[k]-dofpowers[k].sum(2)).max());assert np.allclose(powers[k],dofpowers[k].sum(2),atol=1e-10,rtol=1e-12)
  assert np.allclose(vel,a['actuator_velocity'],atol=1e-10,rtol=1e-12);assert max(errors.values())<1e-10,errors
  tau=torques['active'];pos=np.maximum(tau,0).sum(1);neg=np.minimum(tau,0).sum(1);opposing=np.minimum(pos,-neg)
  share=np.where(tau>0,np.divide(tau,pos[:,None,:],out=np.zeros_like(tau),where=pos[:,None,:]>0),np.divide(-tau,-neg[:,None,:],out=np.zeros_like(tau),where=neg[:,None,:]<0))*opposing[:,None,:]/2
  assert np.allclose(share.sum(1),opposing,atol=1e-12,rtol=1e-12)
  work={k:{role:cum(fn(power),t) for role,fn in [('signed',lambda x:x),('positive',lambda x:np.maximum(x,0)),('negative_signed',lambda x:np.minimum(x,0))]} for k,power in powers.items()}
  legs=meta['foot_order'];F=np.zeros((n,6,3))
  for i in range(n):
   for c in np.flatnonzero(a['contact_valid_mask'][i]):
    g1,g2=map(int,a['contact_geom_ids'][i,c]);floor=set(meta['floor_geom_ids']);foot=g2 if g1 in floor else g1 if g2 in floor else -1;leg=meta['geom_map'].get(str(foot))
    if leg is not None:F[i,legs.index(leg)]+=(1 if g1 in floor else -1)*(a['contact_frame'][i,c].T@a['contact_force_raw6'][i,c,:3])
  errors['foot_force']=float(abs(F-a['foot_force']).max());assert errors['foot_force']<1e-10
  ev={leg:event(F[:,j,2],t) for j,leg in enumerate(legs)};ev['total']=event(F[:,:,2].sum(1),t)
  for i in range(n):
   for d in range(48):
    dofrows.append(dict(arm=label,sample=i,time_s=float(t[i]),dof=d,joint=mm['joint_names'][mm['dof_jntid'][d]],qvel=float(a['qvel'][i,d]),near_zero=bool(abs(a['qvel'][i,d])<=1e-9),power_role='NO_FUNCTIONAL_POWER_ROLE' if abs(a['qvel'][i,d])<=1e-9 else 'SIGNED_MECHANICAL_ONLY',positive_active_torque=float(pos[i,d]),negative_active_torque=float(neg[i,d]),net_active_torque=float(tau[i,:,d].sum()),mechanical_opposing_not_harmful=float(opposing[i,d]),active_power=float(dofpowers['active'][i,:,d].sum())))
   for u,r in enumerate(maps):
    row=dict(arm=label,sample=i,time_s=float(t[i]),actuator=u,name=r['name'],leg=r['leg'],act=float(a['act'][i,u]),ctrl=float(a['ctrl'][i,u]),gain=float(gain[i,u]),velocity_native=float(a['actuator_velocity'][i,u]),velocity_moment_qvel=float(vel[i,u]),velocity_error=float(vel[i,u]-a['actuator_velocity'][i,u]))
    for k in forces:
     row[k+'_force']=float(forces[k][i,u]);row[k+'_power']=float(powers[k][i,u])
     for role in work[k]:row[k+'_cumulative_'+role+'_work']=float(work[k][role][i,u])
     row[k+'_cumulative_negative_absorbed_magnitude']=float(-work[k]['negative_signed'][i,u])
    actrows.append(row)
    for d in np.flatnonzero(M[i,u]):
     contrib.append(dict(arm=label,sample=i,time_s=float(t[i]),actuator=u,name=r['name'],leg=r['leg'],dof=int(d),joint=mm['joint_names'][mm['dof_jntid'][d]],moment=float(M[i,u,d]),qvel=float(a['qvel'][i,d]),near_zero=bool(abs(a['qvel'][i,d])<=1e-9),active_torque=float(tau[i,u,d]),passive_torque=float(torques['passive'][i,u,d]),total_torque=float(torques['total'][i,u,d]),active_power=float(dofpowers['active'][i,u,d]),passive_power=float(dofpowers['passive'][i,u,d]),total_power=float(dofpowers['total'][i,u,d]),opposing_allocation_not_harmful=float(share[i,u,d])))
   for l,leg in enumerate(legs):
    us=[r['actuator'] for r in maps if r['leg']==leg];legrows.append(dict(arm=label,sample=i,time_s=float(t[i]),leg=leg,Fz=float(F[i,l,2]),active_power=float(powers['active'][i,us].sum()),active_signed_work=float(work['active']['signed'][i,us].sum()),active_positive_work=float(work['active']['positive'][i,us].sum()),active_negative_signed_work=float(work['active']['negative_signed'][i,us].sum())))
  for name in ['lm','rm','total']:
   center=ev[name]['sustained_sample']
   if center is None:continue
   ids=np.flatnonzero(abs(t-t[center])<=.0001+1e-14);tw=t[ids]
   for i in ids:
    eventrows.append(dict(arm=label,event=name,center_sample=center,center_ms=float(t[center]*1000),sample=int(i),time_s=float(t[i]),Fz=float(F[i,legs.index(name),2]) if name!='total' else float(F[i,:,2].sum()),total_Fz=float(F[i,:,2].sum()),active_power=float(powers['active'][i].sum()),opposing_magnitude=float(opposing[i].sum())))
   ow=cum(share[ids].sum(2),tw)[-1];nw=cum(np.minimum(powers['active'][ids],0),tw)[-1]
   order=np.argsort(-ow);nr=np.argsort(nw)
   for u,r in enumerate(maps):ranks.append(dict(arm=label,event=name,actuator=u,name=r['name'],leg=r['leg'],dofs=json.dumps(r['dofs']),window_first_sample=int(ids[0]),window_last_sample=int(ids[-1]),opposing_integral=float(ow[u]),opposing_rank=int(np.flatnonzero(order==u)[0]+1),active_negative_signed_work=float(nw[u]),negative_work_rank=int(np.flatnonzero(nr==u)[0]+1)))
  opposing_dofs=np.flatnonzero(np.any(opposing>1e-10,axis=0)).tolist();multi=np.any(np.max(dofpowers['active'],axis=2)>1e-12,axis=0)&np.any(np.min(dofpowers['active'],axis=2)<-1e-12,axis=0)
  simultaneous=np.any((np.max(dofpowers['active'],axis=2)>1e-12)&(np.min(dofpowers['active'],axis=2)<-1e-12),axis=0)
  summary[label]=dict(samples=n,end_ms=float(t[-1]*1000),errors_maxabs=errors,events=ev,opposing_dofs=opposing_dofs,opposing_joint_names=[mm['joint_names'][mm['dof_jntid'][d]] for d in opposing_dofs],simultaneous_inject_absorb_actuator_ids=np.flatnonzero(simultaneous).tolist(),total_work={k:{role:float(v[-1].sum()) for role,v in roles.items()} for k,roles in work.items()},top10_negative_active_work=[dict(actuator=int(u),name=maps[u]['name'],work=float(work['active']['negative_signed'][-1,u])) for u in np.argsort(work['active']['negative_signed'][-1])[:10]])
  saved=dict(time=t,moment=M,active_gain=gain,active_force=active,passive_bias=bias,velocity_moment_qvel=vel,opposing=opposing,opposing_allocation=share)
  for k in forces:
   saved[k+'_power']=powers[k]
   for role,v in work[k].items():saved[k+'_cumulative_'+role+'_work']=v
  np.savez_compressed(out/(label+'_RECONSTRUCTED.npz'),**saved);arrays[label]=(a,powers)
 a,pow0=arrays['S149'];b,pow1=arrays['S151'];other=np.arange(90)!=4;comparison={}
 for key in ['spikes','features','decoded','applied','ctrl','act','actuator_force','actuator_velocity']:
  delta=abs(b[key].astype(float)-a[key].astype(float));comparison[key]=dict(equal=bool(np.array_equal(a[key],b[key])),maxabs=float(delta.max()))
  if delta.ndim==2 and delta.shape[1]==90:comparison[key]['other89_maxabs']=float(delta[:,other].max());comparison[key]['changed_channel_ids']=np.flatnonzero(np.any(delta!=0,axis=0)).tolist()
 for key in ['spikes','features','decoded']:assert comparison[key]['equal'],comparison[key]
 for key in ['applied','ctrl']:assert comparison[key]['changed_channel_ids']==[4]
 assert comparison['act']['other89_maxabs']==0
 change=[]
 for u in np.flatnonzero(other):
  change.append(dict(actuator=int(u),name=maps[u]['name'],leg=maps[u]['leg'],force_maxabs=float(abs(b['actuator_force'][:,u]-a['actuator_force'][:,u]).max()),active_power_maxabs=float(abs(pow1['active'][:,u]-pow0['active'][:,u]).max()),total_power_maxabs=float(abs(pow1['total'][:,u]-pow0['total'][:,u]).max())))
 for key in pow0:comparison[key+'_power']=dict(maxabs=float(abs(pow1[key]-pow0[key]).max()),other89_maxabs=float(abs(pow1[key][:,other]-pow0[key][:,other]).max()))
 change.sort(key=lambda r:r['total_power_maxabs'],reverse=True)
 table(out/'DOF_OPPOSING.csv',dofrows);table(out/'ACTUATOR_SIGNED_WORK.csv',actrows);table(out/'NONZERO_DOF_CONTRIBUTIONS.csv',contrib);table(out/'SIX_LEG_WORKLOAD.csv',legrows);table(out/'EVENT_NATIVE_RECORDS.csv',eventrows);table(out/'EVENT_ACTUATOR_RANKS.csv',ranks);table(out/'OTHER89_CHANGED_RANK.csv',change)
 dump(out/'COMMAND_ACT_FORCE_POWER_COMPARISON.json',comparison)
 result=dict(task='S152/P1',arms=summary,comparison=comparison,other89_top10_changed=change[:10],calls=dict(model=0,forward=0,step=0,neural=0),physical_units='NOT_ESTABLISHED',minimal_next_hypothesis=dict(status='INSUFFICIENT_FOR_UNIQUE_INTERVENTION',hypothesis='S149→S151 other89 force/power differences arise from changed mechanical state despite identical other89 activation inputs, not altered other89 neural recruitment.',falsifiable_criteria='At matched saved length/velocity and activation, scalar gain/bias must reproduce each other89 force to atol1e-10; raw spikes/features/decoded and other89 ctrl/act must remain identical. A residual force difference >1e-10 under identical inputs rejects this explanation. This is a minimal offline matched-state decomposition hypothesis, NOT proof of unloading cause or a parameter recommendation.',not_executed=True))
 dump(out/'SUMMARY.json',result)
 ds=summary['S149']['opposing_dofs'];fig,axs=plt.subplots(4,3,figsize=(15,12),sharex=True)
 for ax,d in zip(axs.flat,ds):
  for label in SOURCES:
   rows=[r for r in dofrows if r['arm']==label and r['dof']==d];ax.plot([r['time_s']*1000 for r in rows],[r['mechanical_opposing_not_harmful'] for r in rows],label=label)
  ax.set_title(f'DOF {d} '+rows[0]['joint']);ax.grid(alpha=.3);ax.legend(fontsize=7);ax.set_xlabel('physical ms')
 fig.suptitle('12 DOF mechanical opposition != harmfulness; native torque units NOT_ESTABLISHED');fig.tight_layout();fig.savefig(out/'GROUPED_12_DOF_OPPOSING.png',dpi=140);plt.close(fig)
 fig,axs=plt.subplots(6,3,figsize=(15,18),sharex=True)
 for l,leg in enumerate(legs):
  for label in SOURCES:
   rows=[r for r in legrows if r['arm']==label and r['leg']==leg];x=[r['time_s']*1000 for r in rows]
   for ax,key in zip(axs[l],['Fz','active_positive_work','active_negative_signed_work']):ax.plot(x,[r[key] for r in rows],label=label)
  for ax,title in zip(axs[l],['foot-floor Fz','positive active work','negative signed active work']):ax.set_title(leg+' '+title);ax.grid(alpha=.3);ax.legend(fontsize=7);ax.set_xlabel('physical ms')
 fig.suptitle('S152 native six-leg workload; negative work != waste; S146 separate shorter reference');fig.tight_layout();fig.savefig(out/'SIX_LEG_WORKLOAD.png',dpi=135);plt.close(fig)
 report=['# S152 / P1 同DOF机械对抗与肌腱有符号功离线审查','不是成品、训练、预热或自主行走成功；0模型/forward/step/神经调用。绝对单位 NOT_ESTABLISHED。','## 预登记范围与定义','S149/S151共同80个实际25us样本0–1.975ms；S146仅78样本0–1.925ms单列参考，不补齐。末失败样本包含，不称全门通过。每一步90路scalar gain×act+bias、CSR90×48、原生肌腱速度及每执行器三类功率全48DOF恒等式均断言；误差见SUMMARY。','同DOF正主动矩和、负主动矩和与net均输出；min(正和,−负和)仅机械对抗，不等同有害。|qvel|≤1e−9明示无功能功率角色。跨关节肌腱同刻可向一个DOF注功而从另一DOF吸功，不标签好坏；映射只保留真实名字/ID/跨关节，无屈伸功能推断。','正功与负吸收功按每执行器功率正负截断后梯形积分，signed=positive+negative_signed；负功不是浪费或代谢能测量。CSV含全累积和稀疏非零DOF贡献。','持续卸载定义完全沿S150：从首个i到声明窗末所有diff<−1e−10，非末样本真空成立；这是窗末条件描述而非因果起点。LM/RM及总载荷起点±.1ms仅真实25us点，窗口排名按已预登记机械对抗分配积分和负主动功分别排序。','## 实际结果',json.dumps(summary,ensure_ascii=False,indent=2),'## 神经/命令/激活与力功率分离',json.dumps(comparison,ensure_ascii=False,indent=2),'RAW spikes/features/decoded逐项独立比较，不能拿applied差冒充神经差；仅channel4 applied/ctrl改变。其他89路act相同不要求其力或功率相同，真实几何/速度改变仍可改变gain/bias/moment。OTHER89_CHANGED_RANK按总功率最大差排序，不按名称猜功能。','## 唯一最小下一假设（未执行）',json.dumps(result['minimal_next_hypothesis'],ensure_ascii=False,indent=2),'现有短窗机械对抗/负功不足以唯一选择导致LM/RM卸载的通道或确定应改何参数；不硬选新的gain/tau/权重、不扩盲扫。上述唯一假设仅可证伪的离线状态依赖解释，不能升级为总失撑原因。S113/checkpoint保持、无实际干预。']
 (out/'REPORT.zh-CN.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
 after={str(p):sha(p) for p in inputs};assert after==hashes;dump(out/'INPUT_HASHES_AFTER.json',after);dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.rglob('*') if p.is_file()});dump(out/'READY.json',dict(ready=True,output_root=str(out),source_sha256=sha(Path(__file__)),calls=result['calls']));print('READY '+str(out),flush=True)
if __name__=='__main__':main()
