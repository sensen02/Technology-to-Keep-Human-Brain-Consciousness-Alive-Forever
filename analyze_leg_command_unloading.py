"""S150/P1 saved-array contact/command allocation; no model or neural calls."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_muscle_reconstruction import evaluate
P=Path('/run/media/sensen/Data2/cell_wound_prototype'); B=Path('/home/sensen/Desktop/engine修复验收')
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
SOURCES={k:B/v/'init80ms' for k,v in {'S146':'s146-secant-pose-free-physical/20261024T203645678947Z','S149':'s149-half-gain-physical/20261024T213108429114Z','S148':'s148-zero-control-ablation/20261024T212511892865Z'}.items()}
FILES=['TRAJECTORY.npz','INITIALIZATION.npz','NATIVE_MODEL_ARRAYS.npz','MOMENT_METADATA.json','COM_CONTACT_METADATA.json','MODEL_PROVENANCE.json','SCHEDULE.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(p,rows):
 with p.open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def events(y,t):
 peak=int(np.argmax(y)); start=next((i for i in range(len(y)-1) if np.all(np.diff(y[i:]) < -1e-10)),None)
 return dict(peak_sample=peak,peak_ms=float(t[peak]*1000),peak=float(y[peak]),sustained_decrease_sample=start,sustained_decrease_ms=None if start is None else float(t[start]*1000),final=float(y[-1]))
def mapping(root,mm,legs):
 # Spatial tendons: crossed joints are ancestor-path symmetric differences, not muscle-name semantics.
 sites={}; jointpaths={}
 def walk(body,ancestry):
  here=ancestry+[x.get('name') for x in body.findall('joint')]
  for x in body.findall('joint'):jointpaths[x.get('name')]=here
  for x in body.findall('site'):sites[x.get('name')]=set(here)
  for child in body.findall('body'):walk(child,here)
 for body in root.find('worldbody').findall('body'):walk(body,[])
 tendons={x.get('name'):x for x in root.find('tendon')}; rows=[]
 for a,node in enumerate(root.find('actuator')):
  tendon=tendons[node.get('tendon')]; refs=set(x.get('joint') for x in tendon.findall('joint'))
  paths=[sites[x.get('site')] for x in tendon.findall('site')]
  if paths:refs |= set.union(*paths)-set.intersection(*paths)
  ids=[mm['joint_names'].index(j) for j in sorted(refs)]; dofs=[mm['jnt_dofadr'][j] for j in ids]
  owners={leg for j in refs for leg in legs if j.lower().startswith('joint_'+leg)}
  rows.append(dict(actuator=a,name=node.get('name'),tendon=node.get('tendon'),wrap_type=tendon.tag,referenced_crossed_joints=sorted(refs),dofs=dofs,leg=next(iter(owners)) if len(owners)==1 else 'UNKNOWN',functional_role='UNKNOWN: no name-based flex/ext inference'))
 return rows
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path);out=pa.parse_args().output or B/'s150-leg-command-unloading'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
 inputs=[Path(__file__),P/'verify_muscle_reconstruction.py',P/'run_secant_weight_probe.py',XML]+[p/f for p in SOURCES.values() for f in FILES]
 hashes={str(p):sha(p) for p in inputs};dump(out/'INPUT_HASHES.json',hashes)
 dump(out/'PREANALYSIS.json',dict(task='S150/P1 offline diagnostic, not walking acceptance',sources={k:str(v) for k,v in SOURCES.items()},windows={'S146/S149':'0..1.925ms 78 samples, includes S146 failure terminal','S148':'0..0.25ms 11 samples; never extrapolate'},endpoint='First i < last with EVERY remaining diff(Fz) < -1e-10, to end of declared saved window; no vacuous final sample. Peak first argmax. Endpoint-conditioned not causal onset.',bearing='foot-floor world Fz > 1e-10; negative distance counts separately',command_hold='sample0 actual ctrl=0; sample i>0 uses decoded/applied tick floor((i-1)/20), post-tick command held during following 20 steps',mapping='actual XML tendon site ancestor paths/joint refs + MOMENT_METADATA DOF map; source runner joint_LEG rule only, never muscle-name roles',antagonism='same DOF opposite signed moment with simultaneous |active force|>1e-10, additionally output positive/negative active torque',physical_units='NOT_ESTABLISHED',calls=dict(model=0,forward=0,step=0,neural=0)))
 root=ET.parse(XML).getroot();nodes=list(root.find('actuator'));contactrows=[];legrows=[];musclerows=[];jointrows=[];summary={};arrays={}
 for label,p in SOURCES.items():
  a=load(p/'TRAJECTORY.npz');m=load(p/'NATIVE_MODEL_ARRAYS.npz');mm=json.loads((p/'MOMENT_METADATA.json').read_text());meta=json.loads((p/'COM_CONTACT_METADATA.json').read_text());legs=meta['foot_order'];n=11 if label=='S148' else 78;t=a['time'][:n];assert np.allclose(t,np.arange(n)*25e-6,atol=1e-14,rtol=0)
  maps=mapping(root,mm,legs);assert [x['name'] for x in maps]==mm['actuator_names'];dump(out/(label+'_ACTUATOR_MAP.json'),maps)
  assert len(nodes)==90 and np.array_equal(m['actuator_actadr'],np.arange(90))
  for field in ['gainprm','biasprm','dynprm']:
   vals=[list(map(float,node.get(field).split())) for node in nodes];assert np.array_equal(np.array([v+[0.]*(10-len(v)) for v in vals]),m['actuator_'+field])
  prov=json.loads((p/'MODEL_PROVENANCE.json').read_text());assert prov['xml_sha256']==sha(XML)
  force=np.zeros((n,6,3));negative=np.zeros((n,6),int);bearing=negative.copy();counts=negative.copy();depth=np.zeros((n,6));positions=[[[] for _ in legs] for _ in t];active=np.zeros((n,90));bias=active.copy();M=np.zeros((n,90,48));raw=np.zeros((n,90));held=raw.copy()
  for i in range(n):
   if i:tick=(i-1)//20;raw[i]=a['decoded'][tick];held[i]=a['applied'][tick]
   for c in np.flatnonzero(a['contact_valid_mask'][i]):
    g1,g2=map(int,a['contact_geom_ids'][i,c]);floor=set(meta['floor_geom_ids']);foot=g2 if g1 in floor else g1 if g2 in floor else -1;leg=meta['geom_map'].get(str(foot))
    if leg is None:continue
    l=legs.index(leg);f=(1 if g1 in floor else -1)*(a['contact_frame'][i,c].T@a['contact_force_raw6'][i,c,:3]);dist=float(a['contact_dist'][i,c]);pos=a['contact_pos_world'][i,c];force[i,l]+=f;negative[i,l]+=dist<0;bearing[i,l]+=f[2]>1e-10;counts[i,l]+=1;depth[i,l]=max(depth[i,l],-dist);positions[i][l].append(pos.tolist())
    contactrows.append(dict(arm=label,sample=i,time_s=float(t[i]),contact=int(c),leg=leg,geom1=g1,geom2=g2,Fx=float(f[0]),Fy=float(f[1]),Fz=float(f[2]),negative_distance=dist<0,bearing=bool(f[2]>1e-10),dist=dist,depth=max(0.,-dist),x=float(pos[0]),y=float(pos[1]),z=float(pos[2])))
   for j,node in enumerate(nodes):
    f,g,b=evaluate(a['actuator_length'][i,j],a['actuator_velocity'][i,j],a['act'][i,j],node);active[i,j]=g*a['act'][i,j];bias[i,j]=b
    adr=int(a['moment_rowadr'][i,j]);nnz=int(a['moment_rownnz'][i,j]);cols=a['moment_colind'][i,adr:adr+nnz];M[i,j,cols]=a['actuator_moment_raw'][i,adr:adr+nnz]
    assert set(cols[np.abs(M[i,j,cols])>1e-12]).issubset(set(maps[j]['dofs']))
    musclerows.append(dict(arm=label,sample=i,time_s=float(t[i]),actuator=j,leg=maps[j]['leg'],name=maps[j]['name'],rawdecoded=float(raw[i,j]),actualctrl=float(a['ctrl'][i,j]),act=float(a['act'][i,j]),active_gain=float(g),active_force=float(active[i,j]),passive_bias=float(b),total_force=float(f)))
   for d in range(6,48):
    joint=mm['joint_names'][mm['dof_jntid'][d]];prod=M[i,:,d]*active[i];pos=prod[prod>0].sum();neg=prod[prod<0].sum();live=np.abs(active[i])>1e-10;opposing=bool(np.any(live&(M[i,:,d]>1e-10)) and np.any(live&(M[i,:,d]<-1e-10)))
    jointrows.append(dict(arm=label,sample=i,time_s=float(t[i]),dof=d,joint=joint,positive_active_torque=float(pos),negative_active_torque=float(neg),net_active_torque=float(prod.sum()),opposing_active=opposing,cancelled_magnitude=float(min(pos,-neg)),passive_bias_torque=float(M[i,:,d]@bias[i]),total_torque=float(M[i,:,d]@(active[i]+bias[i]))))
   for l,leg in enumerate(legs):
    js=[r['actuator'] for r in maps if r['leg']==leg]
    legrows.append(dict(arm=label,sample=i,time_s=float(t[i]),leg=leg,Fx=float(force[i,l,0]),Fy=float(force[i,l,1]),Fz=float(force[i,l,2]),contact_count=int(counts[i,l]),negative_count=int(negative[i,l]),bearing_count=int(bearing[i,l]),maxdepth=float(depth[i,l]),contact_positions=json.dumps(positions[i][l]),known_actuators=len(js),raw_sum=float(raw[i,js].sum()),raw_max=float(raw[i,js].max(initial=0)),raw_sat_count=int((raw[i,js]>=1-1e-12).sum()),actual_sum=float(a['ctrl'][i,js].sum()),actual_sat_count=int((a['ctrl'][i,js]>=1-1e-12).sum()),act_sum=float(a['act'][i,js].sum()),act_max=float(a['act'][i,js].max(initial=0)),active_force_L1=float(abs(active[i,js]).sum()),passive_bias_L1=float(abs(bias[i,js]).sum()),active_power=float(active[i,js]@a['actuator_velocity'][i,js])))
  err=dict(foot_force=float(abs(force-a['foot_force'][:n]).max()),negative_count=float(abs(negative-a['contacts'][:n]).max()),load_numerator=float(abs(force.sum(1)[:,2]-a['load_ratio'][:n]*(-a['gravity_force_world'][:n,2])).max()),force_scalar=float(abs(active+bias-a['actuator_force'][:n]).max()),moment_CSR=float(abs(M-a['actuator_moment'][:n]).max()),projection=float(abs(np.einsum('tud,tu->td',M,active+bias)-a['qfrc_actuator'][:n]).max()),command_hold=float(abs(held-a['ctrl'][:n]).max()))
  assert max(err.values())<1e-10,err
  fail=np.flatnonzero(~a['all_six_gates']);endpoints={leg:events(force[:,l,2],t) for l,leg in enumerate(legs)}
  for v in endpoints.values():v['unload_before_first_gate']=bool(v['sustained_decrease_sample'] is not None and (not len(fail) or v['sustained_decrease_sample']<int(fail[0])))
  summary[label]=dict(samples=n,end_ms=float(t[-1]*1000),errors_maxabs=err,known_leg_actuators=sum(r['leg']!='UNKNOWN' for r in maps),unknown_leg_actuators=[r['actuator'] for r in maps if r['leg']=='UNKNOWN'],activation_maxabs=float(abs(a['act'][:n]).max()),actual_saved_full_activation_maxabs=float(abs(a['act']).max()),total=events(force.sum(1)[:,2],t),perleg=endpoints,first_gate_failure_ms=None if not len(fail) else float(a['time'][fail[0]]*1000),terminal_negative_counts=negative[-1].tolist(),terminal_bearing_counts=bearing[-1].tolist(),raw_sat_sample_channel_count=int((raw>=1-1e-12).sum()),actual_sat_sample_channel_count=int((a['ctrl'][:n]>=1-1e-12).sum()))
  arrays[label]=(a,active,force,raw)
 table(out/'CONTACT_ROWS.csv',contactrows);table(out/'LEG_TIMESTEPS.csv',legrows);table(out/'ACTUATOR_TIMESTEPS.csv',musclerows);table(out/'JOINT_OPPOSING_ACTIVE_TORQUE.csv',jointrows)
 comparisons={}
 for n in [11,78]:
  ref=arrays['S146'];half=arrays['S149'];comparisons[str(n)]=dict(half_ctrl_vs_half_reference_maxabs=float(abs(half[0]['ctrl'][:n]-.5*ref[0]['ctrl'][:n]).max()),half_activation_vs_half_reference_maxabs=float(abs(half[0]['act'][:n]-.5*ref[0]['act'][:n]).max()),half_activeforce_vs_half_reference_maxabs=float(abs(half[1][:n]-.5*ref[1][:n]).max()),half_legFz_vs_half_reference_maxabs=float(abs(half[2][:n,:,2]-.5*ref[2][:n,:,2]).max()))
 dump(out/'SUMMARY.json',dict(task='S150/P1',arms=summary,nonlinear_comparison=comparisons,S149_activation_zero_correction='activation_maxabs=0 is a reconstruction error, NOT real zero; actual saved act maxima reported above',calls=dict(model=0,forward=0,step=0,neural=0),physical_units='NOT_ESTABLISHED',unloading_is_not_liftoff=True))
 fig,axs=plt.subplots(6,3,figsize=(15,18),sharex=True)
 for l,leg in enumerate(legs):
  for label,(a,active,force,raw) in arrays.items():
   rows=[r for r in legrows if r['arm']==label and r['leg']==leg];x=np.array([r['time_s'] for r in rows])*1000
   for ax,key in zip(axs[l],['Fz','act_sum','actual_sum']):ax.plot(x,[r[key] for r in rows],label=label)
   axs[l,2].plot(x,[r['raw_sum'] for r in rows],':',alpha=.55)
  for ax,title in zip(axs[l],['foot-floor Fz (native)','sum activation','sum actual ctrl; dotted raw']):ax.set_title(leg+' '+title);ax.grid(alpha=.3);ax.legend(fontsize=7);ax.set_xlabel('physical ms')
 fig.suptitle('S150 saved-array six-leg allocation; unloading != liftoff; terminal S146 fail included');fig.tight_layout();fig.savefig(out/'SIX_LEG_LOAD_COMMAND_DYNAMICS.png',dpi=135);plt.close(fig)
 report=['# S150 / P1 六腿载荷下降与神经命令分配（离线诊断）','不是成品、预热成功或自主行走验收。模型/forward/step/神经调用均为0；native物理单位 NOT_ESTABLISHED。','## 定义与范围','持续下降起点严格按跑前 PREANALYSIS 定义：从该样本到声明窗末，每个差分均 < −1e−10；这是依赖窗末的描述，不是独立因果起点。载荷下降绝不等于离地。S146/S149仅共同78样本0–1.925ms（含S146失败末样本）；S148仅11样本0–0.25ms，不外推。','映射使用实际XML空间肌腱site祖先路径差集/固定joint引用、真实MOMENT_METADATA关节DOF、原runner joint_LEG归属规则；不按肌肉名称猜功能。UNKNOWN角色保留。逐时CSR非零DOF均必须位于真实肌腱跨越关节内。拮抗仅同DOF异号moment且同时有主动力；不凭名字判断。','## 实测摘要',json.dumps(summary,ensure_ascii=False,indent=2),'## 半幅接口非线性比较',json.dumps(comparisons,ensure_ascii=False,indent=2),'S149 activation_maxabs=0 是 RECONSTRUCTIONERROR，不是真实零激活；本报告从实际保存act全通道直接计算。sample0控制仍为0；其后严格按post-tick hold映射rawdecoded与applied，不能把tick后的命令回填sample0。','接触CSV保留世界力、接触点和距离；逐腿负距离与bearing分列。六腿合计核对foot_force及load分子，scalar gain×act + bias与实际力及全48DOF投影闭合。','## 唯一建议（不执行）','在保持神经网络/权重/感觉/姿态与原门不变前提下，仅将神经命令到肌肉的激活建立时间常数作为一个有界干预变量做一次预登记对照；不再盲扫全局幅度。依据是半幅虽减小反弹与主动功仍几乎同窗失撑，且真实激活动力学/FLV使幅度非线性，不是消除神经或外置控制器的证据。具体新数值需另行预登记，本轮没有实测或改参数。']
 (out/'REPORT.zh-CN.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
 after={str(p):sha(p) for p in inputs};assert after==hashes;dump(out/'INPUT_HASHES_AFTER.json',after);dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()});print(str(out))
if __name__=='__main__':main()
