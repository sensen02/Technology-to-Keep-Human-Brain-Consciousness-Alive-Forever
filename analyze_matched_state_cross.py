"""S153/P1: bounded saved-array arithmetic only; no engine/model/neural calls."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, shutil, traceback
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_muscle_reconstruction import evaluate
P=Path('/run/media/sensen/Data2/cell_wound_prototype')
B=Path('/home/sensen/Desktop/engine修复验收')
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
MAP=B/'s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json'
S152=B/'s152-opposing-tendon-work/20261024T214726426780Z'
SOURCES={'S149':B/'s149-half-gain-physical/20261024T213108429114Z/init80ms','S151':B/'s151-lf-channel-gain/20261024T214136681607Z/init80ms'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def table(p,rows):
 with p.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cum(y,t):
 return np.concatenate([np.zeros_like(y[:1]),np.cumsum((y[1:]+y[:-1])*np.diff(t).reshape((-1,)+(1,)*(y.ndim-1))/2,axis=0)])
def err(a,b):return float(np.max(np.abs(a-b)))
def check(a,b):
 e=err(a,b);assert e<=1e-10,e
 return e
def compute(out):
 inputs=[Path(__file__),P/'verify_muscle_reconstruction.py',XML,MAP,S152/'SUMMARY.json',S152/'PREANALYSIS.json',S152/'S149_RECONSTRUCTED.npz',S152/'S151_RECONSTRUCTED.npz']+[p/f for p in SOURCES.values() for f in ['TRAJECTORY.npz','NATIVE_MODEL_ARRAYS.npz','MOMENT_METADATA.json','MODEL_PROVENANCE.json']]
 hashes={str(p):sha(p) for p in inputs};dump(out/'INPUT_HASHES.json',hashes)
 dump(out/'PREANALYSIS.json',dict(task='S153/P1 bounded OFFLINE matched-state cross',sources={k:str(v) for k,v in SOURCES.items()},reference=str(S152),samples='80 common native samples 0..1.975ms, terminal failed gate retained',frozen='Original XML gain/bias/lengthrange; host act, host CSR moment and host qvel fixed. Swap only saved L and/or V, never model geometry or neural signals.',corners='For each host: native, L-only, V-only, LV from other donor; repeated identical formula and matched other89 force <=1e-10; channel4 activation difference separate.',power='Scalar F*donorV generally differs from F*(HOSTM@HOSTqvel). Per-actuator minimum-norm compatible qvel=hostqvel+Mrow*(V-Mrow@hostqvel)/norm(Mrow)^2. Each actuator has its OWN synthetic velocity; NOT one physical state. Save both mismatch and compatible algebra closure. Zero rows require target V within 1e-10; no invented solution.',decompositions='Native donor minus native host torque = Mhost.T@(Fdonor-Fhost)+(Mdonor-Mhost).T@Fdonor. Force term further split into fixed-host-act LV change and donor activation residual. Power delta=(Fd-Fh)*Vh+Fd*(Vd-Vh).',work='Sampled signed trapezoids only, positive/negative signed separately, active/bias/total, integration errors retained; no continuous error bound or metabolic meaning.',opposition='Recompute positive/negative/net torque per DOF separately per native/cross and active/bias/total; min(positive,-negative) mechanical only.',conclusion='Only existing-formula explanation or residual failure; no LM/RM causal attribution, no new physical proposal; absolute units NOT_ESTABLISHED.',calls=dict(model=0,forward=0,step=0,neural=0)))
 freeze=out/'SOURCE_FREEZE';freeze.mkdir()
 for p in [Path(__file__),P/'verify_muscle_reconstruction.py',XML,MAP]:shutil.copyfile(p,freeze/p.name)
 nodes=list(ET.parse(XML).getroot().find('actuator'));maps=json.loads(MAP.read_text());assert len(nodes)==len(maps)==90
 arrays={};moments={};native={};errors={};results={};rows=[];works=[];dofrows=[];decomprows=[]
 for h,p in SOURCES.items():
  a=load(p/'TRAJECTORY.npz');m=load(p/'NATIVE_MODEL_ARRAYS.npz');meta=json.loads((p/'MOMENT_METADATA.json').read_text());assert [n.get('name') for n in nodes]==meta['actuator_names']==[r['name'] for r in maps]
  assert json.loads((p/'MODEL_PROVENANCE.json').read_text())['xml_sha256']==sha(XML)
  for k in ['gainprm','biasprm','dynprm','lengthrange']:
   v=[list(map(float,n.get(k).split())) for n in nodes]
   if k!='lengthrange':v=[x+[0.]*(10-len(x)) for x in v]
   assert np.array_equal(np.array(v),m['actuator_'+k])
  t=a['time'];assert len(t)==80;check(t,np.arange(80)*25e-6);M=np.zeros((80,90,48))
  for i in range(80):
   for u in range(90):
    adr=int(a['moment_rowadr'][i,u]);nnz=int(a['moment_rownnz'][i,u]);cols=a['moment_colind'][i,adr:adr+nnz];M[i,u,cols]=a['actuator_moment_raw'][i,adr:adr+nnz]
    assert set(cols[np.abs(M[i,u,cols])>1e-12]).issubset(maps[u]['dofs'])
  errors[h+'_moment']=check(M,a['actuator_moment']);errors[h+'_native_velocity']=check(np.einsum('tud,td->tu',M,a['qvel']),a['actuator_velocity'])
  arrays[h]=a;moments[h]=M
 assert np.array_equal(arrays['S149']['time'],arrays['S151']['time'])
 other=np.arange(90)!=4
 assert np.array_equal(arrays['S149']['act'][:,other],arrays['S151']['act'][:,other])
 for h,a in arrays.items():
  d='S151' if h=='S149' else 'S149';donor=arrays[d];M=moments[h];basev=np.einsum('tud,td->tu',M,a['qvel']);norm=np.sum(M*M,axis=2)
  for mode in ['native','L','V','LV']:
   key=h+'_'+mode;L=donor['actuator_length'] if mode in ['L','LV'] else a['actuator_length'];V=donor['actuator_velocity'] if mode in ['V','LV'] else a['actuator_velocity'];gain=np.zeros((80,90));bias=gain.copy();repeat=gain.copy()
   for i in range(80):
    for u,node in enumerate(nodes):
     f,g,b=evaluate(L[i,u],V[i,u],a['act'][i,u],node);gain[i,u]=g;bias[i,u]=b;repeat[i,u]=evaluate(L[i,u],V[i,u],a['act'][i,u],node)[0]
   forces={'active':gain*a['act'],'bias':bias,'total':gain*a['act']+bias};errors[key+'_repeated_force']=check(forces['total'],repeat)
   if mode=='native':
    errors[key+'_native_force']=check(forces['total'],a['actuator_force']);errors[key+'_native_torque']=check(np.einsum('tud,tu->td',M,forces['total']),a['qfrc_actuator']);native[h]=forces
    ref=load(S152/(h+'_RECONSTRUCTED.npz'));errors[key+'_S152_active']=check(forces['active'],ref['active_force']);errors[key+'_S152_bias']=check(bias,ref['passive_bias'])
   dv=V-basev;assert np.all(np.abs(dv[norm==0])<=1e-10)
   correction=np.divide(dv,norm,out=np.zeros_like(dv),where=norm>0);synthetic=a['qvel'][:,None,:]+M*correction[:,:,None];compatible=np.einsum('tud,tud->tu',M,synthetic)
   errors[key+'_compatible_velocity']=check(V,compatible)
   saved=dict(time=t,host_act=a['act'],length=L,velocity=V,host_moment=M,host_qvel=a['qvel'],host_projected_velocity=basev,velocity_mismatch=dv,compatible_qvel_per_actuator=synthetic,active_gain=gain)
   result=dict(errors={},work={},opposing_dofs={})
   for part,F in forces.items():
    tau=M*F[:,:,None];power=F*V;hostpower=F*basev;cp=(tau*synthetic).sum(2);errors[key+'_'+part+'_compatible_power']=check(power,cp)
    pos=np.maximum(tau,0).sum(1);neg=np.minimum(tau,0).sum(1);net=tau.sum(1);opp=np.minimum(pos,-neg)
    saved.update({part+'_force':F,part+'_torque_per_actuator':tau,part+'_torque':net,part+'_scalar_power':power,part+'_host_kinematic_power':hostpower,part+'_power_mismatch':power-hostpower,part+'_compatible_power':cp,part+'_positive_torque':pos,part+'_negative_torque':neg,part+'_opposing':opp})
    result['work'][part]={};result['opposing_dofs'][part]=np.flatnonzero(np.any(opp>1e-10,axis=0)).tolist()
    for role,y in [('signed',power),('positive',np.maximum(power,0)),('negative_signed',np.minimum(power,0))]:
     w=cum(y,t);saved[part+'_cumulative_'+role+'_work']=w;result['work'][part][role]=float(w[-1].sum())
     for u,r in enumerate(maps):works.append(dict(host=h,mode=mode,donor=d,part=part,role=role,actuator=u,name=r['name'],leg=r['leg'],sampled_work=float(w[-1,u])))
    errors[key+'_'+part+'_work_split']=check(saved[part+'_cumulative_signed_work'],saved[part+'_cumulative_positive_work']+saved[part+'_cumulative_negative_signed_work'])
    for i in range(80):
     for j in range(48):dofrows.append(dict(host=h,mode=mode,part=part,sample=i,time_s=float(t[i]),dof=j,positive=float(pos[i,j]),negative_signed=float(neg[i,j]),net=float(net[i,j]),mechanical_opposing=float(opp[i,j])))
   result['max_abs_velocity_mismatch']=float(abs(dv).max());result['max_abs_total_power_mismatch']=float(abs(saved['total_power_mismatch']).max());results[key]=result
   for i in range(80):
    for u,r in enumerate(maps):
     row=dict(host=h,mode=mode,donor=d,sample=i,time_s=float(t[i]),actuator=u,name=r['name'],leg=r['leg'],host_act=float(a['act'][i,u]),length=float(L[i,u]),velocity=float(V[i,u]),host_projected_velocity=float(basev[i,u]),velocity_mismatch=float(dv[i,u]),gain=float(gain[i,u]))
     for part in forces:
      for field in ['force','scalar_power','host_kinematic_power','power_mismatch','compatible_power']:row[part+'_'+field]=float(saved[part+'_'+field][i,u])
     rows.append(row)
   np.savez_compressed(out/(key+'_REBUILT.npz'),**saved)
 # Native comparisons/decompositions use donor NATIVE force, not the cross force at host activation.
 for h,a in arrays.items():
  d='S151' if h=='S149' else 'S149';b=arrays[d];Mh=moments[h];Md=moments[d];cross=load(out/(h+'_LV_REBUILT.npz'));parts={}
  errors[h+'_matched_other89']=check(cross['total_force'][:,other],native[d]['total'][:,other]);errors[h+'_channel4_activation_residual']=check(native[d]['total'][:,4]-cross['total_force'][:,4],cross['active_gain'][:,4]*(b['act'][:,4]-a['act'][:,4]))
  for part in ['active','bias','total']:
   Fh=native[h][part];Fd=native[d][part];Fc=cross[part+'_force'];force_term=np.einsum('tud,tu->td',Mh,Fd-Fh);geometry=np.einsum('tud,tu->td',Md-Mh,Fd);delta=np.einsum('tud,tu->td',Md,Fd)-np.einsum('tud,tu->td',Mh,Fh);state=np.einsum('tud,tu->td',Mh,Fc-Fh);activation=np.einsum('tud,tu->td',Mh,Fd-Fc)
   pe=(Fd-Fh)*a['actuator_velocity'];pv=Fd*(b['actuator_velocity']-a['actuator_velocity']);pd=Fd*b['actuator_velocity']-Fh*a['actuator_velocity']
   errors[h+'_'+part+'_torque_decomp']=check(delta,force_term+geometry);errors[h+'_'+part+'_force_subsplit']=check(force_term,state+activation);errors[h+'_'+part+'_power_decomp']=check(pd,pe+pv)
   parts.update({part+'_delta_torque':delta,part+'_force_term':force_term,part+'_geometry_term_at_donor_force':geometry,part+'_matched_state_force_term':state,part+'_activation_residual_term':activation,part+'_delta_power':pd,part+'_power_force_term':pe,part+'_power_velocity_term':pv})
   for i in range(80):
    for j in range(48):decomprows.append(dict(host=h,donor=d,part=part,sample=i,time_s=float(t[i]),dof=j,full_torque_delta=float(delta[i,j]),force_component=float(force_term[i,j]),geometry_at_donor_force=float(geometry[i,j]),matched_state_force_component=float(state[i,j]),activation_residual=float(activation[i,j]),residual=float(delta[i,j]-force_term[i,j]-geometry[i,j])))
  np.savez_compressed(out/(h+'_TO_'+d+'_DECOMPOSITION.npz'),time=t,**parts)
 # Retain endpoint-preserving 50us sampled quadrature sensitivity; NOT a continuous integral error bound.
 sensitivity={}
 for h in arrays:
  for mode in ['native','L','V','LV']:
   z=load(out/(h+'_'+mode+'_REBUILT.npz'));ids=np.unique(np.r_[np.arange(0,80,2),79]);sensitivity[h+'_'+mode]={part:dict(fine=float(cum(z[part+'_scalar_power'],t)[-1].sum()),coarse_endpoint_retained=float(cum(z[part+'_scalar_power'][ids],t[ids])[-1].sum()),coarse_minus_fine=float(cum(z[part+'_scalar_power'][ids],t[ids])[-1].sum()-cum(z[part+'_scalar_power'],t)[-1].sum())) for part in ['active','bias','total']}
 table(out/'ACTUATOR_TIMESTEP.csv',rows);table(out/'SAMPLED_WORK_SUMMARY.csv',works);table(out/'DOF_OPPOSITION.csv',dofrows);table(out/'TORQUE_DECOMPOSITION.csv',decomprows);dump(out/'SAMPLED_QUADRATURE_SENSITIVITY.json',sensitivity)
 fig,axs=plt.subplots(2,2,figsize=(13,9))
 for h in arrays:
  for mode in ['native','LV']:
   z=load(out/(h+'_'+mode+'_REBUILT.npz'));label=h+' '+mode
   axs[0,0].plot(t*1000,z['active_cumulative_signed_work'].sum(1),label=label);axs[0,1].plot(t*1000,z['active_cumulative_negative_signed_work'].sum(1),label=label);axs[1,0].plot(t*1000,z['total_power_mismatch'].sum(1),label=label);axs[1,1].plot(t*1000,z['active_opposing'].sum(1),label=label)
 for ax,title in zip(axs.flat,['Sampled signed active work','Sampled negative signed active work','Donor scalar minus HOST kinematic power','Recomputed active mechanical opposition']):ax.set_title(title);ax.set_xlabel('saved time (ms)');ax.grid(alpha=.3);ax.legend(fontsize=8)
 fig.suptitle('S153 offline formula cross: synthetic velocities are per actuator, NOT a physical state');fig.tight_layout();fig.savefig(out/'MATCHED_STATE_CROSS.png',dpi=150);plt.close(fig)
 result=dict(task='S153/P1',conclusion='EXISTING_FORMULA_EXPLAINS_MATCHED_OTHER89',errors_maxabs=errors,arms=results,calls=dict(model=0,forward=0,step=0,neural=0),physical_units='NOT_ESTABLISHED',causal_LM_RM='NOT_ESTABLISHED',future_physical_proposal=None)
 dump(out/'SUMMARY.json',result)
 text=['# S153 / P1 匹配状态离线交叉核验','结论：现有 active_gain(L,V) × host act + bias(L) 公式解释其他89路力差；未发现超过1e−10的匹配输入残差。索引4另有真实activation差，不归入相同输入。','仅保存80样本0–1.975ms；包含历史末失败样本，不是预热/自主成功。0模型/forward/step/神经调用；单位 NOT_ESTABLISHED。','每个host固定原act、原CSR moment、原gain/bias/lengthrange；分别仅换L、仅换V、换LV。反事实矩为HOST M.T @ F，不偷换donor几何。原生及重复公式均逐项校验。','## 非物理功率投影边界','DONOR V一般不等于HOST M @ HOST qvel。CSV/NPZ保存速度不匹配与F*DONOR V − F*(HOST M@HOST qvel)，此差是预期，不宣称原host功率闭合。为代数验证，每个actuator各自用minimum-norm row correction构造compatible qvel，并验证F*V=sum_d(Mrow_d*F*synthetic_qvel_d)。90个synthetic速度并非单个身体状态，不能作接触/运动预测。','## 精确差分','原生全矩差=HOST M.T@(Fdonor−Fhost)+(Mdonor−Mhost).T@Fdonor；第一项再拆成host-act匹配LV力项和donor-act残差。功率差=(Fd−Fh)*Vh+Fd*(Vd−Vh)。全部48DOF及每执行器power残差保留于NPZ/CSV。','active/bias/total逐样本正负功率分别梯形积分，signed=positive+negative_signed；保留50us对25us端点不丢的采样敏感性，非连续积分误差界。逐DOF正矩、负矩、net与mechanical opposition对各原生和交叉独立复算；机械对抗或负功不等有害/浪费。','## 实测残差',json.dumps(errors,ensure_ascii=False,indent=2),'## 结论边界','仅现有公式解释通过；不作LM/RM卸载因果结论，不提出未来物理干预，不改变S113/checkpoint、tracker及任何历史文件。']
 (out/'REPORT.zh-CN.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
 after={str(p):sha(p) for p in inputs};dump(out/'INPUT_HASHES_AFTER.json',after);assert hashes==after
 dump(out/'READY.json',dict(ready=True,output_root=str(out),calls=result['calls'],source_sha256=sha(Path(__file__))))
 dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.rglob('*') if p.is_file()});print('READY '+str(out),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);out=ap.parse_args().output or B/'s153-matched-state-cross'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
 try:compute(out)
 except Exception:
  (out/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8');print('FAILED preserved '+str(out),flush=True);raise
if __name__=='__main__':main()
