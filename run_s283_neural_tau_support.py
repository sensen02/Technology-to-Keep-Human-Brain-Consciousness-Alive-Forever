"""S283: original S276 loop, isolated synchronous diagnostics, single tau pair."""
from pathlib import Path
import sys,json,csv,hashlib,argparse,dataclasses,traceback
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');OLD=Path('/home/sensen/Desktop/engine修复验收/s276-the-one-cell-that-stops/20261027T010000000000Z/scripts/s276.py');REF=Path('/home/sensen/Desktop/engine修复验收/s277-silence-attribution/20261027T030000000000Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(str(type(x)))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=serial)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
 if not out.is_absolute() or out.exists():raise ValueError('fresh absolute output required')
 source=OLD.read_text();prefix=source[:source.index('mm=mujoco.MjModel')];setup=source[source.index('joints={}'):source.index('EVERY=10')]
 ns={'__name__':'s283_declarations'};exec(compile(prefix,str(OLD)+'::declarations','exec'),ns)
 paths=[OLD,Path(__file__),REF/'CTRL_S277.npz',REF/'STATE_S277.csv',ns['FROZEN'],ns['S177'],ns['S244']/'CMD_A3_sym_matched.npz',ns['S255']/'ARM_VECTORS.json',ns['POOLS'],ROOT/'data/flywire/banc_connectome.npz',ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json',*ROOT.glob('engine/**/*.py')]
 from run_initialized_translation_training import mesh_dependencies
 paths+=mesh_dependencies();ident={str(p):sha(p) for p in dict.fromkeys(paths)};out.mkdir(parents=True)
 dump(out/'PREREG.json',{'round':'S283','arms':['baseline','tau_equal'],'duration_ms':400,'seed':0,'network':'all_empty ASSUMED_NETWORK_SCORE','parameter_diff':{'inh_tau_ms':[8.,3.]},'exc_tau_ms':3.,'source_hashes':ident,'dt_s':.0001,'neural_dt_ms':.5,'sensory_ms':1,'gate_columns':['dz','dx','dy','horiz','leg_up_pen_W','tib_excursion'],'gate':'first200ms baseline CTRL and six CSV columns exact; failure stops before candidate','budget':{'model_load':1,'tier_build':1,'network_init':2,'live_step':8000,'neural_tick':1600,'live_fwd_position':2,'live_forward':0,'diagnostic_forward':8002},'metrics':['target spikes0..200/200..400,last spike,max gap','first body contact','foot loads and COM dz400','root horizontal,orientation,foot slip,limits'],'sync':'live legacy loop unchanged; separate MjData mj_copyData then mj_forward at each saved qpos, no diagnostic feedback','limits':'uncalibrated sensitivity, no support success inferred from spikes','legacy':'frame@raw up=-z for sensation/history only; physical diagnostics frame.T raw on geom2 signed by floor side'})
 dump(out/'CONFIG_DIFF.json',{'baseline':{'exc_tau_ms':3.,'inh_tau_ms':8.},'tau_equal':{'exc_tau_ms':3.,'inh_tau_ms':3.},'unchanged':'all other parameters,seed,connectivity,model,decoder,V1,sensation; model default integrator'})
 mj=ns['mujoco'];m=mj.MjModel.from_xml_path(str(ns['FROZEN']));ns['mm']=m;built=ns['build_neural_tier'](ns['TierConfig'](sign_fill='all_empty').validate());assert built['fingerprint']=='56e0322a33c51b7f03aaae9f7df1ef3f'
 N=int(built['n']);groups,_=ns['channel_index_map'](N,np.asarray(built['tier']['root_ids']));names=ns['NAMES'];man=ns['build_motor_manifest'](json.loads(ns['POOLS'].read_text()),np.asarray(built['tier']['root_ids']),names);ns.update(built=built,N=N,groups=groups,man=man);exec(compile(setup,str(OLD)+'::legacy_ledger','exec'),ns)
 ledger=ns['ledger'];joints=ns['joints'];legs=ns['LEGS'];floor=ns['FLOOR'];geom_leg=ns['geom_leg'];qrel=ns['qrel'];target=names.index('RHC_pleural_remotor_and_abductor');roots=np.asarray(built['tier']['root_ids']);resolved={r['actuator']:r for r in json.loads(ns['POOLS'].read_text())['measured']['resolved']};cells=[int(np.flatnonzero(roots==int(c))[0]) for c in resolved['RHC_pleural_remotor_and_abductor']['pool_root_ids'] if np.any(roots==int(c))]
 calls={'model_load':1,'tier_build':1,'network_init':0,'live_step':0,'neural_tick':0,'live_fwd_position':0,'live_forward':0,'diagnostic_forward':0};dump(out/'CALL_LEDGER.json',calls)
 dump(out/'MODEL_METADATA.json',{'names':names,'body_names':[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i) for i in range(m.nbody)],'geom_leg':geom_leg,'floor':floor,'integrator':int(m.opt.integrator),'timestep':m.opt.timestep,'nmesh':m.nmesh,'target_index':target,'target_cells':cells,'fingerprint':built['fingerprint']})
 assert m.opt.timestep==.0001 and (m.nq,m.nv,m.nu,m.nmesh)==(49,48,90,71)
 def run(label):
  dest=out/label;dest.mkdir();d=mj.MjData(m);d.qpos[:]=qrel;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mj.mj_fwdPosition(m,d);calls['live_fwd_position']+=1;q0=d.qpos.copy();diag=mj.MjData(m)
  tier=ns['NeuralTier'](built,ns['TierConfig'](sign_fill='all_empty',background_seed=0).validate(),ns['OwnershipLedger']());calls['network_init']+=1
  if label=='tau_equal':tier.net.p=dataclasses.replace(tier.net.p,inh_tau_ms=tier.net.p.exc_tau_ms);tier.net.p.validate()
  dump(dest/'PARAMETERS.json',dataclasses.asdict(tier.net.p));enc=ns['ChannelSensoryEncoder'](groups,N,ns['ChannelConfig']());dec=ns['MotorOutput'](man,ns['MotorOutputConfig'](rate_for_full_control_hz=450.,per_actuator_gain=ns['V1'],filter_tau_s=5e-3));carry=np.zeros(90);traj=[];ctrls=[];rows=[];ticks=[];sens=[];rates=[];rawrates=[];channels=[]
  step=tier.net.step
  def tick(*a,**kw):
   sp=step(*a,**kw);ticks.append({'time_ms':tier.net.t_ms,'spikes':sp[cells].copy(),'ge':tier.net.ge[cells].copy(),'gi':tier.net.gi[cells].copy(),'v':tier.net.v[cells].copy()});return sp
  tier.net.step=tick
  def snapshot():
   mj.mj_copyData(diag,m,d);mj.mj_forward(m,diag);calls['diagnostic_forward']+=1;mass=m.body_mass;total=mass.sum();cv=np.zeros((m.nbody,6))
   for b in range(m.nbody):mj.mj_objectVelocity(m,diag,mj.mjtObj.mjOBJ_BODY,b,cv[b],0)
   force=np.zeros((6,3));penetrating=np.zeros(6,int);body=0;raw=[]
   for ci in range(diag.ncon):
    c=diag.contact[ci];g1,g2=int(c.geom1),int(c.geom2);fx=np.zeros(6);mj.mj_contactForce(m,diag,ci,fx);raw.append([g1,g2,float(c.dist),*c.pos,*np.asarray(c.frame).ravel(),*fx])
    if floor not in (g1,g2):continue
    other=g2 if g1==floor else g1;fw=(1 if g1==floor else -1)*(np.asarray(c.frame).reshape(3,3).T@fx[:3])
    if other in geom_leg:li=legs.index(geom_leg[other]);force[li]+=fw;penetrating[li]+=int(c.dist<0)
    elif c.dist<0:body+=1
   limited=np.flatnonzero(m.jnt_limited);qa=m.jnt_qposadr[limited];lim=np.maximum(np.maximum(m.jnt_range[limited,0]-diag.qpos[qa],diag.qpos[qa]-m.jnt_range[limited,1]),0).max()
   return {'time':float(d.time),'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'act':d.act.copy(),'ctrl':d.ctrl.copy(),'qacc':diag.qacc.copy(),'body_xpos':diag.xpos.copy(),'body_xipos':diag.xipos.copy(),'body_velocity':cv,'com':(mass[:,None]*diag.xipos).sum(0)/total,'com_velocity':(mass[:,None]*cv[:,3:]).sum(0)/total,'foot_force_world':force,'foot_penetrating_count':penetrating,'body_penetrating_contacts':body,'max_limit':lim,'external_applied':diag.xfrc_applied.copy(),'qfrc_applied':diag.qfrc_applied.copy(),'contact_raw':raw,'qfrc_actuator':diag.qfrc_actuator.copy(),'qfrc_constraint':diag.qfrc_constraint.copy()}
  traj.append(snapshot())
  def save():
   np.savez_compressed(dest/'TRAJECTORY.npz',**{k:np.asarray([r[k] for r in traj]) for k in traj[0] if k!='contact_raw'});dump(dest/'CONTACTS.json',[r['contact_raw'] for r in traj]);np.savez_compressed(dest/'NEURAL.npz',time_ms=np.array([r['time_ms'] for r in ticks]),**{k:np.array([r[k] for r in ticks]) for k in ['spikes','ge','gi','v']},sensory_current=np.asarray(sens),rates_hz=np.asarray(rates),raw_rates_hz=np.asarray(rawrates));np.savez_compressed(dest/'CTRL.npz',ctrl=np.asarray(ctrls));dump(dest/'CHANNELS.json',channels)
   with (dest/'STATE.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   dump(out/'CALL_LEDGER.json',calls)
  for i in range(400):
   d.ctrl[:]=carry;lp,per,bf=ledger(d);qv={lg:ns['ChannelQuantities'](tibia_joint_dev_rad=float(d.qpos[joints[lg]['tib_q']]-q0[joints[lg]['tib_q']]),tibia_joint_vel_rad_s=float(d.qvel[joints[lg]['tib_v']]),multi_joint_dev_rad=float(np.mean(d.qpos[joints[lg]['q']]-q0[joints[lg]['q']])),contact_force_norm=float(np.linalg.norm(per[lg]['F_world'])),foot_contact=float(per[lg]['touch'])) for lg in legs};cur,rec=enc.encode(qv);sens.append(cur.copy());channels.append(rec);sp=tier.advance(2,cur);calls['neural_tick']+=2;rawrates.append([np.array([sample[idx].mean()/.0005 if len(idx) else 0 for idx in dec.groups]) for sample in sp]);carry,info=dec.consume(sp,spike_dt_s=.0005,start_s=i*.001,end_s=i*.001+.001,connected=True);rates.append(np.asarray(info['rates_hz']));a=np.clip(carry,0,1.)
   for j in range(10):d.ctrl[:]=a;ctrls.append(a.copy());mj.mj_step(m,d);calls['live_step']+=1;traj.append(snapshot())
   lp,_,_=ledger(d);tib=np.array([d.qpos[joints[lg]['tib_q']] for lg in legs]);dx=float(d.qpos[0]-q0[0]);dy=float(d.qpos[1]-q0[1]);rows.append(dict(t_ms=float(i+1),dx=dx,dy=dy,dz=float(d.qpos[2]-q0[2]),horiz=float(np.hypot(dx,dy)),leg_up_pen_W=lp['lp']/ns['W'],tib_excursion=float(np.mean(abs(tib-ns['TIB0']))),n_body_ground=int(lp['np_']+lp['nm']),nbg_pen=int(lp['np_'])))
   if label=='baseline' and i==199:
    rc=np.load(REF/'CTRL_S277.npz',allow_pickle=True)['ctrl'];rr=list(csv.DictReader((REF/'STATE_S277.csv').open()));err={k:max(abs(r[k]-float(rr[n][k])) for n,r in enumerate(rows)) for k in ['dz','dx','dy','horiz','leg_up_pen_W','tib_excursion']};ce=float(abs(np.asarray(ctrls)-rc).max());ok=ce==0 and all(x==0 for x in err.values());dump(out/'GATE_BASELINE.json',{'passed':ok,'ctrl_max_abs':ce,'state_max_abs':err})
    if not ok:save();return False
  save();return True
 try:
  if not run('baseline'):dump(out/'STOP.json',{'reason':'baseline reproduction failed; candidate not run'});return 2
  run('tau_equal');dump(out/'COMPLETE.json',{'capture_complete':True,'calls':calls,'source_unchanged':all(sha(p)==h for p,h in ident.items()),'auxiliary_force':False,'training':False})
 except BaseException:dump(out/'FAILED.json',{'traceback':traceback.format_exc(),'calls':calls});raise
 return 0
if __name__=='__main__':raise SystemExit(main())
