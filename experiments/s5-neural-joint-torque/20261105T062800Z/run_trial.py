"""Isolated S5 torque readout pilot. Archived inputs are read-only.
Algorithmic warmup is explicitly excluded from autonomous scoring.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, hashlib, time, copy, traceback, resource
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
R = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path[:0] = [str(R), str(R/'tools')]
O = Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T062800Z')
P = Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z')
B = Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
A = Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
W = Path('/home/sensen/Desktop/engine修复验收/s5-drive-chain-diagnosis/20261105T054144Z/baseline/EVAL/RUN')
AB = Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS')
DT=0.0000125; NDT=.0005; CDT=.001; RELEASE=.2
LEGS=('LF','LM','LH','RF','RM','RH')
def js(p): return json.loads(Path(p).read_text())
def dump(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
    return [P/n for n in ['MODEL_JOINT_TORQUE.xml','POOL_TO_JOINT.json','JOINT_PLANT_LIMITS.json']] + [B/'RUN/MAPPING.json',B/'IDENTITIES.npz',A/'SENSORY_LOOKUP_SNAPSHOT888.json'] + [AB/a/('ACTUAL_'+k+'.npz') for a in ['A','C'] for k in ['We','Wi']] + [W/n for n in ['GAIT.npz','GAINS.json']] + [Path(__file__),R/'engine/neural_cond.py',R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py',R/'tools/dynamic_support_metrics.py',R/'tools/train_dynamic_support.py']
def prereg():
    O.mkdir(parents=True,exist_ok=False)
    cfg=dict(stage='S5 pilot; not walking certification',body_dt_s=DT,neural_dt_s=NDT,command_dt_s=CDT,warmup_s=1.3,neural_prewarm_start_s=.2,release_s=RELEASE,
        readout='clip(g_j * sum_p(s_pj * w_p * filtered_rate_p), +/- fixed_limit_j)',pool_filter_tau_s=.005,
        sign='dominant joint only, signed union per unique pool; conflicting signs abort',weights='w=exp(z_pool), g=limit/100 * exp(z_joint); z bounded [-2,2]; nonnegative gains preserve sign',
        neural='archived A and C matrices directly, no extra 0.2 scaling; default ConductanceParams and existing sensory encoder defaults; prior muscle-trained thresholds/gains are NOT transferred',
        warmup='validated PD torque-only law drives plant; SNN observes actual body from 0.2s; proposed readout is withheld until 1.3s. Same full physical/neural/filter state continues at release. Shared prewarm can be cached as it does not depend on learned readout.',
        budget=dict(total_cap=200,pilot_training_per_arm=17,description='each of P0/P2: baseline + two generations of 8 Gaussian proposals, sigma .35 then .20, select best incumbent; same candidate perturbation seeds; P1 cut and exact repeats after training'),
        seeds=dict(neural=0,search=20261105),objective='unchanged train_dynamic_support.objective; same-run assisted reference last .2s',
        criteria='E1 exact repeated score/trajectory; E2 zero qfrc_applied/xfrc_applied in release; E3 finite state and zero warnings; D1-D8 unchanged, D1 checked positive nonfoot max; C1/C2 exploratory single seed; no statistical significance claim',
        stop='abort numerical/identity/hash failures; retain all failures. No extended gait claims if dynamic gates fail.',
        resource='one process, one heavy job, full trajectory every pilot candidate retained',units='absolute forces and SI NOT_ESTABLISHED',importance='核心必需')
    dump(O/'PREREGISTRATION.json',cfg)
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in sources()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(P/'PREREGISTRATION_NEXT.json'),old_sha256=sha(P/'PREREGISTRATION_NEXT.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='Make pilot initialization, budget, nonnegative parameter domain, and teacher-only warmup explicit before simulation. Freeze neural/sensory defaults rather than transfer muscle-trained parameters.'))
    print(str(O),flush=True)

class Trial:
    def __init__(self):
        import mujoco
        from flygym.compose.fly.musculoskeletal import _load_mjcf
        from run_integrated_repaired_closure import identities
        self.mj=mujoco; self.m=_load_mjcf(P/'MODEL_JOINT_TORQUE.xml').compile(); m=self.m
        assert (m.nq,m.nv,m.nu,m.nmesh,m.neq)==(49,48,42,71,0)
        assert m.opt.timestep==DT
        self.q0=m.key_qpos[0].copy(); self.limits=m.actuator_ctrlrange[:,1].copy()
        self.jids=m.actuator_trnid[:,0].copy();self.qa=m.jnt_qposadr[self.jids]; self.va=m.jnt_dofadr[self.jids]
        assert len(set(self.jids))==42 and np.all(self.va>=6)
        self.thorax=m.body('Thorax').id;self.joints,self.geom=identities(m)
        self.bids=[m.body(l+'Tarsus5').id for l in LEGS]
        self.planes=m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
        self.feet=np.array([next((i for i,l in enumerate(LEGS) if (m.geom(g).name or '').startswith(l) and 'Tars' in (m.geom(g).name or '')),-1) for g in range(m.ngeom)])
        rows=js(P/'POOL_TO_JOINT.json')['muscle_to_joint']; man=js(B/'RUN/MAPPING.json')
        self.pools=[];lookup={}
        for a,r in enumerate(rows):
            assert r['local_indices']==man['rows'][a]['local_indices']
            key=tuple(r['local_indices'])
            if key and key not in lookup:lookup[key]=len(self.pools);self.pools.append(np.array(key,int))
        assert len(self.pools)==66 and len(set(np.concatenate(self.pools)))==204
        self.S=np.zeros((42,66)); seen={}
        for r in rows:
            key=tuple(r['local_indices'])
            if not key:continue
            j=int(np.flatnonzero(self.jids==m.joint(r['joint']).id)[0]);p=lookup[key]
            if (j,p) in seen:assert seen[j,p]==r['sign'],'conflicting signs in shared pool'
            seen[j,p]=r['sign'];self.S[j,p]=r['sign']
        coverage=[dict(joint=m.joint(int(jid)).name,positive_pools=int((self.S[j]>0).sum()),negative_pools=int((self.S[j]<0).sum()),limit=float(self.limits[j])) for j,jid in enumerate(self.jids)]
        dump(O/'COVERAGE.json',dict(pools=[p.tolist() for p in self.pools],sign_matrix=self.S,joints=coverage,uncovered=sum(not np.any(row) for row in self.S),bidirectional=sum(np.any(row>0) and np.any(row<0) for row in self.S)))
        self.poolmat=sparse.csr_matrix((np.concatenate([np.full(len(p),1/len(p)) for p in self.pools]),(np.repeat(np.arange(66),[len(p) for p in self.pools]),np.concatenate(self.pools))),shape=(66,12000))
        g=np.load(W/'GAIT.npz');self.gait=CubicSpline(g['phase'],g['qpos_hinges'],axis=0,bc_type='periodic');self.gains=js(W/'GAINS.json')
        assert np.array_equal(g['qpos_addresses'],self.qa)
        from run_neural_muscle_closure import channel_index_map
        roots=np.load(B/'IDENTITIES.npz')['root_ids'];assert len(roots)==12000
        self.groups,_=channel_index_map(12000,roots,lookup_path=A/'SENSORY_LOOKUP_SNAPSHOT888.json')
        self.cache={}
    def snapshot(self,d):
        mj=self.mj; spec=mj.mjtState.mjSTATE_INTEGRATION
        s=np.zeros(mj.mj_stateSize(self.m,spec));mj.mj_getState(self.m,d,s,spec);return s
    def contact(self,d):
        f=np.zeros(6); nf=0.;nfs=0.;nc=0;v=np.zeros(6)
        for i in range(d.ncon):
            co=d.contact[i];a,b=int(co.geom1),int(co.geom2)
            if not(self.planes[a] or self.planes[b]):continue
            self.mj.mj_contactForce(self.m,d,i,v)
            z=float((co.frame.reshape(3,3).T@v[:3])[2])*(1 if self.planes[a] else -1)
            li=self.feet[b if self.planes[a] else a]
            if li<0:nf+=max(0,z);nfs+=z;nc+=1
            else:f[li]+=z
        return np.r_[d.time,f,nf,nfs,nc,d.subtree_com[self.thorax,2],d.qpos[:7],np.mean(d.ctrl**2),np.max(abs(d.qfrc_applied[:6]))]
    def bodyrow(self,d):
        return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),com=d.subtree_com[self.thorax].copy(),feet=d.xpos[self.bids].copy())
    def observe_neural(self,d,net,enc,rates):
        from run_neural_muscle_closure import observe_quantities
        cur,ch=enc.encode(observe_quantities(self.m,d,self.q0,self.joints,self.geom));cur=np.asarray(cur)
        decay=np.exp(-NDT/.005);counts=np.zeros(12000,dtype=np.int32)
        for _ in range(2):
            sp=net.step(current_nA=cur);counts+=sp
            rates=decay*rates+(1-decay)*np.asarray(self.poolmat@sp,dtype=float)/NDT
        return rates,counts,cur
    def warm(self,arm):
        from engine.neural_cond import ConductanceNetwork,ConductanceParams
        from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
        np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=NDT*1000)
        graph='C' if arm=='P2' else 'A'
        net.set_connectivity(*[sparse.load_npz(AB/graph/('ACTUAL_'+k+'.npz')) for k in ['We','Wi']])
        enc=ChannelSensoryEncoder(self.groups,12000,ChannelConfig())
        m=self.m;mj=self.mj;d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d)
        rates=np.zeros(66);body=[self.bodyrow(d)];native=[];neural=[]
        kp=self.gains['kp'];kd=np.asarray(self.gains['kd']);clip=self.gains['clip']
        for step in range(round(1.3/DT)):
            if step>=round(.2/DT) and step%80==0:
                rates,sp,cur=self.observe_neural(d,net,enc,rates)
                neural.append((float(d.time),rates.copy(),sp.copy(),cur.copy()))
            t=float(d.time);ramp=np.clip((t-.15)/.10,0,1);phi=max(0,t-.15)*2*np.pi*5
            qg=self.gait(phi%(2*np.pi));des=(1-ramp)*self.q0[self.qa]+ramp*qg
            vel=ramp*self.gait(phi%(2*np.pi),1)*2*np.pi*5+(qg-self.q0[self.qa])*(10. if .15<t<.25 else 0.)
            tau=kp*(des-d.qpos[self.qa])+kd*(vel-d.qvel[self.va])+d.qfrc_bias[self.va]-d.qfrc_passive[self.va]
            d.ctrl[:]=np.clip(np.clip(tau,-clip,clip),-self.limits,self.limits);d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
            mj.mj_step(m,d);mj.mj_forward(m,d)
            if step>=round(1.1/DT):native.append(self.contact(d))
            if (step+1)%40==0:body.append(self.bodyrow(d))
            assert not d.warning.number.any() and np.isfinite(d.qpos).all()
        folder=O/('WARM_'+arm);folder.mkdir()
        np.savez_compressed(folder/'STATE.npz',integration=self.snapshot(d),rates=rates,**{'net_'+k:v for k,v in vars(net).items() if isinstance(v,np.ndarray)})
        np.savez_compressed(folder/'BODY.npz',**{k:np.asarray([r[k] for r in body]) for k in body[0]})
        np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(native))
        np.savez_compressed(folder/'NEURAL.npz',time=[r[0] for r in neural],rates=[r[1] for r in neural],spikes=[r[2] for r in neural],currents=[r[3] for r in neural])
        native=np.asarray(native)
        assist=dict(foot_support_mean=float(native[:,1:7].sum(1).mean()/(m.body_mass.sum()*abs(m.opt.gravity[2]))),com_z_mean=float(native[:,10].mean()))
        self.cache[arm]=dict(state=self.snapshot(d),net=net,enc=enc,rates=rates,assist=assist)
        print('WARM',arm,assist,flush=True)
    def evaluate(self,arm,z,label,cut=False,previous=None):
        import dynamic_support_metrics as dsm
        import train_dynamic_support as dst
        stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+f'{time.time_ns()%1000000000:09d}Z'
        folder=O/(stamp+'_'+label);folder.mkdir();start=time.monotonic()
        dump(folder/'PARAMETERS.json',dict(arm=arm,cut=cut,z=z,reason=label))
        dump(folder/'PREREGISTRATION.json',dict(parent=str(O/'PREREGISTRATION.json'),parameters_sha256=sha(folder/'PARAMETERS.json'),previous_file=previous,previous_sha256=sha(previous) if previous else None,new_file=str(folder/'PARAMETERS.json'),reason=label))
        c=self.cache[arm];net=copy.deepcopy(c['net']);enc=copy.deepcopy(c['enc']);rates=c['rates'].copy();m=self.m;mj=self.mj;d=mj.MjData(m)
        mj.mj_setState(m,d,c['state'],mj.mjtState.mjSTATE_INTEGRATION);mj.mj_forward(m,d)
        before=self.snapshot(d);q_before=d.qpos.copy();v_before=d.qvel.copy();w=np.exp(z[:66]);gain=self.limits/100*np.exp(z[66:])
        decode=lambda rr:np.clip(gain*(self.S@(w*rr)),-self.limits,self.limits)
        pending=decode(rates);d.ctrl[:]=0 if cut else pending;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;mj.mj_forward(m,d)
        assert np.array_equal(d.qpos,q_before) and np.array_equal(d.qvel,v_before), 'physical handover discontinuity'
        native=[self.contact(d)];body=[self.bodyrow(d)];tr=[];error=None
        np.savez_compressed(folder/'BOUNDARY.npz',before=before,after=self.snapshot(d),qpos=d.qpos,qvel=d.qvel,rates=rates,pending=pending)
        try:
            for k in range(round(RELEASE/CDT)):
                applied=np.zeros(42) if cut else pending.copy();source=rates.copy()
                rates,sp,cur=self.observe_neural(d,net,enc,rates)
                d.ctrl[:]=applied
                for j in range(80):
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    mj.mj_step(m,d);mj.mj_forward(m,d);native.append(self.contact(d))
                    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not d.warning.number.any()
                    if (j+1)%40==0:body.append(self.bodyrow(d))
                pending=decode(rates)
                tr.append(dict(time=float(d.time),source_rates=source,rates=rates.copy(),applied=applied,next=pending.copy(),spikes=sp,currents=cur))
            rows=np.asarray(native);J,parts=dst.objective(rows,c['assist']);full=dsm.evaluate_native(c['assist'],rows)
            full['checks']['D1_no_nonfoot_ground_force']=bool(np.max(rows[:,7])==0)
            full['passed']=all(full['checks'].values())
            result=dict(label=label,arm=arm,cut=cut,J=J,parts=parts,validation=full,folder=str(folder),wall_s=time.monotonic()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,release_aux_max=0.,release_external_max=0.,warnings=int(d.warning.number.sum()),finite=True,walking_certified=False,
                displacement_xy=(d.qpos[:2]-body[0]['qpos'][:2]),saturation_fraction=float(np.mean(np.abs(np.array([r['applied'] for r in tr]))>=self.limits[None,:])),units='NOT_ESTABLISHED')
            dump(folder/'RESULT.json',result)
        except BaseException:
            error=traceback.format_exc();dump(folder/'FAILURE.json',dict(error=error));raise
        finally:
            np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(native))
            np.savez_compressed(folder/'BODY.npz',**{k:np.asarray([r[k] for r in body]) for k in body[0]})
            if tr:np.savez_compressed(folder/'NEURAL.npz',**{k:np.asarray([r[k] for r in tr]) for k in tr[0]})
        print(label,round(J,6),'pass',full['passed'],'wall',round(result['wall_s'],2),flush=True)
        return result

def run():
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in sources()}
    t=Trial();results={};history=[]
    try:
        for arm in ['P0','P2']:
            t.warm(arm);x=np.zeros(108);base=t.evaluate(arm,x,arm+'_baseline');best=base;bestx=x.copy();rng=np.random.default_rng(20261105)
            history.append(base)
            for gen,sigma in enumerate([.35,.20]):
                center=bestx.copy();prev=str(Path(best['folder'])/'PARAMETERS.json')
                for i in range(8):
                    z=np.clip(center+sigma*rng.standard_normal(108),-2,2)
                    r=t.evaluate(arm,z,f'{arm}_g{gen}_i{i}',previous=prev);history.append(r)
                    if r['J']<best['J']:best=r;bestx=z.copy()
                    dump(O/'HISTORY.json',history)
            repeat=t.evaluate(arm,bestx,arm+'_repeat');cut=t.evaluate(arm,bestx,arm+'_P1_cut',cut=True)
            a=np.load(Path(best['folder'])/'BODY.npz');b=np.load(Path(repeat['folder'])/'BODY.npz')
            exact=all(np.array_equal(a[k],b[k]) for k in a.files) and best['J']==repeat['J']
            assert exact,'E1 repeat differs'
            results[arm]=dict(baseline=base,best=best,repeat=repeat,cut=cut,exact_repeat=exact,training_evaluations=17,best_z=bestx)
            dump(O/'SUMMARY.json',results)
        print('FINISHED',flush=True)
    finally:
        after={str(p):sha(p) for p in sources()};dump(O/'INPUT_HASHES_AFTER.json',after);assert before==after
if __name__=='__main__':
    if sys.argv[1]=='prereg':prereg()
    elif sys.argv[1]=='run':run()
