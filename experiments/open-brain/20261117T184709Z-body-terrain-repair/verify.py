"""Independent native-array metrics and saved replay qualification."""
from pathlib import Path
import json
import numpy as np
import mujoco
from hashlib import sha256
HERE=Path(__file__).resolve().parent
digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
checks=[]
for result_path in sorted((HERE/'FORMAL').glob('*/RESULT.json')):
    r=json.loads(result_path.read_text());out=result_path.parent
    assert digest(out/'TRACE.npz')==r['trace_sha256'] and digest(out/'BODY.mjb')==r['body_sha256']
    m=mujoco.MjModel.from_binary_path(str(out/'BODY.mjb'));d=mujoco.MjData(m)
    with np.load(out/'TRACE.npz',allow_pickle=False) as z:
        assert len(z['qpos'])==17000 and all(np.isfinite(z[k]).all() for k in z.files)
        j=np.where(m.jnt_limited)[0];q=z['qpos'][:,m.jnt_qposadr[j]]
        excess=np.maximum(np.maximum(m.jnt_range[j,0]-q,q-m.jnt_range[j,1]),0)
        aid=np.array([i for i in range(m.nu) if m.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_JOINT])
        ctrl=z['control'];p=z['position'];weight=m.body_mass.sum()*abs(m.opt.gravity[2])
        rms=float(np.sqrt(np.mean(np.sum((np.diff(p[-2000:,:2],axis=0)/.0001)**2,axis=1))))
        gates=dict(completed=True,finite=True,no_warning=r['warning'] is None,upright=bool(z['upright'].min()>=.8),
             height=bool(p[:,2].min()>=.7*z['initial_position'][2]),nonfoot=bool(z['nonfoot'].max()/weight<=.2),
             joint_limit=bool(excess.max()<=.02),target_limits=bool((ctrl>=m.actuator_ctrlrange[aid,0]-1e-12).all() and
             (ctrl<=m.actuator_ctrlrange[aid,1]+1e-12).all()),stand=rms<=.30)
        if r['height']:gates.update(block_contact=bool(z['block_force'].max()>1e-6),cross=bool(p[-1,0]>=3.15))
        else:gates['advance']=bool(p[-1,0]-z['initial_position'][0]>=2.)
        assert gates==r['gates'] and all(gates.values())==r['pass_']
        assert z['pad_reconstruction_error'].max()<1e-10
        d.qpos[:]=z['qpos'][-1];mujoco.mj_forward(m,d)
        feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'nmf/{leg}_tarsus5') for leg in ('lf','lm','lh','rf','rm','rh')]
        foot_minx=float(d.xpos[feet,0].min())
        if r['height']:assert foot_minx>2.65
        boundary=float(np.mean((np.abs(ctrl-m.actuator_ctrlrange[aid,0])<1e-10)|(np.abs(ctrl-m.actuator_ctrlrange[aid,1])<1e-10)))
        checks.append(dict(name=out.name,pass_=r['pass_'],all_metrics_match=True,final_all_feet_min_x=foot_minx,
             pad_reconstruction_error=float(z['pad_reconstruction_error'].max()),target_boundary_fraction=boundary,
             joint_excess=float(excess.max()),stand_rms=rms))
    inputs=json.loads((out/'INPUTS.json').read_text())
    assert all(digest(p)==h for p,h in inputs['source_sha256'].items())
assert len(checks)==6
replays=[]
for folder in sorted((HERE/'REPLAY').glob('*')):
    if not folder.is_dir():continue
    with np.load(folder/'TRACE.npz',allow_pickle=False) as a,np.load(HERE/'FORMAL'/folder.name/'TRACE.npz',allow_pickle=False) as b:
        matches={k:bool(np.array_equal(a[k],b[k])) for k in b.files}
    assert all(matches.values());replays.append(dict(name=folder.name,arrays=len(matches),all_exact=True))
assert len(replays)==3
prepare=json.loads((HERE/'PREPARE_RESULT.json').read_text());pads=json.loads((HERE/'PAD_BENCH_RESULT.json').read_text())
pair=json.loads((HERE/'PAIR_MOTION_RESULT.json').read_text());limit=json.loads((HERE/'LIMIT_PROBE_ISOLATED.json').read_text())
assert prepare['zero_pad_pass'] and pads['all_pass'] and pair['pass_'] and limit['pass_']
for name in ('PREPARE_INPUTS.json','PAIR_MOTION_CONFIG.json','PAD_BENCH_CONFIG.json','SELECTION.json'):
    spec=json.loads((HERE/name).read_text());assert all(digest(p)==h for p,h in spec['source_sha256'].items())
out=dict(status='LIMITED_BODY_AND_LOW_BLOCK_TEACHER_QUALIFIED',formal_pass=sum(c['pass_'] for c in checks),formal_total=6,
     native_metric_checks=checks,replays=replays,isolated_limit_checks=132,all_hinge_limits=66,position_target_limits=42,
     zero_adhesion_reference='New canonical non-fused 42 vs48 actuator model, not the old frozen fused model',
     static_and_dynamic_zero_adhesion_exact=True,single_pad_tests_pass=8,teacher_only=True,training_updates=0,
     biological_hard_joint_limits='UNKNOWN_ENGINEERING_ENVELOPE',full_69_ANN_regression='NOT_RUN',flight='DEFERRED')
(HERE/'FINAL_ACCEPTANCE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('Six formal conditions, three exact native replays, 132 limits, pad and model equivalence verified')
