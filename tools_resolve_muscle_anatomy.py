#!/usr/bin/env python3
"""Source/route/physical support resolved motor candidates, NOT validated anatomy.

Read-only source assets; outputs only outputs/muscle_anatomy_resolution. No
controllers, simulation integration, muscle edits or neural stimulation. MuJoCo
moment derivatives are model geometry diagnostics, not physiological forces;
engine/embodied/force_channel.py limitations remain applicable.
"""
from pathlib import Path
import argparse,collections,csv,gzip,hashlib,json,re,sys,unittest
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/muscle_anatomy_resolution'
LEGS=('LF','RF','LM','RM','LH','RH')

def normalized(name):return '_'.join(re.findall(r'[a-z0-9]+',name.lower()))
def decode_asset(name):
    leg=name[:2]
    if leg not in LEGS:raise ValueError('unknown leg')
    rest=name[2:];compartment='Tibia'
    if rest.startswith(('C_','F_')):compartment=rest[0];rest=rest[2:]
    suffix=re.search(r'_(\d+)$',rest)
    if suffix:rest=rest[:suffix.start()]
    return leg,compartment,normalized(rest),suffix.group(1) if suffix else None

def allowed_nerves(leg):
    side='left' if leg[0]=='L' else 'right'
    if leg[1]=='F':return [side+'_'+s for s in ('prothoracic_leg_nerve','prothoracic_accessory_nerve','dorsal_prothoracic_nerve','ventral_prothoracic_nerve')]
    return [side+('_mesothoracic_leg_nerve' if leg[1]=='M' else '_metathoracic_leg_nerve')]

def name_rule(key):
    # Only enumerated model-specific spelling/abbreviation rules, no token bag.
    if key.endswith(('_a','_b')):return None,'REJECT_UNRESOLVED_SUBDIVISION'
    if key=='pleural_promotor':return None,'REJECT_SHARED_MILLER28_30_POOL'
    aliases={'accesory_trochanter_flexor':'accessory_trochanter_flexor',
             'tibia_flex':'tibia_flexor',
             'pleural_remotor_and_abductor':'pleural_remotor_abductor'}
    return aliases.get(key,key),'EXPLICIT_MODEL_LEXICAL_EQUIVALENCE' if key in aliases else 'EXACT_PRIMARY_NAME'

def candidates(name,rows):
    leg,comp,key,suffix=decode_asset(name);target,rule=name_rule(key)
    if target is None:return [],[],rule
    selected=[];conflicts=[]
    for r in rows:
        if r['Nerve'] not in allowed_nerves(leg):continue
        primary=normalized(r['Primary Cell Type'])
        aliases=[normalized(t) for t in r['Alternative Cell Type(s)'].split(',') if t]
        if primary==target:selected.append(r)
        elif target in aliases or key in aliases:conflicts.append(r)
    return selected,conflicts,rule

def parse_routes(path):
    root=ET.parse(path).getroot();sites={}
    for b in root.iter('body'):
        for s in b.findall('site'):sites[s.get('name')]={'body':b.get('name'),'pos':s.get('pos'),'name':s.get('name')}
    tendons={t.get('name'):[sites[x.get('site')] for x in t if x.tag=='site'] for t in root.find('tendon')}
    acts=[a for a in root.find('actuator') if a.get('class')=='muscle']
    return acts,tendons

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Tests(unittest.TestCase):
    def test_no_subdivision_merge(self):
        for s in ('trochanter_flexor_a','trochanter_flexor_b','tergopleural_promotor_a','sterno_tergo_trochanter_extensor_b'):
            self.assertIsNone(name_rule(s)[0])
    def test_label_rules(self):
        self.assertEqual(name_rule('accesory_trochanter_flexor')[0],'accessory_trochanter_flexor')
        self.assertEqual(name_rule('tibia_flex')[0],'tibia_flexor')
        self.assertEqual(name_rule('accessory_tibia_flexor')[0],'accessory_tibia_flexor')
    def test_suffix_compartment(self):
        self.assertEqual(decode_asset('LFTibia_flex_93434'),('LF','Tibia','tibia_flex','93434'))
        self.assertEqual(decode_asset('LFF_trochanter_flexor_a')[2],'trochanter_flexor_a')
    def test_nerve_not_soma_side(self):
        r={'Nerve':'left_ventral_prothoracic_nerve','Primary Cell Type':'sternal_anterior_rotator','Alternative Cell Type(s)':'','Soma side':'right'}
        self.assertEqual(len(candidates('LFC_sternal_anterior_rotator',[r])[0]),1)
        self.assertEqual(len(candidates('RFC_sternal_anterior_rotator',[r])[0]),0)
    def test_conflicting_alias(self):
        r={'Nerve':'left_mesothoracic_leg_nerve','Primary Cell Type':'femur_reductor','Alternative Cell Type(s)':'pleural_remotor_abductor'}
        good,bad,_=candidates('LMC_pleural_remotor_and_abductor',[r]);self.assertFalse(good);self.assertEqual(len(bad),1)
    def test_shared_pool_not_copied(self):self.assertIsNone(name_rule('pleural_promotor')[0])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--selftest',action='store_true');args=ap.parse_args()
    if args.selftest:return 0 if unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    import mujoco as mj,flygym
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    from engine.embodied.adapters import build_neural_tier,TierConfig
    canonical=ROOT/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
    upstream=Path(flygym.__file__).parent/'assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml'
    csvpath=ROOT/'data/flywire/banc_neurons.csv.gz'
    with gzip.open(csvpath,'rt') as f:rows=[r for r in csv.DictReader(f) if r['Class']=='leg_motor_neuron' and r['Super Class']=='motor']
    built=build_neural_tier(TierConfig());loc={str(int(r)):i for i,r in enumerate(built['tier']['root_ids'])}
    acts,routes=parse_routes(canonical);upacts,uproutes=parse_routes(upstream)
    upnames={a.get('name'):a for a in upacts}
    model=_load_mjcf(str(canonical)).compile();d=mj.MjData(model);mj.mj_resetDataKeyframe(model,d,0)
    q0=d.qpos.copy();hinges=np.flatnonzero(model.jnt_type==mj.mjtJoint.mjJNT_HINGE)
    mats=[]
    for delta in (0.,-.03,.03):
        d.qpos[:]=q0
        for j in hinges:d.qpos[model.jnt_qposadr[j]]+=delta
        mj.mj_forward(model,d);a=np.zeros((model.nu,model.nv))
        mj.mju_sparse2dense(a,d.actuator_moment,d.moment_rownnz,d.moment_rowadr,d.moment_colind);mats.append(a)
    d.qpos[:]=q0;mj.mj_forward(model,d)
    # Numerically verify ALL tendon derivative columns at initial pose (no mj_step).
    finite=np.zeros((model.nu,len(hinges)));eps=1e-6
    for k,j in enumerate(hinges):
        qa=int(model.jnt_qposadr[j]);d.qpos[:]=q0;d.qpos[qa]+=eps;mj.mj_forward(model,d);plus=d.actuator_length.copy()
        d.qpos[:]=q0;d.qpos[qa]-=eps;mj.mj_forward(model,d);minus=d.actuator_length.copy()
        finite[:,k]=(plus-minus)/(2*eps)
    err=float(np.max(abs(finite-mats[0][:,model.jnt_dofadr[hinges]])))
    assert err<1e-6,err
    old=json.loads((ROOT/'outputs/autonomous_mapping_round2/mapping_manifest.json').read_text())
    oldexact={r['actuator'] for r in old['actuators'] if r['resolution']['status']=='EXACT_ANNOTATION_CANDIDATE'}
    records=[]
    for el in acts:
        name=el.get('name');aid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,name)
        leg,comp,key,suffix=decode_asset(name);same='LF'+name[2:]
        assert same in upnames
        route=routes[el.get('tendon')];ur=uproutes[upnames[same].get('tendon')]
        same_route=[s['name'][2:] for s in route]==[s['name'][2:] for s in ur] if leg=='LF' else None
        supported=[];bad=[]
        for j in hinges:
            dof=int(model.jnt_dofadr[j]);values=[float(a[aid,dof]) for a in mats]
            if max(abs(v) for v in values)<1e-6:continue
            jn=mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,int(j))
            expected_segment='Coxa' if comp=='C' else 'Tibia' if comp=='Tibia' else 'Trochanter'  # mid/hind joints retain Trochanter names on Femur bodies
            okay=jn.startswith('joint_'+leg+expected_segment+'_')
            if not okay:bad.append(jn)
            supported.append({'joint':jn,'dof':dof,'d_length_d_q_mm_per_rad_key_minus_plus':values,'expected_compartment':okay})
        selected,conflicts,rule=candidates(name,rows)
        # Out-of-main-leg nerve anatomical candidates shown separately, not guessed into wiring.
        possible_hind=[]
        target,_=name_rule(key)
        if leg[1]=='H' and target:
            nerve=('left' if leg[0]=='L' else 'right')+'_first_abdominal_nerve'
            possible_hind=[r for r in rows if r['Nerve']==nerve and normalized(r['Primary Cell Type'])==target]
        def pack(rr):return [{'root_id':r['Root ID'],'primary':r['Primary Cell Type'],'alternative':r['Alternative Cell Type(s)'],'nerve':r['Nerve'],'soma_side':r['Soma side'],'local_index':loc.get(r['Root ID'])} for r in rr]
        status='CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE' if selected and supported and not bad else rule if target is None else 'UNRESOLVED_NO_SAFE_POOL' if not selected else 'REJECT_PHYSICAL_COMPARTMENT'
        records.append({'actuator':name,'status':status,'compartment':comp,'label':key,'asset_suffix':suffix,
            'normalization_rule':rule,'target_primary':target,'candidates':pack(selected),'conflicting_alternative_only_rows':pack(conflicts),
            'hind_MetaAN_candidates_REQUIRE_NERVE_ASSIGNMENT_EVIDENCE':pack(possible_hind),
            'tendon':el.get('tendon'),'canonical_route':route,'upstream_actuator':same,'upstream_route':ur,
            'canonical_route_names_equal_upstream_LF':same_route,'moment_arm_support':supported,'unexpected_joints':bad,
            'was_round2_exact_candidate':name in oldexact,'individual_anatomy_validated':False,'runtime_authorization':False,
            'evidence':['original installed FlyGym asset actuator/tendon/site names','BANC CSV exact motor primary+explicit alternative names and nerve','canonical compiled tendon derivative; verified finite differences'],
            'uncertainty':['geometric derivative is pose dependent, not physiological torque sign','five legs mirrored with trochanter-to-femur remap; route evidence not anatomical registration','spike-force gains and motor unit recruitment uncalibrated']})
    sources=[Path(__file__),canonical,upstream,csvpath,ROOT/'tools_mirror_muscles.py',ROOT/'tools_stand_and_ground.py',ROOT/'tools_repair_moment_arms.py',ROOT/'outputs/autonomous_mapping_round2/mapping_manifest.json']
    counter=collections.Counter(r['status'] for r in records)
    report={'status':'ANATOMICAL_CANDIDATE_RESOLUTION_NOT_VALIDATED_INDIVIDUAL_ANATOMY','source_sha256':{str(p.resolve()):sha(p) for p in sources},
        'tier_fingerprint':built['fingerprint'],'moment_derivative_fd_max_abs_error':err,'poses':'keyframe and all hinge offsets +/-0.03rad; no dynamics',
        'rules':{'a_b':'never merged','foreleg_nerves':'explicit prothoracic LN/AN/dorsal/ventral names, not soma laterality','hind_first_abdominal':'reported but excluded pending independent MetaAN nerve assignment','aliases':'only accesory->accessory, tibia_flex->tibia_flexor, remotor_and_abductor->remotor_abductor; supported by route/CSV, no generic token deletion','shared_pool':'Miller28/30 alias cannot partition model 28a/28b/30, rejected'},
        'summary':dict(counter),'previous22_physical_compartment_failures':[r['actuator'] for r in records if r['was_round2_exact_candidate'] and r['unexpected_joints']],
        'candidate_count_by_leg':{l:sum(r['actuator'].startswith(l) and r['status']=='CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE' for r in records) for l in LEGS},
        'new_candidate_actuators':[r['actuator'] for r in records if r['status']=='CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE' and not r['was_round2_exact_candidate']],
        'unmatched_candidate_root_ids':sorted({x['root_id'] for r in records if r['status']=='CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE' for x in r['candidates'] if x['local_index'] is None}),
        'body_joint_naming_caution':'Middle/hind Trochanter-named DOFs are attached to Femur bodies; do not reject by body-name equals joint-name assumption.',
        'actuators':records}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'resolution.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    axs[0].bar(LEGS,list(report['candidate_count_by_leg'].values()));axs[0].axhline(15,ls='--',color='gray');axs[0].set_title('Label + nerve + model compartment candidates');axs[0].set_ylabel('Actuators per leg (15 total)')
    im=axs[1].imshow(np.abs(mats[0][:,model.jnt_dofadr[hinges]]),aspect='auto');axs[1].set_title('Canonical |d tendon length / d joint angle|');axs[1].set_xlabel('42 leg hinge DOFs');axs[1].set_ylabel('90 muscle actuators');fig.colorbar(im,ax=axs[1],label='model mm/rad')
    fig.suptitle('No individual anatomical validation; no a/b merging or controller');fig.tight_layout();fig.savefig(OUT/'resolution.png',dpi=140);plt.close(fig)
    text='# 肌肉命名、神经归属与物理关节支持解析\n\n手工实现的候选映射审计，不是个体解剖验证，不修改规范模型。\n\n'
    text+=f'分类：{dict(counter)}。各腿候选：{report["candidate_count_by_leg"]}。有限差分最大误差{err:.3g}。\n\n'
    text+='## 实质发现\n\n旧22/90仅查腿主神经，漏掉前腿背/腹/辅助神经的明确同名MN。这里依据神经名拓展前腿候选，不用胞体侧别决定效应腿。原始资产含Miller28a/28b/29/30/31/32/33站点标签，可追溯命名。\n\n只有三项明确词法等价：accesory拼写、Tibia_flex资产缩写、pleural_remotor_and_abductor与CSV主类型/别名。a/b保持隔离；Miller28/30共用注释不能分配给28a/28b/30；sterno-tergo合并名称不能武断拆成sternotrochanter或tergotrochanter。后腿MetaAN候选单列而未授权。\n\n'
    text+=f'旧22项物理分区失败：{report["previous22_physical_compartment_failures"]}。这不证明旧22项解剖正确，只说明三个位姿探针未发现作用于错误关节分区。所有90条肌腱原始与规范站点、局部位置、承载身体及数值关节导数逐项记录于JSON。收缩力通常为负，因此导数正负不能直接称屈曲/伸展方向。\n\n'
    text+='## 可执行下一步\n\n可基于JSON候选池作显式假设实验，但runtime_authorization仍false，个体解剖未验证。先解决共享Miller28/30与a/b分区，核对后腿MetaAN来源；不要从力臂给MN反推名字或为了覆盖率合并不同池。使用原始与规范路由差异检查镜像误差。\n'
    (OUT/'REPORT.zh-CN.md').write_text(text)
    assert sha(canonical)==report['source_sha256'][str(canonical)]
    print(json.dumps({'summary':dict(counter),'per_leg':report['candidate_count_by_leg'],'fd_error':err,'previous22_compartment_failures':report['previous22_physical_compartment_failures']},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
