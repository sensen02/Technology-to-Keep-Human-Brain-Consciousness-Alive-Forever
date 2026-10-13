#!/usr/bin/env python3
"""Fail-closed annotation resolver. Exact nomenclature is NOT validated anatomy.

Only lexical punctuation/case normalization, actuator leg/compartment prefixes and
numeric model asset suffix removal are automatic. Subdivision a/b and anatomical
words are never dropped. Declared spelling aliases are displayed, never accepted.
Does not modify model, old wiring, neural configuration or physical parameters.
"""
from pathlib import Path
import argparse,collections,csv,gzip,hashlib,json,re,sys,unittest
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/autonomous_mapping_round2'
NERVES={l:f'{s}_{t}_leg_nerve' for l,s,t in [('LF','left','prothoracic'),('RF','right','prothoracic'),('LM','left','mesothoracic'),('RM','right','mesothoracic'),('LH','left','metathoracic'),('RH','right','metathoracic')]}
# Lexical suggestions, not cited anatomical correspondence. Never used by resolve.
ALIASES={'accesory':'accessory','flex':'flexor'}


def normalize(s):
    return '_'.join(re.findall(r'[a-z0-9]+',s.lower()))


def model_name(name):
    leg=name[:2]
    if leg not in NERVES:raise ValueError('unknown leg prefix '+name)
    rest=name[2:]
    # C_/F_ are explicit compartment labels in this XML, NOT anatomical synonyms.
    if rest.startswith(('C_','F_')):rest=rest[2:]
    rest=re.sub(r'_\d+$','',rest)
    return leg,normalize(rest)


def resolve(key,candidates):
    """Return one exact annotation label or reject; multiple neurons per type allowed."""
    labels=sorted({s for s in candidates if normalize(s)==key})
    if len(labels)==1:return {'status':'EXACT_ANNOTATION_CANDIDATE','label':labels[0]}
    return {'status':'AMBIGUOUS' if labels else 'UNMATCHED','labels':labels}


def require_unique(key,candidates):
    r=resolve(key,candidates)
    if r['status']!='EXACT_ANNOTATION_CANDIDATE':raise ValueError(r['status'])
    return r['label']


class Tests(unittest.TestCase):
    def test_exact(self):self.assertEqual(require_unique('tibia_extensor',['tibia_extensor']),'tibia_extensor')
    def test_population(self):self.assertEqual(require_unique('x',['x','x']),'x')
    def test_ambiguous(self):
        with self.assertRaises(ValueError):require_unique('a_b',['a-b','a_b'])
    def test_no_fuzzy(self):
        for key in ('trochanter_flexor_a','accesory_trochanter_flexor','tibia_flex'):
            with self.assertRaises(ValueError):require_unique(key,['trochanter_flexor','accessory_trochanter_flexor','tibia_flexor'])
    def test_structural(self):self.assertEqual(model_name('LFTibia_extensor_93932'),('LF','tibia_extensor'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--selftest',action='store_true');args=ap.parse_args()
    if args.selftest:
        return 0 if unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    OUT.mkdir(parents=True,exist_ok=True)
    xml=ROOT/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
    npz=ROOT/'data/flywire/banc_connectome.npz';csvpath=ROOT/'data/flywire/banc_neurons.csv.gz'
    from engine.embodied.adapters import build_neural_tier,TierConfig
    cfg=TierConfig();built=build_neural_tier(cfg)
    z=np.load(npz,allow_pickle=False)
    prim=z['ann_primary_type'];nerve=z['ann_nerve'];cls=z['ann_class'];sup=z['ann_super_class'];roots=z['root_ids']
    sampled={str(x):i for i,x in enumerate(built['tier']['root_ids'])}
    # CSV soma-side independent corroboration, no FAFB cross-specimen joins.
    with gzip.open(csvpath,'rt') as f:
        records={r['Root ID']:r for r in csv.DictReader(f)}
    def neurons(indices):
        return [{'root_id':str(roots[i]),'primary_type':str(prim[i]),'class':str(cls[i]),'super_class':str(sup[i]),
                 'nerve':str(nerve[i]),'soma_side':records.get(str(roots[i]),{}).get('Soma side',''),
                 'sampled_local_index':sampled.get(str(roots[i]))} for i in indices]
    tree=ET.parse(xml);acts=[a for a in tree.getroot().find('actuator') if a.get('class')=='muscle']
    assert len(acts)==90,'must audit all 90 muscles'
    rows=[]
    for a in acts:
        name=a.get('name');leg,key=model_name(name)
        mask=(nerve==NERVES[leg])&(sup=='motor')&(cls=='leg_motor_neuron')
        labels=sorted(set(map(str,prim[mask])))
        r=resolve(key,labels)
        indices=np.flatnonzero(mask&(prim==r['label'])) if 'label' in r else np.array([],int)
        # Only declared lexical suggestions; cannot upgrade status.
        alias_key='_'.join(ALIASES.get(t,t) for t in key.split('_'))
        suggestions=[t for t in labels if normalize(t)==alias_key] if alias_key!=key else []
        nd=neurons(indices)
        rows.append({'actuator':name,'leg':leg,'model_normalized_name':key,'resolution':r,
                     'candidate_neurons':nd,'missing_sampled_root_ids':[n['root_id'] for n in nd if n['sampled_local_index'] is None],
                     'declared_alias_suggestions':suggestions,'alias_evidence':'DECLARED_LEXICAL_NOT_ANATOMICAL; not used for acceptance',
                     'annotation_level':r['status'],'biological_anatomy_validated':False,
                     'strict_runtime_wiring_authorized':False,'model_anatomy':'derived six-leg muscle asset; not specimen registration'})
    # A unique label can still be shared by distinct model compartments: block that too.
    reverse=collections.defaultdict(list)
    for r in rows:
        if 'label' in r['resolution']:reverse[(r['leg'],r['resolution']['label'])].append(r['actuator'])
    for r in rows:
        if 'label' in r['resolution']:
            shared=reverse[(r['leg'],r['resolution']['label'])]
            r['model_actuators_sharing_annotation']=shared
            if len(shared)>1:r['annotation_level']='AMBIGUOUS_MODEL_SUBDIVISION'
    sensory=[]
    # Do not infer modalities from abbreviations. Use explicit BANC class annotation.
    types=('chordotonal_organ_neuron','campaniform_sensillum_neuron','hair_plate_neuron','bristle_neuron','multidendritic_neuron','taste_bristle_tactile_neuron')
    for leg,nrv in NERVES.items():
        for subtype in types:
            ids=np.flatnonzero((nerve==nrv)&(cls==subtype))
            if not len(ids):continue
            names=sorted(set(map(str,prim[ids])))
            joints=[j.get('name') for j in tree.getroot().iter('joint') if (j.get('name') or '').startswith('joint_'+leg)]
            sensory.append({'leg':leg,'nerve':nrv,'explicit_annotation_class':subtype,'primary_types':names,
                'neurons':neurons(ids),'count':len(ids),'sampled_count':sum(str(roots[i]) in sampled for i in ids),
                'same_leg_model_joint_candidates':joints,'resolved_model_joint':None,'strict_runtime_wiring_authorized':False,
                'evidence_level':'EXACT_CLASS_AND_NERVE_ONLY',
                'missing_evidence':['receptor attachment/receptive field','specific joint or cuticle site','direction selectivity','transduction gain/kinetics'],
                'warning':'same-leg joint list is search space, NOT mapping; name codes are not decoded as anatomy'})
    counts=dict(collections.Counter(r['annotation_level'] for r in rows))
    sources=[Path(__file__),xml,npz,csvpath,ROOT/'muscle_wiring.py',ROOT/'tools_mn_muscle_map.py',ROOT/'engine/embodied/adapters.py']
    report={'status':'AUDIT_COMPLETED_AUTONOMY_NOT_ACHIEVED','all_actuators_count':len(rows),'counts':counts,
        'normalization_policy':__doc__,'declared_aliases_not_accepted':ALIASES,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'tier':{'max_neurons':cfg.max_neurons,'fingerprint':built['fingerprint'],'sampled_neurons':built['n']},
        'actuators':rows,'sensory_groups':sensory,
        'leg_motor_annotation_inventory':{leg:neurons(np.flatnonzero((nerve==nrv)&(sup=='motor')&(cls=='leg_motor_neuron'))) for leg,nrv in NERVES.items()},
        'sensory_class_counts':dict(collections.Counter({t:sum(s['count'] for s in sensory if s['explicit_annotation_class']==t) for t in types})),
        'biologically_validated_actuator_links':0,'validated_joint_specific_sensory_links':0,
        'old_jaccard_policy':'muscle_wiring.actuator_to_mn_type accepts best token Jaccard >= .40; this resolver does NOT call it or accept its result',
        'scope':'Exact annotations improve auditable candidate selection; not proof of biological innervation or autonomous behavior.'}
    (OUT/'mapping_manifest.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    labels=list(counts);axs[0].bar(range(len(labels)),list(counts.values()))
    axs[0].set_xticks(range(len(labels)),[x.replace('_','\n') for x in labels],fontsize=8)
    axs[0].set_ylabel('Actuators (all 90 audited)')
    subcounts=collections.Counter()
    for s in sensory:subcounts[s['explicit_annotation_class']]+=s['count']
    axs[1].bar(range(len(subcounts)),list(subcounts.values()))
    axs[1].set_xticks(range(len(subcounts)),[x.replace('_','\n') for x in subcounts],fontsize=8)
    axs[1].set_ylabel('Annotated sensory neurons in six leg nerves')
    fig.suptitle('Exact annotation audit — NOT validated anatomy\n0 validated actuator links; 0 validated joint-specific sensory links')
    fig.tight_layout();fig.savefig(OUT/'mapping_coverage.png',dpi=140)
    (OUT/'REPORT.zh-CN.md').write_text('# 自主翻译映射审计 R2\n\n全部90个执行器已审计。名称完全匹配只是候选注释，不是解剖验证。\n\n分类：'+json.dumps(counts,ensure_ascii=False)+'\n\n不再接受最高Jaccard分数作为确认连接。保留a/b等亚分区，不把不同肌肉合并。拼写别名仅列为声明建议，不自动放行。每个候选记录motor类别、腿神经、root ID与缺失采样ID。严格运行授权全部为false。\n\n感觉细胞按明确class/nerve分组，chordotonal/campaniform等注释可确认类别，不能确定模型中的具体关节或受力点；未猜测类型代码。完整信息见同目录mapping_manifest.json。\n\n目标未完成；本工具不修改任何旧模块或物理模型。\n')
    print(json.dumps({'actuators':len(rows),'counts':counts,'sensory_neurons':sum(s['count'] for s in sensory),'validated_actuator_links':0,'validated_sensory_links':0,'out':str(OUT)}))
    return 0

if __name__=='__main__':raise SystemExit(main())
