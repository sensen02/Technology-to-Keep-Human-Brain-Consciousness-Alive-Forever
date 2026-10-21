"""Force-scale absolute units, REDONE after the blocking muscle-selection failure.

RESEARCH SCRIPT, NOT A PRODUCT. Read-only. One model load (to verify that the compiled actuator
order/names/gains agree with the canonical XML) plus pure-text XML parsing and archived matrices.
No mj_step, no dynamics, no neural run, no parameter change, no network.

Why this rewrite exists: the previous run selected muscles with the positional slice gain[:-2]
("exclude the last two adhesion pads"). The canonical model has 90 actuators and ALL of them are
muscles; adhesion pads exist only in the scene builder (engine/embodied/scene.py, fg_add_actuator
"adhesion") and cp.load_model() does not load them. The slice therefore deleted two REAL muscles
(RHTibia_flex_93434, RHTibia_extensor_93932) and made the right hind leg look deficient.
"""
from pathlib import Path
import json,sys,time,hashlib,re
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parent
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
GATE=Path('/home/sensen/Desktop/engine修复验收/S2-support-gate-20261014T172624/native_diamond.npz')
CERT=Path('/home/sensen/Desktop/engine修复验收/S2-support-certificate-20261014T175213/certificate.json')
LIT=ROOT/'RESEARCH_TARSUS_FEMUR_ANATOMY.zh-CN.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from engine import canonical_physics as cp
    from engine.embodied.research_report import atomic_report
    # ---------- XML text path (no model needed): actuator elements, classes, gains
    xml_text=cp.CANONICAL_XML.read_text()
    world=ET.fromstring(xml_text)
    actblock=list(world.iter('actuator'))[0]
    els=list(actblock.iter('general'))
    xml_names=[e.get('name') for e in els]
    xml_gains=[float(e.get('gainprm').split()[0]) for e in els]
    xml_classes=[e.get('class') for e in els]
    assert len(els)==90,('XML actuator count != 90',len(els))
    assert set(xml_classes)=={'muscle'},('XML classes not all muscle',set(xml_classes))
    assert not re.search('adhesion',xml_text,re.I),'the canonical XML mentions adhesion'
    # ---------- compiled model path (declared single load, read-only)
    m=cp.load_model()
    m_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)]
    m_gains=[float(m.actuator_gainprm[i,0]) for i in range(m.nu)]
    assert m.nu==90,('compiled actuator count',m.nu)
    assert m_names==xml_names,('compiled actuator order/names differ from the XML',[ (a,b) for a,b in zip(m_names,xml_names) if a!=b][:3])
    assert all(abs(a-b)<1e-12 for a,b in zip(m_gains,xml_gains)),'gainprm[0] differs between XML and compiled model'
    mus_ids=[i for i in range(m.nu) if int(m.actuator_dyntype[i])==int(mujoco.mjtDyn.mjDYN_MUSCLE)]
    assert len(mus_ids)==90,('not all actuators are muscles',len(mus_ids))
    assert not any('adhesion' in (n or '').lower() for n in m_names),'an adhesion actuator exists in the compiled model'
    # ---------- the two muscles the positional slice killed
    killed=dict(names=[m_names[88],m_names[89]],gains=[m_gains[88],m_gains[89]],sum_gain=m_gains[88]+m_gains[89],
        note='these are the last two actuators in BOTH orders and are real muscles; gain[:-2] deleted exactly these')
    assert abs(killed['sum_gain']-1.864675)<1e-9,('killed gain sum',killed['sum_gain'])
    # ---------- weights and muscle budget
    mass_sum=float(m.body_mass.sum());g=float(abs(m.opt.gravity[2]));weight=mass_sum*g
    readings={'gram(primary, INFERRED)':1e-3,'kg(module text)':1.0,'mg(round-39 constant)':1e-6}
    def uN_per(kg):return kg*1e-3/1e-6
    per_leg={}
    for i in mus_ids:per_leg.setdefault(m_names[i][:2],[]).append(m_gains[i])
    leg_peak={k:float(np.sum(v)) for k,v in per_leg.items()}
    total_gain=float(np.sum(m_gains))
    conv={k:dict(uN_per_force_unit=uN_per(v),
            total_weight_uN=weight*uN_per(v),per_leg_share_uN=weight*uN_per(v)/6.0,
            total_muscle_peak_uN=total_gain*uN_per(v),mean_leg_peak_uN=float(np.mean(list(leg_peak.values())))*uN_per(v),
            leg_peak_uN={kk:vv*uN_per(v) for kk,vv in leg_peak.items()}) for k,v in readings.items()}
    gram=conv['gram(primary, INFERRED)']
    ratio_peak_over_load=gram['mean_leg_peak_uN']/gram['per_leg_share_uN']
    # ---------- direction-resolved certificate quantities + decomposition
    with np.load(GATE) as z:
        A=z['A'].copy();b=z['target'].copy();lo=z['lo'].copy();hi=z['hi'].copy()
    y=np.array([x for x in json.loads(CERT.read_text())['results'] if x['floor']=='native'][0]['y_float'])
    R=A[:,:90];t=R.T@y;P=float(b@y)
    contrib=np.maximum(lo*t,hi*t);U=float(contrib.sum())
    pos=t>0;neg=t<0;exact_zero=(t==0);negligible=np.abs(t)<=1e-15
    assert int(pos.sum())+int(neg.sum())+int(exact_zero.sum())==len(t),'projection partition does not cover the muscle columns'
    decomp=dict(n_positive_projection=int(pos.sum()),n_negative_projection=int(neg.sum()),n_exact_zero_projection=int(exact_zero.sum()),
        n_numerically_negligible_abs_t_le_1e_15=int(negligible.sum()),
        partition_check='positive + negative + exact-zero = %d = number of muscle columns'%len(t),
        negligible_warning='n_numerically_negligible overlaps the positive/negative classes (it counts |t|<=1e-15, including the exact zeros); it is NOT a third partition class',
        sum_positive_t=float(t[pos].sum()),sum_negative_t=float(t[neg].sum()),
        U_from_positive=float(contrib[pos].sum()),U_from_negative_floor=float(contrib[neg].sum()),
        note='with hi=1 and lo=0.0001, U_i = t_i for t_i>0 and 0.0001*t_i for t_i<0; the negative side contributes only a tiny floor term')
    per_leg_U={}
    for i in mus_ids:per_leg_U.setdefault(m_names[i][:2],0.0)
    for i in range(90):per_leg_U[m_names[i][:2]]+=float(contrib[i])
    s_gap=dict(P=P,U=U,shortfall=P-U,shortfall_fraction=(P-U)/P,required_scale_P_over_U=(P/U if U>0 else None))
    # ---------- how the corrected budget relates to s (order-of-magnitude only)
    s_glob=1.0450258
    link=dict(
        statement='order-of-magnitude comparison only: NOT a support claim, NOT a biological claim, NOT a calibration',
        per_leg_peak_uN=gram['mean_leg_peak_uN'],per_leg_weight_share_uN=gram['per_leg_share_uN'],ratio_peak_over_share=ratio_peak_over_load,
        peak_needed_if_s_applied=gram['mean_leg_peak_uN']*s_glob,
        share_after_s=gram['per_leg_share_uN']*s_glob,
        interpretation=('the per-leg muscle PEAK budget is about %.2fx the per-leg weight share in raw force terms, while the static relaxation needs only a x%.4f change of the muscle-force-to-weight ratio to close. '
                        'A budget that is already ~%.1fx the load cannot be closed by "more raw force": the deficit is therefore NOT a force-scale/strength shortage; it is how the available force is ROUTED - moment arms (direction of R^T y), joint limits and the contact cone.'%(
                        ratio_peak_over_load,s_glob,ratio_peak_over_load)))
    r=dict(status='PASS_WITH_CORRECTED_MUSCLE_BUDGET_AND_TWO_VERIFIED_COUNTING_PATHS',
        scope='FORCE_SCALE_CALIBRATION_QUESTION_ONLY',research_script_not_product=True,
        correction=dict(superseded_values=dict(total_muscle_peak_uN=62.4039,mean_leg_peak_uN=10.4006,RH_leg_peak_uN=8.847,
                ratio_per_leg_peak_over_takeoff=0.10400642,approx_factor_low='9.6x'),
            superseded_reason='positional slice gain[:-2] deleted two real muscles; the model has 90 muscles and no adhesion actuators',
            not_superseded=['unit conversion','direction-resolved certificate quantities','reproduction/mass-scale claim','literature anchors'],
            superseded_by_this_directory=True),
        muscle_set=dict(rule='class/name based (actuator_dyntype == mjDYN_MUSCLE); NO positional slices',
            n_muscles=len(mus_ids),n_legs=len(per_leg),muscles_per_leg={k:len(v) for k,v in per_leg.items()},
            compiled_names=m_names,xml_names_match_compiled=True,gainprm0_match=True,
            asserts=dict(actuator_count_equals_90=True,all_actuators_are_muscle=True,adhesion_actuators_equals_0=True,
                         xml_actuator_elements_equals_90=True,xml_classes_all_muscle=True,adhesion_tokens_in_xml=0),
            killed_by_old_slice=killed),
        unit_readings=conv,muscle_budget_gram=gram,ratio_peak_over_leg_load=ratio_peak_over_load,
        direction_resolved=dict(P=P,U=U,shortfall=P-U,shortfall_fraction=(P-U)/P,required_scale_P_over_U=s_gap['required_scale_P_over_U'],
            per_leg_U_along_y=per_leg_U,decomposition=decomp,
            carried_over_global_window=s_glob,carried_over_note='the x1.0450258 global LP window is carried over from an earlier round; the value computed HERE along the certified direction is x%.10f'%s_gap['required_scale_P_over_U']),
        s_versus_muscle_budget=link,
        anchor_policy=dict(
            unverified_quote='the ~100 uN takeoff peak appears in RESEARCH_TARSUS_FEMUR_ANATOMY.zh-CN.md only as an assertion citing Azevedo 2020 (lines 723/798); the nearest verbatim quote in that file is "up to 47 uN load". Marked UNVERIFIED_QUOTE and excluded from primary conclusions.',
            derived_value='"six legs ~ 1 mg body weight" is DERIVED from 10 uN / 9.81 m/s^2 and is marked DERIVED, not recorded text.'),
        environment=dict(python=sys.version.split()[0],executable=sys.executable,mujoco=mujoco.__version__,numpy=np.__version__),
        boundaries=dict(not_done='no mj_step, no dynamics, no neural run, no parameter change, no network',
            claim_limit='force-scale calibration question only; not a biological calibration, not a support result, not a standing result'))
    r['input_hashes']={str(p):sha(p) for p in (Path(__file__),GATE,CERT,LIT,cp.CANONICAL_XML)}
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=['total\nweight','per-leg\nload','per-leg peak\n(90 muscles)','takeoff peak\n(quote unverified)']
    axes[0].bar(labels,[gram['total_weight_uN'],gram['per_leg_share_uN'],gram['mean_leg_peak_uN'],np.nan],color='tab:blue',label='model (inferred gram reading)')
    axes[0].bar(labels,[10.0,1.7,np.nan,100.0],color='tab:orange',label='in-repo anchors (10 & 1.7 uN verbatim; 100 uN quote-unverified)')
    axes[0].set_ylabel('force (uN)');axes[0].legend(fontsize=6);axes[0].set_title('absolute units after the muscle-budget correction')
    for i,v in enumerate((gram['total_weight_uN'],gram['per_leg_share_uN'],gram['mean_leg_peak_uN'])):
        axes[0].text(i,v,'  %.4f'%v,ha='center',va='bottom',fontsize=7)
    axes[1].bar(['P (required)','U (achievable,\nall 90 muscles at bounds)'],[P,U],color=['tab:red','tab:blue'])
    axes[1].set_ylabel('model force units along the certified direction')
    axes[1].set_title('shortfall %.2f%% (needs x%.4f); budget/load = %.2fx'%(100*(P-U)/P,P/U,ratio_peak_over_load))
    fig.suptitle('force-scale absolute units, corrected (read-only) - NOT a calibration, NOT standing',fontsize=9)
    fig.tight_layout();fig.savefig(out/'force-scale-units.png');plt.close(fig)
    atomic_report(out/'report.json',r)
    print(out);print(r['status'])
    print('total peak uN',gram['total_muscle_peak_uN'],'| per-leg',gram['mean_leg_peak_uN'],'| leg peaks',json.dumps({k:round(v,6) for k,v in gram['leg_peak_uN'].items()}))
    print('ratio peak/load',round(ratio_peak_over_load,4),'| P',P,'U',U,'needed x',s_gap['required_scale_P_over_U'])
    print('decomposition',json.dumps(decomp))
    print('killed',json.dumps(killed))
    return 0
if __name__=='__main__':
    out=Path('/home/sensen/Desktop/engine修复验收')/('force-scale-absolute-units-fix-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    raise SystemExit(main())
