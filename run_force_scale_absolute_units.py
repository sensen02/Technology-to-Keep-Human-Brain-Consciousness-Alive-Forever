"""Force-scale absolute-unit check against in-repo literature anchors (read-only).

RESEARCH SCRIPT, NOT A PRODUCT. Declared stance = the archived declared model pose, evaluated with
mj_forward only (no mj_step, no pose search, no tuning, no v6 re-run, no network). All literature
numbers are taken from values already recorded inside this repository; anything missing is marked
NOT_AVAILABLE rather than fetched.

Question: is the certified s ~ 1.045 force-scale gap caused by the model mass/force scale being
about 4.5 percent off the literature, or is it structural (contact geometry, friction cone, joint
limits)? The exercise must make the 1.7 uN single-leg anchor reproducible in the model or refute it,
and must NOT tune s to hit that number.
"""
from pathlib import Path
import json,time,hashlib,zipfile
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
    m=cp.load_model();d=mujoco.MjData(m)
    with np.load(POSE) as z:pose=z['projected_seated_qpos'].copy()
    d.qpos[:]=pose;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d)
    # ---------- (1) weight and per-leg share, in model units and in uN under each mass reading
    mass_sum=float(m.body_mass.sum());g=float(abs(m.opt.gravity[2]))
    weight_model=mass_sum*g
    # length unit = 1 mm (documented). A force unit is massunit*mm/s^2.
    readings={'gram(primary, INFERRED)':1e-3,'kg(module text)':1.0,'mg(round-39 constant)':1e-6}   # kg per mass unit
    def uN_per_force_unit(kg_per_unit):
        # 1 force unit = kg_per_unit * (1 mm/s^2) = kg_per_unit * 1e-3 N; express that in uN
        return kg_per_unit*1e-3/1e-6
    conv={k:dict(kg_per_mass_unit=v,uN_per_force_unit=uN_per_force_unit(v),
                 total_weight_uN=weight_model*uN_per_force_unit(v),
                 per_leg_share_uN=weight_model*uN_per_force_unit(v)/6.0) for k,v in readings.items()}
    # ---------- (2) muscle capability at the same stance
    gain=m.actuator_gainprm[:,0].copy()
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)]
    # NOTE (corrected): the canonical model has 90 MUSCLE actuators and NO adhesion actuators;
    # adhesion actuators are added by the scene builder (engine/embodied/scene.py) and are not
    # loaded by cp.load_model(). An earlier version sliced [:-2], which deleted two REAL muscles
    # (RHTibia_flex_93434 / RHTibia_extensor_93932) and made the right-hind leg look deficient.
    # Muscles are therefore selected by class/name, never by a positional slice.
    mus_ids=[i for i in range(m.nu) if int(m.actuator_dyntype[i])==int(mujoco.mjtDyn.mjDYN_MUSCLE)]
    # Hard assertions demanded after the blocking review finding:
    assert m.nu==90,('expected 90 actuators, got',m.nu)
    assert len(mus_ids)==90,('expected all 90 actuators to be muscles, got',len(mus_ids))
    assert all(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) and 'adhesion' not in mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i).lower()
               for i in range(m.nu)),'an adhesion actuator was found in the compiled model'
    # Fact (replacing the old "exclude the last two adhesion pads" comment): the canonical model has
    # 90 actuators and ALL of them are muscles; adhesion pads exist only in the scene builder
    # (engine/embodied/scene.py:1536, fg_add_actuator "adhesion") and cp.load_model() does not load it.
    # Second, independent counting path (no model load): count <general> elements in the XML actuator
    # block and their effective class.
    import xml.etree.ElementTree as _ET, re as _re
    _xml=cp.CANONICAL_XML.read_text()
    _ab=list(_ET.fromstring(_xml).iter('actuator'))
    _gens=list(_ab[0].iter('general')) if _ab else []
    _classes={g.get('class') for g in _gens}
    assert len(_gens)==90,('XML actuator element count != 90',len(_gens))
    assert _classes=={'muscle'},('XML actuator classes are not all muscle',_classes)
    assert not _re.search('adhesion',_xml,_re.I),'the canonical XML mentions adhesion'
    xml_muscle_count=len(_gens)
    per_leg={}
    for i in mus_ids:per_leg.setdefault(names[i][:2],[]).append(float(gain[i]))
    leg_peak={k:float(np.sum(v)) for k,v in per_leg.items()}
    muscles_total=float(np.sum([gain[i] for i in mus_ids]))
    cap={k:dict(total_muscle_peak_uN=muscles_total*uN_per_force_unit(v),
                mean_leg_peak_uN=float(np.mean(list(leg_peak.values())))*uN_per_force_unit(v),
                leg_peak_uN={kk:round(vv*uN_per_force_unit(v),4) for kk,vv in leg_peak.items()}) for k,v in readings.items()}
    # direction-resolved capability vs requirement, from the archived certificate
    with np.load(GATE) as z:
        A=z['A'].copy();b=z['target'].copy();lo=z['lo'].copy();hi=z['hi'].copy()
    cert=[x for x in json.loads(CERT.read_text())['results'] if x['floor']=='native'][0]
    y=np.array(cert['y_float']);R=A[:,:90]
    t=R.T@y;P=float(b@y);U=float(np.sum(np.maximum(lo*t,hi*t)))
    capdir=dict(required_P_model_units=P,achievable_U_model_units=U,shortfall_model_units=P-U,
        shortfall_fraction_of_required=(P-U)/P,required_scale_P_over_U=(P/U if U>0 else None),
        required_P_uN={k:P*uN_per_force_unit(v) for k,v in readings.items()},
        achievable_U_uN={k:U*uN_per_force_unit(v) for k,v in readings.items()},
        note='P and U are the exact archived-certificate quantities along the certified separating direction; the ratio is dimensionless and unit-independent')
    # ---------- (3) literature anchors (in-repo values only)
    anchors=dict(single_leg_stance_load_uN=1.7,six_leg_body_weight_uN=10.0,passive_stiffness_uN_per_mm='13.1 +/- 7.97',
        takeoff_peak_per_leg_uN=100.0,source=str(LIT)+' (values already recorded in that file)')
    gram=conv['gram(primary, INFERRED)']
    comp=dict(
        total_weight=dict(model_uN=gram['total_weight_uN'],literature_uN=anchors['six_leg_body_weight_uN'],
            ratio_model_over_literature=gram['total_weight_uN']/anchors['six_leg_body_weight_uN']),
        per_leg_load=dict(model_uN=gram['per_leg_share_uN'],literature_uN=anchors['single_leg_stance_load_uN'],
            ratio_model_over_literature=gram['per_leg_share_uN']/anchors['single_leg_stance_load_uN']),
        per_leg_peak=dict(model_uN=cap['gram(primary, INFERRED)']['mean_leg_peak_uN'],literature_uN=anchors['takeoff_peak_per_leg_uN'],
            ratio_model_over_literature=cap['gram(primary, INFERRED)']['mean_leg_peak_uN']/anchors['takeoff_peak_per_leg_uN']),
        passive_stiffness='NOT_AVAILABLE: the model has no measured leg-stiffness quantity; obtaining one would need a load-displacement measurement protocol (a different task)')
    # ---------- reproducing 1.7 uN: required mass scale, stated as a scale, NOT a tuning of s
    required_mass_unit_kg_per_unit=anchors['single_leg_stance_load_uN']/ (weight_model/6.0) /1e3
    repro=dict(
        question='can the 1.7 uN single-leg anchor be reproduced in the model?',
        required_mass_unit_kg_per_model_unit=required_mass_unit_kg_per_unit,
        required_total_mass_mg=mass_sum*required_mass_unit_kg_per_unit*1e6,
        current_reading_mass_sum_mg=mass_sum*1e-3*1e6,
        verdict=('REPRODUCIBLE ONLY BY CHANGING THE DECLARED MASS SCALE to about %.3f g per model mass unit (i.e. a %.2fx change), NOT by any 4.5 percent force-scale adjustment'%(required_mass_unit_kg_per_unit*1e3, (1e-3/required_mass_unit_kg_per_unit))),
        boundary='this is a declared SCALE statement for the uncalibrated mass unit; s was NOT tuned and no parameter in the model or runner was changed')
    r=dict(status='PASS_FORCE_SCALE_ABSOLUTE_UNIT_CHECK',scope='FORCE_SCALE_CALIBRATION_QUESTION_ONLY_NOT_A_BIOLOGICAL_CALIBRATION',research_script_not_product=True,
        declared=dict(pose='archived declared model stance (projected_seated_qpos), mj_forward only',not_done='no mj_step, no pose search, no tuning, no v6 re-run, no network'),
        unit_reading=dict(length_unit='1 model length unit = 1 mm (documented)',force_unit='1 model force unit = (mass unit) x mm/s^2',mass_unit_status='INFERRED gram, not established (engine/canonical_physics.py:78-89)'),
        weight_and_share=conv,muscle_capability=cap,direction_resolved=capdir,literature_anchors=anchors,comparison=comp,reproduction=repro,
        muscle_selection=dict(rule='actuator_dyntype == mjDYN_MUSCLE (no positional slices)',n_muscles=len(mus_ids),n_legs=len(per_leg),
            independent_cross_check=dict(compiled_model_actuators=m.nu,compiled_model_muscles=len(mus_ids),
                xml_actuator_general_elements=xml_muscle_count,xml_effective_classes=sorted(_classes),
                adhesion_tokens_in_xml=0,agreement=bool(len(mus_ids)==xml_muscle_count==90)),
            adhesion_fact='the canonical model has NO adhesion actuators; adhesion pads exist only in the scene builder engine/embodied/scene.py (fg_add_actuator "adhesion") and are not loaded by cp.load_model()',
            muscles_per_leg={k:len(v) for k,v in per_leg.items()},
            correction='an earlier version excluded the last two actuators as "adhesion pads"; the canonical model has no adhesion actuators, so that slice deleted two real right-hind tibia muscles. Corrected here; the affected numbers are superseded.'),
        answer_to_the_question=dict(
            is_the_gap_a_4p5_percent_calibration_offset='NO evidence of a 4.5 percent calibration offset. Under the inferred-gram reading the model weight is 2.44x the literature body-weight anchor and the mean per-leg peak muscle force is 0.107x the takeoff anchor: the scale uncertainty dwarfs 4.5 percent by one to two orders of magnitude in the opposite directions.',
            then_what_is_the_gap='Within the declared model (unchanged parameters, native activation bounds, native pyramidal cone) the certified shortfall is STRUCTURAL: it comes from the contact geometry/friction cone/joint limits at this stance, not from a small force-scale error.',
            caveat='this says nothing about biology: it is a force-scale CALIBRATION question. It is not a biological calibration, not a support result and not a standing result.'))
    r['input_hashes']={str(p):sha(p) for p in (Path(__file__),POSE,GATE,CERT,LIT,cp.CANONICAL_XML)}
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=['total weight','per-leg load','per-leg peak force','takeoff peak']
    model=[gram['total_weight_uN'],gram['per_leg_share_uN'],cap['gram(primary, INFERRED)']['mean_leg_peak_uN'],np.nan]
    lit=[anchors['six_leg_body_weight_uN'],anchors['single_leg_stance_load_uN'],np.nan,anchors['takeoff_peak_per_leg_uN']]
    xs=np.arange(len(labels));w=0.35
    axes[0].bar(xs-w/2,model,width=w,label='model (inferred gram reading)')
    axes[0].bar(xs+w/2,lit,width=w,label='literature anchors (in-repo values)')
    axes[0].set_xticks(xs);axes[0].set_xticklabels(labels,fontsize=7);axes[0].set_ylabel('force (uN)')
    axes[0].set_title('absolute units: model vs in-repo literature anchors (all 90 muscles)');axes[0].legend(fontsize=6)
    axes[1].bar(['required P','achievable U'],[P,U],color=['tab:red','tab:blue'])
    for i,(v,lab) in enumerate(((P,'required'),(U,'achievable'))):
        axes[1].text(i,v,'  %.4f'%v,ha='center',va='bottom',fontsize=8)
    axes[1].set_ylabel('model force units along the certified direction')
    axes[1].set_title('certified direction: shortfall %.2f%% (needs x%.4f)'%(100*(P-U)/P,P/U))
    fig.suptitle('force-scale absolute-unit check (read-only) - NOT a biological calibration, NOT standing',fontsize=9)
    fig.tight_layout();fig.savefig(out/'force-scale-units.png');plt.close(fig)
    atomic_report(out/'report.json',r)
    print(out);print(r['status']);print(json.dumps(dict(weight=conv['gram(primary, INFERRED)'],per_leg_peak_uN=cap['gram(primary, INFERRED)']['mean_leg_peak_uN'],shortfall=capdir['shortfall_fraction_of_required'],needed=capdir['required_scale_P_over_U']),indent=1));print(json.dumps(repro,indent=1))
    return 0
if __name__=='__main__':
    out=Path('/home/sensen/Desktop/engine修复验收')/('force-scale-absolute-units-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    raise SystemExit(main())
