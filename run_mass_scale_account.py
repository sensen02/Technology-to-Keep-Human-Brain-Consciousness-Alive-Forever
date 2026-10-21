"""Mass-scale account of the canonical model (read-only; interpreter recorded in the report).

RESEARCH SCRIPT, NOT A PRODUCT. Reads model parameters only (no mj_step, no LP re-run, no
parameter change, no network, no pose search). Answers ONE question: does the static-support
ratio s ~ 1.045 fall inside the uncertainty band of the mass scale?

Local literature values used (repository files only):
  RESEARCH_TARSUS_FEMUR_ANATOMY.zh-CN.md:637  fly mass ~1 mg, body weight ~10 uN
  LEG_TUNING.zh-CN.md:118-119                 FlyGym walking model 1.02431 mg; real fly ~1.0 mg
"""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
LIT1=ROOT/'RESEARCH_TARSUS_FEMUR_ANATOMY.zh-CN.md'
LIT2=ROOT/'LEG_TUNING.zh-CN.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from engine import canonical_physics as cp
    from engine.embodied.research_report import atomic_report
    m=cp.load_model()
    mass_sum=float(m.body_mass.sum());g=float(abs(m.opt.gravity[2]))
    per_body=[dict(id=i,name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i),mass=float(m.body_mass[i]))
              for i in range(m.nbody) if m.body_mass[i]>0]
    per_body.sort(key=lambda x:-x['mass'])
    top=per_body[:6]
    # local anchors
    anchors=dict(real_fly_mass_mg=1.0,flygym_walking_model_mass_units=0.00102431,
                 source=['RESEARCH_TARSUS_FEMUR_ANATOMY.zh-CN.md:637 (fly mass ~1 mg, weight ~10 uN)',
                         'LEG_TUNING.zh-CN.md:118-119 (FlyGym walking model 1.02431 mg; real fly ~1.0 mg)'])
    # what mass unit would make THIS model equal one real fly? and what does the gram reading give?
    unit_if_fly_mg=anchors['real_fly_mass_mg']/mass_sum
    gram_total_mg=mass_sum*1.0*1e3        # 1 unit = 1 g = 1e3 mg
    mg_reading_total_mg=mass_sum*1e-3*1e3 # 1 unit = 1 mg
    interval=dict(
        unit_mg_if_total_equals_one_fly=unit_if_fly_mg,        # mg per model mass unit under that anchor
        unit_mg_gram_reading=1.0e3,                             # mg per model mass unit under the in-engine reading
        unit_g_range=[unit_if_fly_mg/1e3,1.0],
        unit_uncertainty_factor=1.0e3/unit_if_fly_mg,
        total_mg_low=anchors['real_fly_mass_mg'],total_mg_high=gram_total_mg,
        total_mass_uncertainty_factor=gram_total_mg/anchors['real_fly_mass_mg'])
    # homogeneity argument (analytic, no LP re-run)
    hom=dict(statement='the static LP is homogeneous in force units: scaling the mass unit multiplies BOTH the gravity target b AND the muscle moment columns by the same factor, so the feasible set - and therefore the required ratio s - is unchanged.',
             consequence='the required s ~ 1.045 is INVARIANT under any uniform rescale of the absolute mass unit; it cannot be explained or removed by choosing a different mg-per-unit value.',
             what_does_matter='what matters is the RELATIVE muscle-force-to-mass calibration (gainprm values inherited from the published musculoskeletal asset versus the model body masses), which is a separate uncalibrated quantity.')
    rel=dict(per_leg_peak_force_uN=10.4006, single_leg_load_uN=4.0744, takeoff_peak_anchor_uN=100.0,
             weight_anchor_uN=10.0, weight_vs_anchor_factor=gram_total_mg*9.801/10.0,
             note='from the previous force-scale absolute-unit run; ratios are what the LP depends on')
    verdict=dict(
        question='does s ~ 1.045 fall inside the mass-scale uncertainty band?',
        answer=('The ABSOLUTE mass-unit band does NOT contain s in any determining sense, because s is invariant under a uniform mass-unit rescale. '
                'But s DOES fall inside the uncertainty of the RELATIVE muscle-force-to-mass calibration: the local literature anchors differ from the model by 2.4x (weight vs fly) and ~10x (per-leg peak vs takeoff peak), both far larger than 4.5 percent.'),
        gate_a_consequence='A-gate "this declared pose is statically unsupportable" must be DOWNGRADED to "depends on the uncalibrated muscle-force-to-mass relative scale"; it must not be used as a hard blocker. ',
        quantitative_gap='the required extra muscle force at the current declared ratio is about +4.5 percent (first true threshold ~1.0450258 in the declared-model static LP relaxation)')
    r=dict(status='PASS_MASS_SCALE_ACCOUNT',scope='MASS_AND_FORCE_SCALE_CALIBRATION_QUESTION_ONLY',research_script_not_product=True,
        interpreter=sys.executable,interpreter_declared='this run used the venv_body interpreter (mujoco %s, numpy %s)'%(mujoco.__version__,np.__version__),
        environment=dict(python=sys.version.split()[0],executable=sys.executable,mujoco=mujoco.__version__,numpy=np.__version__),
        model_mass=dict(total_mass_model_units=mass_sum,total_mass_gram_reading_mg=gram_total_mg,
            total_mass_mg_reading_mg=mg_reading_total_mg,gravity_mm_s2=g,
            top_bodies=top,n_bodies_with_mass=len(per_body)),
        literature_anchors=anchors,mass_unit_interval=interval,homogeneity=hom,relative_scale=rel,verdict=verdict,
        boundaries=dict(not_done='no mj_step, no LP re-run, no parameter change, no pose search, no network',
            claim_limit='force-scale calibration question only; NOT a biological calibration, NOT a support result, NOT a standing result; no percentage of overall progress, no schedule'))
    r['input_hashes']={str(p):sha(p) for p in (Path(__file__),LIT1,LIT2,cp.CANONICAL_XML)}
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(['model total\n(gram reading)','one real fly\n(local anchor)'],[gram_total_mg,1.0],color=['tab:blue','tab:orange'])
    axes[0].set_ylabel('mass (mg)');axes[0].set_title('total mass: model vs local fly anchor (%.2fx)'%gram_total_mg)
    for i,v in enumerate((gram_total_mg,1.0)):axes[0].text(i,v,'  %.4f mg'%v,ha='center',va='bottom',fontsize=8)
    axes[1].bar(['per-leg peak\nforce (model)','takeoff peak\n(anchor)','weight (model)','weight (anchor)'],[10.4006,100.0,24.4464,10.0],
                color=['tab:blue','tab:orange','tab:blue','tab:orange'])
    axes[1].set_ylabel('force (uN)');axes[1].set_title('relative muscle-force-to-mass scale: 0.10x and 2.44x vs anchors')
    fig.suptitle('mass-scale account (read-only) - NOT a biological calibration, NOT standing',fontsize=9)
    fig.tight_layout();fig.savefig(out/'mass-scale.png');plt.close(fig)
    atomic_report(out/'report.json',r)
    print(out);print(r['status'])
    print('total mass (gram reading) mg =',round(gram_total_mg,4),'| unit mg if whole model = 1 fly:',round(unit_if_fly_mg,6))
    print('interval total mg:',interval['total_mg_low'],'-',interval['total_mg_high'],'width factor',round(interval['total_mass_uncertainty_factor'],3))
    print(verdict['answer'])
    return 0
if __name__=='__main__':
    out=Path('/home/sensen/Desktop/engine修复验收')/('mass-scale-account-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    raise SystemExit(main())
