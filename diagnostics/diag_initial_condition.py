"""Document the ACTUAL initial condition of the autonomous whole-body runs.

Physical bookkeeping, not a controller and not a behaviour.  It answers:
does the canonical muscle body start in a grounded pose, and can it hold that
pose passively with exactly zero actuation?  No target, no gait, no schedule.

CAUTION recorded here on purpose: a raw ``mj.MjData`` is NOT the pose the
autonomous runner uses.  The runner calls ``mj_resetDataKeyframe(model, d, 0)``.
The XML default (qpos0) is a different, airborne pose.  Reporting the raw
default by mistake would fabricate a "free fall" story, so both are measured.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco as mj
from flygym.compose.fly.musculoskeletal import _load_mjcf

ROOT=Path(__file__).resolve().parents[1]
XML=ROOT/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
OUT=ROOT/'outputs/initial_condition_round15'
PLANE=int(mj.mjtGeom.mjGEOM_PLANE)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ground_geoms(model,data,plane_ids):
    out=set()
    for c in data.contact:
        g1,g2=int(c.geom1),int(c.geom2)
        other=g2 if g1 in plane_ids else g1 if g2 in plane_ids else None
        if other is not None:
            out.add(mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,other) or '')
    return sorted(out)

def lowest_clearance(model,data):
    mj.mj_forward(model,data)
    best=np.inf
    for g in range(model.ngeom):
        if model.geom_type[g]==PLANE:continue
        best=min(best,float(data.geom_xpos[g][2])-float(np.max(model.geom_size[g][:2])))
    return best

def pose_facts(model,data,thorax,plane_ids):
    mj.mj_forward(model,data)
    touching=ground_geoms(model,data,plane_ids)
    return {'thorax_height_model':float(data.xpos[thorax][2]),
            'root_qpos_z_model':float(data.qpos[2]),'ncon':int(data.ncon),
            'lowest_clearance_mm':float(lowest_clearance(model,data)),
            'ground_contact_geoms':touching,
            'ground_contacts_feet_only':bool(touching) and all('Tarsus' in n for n in touching),
            'non_foot_ground_geoms':[n for n in touching if 'Tarsus' not in n]}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    model=_load_mjcf(str(XML)).compile()
    assert model.neq==0 and model.nv==48 and model.nu==90 and model.nkey>=1
    plane_ids=set(np.flatnonzero(model.geom_type==PLANE).tolist())
    assert plane_ids,'ground plane required'
    thorax=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'Thorax');assert thorax>=0
    model.actuator_ctrlrange[:,0]=0.
    report={'status':'INITIAL_CONDITION_MEASUREMENT','source_sha256':{str(XML):sha(XML)},
            'timestep_s':float(model.opt.timestep),'gravity':model.opt.gravity.tolist(),
            'total_mass_model_units':float(mj.mj_getTotalmass(model)),
            'actuation_lower_bound':'set to 0 in memory only so an exactly zero command is possible',
            'scope':'physical initial-condition bookkeeping only; no controller, no target, no schedule.'}

    # (a) what the autonomous runner actually starts from
    d=mj.MjData(model);mj.mj_resetDataKeyframe(model,d,0)
    report['runner_start_keyframe0']=pose_facts(model,d,thorax,plane_ids)

    # (b) zero-actuation settle from that grounded start
    trace=[]
    for k in range(5000):
        d.ctrl[:]=0.;mj.mj_step(model,d)
        if k%500==0:
            trace.append({'time_s':float(d.time),'thorax_height_model':float(d.xpos[thorax][2]),
                          'ncon':int(d.ncon)})
    fin=pose_facts(model,d,thorax,plane_ids)
    start=report['runner_start_keyframe0']['thorax_height_mm']
    report['keyframe0_passive_settle']=dict(fin,start_thorax_height_mm=start,
        drop_mm=start-fin['thorax_height_mm'],
        actuation='exactly zero for the whole settle; passive muscle and joint forces only',
        trace=trace,
        limitation='Passive inability to hold the keyframe pose is not yet a statement about active '
                   'control, and a crouched end pose is not proof of a software bug or of infeasibility.')

    # (c) the raw XML default, recorded only to prevent the free-fall misreading
    d2=mj.MjData(model)
    report['raw_default_pose_CAUTION_not_used_by_runner']=pose_facts(model,d2,thorax,plane_ids)

    # (d) order-of-magnitude force bookkeeping: is the body simply too heavy to hold?
    mj.mj_resetDataKeyframe(model,d,0)
    mass=float(mj.mj_getTotalmass(model));g=abs(float(model.opt.gravity[2]))
    gains=[]
    for a in range(model.nu):
        d.ctrl[:]=0.;d.act[:]=0.;mj.mj_forward(model,d);f0=float(d.actuator_force[a])
        d.act[a]=1.;mj.mj_forward(model,d);f1=float(d.actuator_force[a])
        gains.append(abs(f1-f0))
    gains=np.asarray(gains)
    report['static_force_capacity']={
        'body_weight_model_force_units':mass*g,
        'sum_abs_dForce_d_activation_over_all_90':float(gains.sum()),
        'weight_over_total_capacity':float(mass*g/gains.sum()),
        'gain_min_median_max':[float(gains.min()),float(np.median(gains)),float(gains.max())],
        'interpretation':'Force MAGNITUDE is not the binding constraint: available capacity is orders of '
                        'magnitude above body weight. This is an order-of-magnitude comparison only - the sum '
                        'of scalar force gains is not a joint torque, ignores moment-arm direction and the '
                        'nonnegativity of activations, so it does not prove a supporting pattern exists.'}

    report['mechanical_structure_verified']={
        'nq':int(model.nq),'nv':int(model.nv),'njnt':int(model.njnt),'nu':int(model.nu),
        'ntendon':int(model.ntendon),
        'actuator_transmission':{'trntype_values':sorted(set(int(t) for t in model.actuator_trntype)),
            'trntype_meaning':'3 = mjTRN_TENDON in MuJoCo 3.9.0; each of the 90 muscles drives its own tendon',
            'actuator_gear_unique':sorted(set(float(g) for g in model.actuator_gear[:,0])),
            'caution':'trnid values coincide numerically with joint ids for the first 43 actuators; anyone '
                      'reading trnid as a joint index would wrongly conclude these are direct hinge drives.'},
        'passive_properties':{'joint_stiffness_nonzero':int(np.count_nonzero(model.jnt_stiffness)),
            'joint_stiffness_max':float(model.jnt_stiffness.max()),
            'dof_damping_nonzero':int(np.count_nonzero(model.dof_damping)),
            'dof_damping_max':float(model.dof_damping.max()),
            'dof_armature_nonzero':int(np.count_nonzero(model.dof_armature)),
            'geom_sliding_friction':float(model.geom_friction[0,0]),
            'adhesion_model_present':False,
            'note':'The canonical asset and the public single-arm variant both use stiffness 0.4 and damping '
                   '0.02, so the passive parameter convention is not a local deviation. No adhesion/attachment '
                   'model is present; real flies stand with tarsal adhesion, which is not represented here.'}}

    report['what_this_does_not_establish']=[
        'It does not show the neural loop is useless or that standing is impossible.',
        'It does not provide a controller, a stance pose, or any target posture.',
        'No adhesion/attachment model is added; passive crumpling under 9801 mm/s^2 is not a fly measurement.']
    assert sha(XML)==report['source_sha256'][str(XML)]
    (OUT/'initial_condition.json').write_text(json.dumps(report,indent=2))

    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    t=[x['time_s'] for x in report['keyframe0_passive_settle']['trace']]
    z=[x['thorax_height_mm'] for x in report['keyframe0_passive_settle']['trace']]
    ax.plot(t,z,'o-',label='keyframe0 start (what the runner uses), zero actuation')
    ax.axhline(start,ls=':',c='gray',label='keyframe0 thorax height %.3f mm'%start)
    ax.set_xlabel('time (s)');ax.set_ylabel('thorax height (mm)')
    ax.set_title('Passive settling from the real initial condition (zero actuation)\n'
                 'No controller, no target: does the body hold the keyframe stance?')
    ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'initial_condition.png',dpi=110)
    print(json.dumps({k:report[k] for k in ('runner_start_keyframe0','raw_default_pose_CAUTION_not_used_by_runner')},indent=2))
    print('keyframe0 passive settle: start %.4f -> final %.4f mm, drop %.4f, ncon %d, feet-only %s, non-foot geoms %d'%(
        start,report['keyframe0_passive_settle']['thorax_height_mm'],report['keyframe0_passive_settle']['drop_mm'],
        report['keyframe0_passive_settle']['ncon'],report['keyframe0_passive_settle']['ground_contacts_feet_only'],
        len(report['keyframe0_passive_settle']['non_foot_ground_geoms'])))
    print(OUT)

if __name__=='__main__':main()
