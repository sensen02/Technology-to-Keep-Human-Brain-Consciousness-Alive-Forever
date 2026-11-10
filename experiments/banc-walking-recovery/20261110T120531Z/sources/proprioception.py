"""Declared physical sensory features for later teacher-to-ANN imitation.

No action generation, phase clock, contact-anchor algorithm, root/world position
or trajectory lookup. This input contract is for a replacement ANN, not a claim
that the original connectome has acquired proprioception or autonomous walking.
"""
import numpy as np

LEGS=('LF','LM','LH','RF','RM','RH')
FEATURE_NAMES=(
    [f'joint_offset_{i}' for i in range(42)]
    +[f'joint_target_error_{i}' for i in range(42)]
    +[f'joint_velocity_{i}' for i in range(42)]
    +[f'target_offset_{i}' for i in range(42)]
    +[f'foot_normal_load_{leg}' for leg in LEGS]
    +['gravity_body_x','gravity_body_y','gravity_body_z']
    +['root_linear_velocity_body_x','root_linear_velocity_body_y','root_linear_velocity_body_z']
    +['root_angular_velocity_body_x','root_angular_velocity_body_y','root_angular_velocity_body_z'])


def features(plant,data,target,foot_load):
    """One causal183-vector from CURRENT actual sensed state and own target.

    Joint position scales = native joint half-range, velocities /10rad/s;
    foot load = normal force/nativeweight; body velocities /1native per sec.
    Rotations/body tilt are derived from current Thorax xmat, never overridden.
    Scalar clipping is declared sensory saturation only, not action clipping.
    """
    qa,va,jids=plant.qa,plant.va,plant.jids
    q=data.qpos[qa];velocity=data.qvel[va]
    half=np.maximum(np.ptp(plant.m.jnt_range[jids],axis=1)/2,.1)
    q0=plant.q0[qa]
    rotation=data.xmat[plant.thorax].reshape(3,3)
    gravity_body=rotation.T@np.array([0.,0.,-1.])
    # MuJoCo free joint linear qvel is world, rotational qvel is body-local.
    linear_body=rotation.T@data.qvel[:3]
    angular_body=data.qvel[3:6]
    x=np.r_[(q-q0)/half,(np.asarray(target)-q)/half,velocity/10.,
            (np.asarray(target)-q0)/half,np.asarray(foot_load)/24.446354755878005,
            gravity_body,linear_body,angular_body]
    assert x.shape==(len(FEATURE_NAMES),) and len(x)==183
    assert np.isfinite(x).all()
    return np.clip(x,-10.,10.).astype(np.float32)


def contract():
    return dict(names=FEATURE_NAMES,dimensions=len(FEATURE_NAMES),
        time='current policy-tick state only, before action',
        frame='joint features nativeactuator order; load LF LM LH RF RM RH; velocities/tilt thoraxbody frame',
        normalization='positions/nativehalfjoint range(min.1), jointvelocity/10, footforce/nativeweight; rootlinear/nativeunit/sec, angular/rad/sec',
        saturation=[-10.,10.],
        excluded='worldposition, rootheight, yawheadingreference, phase,time,footanchors,gaitlookup,teacherfuturestates',
        control_source='features for learned replacementANN only; featurecreation neitherintegrates norcommandsbody')
