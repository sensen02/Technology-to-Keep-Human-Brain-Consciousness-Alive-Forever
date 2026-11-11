"""Fixed .012 actual-height teacher feedback, pure arithmetic, no model imports."""
import math


def feedback(decision,actual_h,before_h):
    if not isinstance(decision,int) or decision<0 or not all(math.isfinite(float(x)) for x in (actual_h,before_h)):
        raise ValueError('Finite heights and nonnegative integer decision required')
    actual_h=float(actual_h);before_h=float(before_h)
    deficit=max(0.,.012-actual_h)
    active=decision>960 and actual_h<.012
    applied=deficit if active else 0.
    return dict(trigger=active,actual_deficit=deficit,applied_deficit=applied,desired_shadow_height=float(before_h)+applied)


def support_flags(load_W,padz,total_support_W,com_z):
    if not all(math.isfinite(float(x)) for x in (load_W,padz,total_support_W,com_z)):raise ValueError('Finite actual support diagnostics required')
    loaded=bool(load_W>=.1 and padz<=.004)
    return dict(actual_RH_load_W=float(load_W),actual_RH_padz=float(padz),actual_RH_loaded_ground=loaded,
        actual_RH_unloaded=bool(load_W<=.05),actual_RH_unsupported=not loaded,
        actual_total_foot_support_W=float(total_support_W),actual_COM_z=float(com_z))
