"""Pure mixed-source schedule. No backend, neural or physical imports."""
import math


def source_at(decision):
    if type(decision) is not int or not 0<=decision<1920:raise ValueError('decision integer0..1919 required')
    if decision<960:return 2
    return 3 if (decision-960)%40<5 else 5


def metadata(decision):
    source=source_at(decision)
    return dict(source=source,teacher_elapsed_s=decision*.005,collection=decision>=960,
                burst_boundary=decision==960 or decision>960 and (decision-960)%40 in (0,5),
                phase_is_policy_input=False)


def admit(summary,complete,early_stop,error):
    # Full collection safety admission; never label autonomous success.
    return bool(complete and early_stop is None and error is None and summary.get('nonfoot_force_max_native',math.inf)==0 and
                summary.get('foot_support_mean',-math.inf)>=.90 and summary.get('com_height_min',-math.inf)>=1.20 and
                summary.get('slip_ratio_max',math.inf)<=.5)
