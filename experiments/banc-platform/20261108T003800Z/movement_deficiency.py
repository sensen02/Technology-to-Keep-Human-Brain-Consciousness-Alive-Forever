"""Measured movement-layer deficiency as explicit platform state.

Round 15's freeze asserted the replacement ANN fails the material foot-slip criterion.
This module replaces that assertion with a measured record: the frozen verifier's own
per-leg force-weighted material contact speeds and their ratios to forward speed, quoted
with source hash. Values are NOT recomputed here (the native contact record needed for an
independent recomputation is not in the saved arrays); they are transcribed from the
verifier output and the addendum verifier checks their internal consistency.

Verifier leg order is ('LF','LM','LH','RF','RM','RH') - the frozen verifier's LEGS constant.
"""
LEG_ORDER = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
THRESHOLD = .5  # frozen verifier max per-leg slip-to-forward-speed ratio
FORWARD_SPEED_NATIVE = .3024167195047777  # gait cycle mean forward speed, same verifier
MEAN_TANGENTIAL_SPEED = (0.18447550105983676, 0.1415518456321517, 0.3131971104916836,
                         0.1526367493510085, 0.259249126863852, 0.29134139445700563)
RMS_TANGENTIAL_SPEED = (0.411353824437029, 0.2816078704249033, 0.5227198299821968,
                        0.2949573942759871, 0.4223065144099467, 0.49749251931859806)
SLIP_RATIO = (0.6100043058529452, 0.4680688483889046, 1.0356474701681815,
              0.5047232494319716, 0.8572579164550994, 0.9633772727053304)
LOADED_SAMPLES = (4144, 5101, 5143, 4242, 5108, 5134)
SOURCE = '/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/evaluation/ann_residual_u32/VERIFICATION.json'
SOURCE_SHA256 = 'cea048fb16c7d00ef136fa21935729223b259857bdcd40504ef1311a9be8fdb6'


def summary():
    failing = [{'leg': leg, 'slip_ratio': r} for leg, r in zip(LEG_ORDER, SLIP_RATIO) if r > THRESHOLD]
    passing = [{'leg': leg, 'slip_ratio': r} for leg, r in zip(LEG_ORDER, SLIP_RATIO) if r <= THRESHOLD]
    return dict(
        controller='replacement_ANN_not_connectome',
        accepted=False,
        criterion='max per-leg force-weighted material slip speed / forward speed <= 0.5 (frozen verifier)',
        threshold=THRESHOLD, leg_order=list(LEG_ORDER), slip_ratio={l: r for l, r in zip(LEG_ORDER, SLIP_RATIO)},
        mean_ratio=sum(SLIP_RATIO) / len(SLIP_RATIO), max_ratio=max(SLIP_RATIO), min_ratio=min(SLIP_RATIO),
        legs_failing=failing, legs_passing=passing, failing_count=len(failing), passing_count=len(passing),
        forward_speed_native=FORWARD_SPEED_NATIVE,
        support_gates='D1-D8 pass at native resolution',
        source=SOURCE, source_sha256=SOURCE_SHA256,
        deficiency_statement=('the replacement ANN produces real displacement with all support gates passing, '
                              'but its stance feet slide at 0.47-1.04 times forward speed, so it is pushing/slipping '
                              'rather than planting; 5 of 6 legs exceed the 0.5 frozen threshold and nothing in this '
                              'workspace calibrates or repairs that'),
        not_measured_here=('independent recomputation of material contact speeds from saved arrays: the saved native '
                           'record has no contact positions or velocities, so the verifier output is quoted, not re-derived'),
    )
