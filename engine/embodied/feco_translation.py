"""Fail-closed FeCO input translation prerequisites; no movement decisions.

Evidence: https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/ and
https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/ . Claw encodes joint position,
hook directional movement (not established linear velocity), club movement and
vibration. Population identity alone cannot determine per-root direction/tuning.
This module intentionally does not assign currents from those broad labels.
"""
import math


def assess_feco_channel(channel, sample_interval_s):
    """Explain why a candidate receptor cannot yet inject current automatically."""
    if not math.isfinite(sample_interval_s) or sample_interval_s<=0:
        raise ValueError('positive finite sampling interval required')
    subtype=channel.get('subtype')
    if subtype not in ('claw','hook','club'):
        raise ValueError('explicit FeCO subtype required; broad claw text is insufficient')
    missing=[]
    for key in ('root_id_verified','joint_correspondence_verified','current_conversion_calibrated'):
        if channel.get(key) is not True:missing.append(key)
    if subtype=='claw':
        requirements=('direction_verified','angle_tuning_verified','hysteresis_model_verified')
    elif subtype=='hook':
        requirements=('direction_verified','phasic_response_verified')
    else:
        requirements=('movement_response_verified','frequency_tuning_verified')
        upper=channel.get('max_frequency_hz')
        if not isinstance(upper,(int,float)) or not math.isfinite(upper) or upper<=0:
            missing.append('max_frequency_hz')
        elif sample_interval_s>=1/(2*upper):
            missing.append('sampling_rate_above_nyquist_for_declared_band')
    missing.extend(k for k in requirements if channel.get(k) is not True)
    return {'status':'REQUIRES_TRANSLATOR_REVIEW' if not missing else 'UNREADY_NO_CURRENT_AUTHORIZED',
            'missing':missing,'subtype':subtype,
            'scope':'declaration gate, not a calibrated receptor model or runtime authorization'}
