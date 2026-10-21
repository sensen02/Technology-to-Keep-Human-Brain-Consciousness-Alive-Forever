"""Episode-only dense objective; no neural input, controller or action target.

One declared reward reformulation: exact clipped cosine replaces the old tilt
cutoff, and contact-count magnitude is diagnostic only. Not biological calibration.
"""
import numpy as np
from .body_reward import body_return


def dense_body_return(samples, weight, initial_height, *, dt=.0001, duration=.55):
    """Use unchanged legacy evidence validation and full right-endpoint window."""
    legacy = body_return(samples, weight, initial_height, dt=dt, duration=duration)
    s = np.asarray(samples, dtype=float)
    positive = (np.clip(s[:, 1] / weight, 0, 1)
                * np.clip(s[:, 3] / initial_height, 0, 1)
                * np.clip(np.cos(s[:, 4]), 0, 1))
    return dict(score=float(np.mean(positive - 2 * (s[:, 2] > 0))),
                capped_support_height_upright=float(positive.mean()),
                nonfoot_duration_s=legacy['nonfoot_duration_s'],
                nonfoot_count_integral=legacy['nonfoot_count_integral'],
                interval_count=legacy['interval_count'], duration_s=duration)
