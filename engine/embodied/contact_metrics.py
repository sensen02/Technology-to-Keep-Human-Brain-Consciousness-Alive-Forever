"""Offline contact episodes from synchronous, unmodified physical samples.

A load flag at t[i] describes [t[i], t[i+1]); final sample has no duration.
Threshold choice belongs to preregistration. No debounce merges short episodes.
Foot positions are in a fixed ground/world frame, using consistent units.
A touchdown anchor is reset on EVERY false->true transition. Swing travel is
excluded. The first unloaded position is excluded from slip (its interval is
not in contact). Window-edge episodes are reported separately as censored.
This measures contact kinematics; it cannot establish walking or load bearing.
"""
from math import hypot, isfinite


def contact_episodes(times, loaded, xy):
    """Return per-episode planar drift/path, durations and censoring metadata."""
    times, loaded, xy = list(times), list(loaded), list(xy)
    if not (len(times) == len(loaded) == len(xy)) or len(times) < 2:
        raise ValueError('need >=2 synchronous samples of equal length')
    if any(not isfinite(t) for t in times):
        raise ValueError('times must be finite')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('times must strictly increase')
    if any(len(p) != 2 or any(not isfinite(v) for v in p) for p in xy):
        raise ValueError('xy must contain finite planar positions')
    if any(v not in (False, True) for v in loaded):
        raise ValueError('loaded must be boolean or zero/one')
    episodes = []
    current = None
    anchor = previous = None
    for i, (t, on, position) in enumerate(zip(times, loaded, xy)):
        if on:
            if current is None:
                anchor = previous = position
                current = dict(start_time=t, left_censored=(i == 0),
                               duration=0.0, max_anchor_drift=0.0,
                               contact_path=0.0, samples=0)
            current['samples'] += 1
            current['max_anchor_drift'] = max(
                current['max_anchor_drift'],
                hypot(position[0] - anchor[0], position[1] - anchor[1]))
            current['contact_path'] += hypot(position[0] - previous[0],
                                             position[1] - previous[1])
            previous = position
            if i + 1 < len(times):
                current['duration'] += times[i + 1] - t
        elif current is not None:
            current.update(end_time=t, right_censored=False)
            episodes.append(current)
            current = None
            anchor = previous = None
    if current is not None:
        current.update(end_time=times[-1], right_censored=True)
        episodes.append(current)
    complete = [e for e in episodes if not e['left_censored'] and not e['right_censored']]
    return dict(episodes=episodes, complete_count=len(complete),
                mean_complete_duration=(sum(e['duration'] for e in complete) / len(complete)
                                        if complete else None),
                observed_contact_time=sum(e['duration'] for e in episodes),
                max_observed_anchor_drift=(max(e['max_anchor_drift'] for e in episodes)
                                           if episodes else None),
                duration_convention='left-held intervals; last sample has zero duration',
                slip_convention='world planar drift from each touchdown; only loaded samples')
