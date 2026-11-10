"""Measured swing shapes in an adapted quasistatic algorithm-teacher wave schedule.

This is NOT original animal timing, a deployed neural phase input, or physical
walking evidence. Only foot targets are returned; no simulator/state is imported
or written. Source world heights refer to an inferred plane, not measured contact.
"""
from pathlib import Path
import hashlib
import numpy as np

DEFAULT_REFERENCE = Path('/run/media/sensen/Data2/cell_wound_prototype/references/flybody-measured-2025/selected_reference.npz')
REFERENCE_SHA256 = 'af2f1b6b323a8f97f2761660be2b721cd52754b07a5611141aa45e5dd3d39952'
SOURCE_LEGS = ('RF', 'RM', 'RH', 'LH', 'LM', 'LF')
TARGET_LEGS = ('LF', 'LH', 'LM', 'RF', 'RH', 'RM')
SOURCE_INDEX = (5, 3, 4, 0, 2, 1)
LOW_HEIGHT_CM = .005
MIN_ADVANCE_CM = .03
MIN_PEAK_CM = .015
GRID_SIZE = 201


def _readonly(a):
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def _smooth(a):
    padded = np.pad(a, [(2, 2)] + [(0, 0)] * (a.ndim - 1), mode='edge')
    return sum(padded[k:k+len(a)] for k in range(5)) / 5.


def _crossing(h, lo, hi):
    return float(lo + (LOW_HEIGHT_CM-h[lo])/(h[hi]-h[lo]))


def _at(a, indices):
    """Linear interpolation in native frame-index coordinates."""
    idx = np.asarray(indices, dtype=float)
    lo = np.minimum(idx.astype(int), len(a)-2)
    w = idx-lo
    return a[lo] + w.reshape(w.shape + (1,)*(a.ndim-1))*(a[lo+1]-a[lo])


def _extract(xy, height, times, native_times, source_leg):
    """First complete low-crossing interval with a valid local swing peak."""
    above = height > LOW_HEIGHT_CM
    starts = np.flatnonzero(~above[:-1] & above[1:])
    ends = np.flatnonzero(above[:-1] & ~above[1:])
    candidates = []
    selected = None
    for lo in starts:
        later = ends[ends > lo]
        if not len(later):
            continue  # right-clipped swing is not complete
        hi = int(later[0])
        peak = int(lo+1+np.argmax(height[lo+1:hi+1]))
        start = _crossing(height, int(lo), int(lo+1))
        end = _crossing(height, hi, hi+1)
        endpoints = _at(xy, np.array([start, end]))
        advance = float(endpoints[1, 0]-endpoints[0, 0])
        valid = (advance >= MIN_ADVANCE_CM and height[peak] > MIN_PEAK_CM
                 and height[peak] > height[peak-1] and height[peak] >= height[peak+1])
        record = dict(start_bracket_indices=[int(lo), int(lo+1)],
                      end_bracket_indices=[hi, hi+1], peak_native_index=peak,
                      start_fractional_index=start, end_fractional_index=end,
                      source_elapsed_range_s=_at(times, np.array([start, end])).tolist(),
                      source_time_range_s=_at(native_times, np.array([start, end])).tolist(),
                      peak_source_time_s=float(native_times[peak]),
                      advance_root_local_x_cm=advance, peak_height_cm=float(height[peak]),
                      accepted=bool(valid))
        candidates.append(record)
        if valid and selected is None:
            selected = record
    if selected is None:
        raise ValueError('No complete positive measured swing for '+source_leg)
    start, end = selected['start_fractional_index'], selected['end_fractional_index']
    # Insert exact interpolated low crossings around the native interior frames.
    indices = np.r_[start, np.arange(int(np.floor(start))+1, int(np.ceil(end))), end]
    u_native = (indices-start)/(end-start)
    points = _at(xy, indices)
    heights = _at(height, indices)
    span = selected['advance_root_local_x_cm']
    x = np.clip((points[:, 0]-points[0, 0])/span, 0., 1.)
    y = (points[:, 1] - ((1-u_native)*points[0, 1]+u_native*points[-1, 1]))/span
    z = np.maximum(0., heights-((1-u_native)*heights[0]+u_native*heights[-1]))
    peak_z = float(z.max())
    if peak_z <= 0.:
        raise ValueError('Measured swing has no positive baseline-subtracted height')
    z /= peak_z
    grid = np.linspace(0., 1., GRID_SIZE)
    shape = np.column_stack([np.interp(grid, u_native, a) for a in (x, y, z)])
    # Grid peak normalization preserves an exact unit lift after resampling.
    resampled_peak = float(shape[:, 2].max())
    shape[:, 2] /= resampled_peak
    shape[0] = (0., 0., 0.)
    shape[-1] = (1., 0., 0.)
    record = dict(selected)
    record.update(source_leg=source_leg, method='first_valid_complete_low_height_crossings',
                  candidates=candidates, native_fractional_indices=indices.tolist(),
                  native_shape_u=u_native.tolist(), native_smoothed_xy_cm=points.tolist(),
                  native_smoothed_height_cm=heights.tolist(),
                  baseline_subtracted_peak_cm=peak_z,
                  resampled_pre_normalization_z_peak=resampled_peak,
                  shape_u=grid.tolist(), shape_xyz=shape.tolist())
    return shape, record


class WaveReference:
    """Finite sequential swing teacher, with five stance targets at all times.

    At step i within a cycle all feet move backward by (i+u)*stride/6;
    completed legs have advanced stride and the active leg follows measured X.
    Each cycle returns exactly to rest XY. Time is elapsed AFTER parent settle;
    there is no hold/blend. Piecewise linear shape derivatives are right-hand
    inside steps. Velocities can jump at knots/seams; the physical teacher must
    bound commands through its finite-rate interface. No root trajectory exists.
    """
    def __init__(self, rest_body_feet, stride_native=.3, lift_native=.12,
                 step_s=.8, cycles=2, order=None):
        rest = np.asarray(rest_body_feet, dtype=float)
        if rest.shape != (6, 3) or not np.isfinite(rest).all():
            raise ValueError('rest_body_feet must be finite (6,3) in TARGET_LEGS order')
        if not np.isfinite([stride_native, lift_native, step_s]).all() or stride_native <= 0 or lift_native < 0 or step_s <= 0:
            raise ValueError('Require finite stride>0, lift>=0, step_s>0')
        if isinstance(cycles, (bool, np.bool_)) or not isinstance(cycles, (int, np.integer)) or cycles < 1:
            raise ValueError('cycles must be a positive integer')
        digest = hashlib.sha256(DEFAULT_REFERENCE.read_bytes()).hexdigest()
        if digest != REFERENCE_SHA256:
            raise ValueError('Reference SHA256 mismatch: '+digest)
        with np.load(DEFAULT_REFERENCE, allow_pickle=False) as data:
            self.source_arrays = {key: _readonly(data[key]) for key in data.files}
        a = self.source_arrays
        if tuple(a['leg_order'].tolist()) != SOURCE_LEGS:
            raise ValueError('Source leg order mismatch')
        times, native_times = a['time_s'], a['source_time_s']
        root, heights = a['root2site'], a['height_above_inferred_plane_cm']
        if times.shape != (301,) or root.shape != (301, 6, 3) or heights.shape != (301, 6):
            raise ValueError('Unexpected verified source shape')
        if not all(np.isfinite(v).all() for v in (times, native_times, root, heights)) or not np.allclose(times, np.arange(301)*.002, atol=1e-12, rtol=0) or not np.allclose(native_times, times+.1, atol=1e-12, rtol=0):
            raise ValueError('Invalid source values/time contract')
        xy = _smooth(root[:, SOURCE_INDEX, :2].astype(float))
        h = _smooth(heights[:, SOURCE_INDEX].astype(float))
        shapes, records = [], {}
        for j, leg in enumerate(TARGET_LEGS):
            shape, record = _extract(xy[:, j], h[:, j], times, native_times, leg)
            shapes.append(shape)
            records[leg] = record
        default_order = tuple(sorted(TARGET_LEGS, key=lambda leg: records[leg]['peak_source_time_s']))
        self.order = default_order if order is None else tuple(order)
        if len(self.order) != 6 or set(self.order) != set(TARGET_LEGS):
            raise ValueError('order must be a permutation of TARGET_LEGS names')
        self.order_indices = tuple(TARGET_LEGS.index(leg) for leg in self.order)
        self.rest_body_feet = _readonly(rest)
        self.shapes = _readonly(shapes)
        self.stride_native, self.lift_native = float(stride_native), float(lift_native)
        self.step_s, self.cycles = float(step_s), int(cycles)
        self.duration_s = float(self.cycles*6*self.step_s)
        self.segments = records
        self.metadata = dict(
            label='adapted quasistatic algorithm teacher using measured swing shapes',
            cfg=dict(stride_native=self.stride_native, lift_native=self.lift_native,
                     step_s=self.step_s, cycles=self.cycles, order=list(self.order)),
            duration_s=self.duration_s, source_path=str(DEFAULT_REFERENCE), source_sha256=digest,
            source_trajectory_id=str(a['source_trajectory_id'].item()),
            source_order=list(SOURCE_LEGS), target_order=list(TARGET_LEGS),
            source_index_for_target=list(SOURCE_INDEX), segments=records,
            default_order=list(default_order), order_selection='selected measured peak source times ascending' if order is None else 'explicit caller override',
            smoothing=dict(samples=5, weights=[.2]*5, padding='edge', symmetric=True,
                           future_samples=True, stage='XY and inferred-plane world height before detection'),
            extraction=dict(low_height_cm=LOW_HEIGHT_CM, min_positive_x_advance_cm=MIN_ADVANCE_CM,
                            min_peak_height_cm=MIN_PEAK_CM, complete_crossings_required=True,
                            fallback='none; raise ValueError if no measured positive segment'),
            normalization=dict(grid_size=GRID_SIZE, x='(source_x-start_x)/positive endpoint span; clipped [0,1]',
                               y='remove linear endpoint drift; divide by positive source x span',
                               z='subtract linear endpoint height baseline; clamp >=0; normalize native then resampled peak to 1'),
            schedule='one leg at a time; d=stride/6; completed +stride; all -(i+u)*d; active +stride*xshape; cycle resets algebraically to zero',
            frames=dict(target_xy='root-local native', target_z='pad-bottom world height native',
                        velocity='native per elapsed second', source_time_s='active selected measured segment clock, remapped each step; not a continuous animal clock'),
            derivatives='piecewise linear right-hand within steps; may jump at shape knots and step/cycle seams; zero before start and at/after end; sample-hold teacher commands must be physically bounded',
            stance='all true except active leg when target_z>1e-12; target labels are not measured contact',
            limitations=['Adapted wave timing, NOT exact original animal timing',
                         'Teacher-only virtual phase/clock; never deployment NN phase input',
                         'Targets only; no root writes, no physics claims',
                         'Source is keypoint-fitted kinematics; inferred plane is not contact/force',
                         'No hold or blend; parent supplies settle and optional transition',
                         'Physical feasibility and absolute target SI NOT_ESTABLISHED'])

    def sample(self, elapsed_s):
        t = float(elapsed_s)
        if not np.isfinite(t):
            raise ValueError('elapsed_s must be finite')
        clamped = t < 0. or t >= self.duration_s
        if t >= self.duration_s:
            leg = self.order[-1]
            return dict(target_xy=self.rest_body_feet[:, :2].copy(), target_z=np.zeros(6),
                        velocity_xy=np.zeros((6, 2)), velocity_z=np.zeros(6),
                        stance=np.ones(6, dtype=bool),
                        source_time_s=float(self.segments[leg]['source_time_range_s'][1]), phase='end')
        step_float = max(0., t)/self.step_s
        global_step = int(np.floor(step_float))
        i, u = global_step % 6, step_float-global_step
        active = self.order_indices[i]
        v = u*(GRID_SIZE-1)
        k = min(int(np.floor(v)), GRID_SIZE-2)
        shape = self.shapes[active, k] + (v-k)*(self.shapes[active, k+1]-self.shapes[active, k])
        deriv = (GRID_SIZE-1)*(self.shapes[active, k+1]-self.shapes[active, k])/self.step_s
        d = self.stride_native/6.
        xy = self.rest_body_feet[:, :2].copy()
        xy[:, 0] -= (i+u)*d
        xy[list(self.order_indices[:i]), 0] += self.stride_native
        xy[active, 0] += self.stride_native*shape[0]
        xy[active, 1] += self.stride_native*shape[1]
        z = np.zeros(6)
        z[active] = self.lift_native*shape[2]
        velxy = np.zeros((6, 2))
        velxy[:, 0] = -d/self.step_s
        velxy[active] += self.stride_native*deriv[:2]
        velz = np.zeros(6)
        velz[active] = self.lift_native*deriv[2]
        if clamped:
            velxy[:] = 0.
            velz[:] = 0.
        stance = z <= 1e-12
        source_range = self.segments[self.order[i]]['source_time_range_s']
        return dict(target_xy=xy, target_z=z, velocity_xy=velxy, velocity_z=velz,
                    stance=stance, source_time_s=float(source_range[0]+u*(source_range[1]-source_range[0])),
                    phase='reference')
