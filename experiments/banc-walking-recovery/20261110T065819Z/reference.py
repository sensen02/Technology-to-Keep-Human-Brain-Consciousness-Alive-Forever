"""S2/S5 offline privileged foot targets, never a physical gait claim.

XY is body/root-local, Z is desired pad-bottom world height. Source heights
refer to an inferred lower-envelope plane, not measured contact or forces.
No simulator is imported and no physical state is written.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

DEFAULT_REFERENCE = Path('/run/media/sensen/Data2/cell_wound_prototype/references/flybody-measured-2025/selected_reference.npz')
REFERENCE_SHA256 = 'af2f1b6b323a8f97f2761660be2b721cd52754b07a5611141aa45e5dd3d39952'
SOURCE_LEGS = ('RF', 'RM', 'RH', 'LH', 'LM', 'LF')
TARGET_LEGS = ('LF', 'LH', 'LM', 'RF', 'RH', 'RM')
SOURCE_INDEX = (5, 3, 4, 0, 2, 1)


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _smooth(values, weights):
    radius = len(weights) // 2
    padded = np.pad(values, [(radius, radius)] + [(0, 0)] * (values.ndim - 1), mode='edge')
    return sum(w * padded[i:i + len(values)] for i, w in enumerate(weights))


class FootReference:
    """Finite reference with fixed phase during blend and no phase wrapping.

    ``source_time_s`` in sample output is the original dataset clock (.1..7),
    and stays .1 during blend. Source elapsed phase is that value minus .1.
    Source mean XY is each leg's temporal mean over the FULL verified clip,
    before smoothing; target trajectories are centered on supplied rest feet.
    Linear interpolation has piecewise constant derivatives (right-hand at
    source knots). Position is continuous at blend/end; velocity need not be.
    Smoothing is symmetric and uses future samples: offline teacher use only.
    """
    def __init__(self, rest_body_feet, scale=10., vertical_gain=.25,
                 time_warp=10., stance_height_cm=.005, blend_s=.5,
                 source_duration_s=.24, referencepath=DEFAULT_REFERENCE, *,
                 smoothing_samples=5, smoothing_kernel='box', smooth_xy=True,
                 smooth_height=True, blend_curve='smoothstep'):
        rest = np.array(rest_body_feet, dtype=float, copy=True)
        if rest.shape != (6, 3) or not np.isfinite(rest).all():
            raise ValueError('rest_body_feet must be finite (6,3), in TARGET_LEGS order')
        values = (scale, vertical_gain, time_warp, stance_height_cm, blend_s, source_duration_s)
        if not np.isfinite(values).all() or scale <= 0 or time_warp <= 0 or vertical_gain < 0 or stance_height_cm < 0 or blend_s < 0 or source_duration_s <= 0:
            raise ValueError('Invalid finite scale/gain/warp/threshold/blend/duration')
        if smoothing_samples not in (1, 3, 5) or smoothing_kernel not in ('box', 'triangular'):
            raise ValueError('Smoothing uses 1 (disabled), 3 or 5 samples and box/triangular kernel')
        if blend_curve not in ('linear', 'smoothstep'):
            raise ValueError('blend_curve must be linear or smoothstep')
        path = Path(referencepath).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != REFERENCE_SHA256:
            raise ValueError('Reference SHA256 mismatch: ' + digest)
        kinematics_path = path.with_name('KINEMATICS.json')
        kinematics_bytes = kinematics_path.read_bytes()
        kinematics = json.loads(kinematics_bytes)
        if tuple(kinematics['schema']['leg_order']) != SOURCE_LEGS:
            raise ValueError('KINEMATICS source leg order mismatch')
        with np.load(path, allow_pickle=False) as data:
            if tuple(data['leg_order'].tolist()) != SOURCE_LEGS:
                raise ValueError('NPZ source leg order mismatch')
            self.source_arrays = {key: _readonly(data[key]) for key in data.files}
        a = self.source_arrays
        times = a['time_s']
        root = a['root2site']
        heights = a['height_above_inferred_plane_cm']
        if times.shape != (301,) or root.shape != (301, 6, 3) or heights.shape != (301, 6):
            raise ValueError('Unexpected verified source array shapes')
        for key, value in a.items():
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError('Nonfinite source array: ' + key)
        if not np.all(np.diff(times) > 0) or not np.allclose(times, np.arange(301)*.002, rtol=0, atol=1e-12) or not np.allclose(a['source_time_s'], times + .1, rtol=0, atol=1e-12):
            raise ValueError('Unexpected source time contract')
        if source_duration_s > times[-1]:
            raise ValueError('source_duration_s exceeds verified clip')
        n = int(smoothing_samples)
        weights = np.ones(n) if smoothing_kernel == 'box' else np.minimum(np.arange(n)+1, np.arange(n, 0, -1)).astype(float)
        weights /= weights.sum()
        xy = root[:, SOURCE_INDEX, :2].astype(float)
        height = heights[:, SOURCE_INDEX].astype(float)
        mean_xy = xy.mean(axis=0)
        self.source_mean_xy = _readonly(mean_xy)
        self.xy_cm = _readonly(_smooth(xy, weights) if smooth_xy else xy)
        self.height_cm = _readonly(_smooth(height, weights) if smooth_height else height)
        self.rest_body_feet = _readonly(rest)
        self.time_s = _readonly(times)
        self.scale, self.vertical_gain, self.time_warp = float(scale), float(vertical_gain), float(time_warp)
        self.stance_height_cm, self.blend_s = float(stance_height_cm), float(blend_s)
        self.source_duration_s, self.blend_curve = float(source_duration_s), blend_curve
        self.duration_s = self.blend_s + self.source_duration_s * self.time_warp
        self.cfg = dict(scale=float(scale), vertical_gain=float(vertical_gain), time_warp=float(time_warp), stance_height_cm=float(stance_height_cm), blend_s=float(blend_s), source_duration_s=float(source_duration_s), referencepath=str(path), smoothing_samples=n, smoothing_kernel=smoothing_kernel, smooth_xy=bool(smooth_xy), smooth_height=bool(smooth_height), blend_curve=blend_curve)
        nominal = {}
        for i, leg in enumerate(TARGET_LEGS):
            mask = height[:, i] <= stance_height_cm
            stance_mean = (xy[mask, i].mean(axis=0)*scale).tolist() if mask.any() else None
            nominal[leg] = dict(rest_body_xy=rest[i, :2].tolist(), nominal_source_stance_mean_xy=stance_mean)
        self.metadata = dict(cfg=dict(self.cfg), duration_s=self.duration_s, source_sha256=digest,
            source_order=list(SOURCE_LEGS), target_order=list(TARGET_LEGS), source_index_for_target=list(SOURCE_INDEX),
            kinematics_path=str(kinematics_path), kinematics_sha256=hashlib.sha256(kinematics_bytes).hexdigest(),
            source_provenance=kinematics['source'], source_selection=kinematics['selection']['selected'],
            source_time_range_s=[float(a['source_time_s'][0]), float(a['source_time_s'][-1])],
            source_mean_xy_cm=mean_xy.tolist(), xy_center='per-leg first smoothed source phase anchored to measured current footprint; full-clip temporal mean retained for provenance only',
            smoothing=dict(weights=weights.tolist(), padding='edge', symmetric=True, future_samples=True, stage='before interpolation and height clipping'),
            frames=dict(target_xy='body/root-local native', target_z='pad-bottom world height native', velocity='native per elapsed second', source_time_s='absolute source dataset clock'),
            stance='interpolated smoothed source height <= stance_height_cm; blend effective height = threshold + blend_weight*max(0,height0-threshold)',
            derivatives='piecewise linear right-hand at knots; zero at clamped end; blend/source velocity may jump',
            geometry=nominal, foot_excursions=self._excursions(),
            limitations=['Offline privileged teacher only', 'Inferred plane is not measured contact/force', 'Targets are not achieved physical kinematics or demonstrated walking', 'Native units per cm is a nominal geometric mapping; absolute target SI NOT_ESTABLISHED'])

    def _interpolate(self, source_t):
        i = min(int(np.searchsorted(self.time_s, source_t, side='right')) - 1, len(self.time_s)-2)
        i = max(i, 0)
        dt = self.time_s[i+1] - self.time_s[i]
        f = (source_t-self.time_s[i])/dt
        xy = self.xy_cm[i] + f*(self.xy_cm[i+1]-self.xy_cm[i])
        h = self.height_cm[i] + f*(self.height_cm[i+1]-self.height_cm[i])
        return xy, h, (self.xy_cm[i+1]-self.xy_cm[i])/dt, (self.height_cm[i+1]-self.height_cm[i])/dt

    def _excursions(self):
        ts = np.r_[self.time_s[self.time_s < self.source_duration_s], self.source_duration_s]
        xy = np.array([self._interpolate(float(t))[0] for t in ts])*self.scale
        z = np.array([np.maximum(0., self._interpolate(float(t))[1]-self.stance_height_cm) for t in ts])*self.scale*self.vertical_gain
        return {leg: dict(xy_peak_to_peak_native=np.ptp(xy[:,i], axis=0).tolist(), max_padbottom_height_native=float(z[:,i].max()), padbottom_peak_to_peak_native=float(np.ptp(z[:,i]))) for i, leg in enumerate(TARGET_LEGS)}

    def sample(self, elapsed_s):
        elapsed = float(elapsed_s)
        if not np.isfinite(elapsed):
            raise ValueError('elapsed_s must be finite')
        source_t = min(max((elapsed-self.blend_s)/self.time_warp, 0.), self.source_duration_s)
        xy, h, dxy, dh = self._interpolate(source_t)
        # Anchoring at the first measured phase preserves the current supporting
        # footprint. Temporal-mean anchoring would rearrange all loaded feet
        # during blend before any measured cycle has started.
        target_xy = self.rest_body_feet[:, :2] + self.scale*(xy-self.xy_cm[0])
        target_z = self.scale*self.vertical_gain*np.maximum(0., h-self.stance_height_cm)
        stance = h <= self.stance_height_cm
        if elapsed < self.blend_s:
            phase = 'blend'
            u = np.clip(elapsed/self.blend_s, 0., 1.)
            w = u*u*(3.-2.*u) if self.blend_curve == 'smoothstep' else u
            dw = 6.*u*(1.-u)/self.blend_s if self.blend_curve == 'smoothstep' else 1./self.blend_s
            if elapsed < 0.:
                dw = 0.
            delta_xy = target_xy-self.rest_body_feet[:, :2]
            velocity_xy, velocity_z = dw*delta_xy, dw*target_z
            target_xy = self.rest_body_feet[:, :2] + w*delta_xy
            target_z = w*target_z
            # Same source-height criterion at the virtual threshold during blend.
            stance = (w*np.maximum(0., h-self.stance_height_cm)) <= 0.
        elif elapsed >= self.duration_s:
            phase = 'end'
            velocity_xy, velocity_z = np.zeros((6,2)), np.zeros(6)
        else:
            phase = 'reference'
            velocity_xy = self.scale*dxy/self.time_warp
            velocity_z = self.scale*self.vertical_gain*dh/self.time_warp*(h > self.stance_height_cm)
        return dict(target_xy=target_xy.copy(), target_z=target_z.copy(), velocity_xy=velocity_xy.copy(), velocity_z=velocity_z.copy(), stance=stance.copy(), source_time_s=float(self.source_arrays['source_time_s'][0]+source_t), phase=phase)
