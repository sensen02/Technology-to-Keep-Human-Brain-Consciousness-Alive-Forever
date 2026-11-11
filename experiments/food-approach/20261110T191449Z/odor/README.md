# Engineering food odor and bilateral receptors

This S4 sensory implementation supports later S5 approach experiments. It produces no steering or actions, trains no network, and makes no anatomical or biological calibration claim. Lengths use the caller's simulation coordinates; independent SI calibration is NOT_ESTABLISHED. Existing walking evidence and parent trackers are untouched.

The NumPy-only module is `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/food_field.py`.

```python
from food_field import FoodSource, BilateralOdorSensory, bilateral_positions

# Example engineering numbers only; parent must preregister actual experiment values.
source = FoodSource(center_xy=(4.0, 0.0), patch_radius=0.4,
                    amplitude=1.0, sigma=2.0, onset_s=1.0, rise_s=0.2)
sensory = BilateralOdorSensory(source, K=1.0, tau_s=0.05)
# Actual head body pose, with local +x forward and +y left:
xy = bilateral_positions(head_position, head_rotation,
                         forward_offset=0.1, lateral_offset=0.1)
receptors_lr = sensory.sample(xy, time_s=0.0, dt_s=0.005)
# Actor receives receptors_lr only. Do not pass source, xy, distance or time.
```

`FoodSource` is a frozen validated configuration, defensively copying `center_xy` into an immutable tuple. Its Gaussian is `amplitude * smoothonset * exp(-distance_squared/(2*sigma**2))`. Smooth onset is cubic smoothstep from zero at `onset_s` to one at `onset_s+rise_s`. `patch_radius` describes the contact patch for external food-contact logic; it does not flatten or truncate the Gaussian. `source.concentration(xy,time_s)` is an environment diagnostic, returning shape `(2,)`.

`bilateral_positions(head_position,head_rotation,forward_offset,lateral_offset)` transforms local `(forward,+lateral,0)` and `(forward,-lateral,0)` points into the world and then projects to XY. It accepts either 3D position/proper 3x3 local-to-world rotation or a planar 2D position/proper 2x2 rotation. For MuJoCo use actual head `data.xpos[head_id]` and `data.xmat[head_id].reshape(3,3)`. The parent must verify that body's local axis convention. Offsets are explicit engineering choices; lateral offset is nonnegative and forward offset can be signed.

`BilateralOdorSensory(source,K=1.0,tau_s=0.05,noise_std=0.0,seed=None)` starts with zero latent receptor state. `sample(sensor_positions_xy,time_s,dt_s=0.005)` returns a fresh float64 shape `(2,)` array ordered left/right, bounded to `[0,1]`. `time_s` is the **start** of the interval. The current concentration is held over `[time_s,time_s+dt_s)` and exact exponential integration returns its endpoint response:

```text
u = C/(K+C)
r_next = r + (u-r) * (-expm1(-dt_s/tau_s))
```

Call every 5 ms in production. Positive alternative steps support exact constant-stimulus composition checks, not invariance for moving or changing stimuli. Calls must be contiguous after the first; skipped, repeated or reversed intervals raise `ValueError`. Construct a new instance for each episode. The first time can be any nonnegative interval start, with zero initial receptor state. A crossing-onset interval samples only its start, so onset is resolved causally to the update cadence.

Optional output noise requires an explicit nonnegative integer seed. Independent Gaussian noise is added per receptor per call and clipped to `[0,1]`; latent dynamics remain noise-free. Noisy trajectories depend on the number of calls. Rejected calls do not change state or RNG. Finite inputs are validated, including positive widths, time constants and dt, advancing finite endpoints, proper rotations, and exact sensor array shape `(2,2)`.

Preregistration: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/PREREG.json`.

Tests: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/test_food_field.py`.

Results and source SHA-256: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/TEST_RESULTS.json`.
