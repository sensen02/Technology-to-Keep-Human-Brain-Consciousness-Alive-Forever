from arena_trial import MixedBody, TRIAL, live
import numpy as np
import time

body = MixedBody(10901)
receipt = body.initial_receipt()
np.testing.assert_allclose(np.array(receipt["qpos"])[1:],
                           np.load(live.BASE/"20261117T184709Z-body-terrain-repair/CANONICAL_INIT.npz")["qpos"][1:])
print("spawn diagnostics", receipt["thorax"], [p[0] for p in receipt["stance_feet"]], receipt["native_pairs"], flush=True)
x = body.d.xpos[body.feet, 0]
assert np.all(((2 < x) & (x < 5)) | ((.7 < x) & (x < 1.7)))
print(receipt, flush=True)
begin = time.monotonic()
for tick in range(300):
    np.testing.assert_allclose(body.detail()[0], body.contact()[0], atol=1e-12)
    control, pads = body.teacher(np.array([.8, .8]))
    body.apply(control, pads)
assert not body.d.warning.number.any()
print("official arena 300 native ticks", time.monotonic()-begin, flush=True)
live.write(TRIAL/"SMOKE.json", receipt)
