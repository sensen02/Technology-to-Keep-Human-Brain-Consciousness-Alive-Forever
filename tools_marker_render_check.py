#!/usr/bin/env python3
"""Renders ONE flat-scene frame and reports whether the FLY-BORNE markers show up.

WHY THIS EXISTS.  Every question about the marker set -- are the geoms in the compiled model,
do they render, how bright are they, is the emission working -- is answered by ONE frame plus a
comparison against a build with no markers.  Doing it by eye found five separate defects that no
assertion had caught, so it is kept as a tool rather than thrown away.

HOW TO USE IT (ONE RENDER PER PROCESS: two EGL renderers in one process segfault):
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_marker_render_check.py 0.13 r013 0.18 0.55
    OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_marker_render_check.py 0.13 r013b 0.30 0.80

    THE ARGUMENTS ARE IN MODEL UNITS (mm), which is the trap that hid this for a whole round.

MEASURED RESULT, AND IT IS THE CURRENT BLOCKER: the markers ARE in the model (20 geoms) and DO
change the image -- a with-markers vs without-markers diff differs on 1426 px -- but only by up
to 26 grey levels, whereas the STATIC fiducials, whose material has byte-identical parameters
(emission=1.0, rgba white, reflectance/specular/shininess 0), render at 255.  The fly's copy of
the material therefore does not emit, even with the markers pushed 5 mm clear of the body.  Until
that is resolved the brightness-based detector cannot separate a marker from meadow grass.
"""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0] or ".")
import numpy as np
from pathlib import Path
from PIL import Image
from engine.embodied import BodyBackend, BodyConfig
import run_arena_record as R
rad = float(sys.argv[1]); tag = sys.argv[2]
cfg = BodyConfig(seed=0, scene_preset='meadow_grass', add_world_camera=False, add_tracking_camera=False,
                 add_vision=False, extra_cameras=R.arena_cameras(), extra_geoms=R.marker_specs(),
                 extra_materials=({"name": R.MARKER_MATERIAL_NAME, "rgba":[1,1,1,1],
                                   "emission":1.0,"reflectance":0.0,"shininess":0.0,"specular":0.0},),
                 fly_markers={"segments": R.MARKER_BODY_SEGMENTS, "radius_mm": rad,
                              "material": R.MARKER_MATERIAL_NAME, "offset_m": "outward",
                              "offset_outward_m": float(sys.argv[3])})
be = BodyBackend(cfg, gl_backend="egl")
be = be.attach_cpg_baseline()
rend = be.sim.set_renderer(["cam0"], camera_res=(800,640), playback_speed=1.0, output_fps=100000.0)
for k in range(20):
    be.step()
    if k % 5 == 0: be.sim.render_as_needed()
f = np.asarray(rend.frames["cam0"][-1]); g = f[:,:,:3].astype(float).mean(axis=2)
red = (f[:,:,:3][:,:,0].astype(int) - f[:,:,:3][:,:,1].astype(int) > 60)
print(f"{tag}: n>=250 {int((g>=250).sum())} max {g.max():.0f}  RED-dominant px {int(red.sum())}")
Path("outputs/arena/_marker_check").mkdir(parents=True, exist_ok=True)
Image.fromarray(f[:,:,:3]).crop((320-120,400-120,320+120,400+120)).resize((480,480), Image.NEAREST).save(f"outputs/arena/_marker_check/vis_{tag}.png")
# The ARRAY is saved as well as the picture: the question "do the markers render at all" is
# answered by DIFFING two builds, and a diff of pixels is a measurement while a look is not.
np.save(f"/tmp/mrk_{tag}.npy", f[:,:,:3].astype(np.int16))
print("saved", f"/tmp/mrk_{tag}.npy")
