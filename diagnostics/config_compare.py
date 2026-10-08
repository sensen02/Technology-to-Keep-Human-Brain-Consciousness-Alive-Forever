import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np
from engine.embodied import BodyBackend, BodyConfig
import run_arena_record as R
MODE = sys.argv[1]
if MODE == "tactile_like":
    cfg = BodyConfig(seed=0, scene_preset="meadow_grass", add_tracking_camera=False,
                     add_world_camera=False, add_vision=False)
else:
    cfg = BodyConfig(seed=0, scene_preset="meadow_grass", add_tracking_camera=False,
                     add_world_camera=False, add_vision=False,
                     extra_cameras=R.arena_cameras(), extra_geoms=R.marker_specs(),
                     extra_materials=({"name": R.MARKER_MATERIAL_NAME, "rgba":[1,1,1,1],
                                       "emission":1.0,"reflectance":0.0,"shininess":0.0,"specular":0.0},))
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
dt = cfg.timestep_s
legs = []
zs = []
for k in range(int(1.0/dt)):
    be.step()
    if k % max(1, int(0.005/dt)) == 0:
        obs = be.observe()
        legs.append(int((np.asarray(obs.contact_found_raw) > 0).sum()))
        zs.append(float(be.data.xpos[1][2]))
legs = np.array(legs); zs = np.array(zs)
print(f"{MODE}: frames {legs.size}  legs down mean {legs.mean():.2f}  zero-leg {100*(legs==0).mean():.0f}%")
print(f"   thorax z: mean {zs.mean():.4f} min {zs.min():.4f} max {zs.max():.4f} std {zs.std():.4f}")
