"""Current geometric proprioception only; no terrain/world-location/history input."""
from shared import *


def extra_live(b):
    feet=b.d.xpos[b.feet];thorax=b.d.xpos[b.thorax];r=b.d.xmat[b.thorax].reshape(3,3)
    relative=(feet-thorax)@r
    height=thorax[2]-feet[:,2]
    return np.r_[relative.ravel(),height].astype(np.float32)


def extra_saved(path):
    path=Path(path);cache=HERE/'PROPRIO_FAST_CACHE';cache.mkdir(exist_ok=True)
    key=sha(path);file=cache/(key+'.npz')
    if file.exists():
        with np.load(file,allow_pickle=False) as a:return a['extra'].copy()
    m=mujoco.MjModel.from_binary_path(str(path.parent/'BODY.mjb'));d=mujoco.MjData(m)
    thorax=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'nmf/c_thorax')
    feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'nmf/{leg}_tarsus5') for leg in ('lf','lm','lh','rf','rm','rh')]
    with np.load(TEACHER/'CANONICAL_INIT.npz',allow_pickle=False) as a:initial=a['qpos'].copy()
    out=[];error=0.
    with np.load(path,allow_pickle=False) as a:
        poses=a['qpos'].copy();positions=a['position'].copy();initial_position=a['initial_position'].copy();count=len(a['x'])
        for row in range(count):
            tick=row*HOLD
            # mj_step leaves geometric caches at its pre-integration configuration.
            d.qpos[:]=initial if tick==0 else poses[tick-2]
            mujoco.mj_kinematics(m,d)
            expected=initial_position if tick==0 else positions[tick-1]
            error=max(error,float(np.max(np.abs(d.xpos[thorax]-expected))))
            relative=(d.xpos[feet]-d.xpos[thorax])@d.xmat[thorax].reshape(3,3)
            out.append(np.r_[relative.ravel(),d.xpos[thorax,2]-d.xpos[feet,2]])
    assert error<1e-10,('geometry observation timing mismatch',error)
    values=np.asarray(out,dtype=np.float32)
    previous=HERE/'PROPRIO_CACHE'/(key+'.npz')
    if previous.exists():
        with np.load(previous,allow_pickle=False) as old:assert np.array_equal(old['extra'],values)
    np.savez_compressed(file,extra=values)
    write(cache/(key+'.json'),dict(trace=str(path),trace_sha256=key,body_sha256=sha(path.parent/'BODY.mjb'),
               cached_geometry_position_max_error=error,frames=len(out),physical_steps=0,forward_calls=len(out),
               source_sha256=sha(Path(__file__))))
    return values
