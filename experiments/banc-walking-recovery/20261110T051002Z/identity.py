"""Immutable runtime provenance. Assets resolved by the same FlyGym loader as the plant."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / '20261109T212050Z' / 'WALK_MODEL_best.npz'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as f:
            temp = Path(f.name)
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def capture(core, config, extra_paths=()):
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    import physics
    snapshot = physics.rc if hasattr(physics, 'rc') else None
    # Source snapshot contains the declared unchanged XML/pose references.
    if snapshot is None:
        xml = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml')
        pose = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-stand-pose/20261110T030000Z/STAND_POSE_TRIAD_1.45.json')
    else:
        xml, pose = Path(snapshot.PAD_XML), Path(snapshot.POSE_FILE)
    paths = {xml, pose, LEGACY, HERE / 'PREREG.json', HERE / 'env' / 'pyvenv.cfg',
             HERE / 'env/lib/python3.12/site-packages/frozen_physics_base.pth'}
    # Actual imported modules below include __main__ and local dependencies;
    # unrelated development files must not enter a run's executable identity.
    paths.update(Path(p).resolve() for p in extra_paths)
    paths.update((HERE / 'sources').glob('*.py'))
    spec = _load_mjcf(xml)
    for asset in list(spec.meshes) + list(spec.textures):
        if asset.file:
            p = Path(asset.file).resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            paths.add(p)
    imported = []
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, '__file__', None)
        if source and source.endswith('.py'):
            p = Path(source).resolve()
            if p.is_file() and ('cell_wound_prototype' in str(p) or 'flygym' in str(p)):
                # Includes actual physical loader/dependencies and torch implementation.
                paths.add(p)
                imported.append(dict(module=name, path=str(p)))
    files = {str(p.resolve()): dict(sha256=sha256(p), size=p.stat().st_size) for p in sorted(paths)}
    arrays = {}
    for name in dir(core.plant):
        if name.startswith('_'):
            continue
        a = getattr(core.plant, name, None)
        if isinstance(a, np.ndarray):
            arrays[name] = dict(shape=list(a.shape), dtype=str(a.dtype),
                                sha256=hashlib.sha256(a.tobytes()).hexdigest())
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get('Name')
        if name:
            packages[name] = distribution.version
    value = dict(format_version=1, config=config, files=files, compiled_arrays=arrays,
                 imported_modules=sorted(imported, key=lambda x: x['module']),
                 packages=dict(sorted(packages.items())), python=sys.version,
                 executable=sys.executable, platform=platform.platform(),
                 physics=dict(nq=core.plant.nq, nv=core.plant.nv, nu=core.plant.nu,
                              timestep=float(core.plant.opt.timestep),
                              gravity=core.plant.opt.gravity.tolist()),
                 actuator={name: np.asarray(getattr(core.actuator, name)).tolist()
                           for name in ('kp', 'kd', 'torque_limit', 'max_rate_rad_s', 'target_bounds_rad')},
                 adhesion=dict(enabled=core.adhesion_on, per_loaded_leg_weight_fraction=float(physics.rc.ADH_FRAC),
                               stance_load_native=float(physics.rc.STANCE_LOAD), update_period_s=.0005),
                 environment_scope='isolated torch venv with read-only base physics dependency path; no source-model changes')
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    value['identity_sha256'] = hashlib.sha256(payload).hexdigest()
    return value


def verify(manifest):
    canonical = {k: v for k, v in manifest.items() if k != 'identity_sha256'}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != manifest['identity_sha256']:
        raise RuntimeError('manifest identity digest mismatch')
    changed = [p for p, old in manifest['files'].items()
               if not Path(p).is_file() or sha256(p) != old['sha256']]
    if changed:
        raise RuntimeError('runtime input identity changed: ' + ', '.join(changed[:5]))
    return True


def verify_runtime(manifest, core):
    """Validate behavior-relevant compiled parameters and numerical library versions."""
    verify(manifest)
    changed=[]
    for name,old in manifest['compiled_arrays'].items():
        current=np.asarray(getattr(core.plant,name))
        if list(current.shape)!=old['shape'] or str(current.dtype)!=old['dtype'] or hashlib.sha256(current.tobytes()).hexdigest()!=old['sha256']:
            changed.append(name)
    if changed:
        raise RuntimeError('compiled physical parameter drift: '+', '.join(changed[:5]))
    for name in ('numpy','mujoco','flygym','torch'):
        if importlib.metadata.version(name)!=manifest['packages'].get(name):
            raise RuntimeError('numerical package version drift: '+name)
    for name,old in manifest['actuator'].items():
        if not np.array_equal(np.asarray(getattr(core.actuator,name)),np.asarray(old)):
            raise RuntimeError('actuator parameter drift: '+name)
    if platform.python_version()!=manifest['python'].split()[0]:
        raise RuntimeError('Python version drift')
    return True
