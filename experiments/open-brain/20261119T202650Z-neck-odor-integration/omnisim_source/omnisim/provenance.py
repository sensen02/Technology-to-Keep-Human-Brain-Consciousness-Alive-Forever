"""Reproducible source and input fingerprints for simulation reports."""

from hashlib import sha256
from pathlib import Path
from subprocess import run
from tarfile import open as tar_open

ROOT = Path(__file__).resolve().parents[1]


def file_digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshot_digest(directory):
    root = Path(directory)
    return {name: file_digest(root / name) for name in ('manifest.json', 'materials.zip', 'state.npz')}


def _source_files():
    files = [ROOT / 'pyproject.toml', *ROOT.glob('omnisim/**/*.py'), *ROOT.glob('native/*')]
    return sorted(p for p in files if p.is_file())


def save_source_bundle(path):
    with tar_open(path, 'w:gz') as archive:
        for file in _source_files():
            archive.add(file, arcname=file.relative_to(ROOT))
    return file_digest(path)


def source_version(build=None):
    digest = sha256()
    for path in _source_files():
        digest.update(str(path.relative_to(ROOT)).encode() + b'\0')
        digest.update(file_digest(path).encode())
    try:
        revision = run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except FileNotFoundError:
        commit = None
    result = {'source_sha256': digest.hexdigest(), 'git_commit': commit}
    if build is not None:
        binary = Path(build) / 'libomnisim_hip.so'
        result['hip_library_sha256'] = file_digest(binary) if binary.is_file() else None
    return result
