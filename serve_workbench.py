"""Loopback-only workbench HTTP adapter. See workbench/API.md for P2 contract."""
import argparse
import json
import re
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from workbench import ArtifactStore, DatasetRegistry, RunManager, BenchService
from workbench.artifacts import encoded, strict_loads
from workbench.runs import Conflict

ROOT = Path(__file__).resolve().parent
MODULES = {'data', 'timeline', 'transforms', 'service', 'inspection', 'navigation', 'layers'}
ASSETS = {'/', '/index.html', '/app.js', '/style.css', '/bootstrap.js',
          '/vendor/three.min.js', '/vendor/THREE-LICENSE.txt',
          '/scene/scene.js', '/scene/hud.css'} | {'/modules/' + name + '.js' for name in MODULES}
# Scene chunks are binary payloads produced by scene_build_cli.py. Only names
# listed in the scene manifest are servable, and the name must be a bare
# filename: no separators, no traversal, no dotfiles.
# Names may carry ONE extension; the stem stays restricted to [A-Za-z0-9_] so a
# dot cannot be used to walk the filesystem, but an image chunk (ground.jpg) must
# be nameable -- the earlier pattern only allowed .bin and .json and rejected it.
SCENE_NAME = re.compile(r'^[A-Za-z0-9_]+\.(bin|json|jpg|png)$')
SCENE_TYPES = {'.bin': 'application/octet-stream',
               '.json': 'application/json; charset=utf-8',
               '.jpg': 'image/jpeg', '.png': 'image/png'}


class Workbench:
    def __init__(self, root=ROOT, state=None, **run_options):
        self.root = Path(root)
        self.view = self.root / 'viewer'
        self.scene = self.view / 'scene'
        self.store = ArtifactStore(state or self.root / 'outputs/workbench')
        self.datasets = DatasetRegistry(self.view / 'data.js')
        self.bench = BenchService(self.store, self.root / 'outputs/bench/status.json')
        self.runs = RunManager(self.root, self.store, **run_options)

    def scene_manifest(self):
        path = self.scene / 'manifest.json'
        if not path.is_file():
            raise FileNotFoundError('scene manifest has not been built yet')
        return json.loads(path.read_text())

    def scene_chunk(self, name):
        """Serve one scene chunk, but only if the manifest lists it."""
        if not SCENE_NAME.match(name):
            raise KeyError('scene chunk name rejected')
        manifest = self.scene_manifest()
        listed = set(manifest.get('chunks', {}))
        listed.add('manifest.json')
        if name not in listed:
            raise KeyError('scene chunk not in manifest')
        target = (self.scene / name).resolve()
        if not target.is_relative_to(self.scene.resolve()) or not target.is_file():
            raise KeyError('scene chunk unavailable')
        return target, SCENE_TYPES[target.suffix]

    def status(self):
        return {'service': 'cell-workbench', 'api_version': 'workbench-2',
                'mode': 'simulation replay + backend bench validation; not live acquisition',
                'job': self.runs.latest(), 'bench': self.bench.latest(),
                'queue': {'max_pending': self.runs.max_pending,
                          'queued': sum(j['state'] == 'queued' for j in self.runs.listing()['jobs'])},
                'recovery_warning': self.runs.recovery_warning,
                'worker_error': self.runs.worker_error,
                'healthy': self.runs.worker_error is None}

    def close(self):
        self.runs.close()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(args[2].workbench.view), **kwargs)

    @property
    def app(self):
        return self.server.workbench

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def reply(self, code, data):
        body = encoded(data)
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def local(self):
        host = self.headers.get('Host')
        origin = self.headers.get('Origin')
        return (host in ('127.0.0.1:' + str(self.server.server_port), 'localhost:' + str(self.server.server_port))
                and (not origin or origin == 'http://' + host))

    def do_HEAD(self):
        self.reply(405, {'error': 'HEAD unsupported'})

    def do_GET(self):
        if not self.local():
            return self.reply(403, {'error': 'local same-origin requests only'})
        path = urlparse(self.path).path
        try:
            if path == '/api/scene/manifest':
                return self.reply(200, self.app.scene_manifest())
            if path.startswith('/api/scene/'):
                name = path[len('/api/scene/'):]
                target, ctype = self.app.scene_chunk(name)
                body = target.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == '/api/status':
                return self.reply(200, self.app.status())
            if path == '/api/manifest':
                return self.reply(200, self.app.datasets.manifest())
            if path.startswith('/api/datasets/'):
                # Registry enforces a fixed section-name whitelist, never a filesystem path.
                query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
                if set(query) - {'snapshot'} or len(query.get('snapshot', [])) > 1:
                    return self.reply(400, {'error': 'only one optional snapshot hash accepted'})
                snapshot = query.get('snapshot', [None])[0]
                section = self.app.datasets.section(path[len('/api/datasets/'):], snapshot=snapshot)
                return self.reply(200, section)
            if path == '/api/data':
                data = self.app.datasets.snapshot()
                chain = data.get('bench', {}).get('chain', [])
                data['bench'] = dict(self.app.bench.latest(), chain=chain)
                return self.reply(200, data)
            if path == '/api/recording':
                return self.reply(200, self.app.runs.recording())
            if path == '/api/jobs':
                return self.reply(200, self.app.runs.listing())
            parts = path.strip('/').split('/')
            if len(parts) in (3, 4) and parts[:2] == ['api', 'jobs']:
                if len(parts) == 3:
                    return self.reply(200, self.app.runs.get(parts[2]))
                if parts[3] == 'result':
                    return self.reply(200, self.app.runs.result(parts[2]))
            if path.startswith('/api/'):
                return self.reply(404, {'error': 'unknown endpoint'})
            if path not in ASSETS:
                return self.reply(404, {'error': 'asset unavailable'})
            target = (self.app.view / ('index.html' if path == '/' else path.lstrip('/'))).resolve()
            if not target.is_relative_to(self.app.view.resolve()) or not target.is_file():
                return self.reply(404, {'error': 'asset unavailable'})
            super().do_GET()
        except KeyError as exc:
            self.reply(404, {'error': str(exc)})
        except FileNotFoundError as exc:
            self.reply(404, {'error': str(exc), 'status': 'SCENE_NOT_BUILT'})
        except Conflict as exc:
            self.reply(409, {'error': str(exc)})
        except (ValueError, TypeError, OSError) as exc:
            self.reply(503, {'error': str(exc), 'status': 'INVALID_OR_UNAVAILABLE_ARTIFACT'})

    def do_POST(self):
        if not self.local():
            return self.reply(403, {'error': 'local same-origin requests only'})
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            return self.reply(415, {'error': 'application/json required'})
        if self.headers.get('Transfer-Encoding'):
            return self.reply(400, {'error': 'chunked requests unsupported'})
        try:
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1:
                return self.reply(400, {'error': 'one Content-Length required'})
            size = int(lengths[0])
            if size < 0 or size > 2 * 1024**2:
                return self.reply(413, {'error': 'request limit 2MiB'})
            self.connection.settimeout(10)
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError('incomplete request')
            data = strict_loads(raw or b'{}')
            if not isinstance(data, dict):
                raise ValueError('JSON object required')
            path = urlparse(self.path).path
            if path == '/api/jobs/recording':
                job, duplicate = self.app.runs.submit(data)
                return self.reply(200 if duplicate else 202, dict(job, duplicate=duplicate))
            parts = path.strip('/').split('/')
            if len(parts) == 4 and parts[:2] == ['api', 'jobs'] and parts[3] == 'cancel':
                if data:
                    raise ValueError('cancel accepts empty object only')
                return self.reply(200, self.app.runs.cancel(parts[2]))
            if path == '/api/bench/evaluate':
                result = self.app.bench.evaluate(data)
                return self.reply(400 if result['status'] == 'INVALID_INPUT' else 200, result)
            return self.reply(404, {'error': 'unknown endpoint'})
        except KeyError as exc:
            self.reply(404, {'error': str(exc)})
        except Conflict as exc:
            self.reply(409, {'error': str(exc)})
        except (ValueError, TypeError) as exc:
            self.reply(400, {'error': str(exc), 'status': 'INVALID_INPUT'})
        except Exception as exc:
            self.reply(500, {'error': type(exc).__name__ + ': ' + str(exc)})


def make_server(root=ROOT, state=None, port=8766, **run_options):
    app = Workbench(root, state, **run_options)
    try:
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        server.workbench = app
        return server
    except Exception:
        app.close()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = make_server(port=args.port)
    print('Workbench http://127.0.0.1:%d' % server.server_port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.workbench.close()


if __name__ == '__main__':
    main()
