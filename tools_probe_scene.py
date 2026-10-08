"""Read-only browser probe for the integrated scene layers.

Usage: python3 tools_probe_scene.py [--port 9335] [--focus neural] [--progress 0.25]
Prints the scene manifest summary, per-layer object state, world bounds and any
page errors. Never writes to the project except an optional screenshot.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import urllib.request

PROBE = r"""
(() => {
  const S = window.Workbench && Workbench.scene;
  const vt = window.viewerTest;
  const T = window.THREE;
  if (!S || !vt || !T) return JSON.stringify({error: 'scene or viewer missing', hasScene: !!S, hasViewer: !!vt});
  const l = S.layers;
  const out = {
    ready: S.ready, errors: S.errors, superseded: S.superseded,
    layerKeys: Object.keys(l),
    cnsKeys: l.cns ? Object.keys(l.cns) : null,
    hasSomaPoints: !!(l.cns && l.cns.somaPoints),
  };
  if (l.cns && l.cns.somaPoints) {
    const sp = l.cns.somaPoints;
    out.somaVisible = sp.visible;
    out.somaCount = sp.geometry.attributes.position.count;
    out.material = {opacity: sp.material.opacity, size: sp.material.size, depthTest: sp.material.depthTest};
    const chain = []; let o = sp; while (o) { chain.push(o.visible); o = o.parent; }
    out.visibilityChain = chain;
    const pos = sp.geometry.attributes.position.array;
    let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    for (let i = 0; i < pos.length; i += 3) {
      for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], pos[i + k]); mx[k] = Math.max(mx[k], pos[i + k]); }
    }
    out.localExtent = {min: mn.map(v => +v.toFixed(3)), max: mx.map(v => +v.toFixed(3))};
    const g = l.cns.group;
    out.groupPosition = g.position.toArray().map(v => +v.toFixed(4));
    const world = new T.Vector3().fromArray(pos.slice(0, 3));
    g.updateWorldMatrix(true, false);
    out.firstSomaWorld = world.applyMatrix4(g.matrixWorld).toArray().map(v => +v.toFixed(3));
    const cam = vt.camera;
    out.camera = {pos: cam.position.toArray().map(v => +v.toFixed(2)), near: cam.near, far: cam.far};
    const wb = new T.Box3().setFromObject(l.cns.group);
    out.cnsGroupWorldBox = {min: wb.min.toArray().map(v => +v.toFixed(3)),
                            max: wb.max.toArray().map(v => +v.toFixed(3)),
                            empty: wb.isEmpty()};
    const frustum = new T.Frustum();
    const m = new T.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    frustum.setFromProjectionMatrix(m);
    const sphere = new T.Sphere(); wb.getBoundingSphere(sphere);
    out.cnsSphere = {center: sphere.center.toArray().map(v => +v.toFixed(3)), radius: +sphere.radius.toFixed(4)};
    out.cnsInFrustum = frustum.intersectsSphere(sphere);
    out.parentVisible = (() => { let o = l.cns.group, ch = []; while (o) { ch.push(o.type + ':' + o.visible); o = o.parent; } return ch; })();
  }
  if (l.legs) {
    out.legs = {nLegs: l.legs.nLegs, frames: l.legs.frames, maxForce: +l.legs.maxForce.toFixed(4)};
  }
  if (l.electrodes) {
    out.electrodes = {n: l.electrodes.n, maxCount: l.electrodes.maxCount,
                      radiusMm: +l.electrodes.radiusMm.toFixed(4)};
  }
  out.progress = vt.progress;
  // follow check: the CNS layer must track the recorded body, not sit still
  if (l.cns && l.cns.somaPoints) {
    const sp = l.cns.somaPoints, pos = sp.geometry.attributes.position.array;
    const V = T.Vector3;
    const world = (f) => {
      const g = l.cns.group;
      const p = window.VIEWER_DATA.body.frames[f][1];
      const q = window.VIEWER_DATA.body.rotations_wxyz[f][1];
      g.position.fromArray(p);
      if (q) g.quaternion.set(q[1], q[2], q[3], q[0]).normalize();
      g.updateWorldMatrix(true, false);
      return new V(pos[0], pos[1], pos[2]).applyMatrix4(g.matrixWorld).toArray().map(v => +v.toFixed(3));
    };
    out.followCheck = {frame0: world(0), frame99: world(99)};
  }
  return JSON.stringify(out);
})()
"""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9335)
    ap.add_argument('--url', default='http://127.0.0.1:8766/')
    ap.add_argument('--focus', default=None)
    ap.add_argument('--progress', type=float, default=None)
    ap.add_argument('--shot', default=None)
    ap.add_argument('--wait', type=float, default=12.0)
    args = ap.parse_args()

    import websockets
    proc = subprocess.Popen(
        ['/usr/bin/microsoft-edge', '--headless=new', '--disable-gpu',
         '--remote-debugging-port=%d' % args.port, '--window-size=1400,900',
         '--hide-scrollbars', 'about:blank'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        tabs = None
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen('http://127.0.0.1:%d/json' % args.port, timeout=2))
                break
            except Exception:
                time.sleep(0.5)
        if not tabs:
            print('devtools not reachable')
            return 2
        ws_url = [t for t in tabs if t['type'] == 'page'][0]['webSocketDebuggerUrl']
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            counter = 0

            async def cmd(method, params=None):
                nonlocal counter
                counter += 1
                await ws.send(json.dumps({'id': counter, 'method': method, 'params': params or {}}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get('id') == counter:
                        return msg

            await cmd('Page.enable')
            await cmd('Runtime.enable')
            await cmd('Page.navigate', {'url': args.url})
            await asyncio.sleep(args.wait)
            if args.progress is not None:
                await cmd('Runtime.evaluate', {'expression': 'viewerTest.setProgress(%f)' % args.progress})
            if args.focus:
                if args.focus == 'tissue3d':
                    expr = ('(() => { const S = Workbench.scene, T = THREE, sp = S.layers.tissue3d.points;'
                            ' const box = new T.Box3().setFromObject(sp);'
                            ' const s = box.getBoundingSphere(new T.Sphere());'
                            ' viewerTest.orbit.target.copy(s.center);'
                            ' viewerTest.orbit.distance = Math.max(0.05, s.radius * 6.0);'
                            ' viewerTest.cameraUpdate();'
                            ' return {center: s.center.toArray(), radius: s.radius}; })()')
                else:
                    expr = ('window.Workbench && Workbench.scene && Workbench.scene.focusScene'
                            ' ? Workbench.scene.focusScene("%s") : viewerTest.focus("%s")' % (args.focus, args.focus))
                r0 = await cmd('Runtime.evaluate', {'expression': expr, 'returnByValue': True})
                print('focus ->', json.dumps(r0['result']['result'].get('value'))[:200])
            await asyncio.sleep(2)
            r = await cmd('Runtime.evaluate', {'expression': PROBE, 'returnByValue': True})
            res = r['result']['result']
            payload = res.get('value')
            if payload:
                print(json.dumps(json.loads(payload), indent=1, ensure_ascii=False))
            else:
                print(json.dumps(res, ensure_ascii=False)[:1200])
            if args.shot:
                shot = await cmd('Page.captureScreenshot', {'format': 'png'})
                import base64
                with open(args.shot, 'wb') as fh:
                    fh.write(base64.b64decode(shot['result']['data']))
                print('screenshot:', args.shot)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
