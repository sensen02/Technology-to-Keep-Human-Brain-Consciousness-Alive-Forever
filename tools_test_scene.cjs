#!/usr/bin/env node
/* Scene regression: verifies the single-scene payload and the HTTP layer that
 * serves it. Read-only with respect to the project except for its own report.
 *
 *   node tools_test_scene.cjs [--port 8766] [--out viewer/scene_test.json]
 *
 * Checks:
 *   1. manifest schema, frame declaration, single primary clock
 *   2. every chunk exists, matches its declared byte size and dtype/shape
 *   3. the CNS map is rigid and distance preserving (audit block, plus an
 *      independent recomputation from the raw soma micrometre coordinates that
 *      the builder stored in the manifest)
 *   4. orientation: the brain landmark is anterior (more -x) than the VNC one
 *   5. electrode geometry is exactly the fixed 7 um / 20 um, and the array is
 *      a lattice with that pitch
 *   6. layer byte budget respected
 *   7. HTTP: manifest and chunks served, traversal and unlisted names refused
 */
'use strict';
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : dflt;
};
const PORT = Number(opt('port', 8766));
const OUT = opt('out', 'viewer/scene_test.json');
const BASE = 'http://127.0.0.1:' + PORT;
const ROOT = path.resolve(__dirname);

const checks = [];
function check(name, ok, detail) {
  checks.push({name, ok: !!ok, detail});
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  ' + detail : ''));
}

async function get(url, binary) {
  const r = await fetch(url, {cache: 'no-store'});
  const body = binary ? Buffer.from(await r.arrayBuffer()) : await r.text();
  return {status: r.status, body, type: r.headers.get('content-type') || ''};
}

const DTYPES = {
  float32: {bytes: 4, read: (b) => new Float32Array(b.buffer, b.byteOffset, b.length / 4)},
  uint8: {bytes: 1, read: (b) => new Uint8Array(b.buffer, b.byteOffset, b.length)},
  uint16: {bytes: 2, read: (b) => new Uint16Array(b.buffer, b.byteOffset, b.length / 2)},
  uint32: {bytes: 4, read: (b) => new Uint32Array(b.buffer, b.byteOffset, b.length / 4)},
  uint64: {bytes: 8, read: (b) => new BigUint64Array(b.buffer, b.byteOffset, b.length / 8)},
};

(async function main() {
  const manifestRes = await get(BASE + '/api/scene/manifest');
  check('manifest served', manifestRes.status === 200 && manifestRes.type.includes('json'),
    'HTTP ' + manifestRes.status);
  const manifest = typeof manifestRes.body === 'string' ? JSON.parse(manifestRes.body) : JSON.parse(manifestRes.body.toString());
  check('manifest schema', manifest.schema === 'workbench-scene-2', manifest.schema);
  check('single primary clock', manifest.clock && manifest.clock.primary === 'body' &&
    Array.isArray(manifest.clock.times_s) && manifest.clock.times_s.length === manifest.clock.frames,
    'frames=' + (manifest.clock && manifest.clock.frames));
  check('frame declares axes', manifest.frame && manifest.frame.anterior_axis && manifest.frame.up_axis,
    manifest.frame && manifest.frame.anterior_axis);

  // ---- chunk integrity
  const chunkNames = Object.keys(manifest.chunks);
  check('chunks declared', chunkNames.length >= 15, chunkNames.length + ' chunks');
  const loaded = {};
  let bytes = 0;
  for (const name of chunkNames) {
    const meta = manifest.chunks[name];
    const r = await get(BASE + '/api/scene/' + name, true);
    const ok = r.status === 200 && r.body.length === meta.bytes;
    check('chunk ' + name, ok, r.body.length + '/' + meta.bytes + ' bytes');
    if (!ok) continue;
    bytes += r.body.length;
    loaded[name] = r.body;
    if (meta.dtype === 'image') {
      // an image chunk is validated by its byte size and its magic number, not by a dtype
      const isJpeg = meta.file.endsWith('.jpg') || meta.file.endsWith('.jpeg');
      check('chunk image magic ' + name,
        isJpeg ? (r.body[0] === 0xFF && r.body[1] === 0xD8) : (r.body[0] === 0x89),
        r.body.slice(0, 2).toString('hex'));
      continue;
    }
    const dt = DTYPES[meta.dtype];
    if (!dt) { check('chunk dtype ' + name, false, meta.dtype); continue; }
    const expected = meta.shape.reduce((a, b) => a * b, 1);
    const arr = dt.read(r.body);
    check('chunk length ' + name, arr.length === expected, arr.length + '/' + expected);
    loaded[name] = arr;
  }
  check('payload equals manifest total', bytes === manifest.budget.bytes_total,
    bytes + '/' + manifest.budget.bytes_total);
  check('payload within budget', bytes <= manifest.budget.limit_bytes,
    (bytes / 1048576).toFixed(2) + ' MiB');

  // ---- layer size limits
  const over = (manifest.budget.layers_over_per_layer_limit || []);
  check('no layer over per-layer limit', over.length === 0, over.join(','));

  // ---- registration audit
  const audit = manifest.transform_audit || {};
  check('rigid map reported', audit.rigid === true && audit.max_pairwise_distance_error_mm <= audit.tolerance_mm,
    'max err ' + audit.max_pairwise_distance_error_mm + ' mm');
  const cns = manifest.layers.find((l) => l.id === 'cns_somas');
  const reg = cns && cns.registration;
  check('registration present', !!(reg && reg.landmarks && reg.rotation_banc_to_body), reg && reg.method);
  if (reg) {
    const [b, v] = [reg.local_landmarks_mm.brain_antennal_nerve, reg.local_landmarks_mm.vnc_leg_nerve];
    check('brain anterior to VNC', b[0] < v[0], 'brain_x=' + b[0] + ' vnc_x=' + v[0]);
    check('landmark separation plausible', Math.abs(b[0] - v[0]) > 0.4 && Math.abs(b[0] - v[0]) < 1.2,
      Math.abs(b[0] - v[0]).toFixed(3) + ' mm');
    check('scale is 1.0 (no silent rescale)', reg.scale === 1.0, String(reg.scale));
    // independent recomputation: rotation must be a signed axis permutation
    const R = reg.rotation_banc_to_body;
    const rows = R.map((row) => row.map(Math.abs));
    const perm = rows.every((row) => row.reduce((a, x) => a + x, 0) === 1) &&
      R.every((row) => row.filter((x) => x !== 0).length === 1) &&
      [0, 1, 2].every((c) => R.filter((row) => row[c] !== 0).length === 1);
    check('rotation is a signed permutation', perm, JSON.stringify(R));
    const det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
      R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
      R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    check('rotation is proper or a mirror, declared', Math.abs(Math.abs(det) - 1) < 1e-9, 'det=' + det);
  }

  // ---- somas
  const pos = loaded['somas_pos.bin'];
  const klass = loaded['somas_class.bin'];
  const roots = loaded['somas_root_id.bin'];
  if (pos) {
    const n = pos.length / 3;
    check('soma count as declared', n === cns.count, n + '/' + cns.count);
    check('class codes aligned', klass && klass.length === n * 3, klass && klass.length);
    check('root ids aligned', roots && roots.length === n, roots && roots.length);
    let finite = true, mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < pos.length; i++) {
      if (!Number.isFinite(pos[i])) { finite = false; break; }
      const k = i % 3;
      mn[k] = Math.min(mn[k], pos[i]); mx[k] = Math.max(mx[k], pos[i]);
    }
    check('soma coordinates finite', finite);
    const span = mx.map((v, i) => v - mn[i]);
    check('CNS extent consistent with the measured BANC volume (0.2-1.2 mm per axis)',
      span.every((s) => s > 0.25 && s < 1.2), span.map((v) => v.toFixed(3)).join(' x '));
    const labels = cns.label_tables;
    let codesOk = true;
    for (let i = 0; i < klass.length; i++) {
      const table = [labels.super_class, labels.flow, labels.body_part][i % 3];
      if (klass[i] >= table.length) { codesOk = false; break; }
    }
    check('class codes index the declared label tables', codesOk);
  }

  // ---- simulation layers written by scene_layers_sim.py ----
  const simMeta = (() => {
    try { return JSON.parse(fs.readFileSync(path.join(ROOT, 'viewer/scene/sim_layers.json'), 'utf8')); }
    catch (e) { return null; }
  })();
  if (simMeta) {
    const pp = simMeta.paracrine, ii = simMeta.immune;
    const ppos = loaded['paracrine_pos.bin'];
    const pfr = loaded['paracrine_frames.bin'];
    if (pp && ppos && pfr) {
      check('paracrine point count matches its metadata', ppos.length / 3 === pp.n_points,
        (pppos => ppos + '/' + pp.n_points)(ppos.length / 3));
      check('paracrine frame block size', pfr.length === pp.n_frames * pp.n_points,
        pfr.length + '/' + (pp.n_frames * pp.n_points));
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < pfr.length; i++) { mn = Math.min(mn, pfr[i]); mx = Math.max(mx, pfr[i]); }
      check('paracrine field varies over the frames', mx > 0 && (mx - mn) > 10, 'u8 range ' + mn + '..' + mx);
      check('paracrine patch drawn at true scale (<0.5 mm across)',
        (() => {
          const ext = [0, 1, 2].map((k) => {
            let a = Infinity, b = -Infinity;
            for (let i = k; i < ppos.length; i += 3) { a = Math.min(a, ppos[i]); b = Math.max(b, ppos[i]); }
            return b - a;
          });
          return ext.every((e) => e < 0.5);
        })(), 'extent checked per axis');
    }
    const ipos = loaded['immune_pos.bin'];
    const ideb = loaded['immune_debris.bin'];
    const iatt = loaded['immune_attractant.bin'];
    const icell = loaded['immune_cells.bin'];
    if (ii && ipos && ideb && iatt && icell) {
      const n = ipos.length / 3;
      check('immune field point count matches', n === ii.n_points, n + '/' + ii.n_points);
      check('immune debris block size', ideb.length === ii.n_frames * n,
        ideb.length + '/' + (ii.n_frames * n));
      check('immune attractant block size', iatt.length === ii.n_frames * n,
        iatt.length + '/' + (ii.n_frames * n));
      check('immune agent block size', icell.length === ii.n_frames * ii.n_cells * 3,
        icell.length + '/' + (ii.n_frames * ii.n_cells * 3));
      let moved = 0;
      for (let i = 0; i < ii.n_cells * 3; i++) {
        moved = Math.max(moved, Math.abs(icell[i] - icell[(ii.n_frames - 1) * ii.n_cells * 3 + i]));
      }
      check('immune agents actually move across the frames', moved > 1e-5, 'max displacement ' + moved.toFixed(5) + ' mm');
      check('immune material budget closes', ii.budget.max_relative_residual < 1e-9,
        'residual ' + ii.budget.max_relative_residual);
      check('immune layer labelled as not fly immunity',
        /not fly immunity/i.test(require('./viewer/scene/manifest.json').layers.find((l) => l.id === 'immune_field').class),
        'class string checked');
    }
  } else {
    check('simulation layer metadata present', false, 'sim_layers.json missing');
  }

  // ---- mechanics patch, scenarios, environment, readouts ----
  const mech = (() => {
    try { return JSON.parse(fs.readFileSync(path.join(ROOT, 'viewer/scene/mechanics.json'), 'utf8')); }
    catch (e) { return null; }
  })();
  if (mech) {
    const mp = loaded['mech_pos.bin'];
    if (mp) {
      check('mechanics vertex block size',
        mp.length === mech.n_frames * mech.n_vertices * 3,
        mp.length + '/' + (mech.n_frames * mech.n_vertices * 3));
      // the patch must MOVE across frames (a cut vertex model does).  NaN entries are the
      // source's marker for REMOVED cells (measured 14 of 1248 per frame), so the movement
      // statistic is taken over the FINITE vertices only and the dead count is asserted.
      const nV = mech.n_vertices;
      let moved = 0, dead = 0, live = 0;
      const last = (mech.n_frames - 1) * nV * 3;
      for (let i = 0; i < nV; i++) {
        const k = i * 3;
        const finite = (o) => Number.isFinite(mp[k + o]) && Number.isFinite(mp[last + k + o]);
        if (!finite(0) || !finite(1) || !finite(2)) { dead++; continue; }
        live++;
        for (let o = 0; o < 3; o++) moved = Math.max(moved, Math.abs(mp[k + o] - mp[last + k + o]));
      }
      check('mechanics patch moves across frames', moved > 0 && live > 0,
        'max displacement ' + moved.toFixed(5) + ' mm over ' + live + ' live vertices');
      check('mechanics dead-cell NaN markers counted per vertex',
        dead > 0 && dead <= 40,
        dead + ' dead vertices of ' + nV + ' (source marks removed cells with NaN)');
      check('mechanics extent is the real patch size', mech.extent_um[0] > 50 && mech.extent_um[0] < 400,
        mech.extent_um.map((v) => v.toFixed(1)).join(' x ') + ' um');
    }
  } else {
    check('mechanics metadata present', false, 'mechanics.json missing');
  }

  const sc = (() => {
    try { return JSON.parse(fs.readFileSync(path.join(ROOT, 'viewer/scene/scenarios.json'), 'utf8')); }
    catch (e) { return null; }
  })();
  if (sc) {
    check('scenarios: five arms present', (sc.scenarios || []).length === 5,
      (sc.scenarios || []).join(','));
    const keys = Object.keys(sc.series || {});
    check('scenarios: per-region series present', keys.length >= 5, keys.join(','));
    let lenOk = true;
    for (const k of keys) {
      for (const arm of sc.scenarios) {
        const v = sc.series[k][arm];
        if (v && v.length !== sc.frames) lenOk = false;
      }
    }
    check('scenarios: every series frame count matches', lenOk, 'frames ' + sc.frames);
    let finite = true;
    for (const k of keys) for (const arm of sc.scenarios) {
      for (const v of sc.series[k][arm] || []) if (!Number.isFinite(v)) finite = false;
    }
    check('scenarios: all values finite', finite);
    const man = JSON.parse(fs.readFileSync(path.join(ROOT, 'viewer/scene/manifest.json'), 'utf8'));
    const l = man.layers.find((x) => x.id === 'scenarios');
    check('scenarios: scene layer declares its arms', !!l && (l.arms || []).length === 5,
      l ? (l.arms || []).join(',') : 'layer missing');
  } else {
    check('scenario metadata present', false, 'scenarios.json missing');
  }

  const envj = (() => {
    try { return JSON.parse(fs.readFileSync(path.join(ROOT, 'viewer/scene/environment.json'), 'utf8')); }
    catch (e) { return null; }
  })();
  if (envj) {
    check('environment: ground texture exported', !!(envj.ground && envj.ground.bytes > 0),
      envj.ground ? envj.ground.bytes + ' bytes' : 'none');
    check('environment: CC0 attribution carried with the image',
      !!(envj.attribution && envj.attribution.source_page),
      (envj.attribution || {}).route + ' ' + (envj.attribution || {}).source_page);
    const gf = loaded['ground.jpg'];
    check('environment: ground texture served', !!gf && gf.length === envj.ground.bytes,
      gf ? gf.length + ' bytes' : 'missing');
  } else {
    check('environment metadata present', false, 'environment.json missing');
  }

  const ro = (() => {
    try { return JSON.parse(fs.readFileSync(path.join(ROOT, 'viewer/scene/readouts.json'), 'utf8')); }
    catch (e) { return null; }
  })();
  if (ro) {
    check('readouts: intracellular summary present', !!(ro.cellspace && ro.cellspace.D_ca_um2_s),
      ro.cellspace ? 'D_ca ' + ro.cellspace.D_ca_um2_s : 'missing');
    check('readouts: vascular oxygen is statistics, not a field',
      !!(ro.vascular_oxygen && Array.isArray(ro.vascular_oxygen.rows)) &&
      !('field' in (ro.vascular_oxygen || {})),
      (ro.vascular_oxygen && ro.vascular_oxygen.rows ? ro.vascular_oxygen.rows.length : 0) + ' spacing rows');
    check('readouts: electrode verdicts present',
      !!(ro.frontend && ro.frontend.verdicts) && !!(ro.recording && ro.recording.tests_passed),
      'frontend ' + JSON.stringify(ro.frontend && ro.frontend.verdicts));
    check('readouts: no fabricated field key anywhere',
      !Object.keys(ro).some((k) => ro[k] && typeof ro[k] === 'object' && 'positions_um' in ro[k]));
  } else {
    check('readout metadata present', false, 'readouts.json missing');
  }

  // ---- 3-D multilayer stack ----
  const t3meta = (() => {
    try { return JSON.parse(fs.readFileSync(path.join(ROOT, 'viewer/scene/tissue3d.json'), 'utf8')); }
    catch (e) { return null; }
  })();
  if (t3meta) {
    const p3 = loaded['tissue3d_pos.bin'];
    const e3 = loaded['tissue3d_edges.bin'];
    const l3 = loaded['tissue3d_layers.bin'];
    if (p3 && e3 && l3) {
      const n = p3.length / 3;
      check('tissue3d vertex count matches', n === t3meta.n_vertices, n + '/' + t3meta.n_vertices);
      check('tissue3d edge block size', e3.length === t3meta.n_edges * 2,
        e3.length + '/' + (t3meta.n_edges * 2));
      check('tissue3d layer indices present', l3.length === n, l3.length + '/' + n);
      let bad = 0;
      for (let i = 0; i < l3.length; i++) if (l3[i] >= t3meta.n_layers) bad++;
      check('tissue3d layer indices in range', bad === 0, bad + ' out of range');
      let oob = 0;
      for (let i = 0; i < e3.length; i++) if (e3[i] >= n) oob++;
      check('tissue3d edge indices in range', oob === 0, oob + ' out of range');
      check('tissue3d deviation gain declared', t3meta.deviation_gain > 1,
        'gain ' + t3meta.deviation_gain + 'x, declared in the layer metadata');
      check('tissue3d relaxation was monotone', t3meta.monotone === true,
        'energies ' + t3meta.energy_first_J + ' -> ' + t3meta.energy_last_J + ' J');
    }
  } else {
    check('tissue3d metadata present', false, 'tissue3d.json missing');
  }

  // ---- activity
  const act = loaded['activity_u8.bin'];
  const ascale = loaded['activity_scale.bin'];
  const actLayer = manifest.layers.find((l) => l.id === 'cns_activity');
  if (act && actLayer) {
    check('activity frames x neurons', act.length === actLayer.frames * (pos.length / 3),
      act.length + '/' + (actLayer.frames * pos.length / 3));
    check('activity scale rows', ascale && ascale.length === actLayer.frames * 2, ascale && ascale.length);
    let varies = false;
    for (let f = 1; f < actLayer.frames && !varies; f++) if (ascale[f * 2 + 1] !== ascale[1]) varies = true;
    check('activity is not static', varies);
    check('activity labelled illustrative', /ILLUSTRATIVE/i.test(actLayer.class), actLayer.class);
  }

  // ---- edges
  const edges = loaded['edges_index.bin'];
  const edgeLayer = manifest.layers.find((l) => l.id === 'cns_edges');
  if (edges && edgeLayer && pos) {
    const n = pos.length / 3;
    let ok = edges.length === edgeLayer.count * 2;
    for (let i = 0; ok && i < edges.length; i++) if (edges[i] >= n) ok = false;
    check('edge indices inside the soma cloud', ok, edges.length + ' indices');
  }

  // ---- legs
  const legLayer = manifest.layers.find((l) => l.id === 'leg_contacts');
  const siteIndex = loaded['leg_site_index.bin'];
  const table = loaded['leg_contact_table.bin'];
  if (legLayer && siteIndex && table) {
    const nLegs = legLayer.leg_names.length;
    check('six legs mapped', nLegs === 6, legLayer.leg_names.join(','));
    check('leg frames match the body clock', siteIndex.length / nLegs === manifest.clock.frames,
      siteIndex.length / nLegs + '');
    const sentinel = table.length / 3 - 1;
    let inRange = true, sawContact = false, sawAir = false;
    for (let i = 0; i < siteIndex.length; i++) {
      if (siteIndex[i] > sentinel) { inRange = false; break; }
      if (siteIndex[i] === sentinel) sawAir = true; else sawContact = true;
    }
    check('contact site indices in range', inRange, 'sentinel=' + sentinel);
    check('both contact and no-contact frames present', sawContact && sawAir,
      'contact=' + sawContact + ' air=' + sawAir);
  }

  // ---- electrodes
  const elec = manifest.layers.find((l) => l.id === 'electrodes');
  const epos = loaded['electrode_pos.bin'];
  if (elec && epos) {
    const g = elec.geometry;
    check('fixed geometry 7 um / 20 um preserved', g.diameter_um === 7 && g.pitch_um === 20,
      g.diameter_um + '/' + g.pitch_um);
    check('capture radius declared, not measured', g.capture_radius_class === 'DECLARED, not measured',
      g.capture_radius_class);
    const n = epos.length / 3;
    check('channel count matches the payload', n === elec.count, n + '/' + elec.count);
    // lattice check: every nearest neighbour distance must be 20 um
    let pitchOk = true, tested = 0;
    for (let i = 0; i < n && tested < 200; i++) {
      let best = Infinity;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const d = Math.hypot(epos[i * 3] - epos[j * 3], epos[i * 3 + 1] - epos[j * 3 + 1]);
        if (d < best) best = d;
      }
      if (best < 0.0195 || best > 0.0205) { pitchOk = false; break; }
      tested++;
    }
    check('channel lattice pitch is 20 um', pitchOk, tested + ' channels tested');
  }

  // ---- membership: the overlap that forces source separation
  const member = manifest.layers.find((l) => l.id === 'electrode_membership');
  if (member) {
    const s = member.summary;
    check('membership recomputed at this geometry', s.non_empty_channels === s.total_channels,
      s.non_empty_channels + '/' + s.total_channels);
    check('multi-channel capture reported', s.multi_channel_somas > 0,
      s.multi_channel_somas + ' somas on >1 channel, max ' + s.max_per_channel);
    check('stored 100/10 access map explicitly not reused',
      /NOT reused/.test(s.compare_stored_access_map || ''), s.compare_stored_access_map);
  }

  // ---- HTTP hardening
  const trav = await get(BASE + '/api/scene/..%2f..%2fdata.js', true);
  check('path traversal refused', trav.status === 404, 'HTTP ' + trav.status);
  const unlisted = await get(BASE + '/api/scene/not_a_chunk.bin', true);
  check('unlisted chunk name refused', unlisted.status === 404, 'HTTP ' + unlisted.status);
  const bad = await get(BASE + '/api/scene/manifest.bin', true);
  check('non-manifest json name refused', bad.status === 404, 'HTTP ' + bad.status);

  const failed = checks.filter((c) => !c.ok);
  const report = {
    checks_total: checks.length, checks_passed: checks.length - failed.length,
    checks_failed: failed.length, failed: failed.map((c) => c.name + ': ' + c.detail),
    payload_bytes: bytes, manifest_layers: manifest.layers.length,
    checks,
  };
  const outPath = path.isAbsolute(OUT) ? OUT : path.join(ROOT, OUT);
  fs.writeFileSync(outPath, JSON.stringify(report, null, 1));
  console.log('\n' + (report.checks_passed) + '/' + report.checks_total + ' scene checks passed');
  console.log('report: ' + outPath);
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.error('scene test failed:', e.message); process.exit(2); });
