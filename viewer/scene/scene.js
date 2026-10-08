/* Single-scene layers: one body frame, one clock, everything layered in place.
 *
 * Loads viewer/scene/manifest.json plus its binary chunks and attaches each
 * layer to the same three.js scene the body uses. Nothing here creates a
 * second page, a second renderer or a second clock: layers with their own
 * clock are labelled and advanced independently, everything else follows the
 * body replay.
 *
 * All values come from the built payload; this file invents no numbers.
 */
(function () {
  'use strict';
  const shell = window.Workbench || (window.Workbench = {});
  const state = {ready: false, manifest: null, layers: {}, errors: [], visible: {}};
  shell.scene = state;

  const el = (tag, text, cls) => {
    const e = document.createElement(tag);
    if (text != null) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  };

  async function fetchJSON(url) {
    const r = await fetch(url, {cache: 'no-store'});
    if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
    return r.json();
  }

  async function fetchBuf(url) {
    const r = await fetch(url, {cache: 'no-store'});
    if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
    return r.arrayBuffer();
  }

  function decode(chunk, buffer) {
    const map = {
      float32: Float32Array, uint8: Uint8Array, uint16: Uint16Array,
      uint32: Uint32Array, uint64: BigUint64Array,
    };
    const Ctor = map[chunk.dtype];
    if (!Ctor) throw new Error('unsupported dtype ' + chunk.dtype);
    const a = new Ctor(buffer);
    if (chunk.shape && chunk.shape.length > 1) {
      const expected = chunk.shape.reduce((x, y) => x * y, 1);
      if (a.length !== expected) {
        throw new Error(chunk.dtype + ' length ' + a.length + ' != declared ' + expected);
      }
    }
    return a;
  }

  function waitForViewer(timeoutMs) {
    return new Promise((resolve, reject) => {
      const t0 = Date.now();
      (function poll() {
        if (window.viewerTest) return resolve(window.viewerTest);
        if (Date.now() - t0 > timeoutMs) return reject(new Error('viewer never became ready'));
        setTimeout(poll, 100);
      })();
    });
  }

  const COLORMAPS = {
    activity: (t) => [0.15 + 0.85 * t, 0.35 + 0.45 * Math.sin(Math.PI * t), 1.0 - 0.85 * t],
    force: (t) => [0.2 + 0.8 * t, 0.9 - 0.6 * t, 0.35],
    membership: (t) => [0.35, 0.55 + 0.4 * t, 0.95],
  };

  function colorAt(name, t) {
    const f = COLORMAPS[name] || COLORMAPS.activity;
    const c = f(Math.max(0, Math.min(1, t)));
    return [c[0], c[1], c[2]];
  }

  function makePoints(T, positions, sizeMm, color, opacity, renderOrder) {
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.BufferAttribute(positions, 3));
    const c = new Float32Array(positions.length);
    for (let i = 0; i < positions.length; i += 3) {
      c[i] = color[0]; c[i + 1] = color[1]; c[i + 2] = color[2];
    }
    g.setAttribute('color', new T.BufferAttribute(c, 3));
    const m = new T.PointsMaterial({
      size: sizeMm, vertexColors: true, transparent: true, opacity: opacity,
      depthWrite: false, sizeAttenuation: true,
    });
    const p = new T.Points(g, m);
    p.renderOrder = renderOrder;
    return p;
  }

  // ---------------------------------------------------------------- layers
  function attachBodyTwin(T, vt, name) {
    const g = new T.Group();
    vt.rig.add(g);
    state.layers[name] = {group: g};
    return g;
  }

  async function loadCNS(T, vt, manifest, group) {
    const byId = Object.fromEntries(manifest.layers.map((l) => [l.id, l]));
    const soma = byId.cns_somas, edges = byId.cns_edges, activity = byId.cns_activity;
    const posChunk = manifest.chunks['somas_pos.bin'];
    const posBuf = await fetchBuf('/api/scene/somas_pos.bin');
    const pos = decode(posChunk, posBuf);          // Float32Array, body-local mm
    const n = pos.length / 3;

    const subclass = decode(manifest.chunks['somas_class.bin'],
      await fetchBuf('/api/scene/somas_class.bin'));
    const rootIds = decode(manifest.chunks['somas_root_id.bin'],
      await fetchBuf('/api/scene/somas_root_id.bin'));

    const labels = soma.label_tables;
    // size is a real scale cue: a fly soma is ~3-5 um across, so 0.03 mm
    // diameter keeps the cloud readable without leaving the body frame
    const somaPoints = makePoints(T, pos, 0.03, colorAt('activity', 0.2), 0.9, 30);
    somaPoints.name = 'cns-somas';
    group.add(somaPoints);

    // edges: sampled synapses, drawn as segments between the two subset somas
    let edgeLines = null;
    if (edges) {
      const idx = decode(manifest.chunks['edges_index.bin'],
        await fetchBuf('/api/scene/edges_index.bin'));
      const syn = decode(manifest.chunks['edges_syn.bin'],
        await fetchBuf('/api/scene/edges_syn.bin'));
      const seg = new Float32Array((idx.length / 2) * 6);
      const segColor = new Float32Array((idx.length / 2) * 6);
      let k = 0;
      for (let e = 0; e < idx.length; e += 2) {
        const a = idx[e], b = idx[e + 1];
        seg[k] = pos[a * 3]; seg[k + 1] = pos[a * 3 + 1]; seg[k + 2] = pos[a * 3 + 2];
        seg[k + 3] = pos[b * 3]; seg[k + 4] = pos[b * 3 + 1]; seg[k + 5] = pos[b * 3 + 2];
        const t = Math.min(1, (syn[e >> 1] || 1) / 20);
        const c = colorAt('membership', t);
        for (let s = 0; s < 2; s++) {
          segColor[k + s * 3] = c[0]; segColor[k + s * 3 + 1] = c[1]; segColor[k + s * 3 + 2] = c[2];
        }
        k += 6;
      }
      const g = new T.BufferGeometry();
      g.setAttribute('position', new T.BufferAttribute(seg, 3));
      g.setAttribute('color', new T.BufferAttribute(segColor, 3));
      edgeLines = new T.LineSegments(g, new T.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.18, depthWrite: false,
      }));
      edgeLines.renderOrder = 25;
      group.add(edgeLines);
    }

    // activity frames: quantised response, expanded to per-vertex colour
    let actData = null, actScale = null, actColors = null;
    if (activity) {
      actData = decode(manifest.chunks['activity_u8.bin'],
        await fetchBuf('/api/scene/activity_u8.bin'));
      actScale = decode(manifest.chunks['activity_scale.bin'],
        await fetchBuf('/api/scene/activity_scale.bin'));
      actColors = somaPoints.geometry.attributes.color;
      if (actData.length !== n * activity.frames) {
        throw new Error('activity frame count disagrees with soma count');
      }
    }

    const color = somaPoints.geometry.attributes.color;

    state.layers.cns = {
      group, n, pos, subclass, rootIds, labels,
      somaPoints, edgeLines, actData, actScale, color,
      update(frame) {
        if (!actData) return;
        const f = Math.min(frame, activity.frames - 1);
        const base = f * n, lo = actScale[f * 2], span = actScale[f * 2 + 1] || 1;
        const arr = color.array;
        for (let i = 0; i < n; i++) {
          const t = (actData[base + i] / 255) * span + lo;
          // log-ish compression keeps sparse responders visible next to peaks
          const u = Math.min(1, Math.log1p(t * 20) / Math.log1p(20) + 0.04);
          const c = colorAt('activity', u);
          arr[i * 3] = c[0]; arr[i * 3 + 1] = c[1]; arr[i * 3 + 2] = c[2];
        }
        color.needsUpdate = true;
      },
      describe(index) {
        if (index == null || index < 0 || index >= n) return null;
        const s = subclass[index * 3], fl = subclass[index * 3 + 1], bp = subclass[index * 3 + 2];
        return {
          root_id: String(rootIds[index]),
          super_class: labels.super_class[s],
          flow: labels.flow[fl],
          body_part: labels.body_part[bp],
        };
      },
    };
  }

  async function loadLegs(T, vt, manifest, group) {
    const byId = Object.fromEntries(manifest.layers.map((l) => [l.id, l]));
    const leg = byId.leg_contacts;
    if (!leg) return;
    const siteIndex = decode(manifest.chunks['leg_site_index.bin'],
      await fetchBuf('/api/scene/leg_site_index.bin'));
    const table = decode(manifest.chunks['leg_contact_table.bin'],
      await fetchBuf('/api/scene/leg_contact_table.bin'));
    const present = decode(manifest.chunks['leg_present.bin'],
      await fetchBuf('/api/scene/leg_present.bin'));
    const force = decode(manifest.chunks['leg_force.bin'],
      await fetchBuf('/api/scene/leg_force.bin'));
    const dir = decode(manifest.chunks['leg_force_dir.bin'],
      await fetchBuf('/api/scene/leg_force_dir.bin'));
    const cpg = decode(manifest.chunks['leg_cpg.bin'],
      await fetchBuf('/api/scene/leg_cpg.bin'));

    const nLegs = leg.leg_names.length, frames = siteIndex.length / nLegs;
    const markerGeo = new T.SphereGeometry(0.05, 8, 6);
    const markers = [];
    for (let i = 0; i < nLegs; i++) {
      const m = new T.Mesh(markerGeo, new T.MeshBasicMaterial({color: 0x8fe3ff, transparent: true, opacity: 0.9}));
      m.renderOrder = 32;
      group.add(m);
      markers.push(m);
    }
    const arrowGeo = new T.BufferGeometry();
    arrowGeo.setAttribute('position', new T.BufferAttribute(new Float32Array(nLegs * 6), 3));
    const arrows = new T.LineSegments(arrowGeo, new T.LineBasicMaterial({color: 0xffd478, transparent: true, opacity: 0.9}));
    arrows.renderOrder = 33;
    group.add(arrows);

    // CPG phase rings: one small torus per leg, rotated by the recorded phase
    const ringGeo = new T.TorusGeometry(0.075, 0.009, 6, 18);
    const rings = [];
    for (let i = 0; i < nLegs; i++) {
      const r = new T.Mesh(ringGeo, new T.MeshBasicMaterial({color: 0x7fe3a0, transparent: true, opacity: 0.4}));
      r.renderOrder = 32;
      group.add(r);
      rings.push(r);
    }

    let maxForce = 0;
    for (let i = 0; i < force.length; i++) maxForce = Math.max(maxForce, force[i]);
    state.layers.legs = {
      nLegs, frames, maxForce,
      update(frame) {
        const f = Math.min(frame, frames - 1);
        const a = arrows.geometry.attributes.position;
        for (let i = 0; i < nLegs; i++) {
          const si = siteIndex[f * nLegs + i];
          const inside = si < table.length / 3 - 1;
          const on = present[f * nLegs + i] && inside;
          markers[i].visible = !!on;
          rings[i].visible = true;
          const px = table[si * 3], py = table[si * 3 + 1], pz = table[si * 3 + 2];
          if (on) markers[i].position.set(px, py, pz);
          rings[i].position.set(on ? px : 0, on ? py : 0, (on ? pz : 0) + 0.05);
          rings[i].rotation.z = cpg[(f * nLegs + i) * 2];
          rings[i].material.opacity = 0.25 + 0.5 * Math.min(1, cpg[(f * nLegs + i) * 2 + 1]);
          const mag = force[f * nLegs + i];
          const scale = maxForce > 0 ? Math.min(1, mag / maxForce) : 0;
          const tip = [px + dir[(f * nLegs + i) * 3] * 0.25 * scale,
                       py + dir[(f * nLegs + i) * 3 + 1] * 0.25 * scale,
                       pz + dir[(f * nLegs + i) * 3 + 2] * 0.25 * scale];
          a.setXYZ(i * 2, px, py, pz);
          a.setXYZ(i * 2 + 1, tip[0], tip[1], tip[2]);
        }
        a.needsUpdate = true;
      },
    };
  }

  async function loadElectrodes(T, vt, manifest, group) {
    const byId = Object.fromEntries(manifest.layers.map((l) => [l.id, l]));
    const spec = byId.electrodes, member = byId.electrode_membership;
    if (!spec) return;
    const posChunk = manifest.chunks['electrode_pos.bin'];
    const pos = decode(posChunk, await fetchBuf('/api/scene/electrode_pos.bin'));
    const counts = member
      ? decode(manifest.chunks['electrode_counts.bin'], await fetchBuf('/api/scene/electrode_counts.bin'))
      : null;
    const n = pos.length / 3;
    let maxCount = 1;
    if (counts) for (let i = 0; i < counts.length; i++) maxCount = Math.max(maxCount, counts[i]);

    const points = makePoints(T, pos, 0.05, colorAt('membership', 0.4), 0.95, 34);
    points.name = 'electrodes-scene';
    group.add(points);
    if (counts) {
      const attr = points.geometry.attributes.color;
      for (let i = 0; i < n; i++) {
        const c = colorAt('membership', Math.min(1, counts[i] / maxCount));
        attr.array[i * 3] = c[0]; attr.array[i * 3 + 1] = c[1]; attr.array[i * 3 + 2] = c[2];
      }
      attr.needsUpdate = true;
    }
    const radiusMm = (spec.geometry.diameter_um / 1000) / 2;
    state.layers.electrodes = {
      n, points, counts, maxCount, radiusMm,
      describe(index) {
        if (index == null || index < 0 || index >= n) return null;
        return {
          channel: index,
          position_local_mm: [pos[index * 3], pos[index * 3 + 1], pos[index * 3 + 2]],
          somas_within_capture_radius: counts ? counts[index] : null,
          capture_radius_um: spec.geometry.capture_radius_um,
          capture_radius_class: spec.geometry.capture_radius_class,
          diameter_um: spec.geometry.diameter_um,
          pitch_um: spec.geometry.pitch_um,
        };
      },
    };
  }

  async function loadCalcium(T, vt, manifest, group) {
    const layer = manifest.layers.find((l) => l.id === 'calcium_epithelium');
    if (!layer) return;
    const pos = decode(manifest.chunks['calcium_pos.bin'],
      await fetchBuf('/api/scene/calcium_pos.bin'));
    const act = decode(manifest.chunks['calcium_u8.bin'],
      await fetchBuf('/api/scene/calcium_u8.bin'));
    const n = pos.length / 3;
    // the patch is a flat sheet: rotate it out of the xy plane so it lies on the
    // abdomen surface instead of standing on edge
    const flat = new Float32Array(pos.length);
    for (let i = 0; i < n; i++) {
      flat[i * 3] = pos[i * 3];
      flat[i * 3 + 1] = pos[i * 3 + 2];        // model z (0) -> body y
      flat[i * 3 + 2] = -pos[i * 3 + 1];       // model y -> body -z
    }
    const points = makePoints(T, flat, 0.02, [1, 0.35, 0.3], 0.95, 31);
    points.name = 'calcium-patch';
    group.add(points);
    const color = points.geometry.attributes.color;
    state.layers.calcium = {
      n, points, frames: layer.frames,
      extent_um: layer.extent_um,
      update(frame) {
        const f = Math.min(frame, layer.frames - 1);
        const base = f * n, arr = color.array;
        for (let i = 0; i < n; i++) {
          const t = act[base + i] / 255;
          arr[i * 3] = 0.25 + 0.75 * Math.min(1, Math.log1p(t * 8) / Math.log1p(8));
          arr[i * 3 + 1] = 0.75 * (1 - t) * 0.6 + 0.05;
          arr[i * 3 + 2] = 0.30 * (1 - t);
        }
        color.needsUpdate = true;
      },
    };
  }

  async function loadSimFields(T, vt, manifest, groups) {
    const byId = Object.fromEntries(manifest.layers.map((l) => [l.id, l]));

    // ---- paracrine ligand field: one value per grid cell, per frame ----
    const pl = byId.paracrine_field;
    if (pl) {
      const pos = decode(manifest.chunks['paracrine_pos.bin'],
        await fetchBuf('/api/scene/paracrine_pos.bin'));
      const fr = decode(manifest.chunks['paracrine_frames.bin'],
        await fetchBuf('/api/scene/paracrine_frames.bin'));
      const n = pos.length / 3;
      const points = makePoints(T, pos, 0.012, [0.4, 0.6, 1.0], 0.9, 31);
      points.name = 'paracrine-field';
      groups.paracrine.add(points);
      const color = points.geometry.attributes.color;
      state.layers.paracrine = {
        n, frames: pl.frames, points,
        update(frame) {
          const f = Math.min(frame, pl.frames - 1);
          const base = f * n, arr = color.array;
          for (let i = 0; i < n; i++) {
            const u = fr[base + i] / 255;
            arr[i * 3] = 0.15 + 0.85 * u;
            arr[i * 3 + 1] = 0.10 + 0.55 * Math.pow(u, 2.0);
            arr[i * 3 + 2] = 1.0 - 0.75 * u;
          }
          color.needsUpdate = true;
        },
        describe: null,
      };
    }

    // ---- immune debris and attractant, with agent positions ----
    const il = byId.immune_field;
    if (il) {
      const pos = decode(manifest.chunks['immune_pos.bin'],
        await fetchBuf('/api/scene/immune_pos.bin'));
      const deb = decode(manifest.chunks['immune_debris.bin'],
        await fetchBuf('/api/scene/immune_debris.bin'));
      const att = decode(manifest.chunks['immune_attractant.bin'],
        await fetchBuf('/api/scene/immune_attractant.bin'));
      const cells = decode(manifest.chunks['immune_cells.bin'],
        await fetchBuf('/api/scene/immune_cells.bin'));
      const n = pos.length / 3;
      const nCells = il.model ? cells.length / (il.frames * 3) : 0;
      const debrisPoints = makePoints(T, pos, 0.014, [0.9, 0.5, 0.2], 0.85, 31);
      debrisPoints.name = 'immune-debris';
      groups.immune.add(debrisPoints);
      const attractantPoints = makePoints(T, pos, 0.020, [0.6, 0.9, 0.5], 0.30, 30);
      attractantPoints.name = 'immune-attractant';
      groups.immune.add(attractantPoints);
      const agentGeo = new T.SphereGeometry(0.010, 8, 6);
      const agents = new T.Group();
      const meshes = [];
      for (let i = 0; i < nCells; i++) {
        const m = new T.Mesh(agentGeo, new T.MeshBasicMaterial({color: 0xfff0a0}));
        m.renderOrder = 33;
        agents.add(m);
        meshes.push(m);
      }
      groups.immune.add(agents);
      const dColor = debrisPoints.geometry.attributes.color;
      const aColor = attractantPoints.geometry.attributes.color;
      state.layers.immune = {
        n, nCells, frames: il.frames, debrisPoints, attractantPoints, agents,
        update(frame) {
          const f = Math.min(frame, il.frames - 1);
          const base = f * n, da = dColor.array, aa = aColor.array;
          for (let i = 0; i < n; i++) {
            const u = deb[base + i] / 255;
            da[i * 3] = 0.25 + 0.72 * u;
            da[i * 3 + 1] = 0.16 + 0.30 * u;
            da[i * 3 + 2] = 0.08;
            const v = att[base + i] / 255;
            aa[i * 3] = 0.25 + 0.45 * v;
            aa[i * 3 + 1] = 0.45 + 0.45 * v;
            aa[i * 3 + 2] = 0.25 + 0.35 * v;
          }
          dColor.needsUpdate = true;
          aColor.needsUpdate = true;
          for (let i = 0; i < meshes.length; i++) {
            const k = (f * nCells + i) * 3;
            meshes[i].position.set(cells[k], cells[k + 1], cells[k + 2]);
          }
        },
      };
    }
  }

  async function loadTissue3D(T, vt, manifest, group) {
    const layer = manifest.layers.find((l) => l.id === 'tissue_3d');
    if (!layer) return;
    const pos = decode(manifest.chunks['tissue3d_pos.bin'],
      await fetchBuf('/api/scene/tissue3d_pos.bin'));
    const edges = decode(manifest.chunks['tissue3d_edges.bin'],
      await fetchBuf('/api/scene/tissue3d_edges.bin'));
    const layerIdx = decode(manifest.chunks['tissue3d_layers.bin'],
      await fetchBuf('/api/scene/tissue3d_layers.bin'));
    const n = pos.length / 3;
    const nLayers = layer.model ? layer.model.n_layers : 1;

    // points, coloured by layer index so the stack reads as layers
    const points = makePoints(T, pos, 0.006, [0.5, 0.8, 1.0], 0.95, 32);
    points.name = 'tissue3d-vertices';
    const col = points.geometry.attributes.color;
    for (let i = 0; i < n; i++) {
      const u = nLayers > 1 ? layerIdx[i] / (nLayers - 1) : 0;
      col.array[i * 3] = 0.25 + 0.7 * u;
      col.array[i * 3 + 1] = 0.85 - 0.5 * u;
      col.array[i * 3 + 2] = 1.0 - 0.6 * u;
    }
    col.needsUpdate = true;
    group.add(points);

    // vertical connectors between consecutive layers
    const seg = new Float32Array((edges.length / 2) * 6);
    for (let e = 0; e < edges.length; e += 2) {
      const a = edges[e], b = edges[e + 1];
      seg[e * 3] = pos[a * 3]; seg[e * 3 + 1] = pos[a * 3 + 1]; seg[e * 3 + 2] = pos[a * 3 + 2];
      seg[e * 3 + 3] = pos[b * 3]; seg[e * 3 + 4] = pos[b * 3 + 1];
      seg[e * 3 + 5] = pos[b * 3 + 2];
    }
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.BufferAttribute(seg, 3));
    const lines = new T.LineSegments(g, new T.LineBasicMaterial({
      color: 0x8fe3ff, transparent: true, opacity: 0.35, depthWrite: false,
    }));
    lines.renderOrder = 31;
    group.add(lines);

    state.layers.tissue3d = {
      n, points, lines, vertices: n, edges: edges.length / 2, nLayers,
      deviationGain: layer.model ? layer.model.deviation_gain : null,
    };
  }

  async function loadMechanics(T, vt, manifest, group) {
    const layer = manifest.layers.find((l) => l.id === 'mechanics_patch');
    if (!layer) return;
    const pos = decode(manifest.chunks['mech_pos.bin'],
      await fetchBuf('/api/scene/mech_pos.bin'));
    const nV = layer.count, nF = layer.frames;
    const geometry = new T.BufferGeometry();
    geometry.setAttribute('position', new T.BufferAttribute(pos.slice(0, nV * 3), 3));
    const points = new T.Points(geometry, new T.PointsMaterial({
      size: 0.004, color: 0xff9a6a, transparent: true, opacity: 0.9,
      depthWrite: false, sizeAttenuation: true,
    }));
    points.renderOrder = 31;
    points.name = 'mechanics-vertices';
    group.add(points);
    // the cut edge: draw the boundary as a line so the wound is visible
    // NaN IS A MEANINGFUL VALUE HERE: the source arrays mark removed/dead cells with NaN
    // (measured: 14 of 1248 vertices per frame).  Those vertices must not be drawn, so they
    // are moved far outside the view rather than left at a stale position.
    const FAR = 1e6;
    state.layers.mechanics = {
      nV, nF, points,
      deadPerFrame: (() => {
        const out = [];
        for (let f = 0; f < nF; f++) {
          let c = 0;
          for (let i = 0; i < nV; i++) {
            const k = (f * nV + i) * 3;
            if (!Number.isFinite(pos[k]) || !Number.isFinite(pos[k + 1])
                || !Number.isFinite(pos[k + 2])) c++;
          }
          out.push(c);
        }
        return out;
      })(),
      update(frame) {
        const f = Math.min(frame, nF - 1);
        const attr = points.geometry.attributes.position;
        const arr = attr.array;
        for (let i = 0; i < nV; i++) {
          const k = (f * nV + i) * 3, d = i * 3;
          const x = pos[k], y = pos[k + 1], z = pos[k + 2];
          if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
            arr[d] = x; arr[d + 1] = y; arr[d + 2] = z;
          } else {
            arr[d] = FAR; arr[d + 1] = FAR; arr[d + 2] = FAR;
          }
        }
        attr.needsUpdate = true;
      },
    };
  }

  async function loadEnvironment(T, vt, manifest, group) {
    const layer = manifest.layers.find((l) => l.id === 'environment');
    if (!layer) return;
    const sizeMm = 24.0;
    const loader = new T.TextureLoader();
    const tex = await new Promise((resolve) => {
      loader.load('/api/scene/' + layer.file, (t) => resolve(t), undefined, () => resolve(null));
    });
    if (tex) {
      tex.wrapS = T.RepeatWrapping;
      tex.wrapT = T.RepeatWrapping;
      tex.repeat.set(8, 8);
    }
    const geo = new T.PlaneGeometry(sizeMm, sizeMm);
    const mat = new T.MeshBasicMaterial({
      map: tex || null, color: tex ? 0x9fb0a8 : 0x2a3330,
      transparent: true, opacity: 0.85, depthWrite: false, side: T.DoubleSide,
    });
    const ground = new T.Mesh(geo, mat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(0, 0, -0.02);
    ground.renderOrder = -10;
    ground.name = 'ground-plane';
    group.add(ground);
    state.layers.environment = {ground, texture: !!tex, sizeMm};
  }

  function scenarioReadout(manifest) {
    const layer = manifest.layers.find((l) => l.id === 'scenarios');
    if (!layer || !layer.series) return null;
    const arms = layer.arms || [];
    const keys = ['eye_mV', 'brain_mV', 'neck_proximal_mV', 'body_mV', 'artifact_recorded_mV'];
    const box = el('div', null, 'hud-phys');
    box.appendChild(el('div', '断颈/眼损五臂（mV，逐帧）', 'hud-sub'));
    const n = layer.frames || 1;
    const table = el('div', null, 'hud-mono');
    // per-frame values are written into this element by the tick loop
    box.appendChild(table);
    box.appendChild(el('div', '比较臂：' + arms.join(' / '), 'hud-mono'));
    return {
      box,
      update(frame) {
        const f = Math.min(frame, n - 1);
        // compact: three significant digits and a fixed field width, because the first
        // version printed full precision and the row ran off the panel
        const short = (k) => k.replace('_recorded_mV', '(伪迹)').replace('_mV', '')
          .replace('neck_proximal', '颈近').replace('neck_distal', '颈远')
          .replace('eye', '眼').replace('brain', '脑').replace('body', '体')
          .replace('stimulus_source_nA', '刺激');
        const lines = [];
        keys.forEach((k) => {
          const s = layer.series[k];
          if (!s) return;
          const shown = arms.map((a) => {
            const v = s[a] ? s[a][f] : null;
            if (v == null) return '    —';
            const a2 = Math.abs(v);
            if (a2 >= 100) return (v < 0 ? '-' : '+') + a2.toFixed(0).padStart(4);
            if (a2 >= 1) return (v < 0 ? '-' : '+') + a2.toFixed(1).padStart(4);
            if (a2 >= 0.01) return (v < 0 ? '-' : '+') + a2.toFixed(2).padStart(4);
            return (v === 0 ? ' 0.00' : (v < 0 ? '-' : '+') + a2.toExponential(1).padStart(4));
          });
          lines.push(short(k).padEnd(4, ' ') + ' ' + shown.join(' '));
        });
        table.textContent = lines.join('\n');
      },
    };
  }

  function readoutPanel(manifest) {
    const layer = manifest.layers.find((l) => l.id === 'layer_readouts');
    if (!layer || !layer.values) return null;
    const v = layer.values;
    const box = el('div', null, 'hud-phys');
    box.appendChild(el('div', '无空间场的层（读数）', 'hud-sub'));
    if (v.cellspace) {
      const c = v.cellspace;
      box.appendChild(el('div', '胞内空间：D_Ca ' + c.D_ca_um2_s + ' µm²/s，守恒偏差 ' +
        Number(c.uniform_preservation_max_dev).toExponential(2), 'hud-mono'));
      if (c.convergence_orders) {
        box.appendChild(el('div', '  收敛阶 ' +
          c.convergence_orders.map((o) => 'L∞ ' + o.order_linf.toFixed(3)).join(' / ') +
          '；大 D 极限收敛 ' + c.large_D_lumped_converged, 'hud-mono'));
      }
    }
    if (v.vascular_oxygen) {
      const rows = v.vascular_oxygen.rows || [];
      box.appendChild(el('div', '血管氧（按间距的统计量，无空间场）：', 'hud-mono'));
      rows.forEach((r) => box.appendChild(el('div', '  ' + r.spacing_um + ' µm 间距：均 ' +
        Number(r.mean_mmHg).toFixed(1) + ' mmHg，最低 ' + Number(r.min_mmHg).toFixed(1) +
        '，缺氧占比 ' + Number(r.hypoxic_fraction).toFixed(3), 'hud-mono')));
    }
    const vv = (k, label) => {
      const e = v[k];
      if (!e) return;
      const s = e.verdicts;
      const txt = s ? JSON.stringify(s) : ('tests passed ' + e.tests_passed);
      box.appendChild(el('div', label + '：' + txt, 'hud-mono'));
    };
    vv('frontend', '电极前端判定');
    vv('stimulation', '刺激判定');
    vv('recording', '记录链测试');
    return {box};
  }

  // ------------------------------------------------------------------ HUD
  function buildHUD(manifest, superseded) {
    const host = document.getElementById('sceneHud') || (() => {
      const d = el('div');
      d.id = 'sceneHud';
      document.body.appendChild(d);
      return d;
    })();
    host.innerHTML = '';

    // live readouts first: the values that have no spatial coordinates are the
    // ones a viewer checks most often, so they are not buried under the layer list
    const readout = el('div', null, 'hud-readout');
    readout.id = 'hudReadout';
    host.appendChild(readout);

    const phys = manifest.layers.find((l) => l.id === 'physiology_ledgers');
    if (phys && phys.values) {
      const v = phys.values;
      const o2 = v.oxygen_nmol || {};
      const tre = v.trehalose_nmol || {};
      const f = (x, d) => (x == null ? '—' : Number(x).toFixed(d == null ? 4 : d));
      const box = el('div', null, 'hud-phys');
      box.appendChild(el('div', '活体读数 · 同一场景', 'hud-sub'));
      box.appendChild(el('div', '气管 O₂ ' + f(o2.initial, 3) + ' + ' + f(o2.inflow, 3) + ' − ' +
        f(o2.consumption, 3) + ' = ' + f(o2.amount_now, 3) + ' nmol', 'hud-mono'));
      box.appendChild(el('div', 'O₂ 残差 ' + (o2.residual == null ? '—' : Number(o2.residual).toExponential(2)) +
        ' nmol · 海藻糖 ' + f(tre.amount_now, 3) + ' nmol（独立）', 'hud-mono'));
      if (v.epochs) box.appendChild(el('div', '行为段 ' + v.epochs.length + ' 段 · 睡眠' +
        (v.sleep_implemented ? '已实现' : '未实现（仅静息）'), 'hud-mono'));
      host.appendChild(box);
    }

    const sc = scenarioReadout(manifest);
    if (sc) host.appendChild(sc.box);
    const ro = readoutPanel(manifest);
    if (ro) host.appendChild(ro.box);

    const details = document.createElement('details');
    details.className = 'hud-layers';
    const summary = el('summary', '图层（' + manifest.layers.length + '）· 同一坐标系 · 同一时钟');
    details.appendChild(summary);
    const list = el('div', null, 'hud-list');
    manifest.layers.forEach((layer) => {
      const row = el('label', null, 'hud-row');
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = true;
      box.dataset.layer = layer.id;
      box.onchange = () => setLayerVisible(layer.id, box.checked);
      row.appendChild(box);
      row.appendChild(el('span', layer.label_zh, 'hud-label'));
      const bytes = (layer.chunks || []).reduce((s, c) => s + (manifest.chunks[c] ? manifest.chunks[c].bytes : 0), 0);
      row.appendChild(el('span', bytes ? (bytes / 1024).toFixed(0) + ' KB' : '读数', 'hud-bytes'));
      list.appendChild(row);
    });
    details.appendChild(list);
    host.appendChild(details);

    const warn = el('div', null, 'hud-warn');
    warn.textContent = manifest.notices.join(' · ');
    host.appendChild(warn);

    if (superseded && superseded.length) {
      host.appendChild(el('div', '已隐藏的旧图层：' + superseded.join('、'), 'hud-audit'));
    }
    const audit = manifest.transform_audit;
    if (audit) {
      host.appendChild(el('div', '配准：标志点 · 刚性 · 残差 ' +
        audit.max_pairwise_distance_error_mm.toExponential(2) + ' mm', 'hud-audit'));
    }
    return {host, readout, scenarios: sc ? {updateFrame: sc.update} : null};
  }

  function setLayerVisible(id, on) {
    state.visible[id] = on;
    const map = {
      cns_somas: state.layers.cns && state.layers.cns.somaPoints,
      cns_edges: state.layers.cns && state.layers.cns.edgeLines,
      cns_activity: state.layers.cns && state.layers.cns.somaPoints,
      leg_contacts: state.layers.legs && state.layers.legs.markers,
      electrodes: state.layers.electrodes && state.layers.electrodes.points,
      calcium_epithelium: state.layers.calcium && state.layers.calcium.points,
      paracrine_field: state.layers.paracrine && state.layers.paracrine.points,
      immune_field: state.layers.immune && state.layers.immune.debrisPoints,
      mechanics_patch: state.layers.mechanics && state.layers.mechanics.points,
      tissue_3d: state.layers.tissue3d && state.layers.tissue3d.points,
      electrode_membership: state.layers.electrodes && state.layers.electrodes.points,
    };
    const target = map[id];
    if (target) target.visible = on;
    if (id === 'leg_contacts' && state.layers.legs) {
      state.layers.legs.group.visible = on;
    }
    return on;
  }

  // -------------------------------------------------------------- driver
  async function boot() {
    try {
      const manifest = await fetchJSON('/api/scene/manifest');
      state.manifest = manifest;
      const vt = await waitForViewer(20000);
      const T = window.THREE;
      if (!T) throw new Error('THREE missing');

      const cnsGroup = attachBodyTwin(T, vt, 'cns');
      const legGroup = attachBodyTwin(T, vt, 'legcontacts');
      const elecGroup = attachBodyTwin(T, vt, 'device');

      await loadCNS(T, vt, manifest, cnsGroup);
      await loadLegs(T, vt, manifest, legGroup);
      await loadElectrodes(T, vt, manifest, elecGroup);
      const caGroup = attachBodyTwin(T, vt, 'epithelium');
      const paraGroup = attachBodyTwin(T, vt, 'paracrine');
      const immGroup = attachBodyTwin(T, vt, 'immune');
      const t3Group = attachBodyTwin(T, vt, 'tissue3d');
      const mechGroup = attachBodyTwin(T, vt, 'mechanics');
      const envGroup = attachBodyTwin(T, vt, 'environment');
      await loadCalcium(T, vt, manifest, caGroup);
      await loadSimFields(T, vt, manifest, {paracrine: paraGroup, immune: immGroup});
      await loadTissue3D(T, vt, manifest, t3Group);
      await loadMechanics(T, vt, manifest, mechGroup);
      await loadEnvironment(T, vt, manifest, envGroup);

      // De-duplicate: the legacy layers drew an ILLUSTRATIVE neuron cloud and a
      // 100/10 um electrode block. Both are now superseded by the real CNS cloud
      // and the fixed 7/20 um array drawn here, so the legacy objects are hidden
      // rather than deleted (they stay one click away in the layer panel).
      const superseded = [];
      const legacy = vt.scenePoints || {};
      if (legacy.neural && state.layers.cns) { legacy.neural.visible = false; superseded.push('neural(示意)'); }
      (vt.electrodeObjects || []).forEach((o) => { o.visible = false; });
      if (vt.electrodeObjects && vt.electrodeObjects.length) superseded.push('electrodes(100/10 旧几何)');
      state.superseded = superseded;

      const hud = buildHUD(manifest, superseded);

      // ONE CLOCK: the body replay index drives every layer that shares the body
      // clock. Poses come from the same recorded body frames app.js uses, so the
      // attached layers cannot drift from the body.
      const B = window.VIEWER_DATA.body;
      const frames = B.frames, quats = B.rotations_wxyz;
      const bodyFrames = manifest.clock.frames;
      if (frames.length !== bodyFrames) {
        throw new Error('scene body frames (' + frames.length + ') disagree with manifest (' + bodyFrames + ')');
      }
      const pose = shell.transforms && shell.transforms.pose;
      // The two embedded simulation patches attach to the ABDOMEN node (12), not the
      // thorax: they were laid on the abdomen at true scale, and pinning them to the
      // thorax would make them drift off the body as the fly walks.
      const ABDOMEN_NODE = 12;
      let last = -1;
      function applyPose(frameIndex) {
        if (!pose) return;
        const p = frames[frameIndex];
        const q = quats[frameIndex];
        [cnsGroup, legGroup, elecGroup].forEach((g) => {
          pose(g, p && p[1], q && q[1]);
        });
        [caGroup, paraGroup, immGroup, t3Group, mechGroup].forEach((g) => {
          pose(g, p && p[ABDOMEN_NODE], q && q[ABDOMEN_NODE]);
        });
      }
      function tick() {
        requestAnimationFrame(tick);
        const vtNow = window.viewerTest;
        if (!vtNow) return;
        const frame = Math.min(bodyFrames - 1, Math.max(0, Math.round(vtNow.progress * (bodyFrames - 1))));
        if (frame === last) return;
        last = frame;
        applyPose(frame);
        if (state.layers.cns) state.layers.cns.update(frame);
        if (state.layers.legs) state.layers.legs.update(frame);
        if (state.layers.calcium) state.layers.calcium.update(frame);
        if (state.layers.paracrine) state.layers.paracrine.update(frame);
        if (state.layers.immune) state.layers.immune.update(frame);
        if (state.layers.mechanics) state.layers.mechanics.update(frame);
        if (hud.scenarios) hud.scenarios.updateFrame(frame);
        const seg = '帧 ' + frame + '/' + (bodyFrames - 1) + ' · t=' +
          (manifest.clock.times_s[frame] || 0).toFixed(2) + ' s';
        if (hud.readout) hud.readout.textContent = seg;
      }
      // Framing helpers. The legacy focus() derives its frame from the old
      // illustrative point cloud, which no longer exists, so the CNS and the
      // electrodes need their own aim points inside the same scene.
      const T3 = window.THREE;
      const centre = (objects) => {
        const box = new T3.Box3();
        objects.filter(Boolean).forEach((o) => box.expandByObject(o));
        if (box.isEmpty()) return null;
        const sphere = box.getBoundingSphere(new T3.Sphere());
        return sphere;
      };
      const aim = (sphere, pad) => {
        if (!sphere) return null;
        const vtNow = window.viewerTest;
        if (!vtNow || !vtNow.orbit || !vtNow.cameraUpdate) return null;
        const cam = vtNow.camera;
        const half = Math.min(T3.MathUtils.degToRad(cam.fov / 2),
          Math.atan(Math.tan(T3.MathUtils.degToRad(cam.fov / 2)) * cam.aspect));
        // drive the viewer's own orbit instead of the camera directly: the
        // render loop recomputes the camera from the orbit every frame, so a
        // direct camera move would be overwritten immediately
        vtNow.orbit.target.copy(sphere.center);
        vtNow.orbit.distance = Math.max(0.002, (sphere.radius / Math.sin(half)) * (pad || 1.25));
        vtNow.cameraUpdate();
        return {center: sphere.center.toArray(), radius: sphere.radius, distance: vtNow.orbit.distance};
      };
      // Publish the live scene objects so the viewer's own focus()/camera-follow
      // can frame them: the legacy focus bounds come from the superseded
      // illustrative cloud, which no longer exists.
      shell.scene.focusTargets = () => [state.layers.cns && state.layers.cns.somaPoints,
        state.layers.cns && state.layers.cns.edgeLines, state.layers.electrodes && state.layers.electrodes.points,
        state.layers.calcium && state.layers.calcium.points].filter(Boolean);

      shell.scene.focusScene = (what) => {
        const l = state.layers;
        if (what === 'cns') return aim(centre([l.cns && l.cns.somaPoints]), 1.15);
        if (what === 'electrodes') return aim(centre([l.electrodes && l.electrodes.points]), 1.6);
        if (what === 'calcium') return aim(centre([l.calcium && l.calcium.points]), 2.0);
        if (what === 'everything') {
          const vtNow = window.viewerTest;
          return vtNow && vtNow.focus ? vtNow.focus('all') : null;
        }
        return null;
      };
      shell.scene.sceneBounds = () => {
        const l = state.layers;
        return {
          cns: (() => { const s = centre([l.cns && l.cns.somaPoints]); return s && {center: s.center.toArray(), radius: s.radius}; })(),
          electrodes: (() => { const s = centre([l.electrodes && l.electrodes.points]); return s && {center: s.center.toArray(), radius: s.radius}; })(),
          calcium: (() => { const s = centre([l.calcium && l.calcium.points]); return s && {center: s.center.toArray(), radius: s.radius}; })(),
        };
      };

      applyPose(0);
      tick();
      state.ready = true;
      window.dispatchEvent(new CustomEvent('workbench-scene-ready', {detail: {manifest}}));
    } catch (err) {
      state.errors.push(String(err && err.message ? err.message : err));
      window.dispatchEvent(new CustomEvent('workbench-scene-error', {detail: {error: String(err)}}));
    }
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
