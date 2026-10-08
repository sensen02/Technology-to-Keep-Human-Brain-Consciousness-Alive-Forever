"""Real FlyWire presynaptic site coordinates mapped onto the validated FAFB subtree.

WHAT THIS MODULE DOES
---------------------
1. `discover_products` / `probe_header` — uses the SAME Codex download-resource
   API pattern as `flywire_download.py` and `flywire_probe_resource.py`
   (`/api/download_resource?data_product=...&dataset=fafb&api_token=...`) to find
   and inspect the FAFB synapse table.  The api_token is read from
   `.flywire_api_token` and is NEVER logged, echoed or written to any output.
2. `stream_filter` — downloads the gzipped synapse table ONCE and decompresses
   it as a stream, keeping only rows whose presynaptic root id is in a small
   bounded target set.  The full table is 2.695 GB gzipped (~10 GB of CSV) and
   is never materialised in memory or on disk by this code path.
3. `assign_nearest_node` — maps each real presynaptic site (x, y, z, nanometres)
   onto the skeleton node with minimal Euclidean distance, using an actual
   distance computation (scipy.spatial.cKDTree over the skeleton's own
   coordinates).  Distances are reported; sites beyond `far_um` are flagged.
4. `compartment_current` — builds the per-compartment inward-current vector (nA)
   that `engine.cable.CableNeuron.step` consumes, for a prescribed distribution
   of N real presynaptic sites over morphology compartments.
5. `distribute_sites` — splits a count of N sites across compartments.

HONESTY / LIMITS (also written into every output file)
------------------------------------------------------
* Real coordinates are DATA (measured presynaptic site centroids from FAFB).
* The compartment assignment is an APPROXIMATION: nearest-skeleton-node
  assignment, not a reconstruction of which compartment a synapse truly formed
  on.  FAFB coordinates are in the fixed FAFB (nm) frame; the skeleton is the
  same frame, so no registration is involved — but the retained subtree is
  TRUNCATED, so real synapses near excluded nodes fall outside it.
* No synaptic physiology is measured: g_max, tau and E_rev are illustrative
  assumptions, not fitted to this cell.
* Cable passive parameters are borrowed from DNp01/DNp03 (PMC11071487).
* No claim about neural coding, consciousness or viability.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "CODEX_API", "SYNAPSE_PRODUCT", "FLYWIRE_LABEL", "DEFAULT_FAFB_DATASET",
    "api_token", "probe_header", "stream_filter", "load_sites_npz", "Sites",
    "assign_nearest_node", "map_sites_to_nodes", "truncation_accounting",
    "truncation_boundary_accounting",
    "distribution", "distribute_sites", "compartment_current",
    "event_conductance", "explicit_stability_limit", "solve_implicit_synaptic",
    "discrete_release_schedule", "placement_matrix", "project_schedule",
    "schedule_totals",
    "run_drive", "pairwise_response_compare", "sha256_file", "read_swc_rows",
    "json_ready",
]

CODEX_API = "https://codex.flywire.ai/api/download_resource"
#: The data product that actually carries presynaptic site coordinates.
#: Verified by live probe against the Codex download-resource API on 2026-02-14:
#: HTTP 200, Content-Length 2_695_106_039 bytes, gzip magic 1f 8b, and the header
#: row `pre_x,pre_y,pre_z,ctr_x,ctr_y,ctr_z,post_x,post_y,post_z,size,`
#: `pre_root_id_720575940,post_root_id_720575940,neuropil`.
SYNAPSE_PRODUCT = "synapse_table"
DEFAULT_FAFB_DATASET = "fafb"
FLYWIRE_LABEL = "720575940"          # the FAFB root-id label baked into the header
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

EXPECTED_COLUMNS = ("pre_x", "pre_y", "pre_z", "size",
                    "pre_root_id_720575940", "post_root_id_720575940")


# --------------------------------------------------------------------------- #
# credentials                                                                  #
# --------------------------------------------------------------------------- #
def api_token(root=None, filename=".flywire_api_token"):
    """Read the Codex api_token from disk.  Never log or return it in output."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = filename if os.path.isabs(filename) else os.path.join(root, filename)
    with open(path) as fh:
        token = fh.read().strip()
    if not token:
        raise ValueError("empty FlyWire api_token file")
    return token


def _request(data_product, dataset=DEFAULT_FAFB_DATASET, root=None,
             timeout=180):
    """Build the Codex download-resource request (token redacted in __repr__)."""
    token = api_token(root)
    url = CODEX_API + "?" + urllib.parse.urlencode(
        {"data_product": data_product, "dataset": dataset, "api_token": token})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    # Keep the full URL away from tracebacks / logs.
    req.redacted_url = re.sub(r"api_token=[^&]*", "api_token=<redacted>", url)
    return req


# --------------------------------------------------------------------------- #
# 1. discovery                                                                 #
# --------------------------------------------------------------------------- #
def probe_header(product=SYNAPSE_PRODUCT, dataset=DEFAULT_FAFB_DATASET,
                 root=None, sample_bytes=4 << 20, timeout=180):
    """Probe a data product and return its CSV header + declared size.

    Streams (and discards) only `sample_bytes` of decompressed CSV, so no large
    object is ever held.  Raises RuntimeError with the HTTP status if the
    product is not available.
    """
    req = _request(product, dataset, root, timeout)
    out = {"data_product": product, "dataset": dataset,
           "url_redacted": req.redacted_url, "sample_bytes": sample_bytes}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out["http_status"] = resp.status
            out["content_length_bytes"] = (
                int(resp.headers["Content-Length"])
                if resp.headers.get("Content-Length") else None)
            out["content_type"] = resp.headers.get("Content-Type")
            out["content_disposition"] = resp.headers.get("Content-Disposition")
            head = resp.read(4)
            out["gzip_magic"] = head[:2].hex() == "1f8b"
            if not out["gzip_magic"]:
                raise RuntimeError(
                    f"{product}: response is not gzip (first bytes {head!r})")
            stream = _ChainedReader(head + resp.read(), resp)
            with gzip.GzipFile(fileobj=stream) as gz:
                raw = gz.read(sample_bytes)
    except urllib.error.HTTPError as exc:
        body = exc.read()[:200].decode("utf-8", "replace")
        raise RuntimeError(f"{product}: HTTP {exc.code} {exc.reason} -> {body}") from None
    text = raw.decode("utf-8", "replace")
    lines = text.split("\n")
    out["header_line"] = lines[0].rstrip("\r")
    out["header_columns"] = out["header_line"].split(",")
    out["sample_rows"] = [ln for ln in lines[1:6] if ln.strip()]
    out["elapsed_s"] = time.time() - t0
    out["has_presynaptic_coordinates"] = all(
        c in out["header_columns"] for c in ("pre_x", "pre_y", "pre_z"))
    out["probe_note"] = ("coordinates are the FAFB nm frame; header streams the "
                         "whole connection so only the first %d bytes were read"
                         % sample_bytes)
    return out


class _ChainedReader:
    """Small read-only file-like wrapper so gzip can start on the peeked bytes."""

    def __init__(self, prefix, resp):
        self._prefix = memoryview(prefix)
        self._resp = resp
        self._pos = 0

    def read(self, n=-1):
        if self._pos < len(self._prefix):
            chunk = self._prefix[self._pos:self._pos + (n if n >= 0 else len(self._prefix))]
            self._pos += len(chunk)
            return bytes(chunk)
        return self._resp.read(n if n >= 0 else 1 << 20)

    def readable(self):
        return True


# --------------------------------------------------------------------------- #
# 2. bounded streaming filter                                                  #
# --------------------------------------------------------------------------- #
def _open_stream(product, dataset, root, timeout):
    req = _request(product, dataset, root, timeout)
    resp = urllib.request.urlopen(req, timeout=timeout)
    magic = resp.read(4)
    if magic[:2] != b"\x1f\x8b":
        resp.close()
        raise RuntimeError(f"{product}: not a gzip stream ({magic!r})")
    return resp, gzip.GzipFile(fileobj=_ChainedReader(magic, resp))


def stream_filter(product=SYNAPSE_PRODUCT, dataset=DEFAULT_FAFB_DATASET,
                  pre_root_ids=(), post_root_ids=(), any_root_ids=(),
                  root=None, timeout=600, progress_every_s=20.0, log=None,
                  max_rows=None, dest_npz=None, label=None):
    """Stream the synapse table, keep only rows matching the bounded id set.

    Memory is bounded by the number of MATCHING rows: every non-matching row is
    parsed as a fixed number of comma fields and immediately dropped; the
    decompressed table is never stored.

    Returns (matches, stats).  `matches` holds the retained pre/post ids,
    coordinates and synapse size; `stats` holds row counts, timing and bytes.
    """
    targets = {"pre": set(int(i) for i in pre_root_ids),
               "post": set(int(i) for i in post_root_ids),
               "any": set(int(i) for i in any_root_ids)}
    if not any(targets.values()):
        raise ValueError("at least one of pre_root_ids/post_root_ids/any_root_ids required")
    log = log or (lambda *_: None)
    t0 = time.time()
    resp, gz = _open_stream(product, dataset, root, timeout)
    out = {k: [] for k in ("pre_x", "pre_y", "pre_z", "ctr_x", "ctr_y", "ctr_z",
                           "post_x", "post_y", "post_z", "size",
                           "pre_root_id", "post_root_id")}
    neuropil_label = []
    stats = {"data_product": product, "dataset": dataset,
             "pre_root_ids": sorted(targets["pre"]),
             "post_root_ids": sorted(targets["post"]),
             "any_root_ids": sorted(targets["any"]),
             "rows_scanned": 0, "rows_matched": 0,
             "malformed_rows": 0, "declared_size_bytes": None,
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    try:
        stats["declared_size_bytes"] = (
            int(resp.headers["Content-Length"])
            if resp.headers.get("Content-Length") else None)
        header_seen = False
        columns = []
        pending = b""
        read_bytes = 0
        last = t0
        while True:
            block = gz.read(1 << 22)
            if not block:
                break
            read_bytes += len(block)
            pending += block
            *complete, pending = pending.split(b"\n")
            if max_rows is not None and stats["rows_scanned"] >= max_rows:
                break
            for line in complete:
                stats["rows_scanned"] += 1
                if max_rows is not None and stats["rows_scanned"] > max_rows:
                    break
                if not header_seen:
                    columns = line.decode("utf-8", "replace").rstrip("\r").split(",")
                    header_seen = True
                    missing = [c for c in EXPECTED_COLUMNS if c not in columns]
                    if missing:
                        raise RuntimeError(
                            f"{product}: header lacks expected columns {missing}; "
                            f"actual header = {columns}")
                    stats["header_columns"] = columns
                    continue
                fields = line.decode("utf-8", "replace").rstrip("\r").split(",")
                if len(fields) != len(columns):
                    stats["malformed_rows"] += 1
                    continue
                pre = int(fields[columns.index("pre_root_id_720575940")])
                post = int(fields[columns.index("post_root_id_720575940")])
                if not ((pre in targets["pre"]) or (post in targets["post"])
                        or (pre in targets["any"]) or (post in targets["any"])):
                    continue
                out["pre_x"].append(float(fields[columns.index("pre_x")]))
                out["pre_y"].append(float(fields[columns.index("pre_y")]))
                out["pre_z"].append(float(fields[columns.index("pre_z")]))
                out["ctr_x"].append(float(fields[columns.index("ctr_x")]))
                out["ctr_y"].append(float(fields[columns.index("ctr_y")]))
                out["ctr_z"].append(float(fields[columns.index("ctr_z")]))
                out["post_x"].append(float(fields[columns.index("post_x")]))
                out["post_y"].append(float(fields[columns.index("post_y")]))
                out["post_z"].append(float(fields[columns.index("post_z")]))
                out["size"].append(int(float(fields[columns.index("size")])))
                out["pre_root_id"].append(pre)
                out["post_root_id"].append(post)
                neuropil_label.append(fields[columns.index("neuropil")])
                stats["rows_matched"] += 1
            now = time.time()
            if now - last > progress_every_s:
                last = now
                log(f"  scanned {stats['rows_scanned']:,} rows, "
                    f"{stats['rows_matched']:,} matched, "
                    f"{read_bytes/1e9:.2f} GB csv, {now-t0:.0f}s")
            if max_rows is not None and stats["rows_scanned"] >= max_rows:
                break
    finally:
        gz.close()
        resp.close()
    matches = {k: np.asarray(v, dtype=(np.int64 if "id" in k or k == "size" else np.float64))
               for k, v in out.items()}
    # Neuropil labels are stored as integer codes + a label list so the npz stays
    # plain-numeric (no pickle, no object arrays).
    labels, codes = np.unique(np.asarray(neuropil_label, dtype=str), return_inverse=True)
    matches["neuropil_code"] = codes.astype(np.int64)
    matches["neuropil_labels"] = labels
    matches["retained_neuropil"] = np.asarray(neuropil_label, dtype=str)
    stats["neuropil_labels"] = list(map(str, labels))
    stats["csv_bytes_read"] = read_bytes
    stats["elapsed_s"] = time.time() - t0
    stats["elapsed_wall_clock_s"] = stats["elapsed_s"]
    stats["raw_input_bytes"] = stats["csv_bytes_read"]
    stats["matching_rows"] = stats["rows_matched"]
    if dest_npz:
        np.savez_compressed(dest_npz, **{k: v for k, v in matches.items()})
        stats["npz"] = str(dest_npz)
        stats["npz_bytes"] = os.path.getsize(dest_npz)
    return matches, stats


def load_sites_npz(path):
    """Load a `stream_filter` npz back into a Sites object (plain-numeric npz)."""
    with np.load(path, allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files}
    return Sites.from_fields(fields, source=str(path))


# --------------------------------------------------------------------------- #
# 3. nearest-node assignment + flagging                                        #
# --------------------------------------------------------------------------- #
class Sites:
    """A bounded set of real presynaptic sites, in FAFB nanometres."""

    __slots__ = ("xyz_nm", "pre_root_id", "post_root_id", "size", "neuropil",
                 "index", "source")

    def __init__(self, xyz_nm, pre_root_id=None, post_root_id=None, size=None,
                 neuropil=None, index=None, source="unknown"):
        xyz = np.asarray(xyz_nm, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz_nm must be (N, 3)")
        if not np.isfinite(xyz).all():
            raise ValueError("xyz_nm must be finite")
        n = len(xyz)
        self.xyz_nm = xyz
        self.pre_root_id = _idvec(pre_root_id, n, "pre_root_id")
        self.post_root_id = _idvec(post_root_id, n, "post_root_id")
        self.size = (_idvec(size, n, "size") if size is not None
                     else np.ones(n, dtype=np.int64))
        self.neuropil = (np.asarray(neuropil, dtype=object) if neuropil is not None
                         else np.asarray([""] * n, dtype=object))
        if len(self.neuropil) != n:
            raise ValueError("neuropil must have one entry per site")
        self.index = (_idvec(index, n, "index") if index is not None
                      else np.arange(n, dtype=np.int64))
        self.source = str(source)

    def __len__(self):
        return len(self.xyz_nm)

    @classmethod
    def from_fields(cls, fields, source="npz", column_prefix="pre"):
        def get(*names, default=None):
            for nm in names:
                if nm in fields:
                    return fields[nm]
            return default
        coords = get(f"{column_prefix}_x", f"{column_prefix}_x_nm")
        if coords is None:
            raise ValueError(f"no {column_prefix}_x column in fixture")
        xyz = np.column_stack([np.asarray(fields[f"{column_prefix}_{c}"], float)
                               for c in "xyz"])
        neuropil = get("retained_neuropil", "neuropil")
        if neuropil is None and "neuropil_code" in fields and "neuropil_labels" in fields:
            neuropil = np.asarray(fields["neuropil_labels"], dtype=str)[
                np.asarray(fields["neuropil_code"], dtype=np.int64)]
        return cls(xyz,
                   pre_root_id=get("pre_root_id"),
                   post_root_id=get("post_root_id"),
                   size=get("size"),
                   neuropil=neuropil,
                   index=get("index"),
                   source=source)

    @classmethod
    def synthetic(cls, xyz_nm, **kw):
        return cls(xyz_nm, source=kw.pop("source", "synthetic fixture"), **kw)

    def select_root(self, root_id, column="pre_root_id"):
        """Exact-integer filter: ONLY rows whose chosen column equals root_id."""
        ids = self.pre_root_id if column == "pre_root_id" else self.post_root_id
        keep = np.flatnonzero(ids == int(root_id))
        return Sites(self.xyz_nm[keep], self.pre_root_id[keep],
                     self.post_root_id[keep], self.size[keep],
                     self.neuropil[keep], self.index[keep], self.source)

    def as_fields(self):
        return {
            "pre_x": self.xyz_nm[:, 0], "pre_y": self.xyz_nm[:, 1],
            "pre_z": self.xyz_nm[:, 2], "size": self.size,
            "pre_root_id": self.pre_root_id, "post_root_id": self.post_root_id,
            "neuropil": self.neuropil, "index": self.index,
        }


def _idvec(values, n, name):
    if values is None:
        return np.zeros(n, dtype=np.int64)
    a = np.asarray(values, dtype=np.int64)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f"{name} must have one entry per site")
    return a


def assign_nearest_node(sites_nm, morph, *, far_um=2.0, chunk=20000,
                        node_ids_nm=None, node_ids=None):
    """Nearest skeleton node for every site; distances measured, not assumed.

    Parameters
    ----------
    sites_nm : Sites or (N, 3) array of FAFB nanometre coordinates.
    morph    : engine.cable.Morphology (coordinates in um).
    far_um   : distances above this flag the site as far from every node.
    node_ids_nm : optional (n,) array of layer-1 / node xyz in nanometres; when
        omitted the morphology coordinates are converted um -> nm.

    Returns a dict with `node` (row index), `node_id` (SWC id when available),
    `distance_um`, `distance_nm`, `far` (bool mask), `far_ids` (indices into the
    site list) and full distance statistics.
    """
    xyz = sites_nm.xyz_nm if isinstance(sites_nm, Sites) else np.asarray(sites_nm, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("sites must be (N, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError("site coordinates must be finite")
    if not np.isfinite(far_um) or far_um <= 0:
        raise ValueError("far_um must be positive and finite")
    n = morph.n
    if node_ids_nm is None:
        nodes_nm = np.column_stack([morph.x, morph.y, morph.z]) * 1000.0
    else:
        nodes_nm = np.asarray(node_ids_nm, float)
        if nodes_nm.shape != (n, 3):
            raise ValueError("node coordinate array must be (n, 3)")
    if not np.isfinite(nodes_nm).all():
        raise ValueError("node coordinates must be finite")
    tree = cKDTree(nodes_nm)
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    node = np.empty(len(xyz), dtype=np.int64)
    dist_nm = np.empty(len(xyz), dtype=float)
    for start in range(0, len(xyz), chunk):
        d, i = tree.query(xyz[start:start + chunk], k=1,
                          distance_upper_bound=np.inf)
        node[start:start + chunk] = i
        dist_nm[start:start + chunk] = d
    dist_um = dist_nm / 1000.0
    far = dist_um > far_um
    out = {
        "node": node, "distance_um": dist_um, "distance_nm": dist_nm,
        "far": far, "far_ids": np.flatnonzero(far),
        "n_sites": int(len(xyz)), "n_nodes": int(n),
        "far_threshold_um": float(far_um),
        "n_far": int(far.sum()),
        "assignment": "nearest skeleton node by Euclidean distance in the shared FAFB nm frame",
    }
    if node_ids is not None:
        ids = np.asarray(node_ids)
        if len(ids) != n:
            raise ValueError("node_ids must have one entry per morphology row")
        out["node_swc_id"] = ids[node]
    if len(xyz):
        out["distance_stats_um"] = {
            "min": float(dist_um.min()), "max": float(dist_um.max()),
            "mean": float(dist_um.mean()), "median": float(np.median(dist_um)),
            "p90": float(np.percentile(dist_um, 90)),
            "p99": float(np.percentile(dist_um, 99)),
            "n_over_1um": int((dist_um > 1.0).sum()),
            "n_over_5um": int((dist_um > 5.0).sum()),
        }
    else:
        out["distance_stats_um"] = None
    # truncation accounting: how many sites land on retained vs excluded nodes
    out["n_on_retained_subtree"] = int((~far).sum())
    return out


def map_sites_to_nodes(sites_nm, morph, node_ids_nm, node_swc_ids, *, far_um=2.0):
    """Convenience wrapper returning the assignment plus per-row match flags."""
    result = assign_nearest_node(sites_nm, morph, far_um=far_um,
                                 node_ids_nm=node_ids_nm, node_ids=node_swc_ids)
    return result


def truncation_accounting(assignment, excluded_swc_ids, node_swc_ids):
    """Split the mapped sites into retained-subtree vs excluded-node matches.

    A site is counted on the EXCLUDED nodes when its nearest retained node's
    original SWC id is in `excluded_swc_ids`; every other mapped site is on the
    retained subtree.  Sites flagged `far` are reported separately because their
    nearest retained node is not a credible placement.
    """
    ids = np.asarray(node_swc_ids)
    excluded = set(int(i) for i in excluded_swc_ids)
    nearest = np.asarray(assignment.get("node_swc_id", np.full(len(ids), -1)))
    on_excluded = np.isin(nearest, sorted(excluded))
    far = np.asarray(assignment["far"])
    return {
        "n_sites": int(len(nearest)),
        "n_mapped_to_retained_nodes": int((~on_excluded).sum()),
        "n_mapped_to_excluded_nodes": int(on_excluded.sum()),
        "n_flagged_far": int(far.sum()),
        "n_retained_and_close": int(((~on_excluded) & (~far)).sum()),
        "excluded_node_count": len(excluded),
    }


def truncation_boundary_accounting(assignment, boundary_retained_swc_ids,
                                   excluded_nodes_nm, sites_nm):
    """How close the real sites come to the artificial truncation boundary.

    `boundary_retained_swc_ids` are the retained nodes whose parent/child edge to
    an excluded source node was replaced by an artificial sealed boundary.  Sites
    that map onto those nodes sit immediately at the cut; sites whose nearest
    excluded source node is closer than every retained node are the ones the
    truncation plausibly removed.
    """
    ids = np.asarray(assignment["node_swc_id"])
    boundary = set(int(i) for i in boundary_retained_swc_ids)
    on_boundary = np.isin(ids, sorted(boundary)) if boundary else np.zeros(len(ids), bool)
    xyz = sites_nm.xyz_nm if isinstance(sites_nm, Sites) else np.asarray(sites_nm, float)
    out = {
        "n_boundary_retained_nodes": len(boundary),
        "n_sites_on_a_boundary_node": int(on_boundary.sum()),
        "boundary_retained_swc_ids": sorted(boundary),
        "note": ("a site on a boundary node lies exactly at an artificial sealed cut; it is "
                 "inside the retained subtree, but its original edge to the excluded node is gone"),
    }
    if excluded_nodes_nm is not None and len(np.asarray(excluded_nodes_nm)):
        from scipy.spatial import cKDTree
        d_ex, _ = cKDTree(np.asarray(excluded_nodes_nm, float)).query(xyz, k=1)
        out["n_sites_closer_to_an_excluded_node"] = int(
            (d_ex < np.asarray(assignment["distance_nm"])).sum())
        out["min_distance_to_an_excluded_node_um"] = float(np.min(d_ex) / 1000.0)
    return out


# --------------------------------------------------------------------------- #
# 4. distribution over compartments                                            #
# --------------------------------------------------------------------------- #
def distribute_sites(n_sites, weights):
    """Distribute an integer site count across compartments by largest remainder.

    `weights` are nonnegative compartment weights (e.g. mapped real site counts).
    The returned integer vector sums EXACTLY to n_sites and is deterministic.
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)) or n_sites < 0:
        raise ValueError("n_sites must be a nonnegative integer")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a nonempty 1-D sequence")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    if w.sum() <= 0:
        if n_sites:
            raise ValueError("cannot distribute sites with all-zero weights")
        return np.zeros(w.size, dtype=np.int64)
    exact = n_sites * w / w.sum()
    counts = np.floor(exact).astype(np.int64)
    remainder = n_sites - int(counts.sum())
    if remainder:
        # deterministic: largest fractional remainder, ties by lowest index
        order = np.lexsort((np.arange(w.size), -(exact - counts)))
        counts[order[:remainder]] += 1
    assert counts.sum() == n_sites
    return counts


def distribution(site_nodes, n_compartments=None, weights=None, n_sites=None):
    """Count real sites per compartment row.

    `site_nodes` is the per-site assigned morphology row index.  Returns
    (counts, total) where counts[row] is the number of real sites on that row.
    """
    nodes = np.asarray(site_nodes, dtype=np.int64)
    if nodes.ndim != 1:
        raise ValueError("site_nodes must be 1-D")
    if nodes.size and (nodes.min() < 0):
        raise ValueError("site_nodes must be nonnegative row indices")
    size = int(n_compartments if n_compartments is not None
               else (nodes.max() + 1 if nodes.size else 0))
    if nodes.size and nodes.max() >= size:
        raise ValueError("site_nodes exceed the compartment count")
    counts = np.bincount(nodes, minlength=size).astype(np.int64)
    if weights is None:
        w = counts.astype(float)
    else:
        w = np.asarray(weights, float)
        if w.shape != counts.shape:
            raise ValueError("weights must match the compartment count")
    counts_out = counts if n_sites is None else distribute_sites(n_sites, w)
    return counts_out, int(n_sites if n_sites is not None else counts.sum())


# --------------------------------------------------------------------------- #
# 5. conductance events -> per-compartment current vector                      #
# --------------------------------------------------------------------------- #
def compartment_current(per_compartment_conductance_uS, v_mV, E_rev_mV,
                        extra_scale=1.0):
    """Inward current vector (nA) for `CableNeuron.step(i_syn=...)`.

    CableNeuron treats positive as INWARD, and an ohmic synaptic conductance
    contributes g*(E_rev - V).  The vector returned here is
    `sum over synapses on that compartment of g_syn * (E_rev - v)`, i.e. it
    SUMS exactly to sum(g)*E_rev - sum(g*v) over all sites, so the intended
    total drive is preserved by construction.
    """
    g = np.asarray(per_compartment_conductance_uS, dtype=float)
    v = np.asarray(v_mV, dtype=float)
    if g.ndim != 1:
        raise ValueError("conductance must be a 1-D per-compartment vector")
    if v.shape != g.shape:
        raise ValueError("voltage vector must match the conductance vector")
    if not np.isfinite(g).all() or np.any(g < 0):
        raise ValueError("conductances must be finite and nonnegative")
    if not np.isfinite(v).all():
        raise ValueError("voltage must be finite")
    if not np.isfinite(E_rev_mV):
        raise ValueError("reversal potential must be finite")
    if not np.isfinite(extra_scale) or extra_scale < 0:
        raise ValueError("extra_scale must be finite and nonnegative")
    return extra_scale * g * (float(E_rev_mV) - v)


def event_conductance(g_per_site_uS, events_per_compartment, g_decay=None):
    """Conductance per compartment (uS) from integer event counts.

    `g_decay` optionally carries the previous conductance vector to which the
    new events add (exponential-decay handling stays with the caller).
    """
    counts = np.asarray(events_per_compartment, dtype=float)
    if counts.ndim != 1 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("events must be a finite nonnegative per-compartment vector")
    if not np.allclose(counts, np.rint(counts), rtol=0, atol=1e-9):
        raise ValueError("events must be whole numbers")
    counts = np.rint(counts).astype(np.int64)
    if not np.isfinite(g_per_site_uS) or g_per_site_uS < 0:
        raise ValueError("g_per_site_uS must be finite and nonnegative")
    g = counts.astype(float) * float(g_per_site_uS)
    if g_decay is not None:
        prev = np.asarray(g_decay, float)
        if prev.shape != g.shape:
            raise ValueError("g_decay must match the compartment count")
        return prev + g
    return g


def explicit_stability_limit(neuron, per_compartment_sites, safety=1.0):
    """Largest `g_per_site` for which explicit current sampling stays stable.

    CableNeuron.step samples the synaptic current at the PREVIOUS voltage, so a
    compartment with total conductance g over capacitance C is stable only while
    g*dt/C <= 2.  With n sites on one compartment the per-site conductance must
    therefore satisfy g_site <= safety * min_over_occupied(2*C/(dt*n)).
    Returns None when no compartment carries sites.
    """
    n = np.asarray(per_compartment_sites, dtype=float)
    if n.shape != (neuron.m.n,):
        raise ValueError("per_compartment_sites must have one entry per compartment")
    if np.any(n < 0) or not np.isfinite(n).all():
        raise ValueError("per-compartment site counts must be finite and nonnegative")
    occupied = n > 0
    if not np.any(occupied):
        return None
    if not np.isfinite(safety) or safety <= 0:
        raise ValueError("safety must be positive and finite")
    limit = 2.0 * neuron.C[occupied] / (neuron.dt * n[occupied])
    return float(safety * limit.min())


def solve_implicit_synaptic(neuron, g_uS, i_inject=None):
    """One backward-Euler step with the synaptic conductance INSIDE the solve.

    `C/dt (v+ - v) = I_inject + g*(E_rev - v+) - G v+` is solved exactly for the
    given per-compartment conductance vector, which removes the explicit-sampling
    instability of `CableNeuron.step(i_syn=...)` while using the same matrices.
    No active channels are introduced; E_rev is linear (ohmic) as in cable.Synapse.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu
    g = np.asarray(g_uS, dtype=float)
    n = neuron.m.n
    if g.shape != (n,) or not np.isfinite(g).all() or np.any(g < 0):
        raise ValueError("g_uS must be a finite nonnegative (n,) vector")
    rhs = neuron.C / neuron.dt * neuron.v + neuron.g_mem * neuron.E_leak + \
        neuron.g_end * neuron.E_end
    if i_inject is not None:
        cur = np.asarray(i_inject, float)
        if cur.shape not in ((), (n,)) or not np.isfinite(cur).all():
            raise ValueError("i_inject must be a finite scalar or (n,) array")
        rhs = rhs + cur
    A = (sparse.diags(neuron.C / neuron.dt) + neuron.G + sparse.diags(g)).tocsc()
    neuron.v = splu(A).solve(rhs)
    neuron.t += neuron.dt
    return neuron.v


def placement_matrix(site_rows, n_compartments, *, mode="mapped", single_row=None):
    """Map each site to its compartment: (n_compartments, n_sites) one-hot matrix.

    mode="mapped"    each site to its own assigned row (REAL placement);
    mode="single"    every site to one arbitrary row.
    """
    rows = np.asarray(site_rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("site_rows must be 1-D")
    if not isinstance(n_compartments, (int, np.integer)) or n_compartments < 1:
        raise ValueError("n_compartments must be a positive integer")
    if mode == "mapped":
        if rows.size and (rows.min() < 0 or rows.max() >= n_compartments):
            raise ValueError("site_rows outside the compartment range")
        target = rows
    elif mode == "single":
        if not isinstance(single_row, (int, np.integer)) or not 0 <= single_row < n_compartments:
            raise ValueError("single_row must be a valid compartment index")
        target = np.full(rows.size, int(single_row), dtype=np.int64)
    else:
        raise ValueError("mode must be 'mapped' or 'single'")
    M = np.zeros((int(n_compartments), rows.size), dtype=np.int64)
    if rows.size:
        M[target, np.arange(rows.size)] = 1
    return M


def discrete_release_schedule(site_weights, *, dt_ms, duration_ms, rate_hz,
                              t_start_ms, t_end_ms, seed):
    """Release counts per SITE per step, with an exactly shared step total.

    Each site releases independently with probability `rate_hz*dt_ms/1000` per
    step, so the step total is Binomial(sum(site_weights), p).  One shared total
    `k` is drawn per step and then split over the sites by a CONDITIONAL
    multinomial draw (the exact conditional distribution of Poisson/multinomial
    counts given their sum).  Arms that take this split with the same seed and
    the same total site weight therefore schedule exactly `k` releases at every
    step, whatever their spatial arrangement -- the `same total synaptic drive`
    condition -- without collapsing the release count variance.
    """
    w = np.asarray(site_weights, dtype=float)
    if w.ndim != 1 or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("site_weights must be a finite nonnegative vector")
    total = int(round(float(w.sum())))
    if abs(float(w.sum()) - total) > 1e-9:
        raise ValueError("site weights must sum to a whole number")
    p = float(rate_hz) * float(dt_ms) / 1000.0
    if not 0.0 <= p <= 1.0:
        raise ValueError("rate_hz * dt_ms must be within one release per step")
    n_steps = int(round(duration_ms / dt_ms))
    first = max(0, int(round(t_start_ms / dt_ms)))
    last = min(int(round(t_end_ms / dt_ms)), n_steps)
    schedule = np.zeros((n_steps, w.size), dtype=np.int64)
    rng = np.random.default_rng(seed)
    positive = w > 0
    n_pos = int(positive.sum())
    for k in range(first, last):
        releases = int(rng.binomial(total, p)) if total else 0
        if releases:
            if n_pos == 1:
                schedule[k, positive] = releases
            else:
                chunks = np.zeros(n_pos, dtype=np.int64)
                remaining = releases
                left = float(w[positive].sum())
                for j, weight in enumerate(w[positive]):
                    if j == n_pos - 1:
                        chunks[j] = remaining
                        break
                    if remaining <= 0 or left <= 0:
                        break
                    q = min(1.0, float(weight) / left)
                    share = int(rng.binomial(remaining, q))
                    chunks[j] = share
                    remaining -= share
                    left -= float(weight)
                schedule[k, positive] = chunks
    return schedule


def project_schedule(site_schedule, placement):
    """Project per-site events onto compartments: (n_steps, n_comp) counts.

    `placement` is (n_compartments, n_sites) 0/1.  The result preserves the total
    number of events at every step exactly, so two arms that share
    `site_schedule` differ only in placement.
    """
    s = np.asarray(site_schedule)
    M = np.asarray(placement)
    if s.ndim != 2 or M.ndim != 2 or s.shape[1] != M.shape[1]:
        raise ValueError("site_schedule (n_steps, n_sites) and placement "
                         "(n_compartments, n_sites) must agree on n_sites")
    if np.any(s < 0) or np.any((M != 0) & (M != 1)):
        raise ValueError("schedule must be nonnegative and placement 0/1")
    return (M.astype(np.int64) @ s.T.astype(np.int64)).T.astype(np.int64)


def schedule_totals(schedule):
    """Total events per step of a schedule (compare across arms for equality)."""
    s = np.asarray(schedule, dtype=np.int64)
    if s.ndim != 2:
        raise ValueError("schedule must be two-dimensional")
    if np.any(s < 0):
        raise ValueError("schedule event counts must be nonnegative")
    return s.sum(axis=1)


def run_drive(morph, *, site_nodes=None, site_weights=None, g_per_site_uS=5e-4,
              E_rev_mV=0.0, tau_ms=2.0, rate_hz=80.0, dt_ms=0.025,
              duration_ms=20.0, t_start_ms=5.0, t_end_ms=15.0, seed=12345,
              record_rows=None, passive=None, implicit=False,
              clamp_to_stability=False, stability_safety=0.5, schedule=None):
    """Drive a CableNeuron at real-synapse compartments; return traces.

    Two calling conventions:
      * `site_nodes` = per-site morphology row index (REAL sites), or
      * `site_weights` = per-compartment weight vector (compartment counts).

    `implicit=True` puts the synaptic conductance inside the linear solve (exact
    backward Euler for that conductance, unconditionally stable).  The default
    `implicit=False` reproduces the existing practice of sampling the current at
    the previous voltage, which is only stable while g*dt/C <= 2; with
    `clamp_to_stability=True` the per-site conductance is reduced to the largest
    value that satisfies that bound and the reduction is reported.

    `schedule` (n_steps, n_compartments) overrides the random event train, which
    is how two arms are given an identical total drive.

    Returns dict with per-step voltage at the recorded rows, the conductance
    trace, the exact total injected charge (nA*ms) and peak current.
    """
    from .cable import CableNeuron
    passive = dict(passive or {})
    neuron = CableNeuron(morph, dt_ms=dt_ms, **passive)
    n = morph.n
    if site_nodes is None and site_weights is None:
        raise ValueError("site_nodes or site_weights required")
    if site_nodes is not None:
        nodes = np.asarray(site_nodes, dtype=np.int64)
        if nodes.ndim != 1 or np.any(nodes < 0) or (nodes.size and nodes.max() >= n):
            raise ValueError("site_nodes must be valid morphology row indices")
        counts = np.bincount(nodes, minlength=n).astype(np.int64)
    else:
        w = np.asarray(site_weights, float)
        if w.shape != (n,) or np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("site_weights must be a nonnegative (n,) vector")
        counts = w
    sites_per_step = counts.astype(np.float64)
    n_sites = int(round(float(sites_per_step.sum())))
    if abs(float(sites_per_step.sum()) - n_sites) > 1e-9:
        raise ValueError("per-compartment site counts must be whole numbers")
    n_steps = int(round(duration_ms / dt_ms))
    if schedule is None:
        int_counts = np.rint(sites_per_step)
        if not np.allclose(int_counts, sites_per_step, rtol=0, atol=1e-9):
            raise ValueError("site_weights must be whole numbers when no schedule is given; "
                             "use distribute_sites for an integer apportionment")
        rows = np.repeat(np.arange(n), int_counts.astype(np.int64))
        schedule = project_schedule(
            discrete_release_schedule(np.ones(rows.size), dt_ms=dt_ms, duration_ms=duration_ms,
                                      rate_hz=rate_hz, t_start_ms=t_start_ms,
                                      t_end_ms=t_end_ms, seed=seed),
            placement_matrix(rows, n, mode="mapped"))
    schedule = np.asarray(schedule, dtype=np.int64)
    if schedule.shape != (n_steps, n):
        raise ValueError(f"schedule must have shape ({n_steps}, {n}), got {schedule.shape}")
    if np.any(schedule < 0):
        raise ValueError("schedule event counts must be nonnegative")
    g_site_requested = float(g_per_site_uS)
    g_site = g_site_requested
    clamp = None
    explicit_peak_g = None
    if clamp_to_stability:
        explicit_peak_g = float(g_site_requested * schedule.max())
        limit = explicit_stability_limit(
            neuron, np.where(schedule.max(axis=0) > 0, schedule.max(axis=0), 0.0),
            safety=stability_safety)
        if limit is not None and g_site > limit:
            g_site = limit
            clamp = {"requested_g_per_site_uS": g_site_requested,
                     "applied_g_per_site_uS": g_site,
                     "safety_factor": float(stability_safety),
                     "reason": ("explicit current sampling is stable only while "
                                "g*dt/C <= 2; the per-site conductance was reduced to "
                                "the largest value satisfying that bound on the most "
                                "loaded compartment")}
    g = np.zeros(n)
    decay = np.exp(-dt_ms / tau_ms)
    record_rows = list(range(n)) if record_rows is None else [int(r) for r in record_rows]
    V = np.zeros((n_steps, len(record_rows)))
    G = np.zeros((n_steps, n))
    charge_nA_ms = 0.0
    peak_current = 0.0
    total_events = 0
    max_voltage = float(abs(neuron.v).max())
    for k in range(n_steps):
        # exact exponential decay of the conductance state, then new events
        g *= decay
        events = schedule[k]
        if np.any(events):
            g += events * g_site
            total_events += int(events.sum())
        i_syn = compartment_current(g, neuron.v, E_rev_mV)
        charge_nA_ms += float(i_syn.sum()) * dt_ms
        peak_current = max(peak_current, float(np.abs(i_syn).max()))
        if implicit:
            v_new = solve_implicit_synaptic(neuron, g)
        else:
            v_new = neuron.step(i_syn=i_syn)
        max_voltage = max(max_voltage, float(np.abs(v_new).max()))
        V[k] = v_new[record_rows]
        G[k] = g
    return {
        "V_mV": V, "G_uS": G, "record_rows": record_rows,
        "n_sites": n_sites, "g_per_site_uS": g_site,
        "g_per_site_requested_uS": g_site_requested, "stability_clamp": clamp,
        "implicit": bool(implicit), "schedule": schedule,
        "peak_synaptic_conductance_uS": float(G.max()),
        "total_events_scheduled": int(schedule.sum()),
        "E_rev_mV": E_rev_mV, "tau_ms": tau_ms, "rate_hz": rate_hz,
        "dt_ms": dt_ms, "duration_ms": duration_ms,
        "t_start_ms": t_start_ms, "t_end_ms": t_end_ms, "seed": int(seed),
        "total_events": total_events, "charge_nA_ms": charge_nA_ms,
        "peak_current_nA": peak_current, "max_abs_voltage_mV": max_voltage,
        "sites_per_compartment": sites_per_step,
        "driver_note": ("events come from an explicit shared schedule; each site "
                        "contributes g_per_site_uS * (E_rev - v) to its "
                        "compartment, so the current vector is the exact sum "
                        "over sites on that compartment"
                        + ("; the conductance is placed inside the backward-Euler "
                           "solve, so no explicit-sampling stability limit applies"
                           if implicit else
                           "; the current is sampled at the previous voltage, which is "
                           "stable only while g*dt/C <= 2")),
    }


def pairwise_response_compare(morph, *, site_nodes, total_sites, single_row,
                              record_rows, **kw):
    """(a) one arbitrary injection point vs (b) real distributed sites.

    Both arms use the SAME morphology, the SAME total synaptic drive
    (`total_sites` sites, same g/tau/E_rev/rate/seed) and differ only in WHERE
    the drive is delivered.  `implicit=True` (in `kw`) puts the synaptic
    conductance inside the cable solve; otherwise the existing explicit
    pre-voltage current sampling is used.
    """
    site_nodes = np.asarray(site_nodes, dtype=np.int64)
    if site_nodes.ndim != 1 or not site_nodes.size:
        raise ValueError("site_nodes must be a nonempty 1-D array")
    if not isinstance(total_sites, (int, np.integer)) or isinstance(total_sites, bool) or total_sites < 1:
        raise ValueError("total_sites must be a positive integer")
    if not isinstance(single_row, (int, np.integer)) or single_row < 0 or single_row >= morph.n:
        raise ValueError("single_row must be a valid morphology row index")
    weights = np.bincount(site_nodes, minlength=morph.n).astype(float)
    real_counts = weights.copy()
    distributed = distribute_sites(int(total_sites), weights)
    # arm (a): the whole bounded drive parked on one arbitrary row
    point = np.zeros(morph.n, dtype=float)
    point[int(single_row)] = int(total_sites)

    kw = dict(kw)
    kw.setdefault("seed", 4242)
    kw.setdefault("dt_ms", 0.025)
    kw.setdefault("duration_ms", 20.0)
    kw.setdefault("rate_hz", 80.0)
    kw.setdefault("t_start_ms", 5.0)
    kw.setdefault("t_end_ms", 15.0)
    # ONE per-site release train shared by both arms (one unit-weight site per
    # real site); each arm only changes WHERE a site sits, so the number of
    # releases at every step is IDENTICAL.
    site_events = discrete_release_schedule(
        np.ones(int(total_sites)), dt_ms=kw["dt_ms"], duration_ms=kw["duration_ms"],
        rate_hz=kw["rate_hz"], t_start_ms=kw["t_start_ms"],
        t_end_ms=kw["t_end_ms"], seed=kw["seed"])
    site_index = np.arange(int(total_sites))
    # point arm: every site on one arbitrary row; distributed arm: site i on the
    # compartment given by the largest-remainder apportionment of the real sites
    dist_rows = np.repeat(np.flatnonzero(distributed), distributed[distributed > 0].astype(np.int64))
    if dist_rows.size != int(total_sites):
        raise RuntimeError("apportionment produced the wrong number of sites")
    placement = {
        "point": placement_matrix(site_index, morph.n, mode="single", single_row=int(single_row)),
        "distributed": placement_matrix(dist_rows, morph.n, mode="mapped"),
    }
    sched = {name: project_schedule(site_events, placement[name]) for name in placement}
    if not np.array_equal(schedule_totals(sched["point"]),
                          schedule_totals(sched["distributed"])):
        raise RuntimeError("internal error: the two arms must schedule identical event counts")
    a = run_drive(morph, site_weights=point, record_rows=record_rows,
                  schedule=sched["point"], **kw)
    b = run_drive(morph, site_weights=distributed, record_rows=record_rows,
                  schedule=sched["distributed"], **kw)
    placement_a = np.flatnonzero(sched["point"].sum(axis=0))
    placement_b = np.flatnonzero(sched["distributed"].sum(axis=0))
    current_vectors_differ = None
    if a["total_events"] and b["total_events"]:
        # compare the injected current PATTERN (peak-conductance step) at rest
        ka = int(np.argmax(a["G_uS"].sum(axis=1)))
        kb = int(np.argmax(b["G_uS"].sum(axis=1)))
        v_rest = float(a["V_mV"][0, 0])
        ia = compartment_current(a["G_uS"][ka], np.full(morph.n, v_rest), a["E_rev_mV"])
        ib = compartment_current(b["G_uS"][kb], np.full(morph.n, v_rest), b["E_rev_mV"])
        current_vectors_differ = bool(np.abs(ia - ib).max() > 0.0)
    rows = list(record_rows)
    va, vb = a["V_mV"], b["V_mV"]
    delta = vb - va
    summary = {
        "morphology_nodes": int(morph.n),
        "real_site_rows": int(real_counts.sum()),
        "total_site_count_both_arms": int(total_sites),
        "single_point_row": int(single_row),
        "compartments_used_by_real_distribution": int(np.count_nonzero(distributed)),
        "n_compartments": int(morph.n),
        "per_step_max_abs_delta_mV": float(np.abs(delta).max()),
        "synaptic_solve": "implicit backward Euler" if kw.get("implicit") else
                          "explicit pre-voltage current sampling",
        "per_row": {},
        "same_total_drive": {
            "scheme": ("one event train per arm from the same seed: identical total event "
                       "count (and therefore identical total synaptic conductance) at every "
                       "timestep; only the placement of each event differs"),
            "point_arm_total_sites": int(point.sum()),
            "distributed_arm_total_sites": int(distributed.sum()),
            "point_arm_charge_nA_ms": a["charge_nA_ms"],
            "distributed_arm_charge_nA_ms": b["charge_nA_ms"],
            "point_arm_events": a["total_events"],
            "distributed_arm_events": b["total_events"],
            "events_identical": a["total_events"] == b["total_events"],
            "event_counts_identical_per_step": bool(np.array_equal(
                schedule_totals(a["schedule"]), schedule_totals(b["schedule"]))),
            "point_placement_rows": placement_a.tolist()[:32],
            "distributed_placement_rows": placement_b.tolist()[:32],
            "placement_differs": bool(placement_a.size != placement_b.size or
                                      not np.array_equal(placement_a, placement_b)),
            "current_vectors_differ_at_rest": current_vectors_differ,
            "point_arm_peak_synaptic_conductance_uS": a["peak_synaptic_conductance_uS"],
            "distributed_arm_peak_synaptic_conductance_uS": b["peak_synaptic_conductance_uS"],
            "point_arm_g_per_site_uS": a["g_per_site_uS"],
            "distributed_arm_g_per_site_uS": b["g_per_site_uS"],
            "same_g_per_site": a["g_per_site_uS"] == b["g_per_site_uS"],
            "stability_clamp_applied": {"point": a["stability_clamp"],
                                        "distributed": b["stability_clamp"]},
        },
        "numerical": {
            "point_arm_max_abs_voltage_mV": a["max_abs_voltage_mV"],
            "distributed_arm_max_abs_voltage_mV": b["max_abs_voltage_mV"],
            "point_arm_peak_current_nA": a["peak_current_nA"],
            "distributed_arm_peak_current_nA": b["peak_current_nA"],
            "finite": bool(np.isfinite(va).all() and np.isfinite(vb).all()),
        },
    }
    for j, row in enumerate(rows):
        summary["per_row"][str(row)] = {
            "point_peak_mV": float(va[:, j].min()),
            "distributed_peak_mV": float(vb[:, j].min()),
            "point_peak_abs_deviation_mV": float(np.abs(va[:, j] - va[0, j]).max()),
            "distributed_peak_abs_deviation_mV": float(np.abs(vb[:, j] - vb[0, j]).max()),
            "max_abs_difference_mV": float(np.abs(delta[:, j]).max()),
            "rest_mV": float(b["V_mV"][0, j]),
        }
    return {"point": a, "distributed": b, "delta_mV": delta,
            "summary": summary,
            "interpretation_note": ("both arms deliver the same number of sites "
                                    "at the same rate with the same g/tau/E_rev "
                                    "and the same RNG seed; only the spatial "
                                    "placement differs")}


# --------------------------------------------------------------------------- #
# provenance helpers                                                           #
# --------------------------------------------------------------------------- #
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_swc_rows(path):
    """Raw seven-column SWC rows (no validation; callers assert what they need)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            body = line.split("#", 1)[0].strip()
            if body:
                rows.append(body.split())
    return rows


def json_ready(obj):
    """Recursively convert numpy scalars/arrays to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
