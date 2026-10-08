#!/usr/bin/env python3
"""Bounded, deterministic audit of local FAFB/BANC evidence.

This reads only central-directory metadata and a few SWCs in-memory; it never
extracts the 13 GB archive.  It deliberately reports evidence and unknowns,
not a neck cut geometry or cross-specimen correspondence.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, os, statistics, zipfile
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "flywire")
OUT = os.path.join(ROOT, "outputs", "brain_isolation")
ZIP_PATH = os.path.join(DATA, "fafb_skeletons_swc.zip")
CLASS_PATH = os.path.join(DATA, "fafb_classification.csv.gz")
FAFB_NEURONS = os.path.join(DATA, "fafb_neurons.csv.gz")
BANC_NEURONS = os.path.join(DATA, "banc_neurons.csv.gz")
CONN_PATH = os.path.join(DATA, "fafb_connectome.npz")


def read_annotations(path, id_field):
    rows = []
    with gzip.open(path, "rt", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            try: row["_id"] = int(row[id_field])
            except (KeyError, ValueError): continue
            rows.append(row)
    return rows


def swc_stats(raw, expected_id):
    text = raw.decode("utf-8", "replace")
    headers = [x.strip() for x in text.splitlines() if x.startswith("#")]
    units = [x for x in headers if "units" in x.lower()]
    points, ids, parents, radii, labels = [], set(), [], [], []
    row_ids = []
    malformed = 0
    for line in text.splitlines():
        if not line or line.startswith("#"): continue
        f = line.split()
        if len(f) < 7: malformed += 1; continue
        try:
            i, lab, x, y, z, r, p = int(f[0]), int(f[1]), *map(float, f[2:6]), int(f[6])
            points.append((i, x, y, z)); row_ids.append(i); ids.add(i); parents.append(p); radii.append(r); labels.append(lab)
        except ValueError: malformed += 1
    child = Counter(parents)
    roots = [i for i,p in zip(row_ids, parents) if p == -1]
    missing_parent = sorted(set(p for p in parents if p != -1 and p not in ids))
    return {"archive_id": expected_id, "header_units_evidence": units[:2],
            "point_count": len(points), "malformed_rows": malformed,
            "root_count": len(roots), "missing_parent_count": len(missing_parent),
            "negative_radius_count": sum(r < 0 for r in radii),
            "radius_min_nm": min(radii) if radii else None,
            "radius_max_nm": max(radii) if radii else None,
            "label_counts": dict(sorted(Counter(labels).items())),
            "coordinate_min_nm": [min(p[j] for p in points) for j in range(1,4)] if points else None,
            "coordinate_max_nm": [max(p[j] for p in points) for j in range(1,4)] if points else None,
            "sha256_sample_bytes": hashlib.sha256(raw).hexdigest()}


def main():
    result = {"audit": "bounded_brain_isolation", "deterministic": True,
              "scope": "FAFB SWC central directory plus three in-memory samples; local metadata only"}
    with zipfile.ZipFile(ZIP_PATH) as z:
        infos = sorted(z.infolist(), key=lambda x: (x.file_size, x.filename))
        result["archive"] = {"path": ZIP_PATH, "members": len(infos),
            "central_directory_total_uncompressed_bytes": sum(x.file_size for x in infos),
            "central_directory_total_compressed_bytes": sum(x.compress_size for x in infos),
            "smallest_member": infos[0].filename, "largest_member": infos[-1].filename,
            "central_directory_crc_metadata_present": all(x.CRC >= 0 for x in infos)}
        # Deterministic small, middle, and largest bounded members.
        picks = [infos[0], infos[len(infos)//2], infos[-1]]
        samples = []
        for info in picks:
            with z.open(info, "r") as fh: raw = fh.read()  # CRC checked by ZipExtFile
            samples.append({"member": info.filename, "compressed_bytes": info.compress_size,
                            "uncompressed_bytes": info.file_size, "declared_crc32": f"{info.CRC:08x}",
                            "read_bytes": len(raw), "stream_read_complete": len(raw) == info.file_size,
                            "swc": swc_stats(raw, info.filename[:-4])})
        result["stream_crc_samples"] = samples

    cls = read_annotations(CLASS_PATH, "root_id")
    cls_by = {r["_id"]: r for r in cls}
    result["fafb_classification"] = {"rows": len(cls), "flow": dict(Counter(r.get("flow", "") for r in cls)),
        "super_class": dict(Counter(r.get("super_class", "") for r in cls)),
        "visual_class_candidates": sum(r.get("class", "").lower() == "visual" or r.get("super_class", "").lower() in {"optic", "visual_projection"} for r in cls),
        "visual_subclass_examples": sorted(Counter(r.get("sub_class", "") for r in cls if r.get("super_class", "").lower() in {"optic", "visual_projection"}).items(), key=lambda x:(-x[1],x[0]))[:12]}
    fafb_neurons = read_annotations(FAFB_NEURONS, "root_id")
    root_ids = set(int(x) for x in __import__('numpy').load(CONN_PATH, allow_pickle=False)["root_ids"])
    class_ids, neuron_ids = set(cls_by), {r["_id"] for r in fafb_neurons}
    result["id_matching"] = {"classification_rows": len(class_ids), "fafb_neuron_rows": len(neuron_ids),
        "connectome_root_ids": len(root_ids), "classification_intersect_connectome": len(class_ids & root_ids),
        "neuron_table_intersect_connectome": len(neuron_ids & root_ids),
        "classification_not_in_connectome": len(class_ids - root_ids),
        "warning": "ID overlap is within FAFB tables; it is not a cross-specimen match."}

    banc = read_annotations(BANC_NEURONS, "Root ID")
    # BANC is used only to nominate annotation classes, never to infer geometry.
    result["banc_candidates"] = {"rows": len(banc),
        "visual_function_candidates": dict(Counter(r.get("Function", "") for r in banc if "visual" in r.get("Function", "").lower())),
        "visual_class_candidates": dict(Counter(r.get("Class", "") for r in banc if any(q in (r.get("Class", "")+r.get("Function", "")).lower() for q in ("visual", "optic", "transmedullary")))),
        "neck_crossing_candidate_fields": {k: sorted(set(r.get(k, "") for r in banc if r.get(k, "")))[:30] for k in ("Body Part", "Nerve", "Flow", "Function")},
        "interpretation": "annotation candidates for follow-up; no exact neck crossing geometry and no cross-specimen ID claim"}
    result["unknowns_and_caveats"] = [
        "FAFB is brain-only/truncated relative to a whole CNS; no nerve-cord continuation is established here.",
        "No glial annotation or tissue segmentation was audited; SWC neuronal morphology is not a tissue boundary.",
        "SWC headers explicitly report 1 nanometer units; this is provenance evidence, not independent calibration.",
        "Topology checks are bounded to three samples and cannot establish archive-wide validity.",
        "BANC and FAFB are distinct specimens; annotation names cannot establish exact cut geometry or cross-specimen identity.",
        "CRC validation here is stream decompression of three selected members, not a full archive CRC pass.",
    ]
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "anatomy_audit.json")
    with open(out, "w", encoding="utf-8") as fh: json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False); fh.write("\n")
    print(out)

if __name__ == "__main__": main()
