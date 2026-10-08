#!/usr/bin/env python3
"""Read-only disk/duplication audit for the cell_wound_prototype workspace.

Usage:
    python3 tools_audit_disk.py [--root DIR] [--dup-min-bytes N] [--top N] [--json OUT]

Writes nothing unless --json is given. Never deletes anything.
Prints:
  1. total size, per-top-level-directory size
  2. largest files (top N)
  3. duplicate files by (size, sha256 of first 64 KiB + size) candidate set,
     confirmed by full sha256 for candidates above the threshold
  4. stale files: .log/.tmp/.bak older than --stale-days (default 1) and not
     inside a virtualenv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

VENV_MARKERS = ("/venv/", "/venv_body/", "/.venv/", "/venv_pvm/")
SKIP_DIR_NAMES = {".git", "lost+found", "__pycache__"}
# Runtime/service state that must never be treated as reclaimable clutter:
#  * .lock files (flock single-instance guards; deleting one lets a second
#    server start while the first still holds the old inode)
PROTECTED_SUFFIXES = (".lock",)


def human(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"


def in_venv(path: str) -> bool:
    p = path.replace(os.sep, "/")
    return any(m in p for m in VENV_MARKERS)


def digest(path: str, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if limit is None:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    if limit is not None:
        with open(path, "rb") as fh:
            h.update(fh.read(limit))
    return h.hexdigest()


def walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIR_NAMES and not os.path.islink(os.path.join(dirpath, d))
        ]
        for name in filenames:
            full = os.path.join(dirpath, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if os.path.islink(full) or not os.path.isfile(full):
                continue
            yield full, st.st_size, st.st_mtime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dup-min-bytes", type=int, default=64 * 1024)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--stale-days", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    now = time.time()
    by_dir = defaultdict(int)
    by_ext = defaultdict(int)
    files = []
    for full, size, mtime in walk(root):
        files.append((full, size, mtime))
        rel = os.path.relpath(full, root)
        top = rel.split(os.sep)[0]
        by_dir[top] += size
        by_ext[os.path.splitext(full)[1].lower() or "(none)"] += size

    total = sum(s for _, s, _ in files)
    print(f"root            : {root}")
    print(f"files           : {len(files)}")
    print(f"total size      : {human(total)} ({total} bytes)")
    print()

    print("== size by top-level entry ==")
    for name, size in sorted(by_dir.items(), key=lambda kv: -kv[1])[: args.top]:
        share = 100.0 * size / total if total else 0.0
        print(f"{human(size):>10}  {share:5.1f}%  {name}")
    print()

    print(f"== largest {args.top} files ==")
    for full, size, mtime in sorted(files, key=lambda t: -t[1])[: args.top]:
        age = (now - mtime) / 86400.0
        print(f"{human(size):>10}  {age:6.1f}d  {os.path.relpath(full, root)}")
    print()

    print("== duplicate candidates (same size, > threshold) ==")
    by_size = defaultdict(list)
    for full, size, mtime in files:
        if size >= args.dup_min_bytes:
            by_size[size].append(full)
    cand = {s: v for s, v in by_size.items() if len(v) > 1}
    dup_waste = 0
    dup_groups = []
    for size, paths in sorted(cand.items(), key=lambda kv: -kv[0] * len(kv[1])):
        heads = defaultdict(list)
        for p in paths:
            try:
                heads[digest(p, 65536)].append(p)
            except OSError:
                continue
        for head, group in heads.items():
            if len(group) < 2:
                continue
            full_hashes = defaultdict(list)
            for p in group:
                try:
                    full_hashes[digest(p)].append(p)
                except OSError:
                    continue
            for h, same in full_hashes.items():
                if len(same) < 2:
                    continue
                waste = size * (len(same) - 1)
                dup_waste += waste
                dup_groups.append({"sha256": h, "bytes": size, "paths": sorted(same)})
    if not dup_groups:
        print("(none above threshold)")
    for g in sorted(dup_groups, key=lambda g: -g["bytes"] * len(g["paths"])):
        print(f"{human(g['bytes']):>10} each, {len(g['paths'])} copies "
              f"(waste {human(g['bytes'] * (len(g['paths']) - 1))})")
        for p in g["paths"]:
            print(f"            {os.path.relpath(p, root)}")
    print(f"reclaimable by dedup: {human(dup_waste)}")
    print()

    print(f"== stale logs/tmp/bak older than {args.stale_days} d (outside venvs) ==")
    stale = []
    for full, size, mtime in files:
        ext = os.path.splitext(full)[1].lower()
        if ext not in (".log", ".tmp", ".bak", ".old"):
            continue
        if in_venv(full) or full.endswith(PROTECTED_SUFFIXES):
            continue
        age = (now - mtime) / 86400.0
        if age >= args.stale_days:
            stale.append((full, size, age))
    stale.sort(key=lambda t: -t[2])
    print(f"count {len(stale)}, total {human(sum(s for _, s, _ in stale))}")
    for full, size, age in stale[: args.top]:
        print(f"{human(size):>10}  {age:6.1f}d  {os.path.relpath(full, root)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "root": root,
                "files": len(files),
                "total_bytes": total,
                "by_top_level": dict(sorted(by_dir.items(), key=lambda kv: -kv[1])),
                "duplicates": dup_groups,
                "duplicate_waste_bytes": dup_waste,
                "stale_count": len(stale),
                "stale_bytes": sum(s for _, s, _ in stale),
            }, fh, indent=2, sort_keys=False)
        print(f"\njson written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
