"""Download the FlyWire FAFB data products needed for the neural layer.

Uses the Codex static-download API (the documented programmatic path):
    /api/download_resource?data_product=...&dataset=fafb&api_token=...

The api_token is read from .flywire_api_token (mode 600) and is NEVER written
into the delivered output folder.  Both files are gzipped CSV; we stream them
to disk and verify the gzip stream and the row count.

Products:
  connections_princeton    pre/post neuron root ids, synapse count, nt_type
                           (this one file gives edges AND transmitter type)
  consolidated_cell_types  per-neuron cell type annotations
"""

from __future__ import annotations

import gzip
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "flywire")
os.makedirs(DATA, exist_ok=True)
TOKEN_FILE = os.path.join(ROOT, ".flywire_api_token")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

PRODUCTS = ("connections_princeton", "consolidated_cell_types",
            "classification", "neurons")
# dataset is taken from argv[1] so the same script serves fafb / banc / manc
DATASET = sys.argv[1] if len(sys.argv) > 1 else "fafb"


def token():
    with open(TOKEN_FILE) as fh:
        return fh.read().strip()


def download(product, dataset=None):
    dataset = dataset or DATASET
    url = ("https://codex.flywire.ai/api/download_resource?"
           + urllib.parse.urlencode({"data_product": product,
                                     "dataset": dataset,
                                     "api_token": token()}))
    dest = os.path.join(DATA, f"{dataset}_{product}.csv.gz")
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"  {product}: already present ({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        total = r.headers.get("Content-Length")
        total = int(total) if total else None
        done = 0
        last = 0.0
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 3.0:
                    last = now
                    pct = f" ({100*done/total:.1f}%)" if total else ""
                    print(f"    {product}: {done/1e6:.1f} MB{pct} "
                          f"[{now-t0:.0f}s]", flush=True)
        os.replace(tmp, dest)
    print(f"  {product}: done {os.path.getsize(dest)/1e6:.1f} MB in "
          f"{time.time()-t0:.0f}s")
    return dest


def verify(path, max_lines=None):
    """Verify the gzip stream and report header + row count."""
    n = 0
    header = None
    with gzip.open(path, "rt", errors="replace") as fh:
        header = fh.readline().rstrip("\n")
        for _ in fh:
            n += 1
            if max_lines and n >= max_lines:
                break
    return header, n


if __name__ == "__main__":
    print(f"=== FlyWire {DATASET} 下载 ===")
    for p in PRODUCTS:
        try:
            path = download(p)
            hdr, n = verify(path)
            print(f"  列名: {hdr}")
            print(f"  行数: {n}" + (" (已截断)" if False else "") + "\n")
        except urllib.error.HTTPError as e:
            body = e.read()[:200].decode("utf-8", "replace")
            print(f"  {p}: HTTPError {e.code} -> {body}")
        except Exception as e:
            print(f"  {p}: {type(e).__name__}: {e}")
