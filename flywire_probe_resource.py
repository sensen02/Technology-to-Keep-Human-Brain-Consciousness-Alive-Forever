import os, sys, urllib.request
ROOT = os.path.dirname(os.path.abspath(__file__))
cookie = open(os.path.join(ROOT, ".flywire_cookies.txt")).read().strip()
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
for prod in ("connections_princeton", "consolidated_cell_types", "neurons"):
    url = (f"https://codex.flywire.ai/api/download_resource?data_product={prod}"
           f"&dataset=fafb")
    req = urllib.request.Request(url, headers={"Cookie": cookie, "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            cl = r.headers.get("Content-Length")
            ct = r.headers.get("Content-Type")
            cd = r.headers.get("Content-Disposition")
            first = r.read(4)
            magic = "gzip" if first[:2] == b"\x1f\x8b" else repr(first[:4])
            print(f"{prod}: HTTP {r.status}  Content-Length={cl}  type={ct}  "
                  f"disp={cd}  magic={magic}")
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", "replace")
        print(f"{prod}: HTTPError {e.code} {e.reason} -> {body[:200]}")
    except Exception as e:
        print(f"{prod}: {type(e).__name__}: {e}")
