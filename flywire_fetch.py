"""Fetch the FlyWire connectome data products through the Codex API.

Cookies are read from .flywire_cookies.txt (mode 600, never copied into the
delivered output folder).  The session cookie is a live credential; see the
final report for the rotation warning.
"""
import gzip, io, json, os, sys, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
COOKIE_FILE = os.path.join(ROOT, ".flywire_cookies.txt")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def cookie_header():
    with open(COOKIE_FILE) as fh:
        return fh.read().strip()


def get(url, accept="text/html"):
    req = urllib.request.Request(url, headers={
        "Cookie": cookie_header(), "User-Agent": UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read(), dict(r.headers)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "list"
    if what == "list":
        st, body, hdr = get("https://codex.flywire.ai/api/download?dataset=fafb")
        print("status", st, "bytes", len(body), "ctype", hdr.get("Content-Type"))
        txt = body.decode("utf-8", "replace")
        open(os.path.join(ROOT, "outputs", "flywire_download_page.html"), "w").write(txt)
        # the page is a JS/your framework app; look for embedded JSON
        import re
        for pat in (r'data_product[^,}]{0,80}', r'api_token[^,}]{0,60}',
                    r'"name"\s*:\s*"[^"]{3,60}"'):
            hits = re.findall(pat, txt)[:15]
            if hits:
                print(f"  {pat[:22]}... ->", hits[:8])
        print("  page saved to outputs/flywire_download_page.html")
    elif what == "account":
        st, body, hdr = get("https://codex.flywire.ai/account")
        txt = body.decode("utf-8", "replace")
        open(os.path.join(ROOT, "outputs", "flywire_account.html"), "w").write(txt)
        print("account status", st, "bytes", len(body))
        import re
        m = re.findall(r'[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{20,}', txt)
        print("  token-like strings:", m[:5])
