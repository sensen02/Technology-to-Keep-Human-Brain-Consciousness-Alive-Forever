"""engine.record -- self-describing run records.

A result that cannot be reproduced is not a result.  Every run of this engine
emits a record containing everything needed to re-run it and to audit what it
was allowed to claim:

  * environment (python, numpy, numba, platform, cpu count);
  * the project's git commit, if the tree is a git checkout;
  * the organism profile and its REQUIRED caveats;
  * the full parameter snapshot WITH provenance;
  * the pipeline wiring report (which layer read which port, in which unit);
  * every check result;
  * the outputs written;
  * a content hash so two records can be compared for equality.

The caveats are copied into the record by the engine rather than being retyped
by whoever writes the report, because that is how a caveat gets quietly lost.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

__all__ = ["RunRecord"]


def _git_commit(path):
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5)
            return {"commit": out.stdout.strip(),
                    "dirty": bool(dirty.stdout.strip())}
    except Exception:
        pass
    return {"commit": None, "dirty": None}


def _versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform(),
         "machine": platform.machine(), "cpu_count": os.cpu_count()}
    for mod in ("numpy", "scipy", "numba", "matplotlib"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception:
            v[mod] = None
    return v


class RunRecord:
    def __init__(self, title, root=None, seed=None):
        self.data = {
            "title": title,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "root": root or os.getcwd(),
            "seed": seed,
            "environment": _versions(),
            "git": _git_commit(root or os.getcwd()),
            "profile": None,
            "parameters": None,
            "parameter_summary": None,
            "wiring": None,
            "checks": None,
            "outputs": [],
            "notes": [],
            "caveats": [],
            "claim_boundary": [
                "这是模拟，不是永生实验。",
                "唯一可用的定量实验对照是一条对照伤口钙波半径轨迹。",
                "参数若未标注 measured，即为 illustrative/assumed，不是实测值。",
                "本模拟不验证、也不能验证主观体验。",
            ],
        }

    # ------------------------------------------------------------------
    def set_profile(self, profile):
        self.data["profile"] = profile.to_dict()
        self.data["caveats"] = list(profile.required_caveats())
        return self

    def set_parameters(self, registry):
        self.data["parameters"] = registry.snapshot()
        self.data["parameter_summary"] = registry.summary()
        return self

    def set_wiring(self, pipeline):
        self.data["wiring"] = {
            "name": pipeline.name,
            "layers": [
                {"name": l.name, "enabled": bool(l.enabled),
                 "requires": dict(l.requires), "provides": dict(l.provides),
                 "target_dt": str(l.target_dt)}
                for l in pipeline.layers],
            "report": pipeline.wiring_report(),
        }
        return self

    def set_checks(self, suite, results=None):
        results = results if results is not None else suite.run()
        self.data["checks"] = {
            "summary": suite.summary(results),
            "results": [r.to_dict() for r in results],
            "markdown": suite.markdown(results),
        }
        return self

    def add_output(self, path, what=""):
        self.data["outputs"].append({"path": os.path.abspath(path), "what": what})
        return self

    def add_note(self, text):
        self.data["notes"].append(text)
        return self

    def add_caveat(self, text):
        if text not in self.data["caveats"]:
            self.data["caveats"].append(text)
        return self

    def set_metric(self, key, value):
        self.data.setdefault("metrics", {})[key] = value
        return self

    # ------------------------------------------------------------------
    def hash(self):
        blob = json.dumps(
            {k: v for k, v in self.data.items()
             if k not in ("created_utc", "outputs")},
            sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def write(self, path):
        self.data["record_hash"] = self.hash()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False, default=str)
        return path

    def markdown_claim_boundary(self):
        rows = ["## 本记录允许的结论边界", ""]
        for c in self.data["claim_boundary"]:
            rows.append(f"- {c}")
        if self.data["caveats"]:
            rows += ["", "### 由生物档案强制注入的必读警示", ""]
            for c in self.data["caveats"]:
                rows.append(f"- {c}")
        return "\n".join(rows)
