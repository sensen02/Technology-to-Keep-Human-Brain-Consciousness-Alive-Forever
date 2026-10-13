#!/usr/bin/env python3
"""INTERFACE DOCUMENTATION, GENERATED FROM THE CODE SO IT CANNOT ROT.

WHY GENERATED RATHER THAN WRITTEN.  A hand-written interface document is stale the first time
somebody changes a signature, and a stale interface document is worse than none: it is a map that
lies.  This project's whole repair has been about turning claims into checks, so the interface
document is generated from the AST and then CHECKED for drift:

    tools_interface_doc.py            regenerate INTERFACES.zh-CN.md + outputs/interfaces.json
    tools_interface_doc.py --check    compare against the last generated set; report what changed

WHY IT PARSES INSTEAD OF IMPORTING.  Importing `engine.embodied.*` pulls in flygym and mujoco, which
exist only in the body environment; an interface document that cannot be produced without a
particular interpreter is a document that will not be produced.  The AST is available everywhere.

WHAT IT RECORDS PER MODULE, chosen so that ONE module can be changed without reading the others:
  * the module's declared purpose (its own docstring, first paragraph);
  * every public class and function with its FULL signature;
  * the first line of each of their docstrings -- the contract in the author's own words;
  * whether the module declares its units and whether it declares anything ILLUSTRATIVE, because in
    this project those two declarations are what make a number usable;
  * which modules it imports from the project, so the effect of a change can be bounded.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_interface_doc.py
    ./venv/bin/python tools_interface_doc.py --check     # drift alarm
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DOC = HERE / "INTERFACES.zh-CN.md"
STATE = HERE / "outputs" / "interfaces.json"

SKIP = ("site-packages", "node_modules", "vendor", ".git", "__pycache__")

#: Which files are documented.  Engine modules and the embodied package are the platform; the
#: top-level recording scripts are the measurement chain.  Nothing else is included, because a
#: document that covers everything is a document nobody reads.
GLOBS = ("engine/*.py", "engine/embodied/*.py", "run_arena_record.py", "run_arena_reconstruct.py",
         "run_tactile_record.py", "limb_tracking.py")

#: A module smaller than this is usually a shim; it is still listed but marked terse.
TERSE_LINES = 40


def _sig(node: ast.AST) -> str:
    """Render a function/method signature from the AST, with defaults where they are literals."""
    a = node.args
    parts = []
    pos = list(getattr(a, "posonlyargs", [])) + list(a.args)
    ndef = len(a.defaults)
    for i, arg in enumerate(pos):
        di = i - (len(pos) - ndef)
        s = arg.arg
        if di >= 0:
            d = a.defaults[di]
            s += "=" + (ast.unparse(d) if hasattr(ast, "unparse") else "...")
        parts.append(s)
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(a.kwonlyargs):
        s = arg.arg
        d = a.kw_defaults[i]
        if d is not None:
            s += "=" + (ast.unparse(d) if hasattr(ast, "unparse") else "...")
        parts.append(s)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    ret = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({', '.join(parts)}){ret}"


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        if line.strip():
            return line.strip()[:160]
    return ""


def describe(path: Path) -> dict:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError as exc:
        return {"path": str(path.relative_to(HERE)), "error": f"SyntaxError: {exc}"}
    rel = str(path.relative_to(HERE))
    mod_doc = ast.get_docstring(tree) or ""
    first_para = mod_doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    api, imports = [], set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and not sub.name.startswith("_"):
                    methods.append({"name": sub.name, "signature": _sig(sub),
                                    "doc": _first_line(ast.get_docstring(sub))})
            api.append({"kind": "class", "name": node.name,
                        "bases": [ast.unparse(b) for b in node.bases],
                        "doc": _first_line(ast.get_docstring(node)), "methods": methods})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and not node.name.startswith("_"):
            api.append({"kind": "function", "name": node.name, "signature": _sig(node),
                        "doc": _first_line(ast.get_docstring(node))})
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and \
                (node.module.startswith("engine") or node.level):
            imports.add(("." * node.level) + (node.module or ""))
    low = path.read_text(errors="replace")
    return {
        "path": rel, "lines": len(low.splitlines()), "summary": first_para,
        "declares_units": any(k in low for k in ("units", "UNITS", "mV", "nA", "um", "mm")),
        "declares_illustrative": "ILLUSTRATIVE" in low or "illustrative" in low,
        "declares_not_established": "does_not_establish" in low or "NOT ESTABLISH" in low.upper(),
        "project_imports": sorted(imports), "api": api,
    }


def collect() -> dict:
    mods = []
    for pat in GLOBS:
        for p in sorted(HERE.glob(pat)):
            if any(s in p.parts for s in SKIP) or p.name.startswith("__"):
                continue
            mods.append(describe(p))
    mods.sort(key=lambda m: (-m.get("lines", 0), m.get("path", "")))
    return {"modules": mods, "n_modules": len(mods),
            "generated_by": "tools_interface_doc.py (AST, no imports)"}


def signature_index(state: dict) -> dict:
    """A flat {module::symbol: signature} map -- the thing that makes drift detectable."""
    out = {}
    for m in state.get("modules", []):
        for item in m.get("api", []):
            if item["kind"] == "function":
                out[f"{m['path']}::{item['name']}"] = item["signature"]
            else:
                out[f"{m['path']}::{item['name']}"] = "class"
                for meth in item.get("methods", []):
                    out[f"{m['path']}::{item['name']}.{meth['name']}"] = meth["signature"]
    return out


def diff(old: dict, new: dict) -> dict:
    a, b = signature_index(old), signature_index(new)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return {"added": added, "removed": removed, "changed": changed,
            "n_changes": len(added) + len(removed) + len(changed),
            "added_detail": {k: b[k] for k in added}, "removed_detail": {k: a[k] for k in removed},
            "changed_detail": {k: {"was": a[k], "now": b[k]} for k in changed}}


def render(state: dict, seams: dict, drift: dict | None) -> str:
    L = []
    L.append("# 接口文档（从代码生成）\n")
    L.append("> **本文件由 `tools_interface_doc.py` 从 AST 生成，不手工编辑。**")
    L.append("> 改了模块就跑一次 `./venv/bin/python tools_interface_doc.py`；"
             "用 `--check` 可以查出**与上次相比哪些签名变了**。\n")
    L.append(f"生成方式：`{state['generated_by']}`　模块数：**{state['n_modules']}**\n")

    L.append("\n---\n\n## 1. 闭环的四条接缝（本平台最该先看的一张表）\n")
    L.append("| 接缝 | 方向 | 状态 | 实现 | 边界 |")
    L.append("|---|---|---|---|---|")
    for name, s in seams.items():
        # `observation_channels` is a survey entry with a DIFFERENT shape from a seam: it answers
        # "which channels can be recorded at the same time", not "is this interface implemented".
        if name == "observation_channels":
            continue
        impl = ", ".join(f"`{c['candidate'].split('.')[-1]}`"
                         for c in s["candidates_checked"] if c["satisfies_protocol"]) or "—"
        L.append(f"| **{name}** | {s['direction']} | **{s['status']}** | {impl} | {s['limits']} |")
    L.append("\n读法：`DECLARED_NOT_IMPLEMENTED` 不是遗漏，而是**平台形状的一个事实** ——")
    L.append("它意味着那条信息流**不存在**（例如电极刺激从未注入神经元）。\n")

    # FOUR SEAMS IMPLEMENTED IS NOT FOUR CHANNELS OBSERVED.  Printed as its own table so that a
    # reader cannot take "4/4 IMPLEMENTED" above as "4 channels recorded together".
    oc = seams.get("observation_channels")
    if oc:
        detail = oc.get("detail", {})
        L.append("\n### 1b. 能**同时**观测的通道（与上表是两个不同问题）\n")
        L.append(f"同一次 `run()`、同一个躯体、同一个 interval 计数器里真正一起录下来的通道："
                 f"**{oc['n_simultaneous']}** 个 —— "
                 f"{', '.join('`%s`' % c for c in oc['recorded_in_one_run_on_one_body'])}；"
                 f"结构性缺席：{', '.join('`%s`' % c for c in oc['absent_by_construction'])}。\n")
        L.append("| 通道 | 所在栈 | 能否同时 | 实测 | 边界 |")
        L.append("|---|---|:--:|---|---|")
        for ch, spec in detail.items():
            if ch == "together":
                continue
            L.append(f"| `{ch}` | {spec['in_stack']} | "
                     f"{'✅' if spec['together'] else '❌'} | {spec['measured']} | {spec['limits']} |")
        tg = detail.get("together", {})
        if tg:
            L.append(f"\n**该文件修掉的缺陷**：{tg.get('defect_fixed_here','—')}\n")
            L.append(f"**注入验证**：{tg.get('injection','—')}\n")

    L.append("\n---\n\n## 2. 模块索引（按体量排序，`units`/`ILLUSTRATIVE` 两列表示该模块是否自带声明）\n")
    L.append("| 模块 | 行数 | 自身声明的用途 | 声明单位 | 声明 ILLUSTRATIVE | 声明未确立 |")
    L.append("|---|---:|---|:--:|:--:|:--:|")
    for m in state["modules"]:
        if "error" in m:
            L.append(f"| `{m['path']}` | — | ⚠️ {m['error']} | | | |")
            continue
        L.append(f"| `{m['path']}` | {m['lines']} | {m.get('summary','')[:110] or '—'} | "
                 f"{'✅' if m['declares_units'] else '—'} | "
                 f"{'✅' if m['declares_illustrative'] else '—'} | "
                 f"{'✅' if m['declares_not_established'] else '—'} |")

    L.append("\n---\n\n## 3. 与上次生成的差异（drift）\n")
    # ALWAYS emitted, even with no baseline.  MEASURED reason: when this section was conditional on
    # a previous generation existing, the document's TEXT depended on whether a baseline was
    # present, so `--stale` reported the freshly generated document as stale.  A renderer whose
    # output depends on history cannot be compared against itself.
    if drift is None:
        L.append("本次为首次生成，无对比基线。\n")
    elif drift["n_changes"] == 0:
        L.append("**无变化。** 所有公开签名与上次一致。\n")
    else:
        L.append(f"**{drift['n_changes']} 处变化** —— 下面就是「改一个模块会影响谁」的清单：\n")
        for k in drift["added"]:
            L.append(f"- ➕ `{k}` → `{drift['added_detail'][k]}`")
        for k in drift["removed"]:
            L.append(f"- ➖ `{k}`（原 `{drift['removed_detail'][k]}`）")
        for k in drift["changed"]:
            d = drift["changed_detail"][k]
            L.append(f"- 🔁 `{k}`：`{d['was']}` → `{d['now']}`")
        L.append("")

    L.append("\n---\n\n## 4. 逐模块公开接口\n")
    L.append("每个条目给出**完整签名**与作者自己写的第一行契约，"
             "所以改一个模块之前不需要通读其它模块。\n")
    for m in state["modules"]:
        if "error" in m or not m.get("api"):
            continue
        L.append(f"\n### `{m['path']}`  ({m['lines']} 行)\n")
        if m.get("summary"):
            L.append(f"{m['summary']}\n")
        if m.get("project_imports"):
            L.append(f"项目内依赖：{', '.join('`'+i+'`' for i in m['project_imports'])}\n")
        for item in m["api"]:
            if item["kind"] == "function":
                L.append(f"- **`{item['signature']}`**")
                if item["doc"]:
                    L.append(f"    {item['doc']}")
            else:
                base = f"({', '.join(item['bases'])})" if item["bases"] else ""
                L.append(f"- **`class {item['name']}{base}`**")
                if item["doc"]:
                    L.append(f"    {item['doc']}")
                for meth in item["methods"]:
                    L.append(f"    - `{meth['signature']}`"
                             + (f" — {meth['doc']}" if meth["doc"] else ""))
    L.append("\n---\n\n## 5. 这份文档刻意**不**声称的东西\n")
    L.append("- 不声称签名正确、不声称实现符合契约。它只把**签名与作者写在第一行的契约**"
             "集中起来，让你不必通读代码就能定位改动面。")
    L.append("- 不覆盖 `engine/**` 与顶层测量脚本之外的代码（测试、交付脚本、场景前端不在内）。")
    L.append("- 不替代运行时的行为检查：契约是否被遵守由 `run_all_checks.py` 的 37 项与"
             "不变量/注入回归负责。\n")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="对比上次生成的结果，报告签名漂移（给人看；有变化时退出码为 1）")
    ap.add_argument("--stale", action="store_true",
                    help=("作为检查运行：把 INTERFACES.zh-CN.md 与当前代码重新生成的版本比较，"
                          "不一致就是文档过期（退出码 1）。这是可以放进一键检查的那个语义 —— "
                          "它只在**忘记重新生成**时亮灯，合法的接口改动不会让它永久见红。"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    from engine.embodied.seams import survey
    state = collect()
    seams = survey()
    previous = json.loads(STATE.read_text()) if STATE.is_file() else None

    if args.stale:
        # Render exactly as the LAST generation did, using the drift THAT generation recorded.  A
        # plain `diff(previous, state)` looks equivalent and is not: right after a regeneration
        # `previous == state`, so it recomputes to zero while the document on disk still narrates
        # the previous run's changes, and the check then calls a freshly generated document stale.
        want = render(state, seams, (previous or {}).get("drift_at_generation"))
        have = DOC.read_text() if DOC.is_file() else ""
        if not have:
            print("INTERFACES.zh-CN.md is MISSING: run tools_interface_doc.py")
            return 1
        if have == want:
            print(f"INTERFACES.zh-CN.md is up to date with the code "
                  f"({state['n_modules']} modules, {len(signature_index(state))} public symbols)")
            return 0
        old, new = signature_index(previous or {}), signature_index(state)
        d = diff(previous or {}, state) if previous else {"n_changes": 0, "added": [], "removed": [],
                                                         "changed": []}
        print("INTERFACES.zh-CN.md is STALE: the code no longer matches the document.")
        if d["n_changes"]:
            print(f"  {d['n_changes']} signature change(s) since the last generation:")
            for k in d["added"][:10]:
                print(f"    + {k}")
            for k in d["removed"][:10]:
                print(f"    - {k}")
            for k in d["changed"][:10]:
                print(f"    ~ {k}")
        print("  fix: ./venv/bin/python tools_interface_doc.py")
        return 1

    drift = diff(previous, state) if previous else None
    if args.check:
        if previous is None:
            print("no previous outputs/interfaces.json to compare against: NOT_RUN")
            return 0
        print(f"interface drift vs last generation: {drift['n_changes']} change(s)")
        for k in drift["added"]:
            print(f"  + {k}  -> {drift['added_detail'][k]}")
        for k in drift["removed"]:
            print(f"  - {k}  (was {drift['removed_detail'][k]})")
        for k in drift["changed"]:
            d = drift["changed_detail"][k]
            print(f"  ~ {k}: {d['was']}  ->  {d['now']}")
        if drift["n_changes"]:
            print("\nre-run without --check to regenerate INTERFACES.zh-CN.md")
        return 1 if drift["n_changes"] else 0

    if not args.no_write:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({**state, "seams": seams,
                                     "drift_at_generation": drift}, indent=2, sort_keys=True))
        DOC.write_text(render(state, seams, drift))
    print(f"modules documented: {state['n_modules']}")
    print(f"seams: " + ", ".join(f"{k}={v['status']}" for k, v in seams.items()))
    if drift:
        print(f"drift vs previous generation: {drift['n_changes']} change(s)")
    print(f"wrote {DOC}\nwrote {STATE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
