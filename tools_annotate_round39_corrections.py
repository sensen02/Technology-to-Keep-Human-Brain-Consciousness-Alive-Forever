"""ROUND 39 (FIX), part G: annotate the historical claims this round refuted.

WHY
---
Round 39 measured two things that several earlier rounds built on and that are now known to
be wrong in a specific, citable way:

  A. "exact balance (L1 = 0.0)" - the LP residual.  Part C measured the PHYSICAL imbalance
     (M(q) qacc) at the pose the LP was solved for: 5.77% of body weight with the LP's own
     activation, 3.72% at best under a muscle-only rebalance.  The LP's zero is a property of
     its frozen linearisation, not of the body.

  B. "support-margin ceiling 0.370 mm" - the number is really 0.370 cm = 3.70 mm (the same
     `_mm`-holds-cm defect), and the ceiling itself is a property of round 37's NARROW SEARCH
     BOX: a wider search reaches 7.540 mm, and the same search on the round-39 best contact
     set reaches exactly the same 7.540 mm, so the contact model does not set it either.

An earlier round's report is evidence: rewriting it in place would destroy the record of what
was believed and measured at the time.  So this tool does NOT touch any historical sentence.
It APPENDS a clearly marked correction footer to each affected file and writes a
machine-readable index, so the correction travels with the document.

Scoping is deliberate and conservative:
  * only A-markers (`L1 = 0.0`, `精确平衡`) and B-markers (margin + 0.37) are annotated;
  * an unrelated `0.37` (a pixel offset, a bw ratio, a stiffness) is NOT annotated, and the
    tool prints every line it decided about so the decisions are auditable;
  * running it twice does not add a second footer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402

OUT = HERE / "outputs/fix_round39"
FOOTER_MARK = "<!-- ROUND39-CORRECTION -->"
FOOTER_HEAD_MARK = "## 第39轮更正批注（追加，未修改上文任何一句）"

# A: the LP exact-balance claim
A_PATTERNS = (re.compile(r"L1\s*=\s*0(\.0+)?\b"), re.compile(r"精确平衡"),
              re.compile(r"exact(ly)? (force[- ])?balanc", re.I))
# B: the margin ceiling - must mention margin/ceiling AND the number 0.37
B_MARGIN = re.compile(r"margin|支撑|天花板|ceiling", re.I)
B_NUMBER = re.compile(r"0\.37")

# files whose `L1 = 0.0` is about something other than the stance LP
A_EXCLUDE_FILES = {"FIX_ROUND39.zh-CN.md"}
# files whose `0.37` is an unrelated quantity (pixel offsets, bw ratios, stiffness)
B_EXCLUDE_FILES = {"ARENA_FIX_ROUND4.zh-CN.md", "LOAD_SHARE.zh-CN.md",
                   "BALANCE_AND_SYMMETRY.zh-CN.md", "LEG_TUNING.zh-CN.md",
                   "STANDING_FIX.zh-CN.md", "STANDING_PROGRESS.zh-CN.md",
                   "RECORDING_PIPELINE.zh-CN.md", "FIX_ROUND39.zh-CN.md"}

FOOTER_TEMPLATE = """

---

## 第39轮更正批注（追加，未修改上文任何一句）

本文件成文时的结论**保留原样**（那是当时的证据记录）。以下是第39轮修复用**实测**推翻或改写其中两条，
凡引用本文件者必须先读这里。

### 更正 A：「精确平衡（L1 = 0.0）」不是物理平衡

第39轮把该量从 LP 的 `qfrc` 残差域搬到**物理域**（`不平衡 = M(q)·qacc`，真实静态平衡要求它为 0）：

| 量 | 值 | 占体重 |
|---|---|---|
| LP 解激活下的腿部不平衡 | 1.4099 | **5.77%** |
| 零激活对照 | 2.0089 | 8.22% |
| 实测力臂迭代配平的最优 | 0.9589 | **3.92%** |
| LP 声称的残差 | 0.0 | 0% |

用 20 个门控姿态 × 4 个激活种子测得的**配平下限是 3.72% 体重**，**LP 的 0.0 无法复现**。

> **LP 把力-长、力-速、力臂以及接触集全部冻结在某一瞬，这样解出的是平衡点，不是平衡态。**
> **「精确平衡（L1 = 0.0）」不得再作为「这个姿态能站住」的证据。**

证据：`outputs/fix_round39/fix_round39_lp_balance_verification.json`、
`fix_round39_balance_floor.json`；代码 `run_fix_round39_lp_physics.py`。
边界：迭代法不收敛**不构成**「不存在精确平衡」的证明（弱优化器，种子离散比中位 1.4×、最大 3.08×）。

### 更正 B：「支撑 margin 天花板 0.370 mm」是两处错误叠加

1. **单位错**：该数单位是**厘米**（与 `thorax_height_mm` 同族缺陷）。0.370 实为 **3.70 mm**。
   （校准：同一实现精确复现第37轮 keyframe 的 0.5466，故可比。）
2. **它是搜索框的属性，不是果蝇的属性**：把关节偏移范围放宽到 ±1.6 rad 后，
   无约束 margin 最大达 **7.540 mm**；而在第39轮最优接触集（cone=elliptic + noslip3）上
   重跑同一搜索，结果**到小数点后三位完全相同（7.540 mm）**。

> **接触参数不决定支撑几何**（它决定的是倒下前的时间：25.1 → 41.3 ms）。
> 「抗倾覆几何是瓶颈」这条应改述为「**在第37轮的搜索框里** margin 最大 3.70 mm」。

证据：`outputs/fix_round39/fix_round39_margin_research.json`、`fix_round39_contact_one_at_a_time.json`。

{mark}

"""

FOOTER = FOOTER_TEMPLATE.format(mark=FOOTER_MARK)
assert "{mark}" not in FOOTER, "the marker placeholder must be substituted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in HERE.glob("*.md"))
    ann = []
    decisions = []
    for p in files:
        text = p.read_text()
        # never scan our own correction footer: it necessarily quotes the very claims it
        # corrects, so including it would make the scan self-referential (measured: it did)
        scan_text = text.split("<!-- ROUND39-CORRECTION")[0]
        scan_text = scan_text.split("\n## 第39轮更正批注（追加，未修改上文任何一句）")[0]
        lines = scan_text.split("\n")
        hits_a, hits_b, skipped = [], [], []
        for i, line in enumerate(lines, 1):
            a = any(rx.search(line) for rx in A_PATTERNS)
            b = B_MARGIN.search(line) and B_NUMBER.search(line)
            if a and p.name not in A_EXCLUDE_FILES:
                hits_a.append({"line": i, "text": line.strip()[:170]})
            elif a:
                skipped.append({"line": i, "kind": "A", "reason": "file excluded (own fix log)",
                                "text": line.strip()[:120]})
            if b and p.name not in B_EXCLUDE_FILES:
                hits_b.append({"line": i, "text": line.strip()[:170]})
            elif b:
                skipped.append({"line": i, "kind": "B",
                                "reason": "unrelated 0.37 (not the stance margin)",
                                "text": line.strip()[:120]})
        if not hits_a and not hits_b:
            continue
        kinds = []
        if hits_a:
            kinds.append("A")
        if hits_b:
            kinds.append("B")
        ann.append({"file": p.name, "corrections": kinds,
                    "A_lines": hits_a, "B_lines": hits_b, "skipped": skipped})
        for d in skipped:
            decisions.append({**d, "file": p.name})
        if args.apply:
            # strip any previously appended footer (broken or not), then append the good one
            body = text.split(FOOTER_HEAD_MARK)[0]
            body = body.split("\n---\n\n" + FOOTER_MARK)[0].rstrip("\n")
            p.write_text(body + FOOTER)

    if args.apply:
        # idempotency: a second run must not add another footer, and no file may carry an
        # unsubstituted placeholder (a bug that shipped once in this very tool)
        # structural self-check: each annotated file must end with a well formed footer,
        # carry exactly one head marker and one closing marker, and contain no placeholder.
        # (Checking for a literal "{mark}" anywhere is WRONG: the footer text legitimately
        # quotes one, which is how an earlier version of this check reported a false failure.)
        bad = []
        for p in files:
            t = p.read_text()
            if FOOTER_HEAD_MARK not in t:
                if FOOTER_HEAD_MARK in t.replace("\n", "\n"):
                    bad.append((p.name, "head missing"))
                continue
            tail = t.split(FOOTER_HEAD_MARK, 1)[1]
            ok = (t.count(FOOTER_HEAD_MARK) == 1
                  and t.count(FOOTER_MARK) == 1
                  and tail.rstrip().endswith(FOOTER_MARK))
            if not ok:
                bad.append((p.name, "footer not well formed"))
        second = [n for n, why in bad if why == "head missing"]
        idem = len(bad) == 0
        if bad:
            print("FAIL: malformed footer in:", bad)
    else:
        idem = None

    cp.write_result(
        OUT / "fix_round39_historical_annotations.json",
        question=("Which historical documents assert the two claims round 39 refuted, and "
                  "have they been annotated without altering their original text?"),
        status="PASS" if (args.apply and idem) else ("NOT_RUN" if not args.apply else "FAIL"),
        measured={
            "documents_annotated": [a["file"] for a in ann],
            "n_documents": len(ann),
            "correction_A_documents":
                [a["file"] for a in ann if "A" in a["corrections"]],
            "correction_B_documents":
                [a["file"] for a in ann if "B" in a["corrections"]],
            "per_document_hits": ann,
            "explicitly_not_annotated": decisions,
            "original_sentences_modified": False,
            "footer_marker": FOOTER_MARK,
            "idempotent": idem,
        },
        provenance={
            "correction_A_artifacts": ["outputs/fix_round39/"
                                       "fix_round39_lp_balance_verification.json",
                                       "outputs/fix_round39/fix_round39_balance_floor.json"],
            "correction_B_artifacts": ["outputs/fix_round39/"
                                       "fix_round39_margin_research.json",
                                       "outputs/fix_round39/"
                                       "fix_round39_contact_one_at_a_time.json"],
            "canonical_xml_sha256": cp.sha256_file(cp.CANONICAL_XML),
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "Appending a footer does NOT make the earlier documents correct: their original "
            "sentences and numbers are still there, by design, as the record of what was "
            "believed at the time.",
            "The scoping is by keyword and file list, so a claim worded in an unusual way "
            "could be missed; the per-document hit list is written out so that is auditable.",
            "No physics was re-measured here; this tool only carries the round-39 results "
            "into the documents that need them.",
        ],
    )
    print(f"documents to annotate: {len(ann)}")
    for a in ann:
        print(f"  {a['file']:44s} corrections {''.join(a['corrections'])}  "
              f"A-lines {[h['line'] for h in a['A_lines']]}  "
              f"B-lines {[h['line'] for h in a['B_lines']]}")
    print(f"explicitly NOT annotated: {len(decisions)} unrelated '0.37' occurrence(s)")
    if args.apply:
        print(f"idempotent: {idem}")
    else:
        print("(dry run - pass --apply to append the footers)")
    print("wrote", OUT / "fix_round39_historical_annotations.json")


if __name__ == "__main__":
    main()
