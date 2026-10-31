"""S206: independent verification + conclusion + bookkeeping (offline, 0 model loads)."""
import json
import hashlib
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

RUNROOT = Path('/home/sensen/Desktop/engine修复验收/s206-commandability-gap-lp')
DIR = sorted([p for p in RUNROOT.iterdir() if p.is_dir()])[-1]
STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
CENSUS = json.load(open('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/'
                        '20261025T073450411660Z/READOUT_CENSUS.json'))
MAP = np.load('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
BASE_GATE = 3.308620867791373
checks = []


def chk(name, ok, detail):
    checks.append(dict(check=name, ok=bool(ok), detail=detail))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


z = np.load(STATE)
Bm, base, f0, J, T = (np.asarray(z[k], float) for k in ('B', 'base', 'f0', 'J', 'T'))
empty = sorted(c['actuator_index'] for c in CENSUS['rows'] if c['pool_size'] == 0)
req = -(base + J.T @ f0)[6:]
A_all = (Bm @ T)[6:, :]
Bm_cmd = Bm.copy()
Bm_cmd[:, empty] = 0.0
A_cmd = (Bm_cmd @ T)[6:, :]


def solve(A, box):
    n = A.shape[1]
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                 b_ub=np.r_[req, -req], bounds=[(0, box)] * n + [(0, None)], method='highs')
    return float(lp.x[-1])


# 1. gate recomputed here, independently of the round's own script
mm_all1 = solve(A_all, 1.0)
chk('gate:baseline_reproduced', abs(mm_all1 - BASE_GATE) <= 1e-9, '%.15g' % mm_all1)
mm_cmd1 = solve(A_cmd, 1.0)
mm_all2, mm_cmd2 = solve(A_all, 2.0), solve(A_cmd, 2.0)
chk('gap:box1_identical', abs(mm_cmd1 - mm_all1) <= 1e-12, 'all %.15g vs commandable %.15g' % (mm_all1, mm_cmd1))
chk('gap:box2_identical', abs(mm_cmd2 - mm_all2) <= 1e-12, 'all %.15g vs commandable %.15g' % (mm_all2, mm_cmd2))

# 2. WHY: the mapping T gives those 16 actuators no controllable column at all
pool = np.asarray(MAP['actuator_pool']) if 'actuator_pool' in MAP.files else None
T_rows_empty = np.abs(T[empty, :]).max(axis=1)
chk('structure:uncommandable_rows_of_T_are_zero', float(T_rows_empty.max()) == 0.0,
    'max |T[actuator,:]| over the 16 rows = %.3g' % float(T_rows_empty.max()))
nonempty = [i for i in range(90) if i not in empty]
chk('structure:commandable_rows_of_T_are_nonzero', float(np.abs(T[nonempty, :]).max()) > 0.0,
    'the commandable actuators do carry controllable columns')
contrib = np.abs((Bm[:, empty] @ T[empty, :])).max() if T[empty, :].size else 0.0
chk('structure:uncommandable_contribution_to_A_is_zero', contrib == 0.0, 'max |contribution| = %.3g' % contrib)
chk('structure:baseline_force_still_counted', float(np.abs(Bm[:, empty]).max()) > 0.0,
    'their moment x gain columns are nonzero in B, so their constant force still enters through base; only modulation is absent')
if pool is not None:
    chk('structure:pool_slots_are_minus_one_for_them',
        bool(np.all(np.asarray(pool)[empty] == -1)), 'actuator_pool values %s' % np.asarray(pool)[empty].tolist())

# 3. per-row authority loss is empty, consistent with the structure checks
res = json.load(open(DIR / 'COMMANDABILITY_GAP_LP.json'))
chk('rows:no_row_loses_span', res['n_rows_with_lost_authority'] == 0,
    'n rows with any lost span = %d' % res['n_rows_with_lost_authority'])

# 4. the round's verdict follows from its own numbers
chk('verdict:matches_frozen_rule', res['verdict'] == 'GAP_IS_ELSEWHERE',
    '%s at relative delta %.3e' % (res['verdict'], res['delta']['relative']))

# 5. guards
chk('xml:hash_unchanged', sha(XML) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4', sha(XML))
chk('calls:no_physics', res['calls']['model_loads'] == 0 and res['calls']['mj_step'] == 0,
    json.dumps(res['calls']))

n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(
    dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
         note='every ingredient is the frozen S177 state; no model load, no forward, no mj_step'), indent=2, default=str))

# ---------------------------------------------------------------- ledger / budget / hashes
ledger = {"round": "S206", "failures": [
    {"id": "S206-A1", "what": "s206_commandability_lp.py first solve",
     "defect": "req was built as -base[6:] while the frozen definition (s185_reconcile.py) is required = -(base + J.T@f0)[6:]; "
               "the contact term was missing, so the baseline gate returned 0.270064 instead of 3.308620867791373",
     "caught_by": "the pre-registered baseline gate", "status": "fixed; the rerun reproduces the gate exactly"},
    {"id": "S206-A2", "what": "row-DOF map path", "defect": "the S179 round directory was hard-coded to a timestamp that no longer "
     "exists on disk (only its aggregate check file remains there)",
     "caught_by": "FileNotFoundError", "status": "fixed by globbing the newest ROW_DOF_MAP.json"},
    {"id": "S206-A3", "what": "PREREG string", "defect": "a patch replaced text inside the prereg string and broke it (unterminated "
     "literal)", "caught_by": "SyntaxError", "status": "fixed"}],
    "corrections": [
        {"id": "S206-C1", "what": "the implication drawn in S205",
         "defect": ("S205's next-action text suggested that the S188-S196 static verdicts must be read as 'infeasible only because "
                    "58% of the coxa authority is uncommandable', implying the LP had been optimistic. This round shows the opposite: "
                    "the frozen LP never included those 16 actuators as control columns at all (T rows are exactly zero), so the "
                    "3.3086 infeasibility already refers to the 74 commandable actuators only. Removing them changes the minimax by "
                    "exactly 0 in both boxes."),
         "impact": "the S205 gap is real for the BODY-muscle mapping (the coxa authority cannot be used), but it does NOT make any "
                   "recorded LP number wrong; no historical number changes",
         "status": "corrected here and in the tracker"}],
    "design_notes": [
        {"id": "S206-N1", "what": "what the gap does and does not mean",
         "finding": "the gap removes 58% of the moment share on DOF 22/42 from the usable set, but the frozen LP already treated "
                    "those actuators as constants; repairing the coverage would therefore ADD control columns and could only relax "
                    "the LP, never tighten it. That is a statement about a hypothetical repaired model, not about the current one."}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False))

conclusion = f"""# S206 不可命令执行器对静态同时可行性的影响 —— 结论（中文）

目录：`{DIR}`

## 0. 一句话
把 S205 发现的 16 个不可命令执行器从静态 LP 里剔除，**minimax 完全不变**（box [0,1]：3.308620867791373；box [0,2]：3.125786，两组都逐位相同）。原因是**冻结 LP 从一开始就没把这 16 个当作可控列**（它们在映射 `T` 里的行恒为 0，pool slot 是 −1）。所以**已记录的静态不可行性不是"多数了不可命令执行器"造成的**；同时 S205 里"这些结论只能读作'在当前覆盖下不可行'"的**暗示方向是错的**，本轮更正。

## 1. 方法与门
- 全部输入取自冻结状态 `ROOT_BALANCE_STATE.npz`：`B`(48×90)、`base`(48)、`f0`(64)、`J`(64×48)、`T`(90×66)。**0 次模型加载、0 次 forward、0 次 mj_step**，只解 6 次 LP。
- 需求向量按冻结定式：`required = -(base + J.T@f0)[6:]`。
- **基线门**：全执行器、box [0,1] 必须复现 `3.308620867791373` → **通过（精确到 1e-15）**。

## 2. 结果

| 变体 | box [0,1] minimax | box [0,2] minimax |
|---|---|---|
| 全部 90 个执行器的 `B` | **3.308620867791373** | 3.125786… |
| 把 16 个不可命令列清零后的 `B` | **3.308620867791373** | 3.125786… |

- 相对差 **0.0**（阈值 1%），逐行"损失跨度"表**为空**（42 行里没有任何一行失去可调跨度）→ 判定 **`GAP_IS_ELSEWHERE`**。

## 3. 为什么不变（结构性证据，已逐条断言）
1. `T` 的那 16 行**恒为 0**（`max |T[actuator,:]| = 0`）——它们没有可控列；
2. 因此它们对 `A` 的贡献**恰好为 0**（`max |B[:,16] @ T[16,:]| = 0`）；
3. 但它们**不是空气**：`B` 的这 16 列非零，说明它们的力矩×增益仍在，其**常值力经由 `base` 进入需求端**——缺席的只有"可调制性"；
4. `MAPPING.npz` 的 `actuator_pool` 对这 16 个执行器全是 **−1**。

## 4. 这对 S205 的结论意味着什么（S206-C1 更正）
- S205 的**事实**不变：16 个执行器不可命令；DOF 22/42 各有约 58% 的 |moment| 份额落在它们身上。
- 但**推论要改**：冻结 LP **本来就是"只对 74 个可命令执行器求解"**，所以 3.3086 这个数**已经如实反映当前可命令系统**，不需要"限定条件"式的打折；我 S205 写的"应限定为在当前重建覆盖下不可行"容易被读成"LP 偏乐观"，**这个方向是错的**。
- 正确的表述：**当前可命令系统的静态不可行性是成立的**；S205 的缺口意味着**要修就得增加可控列（补齐覆盖或声明的镜像映射），这只会放宽 LP，不会收紧**——但那是对"修好之后"的假设，不能拿来给现在的数打折。

## 5. 这**不是**什么
- 没有因此声明任何身体能力或步态结果；仍是同一个冻结位姿上的**静态**平衡问题。
- 不能推断"补齐 16 行就能站起来"：补齐只是**多出控制列**，LP 可能仍然不可行（S196 已给出同时性耦合的证据）。
- 力单位与生物有效性仍 **NOT_ESTABLISHED**；仍非站立/行走/自主运动。

## 6. 下一步最小动作
1. 若要认真评估"补齐覆盖能带来多少"：把补齐后的 `T` 作为**假设矩阵**（明确标注为假设、不写入冻结产物）跑同一条 LP，给出 minimax 随新增列数的变化曲线——这属于**假设推演**，必须与实测分开记账。
2. 或者转向 S196 已指出的耦合壁垒（腿间运动学耦合/simultaneity），那条线不依赖覆盖补齐。

## 7. 记账
- `PREREG.json`（本轮由同一进程在求解前写入，`written_before_analysis=true`）、`COMMANDABILITY_GAP_LP.json`、`INDEPENDENT_VERIFICATION.json`（{n_ok}/{len(checks)} 通过）、`ATTEMPT_LEDGER.json`（3 条自身失败 + 1 条对 S205 的更正 + 1 条设计说明）。
- 预算：模型加载 0、forward 0、mj_step 0、LP 6 次；0 训练、0 预热、0 XML 改动。
"""
(DIR / 'CONCLUSION.zh-CN.md').write_text(conclusion)

def sha_all(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

budget = {"round": "S206",
          "stop_reason": ("The frozen LP was re-solved with and without the 16 uncommandable actuators; the baseline gate "
                          "reproduced exactly and the answer is unambiguous (identical minimax). No further arm is needed, so the "
                          "round stops. No physics was run."),
          "budget": {"model_loads": 0, "forward_calls": 0, "mj_step_calls": 0, "lp_solves": 6,
                     "neural_ticks": 0, "training_steps": 0, "prewarming": False, "xml_changes": 0},
          "artifacts": sha_all(DIR),
          "sources": {str(p): sha(p) for p in [Path('/run/media/sensen/Data2/cell_wound_prototype/s206_commandability_lp.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/verify_s206.py'), XML]
                      if p.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
(DIR / 'SOURCE_HASHES.json').write_text(json.dumps(dict(sources=budget['sources'], artifacts=budget['artifacts']), indent=2))
print(json.dumps(dict(dir=str(DIR), n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
                      gate=mm_all1, mm_cmd1=mm_cmd1, mm_all2=mm_all2, mm_cmd2=mm_cmd2,
                      failed=[c for c in checks if not c['ok']]), indent=1, default=str))
