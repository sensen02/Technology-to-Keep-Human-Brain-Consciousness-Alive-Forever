"""run_engine_selftest.py -- verification of the engine core itself.

Half of these tests are NEGATIVE: they check that the engine REFUSES things it
is supposed to refuse.  A guard that never fires is not a guard, and this
project has already been bitten by silent failures (a unit conversion off by
1e6; chemotaxis that never fired because the source term was missing, yet
produced bit-identical results and so looked like a null result rather than a
bug).  So every guard gets a test that it actually raises.

Writes: outputs/metrics_engine_selftest.json
        outputs/ENGINE_SELFTEST.zh-CN.md
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (Quantity, Q, UnitError, convert, ParamRegistry, Param,
                    ProvenanceError, MEASURED, DERIVED, ILLUSTRATIVE,
                    Layer, Pipeline, FeedforwardViolation, WiringError,
                    ConstantSource, Integrator, RatioMonitor,
                    DROSOPHILA_NOTUM, HYDRA, MAMMALIAN_TISSUE, ProfileViolation,
                    CheckSuite, RunRecord)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

results = []


def rec(name, kind, passed, detail="", error=None, tolerance=None):
    results.append({"name": name, "kind": kind, "passed": bool(passed),
                    "error": error, "tolerance": tolerance, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")


def expect_raises(name, exc, fn, why):
    """NEGATIVE test: fn() must raise exc."""
    try:
        fn()
    except exc as e:
        rec(name, "invariant", True, f"{why} -> 按预期抛出 {exc.__name__}: {str(e)[:90]}")
        return True
    except Exception as e:
        rec(name, "invariant", False,
            f"{why} -> 抛出了错误的异常 {type(e).__name__}: {e}")
        return False
    rec(name, "invariant", False, f"{why} -> 没有抛出任何异常（守卫失效）")
    return False


# ======================================================================
# 1. units
# ======================================================================
print("=== 1. 单位系统 ===")

# The exact bug this project hit: m^2 -> um^2 with 1e6 instead of 1e12.
v = convert(1.0, "um^2", "m^2")
rec("area_conversion_um2_to_m2", "analytic", abs(v - 1e-12) < 1e-24,
    f"1 µm² = {v:.3e} m²（应为 1e-12；项目曾误用 1e6 得到 1e-6）",
    error=abs(v - 1e-12), tolerance=1e-24)

# and the diffusion constant that depends on it
D = Q(1e-13, "m^2/s")
rec("diffusion_m2s_to_um2s", "analytic", abs(D.in_unit("um^2/s") - 0.1) < 1e-15,
    f"1e-13 m²/s = {D.in_unit('um^2/s'):.4g} µm²/s（应为 0.1）")

# diffusion length of the Merks model must come out at exactly 10 um
Dl = Q(0.1, "um^2/s")
eps = Q(1e-3, "1/s")
L = (Dl / eps) ** 0.5
rec("merks_diffusion_length", "analytic", abs(L.in_unit("um") - 10.0) < 1e-9,
    f"sqrt(D/ε) = {L.in_unit('um'):.6f} µm（原文 10 µm）",
    error=abs(L.in_unit("um") - 10.0), tolerance=1e-9)

rec("pressure_conversion", "analytic", abs(convert(1.0, "mmHg", "Pa") - 133.322) < 1e-9,
    f"1 mmHg = {convert(1.0, 'mmHg', 'Pa'):.3f} Pa")

expect_raises("adding_incompatible_units", UnitError,
              lambda: Q(1, "um") + Q(1, "s"), "把 µm 与 s 相加")
expect_raises("converting_incompatible_units", UnitError,
              lambda: Q(1, "um").to("s"), "把 µm 转成 s")
expect_raises("unknown_unit_rejected", UnitError,
              lambda: Q(1, "furlong"), "使用未知单位 furlong")
rec("mul_div_dimensions", "analytic",
    (Q(2, "um") / Q(4, "s")).dims == Q(1, "um/s").dims,
    "µm/s 的量纲运算正确")

# ======================================================================
# 2. params
# ======================================================================
print("\n=== 2. 参数出处强制 ===")

expect_raises("measured_without_source_rejected", ProvenanceError,
              lambda: Param("x", 1.0, "um", MEASURED),
              "创建 provenance=measured 但没有引用来源的参数")
expect_raises("derived_without_parents_rejected", ProvenanceError,
              lambda: Param("y", 1.0, "um", DERIVED),
              "创建 provenance=derived 但没有 derived_from 的参数")
expect_raises("value_outside_range_rejected", ProvenanceError,
              lambda: Param("z", 99.0, "um", ILLUSTRATIVE, range=(0.0, 10.0)),
              "参数值超出自己声明的范围")

reg = ParamRegistry("t")
reg.define("cell_area", 55.6, "um^2", MEASURED,
           source="Curran 2017, notum 12-13.5 hAPF, 222.3/4 um^2",
           species="Drosophila melanogaster", stage="pupal notum, 12-13.5 hAPF")
reg.define("h", 1.0, "um", DERIVED, derived_from=["cell_area"])
reg.define("lambda_chem", 100.0, "1", ILLUSTRATIVE)

expect_raises("unregistered_parameter_rejected", KeyError,
              lambda: reg["never_declared"],
              "读取一个从未声明的参数")

reg2 = ParamRegistry("bad")
reg2.define("a", 1.0, "um", DERIVED, derived_from=["ghost"])
expect_raises("derived_from_missing_parent", ProvenanceError, reg2.validate,
              "derived 参数引用了不存在的父参数")

def _species_guard():
    m = ParamRegistry("m")
    m.add(Param("tension", 44.0, "pN", MEASURED, source="Bambardekar 2015",
                species="Mus musculus"))
    return m.check_species("Drosophila melanogaster")


expect_raises("species_mixing_rejected", ProvenanceError, _species_guard,
              "果蝇模型里使用小鼠实测参数（不允许跨物种）")

stage_off = reg.check_stage("pupal notum, 18-26 hAPF")
rec("stage_mixing_flagged", "invariant", len(stage_off) >= 1,
    f"阶段不一致被抓出：{stage_off}")

sm = reg.summary()
rec("provenance_summary", "invariant",
    sm["_total"] == 3 and sm[MEASURED] == 1 and sm[ILLUSTRATIVE] == 1,
    f"出处统计 {sm}")
expect_raises("require_measured_on_illustrative", ProvenanceError,
              lambda: reg.require_measured(["lambda_chem"]),
              "把一个 illustrative 参数当作 measured 声称")

# ======================================================================
# 3. layers
# ======================================================================
print("\n=== 3. 分层与单向约束 ===")


class VesselSource(Layer):
    name = "vessel"
    provides = {"vessel_source": "mol/m^3"}
    requires = {}

    def step(self, dt, inputs):
        return {"vessel_source": 1.0}


class TissueO2(Layer):
    name = "tissue_o2"
    requires = {"vessel_source": "mol/m^3"}
    provides = {"o2": "mol/m^3"}

    def step(self, dt, inputs):
        return {"o2": float(inputs["vessel_source"]) * 0.5}


class Growth(Layer):
    name = "growth"
    requires = {"o2": "mol/m^3"}
    provides = {"biomass": "kg"}

    def step(self, dt, inputs):
        self.state["m"] = self.state.get("m", 0.0) + 1e-18 * float(inputs["o2"])
        return {"biomass": self.state["m"]}


pl = Pipeline("ok")
pl.add(VesselSource()).add(TissueO2()).add(Growth())
pl.wire()
rec("feedforward_valid_stack_wires", "invariant", True,
    f"合法三层栈接线成功：{sorted(pl.ports())}")

# ordering violation: the root needs a port only a LATER layer provides
bad = Pipeline("backward")
bad.add(TissueO2())      # requires vessel_source
bad.add(VesselSource())  # provides it, but too late
expect_raises("forward_read_rejected", FeedforwardViolation, bad.wire,
              "某层读取了「后面那层」才提供的端口")

missing = Pipeline("missing")
missing.add(TissueO2())
expect_raises("missing_port_rejected", WiringError, missing.wire,
              "某层要求的端口无人提供")

unit_bad = Pipeline("unitmismatch")
unit_bad.add(VesselSource())
unit_bad.add(Growth())   # requires o2 in mol/m^3, nothing provides o2
expect_raises("unprovided_typed_port_rejected", WiringError, unit_bad.wire,
              "要求的带单位端口无人提供")

unit_bad2 = Pipeline("unitmismatch2")


class WrongUnit(Layer):
    name = "wrong_unit"
    requires = {"vessel_source": "s"}   # vessel_source is mol/m^3 -> mismatch
    provides = {"o2x": "mol/m^3"}

    def step(self, dt, inputs):
        return {"o2x": 0.0}


unit_bad2.add(VesselSource()).add(WrongUnit())
expect_raises("port_unit_mismatch_rejected", WiringError, unit_bad2.wire,
              "端口单位不兼容（mol/m³ vs s）")


class Snooper(Layer):
    """Tries to read a real state key that it did NOT declare."""
    name = "snooper"
    requires = {"o2": "mol/m^3"}
    provides = {"snooped": "1"}

    def step(self, dt, inputs):
        self.saw = sorted(inputs.keys())
        leaked = inputs.get("vessel_source", "ABSENT")
        return {"snooped": 0.0 if leaked == "ABSENT" else 1.0}


sp = Pipeline("snoop")
sn = Snooper()
sp.add(VesselSource()).add(TissueO2()).add(sn)
st, _ = sp.run(Q(1.0, "s"), dt_outer=Q(1.0, "s"))
rec("layer_cannot_read_undeclared_port", "invariant",
    sn.saw == ["o2"] and st["snooped"] == 0.0,
    f"未声明端口无法读取：inputs={sn.saw}，vessel_source 对它是 ABSENT")

# per-layer time stepping must not change a linear integral
def integral_under_dt(dt_outer_s):
    p = Pipeline("dt")
    p.add(ConstantSource("src", "1", 2.0)).add(
        Integrator("i", "src", "1", "x", "1", rate=3.0))
    p.layers[1].target_dt = Q(0.01, "s")
    s, _ = p.run(Q(10.0, "s"), dt_outer=Q(dt_outer_s, "s"))
    return s["x"]


a = integral_under_dt(10.0)
b = integral_under_dt(0.5)
c = integral_under_dt(0.01)
rec("per_layer_substepping_exact", "analytic",
    abs(a - 60.0) < 1e-9 and abs(b - 60.0) < 1e-9 and abs(c - 60.0) < 1e-9,
    f"∫2·3 dt over 10 s = {a:.6f}/{b:.6f}/{c:.6f}（外层 dt 变化不影响，应为 60）",
    error=max(abs(a - 60), abs(b - 60), abs(c - 60)), tolerance=1e-9)

# ======================================================================
# 4. profile guards
# ======================================================================
print("\n=== 4. 生物档案守卫 ===")

dl = Pipeline("drosophila_with_vessel")
dl.add(VesselSource()).add(TissueO2()).add(Growth())
expect_raises("drosophila_rejects_vessel", ProfileViolation,
              lambda: DROSOPHILA_NOTUM.validate_pipeline(dl),
              "给果蝇档案接上血管源层")
expect_raises("hydra_rejects_vessel", ProfileViolation,
              lambda: HYDRA.validate_pipeline(dl),
              "给水螅档案接上血管源层")
rec("mammal_accepts_vessel", "invariant",
    MAMMALIAN_TISSUE.validate_pipeline(dl) is True,
    "哺乳动物档案允许血管源（类别正确）")

rec("drosophila_caveats_enforced", "invariant",
    len(DROSOPHILA_NOTUM.required_caveats()) >= 3,
    f"果蝇档案强制携带 {len(DROSOPHILA_NOTUM.required_caveats())} 条必读警示")

rec("drosophila_is_the_original_organism", "invariant",
    DROSOPHILA_NOTUM.species.startswith("Drosophila"),
    "记录核实的原始实验体 = 果蝇蛹背板上皮（PROTOCOL.md 第一行）")

# ======================================================================
# 4b. declarative rules
# ======================================================================
print("\n=== 4b. 声明式规则 ===")

from engine import Rule, RuleSet, RuleError

rs = RuleSet("notum")
rs.add("cell_o2 < 5 mmHg", "cellstate.death_rate", "set", "0.05 1/h",
       basis="illustrative", note="低氧死亡阈值，由我们选定")
rs.add("cell_o2 > 20 mmHg", "growth.cycle_rate", "scale", "2.0 1",
       basis="illustrative")

expect_raises("rule_measured_without_source_rejected", RuleError,
              lambda: Rule("x < 1 mmHg", "t", "set", "1 1/h", basis="measured"),
              "规则声称 measured 但无引用")

expect_raises("rule_bad_grammar_rejected", RuleError,
              lambda: Rule("oxygen is low", "t", "set", "1 1/h"),
              "条件语法无法解析")

expect_raises("rule_unknown_port_rejected", RuleError,
              lambda: RuleSet("r").add("nonexistent_port < 1 mmHg", "t", "set",
                                       "1 1/h").compile({"cell_o2": "mmHg"}),
              "规则引用了没有层提供的端口（该规则永远不会触发）")

expect_raises("rule_unit_mismatch_rejected", RuleError,
              lambda: RuleSet("r").add("cell_o2 < 5 s", "t", "set", "1 1/h")
              .compile({"cell_o2": "mmHg"}),
              "规则把 mmHg 和 s 相比")

rs.compile({"cell_o2": "mmHg", "growth.cycle_rate": "1/h"})
fired_low = rs.evaluate({"cell_o2": 3.0})
fired_high = rs.evaluate({"cell_o2": 30.0})
rec("rule_fires_on_correct_side", "invariant",
    "cellstate.death_rate" in fired_low and "growth.cycle_rate" in fired_high
    and "cellstate.death_rate" not in fired_high,
    f"低氧触发死亡规则={sorted(fired_low)}；高氧触发增殖规则={sorted(fired_high)}")

ap = rs.apply({"cell_o2": 30.0}, {"growth.cycle_rate": 0.5})
rec("rule_effects_apply", "analytic", abs(ap["growth.cycle_rate"] - 1.0) < 1e-12,
    f"scale 作用正确：0.5 x 2.0 = {ap['growth.cycle_rate']}")

ann = rs.annotate()
rec("rule_annotation_autogenerated", "invariant",
    "自动生成" in ann and "cell_o2 < 5 mmHg" in ann,
    "规则文档自动生成，不依赖手写")

rec("rule_provenance_summary", "invariant",
    rs.provenance_summary()["_fraction_not_measured"] == 1.0,
    f"规则出处统计：{rs.provenance_summary()}")


# ======================================================================
# 5. checks harness
# ======================================================================
print("\n=== 5. 检查框架 ===")

suite = CheckSuite("harness_demo")
suite.analytic("passes", lambda: (1e-9, 1e-6), "故意通过")
suite.conservation("fails", lambda: (1.0, 1e-12), "故意失败")
res = suite.run()
s = suite.summary(res)
rec("check_suite_counts_failures", "invariant",
    s["n_checks"] == 2 and s["n_passed"] == 1 and s["n_failed"] == 1
    and not s["all_passed"],
    f"检查框架正确统计失败：{s['n_checks']} 项 / 通过 {s['n_passed']}")
rec("check_suite_disclaims_biology", "invariant",
    "不说明模型在生物学上正确" in s["disclaimer"],
    "检查框架自带「通过≠生物学正确」声明")
threw = False
try:
    suite.require_all_passed(res)
except Exception:
    threw = True
rec("require_all_passed_raises", "invariant", threw,
    "存在失败项时 require_all_passed 会抛错")

# ======================================================================
# 6. run record
# ======================================================================
print("\n=== 6. 运行记录 ===")
rec_obj = RunRecord("engine selftest", root=os.path.dirname(os.path.abspath(__file__)),
                    seed=0)
rec_obj.set_profile(DROSOPHILA_NOTUM).set_parameters(reg).set_wiring(pl)
rec_obj.set_checks(suite, res)
p = rec_obj.write(os.path.join(OUT, "engine_selftest_record.json"))
rec("record_contains_environment", "invariant",
    rec_obj.data["environment"]["python"] is not None, "记录含运行环境")
rec("record_contains_caveats", "invariant",
    len(rec_obj.data["caveats"]) >= 3, "记录自动携带生物档案警示")
rec("record_hash_stable", "determinism",
    rec_obj.hash() == rec_obj.hash(), f"记录哈希稳定: {rec_obj.hash()}")
rec("record_written", "invariant", os.path.exists(p), f"写入 {p}")

# ======================================================================
npass = sum(1 for r in results if r["passed"])
n = len(results)
by_kind = {}
for r in results:
    d = by_kind.setdefault(r["kind"], [0, 0])
    d[0] += 1
    d[1] += int(r["passed"])

summary = {
    "n_tests": n, "n_passed": npass, "n_failed": n - npass,
    "all_passed": npass == n,
    "by_kind": {k: {"n": v[0], "passed": v[1]} for k, v in by_kind.items()},
    "n_negative_tests": sum(1 for r in results if r["kind"] == "invariant"),
    "results": results,
    "what_this_proves": "引擎的守卫确实会触发、单位与出处强制生效、"
                        "分层单向约束不可绕过",
    "what_this_does_not_prove": "不证明任何生物学结论；不证明所接模块的模型正确",
}
with open(os.path.join(OUT, "metrics_engine_selftest.json"), "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

lines = ["# 引擎自检报告", "",
         f"共 {n} 项测试，通过 {npass}，失败 {n - npass}；"
         f"其中**否定测试 {summary['n_negative_tests']} 项**（验证守卫会触发）。", "",
         "| 测试 | 类型 | 通过 | 说明 |", "|---|---|---|---|"]
for r in results:
    lines.append(f"| `{r['name']}` | {r['kind']} | {'✅' if r['passed'] else '❌'} | "
                 f"{r['detail']} |")
lines += ["", "> 本自检只证明引擎自身的守卫与约束生效；不证明任何生物学结论。"]
with open(os.path.join(OUT, "ENGINE_SELFTEST.zh-CN.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\n=== {npass}/{n} 通过（否定测试 {summary['n_negative_tests']} 项）===")
if npass != n:
    print("失败项：", [r["name"] for r in results if not r["passed"]])
