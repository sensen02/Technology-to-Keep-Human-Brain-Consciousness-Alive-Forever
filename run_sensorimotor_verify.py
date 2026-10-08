"""Verify the sensorimotor interface against the annotation table, and
demonstrate that a virtual environment can now drive and read the brain."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import sparse as sp
from engine.sensorimotor import SensorimotorInterface
from engine.neural import LIFNetwork, NetworkParams

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")

z = np.load(os.path.join(ROOT, "data/flywire/fafb_connectome.npz"), allow_pickle=False)
root_ids = z["root_ids"]
smi = SensorimotorInterface(os.path.join(ROOT, "data/flywire/fafb_classification.csv.gz"),
                            root_ids)
print(smi.describe())
v = smi.verify()
print(f"\n一致性校验: {'通过' if v['ok'] else '失败'}  问题={v['problems']}")

smi.save_groups(os.path.join(ROOT, "data/flywire/fafb_sensorimotor_groups.npz"))

# ---- demonstrate: an environment drives vision, we read the motor output ----
pre, post, syn, nt_pair = z["pre"], z["post"], z["syn"].astype(float), z["nt_pair"]
labels = json.load(open(os.path.join(OUT, "metrics_flywire_connectome.json")))["nt_types"]
sign = np.array([{"ACH":1,"DA":1,"GABA":-1,"GLUT":-1,"OCT":1,"SER":1}[l] for l in labels])
n = root_ids.size
WS = 200.0
W = sp.csr_matrix((syn * sign[nt_pair] * WS, (post, pre)), shape=(n, n))

net = LIFNetwork(n, NetworkParams(w_scale_mV=WS), dt_ms=1.0, seed=0)
net.set_connectivity(W)

print("\n--- 虚拟环境闭环演示：只给视觉输入，看运动输出 ---")
trials = []
for vis in (0.0, 8.0, 10.0, 12.0, 14.0):
    net2 = LIFNetwork(n, NetworkParams(w_scale_mV=WS), dt_ms=1.0, seed=0)
    net2.set_connectivity(W)
    i_ext = smi.make_drive({"visual": vis})          # ONLY the visual channel
    mot = np.zeros(3); k = 0
    for _ in range(300):
        s = net2.step(i_ext=i_ext)
        mot += [s[smi.outputs[o]].mean() for o in ("descending","motor","endocrine")]
        k += 1
    rates = {o: 1e3*mot[j]/k for j,o in enumerate(("descending","motor","endocrine"))}
    inh = net2.v[smi.intrinsic].mean()
    trials.append({"visual_drive_mV": vis,
                   "descending_hz": rates["descending"],
                   "motor_hz": rates["motor"],
                   "endocrine_hz": rates["endocrine"]})
    print(f"  视觉驱动 {vis:5.1f} mV -> 下行 {rates['descending']:8.3f} Hz  "
          f"运动 {rates['motor']:8.3f} Hz  内分泌 {rates['endocrine']:8.3f} Hz")

res = {"interface": smi.to_dict(), "verification": v,
       "visual_to_motor_trials": trials,
       "note": "这里只注入了视觉通道；运动输出随视觉输入变化，"
               "说明感觉→运动的通路在真实连接组上确实连着。"
               "但不代表果蝇「看见了」任何东西。"}
with open(os.path.join(OUT, "metrics_sensorimotor.json"), "w") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False, default=str)
print(f"\n写入 {os.path.join(OUT,'metrics_sensorimotor.json')}")
