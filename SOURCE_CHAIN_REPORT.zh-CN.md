# S163/P1/P2 独立源码—实际保存状态链审计

## 范围与证据身份

仅阅读冻结源码、JSON 与 numpy 保存数组，不导入项目模块、不创建神经网络/模型，不执行 neural step、model load、forward 或 physics step。现有生产文件/历史归档不改写。新脚本 `/run/media/sensen/Data2/cell_wound_prototype/s163_source_audit_saved_only.py` 只使用标准库与 numpy；实际执行退出码 0。父代理负责全局图表/聚合/追踪记录。

基线 `/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z`；提前感觉刷新臂 `/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z`。精确输入 SHA256、完整元数据、按行引用见 `/run/media/sensen/Data2/cell_wound_prototype/s163_neural_source_audit.json` 与本目录同名副本；本报告末尾附冻结源码逐字引文及 SHA256。不是用当前工作树源码替代历史版本。

## 1. 实际注入电流没有被膜积分忽略/覆盖

实际 runner `/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py:340–357` 依序从已积分到当前时刻的身体观察、编码、`currents.append(current.copy())`、`net.step(current_nA=current)`。保存 current 与传给 step 的是同一向量值，不是仅记录但未使用。357–366 行先 step，再把二值 spikes 送入 FIR，最终 `d.ctrl[:] = 0.5 * u`。没有把 actual current 改回 initial current 的后置赋值。

编码器 `/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/BOUND_INPUT_FREEZE/0087_sensory_channels.py:261–297` 每次从零电流向量开始；逐组 `current[idx] += cfg.current_nA_per_unit * float(v)`，不是逐组覆写。signed velocity 允许负电流，0.3 nA/unit 的比例为接口假设而非生理校准。320–352 行通过 root→local 映射，将不在 tier 的细胞记为 unmatched，而非重分配到其他细胞。runner 185–188 行从实际 tier global index 得到 roots 后建立 group。实际保存的 ENCODER_PARAMETERS groups 和 current 是核查对象，不把注释当实际数值。

## 2. 真正的膜方程、更新顺序与单位

实际 `/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/BOUND_INPUT_FREEZE/0049_neural_cond.py` 顶部明确：`C[nF] dV[mV]/dt[ms] = g[uS]*(E-V)[mV] + I[nA] - adaptation[nA]`。因此 V 是 mV，dt=.5 是 ms，C=.02 是 nF；`C/dt=.04`，不能再把电流按秒换算一次。

元数据两臂相同：cold steps0/t0/ring_i0；PRE/release steps160/t80ms/ring_i0；final steps165/t82.5ms/ring_i0；n17099、delay_steps4、voltage_method=`backward_euler`。所有参数：C .02nF、leak .001uS、rest/reset −65mV、threshold −50mV、Eexc0/Einh−70mV、tau_exc3ms/tau_inh8ms、tau_adapt100ms、每 spike adaptation+.002nA、refractory2ms、delay2ms。它们在源码被标作 illustrative/uncalibrated，不能称真实果蝇阈值或膜参数。

实际 step 顺序（119–155 行）：
1. 取 ring 当前槽的 arrivals 并清槽，`ge += We @ arrivals + ee`，`gi += Wi @ arrivals + ei`；矩阵索引 post,pre。
2. interval START 的活动资格 `own & (t_ms+1e-10 >= refractory_until_ms)`。
3. G=leak+ge+gi，`rhs=C/dt*v + leak*rest + ge*Eexc + gi*Einh + I - adaptation`。
4. 当前实际选 backward Euler，`v_active = rhs/(C/dt+G)`；不活动的 owned 细胞钳至 reset。
5. 仅 active 细胞以候选 v>=−50 判 spike；时间推进到 END，然后 spike 细胞重置 v=−65，设置 refractory_until=END+2ms，累计 count。
6. adaptation 先 exp(−dt/100) 衰减再给本次 spike 加 .002；ge/gi 分别 exp(−dt/3)、exp(−dt/8) 衰减；将本次 spike 写回已消费槽，ring index+1。

这不是 exponential integrate-and-fire：无 exp(V) 尖峰项。存在的指数是电导/适应衰减；可选 frozen_exact 电压分支虽用 expm1，但本次没有选它，且源码说明系数冻结精确不等于连续电导网络精确。threshold 数字固定 −50，不存在动态 threshold 参数；动态因素是 refractory 资格和 adaptation 电流，应避免把二者误说成阈值升降。

ring 长度 delay_steps+1=5 是区分 spike END 与接收 START。某步末发出的 spike，在其后4个完整.5ms区间后、相距2ms的 START 接收；不是读完 immediately 回馈同一步，也不能因长度5就称 delay2.5ms。首个改变的 spike 若在释放神经 tick1 末81ms才发出，最早到83ms START，已经超出最后已执行的 tick4 END82.5ms。此为延迟窗口约束，不是声称本次存在改变的 spike。

## 3. 最大 39.691282mV 的细胞到底是什么

actual NEURAL_FINAL 两臂全数组比较最大差 local index945，已保存 encoder group 是 RF / hook_direction，不在206个实际运动 readout union 中。另一独立 tier-index 审计重建n17099并通过manifest核验，父代理交叉确认精确 root_id=`720575941430177161`、primary type SNpp41。不能用 lookup 序号猜：该通道 lookup26个而 tier25个，直接对齐序号可能错。

| 实际项 | S154 | S162 |
|---|---:|---:|
| terminal v mV | −65 | −104.6912820402041 |
| ge / gi uS | 0 / 0 | 0 / 0 |
| adaptation nA | 0 | 0 |
| refractory_until ms | 0 | 0 |
| PRE count / final cumulative count | 0 / 0 | 0 / 0 |
| 五个 release spikes | 全 false | 全 false |
| 五个 current nA | 0,0,0,0,0 | 0,−.18363657862259863,−.38701822970297733,−.5122622626127409,−.5886801612101807 |
| terminal algebraic threshold−v mV | 15 | 54.6912820402041 |

S162−S154=−39.6912820402041mV：是更负的超极化，不是朝阈值推进39.7mV；也不是39.7V。S154 −65 的来源不能仅凭 reset 数字判断，但该细胞保存 PRE/final count0、refractory_until0，明确无 spike reset 历史证据，不得把它说成刚发 spike 被 reset。全群体不能套用该细胞结论。

## 4. 实际 spike 与 readout 距阈值的有界结论

两臂 PRE160 总 spike70496；五个 release tick 全网络 spike 数 `[83,20,11,2884,17]`。actual readout206个唯一细胞 PRE总256，release `[3,3,1,1,1]`。因此“实际 spikes 相同”绝不是“阈值始终没触发/全网没放电”；无差异指两臂逐细胞逐 tick 的二值事件一致，不是否认事件。

逐206个 readout、两臂终态 CSV `/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z/SOURCE_READOUT_FINAL_STATE.csv` 给出 local/root/pool、actual final v/ge/gi/adaptation/refractory、PRE/release/final count 与最后一 tick spike。两臂终态 readout v范围 −68.72076000965254 到 −50.10027421823974mV，代数 gap .10027421823973981 到18.720760009652537mV。82.5ms START（尚未执行的下一步）有200个满足 refractory eligibility、6个仍 refractory；adaptation范围0至.0184122716604413nA。CSV 的 eligible_next_start 只比较保存终态 refractory_until 与82.5，绝不预测下一步 spike。

正确解释：这是 **最后已执行 step 完成 reset 后** 的真实保存电压到固定阈值的代数间隙，不能当所有 tick 的 pre-reset threshold-crossing margin，不能当“只差.1mV再加一点就会发”的可执行预测。动态 refractory 决定能否参与阈值判断，adaptation/ge/gi 影响下一次候选膜积分。最后一步的 spike 会让终态 v变−65并修改 refractory_until；用最终 refractory 反推上一步开始时是否 active 会错，CSV刻意不含这类反推字段。

## 5. FIR / 66×5 / 90 输出 / clipping 的因果位置

实际 `/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/BOUND_INPUT_FREEZE/0104_motor_translation.py:12–59`：manifest的非空 local_indices 去重排序成66个 pools；206个唯一细胞可能跨共享池。90 actuator slots 中74 mapped，16无池输出固定零，不是66个独立神经元直接对应90肌肉。

每 tick pool activity=`mean(binary_spikes)/.0005/100`，把最新值放 history前端，history80tick=40ms；5 bins `[0,5),[5,10),[10,20),[20,40),[40,80)` 即2.5/2.5/5/10/20ms段。features均值乘66×5 weights并求和得到 unclipped pool drive，随后 clip[0,1]，映射到90路，再统一gain.5→[0,.5]。actual weights范围 .08979055178021957–.18483479399906344，非名称推断的默认 [.4,.25,.2,.1,.05]。两臂 PRE+release165 tick 从保存 spikes 独立重算 FIR误差0；saved features×weights→clip→actuator mapping→decoded误差0。五 tick pool drive最大1.0045682403373766、最小0，有3个 pool×tick >1，decoded[0,1]、applied[0,.5]。

确有少量 clipping 不等于它造成感觉无法改变运动。两臂差异在 current/final v之后，**最早已保存可比较的离散边界——全网络二值 spikes——已经完全相同**，后续 FIR features也相同，clipping前的 pool drive亦相同。不能用更下游的 clip 或 gain 解释一个在它们前面已经不存在的差异；也不能称“全部输出饱和”。clipping的潜在 future作用未测试。

独立实际数组再次核对：206 readout 的 final `v/ge/gi/adaptation/refractory_until_ms/spike_count` 两臂逐项 maxabs差均0，不只是spikes相同。父代理独立差分识别552个曾改变输入的细胞，其五tick spike数 `[60,0,0,0,0]`，60个全发生在tick0、第一次刷新之前；刷新后这些细胞没有已保存的发放事件。不能将readout输出未变解释成“readout收到不同电流却被其阈值挡住”：readout未直接收到变化电流，实际终态六项亦完全相同。图路径可达不等于本短窗已产生不同事件传播。

## 6. 不能从现有归档恢复的内容

actual TRAJECTORY 保存85个身体样本和5个 current/spike/features/decoded/applied；没有逐 tick neural v、ge、gi、adaptation、refractory_until。NEURAL_COLD/INITIALIZATION/NEURAL_FINAL 分别提供cold、PRE结束、终态；STOP_FULL_STATE为身体状态，不是每 tick 神经候选膜电位。脚本 JSON 有实际数组 key inventory。

故不恢复、不重跑、不生成伪 per-tick v 曲线；不从相同 spikes 推断全过程阈值从未触发；不声称排除了数值/生理机制；不认为终态最大差就是运动 readout 差。可确认：真实电流注入与部分实际终态膜电位改变存在，但本5tick短窗未改变已保存的输出事件，后续 FIR/译码自然相同。更长延迟传播、生理效力、失撑根因、行走/预热成功均未建立。

## 附录：冻结源 SHA256 与逐字行引文

### `/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/BOUND_INPUT_FREEZE/0049_neural_cond.py`

SHA256 `f4e6d5af5473a187f3da521273e606d726bd5df6272df214457e14e9aa289499`

```text
1: """Conductance point tier, explicit units and illustrative parameters.
2: 
3: C[nF] dV[mV]/dt[ms] = g[uS]*(E-V)[mV] + I[nA] - adaptation[nA].
4: Default backward Euler voltage, optional frozen-coefficient exact voltage.
5: Both split exact exponential conductance/adaptation decay and end-step spikes;
6: frozen_exact is not an exact continuous-time conductance-network solution.
7: Threshold/reset and adaptation are reductions, NOT calibrated fly physiology.
8: Matrices carry nonnegative conductance increments, indexed (post, pre).
9: No transmitter-to-sign inference: caller must supply receptor assumptions.
```

```text
17: class ConductanceParams:
18:     capacitance_nF: float = 0.02
19:     leak_uS: float = 0.001
20:     rest_mV: float = -65.0
21:     threshold_mV: float = -50.0
22:     reset_mV: float = -65.0
23:     exc_reversal_mV: float = 0.0
24:     inh_reversal_mV: float = -70.0
25:     exc_tau_ms: float = 3.0
26:     inh_tau_ms: float = 8.0
27:     adaptation_tau_ms: float = 100.0
28:     adaptation_increment_nA: float = 0.002
29:     refractory_ms: float = 2.0
30:     delay_ms: float = 2.0
```

```text
63:         d = self.p.delay_ms / self.dt
64:         if d < 1 or not np.isclose(d, round(d), rtol=0, atol=1e-9):
65:             raise ValueError('delay must be an integer number of steps, at least one')
66:         self.delay_steps = int(round(d))
67:         self._ring = np.zeros((self.delay_steps + 1, self.n), dtype=bool)
68:         self._ring_i = 0
69:         self.v = np.full(self.n, self.p.rest_mV)
70:         self.ge = np.zeros(self.n)
71:         self.gi = np.zeros(self.n)
72:         self.adaptation = np.zeros(self.n)
73:         self.refractory_until_ms = np.zeros(self.n)
74:         self.spike_count = np.zeros(self.n, dtype=np.int64)
75:         self.We = sparse.csr_matrix((self.n, self.n))
76:         self.Wi = sparse.csr_matrix((self.n, self.n))
77:         self.t_ms = 0.0
78:         self.steps = 0
```

```text
99:     def step(self, current_nA=0.0, exc_events_uS=0.0, inh_events_uS=0.0,
100:              enabled=None):
101:         """Events arrive at interval START, returned spikes at interval END.
102: 
103:         A spike emitted at t arrives at t+delay. External increments are immediate.
104:         ``enabled`` assigns point-tier ownership; disabled cells do not spike or
105:         integrate. Incoming events to disabled targets are discarded, so a hybrid
106:         coordinator must route those edges to their detailed owner instead.
107:         Disabled voltage/adaptation are frozen. This is a fixed-ownership API,
108:         not a state-transfer protocol for dynamic coarse/fine switching.
109:         """
110:         I = self._vector(current_nA, 'current')
111:         ee = self._vector(exc_events_uS, 'exc events', True)
112:         ei = self._vector(inh_events_uS, 'inh events', True)
113:         own = np.ones(self.n, bool) if enabled is None else np.asarray(enabled)
114:         if own.shape != (self.n,) or own.dtype != np.dtype(bool):
115:             raise ValueError('ownership mask must be a boolean vector of neuron count')
116:         p, dt = self.p, self.dt
117:         # Slot at step k contains spikes emitted at end of step k-delay-1.
118:         # Reuse the consumed slot in a delay+1 ring: end->start delay is exact.
119:         arrivals = self._ring[self._ring_i].copy()
120:         self._ring[self._ring_i] = False
121:         self.ge += self.We @ arrivals + ee
122:         self.gi += self.Wi @ arrivals + ei
123:         self.ge[~own] = 0
124:         self.gi[~own] = 0
125:         active = own & (self.t_ms + 1e-10 >= self.refractory_until_ms)
126:         G = p.leak_uS + self.ge + self.gi
127:         rhs = (p.capacitance_nF / dt * self.v + p.leak_uS * p.rest_mV
128:                + self.ge * p.exc_reversal_mV + self.gi * p.inh_reversal_mV
129:                + I - self.adaptation)
130:         if self.voltage_method == 'backward_euler':
131:             self.v[active] = (rhs / (p.capacitance_nF / dt + G))[active]
132:         else:
133:             # Exact only for coefficients frozen at interval start. Conductance
134:             # decay, adaptation and end-step threshold events remain split.
135:             drive = (p.leak_uS * p.rest_mV + self.ge * p.exc_reversal_mV
136:                      + self.gi * p.inh_reversal_mV + I - self.adaptation)
137:             equilibrium = drive / G
138:             fraction = -np.expm1(-dt * G / p.capacitance_nF)
139:             self.v[active] += ((equilibrium - self.v) * fraction)[active]
140:         self.v[own & ~active] = p.reset_mV
141:         spiked = active & (self.v >= p.threshold_mV)
142:         self.steps += 1
143:         self.t_ms = self.steps * dt
144:         self.v[spiked] = p.reset_mV
145:         self.refractory_until_ms[spiked] = self.t_ms + p.refractory_ms
146:         self.spike_count += spiked
147:         self.adaptation[own] *= np.exp(-dt / p.adaptation_tau_ms)
148:         self.adaptation[spiked] += p.adaptation_increment_nA
149:         self.ge *= np.exp(-dt / p.exc_tau_ms)
150:         self.gi *= np.exp(-dt / p.inh_tau_ms)
151:         # Delay >= one step; next slot is consumed after delay full intervals.
152:         # An extra slot is needed to distinguish start from end timestamps.
153:         self._ring[self._ring_i] = spiked
154:         self._ring_i = (self._ring_i + 1) % len(self._ring)
155:         return spiked
```

### `/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py`

SHA256 `c8c5b17e3e70a1c155ae679712182568a1d689b2e62501e9182ca6ca884d17ca`

```text
185:         built = build_neural_tier(); a = built['data']
186:         tier = retain_motor_endpoints(a, downstream_union(a, built['global_index'], ['720575941433155799','720575941566493282'], hops=2, max_neurons=20000), ['720575941391252886','720575941511883985','720575941497037218','720575941559306355'])
187:         roots = a['root_ids'][tier['global_index']]; assert len(roots) == 17099
188:         edges = build_edges(a, tier, built['cut']); groups, _ = channel_index_map(len(roots), roots)
```

```text
225:             d = mj.MjData(m); d.qpos[:] = q  # Sole pose write.
226:             net = ConductanceNetwork(len(roots), dt_ms=.5)
227:             net.set_connectivity(edges['We']*.2, edges['Wi']*.2)
228:             dec = FiniteMotorTranslator(manifest, arm_weights); enc = ChannelSensoryEncoder(groups, len(roots))
229:             assert not dec.history.any() and net.steps == 0 and net.t_ms == 0
230:             atomic_npz(dest/'NEURAL_COLD.npz', **{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)})
231:             atomic_json(dest/'NEURAL_COLD_META.json', neural_meta(net))
232:             connectivity = {}
233:             for name in ('We', 'Wi'):
234:                 matrix = getattr(net, name).tocsr()
235:                 for field in ('data','indices','indptr'): connectivity[name+'_'+field] = getattr(matrix,field)
236:                 connectivity[name+'_shape'] = np.asarray(matrix.shape)
237:             atomic_npz(dest/'CONNECTIVITY.npz', **connectivity)
```

```text
294:             initial_current, _ = enc.encode(initial_observed)
295:             atomic_json(dest/'INITIAL_OBSERVATION.json',raw_observation_record(initial_observed)[0])
296:             warmsp, warmf, warmu = [], [], []
297:             for j in range(PRE):
298:                 reserve(neural_ticks=1); sp = net.step(current_nA=initial_current).copy()
299:                 u, info = dec.consume(sp[None,:],spike_dt_s=.0005,start_s=j*.0005,end_s=(j+1)*.0005)
300:                 warmsp.append(sp); warmf.append(info['pool_features']); warmu.append(u)
301:             release_features = np.stack([dec.history[lo:hi].mean(0) for lo,hi in dec.BINS],axis=1)
302:             pooled = np.clip((release_features*arm_weights).sum(1),0,1)
303:             release_ctrl = np.zeros(m.nu); mapped = dec.actuator_pool >= 0
304:             release_ctrl[mapped] = pooled[dec.actuator_pool[mapped]]
305:             atomic_npz(dest/'INITIALIZATION.npz',spikes=np.asarray(warmsp,dtype=bool),features=np.asarray(warmf),decoded=np.asarray(warmu),warmup_spikes=np.asarray(warmsp,dtype=bool),warmup_features=np.asarray(warmf),warmup_decoded=np.asarray(warmu),
306:                 current=initial_current,history=dec.history,release_features=release_features,release_decoded=release_ctrl,
307:                 weights=arm_weights,body_qpos=d.qpos,body_qvel=d.qvel,body_act=d.act,body_ctrl=d.ctrl,
308:                 **{k:v for k,v in vars(net).items() if isinstance(v,np.ndarray)})
309:             atomic_json(dest/'NEURAL_RELEASE_META.json',neural_meta(net))
```

```text
338:             for tick in range(TICKS):
339:                 if stop_reasons: break
340:                 if tick % 1 == 0:
341:                     assert abs(d.time-tick*.0005) < 1e-12
342:                     observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
343:                     current, encoding_record = enc.encode(observed)
344:                     refresh_times.append(float(d.time))
345:                     atomic_json(dest/f'OBSERVATION_{tick:02d}.json',raw_observation_record(observed)[0])
346:                     atomic_json(dest/f'ENCODING_{tick:02d}.json',dict(
347:                         schema='S162 raw OBSERVATION unchanged; encoder record retained separately',
348:                         tick=tick, physical_sample_time_s=float(d.time),
349:                         body_last_step_index=tick*20, current_nA=current.tolist(),
350:                         features=encoding_record, raw_file=f'OBSERVATION_{tick:02d}.json'))
351:                     sensory_schedule.append(dict(tick=tick, physical_sample_time_s=float(d.time),
352:                         body_last_step_index=tick*20, held_from_tick=None,
353:                         held_current_between_ticks=False, refreshed_every_tick=True,
354:                         source_start_s=(PRE+tick)*.0005,source_end_s=(PRE+tick+1)*.0005,
355:                         order='current integrated body -> observe -> encode -> neural -> decode -> immediate ctrl -> 20 RK4 body substeps'))
356:                 currents.append(current.copy()); reserve(neural_ticks=1)
357:                 sp = net.step(current_nA=current).copy()
358:                 start,end = (PRE+tick)*.0005,(PRE+tick+1)*.0005
359:                 u,info = dec.consume(sp[None,:],spike_dt_s=.0005,start_s=start,end_s=end)
360:                 assert abs(net.t_ms/1000-end) < 1e-12 and dec.interval == PRE+tick
361:                 assert abs(d.time-tick*.0005) < 1e-12
362:                 spikes.append(sp); features.append(info['pool_features']); commands.append(u.copy())
363:                 schedule.append(dict(tick=tick,source_start_s=start,source_end_s=end,
364:                     physical_start_s=float(d.time),planned_physical_end_s=(tick+1)*.0005,
365:                     source_offset_s=.08,consume_id=dec.interval))
366:                 d.ctrl[:] = 0.5 * u
367:                 for sub in range(substeps):
368:                     require_no_callbacks(callback_snapshot(mj))
369:                     assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
370:                     reserve(step=1,forward=1); mj.mj_step(m,d); mj.mj_forward(m,d)
371:                     row = sample(); rows.append(row)
372:                     stop_reasons = s142_reasons(row, q, m)
373:                     if stop_reasons: break
```

```text
376:             arrays.update(spikes=np.asarray(spikes),features=np.asarray(features),
377:                 applied=0.5*np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),
378:                 weights=arm_weights,actuator_pool=dec.actuator_pool.copy())
379:             weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
380:             load = arrays['foot_force'][:,:,2].sum(1)/weight_native
381:             quat = arrays['qpos'][:,3:7]; norms = np.linalg.norm(quat,axis=1)
382:             angle = 2*np.arccos(np.clip(np.abs(quat@quat[0])/(norms*norms[0]),0,1))
383:             gates = np.column_stack((arrays['qpos'][:,2]/s146_original_q[2] >= .85,angle <= .35,
384:                 np.count_nonzero(arrays['contacts'] > 0,axis=1) >= 3,load >= .8,
385:                 arrays['nonfoot_depth'] <= 1e-12,arrays['limit_excess'] <= 1e-6))
386:             arrays.update(six_gates=gates,all_six_gates=gates.all(1),load_ratio=load,root_angle_rad=angle)
387:             # Actual-time trapezoid impulse, including stop sample; NOT work or elastic energy.
388:             for force_key, output_key in [('external_force_world','external_impulse_world'),
389:                                          ('contact_force_world','contact_impulse_world')]:
390:                 f = arrays[force_key]
391:                 arrays[output_key] = np.vstack([np.zeros(3),np.cumsum(
392:                     .5*(f[1:]+f[:-1])*np.diff(arrays['time'])[:,None],axis=0)])
393:             arrays['com_momentum_change_world'] = arrays['com_momentum_world']-arrays['com_momentum_world'][0]
394:             atomic_npz(dest/'TRAJECTORY.npz',**arrays)
395:             atomic_json(dest/'SCHEDULE.json',schedule)
396:             atomic_json(dest/'SENSORY_SCHEDULE.json',sensory_schedule)
397:             atomic_json(dest/'NEURAL_FINAL_META.json',neural_meta(net))
398:             atomic_npz(dest/'NEURAL_FINAL.npz',**{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)},
399:                 FIR_history=dec.history,release_features=release_features,release_decoded=release_ctrl)
```

### `/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/BOUND_INPUT_FREEZE/0087_sensory_channels.py`

SHA256 `85fe4e5cbef021263b242ce26d5db140f7ede319907cb4a5f71b0c0a2fdd9ccf`

```text
255:     def encode(self, per_leg: dict) -> tuple:
256:         """`per_leg[leg]` is a ChannelQuantities.  Returns (current_nA, record)."""
257:         # Config and quantities are mutable dataclasses: validate at the boundary
258:         # as well as construction, including quantities for currently empty groups.
259:         self._validate_config()
260:         cfg = self.cfg
261:         current = np.zeros(self.n)
262:         record = {"per_leg": {}, "unmodelled_channels": [], "scale_nA_per_unit":
263:                   cfg.current_nA_per_unit}
264:         for leg in LEGS:
265:             q = per_leg.get(leg)
266:             if q is None:
267:                 continue
268:             for name in ("tibia_joint_dev_rad", "tibia_joint_vel_rad_s",
269:                          "multi_joint_dev_rad", "contact_force_norm", "foot_contact"):
270:                 self._require_finite_scalar(getattr(q, name), f"{leg}/{name}")
271:             vals = {
272:                 "claw_position": q.tibia_joint_dev_rad / cfg.reference_angle_rad,
273:                 "hook_direction": q.tibia_joint_vel_rad_s / cfg.reference_velocity_rad_s,
274:                 "hairplate_angle": q.multi_joint_dev_rad / cfg.reference_angle_rad,
275:                 "campaniform_strain": q.contact_force_norm / cfg.reference_force,
276:                 "bristle_touch": q.foot_contact,
277:                 "club_vibration": 0.0,
278:                 "multidendritic_noxious": 0.0,
279:             }
280:             leg_record = {}
281:             for ch, v in vals.items():
282:                 self._require_finite_scalar(v, f"{leg}/{ch} normalized quantity")
283:                 _, fn, signed, modelled = CHANNEL_SPEC[ch]
284:                 if not modelled:
285:                     record["unmodelled_channels"].append({"leg": leg, "channel": ch,
286:                                                           "dataset_function": fn,
287:                                                           "value": 0.0})
288:                 idx = self.groups[leg][ch]
289:                 if idx.size:
290:                     with np.errstate(over="ignore", invalid="ignore"):
291:                         current[idx] += cfg.current_nA_per_unit * float(v)
292:                     if not np.all(np.isfinite(current[idx])):
293:                         raise ValueError(f"{leg}/{ch} produces nonfinite current")
294:                 leg_record[ch] = {"value": float(v), "signed": signed, "modelled": modelled,
295:                                   "n_cells": int(idx.size), "dataset_function": fn}
296:             record["per_leg"][leg] = leg_record
297:         return current, record
```

```text
320: def channel_index_map(n_neurons: int, tier_root_ids, *, lookup_path: Path = LOOKUP):
321:     """Partition tier-local indices into (leg, channel) from the EXPORTED lookup.
322: 
323:     `tier_root_ids` is the tier's root-id vector in tier-local order (the tier publishes it).
324:     Returns (channel_groups, stats).  A cell the tier does not contain simply does not appear:
325:     a partial partition is REPORTED, never padded.
326: 
327:     The dataset read itself lives in `tools_export_sensory_channel_lookup.py` because reading
328:     the feather needs pandas, and the interpreter that runs MuJoCo here does not have pandas
329:     (measured: venv_body has mujoco without pandas; venv has pandas without mujoco).
330:     """
331:     lu = load_lookup(lookup_path)
332:     by_root = lu["leg_channel_by_root"]
333:     local_of_root = {int(r): i for i, r in enumerate(np.asarray(tier_root_ids))}
334:     groups = {leg: {ch: [] for ch in CHANNEL_SPEC} for leg in LEGS}
335:     matched = unmatched = 0
336:     for root_s, pair in by_root.items():
337:         leg, ch = pair[0], pair[1]
338:         if leg not in groups or ch not in groups[leg]:
339:             continue
340:         li = local_of_root.get(int(root_s))
341:         if li is None:
342:             unmatched += 1
343:             continue
344:         groups[leg][ch].append(li)
345:         matched += 1
346:     stats = {"cells_placed": matched, "leg_sensory_cells_outside_tier": unmatched,
347:              "n_neurons": int(n_neurons),
348:              "n_dataset_cells_with_a_channel": len(by_root),
349:              "lookup": str(lookup_path),
350:              "note": "a cell that is in the dataset but not in the tier cannot be driven; that "
351:                      "count is reported rather than redistributed"}
352:     return groups, stats
```

### `/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/BOUND_INPUT_FREEZE/0104_motor_translation.py`

SHA256 `5d20bd0051e8c3d066ed64a6827059113923e3aa5c7c00e42ec258dc39121aae`

```text
12: class FiniteMotorTranslator:
13:     DT = .0005
14:     BINS = ((0,5),(5,10),(10,20),(20,40),(40,80))
15:     MAX_WEIGHT = 4.
16: 
17:     def __init__(self, manifest, weights=None):
18:         validated=MotorOutput(manifest)
19:         self.n,self.nu=validated.n,validated.nu
20:         self.pools=sorted({tuple(sorted(g.tolist())) for g in validated.groups if len(g)})
21:         lookup={p:i for i,p in enumerate(self.pools)}
22:         self.actuator_pool=np.array([lookup.get(tuple(sorted(g.tolist())),-1) for g in validated.groups])
23:         identity={'n':self.n,'rows':[{k:r.get(k) for k in ('actuator','actuator_index','local_indices','tier_root_ids','pool_root_ids')} for r in manifest['rows']]}
24:         self.mapping_digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
25:         self._weights=np.tile([.4,.25,.2,.1,.05],(len(self.pools),1)) if weights is None else np.asarray(weights,float).copy()
26:         if self._weights.shape!=(len(self.pools),5) or not np.isfinite(self._weights).all() or np.any((self._weights<0)|(self._weights>self.MAX_WEIGHT)):
27:             raise ValueError('weights require shape pools x5, finite and within [0,4]')
28:         self._weights.flags.writeable=False
29:         self.reset()
```

```text
35:     def reset(self):
36:         self.history=np.zeros((80,len(self.pools)))
37:         self.elapsed_s=0.;self.interval=-1
38: 
39:     def consume(self, spikes, *, spike_dt_s, start_s, end_s, connected=True):
40:         sp=np.asarray(spikes)
41:         if sp.ndim!=2 or sp.shape[1]!=self.n or sp.shape[0]<1 or not np.all((sp==0)|(sp==1)):
42:             raise ValueError('binary finite time x neuron spikes required')
43:         if not all(math.isfinite(v) for v in (spike_dt_s,start_s,end_s)) or not math.isclose(spike_dt_s,self.DT,rel_tol=0,abs_tol=1e-15):
44:             raise ValueError('fixed 0.5ms neural ticks required')
45:         if not math.isclose(start_s,self.elapsed_s,rel_tol=0,abs_tol=1e-12) or not math.isclose(end_s-start_s,len(sp)*self.DT,rel_tol=0,abs_tol=1e-12):
46:             raise ValueError('contiguous completed source intervals required')
47:         counts=np.array([sp[:,p].sum() for p in self.pools],dtype=int)
48:         if connected:
49:             activity=np.column_stack([sp[:,p].mean(axis=1)/self.DT/100 for p in self.pools]) if self.pools else np.zeros((len(sp),0))
50:             self.history=np.concatenate((activity[::-1],self.history),axis=0)[:80]
51:         else:self.history.fill(0)
52:         features=np.stack([self.history[lo:hi].mean(axis=0) for lo,hi in self.BINS],axis=1)
53:         pooled=np.clip((features*self._weights).sum(axis=1),0,1)
54:         ctrl=np.zeros(self.nu);mapped=self.actuator_pool>=0;ctrl[mapped]=pooled[self.actuator_pool[mapped]]
55:         self.elapsed_s=float(end_s);self.interval+=1
56:         return ctrl,{'source_interval':self.interval,'source_start_s':float(start_s),'source_end_s':float(end_s),
57:             'valid_from_s':float(end_s),'connected':bool(connected),'pool_spike_counts':counts.tolist(),
58:             'pool_features':features.tolist(),'memory_horizon_s':.04,'mapping_digest':self.mapping_digest,
59:             'note':'finite causal FIR, no bias; engineering translation not anatomy'}
```
