# Fixed3000 exact continuation: complete actual evaluation

Checkpoint: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/continuation3000/TRAIN/checkpoint_003000.pt`

SHA-256: `17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8`

Training resumed exact parent1000 model/Adam/RNG/full curve, changing only lr0.001→0.0001 for2000 fixed additional updates. No old captured source/data changes, no bound or network change. Training203.682s; all four integrity checkpoints passed22 checks each.

| Condition | Engineering eligible | Saved actual integrity | Forward | Slip max | Yaw rad |
|---|---|---|---:|---:|---:|
|neutral|True|True|0.0713185963743811|0.49278883265360646|-0.006677607974098819|
|left|False|True|0.022630848912034863|5.913961298662751|-0.009376182063800496|
|right|False|True|0.035467127614201406|8.704046926466782|-0.043178652233356846|
|speed085|False|True|-0.06022907354444164|3.3551791604991337|-0.5303814454354159|
|speed090|False|True|0.04312873194290642|10.114005989367353|0.0522124161288866|

All conditions eligible: **False**. Saved-trajectory integrity for all: **True**.

Original all5 gates, all6 sensed cycles, strict nonfoot zero, turn sign/magnitude0.01 and speed reduction versus matched neutral remain unchanged. Integrity proves recorded checkpoint inference/state/force/source/trajectory consistency, not behavioral success. Full native D1–D8 certification was not run. No food behavior optimization, arrival, ingestion or satiety was performed.

Original pipeline219 interrupted before a neutral trace persisted; original directory/log/summary remain intact. No completed condition was repeated. Recovery220 reran the necessary incomplete neutral and executed four previously unstarted cases in fresh directories, retaining frozen evaluator sources. Recovery association files bind exact commands/checkpoint and new orchestration source.

See `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/diagnostics/README.md` for loss-stability and capacity/observability review. Improved teacher fit under lower lr does not establish the cause of a physical failure; onset ramp/step mismatch and any subsequent structural or curriculum change require separately declared experiments.

No automatic training extension or new physical condition is authorized by this report. Parent decides the next bounded task under the active user objective.
