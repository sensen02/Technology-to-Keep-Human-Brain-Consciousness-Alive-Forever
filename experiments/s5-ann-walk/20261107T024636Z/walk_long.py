"""Is the ANN walk transient or progressive? Re-run the memoryless ANN for 6 s.

Measured on the 3 s protocol the memoryless translator matches the teacher's advance
(+0.283 at t=2.5 s, foot support 1.000, non-foot contact 0) and only then breaks down.
This run decides whether that breakdown is a late transient or a progressive instability.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = '/run/media/sensen/Data2/cell_wound_prototype'
ANN = PROJ + '/experiments/s5-ann-incremental/20261107T010000Z'
sys.path[:0] = [PROJ, PROJ + '/tools', ANN]
import walk_full as wf
import learn_stand as ls
from ann_translator import NumpyFullBrainTranslator

OUT = wf.OUT
RUN = OUT / 'full'
ls.O = RUN
wf.RUN = RUN


def main():
    model = NumpyFullBrainTranslator(RUN / 'learning' / 'TRANSLATOR_WALK_FULL.npz')
    rt = wf.WalkRuntime()
    for tag, seconds, disturb in [('walk_long_6s', 6.0, False), ('walk_long_6s_disturb', 6.0, True)]:
        r = rt.run(tag, 'ann', 55, model, seconds=seconds, disturb=disturb)
        r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
        print(json.dumps(dict(tag=tag, disturb=disturb, walk=r['walk'])), flush=True)
        # 分段前向位移：确认是"走一段"还是"一路推进"
        b = np.load(Path(r['folder']) / 'BODY.npz')
        bt, q = b['time'], b['qpos']
        i0 = int(np.searchsorted(bt, 1.0))
        series = {f'{T:.1f}s': round(float(q[int(np.searchsorted(bt, T)) - 1, 0] - q[i0, 0]), 3)
                  for T in np.arange(1.5, seconds + 0.01, 0.5)}
        n = np.load(Path(r['folder']) / 'NATIVE.npz')['rows']
        foot = n[:, 1:7].sum(1) / wf.WEIGHT
        print(json.dumps(dict(tag=tag, forward_by_time=series,
                              foot_min=round(float(foot[n[:, 0] >= 1.0].min()), 3),
                              nonfoot_max=round(float(n[n[:, 0] >= 1.0, 7].max()), 1)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
