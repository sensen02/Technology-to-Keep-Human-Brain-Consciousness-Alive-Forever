import hashlib
import json
import pathlib

B = pathlib.Path('/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259')
p = B / 'README.zh-CN.md'
t = p.read_text()
old = '共命中 6 个起点（<= 1.0 mm/s²），其中过门限 5 个；它们是**不同状态**（root z 与接触数不同），所以"残差为零"不等于"唯一"。'
new = ('共命中 6 个起点（<= 1.0 mm/s²）：满足 qacc 项（<= 1e-3）的 5 个，**四项门限全过的 4 个**'
       '（起点 16 / 117 / 131 / 212；起点 207 的 51 个跗节接触只覆盖 5 条腿、缺 LF，起点 5 的 qacc 项未过）。'
       '**"过门限"一律指四项全过。** 它们是**不同状态**（root z 1.950011–2.028718、接触数 47–51 不同），'
       '所以"残差为零"不等于"唯一"。')
assert t.count(old) == 1, t.count(old)
p.write_text(t.replace(old, new, 1))

c = B / 'CHANGELOG-inplace.md'
c.write_text(c.read_text() + (
    '\n## F2（补记，同轮） — `%s`\n- 位置：`§3 逐平衡明细 末尾那句`\n- 改前：%s\n- 改后：%s\n'
    '- 依据：本轮要求"不得再出现 5 个过门限的字样"\n' % (p, old, new)))

files = sorted(x for x in B.iterdir() if x.is_file() and x.name != 'delivery-hashes.json')
(B / 'delivery-hashes.json').write_text(json.dumps({x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
                                                   ensure_ascii=False, indent=1))
print('fixed; remaining "过门限 5" hits:', p.read_text().count('过门限 5'))
print('hashes:', len(files))
