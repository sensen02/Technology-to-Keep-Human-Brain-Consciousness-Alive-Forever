"""Turn 25, part A: patch the clock in the family scripts, apply the single F-A sentence (76.375 ms) to the four files the
owner named, and append the two immutable event records.

Part B (a separate script) produces the ONE current 25 us batch.
"""

import hashlib
import json
import pathlib
import re
import time

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
V3 = DESK / 'S2-equilibrium-A-corrected-reissued-v3-20261015T170319'
VOID_A = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
VOID_B = DESK / 'S2-accepted-equilibrium-localization-20261015T170058'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
edits = []

SENTENCE_CN = ('首个 `Thorax_geom | floor` 接触 = **步 3055 = 76.375 ms**（3055 × 25 µs；**76.400 ms 属于步 3056**，此前写错一步）；'
               '首次非足接触 = `(floor, A6_geom)`（`A6_geom` 属 body A6），步 2739 = 68.475 ms；'
               '这是**首次非足（躯干/肢体、非跗节）接触，不是"失去足接触"**（`first_tarsus_loss_step = 5111` = 127.775 ms；'
               '`legs_in_contact` 自步 537 = 13.425 ms 起少于 6）')
SENTENCE_EN = ('first Thorax_geom | floor contact = step 3055 = 76.375 ms (3055 x 25 us; 76.400 ms belongs to step 3056, one step off as '
               'previously written); first non-foot contact = (floor, A6_geom), body A6, step 2739 = 68.475 ms; this is a first NON-FOOT '
               '(trunk/limb, non-tarsus) contact, NOT a loss of foot contact (first_tarsus_loss_step = 5111 = 127.775 ms; legs_in_contact '
               'below 6 from step 537 = 13.425 ms)')


def record(file, before, after, reason):
    edits.append(dict(file=str(file), sha256_before=before, sha256_after=after, changed_at=now(),
                      changed_by='the model on the owner\'s instruction', reason=reason))


def main():
    partA = DESK / ('S2-25us-clock-patches-and-events-' + time.strftime('%Y%m%dT%H%M%S'))
    partA.mkdir(exist_ok=False)

    # ---------------- 1) clock patches ----------------
    loc = ROOT / 'run_accepted_equilibrium_localization.py'
    t = loc.read_text()
    b = sha(loc)
    if "time_s=np.arange(root_z.size) * DT" in t:
        t = t.replace("time_s=np.arange(root_z.size) * DT",
                      "time_s=time_s_measured  # taken from d.time during the run, never arange*DT")
        t = t.replace("""    time_s = np.zeros(STEPS + 1)""", """    time_s = np.zeros(STEPS + 1)""") if 'time_s = np.zeros(STEPS + 1)' in t else t
        if 'time_s = np.zeros(STEPS + 1)' not in t:
            t = t.replace("""    root_z = np.zeros(STEPS + 1)""", """    root_z = np.zeros(STEPS + 1)
    time_s_measured = np.zeros(STEPS + 1)""", 1)
            t = t.replace("""        root_z[i] = float(d.qpos[2])""", """        time_s_measured[i] = float(d.time)
        root_z[i] = float(d.qpos[2])""", 1)
        t = t.replace("""    assert float(m.opt.timestep) == DT, 'runtime timestep mismatch'""",
                      """    assert float(m.opt.timestep) == DT, 'runtime timestep mismatch'
    assert abs(float(d.time) - STEPS * DT) <= 1e-9, ('the measured final d.time must equal steps x dt; note the convention: STEPS mj_step calls '
                                                     'give d.time = STEPS * DT = 0.5 s and STEPS + 1 samples')""", 1)
        loc.write_text(t)
        record(loc, b, sha(loc), 'time_s now comes from the measured d.time and the final time is asserted against steps x dt')
    for name in ('measure_openloop_step_qacc.py', 'measure_openloop_step_u_diamond.py'):
        p = ROOT / name
        t = p.read_text()
        b = sha(p)
        t2 = t.replace("time_s=np.arange(qacc.size) * dt", "time_s=time_s_measured")
        t2 = t2.replace("time_s=np.arange(rz.size) * dt", "time_s=time_s_measured")
        if t2 != t:
            t2 = t2.replace("""    for step in range(steps):""", """    time_s_measured = np.zeros(steps + 1)
    for step in range(steps):""", 1)
            t2 = t2.replace("""        mujoco.mj_step(model, d)""", """        mujoco.mj_step(model, d)
        time_s_measured[step + 1] = float(d.time)""", 1)
            p.write_text(t2)
            record(p, b, sha(p), 'the npz time axis is now the measured d.time, not arange*dt')

    # ---------------- 2) the single F-A sentence in the four named files ----------------
    # v1 README
    rmp = V1 / 'README.zh-CN.md'
    t = rmp.read_text()
    b = sha(rmp)
    old = [ln for ln in t.split('\n') if '76.400' in ln]
    if old:
        t = t.replace(old[0], '- ' + SENTENCE_CN)
        rmp.write_text(t)
        record(rmp, b, sha(rmp), 'F-A sentence: 76.375 ms (3055 x 25 us), 76.400 belongs to step 3056')
    # v2 report.json + README
    rp = V2 / 'report.json'
    r = json.loads(rp.read_text())
    b = sha(rp)
    md = r.get('mandatory_disclosure', [])
    for i, s in enumerate(md):
        if 'A6_geom' in s:
            md[i] = ('A declared-cone static equilibrium is NOT dynamic maintainability. ' + SENTENCE_EN)
    r['mandatory_disclosure'] = md
    rp.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    record(rp, b, sha(rp), 'F-A sentence with 76.375 ms')
    rmp2 = V2 / 'README.zh-CN.md'
    t = rmp2.read_text()
    b = sha(rmp2)
    hit = [ln for ln in t.split('\n') if 'A6_geom' in ln]
    if hit:
        rmp2.write_text(t.replace(hit[0], '- ' + SENTENCE_CN))
        record(rmp2, b, sha(rmp2), 'F-A sentence with 76.375 ms')
    # ladder README
    rml = LADDER / 'README.zh-CN.md'
    t = rml.read_text()
    b = sha(rml)
    hit = [ln for ln in t.split('\n') if 'A6_geom' in ln]
    if hit:
        rml.write_text(t.replace(hit[0], '- ' + SENTENCE_CN))
        record(rml, b, sha(rml), 'F-A sentence with 76.375 ms')
    # v3 canonical record + README
    cp3 = V3 / 'contact_identity_sentence_canonical.json'
    if cp3.exists():
        d3 = json.loads(cp3.read_text())
        b = sha(cp3)
        d3['sentence'] = SENTENCE_CN
        d3['applied_here'] = str(cp3.parent / 'README.zh-CN.md')
        d3['prior_wrong_value'] = 'step 3055 was written as 76.400 ms; 76.400 ms belongs to step 3056'
        d3['recorded_at'] = now()
        cp3.write_text(json.dumps(d3, ensure_ascii=False, indent=1))
        record(cp3, b, sha(cp3), 'canonical F-A sentence set to 76.375 ms')

    # ---------------- 3) two appended event records ----------------
    events = dict(
        status='APPENDED_EVENT_LOG', manual_research_not_product=True,
        events=[
            dict(id='EV-01', when='2026-10-15 16:14:08', kind='assembler hash mismatch',
                 file=str(ROOT / 'run_equilibrium_reissue_finalize.py'),
                 sha256_before='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0',
                 sha256_after='9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742',
                 reason='the live script was edited after the batch was delivered; it added the F2 count fields',
                 disposition='recorded, not rewritten: the delivered records keep the old value; no hash was changed to match the present state',
                 note='the 9271e578 bytes are gone, so this event is documented from the verification that measured it'),
            dict(id='EV-02', when='2026-10-15 17:01:26', kind='frozen evidence file rewritten inside a verification window',
                 file=str(V1 / 'integrity_incident_assembler.json'),
                 sha256_before='NOT RECORDED — untraceable (no hash was taken while the file still said INVALID)',
                 sha256_after='d1aa14121aadbbeb35b07e5465f518ff3ad9355222f9301da36335f06c23b034',
                 reason='the status was changed from INVALID to WITHDRAWN and a README section was added, on the turn-18 instruction',
                 disposition='recorded, not rewritten; the file is not touched again',
                 consequence='the incident record inside a frozen delivered artifact is now STALE: its status is a later revision, and the '
                             'verification that had frozen it at 16:58:44 lost that anchor mid-run'),
        ],
        rule='frozen artifacts are not rewritten; every change to one is appended here with before/after hashes, and an unrecorded before-state '
             'is stated as untraceable rather than reconstructed',
        recorded_at=now())
    (partA / 'appended_events.json').write_text(json.dumps(events, ensure_ascii=False, indent=1))
    (partA / 'clock_patch_ledger.json').write_text(json.dumps(dict(status='PASS_CLOCK_PATCHES', edits=edits, recorded_at=now()),
                                                               ensure_ascii=False, indent=1))
    print('edited files:', len(edits))
    for e in edits:
        print('   %-70s %s -> %s' % (pathlib.Path(e['file']).name, e['sha256_before'][:8], e['sha256_after'][:8]))
    print('part A dir:', partA)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
