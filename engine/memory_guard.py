"""Process memory budget, hard cap and single-flight guard.

INCIDENT THAT PRODUCED THIS MODULE (measured, not hypothetical)
---------------------------------------------------------------
The host has 14.5 GB RAM and 4 GB swap, and `systemd-oomd` is ACTIVE.  The kernel log
records a GLOBAL out-of-memory kill:

    msedge invoked oom-killer: ... global_oom
    Out of memory: Killed process 88800 (python)
        total-vm:19841164kB, anon-rss:7578092kB, oom_score_adj:200
        task_memcg=.../ptyxis-spawn-2a6d0c17-....scope

Two concurrently running audit jobs of THIS project were killed before writing any output
(their artifact directory was never created), and the interactive session became unusable.
The other large consumer is a long-running training job of a DIFFERENT project on the same
host (measured 4.88 GB resident), plus the editor/agent host itself (1.9 GB) and a browser.

WHY A GUARD AND NOT JUST "RUN FEWER JOBS"
-----------------------------------------
A project that can take the whole machine down cannot be trusted with overnight runs, and a
job that dies silently at 90% produces no evidence at all.  So every heavy entry point in this
project declares

    * a MINIMUM FREE MEMORY it needs to start (checked against /proc/meminfo MemAvailable),
    * a HARD ADDRESS-SPACE CAP so a runaway allocation raises MemoryError inside this process
      instead of provoking a global OOM kill, and
    * a SINGLE-FLIGHT LOCK so two of this project's heavy runs cannot stack by accident.

WHAT THIS DOES NOT DO
---------------------
It cannot protect the machine from other programs, and it does not make a job small: it only
makes this project's own footprint declared, bounded and visible.  It also does not turn a
MemoryError into a scientific result - a capped run that dies is a FAILED run and is reported
as such, never as a negative finding.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import resource
import time
from pathlib import Path

MB = 1024 * 1024
MEMINFO = Path("/proc/meminfo")
LOCK_DIR = Path(__file__).resolve().parents[1] / "outputs/.locks"


class MemoryBudgetError(RuntimeError):
    """Raised when a declared memory budget cannot be met. Never silently downgraded."""


def meminfo() -> dict:
    out = {}
    for line in MEMINFO.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        field = rest.strip().split()
        if field and field[0].isdigit():
            out[key] = float(field[0]) / 1024.0     # kB -> MB
    return out


def available_mb() -> float:
    return float(meminfo().get("MemAvailable", 0.0))


def swap_used_mb() -> float:
    m = meminfo()
    return float(m.get("SwapTotal", 0.0) - m.get("SwapFree", 0.0))


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def current_vm_mb() -> tuple:
    """(VmPeak, VmRSS, VmSize) in MB, read from this process only."""
    vals = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        for key in ("VmPeak", "VmRSS", "VmSize", "VmSwap"):
            if line.startswith(key + ":"):
                vals[key] = float(line.split()[1]) / 1024.0
    return vals.get("VmPeak", 0.0), vals.get("VmRSS", 0.0), vals.get("VmSize", 0.0)


def cap_address_space_mb(cap_mb: float) -> float:
    """Cap this process's virtual address space; return the previous soft limit in MB.

    The cap is deliberately generous relative to measured use, because EGL/OpenGL and BLAS
    reserve large mappings that are never resident; the purpose is to convert an unbounded
    runaway into a clean MemoryError, not to constrain normal operation.
    """
    want = int(cap_mb * MB)
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = want if old_hard in (resource.RLIM_INFINITY, -1) else min(want, old_hard)
    if new_hard < want:
        raise MemoryBudgetError(
            f"cannot cap address space to {cap_mb} MB: existing hard limit is "
            f"{old_hard / MB:.0f} MB")
    resource.setrlimit(resource.RLIMIT_AS, (new_hard, new_hard))
    return float("inf") if old_soft in (resource.RLIM_INFINITY, -1) else old_soft / MB


def require_available_mb(need_mb: float, label: str) -> dict:
    """Refuse to start when the host cannot afford this job. Raises, never clamps."""
    m = meminfo()
    avail = float(m.get("MemAvailable", 0.0))
    if avail < need_mb:
        raise MemoryBudgetError(
            f"{label}: refusing to start - needs {need_mb:.0f} MB available but the host "
            f"reports {avail:.0f} MB (swap used {swap_used_mb():.0f} MB). Running anyway is "
            f"what caused the recorded global OOM kill; free memory or wait for the "
            f"concurrent job to finish, then re-run.")
    return {"available_mb": avail, "swap_used_mb": swap_used_mb(),
            "MemTotal_mb": float(m.get("MemTotal", 0.0))}


@contextlib.contextmanager
def single_flight(name: str, *, wait_s: float = 0.0):
    """Exclusive lock on ``name``; a second holder either waits or fails with the holder named.

    MEASURED BUG (fixed, 2026-10-11): the lock file was opened with mode ``"w"``, which
    TRUNCATES it before the lock is even attempted - so a blocked run destroyed the holder's
    pid line and the error message could not say who held the lock.  It is now opened
    ``O_RDWR|O_CREAT`` and only rewritten AFTER the lock is acquired.

    The holder's identity is read from the file and reported, because "someone else is running"
    is not an actionable message.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCK_DIR / (name.replace("/", "_") + ".lock")
    fh = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o644), "r+")
    deadline = time.time() + max(0.0, wait_s)
    try:
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.time() >= deadline:
                    fh.seek(0)
                    held = fh.read().strip() or "unknown holder (no pid line)"
                    raise MemoryBudgetError(
                        f"another heavy run holds {path}: {held}. This project serialises heavy "
                        f"jobs because concurrent runs are what produced the recorded OOM kill")
                time.sleep(0.5)
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} started={time.time():.0f} label={name}\n")
        fh.flush()
        yield path
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


#: ONE shared lock name for every heavy entry point of this project.
#:
#: MEASURED BUG (fixed, 2026-10-11): `guarded_run` used to pass its own ``label`` as the lock
#: name, so `neural_muscle_closure` and `measure_memory_footprint` locked DIFFERENT files and
#: ran concurrently - verified end to end: with the kernel lock provably held by one entry
#: point (`flock -n` probe returned "held"), a second entry point still started and completed.
#: A per-job lock cannot prevent the very failure it exists for, because the recorded OOM was
#: caused by different jobs overlapping.  All heavy runs therefore share this single lock.
HEAVY_JOB_LOCK = "heavy_job"


@contextlib.contextmanager
def guarded_run(label: str, *, need_mb: float, cap_mb: float, lock: bool = True,
                wait_s: float = 0.0, lock_name: str | None = None):
    """require_available_mb + single_flight + cap_address_space_mb, with a report.

    ``report()`` returns the measured before/after footprint so the result artifact can state
    what the run actually cost instead of asserting that it was small.
    """
    info = {"label": label, "requested_need_mb": need_mb, "requested_cap_mb": cap_mb,
            "lock_name": lock_name or HEAVY_JOB_LOCK}
    with contextlib.ExitStack() as stack:
        if lock:
            stack.enter_context(single_flight(lock_name or HEAVY_JOB_LOCK, wait_s=wait_s))
        info["start"] = require_available_mb(need_mb, label)
        previous_limit = cap_address_space_mb(cap_mb)
        # JSON has no Infinity; null explicitly denotes an unlimited previous cap.
        info["previous_soft_limit_mb"] = previous_limit if previous_limit != float('inf') else None
        info["previous_soft_limit_unlimited"] = previous_limit == float('inf')
        info["start"]["VmPeak_mb"], info["start"]["VmRSS_mb"], info["start"]["VmSize_mb"] = \
            current_vm_mb()
        t0 = time.time()

        def report(result_status: str = "completed"):
            peak, rss, size = current_vm_mb()
            info.update(peak_VmPeak_mb=peak, end_VmRSS_mb=rss, end_VmSize_mb=size,
                        ru_maxrss_mb=peak_rss_mb(), wall_s=time.time() - t0,
                        end_available_mb=available_mb(), end_swap_used_mb=swap_used_mb(),
                        result_status=result_status,
                        cap_utilisation=peak / cap_mb if cap_mb else None)
            return info

        yield report


#: Declared budgets for this project's heavy entry points.  Derived from a MEASURED stage table
#: (tool_measure_memory.py, outputs/memory_guard/memory_budget.json):
#:
#:     peak VmPeak 3159.2 MB, peak VmRSS 1047.6 MB  (render + 12000-neuron tier + closure prep)
#:     largest single steps: compile_canonical_model +1167 MB VmPeak,
#:                           build_neural_tier_12000 +894 MB VmPeak / +394 MB RSS,
#:                           closure_prepare_body  +492 MB VmPeak
#:
#: ``need_mb`` is a REFUSAL threshold against host MemAvailable, set near 2x the measured
#: resident footprint (~1.05 GB) so the run does not start into a swapping host.
#: ``cap_mb`` is an address-space cap with ~1.4x headroom over the measured worst VmPeak, which
#: turns an unbounded runaway into a clean MemoryError instead of a global OOM kill.
BUDGETS = {
    "neural_muscle_closure": {"need_mb": 2200.0, "cap_mb": 4608.0},
    "mechanics_audit": {"need_mb": 2200.0, "cap_mb": 4608.0},
    "latency_audit": {"need_mb": 2200.0, "cap_mb": 4608.0},
    "render": {"need_mb": 1600.0, "cap_mb": 4608.0},
    "units_audit": {"need_mb": 1200.0, "cap_mb": 4608.0},
    "muscle_floor_audit": {"need_mb": 1800.0, "cap_mb": 4608.0},
    "measure_memory_footprint": {"need_mb": 900.0, "cap_mb": 4096.0},
}
# NOTE: every cap is 4608 MB, i.e. above the measured 3159 MB project peak, because only the
# full-stack job has been MEASURED.  A per-job cap below the project peak may only be set after
# that specific job is measured; test_memory_guard.py enforces this, and it caught
# `units_audit` sitting at 3072 MB (below the measured peak) before any run used it.


