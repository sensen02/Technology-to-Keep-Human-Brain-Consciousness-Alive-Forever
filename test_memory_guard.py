"""Self-test for the memory guard: a guard that has never refused anything is not a guard.

Uses no project data and allocates nothing large, so it is safe to run at any time.
"""
import multiprocessing as mp
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.memory_guard as mg  # noqa: E402


def _hold_lock(name, seconds, queue):
    with mg.single_flight(name):
        queue.put("held")
        time.sleep(seconds)


class GuardTests(unittest.TestCase):
    def test_meminfo_is_readable(self):
        m = mg.meminfo()
        self.assertGreater(m["MemTotal"], 0)
        self.assertGreater(mg.available_mb(), 0)
        self.assertGreaterEqual(mg.swap_used_mb(), 0)

    def test_refuses_when_host_cannot_afford_job(self):
        with self.assertRaises(mg.MemoryBudgetError) as ctx:
            mg.require_available_mb(10 ** 9, "absurd job")
        self.assertIn("refusing to start", str(ctx.exception))
        self.assertIn("absurd job", str(ctx.exception))

    def test_accepts_an_affordable_job(self):
        info = mg.require_available_mb(1.0, "tiny job")
        self.assertIn("available_mb", info)

    def test_cap_is_applied_and_inherited_by_children(self):
        q = mp.Queue()
        p = mp.Process(target=_cap_child, args=(q,))
        p.start()
        p.join(30)
        self.assertEqual(q.get(timeout=5), "ok")

    def test_second_holder_is_refused(self):
        q = mp.Queue()
        name = "selftest_lock"
        with mg.single_flight(name):
            p = mp.Process(target=_hold_lock, args=(name, 2.0, q))
            p.start()
            with self.assertRaises(mg.MemoryBudgetError):
                with mg.single_flight(name, wait_s=0.0):
                    self.fail("two holders of the same heavy-job lock")
            p.terminate()
            p.join(10)

    def test_lock_released_after_use(self):
        name = "selftest_release"
        with mg.single_flight(name):
            pass
        from engine.memory_guard import LOCK_DIR
        self.assertTrue((LOCK_DIR / (name + ".lock")).exists())
        with mg.single_flight(name):   # must not raise
            pass

    def test_guarded_run_reports_footprint_without_lying(self):
        with mg.guarded_run("selftest_run", need_mb=1.0, cap_mb=4096.0) as report:
            blob = bytearray(64 * 1024 * 1024)
            info = report("completed")
        self.assertGreaterEqual(info["peak_VmPeak_mb"], 64.0)
        self.assertEqual(info["result_status"], "completed")
        self.assertIsNotNone(info["cap_utilisation"])
        del blob

    def test_measured_budget_matches_recorded_measurement(self):
        doc = HERE / "outputs/memory_guard/memory_budget.json"
        if not doc.exists():
            self.skipTest("run tool_measure_memory.py first")
        import json
        m = json.loads(doc.read_text())["measured"]
        for name, b in mg.BUDGETS.items():
            if name == "measure_memory_footprint":
                continue
            self.assertGreater(b["cap_mb"], m["peak_VmPeak_mb"],
                               f"{name} cap is below the measured peak - the cap would kill "
                               f"a healthy run")


    def _spawn_holder(self, seconds=2.0, lock_name=None):
        """Start a real holder process and guarantee it is reaped (no lock left behind).

        NOTE: an earlier version of this file forgot to terminate the holder when an assertion
        failed, so a leaked child kept the shared lock and made LATER tests fail with an
        unrelated 'lock held' error. Cleanup is registered here so that cannot recur.
        """
        import multiprocessing as mp
        q = mp.Queue()
        p = mp.Process(target=_hold_shared, args=(seconds, q, lock_name))
        p.start()
        self.addCleanup(self._reap, p)
        self.assertEqual(q.get(timeout=15), "held")
        return p

    @staticmethod
    def _reap(p):
        if p.is_alive():
            p.terminate()
        p.join(15)

    def test_different_labels_still_exclude_each_other(self):
        """REGRESSION for the measured bug: per-label locks let different jobs overlap.

        Before the fix, guarded_run used its label as the lock name, so two DIFFERENT heavy
        entry points locked different files and ran at the same time - which is exactly the
        configuration that caused the recorded OOM. This test fails if that ever returns.
        """
        self._spawn_holder()
        with self.assertRaises(mg.MemoryBudgetError) as ctx:
            with mg.guarded_run("a_completely_different_job", need_mb=1.0, cap_mb=4096.0):
                self.fail("a second heavy job started while another held the shared lock")
        self.assertIn("another heavy run holds", str(ctx.exception))

    def test_lock_error_names_the_holder(self):
        """`single_flight` is per-name by design; the same name must conflict and be named."""
        self._spawn_holder()
        with self.assertRaises(mg.MemoryBudgetError) as ctx:
            with mg.single_flight(mg.HEAVY_JOB_LOCK, wait_s=0.0):
                self.fail("two holders of the same lock name")
        self.assertIn("another heavy run holds", str(ctx.exception))
        self.assertIn("pid=", str(ctx.exception), "error must name the actual holder")
        with mg.single_flight("an_unrelated_lock_name"):   # different name: no conflict
            pass

    def test_measuring_a_different_label_uses_the_shared_lock(self):
        """Distinct labels must map to the SAME lock file, so serialisation is global."""
        self._spawn_holder()
        with self.assertRaises(mg.MemoryBudgetError):
            with mg.guarded_run("measure_memory_footprint", need_mb=1.0, cap_mb=4096.0):
                self.fail("per-label lock regression: shared lock not used")

    def test_lock_file_is_not_truncated_while_held(self):
        """A blocked run must not destroy the holder's pid line (it opened mode 'w' before)."""
        self._spawn_holder(seconds=4.0)
        with self.assertRaises(mg.MemoryBudgetError):
            with mg.single_flight(mg.HEAVY_JOB_LOCK, wait_s=0.0):
                pass
        from engine.memory_guard import LOCK_DIR
        content = (LOCK_DIR / (mg.HEAVY_JOB_LOCK + ".lock")).read_text()
        self.assertIn("pid=", content, "holder metadata was truncated by the blocked run")

    def test_sequential_heavy_jobs_are_allowed(self):
        """Serialisation must not deadlock: after the holder exits, the lock is free again."""
        p = self._spawn_holder()
        self._reap(p)
        with mg.guarded_run("next_heavy_job", need_mb=1.0, cap_mb=4096.0) as report:
            info = report("completed")
        self.assertEqual(info["result_status"], "completed")
        self.assertEqual(info["lock_name"], mg.HEAVY_JOB_LOCK)


def _hold_shared(seconds, queue, lock_name=None):
    with mg.guarded_run("holder_job_under_test", need_mb=1.0, cap_mb=4096.0,
                        lock_name=lock_name):
        queue.put("held")
        time.sleep(seconds)


def _cap_child(q):
    try:
        mg.cap_address_space_mb(2048.0)
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        q.put("ok" if soft == 2048 * 1024 * 1024 else f"wrong {soft}")
    except Exception as exc:      # pragma: no cover
        q.put(f"error {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
