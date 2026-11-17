"""Tests for the continuous batching scheduler."""

import unittest

from continuous_batching import Request, RequestState, Scheduler


def make_request(rid, prompt=8, output=4, deadline=1000, priority=0,
                 kv_pages=None):
    return Request(
        request_id=rid,
        prompt_tokens=prompt,
        max_output_tokens=output,
        deadline=deadline,
        priority=priority,
        kv_pages=kv_pages,
    )


class ContinuousArrivalTest(unittest.TestCase):
    """Requests keep arriving every tick; the scheduler must keep up."""

    def test_throughput_and_completion(self):
        sched = Scheduler(num_kv_pages=64, max_batch_size=8,
                          max_tokens_per_tick=256)
        total = 40
        for tick in range(total):
            sched.add_request(make_request(f"r{tick}", output=4,
                                           deadline=tick + 200))
            sched.step()
        sched.run_until_idle()
        self.assertEqual(sched.stats.completed, total)
        self.assertEqual(sched.stats.timed_out, 0)
        self.assertGreater(sched.stats.throughput, 1.0)
        self.assertEqual(sched.pages_in_use(), 0)

    def test_batch_size_respected(self):
        sched = Scheduler(num_kv_pages=1024, max_batch_size=3)
        for i in range(10):
            sched.add_request(make_request(f"r{i}", output=20))
        for _ in range(50):
            batch = sched.step()
            self.assertLessEqual(len(batch.decoded), 3)
            if not sched.pending():
                break


class LongTailDecodeTest(unittest.TestCase):
    """One very long request must not block short requests."""

    def test_short_requests_not_stuck_behind_long_one(self):
        sched = Scheduler(num_kv_pages=128, max_batch_size=8,
                          max_tokens_per_tick=256)
        sched.add_request(make_request("long", output=300, deadline=10000))
        sched.step()
        for i in range(5):
            sched.add_request(make_request(f"short{i}", output=3,
                                           deadline=10000))
        while any(sched.get(f"short{i}").state is not RequestState.FINISHED
                  for i in range(5)):
            sched.step()
            self.assertLess(sched.now, 100, "short requests starved")
        for i in range(5):
            req = sched.get(f"short{i}")
            self.assertIs(req.state, RequestState.FINISHED)
            self.assertLessEqual(req.finish_tick, 30)
        sched.run_until_idle()
        self.assertIs(sched.get("long").state, RequestState.FINISHED)


class KvPageShortageTest(unittest.TestCase):
    """More demand than KV pages: preemption, no deadlock, no starvation."""

    def test_preemption_and_eventual_completion(self):
        sched = Scheduler(num_kv_pages=20, max_batch_size=8,
                          max_tokens_per_tick=256, aging_interval=5)
        # Two low-priority requests fill the whole page pool.
        for i in range(2):
            sched.add_request(make_request(f"low{i}", output=30, kv_pages=10,
                                           priority=0, deadline=10000))
        for _ in range(3):
            sched.step()
        self.assertEqual(sched.pages_in_use(), 20)
        # A high-priority request arrives and must preempt to get pages.
        sched.add_request(make_request("high", output=10, kv_pages=10,
                                       priority=10, deadline=10000))
        sched.step()
        self.assertGreater(sched.stats.preemptions, 0)
        self.assertIs(sched.get("high").state, RequestState.RUNNING)
        sched.run_until_idle(max_ticks=10000)
        for rid in ("low0", "low1", "high"):
            self.assertIs(sched.get(rid).state, RequestState.FINISHED)
        self.assertEqual(sched.pages_in_use(), 0)

    def test_pages_never_exceeded(self):
        sched = Scheduler(num_kv_pages=24, max_batch_size=8)
        for i in range(6):
            sched.add_request(make_request(f"r{i}", output=15, kv_pages=9,
                                           deadline=10000))
        for _ in range(500):
            if not sched.pending():
                break
            sched.step()
            self.assertLessEqual(sched.pages_in_use(), 24)

    def test_oversized_request_rejected(self):
        sched = Scheduler(num_kv_pages=4)
        with self.assertRaises(ValueError):
            sched.add_request(make_request("big", kv_pages=5))


class CancellationTest(unittest.TestCase):
    """Concurrent cancels of waiting and running requests."""

    def test_cancel_mixed_states(self):
        sched = Scheduler(num_kv_pages=16, max_batch_size=2,
                          max_tokens_per_tick=64)
        ids = [f"r{i}" for i in range(8)]
        for rid in ids:
            sched.add_request(make_request(rid, output=10, kv_pages=4,
                                           deadline=10000))
        sched.step()
        running = [r.request_id for r in sched.pending()
                   if r.state is RequestState.RUNNING]
        waiting = [r.request_id for r in sched.pending()
                   if r.state is RequestState.WAITING]
        self.assertTrue(running)
        self.assertTrue(waiting)
        self.assertTrue(sched.cancel(running[0]))
        self.assertTrue(sched.cancel(waiting[0]))
        self.assertLessEqual(sched.pages_in_use(), 16)
        for rid in ids[4:6]:
            sched.cancel(rid)
            sched.step()
        sched.run_until_idle()
        self.assertEqual(sched.stats.cancelled, 4)
        self.assertEqual(sched.stats.completed, 4)
        self.assertEqual(sched.pages_in_use(), 0)
        self.assertFalse(sched.cancel(ids[0]))
        self.assertFalse(sched.cancel("nope"))


class PriorityInversionTest(unittest.TestCase):
    """Urgent/high-priority requests preempt, but nobody starves."""

    def test_urgent_request_preempts_low_priority(self):
        sched = Scheduler(num_kv_pages=10, max_batch_size=4,
                          max_tokens_per_tick=64, aging_interval=5)
        sched.add_request(make_request("low", output=50, kv_pages=10,
                                       priority=0, deadline=10000))
        sched.step()
        self.assertIs(sched.get("low").state, RequestState.RUNNING)
        sched.add_request(make_request("high", output=5, kv_pages=10,
                                       priority=10,
                                       deadline=sched.now + 6))
        batch = sched.step()
        self.assertIn("low", batch.preempted)
        self.assertIs(sched.get("high").state, RequestState.RUNNING)
        sched.run_until_idle()
        high = sched.get("high")
        self.assertIs(high.state, RequestState.FINISHED)
        self.assertLessEqual(high.finish_tick, high.deadline)
        self.assertIs(sched.get("low").state, RequestState.FINISHED)

    def test_aging_prevents_starvation(self):
        sched = Scheduler(num_kv_pages=10, max_batch_size=1,
                          max_tokens_per_tick=64, aging_interval=3)
        sched.add_request(make_request("low", output=4, kv_pages=10,
                                       priority=0, deadline=100000))
        for i in range(6):
            sched.add_request(make_request(f"hi{i}", output=2, kv_pages=10,
                                           priority=2, deadline=100000))
            sched.step()
        sched.run_until_idle(max_ticks=100000)
        self.assertIs(sched.get("low").state, RequestState.FINISHED)

    def test_timeout_rate(self):
        sched = Scheduler(num_kv_pages=8, max_batch_size=1,
                          max_tokens_per_tick=16)
        for i in range(6):
            sched.add_request(make_request(
                f"r{i}", output=10, kv_pages=4,
                deadline=2 if i % 2 == 0 else 10000))
        sched.run_until_idle()
        self.assertEqual(sched.stats.timed_out, 3)
        self.assertEqual(sched.stats.completed, 3)
        self.assertAlmostEqual(sched.stats.timeout_rate, 0.5)


class DeterminismTest(unittest.TestCase):
    """Identical inputs must produce identical schedules."""

    @staticmethod
    def run_scenario():
        sched = Scheduler(num_kv_pages=24, max_batch_size=4,
                          max_tokens_per_tick=64, aging_interval=4)
        for tick in range(30):
            if tick % 2 == 0:
                sched.add_request(make_request(
                    f"r{tick}", prompt=8 + tick % 5, output=3 + tick % 7,
                    kv_pages=4 + tick % 6, priority=tick % 3,
                    deadline=tick + 40))
            if tick == 10:
                sched.cancel("r4")
            sched.step()
        sched.run_until_idle()
        return sched

    def test_same_input_same_trace(self):
        a = self.run_scenario()
        b = self.run_scenario()
        self.assertEqual(a.trace, b.trace)
        self.assertEqual(a.stats, b.stats)

    def test_trace_is_consistent(self):
        sched = self.run_scenario()
        decoded = sum(len(batch.decoded) for batch in sched.trace)
        self.assertEqual(decoded, sched.stats.generated_tokens)


if __name__ == "__main__":
    unittest.main()
