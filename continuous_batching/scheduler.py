"""Continuous (iteration-level) batching scheduler for LLM inference.

Each decode tick the scheduler picks the set of requests that run next,
subject to a KV-page budget, a per-tick token budget and a batch-size
limit.  Requests close to their deadline become "urgent" and may preempt
lower-priority running requests; waiting requests accumulate an aging
bonus so no request starves.

Time is modeled in discrete ticks owned by the scheduler, which makes
the whole simulation deterministic: identical input sequences always
produce identical schedules.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class RequestState(Enum):
    WAITING = "waiting"        # admitted, holds no KV pages
    RUNNING = "running"        # holds KV pages, prefilling or decoding
    FINISHED = "finished"      # generated all requested tokens
    TIMEOUT = "timeout"        # missed its deadline, evicted
    CANCELLED = "cancelled"    # cancelled by the caller


@dataclass
class Request:
    """A single inference request.

    Attributes:
        request_id: Unique identifier.
        prompt_tokens: Number of prompt tokens to prefill.
        max_output_tokens: Number of tokens to generate.
        deadline: Absolute tick by which the request must finish.
        priority: Base priority, higher is more important.
        kv_pages: KV pages reserved while running.  ``None`` means
            derive it from ``prompt_tokens + max_output_tokens`` and the
            scheduler's page size.
        arrival_tick: Tick at which the request was submitted.  Set by
            :meth:`Scheduler.add_request` when omitted.
    """

    request_id: str
    prompt_tokens: int
    max_output_tokens: int
    deadline: int
    priority: int = 0
    kv_pages: Optional[int] = None
    arrival_tick: int = 0

    # Runtime state, managed by the scheduler.
    state: RequestState = field(default=RequestState.WAITING, init=False)
    prefilled_tokens: int = field(default=0, init=False)
    generated_tokens: int = field(default=0, init=False)
    preemptions: int = field(default=0, init=False)
    finish_tick: Optional[int] = field(default=None, init=False)

    @property
    def prefilled(self) -> bool:
        return self.prefilled_tokens >= self.prompt_tokens

    @property
    def remaining_output(self) -> int:
        return self.max_output_tokens - self.generated_tokens


@dataclass
class Batch:
    """The schedule produced for one tick."""

    tick: int
    prefilled: List[str] = field(default_factory=list)   # got prefill chunks
    decoded: List[str] = field(default_factory=list)     # generated a token
    preempted: List[str] = field(default_factory=list)   # swapped out
    finished: List[str] = field(default_factory=list)
    timed_out: List[str] = field(default_factory=list)


@dataclass
class SchedulerStats:
    completed: int = 0
    timed_out: int = 0
    cancelled: int = 0
    preemptions: int = 0
    generated_tokens: int = 0
    ticks: int = 0

    @property
    def throughput(self) -> float:
        """Average generated tokens per tick."""
        return self.generated_tokens / self.ticks if self.ticks else 0.0

    @property
    def timeout_rate(self) -> float:
        """Fraction of resolved requests that missed their deadline."""
        resolved = self.completed + self.timed_out
        return self.timed_out / resolved if resolved else 0.0


class Scheduler:
    """Iteration-level batching scheduler.

    Args:
        num_kv_pages: Total KV-cache pages available on the device.
        page_size: Tokens per KV page, used to derive a request's page
            requirement when ``Request.kv_pages`` is not given.
        max_batch_size: Maximum number of running requests per tick.
        max_tokens_per_tick: Token budget per tick shared by chunked
            prefills (prompt tokens) and decodes (one token each).
        aging_interval: Ticks of waiting time that earn one extra
            effective-priority point.  Guards against starvation.
    """

    def __init__(
        self,
        num_kv_pages: int,
        page_size: int = 16,
        max_batch_size: int = 8,
        max_tokens_per_tick: int = 512,
        aging_interval: int = 10,
    ) -> None:
        if num_kv_pages <= 0:
            raise ValueError("num_kv_pages must be positive")
        self.num_kv_pages = num_kv_pages
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.max_tokens_per_tick = max_tokens_per_tick
        self.aging_interval = max(1, aging_interval)

        self.now = 0
        self.stats = SchedulerStats()
        self._requests: Dict[str, Request] = {}
        self._seq: Dict[str, int] = {}
        self._counter = itertools.count()
        self.trace: List[Batch] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_request(self, request: Request) -> None:
        """Submit a request at the current tick."""
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request id {request.request_id!r}")
        if request.kv_pages is None:
            tokens = request.prompt_tokens + request.max_output_tokens
            request.kv_pages = max(1, math.ceil(tokens / self.page_size))
        if request.kv_pages > self.num_kv_pages:
            raise ValueError(
                f"request {request.request_id!r} needs {request.kv_pages} "
                f"pages, more than the pool ({self.num_kv_pages})"
            )
        request.arrival_tick = self.now
        self._requests[request.request_id] = request
        self._seq[request.request_id] = next(self._counter)

    def cancel(self, request_id: str) -> bool:
        """Cancel a waiting or running request; returns False if unknown
        or already resolved.  Safe to call between or around ticks."""
        req = self._requests.get(request_id)
        if req is None or req.state not in (
            RequestState.WAITING,
            RequestState.RUNNING,
        ):
            return False
        req.state = RequestState.CANCELLED
        self.stats.cancelled += 1
        return True

    def step(self) -> Batch:
        """Advance one tick and return the batch executed this tick."""
        self.now += 1
        self.stats.ticks += 1
        batch = Batch(tick=self.now)

        self._expire(batch)
        self._admit(batch)
        self._execute(batch)

        self.trace.append(batch)
        return batch

    def run_until_idle(self, max_ticks: int = 100000) -> None:
        """Step until no request is waiting or running."""
        for _ in range(max_ticks):
            if not self.pending():
                return
            self.step()
        raise RuntimeError("scheduler did not drain; possible livelock")

    def pending(self) -> List[Request]:
        """Requests still waiting or running, in schedule order."""
        return sorted(self._active(), key=self._sort_key)

    def pages_in_use(self) -> int:
        return sum(
            r.kv_pages
            for r in self._requests.values()
            if r.state is RequestState.RUNNING
        )

    def get(self, request_id: str) -> Optional[Request]:
        return self._requests.get(request_id)

    # ------------------------------------------------------------------
    # Scheduling policy
    # ------------------------------------------------------------------
    def _active(self) -> List[Request]:
        return [
            r
            for r in self._requests.values()
            if r.state in (RequestState.WAITING, RequestState.RUNNING)
        ]

    def _effective_priority(self, req: Request) -> int:
        waited = self.now - req.arrival_tick
        aging = waited // self.aging_interval
        return req.priority + aging + req.preemptions

    def _is_urgent(self, req: Request) -> bool:
        # Cannot meet the deadline even if decoded every tick from now.
        slack = req.deadline - self.now - req.remaining_output
        return slack <= 0

    def _sort_key(self, req: Request) -> Tuple:
        # Total, deterministic order: urgent first, then effective
        # priority, then earliest deadline, then FIFO by arrival.
        return (
            -int(self._is_urgent(req)),
            -self._effective_priority(req),
            req.deadline,
            self._seq[req.request_id],
        )

    def _expire(self, batch: Batch) -> None:
        for req in self._active():
            if self.now > req.deadline:
                req.state = RequestState.TIMEOUT
                req.finish_tick = self.now
                self.stats.timed_out += 1
                batch.timed_out.append(req.request_id)

    def _admit(self, batch: Batch) -> None:
        """Move waiting requests into the running set, preempting
        strictly lower-ranked running requests when pages or batch
        slots are scarce."""
        running = [r for r in self._active() if r.state is RequestState.RUNNING]
        free_pages = self.num_kv_pages - sum(r.kv_pages for r in running)

        for cand in self.pending():
            if cand.state is RequestState.RUNNING:
                continue
            key = self._sort_key(cand)
            need_pages = cand.kv_pages - free_pages
            need_slots = max(0, (len(running) + 1) - self.max_batch_size)

            victims: List[Request] = []
            if need_pages > 0 or need_slots > 0:
                # Only strictly worse-ranked running requests may be
                # preempted, worst first.
                worse = sorted(
                    (r for r in running if self._sort_key(r) > key),
                    key=self._sort_key,
                    reverse=True,
                )
                gained_pages, gained_slots = 0, 0
                for victim in worse:
                    if gained_pages >= need_pages and gained_slots >= need_slots:
                        break
                    victims.append(victim)
                    gained_pages += victim.kv_pages
                    gained_slots += 1
                if gained_pages < need_pages or gained_slots < need_slots:
                    continue  # cannot fit even by preempting; stay waiting

            for victim in victims:
                victim.state = RequestState.WAITING  # swap-out, keep progress
                victim.preemptions += 1
                self.stats.preemptions += 1
                free_pages += victim.kv_pages
                running.remove(victim)
                batch.preempted.append(victim.request_id)

            cand.state = RequestState.RUNNING
            free_pages -= cand.kv_pages
            running.append(cand)

    def _execute(self, batch: Batch) -> None:
        budget = self.max_tokens_per_tick
        for req in self.pending():
            if req.state is not RequestState.RUNNING:
                continue
            if not req.prefilled:
                if budget <= 0:
                    continue
                chunk = min(req.prompt_tokens - req.prefilled_tokens, budget)
                req.prefilled_tokens += chunk
                budget -= chunk
                batch.prefilled.append(req.request_id)
                if not req.prefilled:
                    continue
                # The final prefill chunk also yields the first token.
            elif budget <= 0:
                continue
            else:
                budget -= 1

            req.generated_tokens += 1
            self.stats.generated_tokens += 1
            batch.decoded.append(req.request_id)
            if req.generated_tokens >= req.max_output_tokens:
                req.state = RequestState.FINISHED
                req.finish_tick = self.now
                self.stats.completed += 1
                batch.finished.append(req.request_id)
