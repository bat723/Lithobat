"""
Deciding *when* to recompute — the part that makes a slider feel alive.

Dragging a slider produces far more parameter changes than the engine can
service. Running every one of them queues work that is already stale before
it starts, which is exactly how a UI ends up lagging seconds behind the
mouse. This module holds the policy for that, kept free of Qt so it can be
tested deterministically instead of by waving a cursor around.

Two rules:

**Coalesce.** Only the newest request matters. If a request arrives while one
is running, it replaces any other request waiting — intermediate states are
dropped unrendered, because nobody wants to watch them.

**Defer what is too slow to be live.** A cheap configuration updates as the
mouse moves; an expensive one waits for the drag to stop. The threshold is a
time budget, not a parameter list, so raising the source grid or switching on
vector imaging changes the behaviour automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Above this, a change waits for the drag to settle rather than running live.
LIVE_BUDGET_MS = 120.0

#: How long "settled" means, once deferred.
SETTLE_MS = 180.0


@dataclass
class Request:
    """One pending unit of work."""

    signature: tuple
    payload: Any
    cost_ms: float

    @property
    def is_live(self) -> bool:
        return self.cost_ms <= LIVE_BUDGET_MS


class Scheduler:
    """Coalescing queue of depth one.

    Not a thread pool and not a timer — it is only the bookkeeping. The caller
    drives it (a Qt timer in the app, a loop in the tests) and does the actual
    work; this decides what is worth doing.
    """

    def __init__(self, live_budget_ms: float = LIVE_BUDGET_MS):
        self.live_budget_ms = live_budget_ms
        self._pending: Request | None = None
        self._running: Request | None = None
        self._last_done: tuple | None = None
        self.dropped = 0

    # -- submitting ---------------------------------------------------
    def submit(self, signature: tuple, payload: Any, cost_ms: float) -> None:
        """Offer a request. Supersedes anything already waiting."""
        if self._pending is not None:
            self.dropped += 1
        self._pending = Request(signature, payload, cost_ms)

    # -- draining -----------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._running is not None

    def take(self, settled: bool = True) -> Request | None:
        """Claim the next request to run, or ``None``.

        Parameters
        ----------
        settled : bool
            Whether the input has stopped changing. An expensive request is
            only released once it has; a cheap one runs either way.
        """
        if self._running is not None or self._pending is None:
            return None

        req = self._pending
        if req.signature == self._last_done:
            self._pending = None       # nothing actually changed
            return None
        if not req.is_live and not settled:
            return None                # too slow to run mid-drag

        self._pending = None
        self._running = req
        return req

    def finish(self, request: Request) -> None:
        """Mark the running request complete."""
        if self._running is not None and self._running.signature == request.signature:
            self._last_done = request.signature
            self._running = None

    def abandon(self) -> None:
        """Give up on the running request without recording it as done."""
        self._running = None


def debounce(fn: Callable, wait_ms: float, clock: Callable[[], float]) -> Callable:
    """Wrap *fn* so it runs at most once per *wait_ms*, trailing edge.

    ``clock`` is injected so tests can advance time rather than sleep.
    """
    state: dict = {"last": None, "pending": None}

    def call(*args, **kwargs):
        now = clock()
        state["pending"] = (args, kwargs)
        if state["last"] is None or (now - state["last"]) * 1000.0 >= wait_ms:
            state["last"] = now
            args_, kwargs_ = state["pending"]
            state["pending"] = None
            return fn(*args_, **kwargs_)
        return None

    def flush():
        if state["pending"] is not None:
            args_, kwargs_ = state["pending"]
            state["pending"] = None
            state["last"] = clock()
            return fn(*args_, **kwargs_)
        return None

    call.flush = flush  # type: ignore[attr-defined]
    return call
