"""Turn-scoped counts of observable SDK entry calls, separate from iteration budgets."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator


@dataclass
class ClientInvocations:
    count: int = 0
    lock: Lock = field(default_factory=Lock)

    def record(self) -> None:
        with self.lock:
            self.count = min(self.count + 1, 1_000_000)

    def snapshot(self) -> dict[str, str | int]:
        with self.lock:
            return {"kind": "main_chat_client_invocation", "count": self.count}


_current: ContextVar[ClientInvocations | None] = ContextVar("observed_client_invocations", default=None)


def record_client_invocation() -> None:
    counter = _current.get()
    if counter is not None:
        counter.record()


@contextmanager
def observe_client_invocations(agent: Any) -> Iterator[ClientInvocations | None]:
    counter = (ClientInvocations() if getattr(agent, "api_mode", None) == "chat_completions"
               and getattr(agent, "provider", None) != "moa" else None)
    token = _current.set(counter)
    try:
        yield counter
    finally:
        _current.reset(token)
