"""Background data refresh: once when the server starts, and on "Refresh now".

The job runs ingest() in a daemon thread so the first render never waits
on ESPN or nba.com. There is one job per server process (st.cache_resource),
shared by every browser tab, and boot_refresh() starts it at most once
per server start however many sessions open.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone

import streamlit as st


class RefreshJob:
    """One refresh at a time; remembers how the last one ended."""

    def __init__(self, fn: Callable[[], object]):
        self._fn = fn
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.runs = 0
        self.error: str | None = None
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def succeeded(self) -> bool:
        return self.finished_at is not None and self.error is None and not self.running

    def start(self) -> bool:
        """Start a refresh unless one is running. True if this call started it."""
        with self._lock:
            if self.running:
                return False
            self.runs += 1
            self.error, self.finished_at = None, None
            self.started_at = datetime.now(timezone.utc)
            self._thread = threading.Thread(target=self._run, name="ninecat-refresh", daemon=True)
            self._thread.start()
            return True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            self._fn()
        except Exception as exc:  # ESPN auth, network: shown in the status line
            self.error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self.finished_at = datetime.now(timezone.utc)


def _ingest() -> None:
    from ninecat import ingest

    ingest.ingest()


@st.cache_resource(show_spinner=False)
def refresh_job() -> RefreshJob:
    """The server's one refresh job."""
    return RefreshJob(_ingest)


@st.cache_resource(show_spinner=False)
def _boot(_job: RefreshJob) -> bool:
    return _job.start()


def boot_refresh(job: RefreshJob, enabled: bool) -> None:
    """Start the job on the server's first session; later sessions and reruns do nothing."""
    if enabled:
        _boot(job)
