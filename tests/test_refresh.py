import threading

import streamlit as st

from ui import refresh
from ui.refresh import RefreshJob, boot_refresh


def test_job_runs_in_background_and_records_success():
    gate, calls = threading.Event(), []
    job = RefreshJob(lambda: (gate.wait(5), calls.append(1)))
    assert job.start() and job.running
    assert not job.start(), "a second refresh must not start while one runs"
    gate.set()
    job.join(5)
    assert calls == [1] and job.succeeded and job.runs == 1 and job.finished_at is not None


def test_job_records_failure_without_raising():
    def boom():
        raise RuntimeError("ESPN said no")

    job = RefreshJob(boom)
    job.start()
    job.join(5)
    assert not job.succeeded and job.error == "RuntimeError: ESPN said no"
    assert job.start(), "a failed refresh can be retried"
    job.join(5)
    assert job.runs == 2


def test_boot_refresh_runs_once_per_server_start():
    st.cache_resource.clear()
    calls = []
    job = RefreshJob(lambda: calls.append(1))
    for _ in range(3):            # three sessions, or three reruns
        boot_refresh(job, enabled=True)
        job.join(5)
    other = RefreshJob(lambda: calls.append(2))
    boot_refresh(other, enabled=True)   # same server: still guarded
    other.join(5)
    assert calls == [1] and job.runs == 1 and other.runs == 0
    st.cache_resource.clear()


def test_boot_refresh_disabled_never_starts():
    st.cache_resource.clear()
    job = RefreshJob(lambda: None)
    boot_refresh(job, enabled=False)
    assert job.runs == 0
    st.cache_resource.clear()


def test_refresh_job_is_shared():
    st.cache_resource.clear()
    assert refresh.refresh_job() is refresh.refresh_job()
    st.cache_resource.clear()
