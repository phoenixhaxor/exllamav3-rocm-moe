"""
AsyncGenerator with the sync Generator stubbed out: threaded step (EXL3_ASYNC_THREAD) and the refusal of a job at
enqueue (upstream: "a rejected job must not stay tracked"; threaded: the worker thread records the refusal and the
loop thread untracks the job and delivers the exception to its consumer)
"""
import asyncio
import pytest

pytestmark = pytest.mark.nogpu

from exllamav3.generator import async_generator as ag


class StubJob:
    def __init__(self, reject = False, **kwargs):
        self.reject = reject


class StubGenerator:
    def __init__(self, *args, **kwargs):
        self.pending = []
        self.closed = False

    def enqueue(self, job):
        if job.reject:
            raise ValueError("prompt too long for the cache")
        self.pending.append(job)

    def cancel(self, job):
        if job in self.pending:
            self.pending.remove(job)

    def num_remaining_jobs(self):
        return len(self.pending)

    def has_idle_work(self):
        return False

    def idle_work(self):
        pass

    def iterate(self):
        out = [{"job": j, "eos": True, "stage": "streaming", "text": "x"} for j in self.pending]
        self.pending.clear()
        return out

    def close(self):
        self.closed = True


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(ag, "Generator", StubGenerator)
    monkeypatch.setattr(ag, "Job", StubJob)


async def _consume(async_job):
    return [r async for r in async_job]


@pytest.mark.parametrize("threaded", [False, True])
def test_accepted_job_delivers_result(stubs, monkeypatch, threaded):
    monkeypatch.setattr(ag, "_async_thread", threaded)

    async def run():
        gen = ag.AsyncGenerator()
        try:
            job = ag.AsyncJob(gen)
            results = await asyncio.wait_for(_consume(job), 10)
            assert len(results) == 1 and results[0]["eos"]
            assert not gen.jobs
        finally:
            await gen.close()
        assert gen.generator.closed

    asyncio.run(run())


def test_rejected_job_not_threaded_raises_at_enqueue(stubs, monkeypatch):
    monkeypatch.setattr(ag, "_async_thread", False)

    async def run():
        gen = ag.AsyncGenerator()
        try:
            with pytest.raises(ValueError):
                ag.AsyncJob(gen, reject = True)
            assert not gen.jobs
        finally:
            await gen.close()

    asyncio.run(run())


def test_rejected_job_threaded_reaches_consumer(stubs, monkeypatch):
    monkeypatch.setattr(ag, "_async_thread", True)

    async def run():
        gen = ag.AsyncGenerator()
        try:
            bad = ag.AsyncJob(gen, reject = True)
            with pytest.raises(ValueError):
                await asyncio.wait_for(_consume(bad), 10)
            assert not gen.jobs, "a refused job must be untracked"
            # the generator keeps working for the next job
            good = ag.AsyncJob(gen)
            results = await asyncio.wait_for(_consume(good), 10)
            assert len(results) == 1
        finally:
            await gen.close()

    asyncio.run(run())
