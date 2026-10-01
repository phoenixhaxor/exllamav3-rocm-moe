"""Opt-in low-idle policy must preserve dispatch correctness."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Idle CPU measurement and sleeping policy are Linux-specific")
def test_cpu_moe_pool_parks_at_idle_and_wakes_without_lost_dispatches():
    child = r'''
import json, resource, time
from exllamav3.ext import exllamav3_ext as ext
assert ext.exl3_moe_cpu_pool_stress(12, 200, 2, 100) == 0
# Let the short-latency spinning phase expire.
time.sleep(0.3)
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime
start_cpu = cpu(); start = time.monotonic(); time.sleep(2)
pct = (cpu() - start_cpu) / (time.monotonic() - start) * 100
print(json.dumps({'idle_cpu_pct': pct}), flush=True)
assert pct < 10, f'CPU MoE pool idle CPU {pct:.2f}% exceeds 10% of one core'
for _ in range(3):
    assert ext.exl3_moe_cpu_pool_stress(12, 1000, 2, 100) == 0
    time.sleep(0.2)
'''
    env = os.environ | {"EXL3_MOE_IDLE_SLEEP_US": "1000"}
    result = subprocess.run([sys.executable, "-c", child], env=env,
                            cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
