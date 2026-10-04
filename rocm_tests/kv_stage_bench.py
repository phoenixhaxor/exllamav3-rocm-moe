# Staging bandwidth for KV streaming: one layer image (2048 pages = 512K tokens at Q8) copied from the pinned host
# copy into VRAM by DMA runs (kv_stage_runs) and by the page-gather kernel (kv_stage_pages); checks the copies
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext

shapes = [((2048, 256, 128), torch.int), ((2048, 256, 128), torch.int), ((2048, 256, 16), torch.half), ((2048, 256, 16), torch.half)]
host = [ext.kv_host_alloc(list(s), d, 0) for s, d in shapes]
for h in host:
    h.copy_(torch.randint(-1000, 1000, h.shape, device = "cuda").to(h.dtype))
dev = [torch.empty(s, dtype = d, device = "cuda") for s, d in shapes]
n = 1600
nbytes = sum(h[:n].numel() * h.element_size() for h in host)
for name, fn in (
    ("dma runs", lambda: ext.kv_stage_runs(torch.tensor([[0, n]], dtype = torch.int), host, dev)),
    ("dma 1-page runs", lambda: ext.kv_stage_runs(torch.stack([torch.arange(n), torch.arange(n) + 1], 1).int().contiguous(), host, dev)),
    ("gather kernel", lambda: ext.kv_stage_pages(torch.arange(n, dtype = torch.int, device = "cuda"), host, dev)),
):
    for d in dev: d.zero_()
    fn(); torch.cuda.synchronize()
    ok = all(torch.equal(h[:n], d[:n]) for h, d in zip(host, dev))
    t = time.perf_counter()
    for _ in range(5): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 5
    print(f"{name:16s}: {nbytes / 1e6:.0f} MB in {dt * 1e3:.1f} ms = {nbytes / dt / 1e9:.1f} GB/s, copy ok {ok}")
