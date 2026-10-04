from __future__ import annotations
import os
import torch
from ..constants import PAGE_SIZE
from exllamav3.ext import exllamav3_ext as ext

# KV streaming for the QSA attention layers (exllamav3_ext/kv_stream.cu has the design).
# EXL3_KV_STREAM=1: a quantized QSA cache layer of at least EXL3_KV_STREAM_MIN tokens keeps its K/V and
# raw indexer keys in pinned, device-mapped host memory. The sparse decode reader goes through a VRAM
# pool of EXL3_KV_STREAM_SLOTS 4-token blocks per layer (8192 = the latest-selected 32K tokens), and
# sparse prefill chunks stage the job's pages into one VRAM copy of a layer, shared by every streamed
# layer of the same shape. The pooled indexer plane (what the block selection scans) stays in VRAM.
kv_stream_enabled = os.environ.get("EXL3_KV_STREAM", "0") != "0"
kv_stream_min = int(os.environ.get("EXL3_KV_STREAM_MIN", 65536))
kv_stream_slots = int(os.environ.get("EXL3_KV_STREAM_SLOTS", 8192))
_kv_stream_stats = int(os.environ.get("EXL3_KV_STREAM_STATS", 0))
_kv_stream_verify = os.environ.get("EXL3_KV_STREAM_VERIFY", "0") != "0"
kv_verify_totals = {"calls": 0, "rows": 0, "mismatch": 0, "max_diff": 0.0}

BLOCK_TOKENS = 4

# (device index, page shapes) -> staging tensors; one VRAM layer image for all streamed layers
_staging = {}


class KVStream:

    def __init__(self, layer, device: torch.device):
        self.layer = layer
        self.device = device
        num_pages = layer.qshape_k[0]
        self.num_blocks = num_pages * PAGE_SIZE // BLOCK_TOKENS
        self.n_slots = kv_stream_slots
        ns = self.n_slots
        i32 = dict(dtype = torch.int, device = device)
        self.page_table = torch.full((self.num_blocks,), -1, **i32)
        self.slot_block = torch.full((ns,), -1, **i32)
        self.slot_stamp = torch.full((ns,), -1, **i32)
        self.slot_ref = torch.zeros((ns,), **i32)
        self.ctl = torch.zeros((16,), **i32)
        self.stats = torch.zeros((4,), dtype = torch.long, device = device)
        self.miss_block = torch.zeros((ns,), **i32)
        self.miss_slot = torch.zeros((ns,), **i32)

        rows = ns * BLOCK_TOKENS
        self.s_qk = torch.zeros((rows, layer.qshape_k[2]), dtype = torch.int, device = device)
        self.s_qv = torch.zeros((rows, layer.qshape_v[2]), dtype = torch.int, device = device)
        self.s_sk = torch.zeros((rows, layer.qshape_s[2]), dtype = torch.half, device = device)
        self.s_sv = torch.zeros((rows, layer.qshape_s[2]), dtype = torch.half, device = device)

        key = (torch.device(device).index, tuple(layer.qshape_k), tuple(layer.qshape_v), tuple(layer.qshape_s))
        st = _staging.get(key)
        if st is None:
            st = _staging[key] = (
                torch.empty(layer.qshape_k, dtype = torch.int, device = device),
                torch.empty(layer.qshape_v, dtype = torch.int, device = device),
                torch.empty(layer.qshape_s, dtype = torch.half, device = device),
                torch.empty(layer.qshape_s, dtype = torch.half, device = device),
            )
        self.staging = st
        self.calls = 0
        self.verify = _kv_stream_verify

    def verify_result(self, o, o_ref, rows):
        t = kv_verify_totals
        t["calls"] += 1
        t["rows"] += rows
        if not torch.equal(o, o_ref):
            t["mismatch"] += 1
            t["max_diff"] = max(t["max_diff"], (o.float() - o_ref.float()).abs().max().item())

    def vram_tensors(self):
        return [self.s_qk, self.s_qv, self.s_sk, self.s_sv, self.page_table]

    def host_tensors(self):
        l = self.layer
        return [l.qk, l.qv, l.sk, l.sv]

    def invalidate(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, length: int):
        sl = cache_seqlens if cache_seqlens.dtype == torch.int else cache_seqlens.int()
        bt = block_table if block_table.dtype == torch.int else block_table.int()
        ext.kv_stream_invalidate(self.page_table, self.slot_block, sl, bt.contiguous(), length, PAGE_SIZE)

    def invalidate_page(self, page: int):
        bpp = PAGE_SIZE // BLOCK_TOKENS
        ext.kv_stream_invalidate_range(self.page_table, self.slot_block, page * bpp, (page + 1) * bpp)

    def slot_fits(self, rows: int, k_pad: int) -> bool:
        return rows * (k_pad // BLOCK_TOKENS + 2) <= self.n_slots

    def resolve(self, indices: torch.Tensor, block_table: torch.Tensor, seq: int) -> torch.Tensor:
        """Make every block the selection names resident; returns the selection as slot rows"""
        bt = block_table if block_table.dtype == torch.int else block_table.int()
        out = torch.empty_like(indices)
        ext.kv_stream_resolve(
            self.page_table, self.slot_block, self.slot_stamp, self.slot_ref, self.ctl, self.stats,
            self.miss_block, self.miss_slot, indices, bt.contiguous(), out, seq, PAGE_SIZE
        )
        ext.kv_stream_copy(
            self.ctl, self.miss_block, self.miss_slot,
            self.host_tensors(), [self.s_qk, self.s_qv, self.s_sk, self.s_sv],
            indices.shape[0] * (indices.shape[1] // BLOCK_TOKENS + 2)
        )
        if _kv_stream_stats:
            self.calls += 1
            if self.calls % _kv_stream_stats == 0:
                s = self.stats.tolist()
                c = self.ctl.tolist()
                print(f" -- kv_stream {id(self) & 0xffff:04x}: calls {s[2]}, lookups {s[1]}, misses {s[0]} "
                      f"({s[0] / max(s[1], 1) * 100:.2f}%), overflow {c[3]}")
        return out

    def stage(self, block_table: torch.Tensor, cache_seqlens_cpu: torch.Tensor, seq: int):
        """Copy the job pages a chunk reads into the shared VRAM layer image; returns the image's tensors.
        Runs of consecutive pages go through the DMA engine, single pages through one gather launch"""
        bt = block_table.cpu()
        pages = []
        for b in range(bt.shape[0]):
            n_p = min(-(-(int(cache_seqlens_cpu[b]) + seq) // PAGE_SIZE), bt.shape[1])
            pages.append(bt[b, :n_p])
        p = torch.unique(torch.cat(pages).int())                    # sorted
        brk = torch.nonzero(p[1:] - p[:-1] != 1).flatten() + 1
        starts = torch.cat([torch.zeros(1, dtype = torch.long), brk])
        ends = torch.cat([brk, torch.tensor([p.numel()])])
        long_run = (ends - starts) >= 2
        runs = torch.stack([p[starts[long_run]], p[ends[long_run] - 1] + 1], dim = 1).int().contiguous()
        if runs.numel():
            ext.kv_stage_runs(runs, self.host_tensors(), list(self.staging))
        singles = p[starts[~long_run]]
        if singles.numel():
            ext.kv_stage_pages(singles.to(self.device, non_blocking = True), self.host_tensors(), list(self.staging))
        return self.staging
