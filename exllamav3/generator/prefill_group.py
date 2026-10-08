from __future__ import annotations
import torch

class PrefillGroup:
    """
    Prompt chunks of several jobs read together (EXL3_PREFILL_GROUP). Each job's chunk forward runs in its own
    greenlet on the one stream; at every CPU-split MoE layer the job hands its rows to submit() and waits. When
    every live chunk has arrived at the same layer, the CPU/streamed expert part runs once over the concatenated
    rows, so the experts streamed over PCIe for a chunk (most of a short chunk's time: ~1.2 s of a 1K-token chunk
    with Qwen3.8-Flash-Next at 362 CPU experts per layer) are streamed once for all of them, and each job gets its
    rows of the result back. Attention, recurrent layers and the GPU-resident experts still run per job.

    Only tensors the job allocated for this call cross the switch; inputs that may live in a module's static buffer
    (the dynamic-placement selection) are copied at submit time, while their producer's stream order still holds.
    """

    def __init__(self):
        self.main = None
        self.live = set()
        self.waiting = []

    def submit(self, host, layer_idx, y, sel, w):
        from greenlet import getcurrent
        g = getcurrent()
        if self.main is None or g not in self.live or len(self.live) < 2:
            return host.submit_prefill(layer_idx, y, sel, w)
        entry = {"g": g, "key": (id(host), layer_idx), "host": host, "layer": layer_idx,
                 "y": y.clone(), "sel": sel.clone(), "w": w.clone(), "out": None}
        self.waiting.append(entry)
        self.main.switch()
        out = entry["out"]
        if isinstance(out, BaseException):
            raise out
        return out

    def run(self, fns: list):
        """Run the callables (one per job) as greenlets until all have finished. A callable's exception is its own
        to handle; one that escapes is re-raised here after the others finished."""
        from greenlet import greenlet, getcurrent
        self.main = getcurrent()
        errors = []
        def wrap(fn):
            def body():
                try:
                    fn()
                except BaseException as e:
                    errors.append(e)
            return body
        lets = [greenlet(wrap(fn)) for fn in fns]
        self.live = set(lets)
        runnable = list(lets)
        try:
            while self.live:
                for g in runnable:
                    g.switch()
                    if g.dead:
                        self.live.discard(g)
                runnable = []
                if not self.waiting:
                    continue
                # Every live greenlet now waits at a layer. Serve the earliest layer's group (normally all of them)
                key = min(self.waiting, key = lambda e: e["layer"])["key"]
                batch = [e for e in self.waiting if e["key"] == key]
                self.waiting = [e for e in self.waiting if e["key"] != key]
                e0 = batch[0]
                try:
                    if len(batch) == 1:
                        outs = [e0["host"].submit_prefill(e0["layer"], e0["y"], e0["sel"], e0["w"])]
                    else:
                        out = e0["host"].submit_prefill(
                            e0["layer"],
                            torch.cat([e["y"] for e in batch], dim = 0),
                            torch.cat([e["sel"] for e in batch], dim = 0),
                            torch.cat([e["w"] for e in batch], dim = 0),
                        )
                        outs = list(out.split([e["y"].shape[0] for e in batch], dim = 0))
                except BaseException as ex:
                    outs = [ex] * len(batch)
                for e, o in zip(batch, outs):
                    e["out"] = o
                    runnable.append(e["g"])
        finally:
            self.main = None
            self.live = set()
            self.waiting = []
        if errors:
            raise errors[0]
