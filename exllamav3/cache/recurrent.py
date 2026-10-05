import os
import time
from collections import OrderedDict
from ..constants import PAGE_SIZE
from ..util.memory import malloc_trim

# EXL3_STASH_LADDER=1 (default): under pressure, drop the restorable checkpoint whose loss costs the least replay
# instead of the least recently used one. Checkpoints form chains along each conversation's page chain; dropping one
# widens the gap below its successors, and that gap (the tokens a prompt diverging just below a successor would
# replay) is the cost, discounted by how long the conversation has been idle (EXL3_STASH_LADDER_IDLE seconds halve
# it). A conversation then keeps a ladder of checkpoints over its whole length instead of only its latest ones, so a
# client that edits or prunes older history replays from the nearest rung, not from the start
_ladder = os.environ.get("EXL3_STASH_LADDER", "1") != "0"
_ladder_idle = float(os.environ.get("EXL3_STASH_LADDER_IDLE", 600))
# A tip (a conversation's latest checkpoint) is where its next turn resumes, nearly certain to be needed, while a
# rung below only helps a prompt that changes older history: its gap counts this many times
_ladder_tip = float(os.environ.get("EXL3_STASH_LADDER_TIP", 4))

# Checkpoint stashes are MB-scale host allocations with LRU (i.e. interleaved) lifetimes —
# exactly the churn glibc retains after free (issue #277). Return memory to the OS once
# enough has been released; per-event cost at this threshold is a few ms. The accumulator
# is per-process, which also gives each tensor-parallel rank its own (their stashes live
# in the child processes)
_TRIM_THRESHOLD = 256 * 1024**2
_freed_bytes = 0

def note_freed(nbytes: int):
    global _freed_bytes
    _freed_bytes += nbytes
    if _freed_bytes >= _TRIM_THRESHOLD:
        _freed_bytes = 0
        malloc_trim()


class RecurrentCache(OrderedDict):
    def __init__(
        self,
        model,
        max_size: int = 4 * 1024**3,
    ):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.model = model

        # Optionally set by the Generator; enables stranded-first eviction and staleness metrics
        self.pagetable = None
        self.metrics = {
            "stash_evictions": 0,           # checkpoints dropped by LRU pressure
            "stash_evictions_stranded": 0,  # of those, checkpoints that were already unrestorable
            "stash_evictions_live_kv": 0,   # of those, checkpoints whose anchor KV page was still cached
            "stash_pruned": 0,              # stranded checkpoints dropped by prune_stranded()
            "stash_evictions_ladder": 0,    # of the evictions, chosen by replay cost (EXL3_STASH_LADDER)
        }


    def get_stashed(self, key, default = None):
        """
        Fetch state from cache and move it to the end of the queue
        """
        if key in self:
            self.move_to_end(key)
            self[key]["t_use"] = time.monotonic()
            return self[key]
        return default


    def put(self, key, state):
        """
        Add state to cache
        """
        if key in self:
            self.move_to_end(key)
            self[key]["t_use"] = time.monotonic()
        else:
            stashed_state = state.stash()
            stashed_state["t_use"] = time.monotonic()
            state_size = stashed_state["checkpoint_size"]
            while self.update_total_size() + state_size > self.max_size:
                assert self.current_size >= 0, "Not enough space in cache for single state"
                pt = self.pagetable

                # A checkpoint whose anchor page chain has been broken by KV eviction can never be restored by
                # an allocation, so drop stranded checkpoints (oldest first) before restorable ones. This is a
                # pure win: if the conversation returns, the replay prefill recreates the same checkpoint at no
                # extra cost, since the missing pages force a replay past this position either way.
                popped_key = None
                if pt is not None:
                    for k in self:
                        if not pt.is_resumable(k):
                            popped_key = k
                            break
                if popped_key is not None:
                    popped = self.pop(popped_key)
                    self.metrics["stash_evictions_stranded"] += 1
                elif _ladder and pt is not None and len(self) > 1 and \
                        (popped_key := self.ladder_victim(pt)) is not None:
                    popped = self.pop(popped_key)
                    self.metrics["stash_evictions_ladder"] += 1
                else:
                    popped_key, popped = self.popitem(last = False)
                    if pt is not None:
                        page = pt.referenced_pages.get(popped_key) or pt.unreferenced_pages.get(popped_key)
                        if page is not None and page.kv_position == PAGE_SIZE:
                            self.metrics["stash_evictions_live_kv"] += 1

                self.metrics["stash_evictions"] += 1
                note_freed(popped["checkpoint_size"])
                if self.model.loaded_tp:
                    self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))

            self[key] = stashed_state
            self.update_total_size()


    def ladder_victim(self, pt):
        """
        The restorable checkpoint whose removal costs the least replay (see EXL3_STASH_LADDER), or None.
        Each checkpoint's parent is the nearest other checkpoint on its page chain; removing a checkpoint
        widens the gap below each of its children to the child's position less its own parent's, and below
        a tip (no children) the conversation's next turn would replay from the parent instead
        """
        keys = list(self.keys())
        pos = {k: self[k]["position"] for k in keys}
        idx = {k: i for i, k in enumerate(keys)}
        parent = {}
        for k in keys:
            # Walk the page chain down from the checkpoint's anchor page to the first other checkpoint
            h = pt.get_live_page(k)
            h = h.prev_hash if h is not None else None
            p = None
            steps = 0
            while h is not None and steps <= pt.max_pages:
                if h in idx:
                    p = h
                    break
                page = pt.get_live_page(h)
                if page is None:
                    break
                h = page.prev_hash
                steps += 1
            parent[k] = p
        children = {k: [] for k in keys}
        for k, p in parent.items():
            if p is not None:
                children[p].append(k)

        # A conversation's idle time is that of its most recently used checkpoint at or above this one
        t_use = {k: self[k].get("t_use", 0.0) for k in keys}
        recent = dict(t_use)
        for k in sorted(keys, key = lambda k: -pos[k]):
            p = parent[k]
            if p is not None:
                recent[p] = max(recent[p], recent[k])
        now = time.monotonic()

        best = None
        best_cost = None
        for k in keys:
            base = pos[parent[k]] if parent[k] is not None else 0
            if children[k]:
                gap = max(pos[c] for c in children[k]) - base
            else:
                gap = (pos[k] - base) * _ladder_tip
            cost = gap / (1.0 + max(0.0, now - recent[k]) / _ladder_idle)
            # Ties (and equal gaps) fall to the least recently used
            if best_cost is None or cost < best_cost:
                best, best_cost = k, cost
        return best


    def prune_stranded(self) -> int:
        """
        Drop all checkpoints whose anchor page chain has been broken by KV eviction. A stranded checkpoint can
        never be restored by an allocation, and if its conversation returns, the replay prefill recreates it at
        no extra cost, so this only frees system RAM that would otherwise sit dead until LRU pressure reaches it.
        Intended to be called when the generator goes idle.
        """
        if self.pagetable is None:
            return 0
        stranded = [k for k in self if not self.pagetable.is_resumable(k)]
        for k in stranded:
            popped = self.pop(k)
            self.metrics["stash_pruned"] += 1
            note_freed(popped["checkpoint_size"])
            if self.model.loaded_tp:
                self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
        if stranded:
            self.update_total_size()
        return len(stranded)


    def update_total_size(self):
        seen = set()
        total = 0
        for v in self.values():
            if id(v) in seen:
                continue
            seen.add(id(v))
            total += v["checkpoint_size"]
        self.current_size = total
        return total


# Checkpoint handles key the per-rank recurrent_cache dicts and must be unique across all
# recurrent module types (GDN, short-conv, SWA states all stash through the same dict)
_next_checkpoint_handle = 0

def new_checkpoint_handle() -> int:
    global _next_checkpoint_handle
    h = _next_checkpoint_handle
    _next_checkpoint_handle += 1
    return h


# Per-rank functions for tensor-parallel mode

def mp_cache_recurrent_clear(local_context: dict, cache_id: int, slot: int):
    recurrent_modules = local_context["recurrent_modules"]
    for module in recurrent_modules:
        recurrent_layer = module.tp_recurrent_lookup[cache_id]
        recurrent_layer.clear(slot)


def mp_cache_recurrent_stash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = []
    for module in recurrent_modules:
        l = module.tp_recurrent_lookup[cache_id]
        stashed.append(l.stash(slot, position))
    recurrent_cache[cp_handle] = stashed


def mp_cache_recurrent_unstash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache[cp_handle]
    for module, s in zip(recurrent_modules, stashed):
        l = module.tp_recurrent_lookup[cache_id]
        l.unstash(slot, s, position)


def _stashed_bytes(obj) -> int:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_stashed_bytes(o) for o in obj)
    return 0


def mp_cache_recurrent_del(local_context: dict, cache_id: int, cp_handle: int):
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache.pop(cp_handle)
    note_freed(_stashed_bytes(stashed))
