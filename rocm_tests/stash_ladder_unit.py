# Recurrent checkpoint retention (EXL3_STASH_LADDER) against plain LRU, on a mock page table: conversations grow turn
# by turn with a checkpoint every 2048 tokens, the stash holds 18. Checks that the ladder keeps checkpoints over the
# whole conversation (a prompt that changes older history finds a rung close below the change), keeps every live
# conversation's tip, and lets an idle conversation's ladder go first. CPU only.
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exllamav3.cache.recurrent as rec
from exllamav3.constants import PAGE_SIZE

class Page:
    def __init__(self, phash, prev_hash):
        self.phash, self.prev_hash, self.kv_position = phash, prev_hash, PAGE_SIZE

class PT:
    max_pages = 1 << 20
    def __init__(self): self.pages = {}
    def get_live_page(self, h): return self.pages.get(h)
    def is_resumable(self, h): return h in self.pages
    referenced_pages = {}
    unreferenced_pages = {}

class State:
    def __init__(self, pos): self.pos = pos
    def stash(self): return {"position": self.pos, "checkpoint_size": 100}

class Model:
    loaded_tp = False

def conv(pt, name, n_pages, base = None, base_pages = 0):
    """Page hashes of a conversation; shares the first base_pages pages of base"""
    hs = []
    for i in range(n_pages):
        h = base[i] if base is not None and i < base_pages else f"{name}-{i}"
        pt.pages.setdefault(h, Page(h, hs[-1] if hs else None))
        hs.append(h)
    return hs

def grow(rc, hs, upto, every = 2048):
    for pos in range(every, upto + 1, every):
        rc.put(hs[pos // PAGE_SIZE - 1], State(pos))

def best_below(rc, hs, pos):
    for pi in range(pos // PAGE_SIZE - 1, -1, -1):
        if hs[pi] in rc:
            return (pi + 1) * PAGE_SIZE
    return 0

def run(ladder):
    rec._ladder = ladder
    fails = []
    pt = PT()
    rc = rec.RecurrentCache(Model(), max_size = 18 * 100)
    rc.pagetable = pt

    # One conversation, 100K tokens
    a = conv(pt, "a", 400)
    grow(rc, a, 100352)
    pos = sorted(v["position"] for v in rc.values())
    gaps = [b - a_ for a_, b in zip([0] + pos, pos)]
    tip = 100352 in pos
    worst = max(d - best_below(rc, a, d) for d in range(2048, 100352, 1024))
    print(f"  one conversation: {len(pos)} checkpoints, lowest {pos[0]}, largest gap {max(gaps)}, tip kept {tip}, "
          f"worst replay for a change below the tip {worst}")
    if ladder:
        if max(gaps) > 3 * 100352 // 18: fails.append("ladder gap too large")
        if not tip: fails.append("tip dropped")
        if worst > 3 * 100352 // 18: fails.append("worst replay too large")

    # Second conversation shares the 8K system prompt, grows while the first goes idle
    for v in rc.values(): v["t_use"] -= 3600
    b = conv(pt, "b", 400, a, 32)
    grow(rc, b, 60416)
    na = sum(1 for k in rc if k in set(a[32:]))
    nb = sum(1 for k in rc if k in set(b[32:]))
    tip_a = a[100352 // PAGE_SIZE - 1] in rc
    print(f"  idle a + active b: a keeps {na}, b keeps {nb}, a's tip kept {tip_a}")
    if ladder and not nb > na: fails.append("idle conversation kept more than the active one")
    return fails

f_lru = run(False)
print("LRU above, ladder below")
f = run(True)
print("STASH_LADDER_UNIT", "PASS" if not f else "FAIL " + "; ".join(f))
