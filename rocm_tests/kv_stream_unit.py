# KV streaming kernels against a torch reference: random multi-step selections over a paged host
# copy with shuffled block tables, writes that invalidate, page copies and eviction pressure. After
# every resolve, the slot rows it returns must hold exactly the host rows of the selected positions
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["EXL3_KV_STREAM"] = "1"
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.kv_stream import KVStream

class FakeLayer:
    def __init__(self, num_pages):
        self.qshape_k = (num_pages, PAGE_SIZE, 128)
        self.qshape_v = (num_pages, PAGE_SIZE, 128)
        self.qshape_s = (num_pages, PAGE_SIZE, 16)
        di = 0
        self.qk = ext.kv_host_alloc(list(self.qshape_k), torch.int, di)
        self.qv = ext.kv_host_alloc(list(self.qshape_v), torch.int, di)
        self.sk = ext.kv_host_alloc(list(self.qshape_s), torch.half, di)
        self.sv = ext.kv_host_alloc(list(self.qshape_s), torch.half, di)

def fill(t, page_rows = None):
    if t.dtype == torch.int:
        v = torch.randint(-2**31, 2**31 - 1, t.shape, dtype = torch.int, device = "cuda")
    else:
        v = torch.randn(t.shape, device = "cuda").half()
    t.copy_(v)

def main():
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    num_pages = 64                      # 16K tokens, 4096 blocks
    os.environ["EXL3_KV_STREAM_SLOTS"] = "2048"
    import exllamav3.cache.kv_stream as ks
    ks.kv_stream_slots = 2048
    L = FakeLayer(num_pages)
    for t in (L.qk, L.qv, L.sk, L.sv): fill(t)
    torch.cuda.synchronize()
    S = KVStream(L, dev)
    bsz, seq, K_pad = 2, 3, 2080
    npr = num_pages // bsz
    perm = torch.randperm(num_pages, dtype = torch.int).view(bsz, npr).to(dev)
    errs = 0
    for step in range(200):
        seqlen = 4000 + step * 20          # stays inside the 32-page block table rows (8192 tokens)
        # selections: mostly stable hot set + random, ascending, -1 padded tail
        idx = torch.full((bsz * seq, K_pad), -1, dtype = torch.int, device = dev)
        for r in range(bsz * seq):
            hot = torch.arange(0, 1200, 3, device = dev) * 4 + (step % 4)
            rnd = torch.randint(0, seqlen, (200 if step % 10 == 0 else 60,), device = dev)
            sel = torch.unique(torch.cat([hot, rnd]))[:K_pad]
            idx[r, :sel.numel()] = sel.int()
        out = S.resolve(idx, perm, seq)
        # reference
        valid = idx >= 0
        b = (torch.arange(bsz * seq, device = dev) // seq).unsqueeze(1).expand_as(idx)
        pos = idx.clamp(min = 0).long()
        page = perm[b, pos // PAGE_SIZE].long()
        hrow = page * PAGE_SIZE + pos % PAGE_SIZE
        srow = out.long()
        if bool(((srow < 0) & valid).any()):
            print(f"step {step}: unresolved rows"); errs += 1; break
        for h, s in ((L.qk, S.s_qk), (L.qv, S.s_qv), (L.sk, S.s_sk), (L.sv, S.s_sv)):
            hv = h.view(-1, h.shape[-1])[hrow[valid]]
            sv = s[srow[valid]]
            if not torch.equal(hv, sv):
                print(f"step {step}: slot contents differ"); errs += 1; break
        # writes: new tokens of both sequences (rewrite host rows + invalidate), sometimes a page copy
        if step % 3 == 0:
            sl = torch.full((bsz,), seqlen - 50, dtype = torch.int, device = dev)
            length = 60
            for bb in range(bsz):
                p = torch.arange(seqlen - 50, seqlen + 10, device = dev)
                rows = perm[bb, p // PAGE_SIZE].long() * PAGE_SIZE + p % PAGE_SIZE
                for h in (L.qk, L.qv, L.sk, L.sv):
                    hv = h.view(-1, h.shape[-1])
                    hv[rows] = (hv[rows] + 1) if h.dtype == torch.int else (hv[rows] + 1).half()
            S.invalidate(sl, perm, length)
        if step % 17 == 5:
            src, dst = 3, int(perm[1, 2])
            for h in (L.qk, L.qv, L.sk, L.sv):
                h[dst].copy_(h[src])
            S.invalidate_page(dst)
    c = S.ctl.tolist(); s = S.stats.tolist()
    print(f"calls {s[2]} lookups {s[1]} misses {s[0]} ({s[0] / max(s[1], 1) * 100:.1f}%) overflow {c[3]}")
    print("KV_STREAM_UNIT", "PASS" if errs == 0 and c[3] == 0 else "FAIL")

if __name__ == "__main__":
    main()
