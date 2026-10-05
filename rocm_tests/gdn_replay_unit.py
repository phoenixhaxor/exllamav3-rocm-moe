# Replay rewind (EXL3_GDN_REPLAY) against stored history, at the kernel level: the same verify pass (random q/k/v,
# decay, beta; Qwen3.8 GDN shapes) through the history kernels and through the replay kernels, then a rewind to every
# accepted count. Outputs and rewound states must be bit-identical. bsz 1 takes the register-resident kernel, bsz 3
# the 128-column kernel; slots are permuted.
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext

torch.manual_seed(0)
dev = torch.device("cuda:0")
NK, NV, HD = 16, 48, 128
F = 2 * NK * HD + NV * HD
TOK = NK * HD * 4 + NV * HD * 2 + NV * 8
H = 4                       # max_history
fails = []

def run(bsz, S, slots_list, accepted):
    n_slots = max(slots_list) + 1
    init = torch.randn((n_slots, NV, HD, HD), device = dev) * 0.05
    qkv = (torch.randn((bsz, S, F), device = dev) * 0.5).to(torch.bfloat16)
    g = -torch.rand((bsz, S, NV), device = dev) * 0.5
    beta = torch.rand((bsz, S, NV), device = dev).to(torch.bfloat16)
    slots = torch.tensor(slots_list, dtype = torch.int, device = dev)

    hist = torch.zeros((n_slots, H + 1, NV, HD, HD), device = dev)
    hist[:, 0].copy_(init)
    out_h = torch.empty((bsz, S, NV, HD), dtype = torch.bfloat16, device = dev)
    ext.cuda_recurrent_gated_delta_rule(qkv, g, beta, hist, out_h, NK, NV, HD, HD, slots, True, None)

    rs = torch.zeros((n_slots, 2, NV, HD, HD), device = dev)
    rs[:, 0].copy_(init)
    rbuf = torch.zeros((n_slots, H + 1, TOK), dtype = torch.uint8, device = dev)
    out_r = torch.empty_like(out_h)
    ext.cuda_recurrent_gated_delta_rule(qkv, g, beta, rs, out_r, NK, NV, HD, HD, slots, True, rbuf)

    tag = f"bsz {bsz} S {S}"
    if not torch.equal(out_h, out_r):
        fails.append(f"{tag}: outputs differ (max {(out_h.float() - out_r.float()).abs().max().item():.3g})")
    if not torch.equal(hist[slots.long(), 0], rs[slots.long(), 0]):
        fails.append(f"{tag}: final states differ")
    if not torch.equal(rs[slots.long(), 1], init[slots.long()]):
        fails.append(f"{tag}: snapshot is not the starting state")

    # Rewind each slot to `accepted` tokens: history copies state index `accepted`, replay re-runs from the snapshot
    for b, slot in enumerate(slots_list):
        a = accepted[b]
        num_tokens = S - a
        last_history = S - 1
        if num_tokens == 0:
            continue
        ref = hist[slot, last_history + 1 - num_tokens].clone()
        es = rs.element_size()
        job = ext.StateReplayJob(rs.data_ptr() + slot * rs.stride(0) * es,
                                 rbuf.data_ptr() + slot * rbuf.stride(0), a, NK, NV, TOK)
        ext.batched_state_replay([job], 0)
        torch.cuda.synchronize()
        if not torch.equal(ref, rs[slot, 0]):
            d = (ref - rs[slot, 0]).abs().max().item()
            fails.append(f"{tag} slot {slot} accepted {a}: rewound state differs (max {d:.3g})")
    print(f"  {tag}, slots {slots_list}, accepted {accepted}: checked", flush = True)

for S in (2, 3, 5):
    for a in range(1, S + 1):
        run(1, S, [0], [a])
for S in (3, 5):
    run(3, S, [2, 0, 1], [1, S, 2])
    run(3, S, [1, 2, 0], [S - 1, 1, S])

# Plain (no-history) pass unchanged by a replay buffer
x = torch.randn((2, 1, NV, HD, HD), device = dev)
y = x.clone()
qkv = (torch.randn((2, 1, F), device = dev) * 0.5).to(torch.bfloat16)
g = -torch.rand((2, 1, NV), device = dev) * 0.5
beta = torch.rand((2, 1, NV), device = dev).to(torch.bfloat16)
o1 = torch.empty((2, 1, NV, HD), dtype = torch.bfloat16, device = dev); o2 = torch.empty_like(o1)
ext.cuda_recurrent_gated_delta_rule(qkv, g, beta, x, o1, NK, NV, HD, HD, None, False, None)
ext.cuda_recurrent_gated_delta_rule(qkv, g, beta, y, o2, NK, NV, HD, HD, None, False,
                                    torch.zeros((2, H + 1, TOK), dtype = torch.uint8, device = dev))
if not (torch.equal(x, y) and torch.equal(o1, o2)):
    fails.append("no-history pass changed by the replay argument")

for f in fails: print("  FAIL", f)
print("GDN_REPLAY_UNIT", "PASS" if not fails else "FAIL")
