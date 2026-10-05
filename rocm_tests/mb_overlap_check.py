# Micro-batch overlap (EXL3_MB_OVERLAP) against the same split run one micro-batch after the other: five greedy jobs
# decode together in the order seq, seq, overlap, seq. Engine decode is not bit-deterministic run to run, so the
# seq/seq pairs give the noise floor: per pair, the first step where some row's argmax differs, the largest logit
# difference before it, and the tokens each row shares with the other run. Corrupted buffers would show up as an
# early divergence and large logit differences in the overlap pairs only. Also reports the extra VRAM of the second
# micro-batch's workspaces and decode speed per mode.
#   python mb_overlap_check.py -m <model> [--mcs 362] [--steps 200]
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator import generator as genmod
from exllamav3.generator.sampler import ComboSampler

PROMPTS = [
    "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry and unit tests.",
    "Write a long, vivid story about a lighthouse keeper who finds a message in a bottle.",
    "Explain in detail how a CPU pipeline handles branch misprediction, with concrete examples.",
    "Write a SQL schema and twenty example queries for a library management system, with comments.",
    "Describe the history of the printing press and its effect on European society, in depth.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 362)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--steps", type = int, default = 200)
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 65536, max_batch_size = 5, layer_type = CacheLayer_quant,
                  k_bits = 8, v_bits = 8)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 5, max_chunk_size = 8192)

    rec = []
    orig = model.forward_overlap
    def fo(batches, interleave = True):
        outs = orig(batches, interleave = interleave)
        rec.append(torch.cat(outs, dim = 0)[:, -1, :].float().cpu())
        return outs
    model.forward_overlap = fo

    def run(mode):
        genmod._mb_overlap = mode
        rec.clear()
        gen.clear_queue()
        for i, p in enumerate(PROMPTS):
            ids = tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                             encode_special_tokens = True)
            gen.enqueue(Job(input_ids = ids, max_new_tokens = args.steps, sampler = ComboSampler(top_k = 1),
                            stop_conditions = [], identifier = i))
        toks = {i: [] for i in range(len(PROMPTS))}
        m0 = torch.cuda.memory_allocated()
        t0 = None; n0 = 0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                    toks[r["identifier"]] += r["token_ids"].view(-1).tolist()
            if t0 is None and len(rec) == 5:
                torch.cuda.synchronize(); t0 = time.perf_counter(); n0 = len(rec)
        torch.cuda.synchronize()
        rate = (len(rec) - n0) * len(PROMPTS) / (time.perf_counter() - t0)
        extra = (torch.cuda.memory_allocated() - m0) / 2**20
        print(f"{mode:8s}: {len(rec)} batched steps, {rate:.1f} tok/s total, allocated +{extra:.0f} MB", flush = True)
        return list(rec), toks

    def compare(tag, a, b):
        la, ta = a; lb, tb = b
        n = min(len(la), len(lb))
        first = n; mx = 0.0
        for s in range(n):
            if la[s].shape != lb[s].shape or not torch.equal(la[s].argmax(-1), lb[s].argmax(-1)):
                first = s
                break
            mx = max(mx, (la[s] - lb[s]).abs().max().item())
        same = []
        for i in ta:
            k = 0
            while k < min(len(ta[i]), len(tb[i])) and ta[i][k] == tb[i][k]: k += 1
            same.append(k)
        print(f"{tag}: argmax equal for {first}/{n} steps, max logit diff before that {mx:.4f}, "
              f"identical tokens per row {same}", flush = True)

    runs = {}
    for name, mode in (("seq1", "seq"), ("seq2", "seq"), ("ovl", "1"), ("seq3", "seq")):
        runs[name] = run(mode)
    compare("baseline seq2 vs seq3", runs["seq2"], runs["seq3"])
    compare("overlap  ovl  vs seq2", runs["ovl"], runs["seq2"])
    compare("overlap  ovl  vs seq3", runs["ovl"], runs["seq3"])
    for i in (0, 1):
        print(f"--- overlap row {i}: " + tok.decode(torch.tensor(runs["ovl"][1][i][:80])).replace("\n", " | "), flush = True)
    gen.clear_queue()
    print("MB_OVERLAP_CHECK_DONE", flush = True)

if __name__ == "__main__":
    main()
