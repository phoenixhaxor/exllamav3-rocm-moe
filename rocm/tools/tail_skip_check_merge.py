# Variant for the merged branch (phase2): local texts, module-level knobs. Original: rocm_tests/tail_skip_check.py.
# Tail skip (EXL3_MOE_TAIL_SKIP) for the CPU-resident experts of decode-size jobs: quality and speed in one load.
# 1. Teacher-forced decode: each text's first --prompt tokens are prefilled, the next --score tokens go through the
#    decode path in windows of --win tokens (an MTP verify window's shape). Logits at every scored position with the
#    tail rank R against R = 0: mean KL, top-1 agreement and the forced text's perplexity ratio; R = 0 against itself
#    is the noise floor. Also the share of CPU-resident picks skipped at each R.
# 2. Speed: greedy generation with MTP drafts, code and prose prompts, R = 0 and R = --speed-r interleaved.
#   TEXT_DIR=<dir with prosa-230k.txt> python rocm/tools/tail_skip_check_merge.py -m <model> [--mcs 362] [--ranks 7,8,9] [--speed-r 7]
import os, sys, time, argparse, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.model import moe_cpu_host
from exllamav3.generator.sampler import ComboSampler

def texts(tok, repo):
    # Local texts only (no download): German prose and code from the test folder, plus repo source files
    td = os.environ.get("TEXT_DIR", "./texts")
    prose = open(os.path.join(td, "prosa-230k.txt"), encoding = "utf-8").read()
    out = [("prose 1", prose[20000:40000]), ("prose 2", prose[400000:420000])]
    for name, f in (("code 1", "exllamav3/generator/generator.py"), ("code 2", "exllamav3/model/moe_cpu_host.py"),
                    ("code 3", "exllamav3/modules/block_sparse_mlp.py")):
        out.append((name, open(os.path.join(repo, f), encoding = "utf-8").read()[30000:60000]))
    chat = ("<|im_start|>user\nExplain how a hash map resolves collisions, with an example in Python.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n" + prose[600000:620000])
    out.append(("chat", chat))
    return [(n, tok.encode(t, encode_special_tokens = True)) for n, t in out]

class _Knobs:
    """The merged code keeps the tail-skip knobs as module-level variables of model.moe_cpu_host (the fork had a TUNING object)"""
    @property
    def tail_skip(self): return moe_cpu_host._tail_skip
    @tail_skip.setter
    def tail_skip(self, v): moe_cpu_host._tail_skip = v
    @property
    def tail_stats(self): return moe_cpu_host._tail_stats
    @tail_stats.setter
    def tail_stats(self, v): moe_cpu_host._tail_stats = v

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 362)
    ap.add_argument("--ranks", default = "7,8,9")
    ap.add_argument("--prompt", type = int, default = 400)
    ap.add_argument("--score", type = int, default = 300)
    ap.add_argument("--win", type = int, default = 3)
    ap.add_argument("--speed-r", type = int, default = 7)
    ap.add_argument("--speed-tokens", type = int, default = 600)
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    q8 = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    cache = Cache(model, max_num_tokens = 16384, max_history = 4, max_batch_size = 1, **q8)
    model.load(progressbar = False)
    dcache = Cache(draft, max_num_tokens = 16384, **q8)
    draft.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    T = _Knobs()
    T.tail_stats = 0

    # 1. Teacher-forced decode
    L = args.prompt + args.score
    data = [(n, ids) for n, ids in texts(tok, repo) if ids.shape[-1] >= L + 1]
    print("texts:", ", ".join(n for n, _ in data), flush = True)
    BS = (L + 8 + 255) // 256 * 256
    def forced(ids):
        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, BS)}
        model.prefill(input_ids = ids[:, :args.prompt], params = params)
        rs = params.get("recurrent_states")
        outs = []
        pos = args.prompt
        while pos < L:
            w = min(args.win, L - pos)
            p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, BS),
                 "recurrent_states": rs}
            outs.append(model.forward(input_ids = ids[:, pos:pos + w], params = p)[0].float().cpu())
            pos += w
        if rs:
            for st in (rs.values() if isinstance(rs, dict) else rs):
                cache.release_state(st)
        return torch.cat(outs, dim = 0)   # (score, vocab): logits predicting tokens prompt+1 .. L
    def run_all(r):
        T.tail_skip = r
        return [forced(ids[:, :L]) for _, ids in data]
    stats_r = {}
    ref = run_all(0)
    ref2 = run_all(0)
    res = {r: None for r in ranks}
    for r in ranks:
        # skipped share at this rank, from the kernel's counters
        T.tail_stats = 10 ** 9
        host = list(config.moe_cpu_hosts.values())[0] if hasattr(config, "moe_cpu_hosts") else None
        if host is not None:
            for k in list(host.dev_count.keys()):
                if isinstance(k, tuple) and k[1] == "tail": host.dev_count[k].zero_()
        res[r] = run_all(r)
        if host is not None:
            tot = [0, 0]
            for k, v in host.dev_count.items():
                if isinstance(k, tuple) and k[1] == "tail":
                    c, s = v.tolist(); tot[0] += c; tot[1] += s
            stats_r[r] = tot
        T.tail_stats = 0
    T.tail_skip = 0

    def compare(a_list, b_list):
        kls, agree, n, nll_a, nll_b = [], 0, 0, 0.0, 0.0
        for (name, ids), a, b in zip(data, a_list, b_list):
            tgt = ids[0, args.prompt + 1: args.prompt + 1 + a.shape[0]]
            a, b = a[:tgt.shape[0]], b[:tgt.shape[0]]
            la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
            kls.append((la.exp() * (la - lb)).sum(-1))
            agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
            n += tgt.shape[0]
            nll_a += -la.gather(-1, tgt.view(-1, 1)).sum().item()
            nll_b += -lb.gather(-1, tgt.view(-1, 1)).sum().item()
        kl = torch.cat(kls)
        return kl.mean().item(), kl.quantile(0.99).item(), agree / n, math.exp((nll_b - nll_a) / n), n
    m = compare(ref, ref2)
    print(f"R=0 vs R=0 (noise): KL mean {m[0]:.5f} p99 {m[1]:.4f}, top-1 {100 * m[2]:.2f}%, ppl x{m[3]:.4f}, "
          f"{m[4]} tokens", flush = True)
    for r in ranks:
        m = compare(ref, res[r])
        sk = stats_r.get(r)
        sk_s = f", skipped {sk[1]}/{sk[0]} CPU picks ({100 * sk[1] / max(sk[0], 1):.1f}%)" if sk else ""
        print(f"R={r} vs R=0: KL mean {m[0]:.5f} p99 {m[1]:.4f}, top-1 {100 * m[2]:.2f}%, ppl x{m[3]:.4f}{sk_s}",
              flush = True)

    # 2. Speed with MTP drafts
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft, draft_cache = dcache,
                    num_draft_tokens = 2, max_batch_size = 1)
    prompts = {
        "code": "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, statistics "
                "counters, a decorator interface and a unittest suite. Output only the code.",
        "prose": "Write a long, vivid story about a lighthouse keeper who finds a message in a bottle.",
    }
    speed = {}
    for rep in range(3):
        for r in ([0, args.speed_r] if rep % 2 == 0 else [args.speed_r, 0]):
            T.tail_skip = r
            for name, p in prompts.items():
                ids = tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                 encode_special_tokens = True)
                job = Job(input_ids = ids, max_new_tokens = args.speed_tokens, sampler = ComboSampler(top_k = 1),
                          stop_conditions = [])
                gen.enqueue(job)
                n = 0; t0 = None
                while gen.num_remaining_jobs():
                    for res_ in gen.iterate():
                        if res_.get("token_ids") is not None and res_["token_ids"].numel():
                            if t0 is None:
                                t0 = time.perf_counter()
                            else:
                                n += res_["token_ids"].numel()
                dt = time.perf_counter() - t0
                speed.setdefault((name, r), []).append(n / dt)
    T.tail_skip = 0
    for name in prompts:
        a = sorted(speed[(name, 0)]); b = sorted(speed[(name, args.speed_r)])
        print(f"speed {name}: R=0 {a[len(a) // 2]:.1f} tok/s ({', '.join(f'{x:.1f}' for x in a)}), "
              f"R={args.speed_r} {b[len(b) // 2]:.1f} tok/s ({', '.join(f'{x:.1f}' for x in b)})", flush = True)
    print("TAIL_SKIP_CHECK_DONE", flush = True)

if __name__ == "__main__":
    main()
