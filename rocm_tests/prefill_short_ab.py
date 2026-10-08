# Short-prompt prefill vs the streaming threshold: the CPU worker computes experts routed by fewer than stream_t of a
# chunk's tokens, the rest are streamed over PCIe and computed on the GPU. One load; for each prompt length the
# thresholds run interleaved over fresh prompts (distinct wikitext offsets, n-gram rows pre-warmed), and the median
# time per (length, threshold) is reported. Agent turns add 1-6K tokens per request, which is this regime.
#   python prefill_short_ab.py -m <model> [--mcs 362] [--lens 1024,2048,4096,6144] [--ts 4,8,12,16,24] [--reps 3]
import os, sys, time, argparse, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 362)
    ap.add_argument("--chunk", type = int, default = 8192)
    ap.add_argument("--lens", default = "1024,2048,4096,6144")
    ap.add_argument("--ts", default = "4,8,12,16,24")
    ap.add_argument("--reps", type = int, default = 3)
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]
    ts = [int(x) for x in args.ts.split(",")]
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 32768, max_batch_size = 1, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    model.load(progressbar = False, max_chunk_size = args.chunk)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_chunk_size = args.chunk)
    text = tok.encode(open(os.path.expanduser("~/wikitext2_test.txt")).read())
    hosts = list(getattr(config, "moe_cpu_hosts", {}).values())

    def set_t(t):
        for h in hosts:
            for st in h.sstate.values():
                st["stream_t"] = t

    def run(ids):
        job = Job(input_ids = ids, max_new_tokens = 1, sampler = ComboSampler(temperature = 0.0, top_k = 1),
                  stop_conditions = [])
        gen.enqueue(job)
        torch.cuda.synchronize(); t0 = time.perf_counter(); first = None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if first is None and r.get("token_ids") is not None and r["token_ids"].numel():
                    first = time.perf_counter()
        gen.pagetable.reset_page_table()
        if getattr(gen, "recurrent_cache", None) is not None:
            gen.recurrent_cache.clear()
        return (first or time.perf_counter()) - t0

    # Prompt offsets: one per (length, rep, threshold); every prompt runs once untimed first (n-gram page cache)
    plan = []
    off = 1000
    for L in lens:
        for rep in range(args.reps):
            order = ts[rep % len(ts):] + ts[:rep % len(ts)]
            for t in order:
                plan.append((L, t, off))
                off += L + 64
    print(f"{len(plan)} prompts, warming", flush = True)
    run(text[:, 500:900])   # creates the stream state
    for L, t, o in plan:
        run(text[:, o:o + L])
    print("stream state:", [(k, st["stream_t"], round(st.get("bw", 0), 1)) for h in hosts for k, st in h.sstate.items()],
          flush = True)
    res = {}
    for L, t, o in plan:
        set_t(t)
        res.setdefault((L, t), []).append(run(text[:, o:o + L]))
    set_t(8)
    for L in lens:
        line = []
        for t in ts:
            m = statistics.median(res[(L, t)])
            line.append(f"t{t} {m:.2f}s ({L / m:.0f}/s)")
        print(f"{L:5d} tokens: " + " | ".join(line), flush = True)
    print("PREFILL_SHORT_AB_DONE", flush = True)

if __name__ == "__main__":
    main()
