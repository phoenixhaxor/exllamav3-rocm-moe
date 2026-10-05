# Where a batched decode step goes with CPU-resident experts: B generator jobs decoding together (no drafts), per
# batch size the step wall time, the host enqueue time of model.forward, and the GPU-stream time per module (CUDA
# events around each hooked call; mlp.cpu_collect is the stall where the GPU waits for the CPU worker). Run with
# EXL3_MOE_HANDOFF_PROF=1 for the worker's own compute/idle split per job (printed by the worker).
#   python batch_prof.py -m <model> [--mcs 362] [--bsz 1,2,3,5] [--steps 40]
import os, sys, time, argparse, collections, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
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
    ap.add_argument("--bsz", default = "1,2,3,5")
    ap.add_argument("--steps", type = int, default = 40)
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

    ev = []
    on = [False]
    def wrap(obj, mn, name):
        f = getattr(obj, mn)
        def w(*a, **kw):
            if not on[0]: return f(*a, **kw)
            e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
            e0.record()
            y = f(*a, **kw)
            e1.record(); ev.append((name, e0, e1))
            return y
        setattr(obj, mn, w)
    for m in model.modules:
        if type(m).__name__ != "TransformerBlock":
            wrap(m, "forward", type(m).__name__)
            continue
        wrap(m.attn, "forward", f"attn[{type(m.attn).__name__}]")
        mm = m.mlp
        for mn, name in (("routing_fn", "mlp.routing"), ("cpu_split_submit", "mlp.cpu_submit"),
                         ("_rdna3_moe_forward", "mlp.gpu_experts"), ("cpu_split_combine", "mlp.cpu_collect")):
            if hasattr(mm, mn): wrap(mm, mn, name)
        if getattr(mm, "shared_experts", None) is not None:
            wrap(mm.shared_experts, "forward", "mlp.shared")
        wrap(mm, "forward", "mlp(total)")
        for an in ("attn_hc", "mlp_hc"):
            if hasattr(m, an):
                for mn in ("mix", "apply_"):
                    if hasattr(getattr(m, an), mn): wrap(getattr(m, an), mn, f"{an}.{mn}")

    fw = []
    orig = model.forward
    def fwd(*a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y = orig(*a, **kw)
        t1 = time.perf_counter(); torch.cuda.synchronize(); t2 = time.perf_counter()
        fw.append((t1 - t0, t2 - t0))
        return y
    model.forward = fwd

    for B in [int(x) for x in args.bsz.split(",")]:
        gen.clear_queue()
        for i in range(B):
            ids = tok.encode(f"<|im_start|>user\n{PROMPTS[i]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                             encode_special_tokens = True)
            gen.enqueue(Job(input_ids = ids, max_new_tokens = 2000, sampler = ComboSampler(temperature = 0.6),
                            stop_conditions = [], identifier = i))
        for _ in range(30): gen.iterate()
        for hooks in (False, True):
            fw.clear(); ev.clear(); on[0] = hooks
            print(f"=== B {B} hooks {int(hooks)}", flush = True)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(args.steps): gen.iterate()
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            on[0] = False
            n = len(fw)
            enq = sorted(x[0] for x in fw)[n // 2] * 1000
            full = sorted(x[1] for x in fw)[n // 2] * 1000
            print(f"B {B} hooks {int(hooks)}: iterate {dt / args.steps * 1000:.2f} ms, forward median {full:.2f} ms "
                  f"(host enqueue {enq:.2f} ms), {B * args.steps / dt:.1f} tok/s total", flush = True)
            if hooks:
                agg = collections.defaultdict(lambda: [0.0, 0])
                for name, e0, e1 in ev:
                    a = agg[name]; a[0] += e0.elapsed_time(e1); a[1] += 1
                for name, (g, c) in sorted(agg.items(), key = lambda kv: -kv[1][0]):
                    print(f"  {name:24s} calls/step {c / n:5.1f}  gpu {g / n:7.3f} ms/step", flush = True)
    gen.clear_queue()
    print("BATCH_PROF_DONE", flush = True)

if __name__ == "__main__":
    main()
