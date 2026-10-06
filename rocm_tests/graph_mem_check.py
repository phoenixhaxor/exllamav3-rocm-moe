# VRAM growth from graph argument updates (issue #4): one greedy job decodes with the decode graphs on (run with
# EXL3_NOGRAPH unset or without "attn"), printing this process's DRM memory (drm-memory-vram + gtt from
# /proc/self/fdinfo, which also counts the ROCm runtime's own allocations, unlike torch's counters) every N tokens.
# Before the fix it grows ~10 KiB per token; with Graph re-instantiation (EXL3_GRAPH_REINSTANTIATE, default 2048
# updates) it stays flat after warm-up.
#   python graph_mem_check.py -m <model> [--mcs 362] [--tokens 3000]
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler

def drm_kib():
    total = 0
    for fd in os.listdir("/proc/self/fdinfo"):
        try:
            lines = open(f"/proc/self/fdinfo/{fd}").read().splitlines()
        except OSError:
            continue
        if not any(l.startswith("drm-driver:") and "amdgpu" in l for l in lines):
            continue
        for l in lines:
            if l.startswith("drm-memory-vram:") or l.startswith("drm-memory-gtt:"):
                total += int(l.split()[1])
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 362)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--tokens", type = int, default = 3000)
    ap.add_argument("--every", type = int, default = 500)
    ap.add_argument("--context", type = int, default = 4000,
                    help = "prompt tokens of real text ahead of the request, so one-time allocations of longer-context "
                           "attention regimes happen before measuring")
    args = ap.parse_args()
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 32768, max_batch_size = 1, layer_type = CacheLayer_quant,
                  k_bits = 8, v_bits = 8)
    model.load(progressbar = False)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 1)
    print(f"EXL3_NOGRAPH={os.environ.get('EXL3_NOGRAPH', '')} "
          f"EXL3_GRAPH_REINSTANTIATE={os.environ.get('EXL3_GRAPH_REINSTANTIATE', 'default')}", flush = True)
    text_in = open(os.path.expanduser("~/wikitext2_test.txt")).read()[: args.context * 4]
    ctx_ids = tok.encode(text_in)[:, : args.context]
    ctx = tok.decode(ctx_ids[0])
    ids = tok.encode("<|im_start|>user\n" + ctx + "\n\nNow write a very long, detailed story about a lighthouse keeper "
                     "and the sea, chapter by chapter.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                     encode_special_tokens = True)
    gen.enqueue(Job(input_ids = ids, max_new_tokens = args.tokens, sampler = ComboSampler(top_k = 1),
                    stop_conditions = []))
    n = 0; m0 = None; t0 = None; text = ""
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") != "streaming": continue
            text += r.get("text", "")
            k = r["token_ids"].numel() if r.get("token_ids") is not None else 0
            n += k
            if m0 is None and n >= 200:
                torch.cuda.synchronize(); m0 = drm_kib(); r0 = torch.cuda.memory_reserved(); t0 = time.perf_counter(); n0 = n
            elif m0 is not None and (n - n0) // args.every > (n - n0 - k) // args.every:
                torch.cuda.synchronize()
                dm = drm_kib() - m0
                dr = (torch.cuda.memory_reserved() - r0) // 1024
                rate = (n - n0) / (time.perf_counter() - t0)
                print(f"  {n - n0:5d} tokens after warm-up: DRM memory {dm:+7d} KiB "
                      f"({dm * 1024 / (n - n0):6.0f} B/token), outside torch's allocator {dm - dr:+7d} KiB, "
                      f"{rate:.1f} tok/s", flush = True)
    print("  text tail:", text[-160:].replace("\n", " | "), flush = True)
    print("GRAPH_MEM_CHECK_DONE", flush = True)

if __name__ == "__main__":
    main()
