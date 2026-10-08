# Grouped prefill (EXL3_PREFILL_GROUP) through the Generator: three short prompts (--lens) arrive together. Time until
# every job has its first token, group off / on in the order off, on, off, on (fresh prompts each run); then the same
# prompts off, off, on, on for the greedy tokens and the first-token logits (KL, top-1) of on against off, next to the
# off/off and on/on pairs (decode is not bit-deterministic run to run, so those are the noise floor).
#   python prefill_group_check.py -m <model> [--mcs 362] [--lens 1024,2048,3072]
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator import generator as genmod
from exllamav3.generator.sampler import ComboSampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--mcs", type = int, default = 362)
    ap.add_argument("--lens", default = "1024,2048,3072")
    ap.add_argument("--gen", type = int, default = 48)
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]
    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = 12
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 131072, max_batch_size = 5, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
    model.load(progressbar = False, max_chunk_size = 8192)
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 5, max_chunk_size = 8192)
    text = tok.encode(open(os.path.expanduser("~/wikitext2_test.txt")).read())
    head = tok.encode("<|im_start|>user\n", encode_special_tokens = True)
    tail = tok.encode("\n\nContinue this text.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      encode_special_tokens = True)
    def prompt(off, n):
        return torch.cat([head, text[:, off:off + n], tail], dim = -1)

    def forget():
        gen.pagetable.reset_page_table()
        if getattr(gen, "recurrent_cache", None) is not None:
            gen.recurrent_cache.clear()

    # 1. Grouped prefill
    off = 2000
    runs = {}
    for name, on in (("off1", False), ("on1", True), ("off2", False), ("on2", True)):
        genmod._pf_group_on = on
        forget()
        jobs = []
        for i, n in enumerate(lens):
            ids = prompt(off, n)
            off += n + 100
            j = Job(input_ids = ids, max_new_tokens = args.gen, sampler = ComboSampler(top_k = 1),
                    stop_conditions = [], identifier = i)
            gen.enqueue(j); jobs.append(j)
        toks = {i: [] for i in range(len(lens))}
        first = {}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and r.get("token_ids") is not None and r["token_ids"].numel():
                    toks[r["identifier"]] += r["token_ids"].view(-1).tolist()
                    first.setdefault(r["identifier"], time.perf_counter() - t0)
        runs[name] = toks
        print(f"{name}: prompts {lens}, all first tokens after {max(first.values()):.2f} s "
              f"(each {', '.join(f'{first[i]:.2f}' for i in range(len(lens)))})", flush = True)
    # the same prompts again, for the token comparison: off / off / on with identical inputs. The first decode forward
    # (all jobs together, the same batch in every mode) gives each job's first-token logits after its prefill
    cmp = {}
    first_logits = {}
    orig_forward = model.forward
    def fwd(input_ids, params = None):
        y = orig_forward(input_ids, params)
        if rec[0] is not None and input_ids.shape[0] == len(lens):
            first_logits[rec[0]] = y[:, -1, :].float().cpu()
            rec[0] = None
        return y
    rec = [None]
    model.forward = fwd
    for name, on in (("A_off", False), ("B_off", False), ("C_on", True), ("D_on", True)):
        genmod._pf_group_on = on
        rec[0] = name
        forget()
        o = 40000
        for i, n in enumerate(lens):
            gen.enqueue(Job(input_ids = prompt(o, n), max_new_tokens = args.gen, sampler = ComboSampler(top_k = 1),
                            stop_conditions = [], identifier = i))
            o += n + 100
        toks = {i: [] for i in range(len(lens))}
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                    toks[r["identifier"]] += r["token_ids"].view(-1).tolist()
        cmp[name] = toks
    def same(a, b):
        out = []
        for i in a:
            k = 0
            while k < min(len(a[i]), len(b[i])) and a[i][k] == b[i][k]: k += 1
            out.append(k)
        return out
    model.forward = orig_forward
    print(f"identical greedy tokens per job (of {args.gen}): off/off {same(cmp['A_off'], cmp['B_off'])}, "
          f"on/on {same(cmp['C_on'], cmp['D_on'])}, on/off {same(cmp['C_on'], cmp['A_off'])}, "
          f"{same(cmp['C_on'], cmp['B_off'])}, {same(cmp['D_on'], cmp['A_off'])}", flush = True)
    def logit_cmp(a, b):
        la, lb = first_logits[a], first_logits[b]
        pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
        kl = (pa.exp() * (pa - pb)).sum(-1)
        top = (la.argmax(-1) == lb.argmax(-1)).tolist()
        mx = (la - lb).abs().max(-1).values
        return (f"KL {', '.join(f'{x:.5f}' for x in kl.tolist())}; top-1 equal {top}; "
                f"max logit diff {', '.join(f'{x:.3f}' for x in mx.tolist())}")
    print("first-token logits per job:", flush = True)
    print("  off/off:", logit_cmp("A_off", "B_off"), flush = True)
    print("  on/on:  ", logit_cmp("C_on", "D_on"), flush = True)
    print("  on/off: ", logit_cmp("C_on", "A_off"), flush = True)
    print("  on/off: ", logit_cmp("D_on", "B_off"), flush = True)
    print("  on, job 0:", tok.decode(torch.tensor(cmp["C_on"][0][:40])).replace("\n", " | "), flush = True)
    genmod._pf_group_on = False

    print("PREFILL_GROUP_CHECK_DONE", flush = True)

if __name__ == "__main__":
    main()
