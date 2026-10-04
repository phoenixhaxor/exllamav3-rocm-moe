# KV streaming end to end: greedy jobs (MTP on) through the generator with EXL3_KV_STREAM off and on,
# token ids saved per run and compared. Streaming moves only where the K/V live, so the tokens must be
# identical. Jobs: a long prompt (sparse prefill chunks + decode), then a second job sharing its prefix
# (prompt-cache page reuse) and a third unrelated one (pages recycled from the first jobs)
#   python kv_stream_check.py -m MODEL --stream 0 --out a.json ; ... --stream 1 --out b.json --ref a.json
import os, sys, time, json, argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("--stream", type = int, default = 0)
    ap.add_argument("--slots", type = int, default = 8192)
    ap.add_argument("--mcs", type = int, default = 444)
    ap.add_argument("--mct", type = int, default = 12)
    ap.add_argument("--cache", type = int, default = 131072)
    ap.add_argument("--len", type = int, default = 40000)
    ap.add_argument("--tokens", type = int, default = 400)
    ap.add_argument("--chunk", type = int, default = 8192)
    ap.add_argument("--out", default = None)
    ap.add_argument("--ref", default = None)
    args = ap.parse_args()
    os.environ["EXL3_KV_STREAM"] = str(args.stream)
    os.environ["EXL3_KV_STREAM_SLOTS"] = str(args.slots)
    os.environ.setdefault("EXL3_KV_STREAM_MIN", "65536")

    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator.sampler import ComboSampler

    def vram():
        f, t = torch.cuda.mem_get_info()
        return (t - f) / 1e9

    def main():
        config = Config.from_directory(args.model)
        config.infer_params.moe_cpu_split = args.mcs
        config.infer_params.moe_cpu_threads = args.mct
        model = Model.from_config(config)
        draft_model = Model.from_config(config, component = "mtp")
        q8 = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
        cache = Cache(model, max_num_tokens = args.cache, max_history = 4, max_batch_size = 1, **q8)
        model.load(progressbar = False)
        draft_cache = Cache(draft_model, max_num_tokens = args.cache, **q8)
        draft_model.load(progressbar = False)
        print(f"loaded, stream {args.stream}, VRAM used {vram():.2f} GB", flush = True)
        tok = Tokenizer.from_config(config)
        gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft_model,
                        draft_cache = draft_cache, num_draft_tokens = 2, max_chunk_size = args.chunk)
        text = open(os.path.expanduser("~/wikitext2_test.txt")).read()
        body = tok.encode(text)[:, :args.len]
        body2 = tok.encode(text[len(text) // 2:])[:, :args.len // 3]
        head = tok.encode("<|im_start|>user\nRead this:\n", encode_special_tokens = True)
        def q(s):
            return tok.encode(f"<|im_end|>\n<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n"
                              "<think>\n\n</think>\n\n", encode_special_tokens = True)
        jobs = [
            ("long", torch.cat([head, body, q("Repeat the text above word for word, from the very beginning.")], 1)),
            ("prefix", torch.cat([head, body, q("Summarize the text above in detail.")], 1)),
            ("other", torch.cat([head, body2, q("List every person named in the text above.")], 1)),
        ]
        res = {}
        for name, ids in jobs:
            job = Job(input_ids = ids, max_new_tokens = args.tokens, stop_conditions = [],
                      sampler = ComboSampler(temperature = 0.0, top_k = 1))
            gen.enqueue(job)
            t0 = time.perf_counter(); t1 = None; out = []
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    t = r.get("token_ids")
                    if t is not None and t.numel():
                        if t1 is None: t1 = time.perf_counter()
                        out += t.flatten().tolist()
            t2 = time.perf_counter()
            res[name] = out
            print(f"{name}: prompt {ids.shape[1]}, TTFT {t1 - t0:.2f} s ({ids.shape[1] / (t1 - t0):.0f} T/s incl. cache), "
                  f"decode {len(out)} tok {len(out) / (t2 - t1):.1f} T/s | {tok.decode(torch.tensor(out[:40]))[:100]!r}",
                  flush = True)
        for m in (model, draft_model):
            for l in (cache if m is model else draft_cache).layers.values():
                kvs = getattr(l, "kv_stream", None)
                if kvs is not None:
                    s = kvs.stats.tolist(); c = kvs.ctl.tolist()
                    print(f"  kv_stream layer: calls {s[2]}, lookups {s[1]}, misses {s[0]} "
                          f"({s[0] / max(s[1], 1) * 100:.2f}%), overflow {c[3]}")
                    break
        if args.stream:
            from exllamav3.cache.kv_stream import kv_verify_totals
            if kv_verify_totals["calls"]:
                print(f"  verify: {kv_verify_totals}")
        if args.out:
            json.dump(res, open(args.out, "w"))
        if args.ref:
            ref = json.load(open(args.ref))
            for name in res:
                a, b = ref[name], res[name]
                same = sum(1 for x, y in zip(a, b) if x == y)
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                print(f"compare {name}: {same}/{min(len(a), len(b))} equal, first diff at {first}")
            ok = all(ref[n] == res[n] for n in res)
            print("KV_STREAM_CHECK", "PASS" if ok else "DIFF")

    main()
