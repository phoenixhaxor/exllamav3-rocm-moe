#!/usr/bin/env python3
# Concurrency A/B over the Tabby API: each request alone, then both at once.
# Streams, so each request reports its own first-token time and decode rate.
import json, sys, threading, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8098"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
N = 800

PROMPTS = {
    "code": "Write a complete Python module implementing an LRU cache with TTL expiry, thread safety, "
            "statistics counters and a decorator interface. Include docstrings and a small test suite "
            "using unittest. Output only the code.",
    "prose": "Write a long, detailed essay about the history of the printing press and how it changed "
             "science, religion and politics in Europe between 1450 and 1650.",
}

def run(name, out, t_start):
    body = {"model": "x", "messages": [{"role": "user", "content": PROMPTS[name]}],
            "max_tokens": N, "min_tokens": N, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; n_chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout = 3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    if t_first is None:
                        t_first = time.time()
                    n_chunks += 1
    t1 = time.time()
    toks = usage["completion_tokens"] if usage else n_chunks
    out[name] = dict(start = t0 - t_start, ttft = t_first - t0, total = t1 - t0, toks = toks,
                     rate = toks / (t1 - t_first), end = t1 - t_start)

def scenario(names):
    out = {}; t_start = time.time()
    th = [threading.Thread(target = run, args = (n, out, t_start)) for n in names]
    for t in th: t.start(); time.sleep(0.2)
    for t in th: t.join()
    wall = time.time() - t_start
    tot = sum(v["toks"] for v in out.values())
    for n in names:
        v = out[n]
        print(f"{TAG} [{'+'.join(names)}] {n:5s}: {v['toks']} tok, ttft {v['ttft']:.2f}s, "
              f"decode {v['rate']:.1f} tok/s, total {v['total']:.1f}s", flush = True)
    print(f"{TAG} [{'+'.join(names)}] aggregate {tot / wall:.1f} tok/s over {wall:.1f}s wall", flush = True)

run("code", {}, time.time())   # warm-up (placement, page cache)
for rep in range(2):
    scenario(["code"]); scenario(["prose"]); scenario(["code", "prose"])
