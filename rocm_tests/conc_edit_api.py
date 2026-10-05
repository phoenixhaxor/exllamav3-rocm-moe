#!/usr/bin/env python3
# Code edits at once over the API: N requests (argv[3]) that each rename a method in a different ~140-line block of
# this repository's code and output the whole block, alone and then together. Copy-heavy answers, so prompt lookup
# carries most of the speed; reports each request's decode rate and the total (EXL3_MTP_LOOKUP_BATCH).
#   python conc_edit_api.py [url] [tag] [n]
import json, os, sys, threading, time, urllib.request
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8098"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
NCONC = int(sys.argv[3]) if len(sys.argv) > 3 else 2
SRC = [("exllamav3/generator/job.py", 560, "receive_logits", "take_logits"),
       ("exllamav3/generator/pagetable.py", 300, "allocate_pages", "claim_pages"),
       ("exllamav3/cache/recurrent.py", 40, "get_stashed", "fetch_stashed"),
       ("exllamav3/generator/generator.py", 1110, "iterate_gen", "run_generation"),
       ("exllamav3/generator/async_generator.py", 20, "deliver_results", "dispatch_results")]

def prompt(i):
    f, line, old, new = SRC[i]
    code = "".join(open(os.path.join(REPO, f)).readlines()[line: line + 140])
    return (f"```python\n{code}\n```\nRename every occurrence of `{old}` to `{new}` in the code above (add a usage "
            f"of `{old}` in a comment at the top first if it does not occur) and output the complete code block, "
            f"unchanged otherwise. No explanation.")

def run(i, out):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt(i)}], "max_tokens": 1800, "stream": True,
            "temperature": 0.0, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = None
    with urllib.request.urlopen(req, timeout = 3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                if c.get("delta", {}).get("content") and tf is None: tf = time.time()
    t1 = time.time()
    out[i] = (usage["completion_tokens"], usage["completion_tokens"] / (t1 - tf), t1)

def scenario(ids, label):
    out = {}; t0 = time.time()
    th = [threading.Thread(target = run, args = (i, out)) for i in ids]
    for t in th: t.start(); time.sleep(0.2)
    for t in th: t.join()
    wall = max(v[2] for v in out.values()) - t0
    each = ", ".join(f"{out[i][1]:.1f}" for i in ids)
    print(TAG, f"[{label}] each {each} tok/s; total {sum(v[0] for v in out.values()) / wall:.1f} tok/s", flush = True)

ids = list(range(NCONC))
run(0, {})   # warm-up
for rep in range(2):
    for i in ids: scenario([i], f"edit {i} alone")
    scenario(ids, f"{NCONC} edits at once")
