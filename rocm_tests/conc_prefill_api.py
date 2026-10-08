#!/usr/bin/env python3
# Short prompts arriving together over the API, as agent turns do: N requests (argv[3]) with ~L-token (argv[4])
# wikitext prompts sent at once, first-token time each and until all have one, over several rounds. Every text is
# read once first with another opening line: the prompt cache misses each round (the first tokens differ), while
# the PLE n-gram rows the text needs are in the page cache, so cold disk reads do not swamp the comparison.
#   python conc_prefill_api.py [url] [tag] [n] [tokens] [rounds] [text offset]
import json, os, sys, threading, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8098"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
L = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
text = open(os.path.expanduser("~/wikitext2_test.txt")).read()
CH = 3.6

R = int(sys.argv[5]) if len(sys.argv) > 5 else 4
OFF = int(sys.argv[6]) if len(sys.argv) > 6 else 300000

def run(i, off, out, t0, head):
    body = {"model": "x", "max_tokens": 8, "stream": True, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": head + "\n\n" + text[off: off + int(L * CH)] +
                          "\n\nSummarize this in one line."}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = 600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            for c in d.get("choices", []):
                if c.get("delta", {}).get("content") and i not in out:
                    out[i] = time.time() - t0

offs = [OFF + i * int(L * CH + 500) for i in range(N)]
stamp = int(time.time())
for i in range(N):
    run(i, offs[i], {}, time.time(), f"Warm-up {stamp}.")
for rnd in range(R):
    out = {}; t0 = time.time()
    th = [threading.Thread(target = run, args = (i, offs[i], out, t0, f"Round {stamp}-{rnd}.")) for i in range(N)]
    for t in th: t.start()
    for t in th: t.join()
    print(TAG, f"[{N} x ~{L} tokens at once, round {rnd}] first tokens " +
          ", ".join(f"{out[i]:.1f}" for i in sorted(out)) + f" s; all after {max(out.values()):.1f} s", flush = True)
