#!/usr/bin/env python3
"""degprobe.py URL LABEL -- degeneration probe: the 5 long conc_test prompts, 500 tokens each at temperature 0.
Reports for each output the repeated-4-gram share and the zlib compression ratio (high values = looping/garbage text),
and saves the texts so two settings can be compared (shared prefix length in characters against the label 'tail0')."""
import json, sys, zlib, urllib.request, os, re

URL, LABEL = sys.argv[1], sys.argv[2]
P = {
 "code": "Write a complete Python module implementing an LRU cache with TTL expiry, thread safety, statistics counters and a decorator interface. Include docstrings and a small test suite using unittest. Output only the code.",
 "prose": "Write a long, detailed essay about the history of the printing press and how it changed science, religion and politics in Europe between 1450 and 1650.",
 "explain": "Explain in depth how TCP congestion control works, from slow start and AIMD to CUBIC and BBR, with worked numeric examples.",
 "story": "Write a long short story about a lighthouse keeper on a remote island who finds a message in a bottle, with vivid detail and dialogue.",
 "sql": "Design a PostgreSQL schema for a multi-tenant invoicing system with migrations, indexes and example queries for monthly revenue per tenant. Output SQL with comments.",
}
out = {}
for name, prompt in P.items():
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "min_tokens": 500,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(URL.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = 900) as r:
        t = json.load(r)["choices"][0]["message"]["content"]
    w = re.findall(r"\w+|\S", t)
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    rep = 1 - len(set(grams)) / max(len(grams), 1)
    comp = len(zlib.compress(t.encode())) / max(len(t.encode()), 1)
    out[name] = t
    print(f"{LABEL} {name:8s} len {len(t):5d}  repeated-4gram {rep:5.1%}  zlib {comp:4.2f}", flush = True)
d = os.environ.get("OUT_DIR", ".")
json.dump(out, open(f"{d}/degprobe-{LABEL}.json", "w"), ensure_ascii = False, indent = 1)
base = f"{d}/degprobe-tail0.json"
if LABEL != "tail0" and os.path.exists(base):
    b = json.load(open(base))
    for k in out:
        n = 0
        for x, y in zip(out[k], b[k]):
            if x != y: break
            n += 1
        print(f"{LABEL} {k:8s} identical prefix with tail0: {n} of {len(b[k])} chars")
