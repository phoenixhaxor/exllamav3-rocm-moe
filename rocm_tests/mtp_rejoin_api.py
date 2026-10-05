#!/usr/bin/env python3
# Speculative drafts after batching: a long code answer alone, then the same request beside a short one. Once the
# short one ends the long one decodes alone again; its rate from then on should match the alone run if the MTP
# layer's cache stayed consistent through the batched rounds (EXL3_MTP_BATCH_*). Greedy, thinking off.
#   python mtp_rejoin_api.py [url] [tag]
import json, sys, threading, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8098"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
LONG = ("Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, statistics counters, "
        "a decorator interface and a unittest suite. Output only the code.", 1500)
SHORT = ("Explain in detail how a hash map handles collisions.", 250)

def run(prompt, n, out, key):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "min_tokens": n,
            "stream": True, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    stamps = []
    with urllib.request.urlopen(req, timeout = 3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            for c in d.get("choices", []):
                t = c.get("delta", {}).get("content")
                if t: stamps.append((time.time(), len(t)))
    out[key] = stamps

def rate(stamps, t0):
    s = [x for x in stamps if x[0] >= t0]
    if len(s) < 2: return 0.0
    return sum(c for _, c in s[1:]) / (s[-1][0] - s[0][0])

out = {}
run(*LONG, out, "alone")
a = out["alone"]
t_mid = a[0][0] + (a[-1][0] - a[0][0]) * 0.3
print(TAG, f"alone: {rate(a, t_mid):.0f} chars/s over the last 70%", flush = True)
th = [threading.Thread(target = run, args = (*LONG, out, "long")),
      threading.Thread(target = run, args = (*SHORT, out, "short"))]
for t in th: t.start(); time.sleep(0.3)
for t in th: t.join()
t_end = out["short"][-1][0]
lo = out["long"]
both = [x for x in lo if x[0] < t_end]
print(TAG, f"beside the short one: {rate(both, both[0][0]) if len(both) > 1 else 0:.0f} chars/s; "
      f"alone again after it ended: {rate(lo, t_end + 1.0):.0f} chars/s", flush = True)
