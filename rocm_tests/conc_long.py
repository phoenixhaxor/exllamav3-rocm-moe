#!/usr/bin/env python3
# Long + short at once: a ~150K-token prompt starts, a short code edit arrives 15 s later. Reports first-token time
# and decode rate of each (streamed), so the short request's wait behind the long prefill is visible.
import json, os, sys, threading, time, urllib.request, glob
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8098"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
src = [open(os.path.expanduser("~/wikitext2_test.txt")).read()]
for f in sorted(glob.glob(os.path.join(REPO, "exllamav3/**/*.py"), recursive = True)):
    src.append(open(f, errors = "ignore").read())
hay = "\n\n".join(src)
while len(hay) < 150000 * 3.6: hay += "\n\n" + hay
hay = hay[: int(150000 * 3.6)]
code = "".join(open(os.path.join(REPO, "exllamav3/generator/job.py")).readlines()[560:700])
reqs = {
    "long": ("Here is a large document:\n\n" + hay + "\n\nSummarize what the Python code in this document does.", 300, 0.0),
    "short": ("```python\n" + code + "\n```\nRename the method receive_logits to take_logits in the code above and "
              "output the complete code block, unchanged otherwise. No explanation.", 1500, 15.0),
}
out = {}
def run(name, t_start):
    prompt, n, delay = reqs[name]
    time.sleep(delay)
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = None
    with urllib.request.urlopen(req, timeout = 3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if (dl.get("content") or dl.get("reasoning_content")) and tf is None: tf = time.time()
    t1 = time.time()
    toks = usage["completion_tokens"]; pt = usage["prompt_tokens"]
    out[name] = f"{name}: prompt {pt}, sent at +{t0 - t_start:.0f}s, first token after {tf - t0:.1f}s, {toks} tok at {toks / (t1 - tf):.1f} tok/s, done at +{t1 - t_start:.0f}s"
t_start = time.time()
th = [threading.Thread(target = run, args = (k, t_start)) for k in reqs]
for t in th: t.start()
for t in th: t.join()
for k in reqs: print(TAG, out[k], flush = True)
