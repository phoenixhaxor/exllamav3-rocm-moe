#!/usr/bin/env python3
# A conversation that grows to ~120K tokens over 30 turns (a ~60K-token document, then ~2K tokens of new text per
# turn), then the same history with one early turn rewritten, as an agent client does when it prunes old tool output.
# Prints each request's prompt size and first-token time; the last one shows how much of the changed history had to
# be read again (the server's EXL3_PREFIX_LOG line has the reused prefix).
#   python prune_api.py [url] [tag]
import json, os, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8098"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
text = open(os.path.expanduser("~/wikitext2_test.txt")).read()
CH = 3.6                                    # characters per token, roughly
doc = text[: int(60000 * CH)]
pos = len(doc)
turns = []
for i in range(30):
    turns.append(text[pos: pos + int(2000 * CH)])
    pos += int(2000 * CH)

def ask(messages, label):
    body = {"model": "x", "messages": messages, "max_tokens": 4, "stream": True, "temperature": 0.0,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
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
    tf = tf or time.time()
    print(TAG, f"{label}: prompt {usage['prompt_tokens']}, first token {tf - t0:.1f}s", flush = True)
    return tf - t0

def history(rewrite = None):
    m = [{"role": "user", "content": "Read this document.\n\n" + doc + "\n\nReply with OK."},
         {"role": "assistant", "content": "OK."}]
    for i, t in enumerate(turns):
        if i == rewrite:
            t = "[earlier text removed]"
        m += [{"role": "user", "content": f"Part {i + 1}:\n\n{t}\n\nReply with OK."},
              {"role": "assistant", "content": "OK."}]
    return m

ask(history()[:1], "turn 0 (document)")
for n in range(1, len(turns) + 1):
    ask(history()[: 2 * n + 1], f"turn {n}")
full = history()
ask(full[:-1] + [{"role": "user", "content": "How many parts were there? Reply with a number."}], "next turn, unchanged")
ask(history(rewrite = 2)[:-1] + [{"role": "user", "content": "How many parts were there? Reply with a number."}],
    "next turn, part 3 rewritten")
