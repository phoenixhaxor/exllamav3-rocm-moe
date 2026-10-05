# Through the API: a client disconnects mid-stream (alone, and beside a running request), then a normal request must still complete
import json, sys, threading, time, urllib.request
URL = sys.argv[1]
def stream(prompt, n, stop_after = None):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    got = 0
    r = urllib.request.urlopen(req, timeout = 600)
    for line in r:
        if line.startswith(b"data:") and b"content" in line:
            got += 1
            if stop_after and got >= stop_after:
                r.close(); return f"closed after {got}"
    return f"complete, {got} chunks"
print("cancel alone:", stream("Write a long story about a lighthouse.", 600, 20), flush = True)
res = {}
t = threading.Thread(target = lambda: res.update(a = stream("Write a long essay about rivers.", 400)))
t.start(); time.sleep(3)
print("cancel beside:", stream("Write a long story about a forest.", 600, 20), flush = True)
t.join(); print("beside request:", res["a"], flush = True)
print("after:", stream("Say hello in five languages.", 60), flush = True)
print("CANCEL_TEST_DONE")
