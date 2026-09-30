# Long-context check through the OpenAI API: a haystack of real text (wikitext + source files) sized to about
# --tokens tokens with three facts ("needles") at 15%, 50% and 85% depth, asked for together in one request; then a
# second request on the same (cached) prefix that writes ~250 words, for decode speed at that context.
# Prints the answers and whether each needle was found; prefill / decode rates come from the server log.
import argparse, glob, json, os, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type = int, required = True)
ap.add_argument("--url", default = "http://127.0.0.1:8098")
ap.add_argument("--chars-per-token", type = float, default = 3.6)
args = ap.parse_args()

src = [open(os.path.expanduser("~/wikitext2_test.txt")).read()]
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for f in sorted(glob.glob(os.path.join(repo, "exllamav3", "**", "*.py"), recursive = True)):
    src.append(open(f, errors = "ignore").read())
text = "\n\n".join(src)
while len(text) < args.tokens * args.chars_per_token:
    text += "\n\n" + text
text = text[: int(args.tokens * args.chars_per_token)]

needles = [
    (0.15, "Note for the reader: the access code for the Aurora vault is 5-8-2-TULIP.", "5-8-2-TULIP"),
    (0.50, "Note for the reader: the ship's cat on the Meridian was named Brisket.", "Brisket"),
    (0.85, "Note for the reader: the budget meeting was moved to Room 4417.", "4417"),
]
for depth, sent, _ in sorted(needles, reverse = True):
    i = int(len(text) * depth)
    i = text.find("\n", i) + 1 or i
    text = text[:i] + sent + "\n" + text[i:]

def ask(question, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": "Read the following material carefully.\n\n" + text + "\n\n" + question}],
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.0, "max_tokens": max_tokens}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(args.url + "/v1/chat/completions", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout = 7200))
    return r["choices"][0]["message"]["content"].strip(), time.time() - t

ans, dt = ask("Questions: (1) What is the access code for the Aurora vault? (2) What was the ship's cat on the Meridian "
              "named? (3) Which room was the budget meeting moved to? Answer the three questions briefly.", 120)
found = [key in ans for _, _, key in needles]
print(f"needles {sum(found)}/3 {found} in {dt:.0f} s | answer: {ans[:300]!r}", flush = True)
ans2, dt2 = ask("Now write about 250 words on what the Python source code in the material is for, in plain prose.", 500)
print(f"summary {len(ans2.split())} words in {dt2:.0f} s | {ans2[:200]!r}", flush = True)
