# Paired variant of iqbench.py for the tail-skip test: every question is asked with each tail rank in turn (same seed),
# the rank is switched through a file the test server polls. Original: rocm_tests/iqbench.py.
# Capability benchmark through an OpenAI-compatible endpoint, identical settings for every model:
# MMLU-Pro (10-way multiple choice), MATH-500 level 4-5 (boxed answer), HumanEval (executed tests).
# Thinking on, temperature 0.6 / top-p 0.95 / top-k 20, per-question seed; a client-side token budget stops
# runaway generations (the servers ignore max_tokens), counted as unanswered.
# Usage: iqbench.py URL NAME [--tasks mmlu,math,code] [--n-mmlu N] [--n-math N] [--n-code N] [--budget TOK]
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("name")
ap.add_argument("--tasks", default = "mmlu,math,code")
ap.add_argument("--n-mmlu", type = int, default = 80)
ap.add_argument("--n-math", type = int, default = 30)
ap.add_argument("--n-code", type = int, default = 60)
ap.add_argument("--budget", type = int, default = 10000, help = "max streamed chunks (~tokens) per answer")
ap.add_argument("--data", default = os.path.expanduser("~/iqbench"))
ap.add_argument("--ranks", default = "0,8", help = "tail ranks compared, asked alternately per question")
ap.add_argument("--rankfile", default = "./tail_rank", help = "file the test server polls for the tail rank (EXL3_MOE_TAIL_RANKFILE)")
args = ap.parse_args()
ranks = [int(x) for x in args.ranks.split(",")]
out_path = os.path.join(args.data, f"result_{args.name}.jsonl")
done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        r = json.loads(line); done.add((r["task"], r["idx"], r["rank"]))

def ask(prompt, seed):
    body = {"model": "x", "stream": True, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": seed}
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); content = []; reasoning = []; n = 0; cut = False
    with urllib.request.urlopen(req, timeout = 3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: d = json.loads(data)["choices"][0].get("delta", {})
            except Exception: continue
            if d.get("content"): content.append(d["content"]); n += 1
            if d.get("reasoning_content"): reasoning.append(d["reasoning_content"]); n += 1
            if n > args.budget: cut = True; break
    return "".join(content), "".join(reasoning), n, time.time() - t0, cut

# ---- graders
def mmlu_pred(text):
    for pat in (r"answer is \(?([A-J])\)?", r"\\boxed\{\(?([A-J])\)?\}", r"[Aa]nswer:\s*\(?([A-J])\)?"):
        m = re.findall(pat, text)
        if m: return m[-1]
    m = re.findall(r"\b([A-J])\b", text[-200:])
    return m[-1] if m else None

def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0: return None
    j = text.find("{", i)
    if j < 0: return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{": depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0: return text[j + 1:k]
    return None

def norm(s):
    s = s.strip().strip("$").replace(" ", "").replace("\\!", "").replace("\\,", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s).replace("\\mbox", "")
    s = s.rstrip(".").replace("dollars", "").replace("\\$", "")
    if s.startswith("x=") or s.startswith("y="): s = s[2:]
    return s

def math_ok(pred, gold):
    if pred is None: return False
    a, b = norm(pred), norm(gold)
    if a == b: return True
    try: return abs(float(a) - float(b)) < 1e-6
    except ValueError: pass
    try:
        from sympy.parsing.latex import parse_latex
        return bool((parse_latex(a) - parse_latex(b)).simplify() == 0)
    except Exception: return False

def code_ok(text, prob):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = blocks[-1] if blocks else text
    if f"def {prob['entry_point']}" not in code:
        code = prob["prompt"] + code
    else:
        imports = "\n".join(l for l in prob["prompt"].splitlines() if l.startswith(("import ", "from ")))
        code = imports + "\n" + code
    prog = code + "\n\n" + prob["test"] + f"\n\ncheck({prob['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix = ".py", delete = False) as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output = True, timeout = 20)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)

tasks = []
if "mmlu" in args.tasks:
    for i, r in enumerate(json.load(open(os.path.join(args.data, "mmlu_pro.json")))[:args.n_mmlu]):
        opts = "\n".join(f"{chr(65 + k)}. {o}" for k, o in enumerate(r["options"]))
        p = (f"Answer the following multiple choice question. Think step by step, then finish with "
             f"'The answer is (X)' where X is the letter of the correct option.\n\n{r['question']}\n\n{opts}")
        tasks.append(("mmlu", i, p, r))
if "math" in args.tasks:
    for i, r in enumerate(json.load(open(os.path.join(args.data, "math500.json")))[:args.n_math]):
        tasks.append(("math", i, r["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.", r))
if "code" in args.tasks:
    for i, r in enumerate(json.load(open(os.path.join(args.data, "humaneval.json")))[:args.n_code]):
        p = ("Complete the following Python function. Return the complete function (with any imports it needs) "
             "in a single ```python code block.\n\n```python\n" + r["prompt"] + "```")
        tasks.append(("code", i, p, r))

def set_rank(r):
    with open(args.rankfile, "w") as f: f.write(str(r))
    time.sleep(1.5)   # the server polls the file every few decode steps

with open(out_path, "a") as out:
    for task, idx, prompt, r in tasks:
        # alternate which rank goes first so drift in the server (caches, placement) does not favour one rank
        order = ranks if idx % 2 == 0 else ranks[::-1]
        for rk in order:
            if (task, idx, rk) in done: continue
            set_rank(rk)
            content, reasoning, n, dt, cut = ask(prompt, 1000 + idx)
            if task == "mmlu": pred = mmlu_pred(content); ok = pred == r["answer"]
            elif task == "math": pred = last_boxed(content); ok = math_ok(pred, r["answer"])
            else: pred = None; ok = code_ok(content, r)
            rec = {"task": task, "idx": idx, "rank": rk, "ok": bool(ok), "pred": pred, "tokens": n, "sec": round(dt, 1), "cut": cut}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"{args.name} rank {rk} {task} {idx}: {'OK ' if ok else 'BAD'} tok {n} {dt:.0f}s{' CUT' if cut else ''}", flush = True)
set_rank(0)

res = [json.loads(l) for l in open(out_path)]
print("--- paired result (same questions, same seeds) ---")
for t in ("mmlu", "math", "code"):
    pairs = {}
    for r in res:
        if r["task"] == t: pairs.setdefault(r["idx"], {})[r["rank"]] = r
    full = [p for p in pairs.values() if len(p) == len(ranks)]
    if not full: continue
    base = ranks[0]
    line = f"{args.name} {t}: {len(full)} paired questions"
    for rk in ranks:
        ok = sum(p[rk]["ok"] for p in full)
        tok = sum(p[rk]["tokens"] for p in full) / len(full)
        line += f" | rank {rk}: {ok}/{len(full)} = {100 * ok / len(full):.1f}% avg tokens {tok:.0f}"
    for rk in ranks[1:]:
        b = sum(p[base]["ok"] and not p[rk]["ok"] for p in full)
        c = sum((not p[base]["ok"]) and p[rk]["ok"] for p in full)
        line += f" | rank {rk} vs {base}: base-only right {b}, rank-only right {c}"
    print(line)
