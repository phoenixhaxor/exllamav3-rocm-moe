#!/usr/bin/env python3
"""codecheck.py URL LABEL  -- small local code-generation check (no downloads): 23 function tasks with assertions.
Each task is generated at temperature 0 (greedy), the code block is extracted and run in a subprocess with a time limit.
Used to compare a tail-skip rank (or any other setting) against the baseline on the same prompts.
Writes codecheck-LABEL.json into $OUT_DIR (default: current directory) and prints the pass count plus, per task, pass/fail and the generation speed."""
import json, re, subprocess, sys, tempfile, time, urllib.request, os

URL, LABEL = sys.argv[1], sys.argv[2]

TASKS = [
 ("roman", "to_roman(n: int) -> str converting 1..3999 to a Roman numeral",
  "assert to_roman(4)=='IV'; assert to_roman(1994)=='MCMXCIV'; assert to_roman(3999)=='MMMCMXCIX'; assert to_roman(58)=='LVIII'"),
 ("rle", "rle_encode(s: str) -> str run-length encoding like 'aaabcc' -> 'a3b1c2' and rle_decode(s: str) -> str as its inverse",
  "assert rle_encode('aaabcc')=='a3b1c2'; assert rle_decode('a3b1c2')=='aaabcc'; assert rle_encode('')==''; assert rle_decode(rle_encode('xxxxxxxxxxxy'))=='xxxxxxxxxxxy'"),
 ("brackets", "is_balanced(s: str) -> bool checking (), [] and {} brackets, ignoring other characters",
  "assert is_balanced('{[()]}'); assert not is_balanced('([)]'); assert is_balanced('a(b)c'); assert not is_balanced('((')"),
 ("bsearch", "binary_search(a: list[int], x: int) -> int returning the index of x in the sorted list or -1",
  "assert binary_search([1,3,5,7,9],7)==3; assert binary_search([1,3,5],4)==-1; assert binary_search([],1)==-1; assert binary_search([2],2)==0"),
 ("intervals", "merge_intervals(iv: list[tuple[int,int]]) -> list[tuple[int,int]] merging overlapping or touching intervals, result sorted",
  "assert merge_intervals([(1,3),(2,6),(8,10),(15,18)])==[(1,6),(8,10),(15,18)]; assert merge_intervals([(1,4),(4,5)])==[(1,5)]; assert merge_intervals([])==[]"),
 ("anagrams", "group_anagrams(words: list[str]) -> list[list[str]] grouping words that are anagrams; groups sorted by first element, words inside a group in input order",
  "r=group_anagrams(['eat','tea','tan','ate','nat','bat']); assert sorted(map(sorted,r))==sorted([sorted(['eat','tea','ate']),sorted(['tan','nat']),['bat']])"),
 ("lev", "levenshtein(a: str, b: str) -> int edit distance with insert, delete, substitute",
  "assert levenshtein('kitten','sitting')==3; assert levenshtein('','abc')==3; assert levenshtein('abc','abc')==0; assert levenshtein('flaw','lawn')==2"),
 ("toposort", "topo_sort(graph: dict[str, list[str]]) -> list[str] returning nodes in an order where each node comes before the nodes it points to; raise ValueError on a cycle",
  "g={'a':['b','c'],'b':['d'],'c':['d'],'d':[]}; r=topo_sort(g); assert r.index('a')<r.index('b')<r.index('d') and r.index('c')<r.index('d')\ntry:\n    topo_sort({'a':['b'],'b':['a']}); raise SystemExit(1)\nexcept ValueError:\n    pass"),
 ("baseconv", "to_base(n: int, base: int) -> str converting a non-negative integer to the given base (2..36) with lowercase digits",
  "assert to_base(255,16)=='ff'; assert to_base(0,2)=='0'; assert to_base(10,2)=='1010'; assert to_base(35,36)=='z'"),
 ("wordfreq", "top_words(text: str, k: int) -> list[tuple[str,int]] the k most frequent words (lowercased, letters only), ties broken alphabetically",
  "assert top_words('the cat and the hat and the bat',2)==[('the',3),('and',2)]; assert top_words('',3)==[]; assert top_words('a b a B',1)==[('a',2)]"),
 ("flatten", "flatten(x) -> list flattening arbitrarily nested lists and tuples of ints into one list",
  "assert flatten([1,[2,[3,(4,5)]],6])==[1,2,3,4,5,6]; assert flatten([])==[]; assert flatten([[],[[]]])==[]"),
 ("gcdlcm", "gcd_lcm(a: int, b: int) -> tuple[int,int] returning (gcd, lcm) for positive integers without using math.gcd",
  "assert gcd_lcm(12,18)==(6,36); assert gcd_lcm(7,13)==(1,91); assert gcd_lcm(5,5)==(5,5)"),
 ("validip", "is_valid_ipv4(s: str) -> bool strict dotted-quad check, no leading zeros, each part 0..255",
  "assert is_valid_ipv4('192.168.0.1'); assert not is_valid_ipv4('256.1.1.1'); assert not is_valid_ipv4('01.2.3.4'); assert not is_valid_ipv4('1.2.3'); assert is_valid_ipv4('0.0.0.0')"),
 ("sieve", "primes_upto(n: int) -> list[int] all primes <= n using a sieve",
  "assert primes_upto(30)==[2,3,5,7,11,13,17,19,23,29]; assert primes_upto(1)==[]; assert primes_upto(2)==[2]"),
 ("matrix", "spiral_order(m: list[list[int]]) -> list[int] elements of the matrix in clockwise spiral order",
  "assert spiral_order([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5]; assert spiral_order([[1,2],[3,4],[5,6]])==[1,2,4,6,5,3]; assert spiral_order([])==[]"),
 ("lcs", "lcs_length(a: str, b: str) -> int length of the longest common subsequence",
  "assert lcs_length('ABCBDAB','BDCABA')==4; assert lcs_length('','a')==0; assert lcs_length('abc','abc')==3"),
 ("palin", "longest_palindrome(s: str) -> str the longest palindromic substring (first one if several)",
  "assert longest_palindrome('babad') in ('bab','aba'); assert longest_palindrome('cbbd')=='bb'; assert longest_palindrome('a')=='a'"),
 ("twosum", "two_sum(nums: list[int], target: int) -> tuple[int,int] indices (i<j) of two numbers summing to target, or None",
  "assert two_sum([2,7,11,15],9)==(0,1); assert two_sum([3,3],6)==(0,1); assert two_sum([1,2],10) is None"),
 ("lru", "class LRUCache with __init__(self, capacity: int), get(key) -> value or -1, put(key, value); evicts the least recently used item",
  "c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1; assert c.get(3)==3; assert c.get(4)==4"),
 ("minstack", "class MinStack with push(x), pop(), top(), get_min() all O(1)",
  "s=MinStack(); s.push(3); s.push(1); s.push(2); assert s.get_min()==1; s.pop(); assert s.top()==1; s.pop(); assert s.get_min()==3"),
 ("camel", "snake_to_camel(s: str) -> str converting snake_case to lowerCamelCase and camel_to_snake(s: str) -> str as the inverse for simple words",
  "assert snake_to_camel('hello_world_foo')=='helloWorldFoo'; assert camel_to_snake('helloWorldFoo')=='hello_world_foo'; assert snake_to_camel('a')=='a'"),
 ("dedupe", "dedupe_keep_order(xs: list) -> list removing duplicates but keeping the first occurrence order; elements are hashable",
  "assert dedupe_keep_order([3,1,3,2,1])==[3,1,2]; assert dedupe_keep_order([])==[]; assert dedupe_keep_order(['a','a'])==['a']"),
 ("pascal", "pascal_row(n: int) -> list[int] row n (0-based) of Pascal's triangle",
  "assert pascal_row(0)==[1]; assert pascal_row(4)==[1,4,6,4,1]; assert pascal_row(1)==[1,1]"),
]

def gen(prompt):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 900,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(URL.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout = 600) as r:
        d = json.load(r)
    dt = time.time() - t0
    return d["choices"][0]["message"]["content"], (d.get("usage") or {}).get("completion_tokens", 0) or max(1, len(d["choices"][0]["message"]["content"]) // 3), dt

def extract(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[0] if m else text

res = []; passed = 0
for name, spec, tests in TASKS:
    prompt = f"Write a Python function or class as specified. Output only the code in one code block, no explanation.\n\nSpec: {spec}"
    text, ntok, dt = gen(prompt)
    code = extract(text)
    with tempfile.NamedTemporaryFile("w", suffix = ".py", delete = False) as f:
        f.write(code + "\n\n" + tests + "\nprint('PASS')\n")
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output = True, text = True, timeout = 10)
        ok = p.returncode == 0 and "PASS" in p.stdout
        err = "" if ok else (p.stderr.strip().splitlines() or ["?"])[-1][:120]
    except subprocess.TimeoutExpired:
        ok, err = False, "timeout"
    os.unlink(path)
    passed += ok
    res.append(dict(task = name, ok = ok, err = err, tokens = ntok, secs = round(dt, 1), code = code))
    print(f"{LABEL} {name:10s} {'ok  ' if ok else 'FAIL'} {ntok:4d} tok {ntok / max(dt, 1e-6):5.1f} tok/s {err}", flush = True)

print(f"{LABEL}: {passed}/{len(TASKS)} passed")
json.dump(res, open(os.path.join(os.environ.get("OUT_DIR", "."), f"codecheck-{LABEL}.json"), "w"), indent = 1)
