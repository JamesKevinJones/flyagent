"""System 1: typed decisions (Choice / Score / Noul) over the fly's state.

Backends, tried in order until one answers:
  laya       local Laya (`pip install laya`), in this process's CUDA context, hard-capped at 1.5 GB VRAM
  http       TypeSafe Jev (Cloudflare Workers AI `typesafe/jev`) or a self-hosted `laya-serve`
  llm        any OpenAI-compatible chat API with your own key: Gemini, OpenAI, Anthropic, Groq, Ollama, ...
  synthetic  random-weight ModernBERT-large graph (same GPU cost as Laya, no download) + rule answers
  rules      deterministic fallback so the agent never stalls without a decision

Models only run in the table filler (agent_loop.py starts it in its own process), so Laya's Python-heavy
tokenisation never takes the GIL from the 15 ms tick, and a live decision is a dict lookup.
"""
import hashlib
import http.client
import itertools
import json
import math
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import NamedTuple

from goals import AVOIDABLE, DEFAULT_GOAL, SEEKABLE
from fruit_fly_circuits import (BEHAVIOURS, OUT_FWD, OUT_HEADING, OUT_HOME_X, OUT_HOME_Y, OUT_KC_ACTIVE,
                                OUT_NOVELTY, OUT_ODOR_ID, OUT_ODOR_MATCH, OUT_VALENCE)

VRAM_BUDGET_GB = 1.5
URGENCY_LEVELS = ["threat none, nothing happening", "threat none, odor present", "threat approaching",
                  "threat imminent"]

# Criteria name the state fields they depend on. Measured with eval_system1.py: describing behaviours
# ("run away from a threat") gave FLEE on 0% of threatened states; naming the field gave 99.2%.
QUESTIONS = {
    "behaviour": {"type": "choice", "instructions": "Pick the fly's behaviour. Safety first: any threat means FLEE.",
                  "criteria": {"FORAGE": "threat is none and an odor is present that is not punished",
                               "FLEE": "threat is approaching or imminent",
                               "ORIENT": "threat is none, no useful odor, and home is not here",
                               "IDLE": "threat is none, no odor, and home is here"}},
    "urgency": {"type": "score", "instructions": "How urgent is it? Driven by the threat field.",
                "criteria": URGENCY_LEVELS},
    "jump": {"type": "noul", "instructions": "Is the threat field approaching or imminent?",
             "criteria": {"true": "threat is approaching or imminent", "false": "threat is none"}},
}


class Decision(NamedTuple):
    probs: tuple          # P(behaviour), BEHAVIOURS order
    urgency: float        # 0..1
    p_jump: float
    backend: str
    ms: float


# ------------------------------------------------------------------ state tensor -> words
_DIRS = ["ahead", "ahead-left", "left", "behind-left", "behind", "behind-right", "right", "ahead-right"]


def describe(out, loom, odor_names, goal=DEFAULT_GOAL):
    """Quantise the brain's output vector into a short word-valued state.

    Laya and Jev are text models that are weak at arithmetic and angle comparison, so they get
    categories, never raw phases or KC indices. The ring-attractor phase only matters relative to a
    goal (home is 'behind-left'), and the KC hash only matters as identity (odor name / new / familiar)
    and learned value. Bins also make the state change rarely, which is what gates System 1 calls.
    """
    if out[OUT_KC_ACTIVE] <= 1:
        odor, familiarity = "none", "n/a"
    else:
        odor = odor_names.get(int(out[OUT_ODOR_ID]), "unknown") if out[OUT_ODOR_MATCH] > 0.6 else "unknown"
        familiarity = "new" if out[OUT_NOVELTY] > 0.5 else "familiar"
    v = float(out[OUT_VALENCE])
    hx, hy = float(out[OUT_HOME_X]), float(out[OUT_HOME_Y])
    dist = math.hypot(hx, hy)
    rel = (math.atan2(hy, hx) - float(out[OUT_HEADING])) % (2 * math.pi)
    return {
        "threat": "imminent" if loom > 0.5 else "approaching" if loom > 0.2 else "none",
        "odor": odor,
        "odor_familiarity": familiarity,
        "odor_memory": "rewarded" if v > 0.02 else "punished" if v < -0.02 else "neutral",
        "home": "here" if dist < 3 else f"{_DIRS[round(rel / (math.pi / 4)) % 8]}, "
                                        f"{'near' if dist < 30 else 'far'}",
        "moving": "yes" if out[OUT_FWD] > 0.05 else "no",
        # the goal rides along in the state, so the decision cache is keyed per goal for free
        "goal_seek": "+".join(i for i in SEEKABLE if i in goal.seek) or "none",
        "goal_avoid": "+".join(i for i in AVOIDABLE if i in goal.avoid) or "none",
        "goal_heading": goal.heading or "none",
        "goal_rest": "yes" if goal.rest else "no",
    }


def _parse(answers, backend, t0):
    p = answers["behaviour"]["probabilities"]
    return Decision(tuple(float(p[b]) for b in BEHAVIOURS),
                    float(answers["urgency"]["score"]) / (len(URGENCY_LEVELS) - 1),
                    float(answers["jump"]["noul"]), backend, (time.perf_counter() - t0) * 1e3)


# ------------------------------------------------------------------ backends
def _check_vram():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    free = torch.cuda.mem_get_info()[0]
    if free < VRAM_BUDGET_GB * 2**30:
        raise RuntimeError(f"only {free / 2**30:.2f} GB VRAM free, need {VRAM_BUDGET_GB}")


def _cap_vram():
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(VRAM_BUDGET_GB * 2**30 / total)  # over-budget -> OOM -> fallback


def laya_backend(model=os.environ.get("LAYA_MODEL", "convaiinnovations/laya-typed-decisions")):
    import laya
    import torch
    _check_vram()
    # fast=True: TileLang kernels + CUDA graphs per shape bucket (Linux only); elsewhere it warns and stays eager
    agent = laya.load(model, device="cuda", fast=True)
    # Laya keeps fp32 weights under bf16 autocast: 1.6 GB resident. bf16 storage: 831 MB, 37 ms vs 54 ms p50,
    # same choice on 191/194 states, max |dP| 0.0066 (measured). Cap only after, or the fp32 load itself OOMs.
    if agent.backend == "eager":
        agent.model.to(torch.bfloat16)
        torch.cuda.empty_cache()
    _cap_vram()
    agent.warmup([(1, 288, 10)])

    def decide(state):
        t0 = time.perf_counter()
        return _parse(agent.system_one(state, QUESTIONS)["answers"], "laya", t0)
    return decide


def _poster(url, key, timeout):
    """POST JSON over one persistent connection, reconnecting after any error. Returns (base path, post).
    Persistent because a fresh TLS handshake to the nearest Cloudflare edge measured 57-300 ms from
    this laptop, before any inference, so reconnecting per decision would dominate."""
    u = urllib.parse.urlsplit(url)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    conn = None

    def post(path, payload):
        nonlocal conn
        conn = conn or conn_cls(u.netloc, timeout=timeout)
        try:
            conn.request("POST", path, json.dumps(payload), headers)
            r = conn.getresponse()
            body = json.load(r)
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {str(body)[:200]}")
            return body
        except Exception:
            conn.close()
            conn = None
            raise
    return u.path.rstrip("/"), post


def http_backend(url=os.environ.get("JEV_URL"), key=os.environ.get("JEV_API_KEY"), timeout=0.5):
    """Two wire shapes, picked from the URL:
      .../ai/run    Cloudflare Workers AI (how Jev is served): {"model": "typesafe/jev", "input": {...}}
      anything else `laya-serve` / Jev-native: POST {url}/v1/systemone with {"state", "questions"}"""
    if not url:
        raise RuntimeError("JEV_URL not set")
    base, post = _poster(url, key, timeout)
    cloudflare = base.endswith("/ai/run")
    path = base if cloudflare else base + "/v1/systemone"

    def decide(state):
        t0 = time.perf_counter()
        payload = {"state": state, "questions": QUESTIONS}
        if cloudflare:
            payload = {"model": "typesafe/jev", "input": payload}
        body = post(path, payload)
        return _parse(body.get("result", body)["answers"], "http", t0)  # Cloudflare wraps in "result"
    return decide


GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
LLM_PROMPT = "\n".join([
    "You choose a fruit fly's next behaviour from its sensed state (the user message, JSON).",
    "Reply with JSON only, in exactly this shape:",
    '{"behaviour": {"FORAGE": p, "FLEE": p, "ORIENT": p, "IDLE": p}, "urgency": n, "jump": p}',
    "Each p is a probability (the four behaviour probabilities sum to 1); n is an integer 0-3.",
    "behaviour: " + QUESTIONS["behaviour"]["instructions"],
    *(f"  {k}: {v}" for k, v in QUESTIONS["behaviour"]["criteria"].items()),
    "urgency: " + "; ".join(f"{i} = {v}" for i, v in enumerate(URGENCY_LEVELS)),
    "jump: probability that " + QUESTIONS["jump"]["criteria"]["true"],
])


def llm_config():
    """(url, key, model, timeout) for any OpenAI-compatible chat API, from env only; None if unconfigured.
      Gemini   GEMINI_API_KEY                     (base URL and model default to Gemini's; LLM_MODEL overrides)
      others   LLM_BASE_URL, LLM_MODEL, LLM_API_KEY  (OpenAI, Anthropic, Groq, OpenRouter, ...)
      local    LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=<ollama model>   (no key needed)"""
    env = os.environ
    gemini = not env.get("LLM_BASE_URL") and env.get("GEMINI_API_KEY")
    url = env.get("LLM_BASE_URL") or (GEMINI_OPENAI_URL if gemini else None)
    key = env.get("LLM_API_KEY") or (env.get("GEMINI_API_KEY") if gemini else None)
    model = env.get("LLM_MODEL") or ("gemini-3.8-flash" if gemini else None)
    if not (url and model):
        return None
    return url, key, model, float(env.get("LLM_TIMEOUT", "5"))


def llm_backend():
    """System 1 through any OpenAI-compatible chat API (see llm_config for the env vars).
    The reply is untrusted: anything but the expected JSON raises, and the chain falls back to rules."""
    config = llm_config()
    if config is None:
        raise RuntimeError("set GEMINI_API_KEY, or LLM_BASE_URL + LLM_MODEL (+ LLM_API_KEY)")
    url, key, model, timeout = config
    base, post = _poster(url, key, timeout)

    def decide(state):
        t0 = time.perf_counter()
        body = post(base + "/chat/completions", {
            "model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": json.dumps(state)}]})
        text = body["choices"][0]["message"]["content"]
        reply = json.loads(text[text.index("{"):text.rindex("}") + 1])   # tolerates code fences and preambles
        p = [max(0.0, float(reply["behaviour"][b])) for b in BEHAVIOURS]
        if not sum(p):
            raise ValueError(f"all-zero behaviour probabilities: {text[:200]}")
        answers = {"behaviour": {"probabilities": {b: x / sum(p) for b, x in zip(BEHAVIOURS, p)}},
                   "urgency": {"score": min(3.0, max(0.0, float(reply["urgency"])))},
                   "jump": {"noul": min(1.0, max(0.0, float(reply["jump"])))}}
        return _parse(answers, "llm", t0)
    return decide


def rules_backend():
    def decide(state):
        t0 = time.perf_counter()
        threat = state["threat"]
        seek, avoid = state.get("goal_seek", "none"), state.get("goal_avoid", "none")
        home_close = state["home"] == "here" or state["home"].endswith("near")
        if threat != "none":                                  # no goal overrides a threat
            p, urg, jump = (0.05, 0.85, 0.05, 0.05), (3 if threat == "imminent" else 2), 0.9
        elif state.get("goal_rest", "no") == "yes":
            p, urg, jump = (0.0, 0.0, 0.0, 1.0), 0, 0.02      # pure IDLE: the blended IDLE answer still creeps
        elif "home" in seek.split("+") or ("home" in avoid.split("+") and home_close):
            p, urg, jump = (0.1, 0.05, 0.75, 0.1), 1, 0.05    # ORIENT; the home sign in the brain sets the direction
        elif "banana" in seek.split("+") or state.get("goal_heading", "none") != "none":
            p, urg, jump = (0.8, 0.05, 0.1, 0.05), 1, 0.05    # FORAGE; odor signs + heading drive steer
        elif state["odor"] != "none" and state["odor_memory"] != "punished":
            p, urg, jump = (0.8, 0.05, 0.1, 0.05), 1, 0.05
        elif state["home"] != "here":
            p, urg, jump = (0.1, 0.05, 0.75, 0.1), 1, 0.05
        else:
            p, urg, jump = (0.1, 0.05, 0.05, 0.8), 0, 0.02
        answers = {"behaviour": {"probabilities": dict(zip(BEHAVIOURS, p))}, "urgency": {"score": urg},
                   "jump": {"noul": jump}}
        return _parse(answers, "rules", t0)
    return decide


def synthetic_backend(tokens=128):
    """Laya's GPU cost without its weights: a random ModernBERT-large forward, graph-captured, then rules.
    For measuring the loop under real System-1 GPU load before downloading the model."""
    import torch
    from transformers import ModernBertConfig, ModernBertModel
    _check_vram()
    _cap_vram()
    cfg = ModernBertConfig(hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
                           intermediate_size=2624, attn_implementation="sdpa")
    m = ModernBertModel(cfg).to("cuda", torch.float16).eval()
    ids = torch.randint(5, 50000, (1, tokens), device="cuda")
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                m(input_ids=ids, attention_mask=mask)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            m(input_ids=ids, attention_mask=mask)
    rules = rules_backend()

    def decide(state):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        d = rules(state)
        return d._replace(backend="synthetic", ms=(time.perf_counter() - t0) * 1e3)
    return decide


BACKENDS = {"laya": laya_backend, "http": http_backend, "llm": llm_backend, "synthetic": synthetic_backend,
            "rules": rules_backend}


# ------------------------------------------------------------------ precompiled tables
# describe() can only emit these words, so a model's whole policy is 1,938 answers: compile them once,
# then a live decision is a dict lookup. Models never read the goal fields, so one table serves every goal.
BASE_KEYS = ("threat", "odor", "odor_familiarity", "odor_memory", "home", "moving")
THREATS = ("none", "approaching", "imminent")
ODORS = [("none", "n/a", "neutral")] + list(itertools.product(("banana", "geosmin", "unknown"), ("new", "familiar"),
                                                              ("rewarded", "punished", "neutral")))
HOMES = ["here"] + [f"{d}, {r}" for d in _DIRS for r in ("near", "far")]


def all_states():
    for threat, (odor, fam, mem), home, moving in itertools.product(THREATS, ODORS, HOMES, ("yes", "no")):
        yield {"threat": threat, "odor": odor, "odor_familiarity": fam, "odor_memory": mem, "home": home,
               "moving": moving}


VOCAB = {k: {s[k] for s in all_states()} for k in BASE_KEYS}
N_STATES = len(ODORS) * len(THREATS) * len(HOMES) * 2


def fill_order():
    """Threat states first: they're the ones where a late answer costs the fly."""
    states = list(all_states())
    return [s for s in states if s["threat"] != "none"] + [s for s in states if s["threat"] == "none"]


def base_key(state):
    return tuple(state[k] for k in BASE_KEYS)


def table_path(name, tables_dir="tables"):
    """tables/<backend>-<slug>-<hash8>.jsonl. The hash covers the model id and the prompt the backend sends, so
    a changed question or model starts a fresh table. Read from env now, without loading the model."""
    env = os.environ
    if name == "laya":
        model_id = slug = env.get("LAYA_MODEL", "convaiinnovations/laya-typed-decisions")
    elif name == "llm":
        config = llm_config()
        if config is None:
            return None
        model_id, slug = config[0] + " " + config[2], config[2]
    elif name == "http":
        model_id, slug = env.get("JEV_URL", ""), "jev"      # never the URL in a filename: it can carry an account id
    elif name == "synthetic":
        model_id = slug = "synthetic"
    else:
        return None
    prompt = LLM_PROMPT if name == "llm" else QUESTIONS
    digest = hashlib.sha256("\n".join([name, model_id, json.dumps(prompt, sort_keys=True)]).encode()).hexdigest()
    return Path(tables_dir) / f"{name}-{re.sub(r'[^A-Za-z0-9._-]+', '-', slug)}-{digest[:8]}.jsonl"


def valid_answer(state, probs, urgency, p_jump):
    try:
        if tuple(state) != BASE_KEYS or any(state[k] not in VOCAB[k] for k in BASE_KEYS):
            return False
        nums = [float(x) for x in probs] + [float(urgency), float(p_jump)]
    except (TypeError, ValueError, KeyError):
        return False
    return (len(nums) == 6 and all(math.isfinite(x) and 0 <= x <= 1 for x in nums)
            and abs(sum(nums[:4]) - 1) <= 1e-3)


def load_table(path, name):
    """The file is editable, so it is input: invalid lines are skipped and counted; the last duplicate wins."""
    table, skipped = {}, 0
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        return table
    with f:
        for line in f:
            try:
                row = json.loads(line)
                state = row["state"]
                if not valid_answer(state, row["probs"], row["urgency"], row["p_jump"]):
                    raise ValueError
                table[base_key(state)] = Decision(tuple(float(p) for p in row["probs"]), float(row["urgency"]),
                                                  float(row["p_jump"]), f"{name} table", float(row["ms"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                skipped += 1
    if skipped:
        print(f"[system1] {path}: skipped {skipped} invalid lines", flush=True)
    return table


def append_table(path, answers):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(json.dumps({"state": {k: s[k] for k in BASE_KEYS}, "probs": list(d.probs),
                                    "urgency": d.urgency, "p_jump": d.p_jump, "ms": round(d.ms, 3)}) + "\n"
                        for s, d in answers))


def lookup(state, table, rules):
    """Threat: the table (the model's own FLEE). Any goal set: rules, the only policy that knows goals.
    Otherwise the table. A miss is always rules."""
    goal_set = any(state.get(k, "none") != "none" for k in ("goal_seek", "goal_avoid", "goal_heading")) \
        or state.get("goal_rest", "no") == "yes"
    if state["threat"] == "none" and goal_set:
        return rules(state)
    return table.get(base_key(state)) or rules(state)


# The filler: the only code that calls a model. It runs in agent_loop's worker process (or in-process for
# --compile), one chunk at a time, so the tick only ever sees finished answers.
CHUNK = 32
STALL_CHUNKS = 5                                # consecutive chunks with no good answer: give up, rules cover
_filler = None


def filler_init(name):
    global _filler
    try:
        _filler = BACKENDS[name]()
    except Exception as e:                      # missing package, no VRAM headroom, no key
        print(f"[system1] {name} unavailable: {type(e).__name__}: {e}", flush=True)
        _filler = None


def compile_chunk(states, decide=None):
    """[(state, Decision or None)]; None for a failed or invalid answer. None overall if no model loaded."""
    decide = decide or _filler
    if decide is None:
        return None
    out = []
    for state in states:
        try:
            d = decide(state)
            ok = valid_answer({k: state[k] for k in BASE_KEYS}, d.probs, d.urgency, d.p_jump)
        except Exception as e:
            print(f"[system1] compile failed on {base_key(state)}: {type(e).__name__}: {e}", flush=True)
            d, ok = None, False
        out.append((state, d if ok else None))
    return out


def missing(table):
    return [s for s in fill_order() if base_key(s) not in table]


def compile_table(name, tables_dir="tables", decide=None):
    """Fill a backend's table in-process until complete or stalled (the --compile CLI)."""
    if decide is None:
        filler_init(name)
        decide = _filler
        if decide is None:
            return {}
    path = table_path(name, tables_dir)
    table = load_table(path, name)
    todo = missing(table)
    print(f"[system1] {path}: {len(table)}/{N_STATES} compiled, {len(todo)} to go", flush=True)
    t0, streak, threats_done = time.perf_counter(), 0, False
    for i in range(0, len(todo), CHUNK):
        good = [(s, d) for s, d in compile_chunk(todo[i:i + CHUNK], decide) if d is not None]
        append_table(path, good)
        table.update((base_key(s), d._replace(backend=f"{name} table")) for s, d in good)
        streak = 0 if good else streak + 1
        if streak >= STALL_CHUNKS:
            print(f"[system1] stalled after {STALL_CHUNKS} chunks with no answer", flush=True)
            break
        if not threats_done and all(s["threat"] == "none" for s in todo[i + CHUNK:i + CHUNK + 1]):
            threats_done = True
            print(f"[system1] threat states done at {time.perf_counter() - t0:.1f}s", flush=True)
        if (i // CHUNK) % 10 == 9:
            print(f"[system1] {len(table)}/{N_STATES} at {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[system1] {len(table)}/{N_STATES} compiled in {time.perf_counter() - t0:.1f}s -> {path}", flush=True)
    return table


# ------------------------------------------------------------------ worker-process entry points
_chain = []


def worker_init(names):
    """Build the fallback chain, e.g. ("laya", "http", "rules"). Backends that fail to load are skipped."""
    for name in names:
        try:
            _chain.append((name, BACKENDS[name]()))
        except Exception as e:  # missing package, no VRAM headroom, no JEV_URL
            print(f"[system1] {name} unavailable: {type(e).__name__}: {e}", flush=True)
    if not any(n == "rules" for n, _ in _chain):
        _chain.append(("rules", rules_backend()))


_cache = {}   # the worded state space is finite (a few thousand keys) and models are deterministic


def worker_decide(state):
    key = tuple(state.values())
    if key in _cache:
        return _cache[key]._replace(ms=0.0)
    for name, fn in list(_chain):
        try:
            d = fn(state)
            if name != "rules":          # don't let an outage's fallback answer stick
                _cache[key] = d
            return d
        except Exception as e:
            print(f"[system1] {name} failed: {type(e).__name__}: {e}", flush=True)
            if name in ("laya", "synthetic"):   # OOM / CUDA error: don't retry a broken local model every tick
                _chain.remove((name, fn))
    raise RuntimeError("unreachable: rules backend cannot fail")


if __name__ == "__main__" and "--compile" in sys.argv:    # python system1_engine.py --compile laya
    compile_table(sys.argv[sys.argv.index("--compile") + 1])
elif __name__ == "__main__":
    import torch
    out = torch.zeros(12)
    out[OUT_KC_ACTIVE], out[OUT_ODOR_MATCH], out[OUT_VALENCE], out[OUT_HOME_X] = 100, 0.9, 0.1, 40
    s = describe(out, loom=0.0, odor_names={0: "banana"})
    assert s == {"threat": "none", "odor": "banana", "odor_familiarity": "familiar", "odor_memory": "rewarded",
                 "home": "ahead, far", "moving": "no", "goal_seek": "none", "goal_avoid": "none",
                 "goal_heading": "none", "goal_rest": "no"}, s
    worker_init(("rules",))
    d = worker_decide(s)
    assert d.probs.index(max(d.probs)) == BEHAVIOURS.index("FORAGE") and 0 <= d.urgency <= 1
    assert worker_decide(describe(out, loom=0.6, odor_names={}))[0][1] > 0.5     # FLEE under threat
    # goal-aware rules: a threat beats every goal; rest > home > forage; no goal = today's rules
    from goals import Goal
    rules = rules_backend()

    def choice(state):
        p = rules(state).probs
        return BEHAVIOURS[p.index(max(p))]

    def G(seek=(), avoid=(), heading=None, rest=False):
        return Goal(seek, avoid, heading, rest, "", "parser")

    names = {0: "banana"}
    for g in (G(rest=True), G(seek=("banana",)), G(seek=("home",)), G(heading="north")):
        assert choice(describe(out, 0.6, names, g)) == "FLEE", g
    assert choice(describe(out, 0.0, names, G(rest=True))) == "IDLE"
    assert choice(describe(out, 0.0, names, G(seek=("home",)))) == "ORIENT"
    assert choice(describe(out, 0.0, names, G(seek=("banana",)))) == "FORAGE"
    assert choice(describe(out, 0.0, names, G(avoid=("home",)))) == "FORAGE"           # home far
    no_odor = out.clone()
    no_odor[OUT_KC_ACTIVE] = 0
    assert choice(describe(no_odor, 0.0, names, G(heading="north"))) == "FORAGE"
    for home_x in (10.0, 1.0):                                                          # "…, near" and "here"
        near = out.clone()
        near[OUT_HOME_X] = home_x
        assert choice(describe(near, 0.0, names, G(avoid=("home",)))) == "ORIENT", home_x
    st = describe(out, 0.0, names, G(seek=("home", "banana"), heading="north-east"))
    assert (st["goal_seek"], st["goal_avoid"], st["goal_heading"], st["goal_rest"]) == ("banana+home", "none", "north-east", "no")
    plain = {k: v for k, v in describe(out, 0.0, names).items() if not k.startswith("goal_")}
    assert rules(plain).probs == rules(describe(out, 0.0, names)).probs                 # old states still work
    # http backend against a local stand-in for laya-serve: parse + persistent connection
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = []

    class Stub(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            seen.append((self.path, self.client_address[1], self.headers.get("Authorization")))
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/chat/completions"):            # OpenAI-compatible: fenced JSON, unnormalised
                assert req["model"] == "m" and json.loads(req["messages"][1]["content"]) == s
                content = '```json\n{"behaviour": {"FORAGE": 1, "FLEE": 3, "ORIENT": 0, "IDLE": 0},' \
                          ' "urgency": 3, "jump": 0.9}\n```'
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            else:
                assert set(req["questions"]) == set(QUESTIONS)
                body = json.dumps({"answers": {"behaviour": {"probabilities": dict(zip(BEHAVIOURS, (0.1, 0.7, 0.1, 0.1)))},
                                               "urgency": {"score": 3.0}, "jump": {"noul": 0.9}}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    hb = http_backend(f"http://127.0.0.1:{srv.server_port}", key=None)
    d1, d2 = hb(s), hb(s)
    assert d1.probs[1] == 0.7 and d1.urgency == 1.0 and d1.p_jump == 0.9
    assert seen[0][0] == "/v1/systemone" and seen[0][1] == seen[1][1], seen   # same client port = reused
    os.environ.update(LLM_BASE_URL=f"http://127.0.0.1:{srv.server_port}/v1", LLM_MODEL="m", LLM_API_KEY="k")
    d3 = llm_backend()(s)
    assert seen[-1][0] == "/v1/chat/completions" and seen[-1][2] == "Bearer k", seen[-1]
    assert d3.probs == (0.25, 0.75, 0.0, 0.0) and d3.urgency == 1.0 and d3.p_jump == 0.9 and d3.backend == "llm"
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        del os.environ[k]
    os.environ["GEMINI_API_KEY"] = "g"                                   # Gemini shortcut resolves its own defaults
    try:
        llm_backend()
    finally:
        del os.environ["GEMINI_API_KEY"]
    srv.shutdown()
    # precompiled tables: the state space, the file, goal precedence, identity
    import tempfile
    from pathlib import Path
    states = list(all_states())
    assert len(states) == N_STATES == 1938 and all(tuple(st) == BASE_KEYS for st in states)
    assert tuple(k for k in describe(out, 0.0, {0: "banana"}) if not k.startswith("goal_")) == BASE_KEYS
    order = fill_order()
    assert len(order) == 1938 and all(st["threat"] != "none" for st in order[:1292]) \
        and all(st["threat"] == "none" for st in order[1292:])
    with tempfile.TemporaryDirectory() as tmp:            # load_table skips bad lines, last duplicate wins
        good = states[0]
        p = Path(tmp) / "t.jsonl"
        append_table(p, [(good, Decision((0.7, 0.1, 0.1, 0.1), 0.3, 0.1, "laya", 40.0))])
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"state": states[1], "probs": [0.9, 0.9, 0, 0], "urgency": 0, "p_jump": 0, "ms": 1}) + "\n")
            f.write(json.dumps({"state": {**states[2], "odor": "durian"}, "probs": [1, 0, 0, 0], "urgency": 0,
                                "p_jump": 0, "ms": 1}) + "\n")
            f.write(json.dumps({"state": good, "probs": [0.1, 0.1, 0.1, 0.7], "urgency": 0, "p_jump": 0, "ms": 2}) + "\n")
            f.write('{"state": {"threat": "no')
        t = load_table(p, "laya")
        assert list(t) == [base_key(good)] and t[base_key(good)].probs == (0.1, 0.1, 0.1, 0.7), t
        assert t[base_key(good)].backend == "laya table" and t[base_key(good)].ms == 2
        assert load_table(Path(tmp) / "missing.jsonl", "laya") == {}
    calm = {**s, "goal_rest": "yes"}                       # goal precedence
    t2 = {base_key(s): Decision((0.25,) * 4, 0.5, 0.5, "laya table", 9.0)}
    assert lookup(calm, t2, rules).backend == "rules" and lookup(calm, t2, rules).probs == (0, 0, 0, 1)
    assert lookup(s, t2, rules).backend == "laya table"
    threat = {**s, "threat": "imminent", "goal_rest": "yes"}
    t2[base_key(threat)] = Decision((0.1, 0.7, 0.1, 0.1), 1.0, 0.9, "laya table", 9.0)
    assert lookup(threat, t2, rules).backend == "laya table"
    assert lookup({**s, "threat": "approaching"}, t2, rules).backend == "rules"     # threat miss -> rules
    assert lookup({**s, "odor": "durian"}, t2, rules).backend == "rules"            # unknown_state_uses_rules
    assert table_path("rules") is None and table_path("laya").name.startswith("laya-convaiinnovations-laya-typed-decisions-")
    assert table_path("http").name.startswith("http-jev-") and table_path("llm") is None
    before = table_path("laya")
    QUESTIONS["jump"]["instructions"] += " "
    assert table_path("laya") != before
    QUESTIONS["jump"]["instructions"] = QUESTIONS["jump"]["instructions"][:-1]
    assert table_path("laya") == before
    calls = []                                             # the filler: failures stay missing, never rules

    def stub(state):
        calls.append(base_key(state))
        if state["home"] == "behind, far":
            raise TimeoutError("stub")
        if state["home"] == "left, near":
            return Decision((float("nan"),) * 4, 0.0, 0.0, "stub", 1.0)          # bad_answer_not_stored
        return rules(state)._replace(backend="stub", ms=5.0)
    res = compile_chunk(fill_order()[:32], decide=stub)
    assert len(res) == 32 and all(d is None or d.backend == "stub" for _, d in res)
    with tempfile.TemporaryDirectory() as tmp:
        tbl = compile_table("synthetic", tmp, decide=stub)
        bad = [st for st in all_states() if st["home"] in ("behind, far", "left, near")]
        assert len(tbl) == N_STATES - len(bad) and all(base_key(st) not in tbl for st in bad)
        assert len(load_table(table_path("synthetic", tmp), "synthetic")) == len(tbl)
        calls.clear()                                      # resume_fills_only_missing (the model is back up)

        def healed(state):
            calls.append(base_key(state))
            return rules(state)._replace(backend="stub", ms=5.0)
        assert len(compile_table("synthetic", tmp, decide=healed)) == N_STATES
        assert sorted(calls) == sorted(base_key(st) for st in bad)
        calls.clear()                                      # stall after 5 empty chunks

        def broken(state):
            calls.append(1)
            raise TimeoutError("down")
        assert compile_table("synthetic", tmp + "/stall", decide=broken) == {} and len(calls) == STALL_CHUNKS * CHUNK
    print("self-check OK", s)
