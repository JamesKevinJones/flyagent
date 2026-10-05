"""Free-text goals: a small Goal, a parser for the world's vocabulary, and an optional LLM on top.

The goal is interpreted once, when it changes; fast rules and circuit buffers execute it every tick.
"""
import json
import math
import re
import threading
import time
from typing import NamedTuple

SEEKABLE = ("banana", "home")
AVOIDABLE = ("geosmin", "home")
COMPASS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")
HEADINGS = {name: i * math.pi / 4 for i, name in enumerate(COMPASS)}   # landmark frame: east 0, north pi/2
MAX_GOAL_CHARS = 200


class Goal(NamedTuple):
    seek: tuple
    avoid: tuple
    heading: str | None
    rest: bool
    text: str
    source: str        # "parser" | "llm" | "default"


DEFAULT_GOAL = Goal((), (), None, False, "", "default")


class Interpretation(NamedTuple):
    goal: Goal
    ignored: tuple
    ms: float
    note: str | None


def validate(goal):
    """Return why `goal` is not a goal this world can execute, or None."""
    if any(s not in SEEKABLE for s in goal.seek) or any(a not in AVOIDABLE for a in goal.avoid):
        return f"unknown item in seek {goal.seek} / avoid {goal.avoid}"
    if goal.heading is not None and goal.heading not in HEADINGS:
        return f"unknown heading {goal.heading!r}"
    if not isinstance(goal.rest, bool):
        return "rest must be true or false"
    if set(goal.seek) & set(goal.avoid):
        return f"{sorted(set(goal.seek) & set(goal.avoid))} both sought and avoided"
    if goal.rest and (goal.seek or goal.heading):
        return "rest cannot be combined with a seek or a heading"
    return None


# ------------------------------------------------------------------ parser
NOUNS = {**dict.fromkeys(("food", "fruit", "banana", "bananas", "sugar", "eat"), "banana"),
         **dict.fromkeys(("smell", "smells", "stink", "stinky", "mould", "mouldy", "mold", "moldy", "earthy",
                          "geosmin"), "geosmin"),
         **dict.fromkeys(("home", "nest", "start", "base"), "home")}
DIRECTIONS = {**{c: c for c in COMPASS},
              **{"up": "north", "top": "north", "down": "south", "bottom": "south", "left": "west", "right": "east",
                 "northeast": "north-east", "northwest": "north-west", "southeast": "south-east",
                 "southwest": "south-west", "ne": "north-east", "nw": "north-west", "se": "south-east",
                 "sw": "south-west"}}
SEEK_VERBS = {"find", "go", "get", "seek", "eat", "reach", "head", "walk", "move", "turn", "look_for", "go_to"}
AVOID_VERBS = {"avoid", "away_from", "stay_away_from", "keep_away_from", "dont_go_near", "flee", "escape"}
NEGATIONS = {"not", "don't", "dont", "never", "no"}
REST_VERBS = {"rest", "stay", "stop", "wait", "sit", "idle"}
PHRASES = {("stay", "away", "from"): "stay_away_from", ("keep", "away", "from"): "keep_away_from",
           ("don't", "go", "near"): "dont_go_near", ("dont", "go", "near"): "dont_go_near",
           ("do", "not", "go", "near"): "dont_go_near", ("away", "from"): "away_from",
           ("look", "for"): "look_for", ("go", "to"): "go_to"}
STOP = {"i", "i'm", "im", "the", "a", "an", "to", "of", "that", "this", "is", "it", "it's", "me", "my", "please",
        "some", "then", "so", "be", "are", "at", "in", "on", "for", "with", "you", "your", "we", "very", "really",
        "just", "towards", "toward", "there", "here", "side", "way", "direction", "near", "fly", "little",
        "let's", "lets", "can", "could", "would", "should", "now", "and", "but", "or", "all", "of", "too"}


def _tokens(text):
    text = text.lower().replace("’", "'").replace("‘", "'").replace("-", " ")   # curly quotes: iOS/macOS
    words = re.sub(r"[^a-z0-9'\s]", " ", text).split()
    out, i = [], 0
    while i < len(words):
        for n in (4, 3, 2):                                   # longest phrase first
            if tuple(words[i:i + n]) in PHRASES:
                out.append(PHRASES[tuple(words[i:i + n])])
                i += n
                break
        else:
            w = words[i]
            if w in ("north", "south") and i + 1 < len(words) and words[i + 1] in ("east", "west"):
                out.append(f"{w}{words[i + 1]}")              # "north east" -> "northeast"
                i += 2
                continue
            out.append(w)
            i += 1
    return out


def parse(text):
    """Map text onto the world's vocabulary. Never returns an invalid goal; returns the words it couldn't use."""
    seek, avoid, ignored = [], [], []
    asked = {"seek": set(), "avoid": set()}                  # every item requested per mode, even unsupported ones
    heading, rest = None, False
    mode, negated = "seek", False                             # a bare noun ("banana!") means seek
    for tok in _tokens(text) + [","]:
        if tok in ("but", ","):
            mode, negated = "seek", False
        elif tok == "and":                                    # "avoid X and Y": the verb carries over, a negation doesn't
            negated = False
        elif tok in NEGATIONS:
            mode, negated = "avoid", True
        elif tok in AVOID_VERBS:
            mode = "avoid"
        elif tok in SEEK_VERBS:
            mode = "avoid" if negated else "seek"
            if tok == "eat" and negated:                      # "don't eat": banana can't be avoided in this world
                ignored.append("eat")
            elif tok == "eat":
                seek.append("banana")
                asked[mode].add("banana")
        elif tok in REST_VERBS:
            rest = rest or not negated
        elif tok in NOUNS:
            item = NOUNS[tok]
            asked[mode].add(item)
            if mode == "seek" and item in SEEKABLE:
                seek.append(item)
            elif mode == "avoid" and item in AVOIDABLE:
                avoid.append(item)
            else:
                ignored.append(tok)                           # e.g. "avoid the banana": not in this world's vocabulary
        elif tok in DIRECTIONS:
            if negated:                                       # "don't go north": there's no "avoid a heading"
                ignored.append(tok)
            else:
                heading = DIRECTIONS[tok]
        elif tok not in STOP and not tok.isdigit():
            ignored.append(tok)
    seek, avoid = list(dict.fromkeys(seek)), list(dict.fromkeys(avoid))
    for both in asked["seek"] & asked["avoid"]:               # sought and avoided: drop from both, report it
        seek = [s for s in seek if s != both]
        avoid = [a for a in avoid if a != both]
        ignored = [w for w in ignored if NOUNS.get(w) != both] + [both]
    if rest and (seek or heading):                            # moving goals win over rest
        rest = False
        ignored.append("rest")
    return Goal(tuple(seek), tuple(avoid), heading, rest, text.strip(), "parser"), tuple(dict.fromkeys(ignored))


# ------------------------------------------------------------------ interpretation
LLM_GOAL_PROMPT = "\n".join([
    "Turn the user's instruction for a simulated fruit fly into JSON. Reply with JSON only, exactly this shape:",
    '{"seek": [...], "avoid": [...], "heading": null or "<compass>", "rest": false}',
    f"seek may contain only: {', '.join(SEEKABLE)} (banana is the only food)",
    f"avoid may contain only: {', '.join(AVOIDABLE)} (geosmin is the earthy, mouldy smell)",
    f"heading is null or one of: {', '.join(COMPASS)} (map directions; up = north)",
    "rest is true only if the fly should stay still; never combine rest with seek or heading.",
    "Never put the same item in seek and avoid. Leave out anything that doesn't fit these values.",
])


def _from_llm(reply, text):
    try:
        obj = json.loads(reply[reply.index("{"):reply.rindex("}") + 1])
    except ValueError as e:
        raise ValueError(f"llm reply is not json ({e})")
    if not isinstance(obj, dict) or set(obj) - {"seek", "avoid", "heading", "rest"}:
        raise ValueError(f"llm reply invalid: unexpected keys {sorted(obj) if isinstance(obj, dict) else obj!r}")
    seek, avoid = obj.get("seek") or [], obj.get("avoid") or []
    if not isinstance(seek, list) or not isinstance(avoid, list):
        raise ValueError("llm reply invalid: seek/avoid must be lists")
    goal = Goal(tuple(dict.fromkeys(seek)), tuple(dict.fromkeys(avoid)), obj.get("heading"),
                obj.get("rest", False), text, "llm")
    problem = validate(goal)
    if problem:
        raise ValueError(f"llm reply invalid: {problem}")
    return goal


def interpret(text, llm=None):
    """Parser first; the LLM only for what the parser couldn't use. The LLM's reply is untrusted input."""
    text = text.strip()
    if len(text) > MAX_GOAL_CHARS:
        raise ValueError(f"goal longer than {MAX_GOAL_CHARS} characters")
    t0 = time.perf_counter()
    if not text:
        return Interpretation(DEFAULT_GOAL, (), 0.0, None)
    goal, ignored = parse(text)
    note = None
    if ignored and llm is not None:
        try:
            return Interpretation(_from_llm(llm(text), text), (), (time.perf_counter() - t0) * 1e3, None)
        except ValueError as e:
            note = str(e)
        except Exception as e:                                # timeout, HTTP error: keep the parser's answer
            note = f"llm failed: {type(e).__name__}: {e}"
    return Interpretation(goal, ignored, (time.perf_counter() - t0) * 1e3, note)


def make_llm():
    """A callable text -> raw reply, from the same env config as the `llm` System-1 backend; None if unconfigured."""
    import system1_engine                                     # here, not at top: system1_engine imports goals
    config = system1_engine.llm_config()
    if config is None:
        return None
    url, key, model, timeout = config
    base, post = system1_engine._poster(url, key, timeout)
    lock = threading.Lock()                                   # one persistent connection: one request at a time

    def ask(text):
        with lock:
            body = post(base + "/chat/completions", {
                "model": model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": LLM_GOAL_PROMPT}, {"role": "user", "content": text}]})
        return body["choices"][0]["message"]["content"]
    return ask


if __name__ == "__main__":
    def core(goal):
        return goal.seek, goal.avoid, goal.heading, goal.rest

    cases = [
        ("find the banana", (("banana",), (), None, False)),
        ("FIND the BANANA!!", (("banana",), (), None, False)),
        ("avoid the smell and head north", ((), ("geosmin",), "north", False)),
        ("don't go near the mould, go home", (("home",), ("geosmin",), None, False)),
        ("head north east", ((), (), "north-east", False)),
        ("go North-East", ((), (), "north-east", False)),
        ("go up", ((), (), "north", False)),
        ("rest", ((), (), None, True)),
        ("rest and find food", (("banana",), (), None, False)),
        ("find the banana but avoid the banana", ((), (), None, False)),
        ("stay away from home", ((), ("home",), None, False)),
        ("", ((), (), None, False)),
    ]
    for text, want in cases:
        got = parse(text)
        assert core(got[0]) == want, (text, core(got[0]), got[1])
        assert got[0].source == "parser" and validate(got[0]) is None, (text, got)
    assert parse("rest and find food")[1] == ("rest",)
    assert parse("find the banana but avoid the banana")[1] == ("banana",)
    assert parse("I'm starving")[1] == ("starving",)
    assert parse("find the banana and a home")[1] == ()                  # stop-words never reported
    # negation and conjunction (final review): never invalid, never inverted
    for text, want in [("don't eat", ((), (), None, False)),
                       ("don’t go home", ((), ("home",), None, False)),   # curly apostrophe (iOS/macOS)
                       ("don't go north", ((), (), None, False)),
                       ("go north, not south", ((), (), "north", False)),
                       ("avoid the smell and the mould", ((), ("geosmin",), None, False)),
                       ("avoid home and the smell", ((), ("home", "geosmin"), None, False)),
                       ("don't go home and find food", (("banana",), ("home",), None, False))]:
        got = parse(text)
        assert core(got[0]) == want and validate(got[0]) is None, (text, core(got[0]), got[1])
    assert "eat" in parse("don't eat")[1] and "south" in parse("go north, not south")[1]

    def never(text):
        raise AssertionError("llm called for fully parsed text")

    i = interpret("find the banana", never)
    assert i.goal.source == "parser" and i.ignored == () and i.note is None
    messy = "I'm starving but that stink is gross"
    i = interpret(messy, lambda t: '{"seek":["banana"],"avoid":["geosmin"],"heading":null,"rest":false}')
    assert i.goal.source == "llm" and i.ignored == () and core(i.goal) == (("banana",), ("geosmin",), None, False), i
    i = interpret(messy, lambda t: '{"seek":["pizza"]}')
    assert i.goal.source == "parser" and "invalid" in i.note and i.ignored, i

    def timeout(text):
        raise TimeoutError("timed out")

    i = interpret(messy, timeout)
    assert i.goal.source == "parser" and "TimeoutError" in i.note, i
    i = interpret(messy, lambda t: "not json")
    assert i.goal.source == "parser" and "json" in i.note, i
    i = interpret(messy, None)
    assert i.goal.source == "parser" and {"starving", "stink", "gross"} <= set(i.ignored), i   # verbless "stink" = seek, unseekable
    try:
        interpret("x" * 201)
        raise AssertionError("201 chars accepted")
    except ValueError:
        pass
    assert interpret("").goal == DEFAULT_GOAL and interpret("   ").goal == DEFAULT_GOAL
    assert validate(Goal(("banana",), ("banana",), None, False, "", "llm")) is not None
    assert validate(Goal((), (), None, True, "", "llm")) is None
    assert validate(Goal(("banana",), (), "up", False, "", "llm")) is not None
    assert abs(HEADINGS["north"] - 1.5707963) < 1e-6 and len(HEADINGS) == 8
    import os
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "GEMINI_API_KEY"):
        os.environ.pop(k, None)
    assert make_llm() is None
    os.environ.update(LLM_BASE_URL="http://127.0.0.1:9/v1", LLM_MODEL="m")   # nothing listens; construction is offline
    assert callable(make_llm())

    # concurrent interpretations share one LLM client: both must succeed (final review)
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class SlowLLM(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(0.3)
            content = '{"seek": ["home"], "avoid": [], "heading": null, "rest": false}'
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), SlowLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ.update(LLM_BASE_URL=f"http://127.0.0.1:{srv.server_port}/v1", LLM_MODEL="m")
    ask, results = make_llm(), []
    threads = [threading.Thread(target=lambda: results.append(interpret("back to the nest please", ask)))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r.goal.source for r in results] == ["llm", "llm"], [r.note for r in results]
    srv.shutdown()
    print("goals self-check OK")
