"""Local web demo: type a goal, watch the fly pursue it.  python serve.py  ->  http://127.0.0.1:8765

Standard library only. One thread steps the simulation every 15 ms; HTTP threads never touch the Sim
directly, they queue commands that the simulation thread applies between ticks.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from goals import interpret, make_llm

MAX_BODY = 2048
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/CascadiaCode.ttf": ("CascadiaCode.ttf", "font/ttf")}


class SimRunner:
    """Steps the Sim every `tick_ms` and publishes every other snapshot. Sleep-only pacing: a spin-wait
    here would hold the GIL against the HTTP threads (the CLI keeps its spin; it has no server)."""

    def __init__(self, sim, tick_ms=15.0):
        self.sim, self.period = sim, tick_ms / 1000
        self.latest, self.cond = None, threading.Condition()
        self.commands = queue.Queue()
        self.periods = deque(maxlen=400)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="sim", daemon=True)

    def do(self, fn):
        """Run fn(sim) on the simulation thread before the next tick."""
        self.commands.put(fn)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        with self.cond:
            self.cond.notify_all()

    def _loop(self):
        next_t = t_prev = time.perf_counter()
        while not self._stop.is_set():
            while not self.commands.empty():
                self.commands.get_nowait()(self.sim)
            snap = self.sim.step()
            if self.sim.tick % 2 == 0:
                with self.cond:
                    self.latest = snap
                    self.cond.notify_all()
            next_t += self.period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            now = time.perf_counter()
            self.periods.append((now - t_prev) * 1e3)
            if len(self.periods) >= 20:
                self.sim.tick_stats = tuple(round(float(v), 3) for v in np.percentile(self.periods, [50, 99]))
            if now - next_t > self.period:               # fell a whole tick behind: don't burst to catch up
                next_t = now
            t_prev = now


def make_server(sim, port=8765, llm=None, runner=None):
    """ThreadingHTTPServer on 127.0.0.1. `runner` applies commands to `sim`; `llm` interprets what the parser
    can't (None = parser only). LLM keys stay in this process's environment."""
    runner = runner or SimRunner(sim)
    seq_lock, seq = threading.Lock(), [0]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, status, body=b"", ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self):
            """Only this server's own names: a foreign Host header is DNS rebinding."""
            port = self.server.server_address[1]
            return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b'{"error": "forbidden host"}')
            if self.path in STATIC:
                name, ctype = STATIC[self.path]
                with open(os.path.join(WEB, name), "rb") as f:
                    return self._send(200, f.read(), ctype)
            if self.path == "/stream":
                return self._stream()
            self._send(404, b'{"error": "not found"}')

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            last = None
            try:
                while not runner._stop.is_set():
                    with runner.cond:
                        runner.cond.wait(timeout=1.0)
                        snap = runner.latest
                    if snap is not None and snap is not last:
                        self.wfile.write(b"data: " + json.dumps(snap).encode() + b"\n\n")
                        self.wfile.flush()
                        last = snap
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass                                    # the tab closed; the simulation carries on

        def do_POST(self):
            if not self._host_ok():
                self.close_connection = True
                return self._send(403, b'{"error": "forbidden host"}')
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self.close_connection = True              # other sites can only send simple (non-JSON) POSTs
                return self._send(415, b'{"error": "send application/json"}')
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self.close_connection = True
                return self._send(413, b'{"error": "body too large"}')
            body = self.rfile.read(length)
            if self.path == "/predator":
                runner.do(lambda s: s.launch_predator())
                return self._send(204)
            if self.path == "/reset":
                runner.do(lambda s: s.reset())
                return self._send(204)
            if self.path != "/goal":
                return self._send(404, b'{"error": "not found"}')
            try:
                text = json.loads(body)["text"]
                if not isinstance(text, str):
                    raise ValueError("text must be a string")
                with seq_lock:
                    seq[0] += 1
                    mine = seq[0]
                result = interpret(text, llm)          # slow only when the LLM runs; this thread, not the sim's
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, json.dumps({"error": str(e)}).encode())
            with seq_lock:
                if mine == seq[0]:                       # a newer goal was typed meanwhile: don't override it
                    runner.do(lambda s, g=result.goal: s.set_goal(g))
            g = result.goal
            reply = {"goal": {"seek": list(g.seek), "avoid": list(g.avoid), "heading": g.heading, "rest": g.rest,
                              "text": g.text, "source": g.source},
                     "source": g.source, "ignored": list(result.ignored), "ms": round(result.ms, 1),
                     "note": result.note}
            self._send(200, json.dumps(reply).encode())

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def __init__(self, *a):
            super().__init__(*a)
            self.errors = []

        def handle_error(self, request, client_address):
            e = sys.exc_info()[1]
            if not isinstance(e, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                self.errors.append(repr(e))
                super().handle_error(request, client_address)

    server = Server(("127.0.0.1", port), Handler)
    server.runner = runner
    return server


def main():
    from agent_loop import Sim
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    sim = Sim(device=args.device, wall=True)
    runner = SimRunner(sim)
    llm = make_llm()
    server = make_server(sim, args.port, llm, runner)
    runner.start()
    print(f"flyagent: http://127.0.0.1:{args.port}   (goal interpreter: parser{' + LLM' if llm else ' only'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        runner.stop()
        sim.close()


def selfcheck():
    import http.client
    import json
    import socket
    import threading
    import time

    from agent_loop import Sim

    def slow_llm(text):
        time.sleep(1.0)
        return '{"seek": [], "avoid": [], "heading": null, "rest": true}'

    sim = Sim(wall=True)
    runner = SimRunner(sim)
    runner.start()
    server = make_server(sim, port=0, llm=slow_llm, runner=runner)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def request(method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request(method, path, body, {"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    status, page = request("GET", "/")
    assert status == 200 and b"<html" in page.lower(), status

    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/stream")
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type").startswith("text/event-stream")
    events = []
    while len(events) < 3:
        line = r.fp.readline().decode()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    keys = {"t", "x", "y", "heading", "bump", "probs", "behaviour", "jump", "odor", "home", "loom", "tick_p50",
            "tick_p99"}
    assert all(keys <= set(e) for e in events) and events[0]["t"] < events[-1]["t"], events[0].keys()
    c.close()

    status, body = request("POST", "/goal", json.dumps({"text": "find the banana"}))
    reply = json.loads(body)
    assert status == 200 and reply["source"] == "parser" and reply["goal"]["seek"] == ["banana"], reply
    assert request("POST", "/goal", "x" * 3000)[0] == 413
    assert request("POST", "/goal", json.dumps({"text": "y" * 201}))[0] == 400
    assert request("POST", "/goal", "{not json")[0] == 400
    assert request("GET", "/../goals.py")[0] == 404
    assert request("POST", "/predator")[0] == 204
    # other websites can't drive the local server (final review): a simple cross-origin POST has to use a
    # non-JSON Content-Type, and DNS rebinding arrives with a foreign Host header
    time.sleep(0.1)                                     # let earlier queued commands land first
    goal_before = sim.goal
    assert request("POST", "/goal", json.dumps({"text": "go home"}), {"Content-Type": "text/plain"})[0] == 415
    assert request("POST", "/reset", None, {"Content-Type": "text/plain"})[0] == 415
    assert request("GET", "/stream", None, {"Host": "evil.example"})[0] == 403
    assert request("POST", "/goal", json.dumps({"text": "go home"}), {"Host": "evil.example:8765"})[0] == 403
    assert request("GET", "/", None, {"Host": f"localhost:{port}"})[0] == 200
    time.sleep(0.1)
    assert sim.goal == goal_before, sim.goal

    replies = {}                                        # last_goal_wins: a slow LLM answer must not override
    first = threading.Thread(target=lambda: replies.update(a=request("POST", "/goal",
                                                                      json.dumps({"text": "rest, I'm sleepy"}))))
    first.start()
    time.sleep(0.1)
    replies["b"] = request("POST", "/goal", json.dumps({"text": "go home"}))
    first.join()
    assert json.loads(replies["a"][1])["source"] == "llm" and sim.goal.seek == ("home",), (replies, sim.goal)

    s = socket.create_connection(("127.0.0.1", port))   # client_disconnect: tab closed mid-stream
    s.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
    s.recv(4096)
    s.close()
    ticks = sim.tick
    time.sleep(0.3)
    assert sim.tick > ticks and server.errors == [], (sim.tick, ticks, server.errors)

    assert request("POST", "/reset")[0] == 204
    server.shutdown()
    runner.stop()
    print("serve self-check OK")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
