from __future__ import annotations

import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from observables import calcular_todos_observables
from simulador import simulate

HOST = "127.0.0.1"
PORT = 8000
FRAME_COUNT = 400
STABILIZATION_FRAMES = 50
DT = 0.05
ROOT = Path(__file__).resolve().parent


class SimulationState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.generation = 0
        self.params: dict[str, float | int] = {
            "v0": 1.8,
            "u0": 0.9,
            "sigma": 0.35,
            "alignment": 1.25,
            "nodes": 84,
            "canvasSize": 100,
        }
        self.equations: dict[str, bool] = {
            "propulsion": True,
            "repulsion": True,
            "alignment": True,
            "noise": True,
        }
        self.subscribers: set[queue.Queue[str]] = set()

    def snapshot(self) -> tuple[int, dict[str, float | int], set[str]]:
        with self.lock:
            return (
                self.generation,
                dict(self.params),
                {kind for kind, enabled in self.equations.items() if enabled},
            )

    def subscribe(self) -> queue.Queue[str]:
        subscriber: queue.Queue[str] = queue.Queue(maxsize=8)
        with self.lock:
            self.subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[str]) -> None:
        with self.lock:
            self.subscribers.discard(subscriber)

    def publish(self, payload: dict[str, Any]) -> None:
        message = f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        with self.lock:
            subscribers = list(self.subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(message)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(message)
                except queue.Empty:
                    pass

    def update(self, body: dict[str, Any]) -> None:
        with self.lock:
            for key in ("v0", "u0", "sigma", "alignment"):
                value = body.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.params[key] = float(value)
            for key, minimum, maximum in (("nodes", 8, 300), ("canvasSize", 40, 300)):
                value = body.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.params[key] = int(max(minimum, min(maximum, round(value))))
            equations = body.get("equations")
            if isinstance(equations, list):
                self.equations = {
                    kind: any(item.get("kind") == kind and item.get("enabled", True) for item in equations if isinstance(item, dict))
                    for kind in self.equations
                }
            if body.get("command") == "reset" or body:
                self.generation += 1
                self.wake.set()


state = SimulationState()


def histogram(values: np.ndarray, bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    low = float(np.min(values)) if values.size else -1.0
    high = float(np.max(values)) if values.size else 1.0
    if math.isclose(low, high):
        low -= 0.5
        high += 0.5
    counts, edges = np.histogram(values, bins=bins, range=(low, high), density=True)
    return (edges[:-1] + edges[1:]) / 2, counts


def make_observables(result: dict[str, Any]) -> dict[str, Any]:
    observables = calcular_todos_observables(result)
    time_axis = observables["tiempo"]
    vx_axis, vx_density = histogram(observables["distribucion_vel"]["vx"])
    vy_axis, vy_density = histogram(observables["distribucion_vel"]["vy"])
    velocity_axis = np.linspace(min(vx_axis[0], vy_axis[0]), max(vx_axis[-1], vy_axis[-1]), 30)
    vx_interp = np.interp(velocity_axis, vx_axis, vx_density, left=0, right=0)
    vy_interp = np.interp(velocity_axis, vy_axis, vy_density, left=0, right=0)
    half = len(time_axis) // 2
    return {
        "msd": [[float(x), float(y)] for x, y in zip(time_axis, observables["msd"])],
        "velocityDistribution": [[float(x), float(x_density), float(y_density)] for x, x_density, y_density in zip(velocity_axis, vx_interp, vy_interp)],
        "temporalCorrelation": [[float(x), float(vx), float(vy)] for x, vx, vy in zip(time_axis[:half], observables["autocor_vx"][:half], observables["autocor_vy"][:half])],
        "spatialCorrelation": [[float(x), float(xx), float(yy)] for x, xx, yy in zip(observables["radios"], observables["cxx_r"], observables["cyy_r"])],
        "pairDistribution": [[float(x), float(y)] for x, y in zip(observables["radios"], observables["gr"])],
    }


def build_frame(result: dict[str, Any], observables: dict[str, Any], index: int, params: dict[str, float | int], include_observables: bool) -> dict[str, Any]:
    vx = result["velx"][index]
    vy = result["vely"][index]
    particles = [
        [float(x), float(y), float(cell_vx), float(cell_vy), float(math.atan2(cell_vy, cell_vx))]
        for x, y, cell_vx, cell_vy in zip(result["posx_caja"][index], result["posy_caja"][index], vx, vy)
    ]
    payload: dict[str, Any] = {
        "type": "telemetry",
        "timestamp": int(time.time() * 1000),
        "step": index,
        "particles": particles,
        "metrics": {
            "msd": float(observables["msd"][index][1]),
            "autocorrelation": float(observables["temporalCorrelation"][min(index, len(observables["temporalCorrelation"]) - 1)][1]) if observables["temporalCorrelation"] else 0.0,
            "meanSpeed": float(np.mean(np.hypot(vx, vy))),
            "density": int(params["nodes"]) / float(params["canvasSize"]) ** 2,
        },
        "optimizer": {"iteration": index, "tolerance": 0.0001, "minimumError": 0.0, "status": "RUNNING"},
        "simulation": {"nodes": int(params["nodes"]), "canvasSize": int(params["canvasSize"])},
    }
    if include_observables:
        payload["observables"] = observables
    return payload


def simulation_worker() -> None:
    while True:
        generation, params, active_equations = state.snapshot()
        result = simulate(
            [params["v0"], params["u0"], params["alignment"], params["sigma"]],
            seed=42 + generation,
            n_pasos=FRAME_COUNT,
            n_estabilizar=STABILIZATION_FRAMES,
            num_celulas=int(params["nodes"]),
            dt=DT,
            limx=float(params["canvasSize"]),
            limy=float(params["canvasSize"]),
            ecuaciones_activas=active_equations,
        )
        observables = make_observables(result)
        for index in range(FRAME_COUNT):
            current_generation, _, _ = state.snapshot()
            if current_generation != generation:
                break
            state.publish(build_frame(result, observables, index, params, True))
            if state.wake.wait(DT):
                state.wake.clear()
                break


class RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        if self.path == "/health":
            with state.lock:
                clients = len(state.subscribers)
                generation = state.generation
            self.send_json({"status": "ok", "clients": clients, "generation": generation})
            return
        if self.path == "/events":
            subscriber = state.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(subscriber.get(timeout=15).encode("utf-8"))
                    self.wfile.flush()
            except (queue.Empty, BrokenPipeError, ConnectionResetError):
                pass
            finally:
                state.unsubscribe(subscriber)
            return
        try:
            content = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except OSError:
            self.send_json({"error": "index.html not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_json({"error": "invalid JSON"}, 400)
            return
        state.update(body if isinstance(body, dict) else {})
        self.send_json({"status": "ok"})

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    threading.Thread(target=simulation_worker, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"Simulación células a contraflujo listening on http://localhost:{PORT}")
    server.serve_forever()
