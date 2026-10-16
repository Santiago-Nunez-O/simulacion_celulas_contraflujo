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
        self.optimization_cancel = threading.Event()
        self.generation = 0
        self.optimization_run_id = 0
        self.optimizer: dict[str, Any] = {
            "status": "IDLE",
            "iteration": 0,
            "message": "Listo para optimizar.",
        }
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

    def optimizer_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.optimizer)

    def start_optimization(self, config: dict[str, Any]) -> bool:
        with self.lock:
            if self.optimizer["status"] == "OPTIMIZING":
                return False
            self.optimization_run_id += 1
            run_id = self.optimization_run_id
            generation = self.generation
            params = dict(self.params)
            equations = {kind for kind, enabled in self.equations.items() if enabled}
            self.optimization_cancel.clear()
            self.optimizer = {
                "status": "OPTIMIZING",
                "iteration": 0,
                "message": "Calculando la referencia y evaluando parámetros...",
            }
        self.publish({"type": "optimizer", "optimizer": self.optimizer_snapshot()})
        threading.Thread(
            target=run_parameter_optimization,
            args=(run_id, generation, params, equations, config),
            daemon=True,
        ).start()
        return True

    def update_optimizer(self, run_id: int, update: dict[str, Any]) -> bool:
        with self.lock:
            if run_id != self.optimization_run_id or self.optimizer["status"] != "OPTIMIZING":
                return False
            self.optimizer.update(update)
            payload = dict(self.optimizer)
        self.publish({"type": "optimizer", "optimizer": payload})
        return True

    def finish_optimization(self, run_id: int, generation: int, result: dict[str, Any]) -> None:
        with self.lock:
            if run_id != self.optimization_run_id or self.optimizer["status"] != "OPTIMIZING":
                return
            if generation != self.generation or self.optimization_cancel.is_set():
                self.optimizer.update({"status": "CANCELLED", "message": "Optimización cancelada por un cambio de parámetros."})
            else:
                self.params.update(result["params"])
                self.generation += 1
                self.wake.set()
                self.optimizer.update({
                    "status": "COMPLETE",
                    "iteration": result["iterations"],
                    "error": result["error"],
                    "params": result["params"],
                    "message": "Optimización completa; los parámetros hallados ya están aplicados.",
                })
            payload = dict(self.optimizer)
        self.publish({"type": "optimizer", "optimizer": payload})

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
        optimizer_update = None
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
                if self.optimizer["status"] == "OPTIMIZING":
                    self.optimization_cancel.set()
                    self.optimizer.update({
                        "status": "CANCELLED",
                        "message": "Optimización cancelada porque cambió la configuración.",
                    })
                    optimizer_update = dict(self.optimizer)
        if optimizer_update:
            self.publish({"type": "optimizer", "optimizer": optimizer_update})


state = SimulationState()


def coordinate_descent(
    initial: list[float],
    objective: Any,
    on_progress: Any = None,
    max_iterations: int = 14,
    tolerance: float = 0.01,
    optimized_indices: list[int] | None = None,
) -> tuple[list[float], float, int]:
    point = np.clip(np.asarray(initial, dtype=float), 0.01, 5.0)
    axes = list(range(len(point))) if optimized_indices is None else optimized_indices
    if not axes:
        raise ValueError("Select at least one parameter to optimize.")
    best_error = float(objective(point.tolist()))
    step = 0.5

    for iteration in range(1, max_iterations + 1):
        improved = False
        for axis in axes:
            for direction in (-1.0, 1.0):
                candidate = point.copy()
                candidate[axis] = np.clip(candidate[axis] + direction * step, 0.01, 5.0)
                if candidate[axis] == point[axis]:
                    continue
                error = float(objective(candidate.tolist()))
                if error < best_error:
                    point = candidate
                    best_error = error
                    improved = True
        if on_progress:
            on_progress(iteration, best_error, point.tolist())
        if best_error <= 1e-12:
            return point.tolist(), best_error, iteration
        if not improved:
            step *= 0.5
        if step < tolerance:
            return point.tolist(), best_error, iteration
    return point.tolist(), best_error, max_iterations


def run_parameter_optimization(
    run_id: int,
    generation: int,
    params: dict[str, float | int],
    equations: set[str],
    config: dict[str, Any],
) -> None:
    node_count = min(int(params["nodes"]), 40)
    simulation_options = {
        "seed": 2718,
        "n_pasos": 50,
        "n_estabilizar": 25,
        "num_celulas": node_count,
        "dt": DT,
        "limx": float(params["canvasSize"]),
        "limy": float(params["canvasSize"]),
        "ecuaciones_activas": equations,
    }

    reference_parameters = config["reference"]
    reference = calcular_todos_observables(simulate(
        [
            reference_parameters["v0"],
            reference_parameters["u0"],
            reference_parameters["alignment"],
            reference_parameters["sigma"],
        ],
        **simulation_options,
    ))
    observable_keys = ("msd", "autocor_prom", "c_prom", "gr")
    reference_values = [np.asarray(reference[key], dtype=float) for key in observable_keys]
    scales = [max(float(np.mean(values * values)), 1e-12) for values in reference_values]
    evaluations = 0

    def objective(values: list[float]) -> float:
        nonlocal evaluations
        if state.optimization_cancel.is_set():
            raise InterruptedError("Optimización cancelada")
        result = simulate(
            [values[0], values[1], values[3], values[2]],
            **simulation_options,
        )
        observables = calcular_todos_observables(result)
        errors = []
        for key, target, scale in zip(observable_keys, reference_values, scales):
            actual = np.asarray(observables[key], dtype=float)
            errors.append(float(np.mean((actual - target) ** 2) / scale))
        evaluations += 1
        return float(np.mean(errors))

    def report_progress(iteration: int, error: float, values: list[float]) -> None:
        state.update_optimizer(run_id, {
            "iteration": iteration,
            "error": error,
            "evaluations": evaluations,
            "params": {
                "v0": round(values[0], 3),
                "u0": round(values[1], 3),
                "sigma": round(values[2], 3),
                "alignment": round(values[3], 3),
            },
            "message": f"Iteración {iteration}: error normalizado {error:.5g}.",
        })

    try:
        initial_parameters = config["initial"]
        initial = [
            float(initial_parameters["v0"]),
            float(initial_parameters["u0"]),
            float(initial_parameters["sigma"]),
            float(initial_parameters["alignment"]),
        ]
        optimized_indices = [
            index for index, key in enumerate(("v0", "u0", "sigma", "alignment"))
            if key in config["optimize"]
        ]
        values, error, iterations = coordinate_descent(
            initial,
            objective,
            report_progress,
            optimized_indices=optimized_indices,
        )
        state.finish_optimization(run_id, generation, {
            "params": {
                "v0": round(values[0], 3),
                "u0": round(values[1], 3),
                "sigma": round(values[2], 3),
                "alignment": round(values[3], 3),
            },
            "error": error,
            "iterations": iterations,
        })
    except InterruptedError:
        state.update_optimizer(run_id, {"status": "CANCELLED", "message": "Optimización cancelada."})
    except (ValueError, FloatingPointError, KeyError, RuntimeError) as error:
        state.update_optimizer(run_id, {"status": "ERROR", "message": f"Falló la optimización: {error}"})


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


def build_frame(
    result: dict[str, Any],
    observables: dict[str, Any],
    index: int,
    step_offset: int,
    params: dict[str, float | int],
    include_observables: bool,
) -> dict[str, Any]:
    vx = result["velx"][index]
    vy = result["vely"][index]
    step = step_offset + index
    particles = [
        [float(x), float(y), float(cell_vx), float(cell_vy), float(math.atan2(cell_vy, cell_vx))]
        for x, y, cell_vx, cell_vy in zip(result["posx_caja"][index], result["posy_caja"][index], vx, vy)
    ]
    payload: dict[str, Any] = {
        "type": "telemetry",
        "timestamp": int(time.time() * 1000),
        "step": step,
        "timeSeconds": round(step * DT, 2),
        "particles": particles,
        "metrics": {
            "msd": float(observables["msd"][index][1]),
            "autocorrelation": float(observables["temporalCorrelation"][min(index, len(observables["temporalCorrelation"]) - 1)][1]) if observables["temporalCorrelation"] else 0.0,
            "meanSpeed": float(np.mean(np.hypot(vx, vy))),
            "density": int(params["nodes"]) / float(params["canvasSize"]) ** 2,
        },
        "optimizer": state.optimizer_snapshot(),
        "simulation": {"nodes": int(params["nodes"]), "canvasSize": int(params["canvasSize"]), "dt": DT},
    }
    if include_observables:
        payload["observables"] = observables
    return payload


def simulation_worker() -> None:
    while True:
        generation, params, active_equations = state.snapshot()
        initial_state = None
        step_offset = 0
        while True:
            current_generation, _, _ = state.snapshot()
            if current_generation != generation:
                break
            result = simulate(
                [params["v0"], params["u0"], params["alignment"], params["sigma"]],
                seed=42 + generation,
                n_pasos=FRAME_COUNT,
                n_estabilizar=STABILIZATION_FRAMES if initial_state is None else 0,
                num_celulas=int(params["nodes"]),
                dt=DT,
                limx=float(params["canvasSize"]),
                limy=float(params["canvasSize"]),
                ecuaciones_activas=active_equations,
                initial_state=initial_state,
            )
            initial_state = result["final_state"]
            observables = make_observables(result)
            for index in range(FRAME_COUNT):
                current_generation, _, _ = state.snapshot()
                if current_generation != generation:
                    break
                state.publish(
                    build_frame(
                        result,
                        observables,
                        index,
                        step_offset,
                        params,
                        index == 0,
                    )
                )
                if state.wake.wait(DT):
                    state.wake.clear()
                    break
            else:
                step_offset += FRAME_COUNT
                continue
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
        if isinstance(body, dict) and body.get("command") == "optimize":
            config = body.get("optimizer")
            parameter_keys = ("v0", "u0", "sigma", "alignment")
            if not isinstance(config, dict):
                self.send_json({"error": "optimizer configuration is required"}, 400)
                return
            initial = config.get("initial")
            reference = config.get("reference")
            optimize = config.get("optimize")
            if (
                not isinstance(initial, dict)
                or not isinstance(reference, dict)
                or not isinstance(optimize, list)
                or not optimize
                or any(key not in parameter_keys for key in optimize)
            ):
                self.send_json({"error": "invalid optimizer configuration"}, 400)
                return
            normalized_config: dict[str, Any] = {
                "initial": {},
                "reference": {},
                "optimize": list(dict.fromkeys(optimize)),
            }
            for key in parameter_keys:
                for source, minimum in ((initial, 0.01), (reference, 0.0)):
                    value = source.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < minimum
                        or value > 5.0
                    ):
                        self.send_json({"error": f"invalid {key} in optimizer configuration"}, 400)
                        return
                normalized_config["initial"][key] = float(initial[key])
                normalized_config["reference"][key] = float(reference[key])
            if not state.start_optimization(normalized_config):
                self.send_json({"error": "optimization already running"}, 409)
                return
            self.send_json({"status": "accepted"})
            return
        state.update(body if isinstance(body, dict) else {})
        self.send_json({"status": "ok"})

    def log_message(self, format: str, *args: Any) -> None:
        return


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        exception_type = __import__("sys").exc_info()[0]
        if exception_type in (BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)


if __name__ == "__main__":
    threading.Thread(target=simulation_worker, daemon=True).start()
    server = QuietThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"Simulación células a contraflujo listening on http://localhost:{PORT}")
    server.serve_forever()
