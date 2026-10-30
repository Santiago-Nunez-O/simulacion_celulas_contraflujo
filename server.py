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

    def cancel_optimization(self) -> bool:
        with self.lock:
            if self.optimizer.get("status") == "OPTIMIZING":
                self.optimization_cancel.set()
                self.optimizer.update({
                    "status": "CANCELLED",
                    "message": "Optimización cancelada por el usuario.",
                })
                payload = dict(self.optimizer)
                self.publish({"type": "optimizer", "optimizer": payload})
                return True
            return False

    def finish_optimization(self, run_id: int, generation: int, result: dict[str, Any]) -> None:
        with self.lock:
            if run_id != self.optimization_run_id or self.optimizer["status"] != "OPTIMIZING":
                return
            if generation != self.generation or self.optimization_cancel.is_set():
                self.optimizer.update({"status": "CANCELLED", "message": "Optimización cancelada por el usuario o cambio de parámetros."})
            else:
                self.params.update(result["params"])
                self.generation += 1
                self.wake.set()
                self.optimizer.update({
                    "status": "COMPLETE",
                    "method": result.get("method"),
                    "methodName": result.get("methodName"),
                    "iteration": result["iterations"],
                    "error": result["error"],
                    "params": result["params"],
                    "message": f"Optimización con {result.get('methodName', 'éxito')} finalizada; los parámetros hallados ya están aplicados.",
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


def nelder_mead_simplex(
    initial: list[float],
    objective: Any,
    on_progress: Any = None,
    max_iterations: int = 40,
    tolerance: float = 0.005,
    optimized_indices: list[int] | None = None,
) -> tuple[list[float], float, int]:
    point = np.clip(np.asarray(initial, dtype=float), 0.01, 5.0)
    axes = list(range(len(point))) if optimized_indices is None else list(optimized_indices)
    k = len(axes)
    if k == 0:
        raise ValueError("Select at least one parameter to optimize.")

    def eval_reduced(sub_x: np.ndarray) -> tuple[np.ndarray, float]:
        full = point.copy()
        for idx, ax in enumerate(axes):
            full[ax] = sub_x[idx]
        return full, float(objective(full.tolist()))

    x0 = np.array([point[ax] for ax in axes], dtype=float)
    vertices = [x0.copy()]
    full_pt, y0 = eval_reduced(x0)
    scores = [y0]

    for i in range(k):
        v = x0.copy()
        step = 0.25 if v[i] + 0.25 <= 5.0 else -0.25
        v[i] = np.clip(v[i] + step, 0.01, 5.0)
        _, y = eval_reduced(v)
        vertices.append(v)
        scores.append(y)

    vertices = np.array(vertices, dtype=float)
    scores = np.array(scores, dtype=float)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    best_full = point.copy()
    best_score = float(np.min(scores))

    for iteration in range(1, max_iterations + 1):
        order = np.argsort(scores)
        vertices = vertices[order]
        scores = scores[order]

        ilo = 0
        ihi = -1
        inhi = -2

        best_score = float(scores[ilo])
        best_full, _ = eval_reduced(vertices[ilo])

        if on_progress:
            on_progress(iteration, best_score, best_full.tolist())

        rtol = 2.0 * abs(scores[ihi] - scores[ilo]) / (abs(scores[ihi]) + abs(scores[ilo]) + 1e-10)
        if rtol < tolerance or best_score < 1e-9:
            return best_full.tolist(), best_score, iteration

        x_bar = np.mean(vertices[:-1], axis=0)

        xr = np.clip(x_bar + alpha * (x_bar - vertices[ihi]), 0.01, 5.0)
        _, yr = eval_reduced(xr)

        if yr < scores[ilo]:
            xe = np.clip(x_bar + gamma * (xr - x_bar), 0.01, 5.0)
            _, ye = eval_reduced(xe)
            if ye < yr:
                vertices[ihi] = xe
                scores[ihi] = ye
            else:
                vertices[ihi] = xr
                scores[ihi] = yr
        elif yr < scores[inhi]:
            vertices[ihi] = xr
            scores[ihi] = yr
        else:
            if yr < scores[ihi]:
                xc = np.clip(x_bar + rho * (xr - x_bar), 0.01, 5.0)
            else:
                xc = np.clip(x_bar - rho * (x_bar - vertices[ihi]), 0.01, 5.0)
            _, yc = eval_reduced(xc)
            if yc < min(yr, scores[ihi]):
                vertices[ihi] = xc
                scores[ihi] = yc
            else:
                for j in range(1, len(vertices)):
                    vertices[j] = np.clip(vertices[ilo] + sigma * (vertices[j] - vertices[ilo]), 0.01, 5.0)
                    _, scores[j] = eval_reduced(vertices[j])

    order = np.argsort(scores)
    best_full, _ = eval_reduced(vertices[order[0]])
    return best_full.tolist(), float(scores[order[0]]), max_iterations


def simplex_annealing(
    initial: list[float],
    objective: Any,
    on_progress: Any = None,
    max_iterations: int = 40,
    tolerance: float = 0.005,
    optimized_indices: list[int] | None = None,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.92,
) -> tuple[list[float], float, int]:
    point = np.clip(np.asarray(initial, dtype=float), 0.01, 5.0)
    axes = list(range(len(point))) if optimized_indices is None else list(optimized_indices)
    k = len(axes)
    if k == 0:
        raise ValueError("Select at least one parameter to optimize.")

    def eval_reduced(sub_x: np.ndarray) -> tuple[np.ndarray, float]:
        full = point.copy()
        for idx, ax in enumerate(axes):
            full[ax] = sub_x[idx]
        return full, float(objective(full.tolist()))

    x0 = np.array([point[ax] for ax in axes], dtype=float)
    vertices = [x0.copy()]
    full_pt, y0 = eval_reduced(x0)
    scores = [y0]

    for i in range(k):
        v = x0.copy()
        step = 0.25 if v[i] + 0.25 <= 5.0 else -0.25
        v[i] = np.clip(v[i] + step, 0.01, 5.0)
        _, y = eval_reduced(v)
        vertices.append(v)
        scores.append(y)

    vertices = np.array(vertices, dtype=float)
    scores = np.array(scores, dtype=float)
    psum = np.sum(vertices, axis=0)
    T = float(initial_temperature)

    best_error_hist = float(np.min(scores))
    best_vert_hist = vertices[np.argmin(scores)].copy()

    def amotry(peor_vert: int, factor: float) -> float:
        nonlocal psum, best_error_hist, best_vert_hist
        factor1 = (1.0 - factor) / k
        factor2 = factor1 - factor
        v_try = np.clip(psum * factor1 - vertices[peor_vert] * factor2, 0.01, 5.0)
        _, res_try = eval_reduced(v_try)
        fluct = -T * np.log(max(float(np.random.rand()), 1e-12))
        res_try_ter = res_try - fluct
        if res_try_ter < scores[peor_vert]:
            scores[peor_vert] = res_try
            psum += v_try - vertices[peor_vert]
            vertices[peor_vert] = v_try
            if res_try < best_error_hist:
                best_error_hist = res_try
                best_vert_hist = v_try.copy()
        return res_try_ter

    for iteration in range(1, max_iterations + 1):
        y_ter = scores.copy()
        for j in range(k + 1):
            y_ter[j] += -T * np.log(max(float(np.random.rand()), 1e-12))

        peor_vert = int(np.argmax(y_ter))
        mejor_vert = int(np.argmin(y_ter))

        copia_y_ter = y_ter.copy()
        copia_y_ter[peor_vert] = -np.inf
        peor2_vert = int(np.argmax(copia_y_ter))

        if scores[mejor_vert] < best_error_hist:
            best_error_hist = float(scores[mejor_vert])
            best_vert_hist = vertices[mejor_vert].copy()

        full_best, _ = eval_reduced(best_vert_hist)
        if on_progress:
            on_progress(iteration, best_error_hist, full_best.tolist())

        rtol = 2.0 * abs(scores[peor_vert] - scores[mejor_vert]) / (abs(scores[peor_vert]) + abs(scores[mejor_vert]) + 1e-10)
        if rtol < tolerance and T < 1e-3:
            return full_best.tolist(), best_error_hist, iteration

        res_ytry_ter = amotry(peor_vert, -1.0)
        if res_ytry_ter <= y_ter[mejor_vert]:
            res_ytry_ter = amotry(peor_vert, 2.0)
        elif res_ytry_ter >= y_ter[peor2_vert]:
            y_save = y_ter[peor_vert]
            res_ytry_ter = amotry(peor_vert, 0.5)
            if res_ytry_ter >= y_save:
                for j in range(k + 1):
                    if j != mejor_vert:
                        vertices[j] = np.clip(0.5 * (vertices[j] + vertices[mejor_vert]), 0.01, 5.0)
                        _, scores[j] = eval_reduced(vertices[j])
                psum = np.sum(vertices, axis=0)
        T *= cooling_rate

    full_best, _ = eval_reduced(best_vert_hist)
    return full_best.tolist(), best_error_hist, max_iterations


def scipy_nelder_mead(
    initial: list[float],
    objective: Any,
    on_progress: Any = None,
    max_iterations: int = 40,
    tolerance: float = 0.005,
    optimized_indices: list[int] | None = None,
) -> tuple[list[float], float, int]:
    try:
        from scipy.optimize import minimize
    except ImportError:
        return nelder_mead_simplex(
            initial, objective, on_progress, max_iterations, tolerance, optimized_indices
        )

    point = np.clip(np.asarray(initial, dtype=float), 0.01, 5.0)
    axes = list(range(len(point))) if optimized_indices is None else list(optimized_indices)
    if not axes:
        raise ValueError("Select at least one parameter to optimize.")

    def sub_obj(sub_x: np.ndarray) -> float:
        full = point.copy()
        for idx, ax in enumerate(axes):
            full[ax] = sub_x[idx]
        return float(objective(full.tolist()))

    x0 = np.array([point[ax] for ax in axes], dtype=float)
    bounds = [(0.01, 5.0) for _ in axes]
    iteration = 0

    def callback(xk: np.ndarray) -> None:
        nonlocal iteration
        iteration += 1
        full = point.copy()
        for idx, ax in enumerate(axes):
            full[ax] = xk[idx]
        err = float(objective(full.tolist()))
        if on_progress:
            on_progress(iteration, err, full.tolist())

    res = minimize(
        sub_obj,
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        callback=callback,
        options={"maxiter": max_iterations, "xatol": tolerance, "fatol": tolerance},
    )
    full_best = point.copy()
    for idx, ax in enumerate(axes):
        full_best[ax] = float(np.clip(res.x[idx], 0.01, 5.0))
    final_err = float(objective(full_best.tolist()))
    return full_best.tolist(), final_err, int(getattr(res, "nit", iteration)) or 1


def scipy_dual_annealing(
    initial: list[float],
    objective: Any,
    on_progress: Any = None,
    max_iterations: int = 30,
    tolerance: float = 0.005,
    optimized_indices: list[int] | None = None,
) -> tuple[list[float], float, int]:
    try:
        from scipy.optimize import dual_annealing
    except ImportError:
        return simplex_annealing(
            initial, objective, on_progress, max_iterations, tolerance, optimized_indices
        )

    point = np.clip(np.asarray(initial, dtype=float), 0.01, 5.0)
    axes = list(range(len(point))) if optimized_indices is None else list(optimized_indices)
    if not axes:
        raise ValueError("Select at least one parameter to optimize.")

    def sub_obj(sub_x: np.ndarray) -> float:
        full = point.copy()
        for idx, ax in enumerate(axes):
            full[ax] = sub_x[idx]
        return float(objective(full.tolist()))

    x0 = np.array([point[ax] for ax in axes], dtype=float)
    bounds = [(0.01, 5.0) for _ in axes]
    iteration = 0

    def callback(x: np.ndarray, f: float, context: int) -> bool:
        nonlocal iteration
        iteration += 1
        full = point.copy()
        for idx, ax in enumerate(axes):
            full[ax] = x[idx]
        if on_progress:
            on_progress(iteration, float(f), full.tolist())
        return False

    res = dual_annealing(
        sub_obj,
        bounds=bounds,
        x0=x0,
        maxiter=max_iterations,
        callback=callback,
    )
    full_best = point.copy()
    for idx, ax in enumerate(axes):
        full_best[ax] = float(np.clip(res.x[idx], 0.01, 5.0))
    final_err = float(objective(full_best.tolist()))
    return full_best.tolist(), final_err, int(getattr(res, "nit", iteration)) or 1


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

    method = str(config.get("method", "coordinate_descent"))
    max_iter = int(config.get("max_iterations", 40))
    tolerance = float(config.get("tolerance", 0.01))
    method_name = {
        "coordinate_descent": "Descenso por Coordenadas",
        "simplex": "Downhill Simplex (Nelder-Mead)",
        "nelder_mead": "Downhill Simplex (Nelder-Mead)",
        "annealing": "Recocido Simulado (Simulated Annealing)",
        "simulated_annealing": "Recocido Simulado (Simulated Annealing)",
        "scipy_simplex": "Scipy Nelder-Mead",
        "scipy_annealing": "Scipy Dual Annealing",
    }.get(method, method)

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
            "method": method,
            "methodName": method_name,
            "iteration": iteration,
            "maxIterations": max_iter,
            "error": error,
            "evaluations": evaluations,
            "params": {
                "v0": round(values[0], 3),
                "u0": round(values[1], 3),
                "sigma": round(values[2], 3),
                "alignment": round(values[3], 3),
            },
            "message": f"[{method_name}] Iteración {iteration}: error normalizado {error:.5g} ({evaluations} evaluaciones).",
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

        if method in ("simplex", "nelder_mead"):
            values, error, iterations = nelder_mead_simplex(
                initial,
                objective,
                report_progress,
                max_iterations=max_iter,
                tolerance=tolerance,
                optimized_indices=optimized_indices,
            )
        elif method in ("annealing", "simulated_annealing"):
            values, error, iterations = simplex_annealing(
                initial,
                objective,
                report_progress,
                max_iterations=max_iter,
                tolerance=tolerance,
                optimized_indices=optimized_indices,
                initial_temperature=float(config.get("temperature", 1.0)),
                cooling_rate=float(config.get("cooling", 0.92)),
            )
        elif method == "scipy_simplex":
            values, error, iterations = scipy_nelder_mead(
                initial,
                objective,
                report_progress,
                max_iterations=max_iter,
                tolerance=tolerance,
                optimized_indices=optimized_indices,
            )
        elif method == "scipy_annealing":
            values, error, iterations = scipy_dual_annealing(
                initial,
                objective,
                report_progress,
                max_iterations=max_iter,
                tolerance=tolerance,
                optimized_indices=optimized_indices,
            )
        else:
            values, error, iterations = coordinate_descent(
                initial,
                objective,
                report_progress,
                max_iterations=max_iter,
                tolerance=tolerance,
                optimized_indices=optimized_indices,
            )

        state.finish_optimization(run_id, generation, {
            "method": method,
            "methodName": method_name,
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
        if isinstance(body, dict) and body.get("command") == "cancel_optimization":
            state.cancel_optimization()
            self.send_json({"status": "cancelled"})
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
                "method": str(config.get("method", "coordinate_descent")),
                "max_iterations": int(config.get("max_iterations", 40)),
                "tolerance": float(config.get("tolerance", 0.01)),
                "temperature": float(config.get("temperature", 1.0)),
                "cooling": float(config.get("cooling", 0.92)),
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
