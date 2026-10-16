import time
import numpy as np
from simulador import simulate
from observables import calcular_todos_observables
from server import coordinate_descent


def test_simulador():
    theta_base = [1.0, 1.0, 1.0, 1.0]
    seed_1 = 42

    t0 = time.time()
    res1 = simulate(theta_base, seed=seed_1, n_pasos=100, n_estabilizar=50, num_celulas=50)
    t_sim = time.time() - t0
    print(f"Simulacion completada en {t_sim:.3f} s. Shape posx: {res1['posx_real'].shape}")

    res1_bis = simulate(theta_base, seed=seed_1, n_pasos=100, n_estabilizar=50, num_celulas=50)
    np.testing.assert_allclose(res1["posx_real"], res1_bis["posx_real"])
    np.testing.assert_allclose(res1["velx"], res1_bis["velx"])

    res2 = simulate(theta_base, seed=999, n_pasos=100, n_estabilizar=50, num_celulas=50)
    diff = np.max(np.abs(res1["posx_real"] - res2["posx_real"]))
    assert diff > 1e-3

    t0_obs = time.time()
    obs = calcular_todos_observables(res1)
    t_obs = time.time() - t0_obs
    print(f"Observables calculados en {t_obs:.3f} s.")


def test_continuous_simulation_and_edge_parameters():
    parameters = [1.0, 0.0, 0.0, 0.0]
    options = {
        "seed": 17,
        "n_estabilizar": 0,
        "num_celulas": 8,
        "ecuaciones_activas": {"propulsion"},
    }
    full = simulate(parameters, n_pasos=12, **options)
    first = simulate(parameters, n_pasos=5, **options)
    second = simulate(
        parameters,
        n_pasos=7,
        initial_state=first["final_state"],
        **options,
    )
    np.testing.assert_allclose(
        np.concatenate((first["posx_real"], second["posx_real"])),
        full["posx_real"],
    )
    np.testing.assert_allclose(
        np.concatenate((first["posy_real"], second["posy_real"])),
        full["posy_real"],
    )
    speeds = np.hypot(full["velx"], full["vely"])
    np.testing.assert_allclose(speeds, 1.0, atol=1e-12)

    stationary = simulate(
        [0.0, 0.0, 0.0, 0.0],
        seed=17,
        n_pasos=12,
        n_estabilizar=0,
        num_celulas=8,
        ecuaciones_activas={"propulsion", "repulsion", "alignment", "noise"},
    )
    np.testing.assert_allclose(stationary["velx"], 0.0)
    np.testing.assert_allclose(stationary["vely"], 0.0)
    expected_x = np.repeat(stationary["posx_real"][:1], 12, axis=0)
    expected_y = np.repeat(stationary["posy_real"][:1], 12, axis=0)
    np.testing.assert_allclose(stationary["posx_real"], expected_x)
    np.testing.assert_allclose(stationary["posy_real"], expected_y)

    no_equations = simulate(
        [1.0, 1.0, 1.0, 1.0],
        seed=17,
        n_pasos=12,
        n_estabilizar=0,
        num_celulas=8,
        ecuaciones_activas=set(),
    )
    np.testing.assert_allclose(no_equations["velx"], 0.0)
    np.testing.assert_allclose(no_equations["vely"], 0.0)


def test_coordinate_descent_fits_bounded_objective():
    target = np.array([2.0, 3.0, 1.0, 0.5])
    result, error, iterations = coordinate_descent(
        [4.5, 0.3, 4.0, 3.5],
        lambda values: float(np.sum((np.asarray(values) - target) ** 2)),
    )
    np.testing.assert_allclose(result, target, atol=0.01)
    assert error < 1e-4
    assert 1 <= iterations <= 14
    fixed_values, _, _ = coordinate_descent(
        [4.5, 3.0, 4.0, 3.5],
        lambda values: float(np.sum((np.asarray(values) - target) ** 2)),
        optimized_indices=[0],
    )
    np.testing.assert_allclose(fixed_values[1:], [3.0, 4.0, 3.5])
    assert abs(fixed_values[0] - target[0]) < 0.01


if __name__ == "__main__":
    test_simulador()
    test_continuous_simulation_and_edge_parameters()
    test_coordinate_descent_fits_bounded_objective()
