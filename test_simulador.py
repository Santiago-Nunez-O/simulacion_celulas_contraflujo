import time
import numpy as np
from simulador import simulate
from observables import calcular_todos_observables


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


if __name__ == "__main__":
    test_simulador()
