import numpy as np


def simulate(
    theta,
    seed,
    n_pasos=100,
    n_estabilizar=50,
    num_celulas=50,
    dt=0.05,
    limx=10.0,
    limy=10.0,
    radio=0.5,
    m=1.0,
    sigma_pos=0.0,
    distribucion_angulos="contraflujo",
    ecuaciones_activas=None,
):
    V0, u0, w, sigma_angulo = theta
    ecuaciones = {"propulsion", "repulsion", "alignment", "noise"} if ecuaciones_activas is None else set(ecuaciones_activas)

    rng = np.random.default_rng(seed)

    pos_x = rng.uniform(0.0, limx, num_celulas)
    pos_y = rng.uniform(0.0, limy, num_celulas)

    if distribucion_angulos == "contraflujo":
        mitad = num_celulas // 2
        angulos = np.zeros(num_celulas, dtype=float)
        angulos[mitad:] = np.pi
    else:
        angulos = rng.uniform(0.0, 2.0 * np.pi, num_celulas)

    posx_real = np.copy(pos_x)
    posy_real = np.copy(pos_y)

    fuerza_x = np.zeros(num_celulas, dtype=float)
    fuerza_y = np.zeros(num_celulas, dtype=float)
    alineacion = np.zeros(num_celulas, dtype=float)
    vel_x = np.zeros(num_celulas, dtype=float)
    vel_y = np.zeros(num_celulas, dtype=float)

    def paso_temporal():
        nonlocal pos_x, pos_y, posx_real, posy_real, angulos, vel_x, vel_y

        # 1. Broadcasting para obtener diferencias (N, N)
        dx = pos_x[:, None] - pos_x[None, :]
        dy = pos_y[:, None] - pos_y[None, :]

        # 2. Condiciones de borde periódicas (camino mas corto)
        dx = (dx + limx / 2.0) % limx - limx / 2.0
        dy = (dy + limy / 2.0) % limy - limy / 2.0

        # 3. Distancia entre todos los pares
        distancia = np.sqrt(dx * dx + dy * dy)

        # 4. Máascara para interacciones validas (excluye dist=0)
        mask = (distancia > 0.0) & (distancia < 2.0 * radio)

        # 5. Calculo de repulsion
        repulsion = np.zeros_like(distancia)
        if "repulsion" in ecuaciones:
            repulsion[mask] = (2.0 * radio - distancia[mask]) / distancia[mask]

        # 6. Suma de fuerzas sobre el eje 1 (columnas)
        fuerza_x[:] = np.sum(repulsion * dx, axis=1)
        fuerza_y[:] = np.sum(repulsion * dy, axis=1)

        # 7. Cálculo de alineación
        d_angulos = angulos[None, :] - angulos[:, None]
        alineacion_matriz = np.zeros_like(distancia)
        if "alignment" in ecuaciones:
            alineacion_matriz[mask] = w * np.sin(m * d_angulos[mask])
        alineacion[:] = np.sum(alineacion_matriz, axis=1)

        # Actualización de ángulos con ruido
        gauss_angulos = rng.normal(0.0, 1.0, num_celulas)
        ruido_angular = sigma_angulo * gauss_angulos * np.sqrt(dt) if "noise" in ecuaciones else 0.0
        angulos[:] = (angulos + (alineacion * dt) + ruido_angular) % (2.0 * np.pi)

        # Ruidos posicionales
        gauss_x = rng.normal(0.0, 1.0, num_celulas)
        gauss_y = rng.normal(0.0, 1.0, num_celulas)

        direc_x = np.cos(angulos)
        direc_y = np.sin(angulos)

        # Desplazamientos
        dx_step = (
            u0 * dt * fuerza_x
            + sigma_pos * np.sqrt(dt) * gauss_x
            + (V0 * direc_x * dt if "propulsion" in ecuaciones else 0.0)
        )
        dy_step = (
            u0 * dt * fuerza_y
            + sigma_pos * np.sqrt(dt) * gauss_y
            + (V0 * direc_y * dt if "propulsion" in ecuaciones else 0.0)
        )

        # Actualización de variables reales
        vel_x[:] = dx_step / dt
        vel_y[:] = dy_step / dt

        posx_real[:] += dx_step
        posy_real[:] += dy_step

        pos_x[:] = (pos_x + dx_step) % limx
        pos_y[:] = (pos_y + dy_step) % limy

    # Ciclo de estabilización
    for _ in range(n_estabilizar):
        paso_temporal()

    hist_posx_real = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_posy_real = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_posx_caja = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_posy_caja = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_velx = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_vely = np.zeros((n_pasos, num_celulas), dtype=float)

    # Ciclo de grabación de datos
    for step in range(n_pasos):
        paso_temporal()
        hist_posx_real[step, :] = posx_real
        hist_posy_real[step, :] = posy_real
        hist_posx_caja[step, :] = pos_x
        hist_posy_caja[step, :] = pos_y
        hist_velx[step, :] = vel_x
        hist_vely[step, :] = vel_y

    return {
        "posx_real": hist_posx_real,
        "posy_real": hist_posy_real,
        "posx_caja": hist_posx_caja,
        "posy_caja": hist_posy_caja,
        "velx": hist_velx,
        "vely": hist_vely,
        "angulos": np.copy(angulos),
        "theta": list(theta),
        "seed": seed,
        "dt": dt,
        "limx": limx,
        "limy": limy,
        "radio": radio,
        "num_celulas": num_celulas,
        "n_pasos": n_pasos,
    }