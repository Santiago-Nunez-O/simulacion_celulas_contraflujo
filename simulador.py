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
):
    V0, u0, w, sigma_angulo = theta

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

    size_grilla = 2.0 * radio
    columnas = max(1, int(limx / size_grilla))
    filas = max(1, int(limy / size_grilla))

    def paso_temporal():
        nonlocal pos_x, pos_y, posx_real, posy_real, angulos, vel_x, vel_y
        fuerza_x.fill(0.0)
        fuerza_y.fill(0.0)
        alineacion.fill(0.0)

        for i in range(num_celulas):
            grid_x1 = int(pos_x[i] / size_grilla)
            grid_y1 = int(pos_y[i] / size_grilla)

            for j in range(num_celulas):
                if i == j:
                    continue

                grid_x2 = int(pos_x[j] / size_grilla)
                grid_y2 = int(pos_y[j] / size_grilla)

                dist_grillax = abs(grid_x1 - grid_x2)
                dist_grillay = abs(grid_y1 - grid_y2)

                if dist_grillax > columnas / 2.0:
                    dist_grillax = columnas - dist_grillax
                if dist_grillay > filas / 2.0:
                    dist_grillay = filas - dist_grillay

                if dist_grillax > 1 or dist_grillay > 1:
                    continue

                dx = pos_x[i] - pos_x[j]
                dy = pos_y[i] - pos_y[j]

                if dx > limx / 2.0:
                    dx -= limx
                elif dx < -limx / 2.0:
                    dx += limx

                if dy > limy / 2.0:
                    dy -= limy
                elif dy < -limy / 2.0:
                    dy += limy

                distancia = np.sqrt(dx * dx + dy * dy)
                if 0.0 < distancia < 2.0 * radio:
                    repulsion = (2.0 * radio - distancia) / distancia
                    fuerza_x[i] += repulsion * dx
                    fuerza_y[i] += repulsion * dy
                    alineacion[i] += w * np.sin(m * (angulos[j] - angulos[i]))

        gauss_angulos = rng.normal(0.0, 1.0, num_celulas)
        angulos[:] = (
            angulos + (alineacion * dt) + sigma_angulo * gauss_angulos * np.sqrt(dt)
        ) % (2.0 * np.pi)

        gauss_x = rng.normal(0.0, 1.0, num_celulas)
        gauss_y = rng.normal(0.0, 1.0, num_celulas)

        direc_x = np.cos(angulos)
        direc_y = np.sin(angulos)

        dx = (
            u0 * dt * fuerza_x
            + sigma_pos * np.sqrt(dt) * gauss_x
            + V0 * direc_x * dt
        )
        dy = (
            u0 * dt * fuerza_y
            + sigma_pos * np.sqrt(dt) * gauss_y
            + V0 * direc_y * dt
        )

        vel_x[:] = dx / dt
        vel_y[:] = dy / dt

        posx_real[:] += dx
        posy_real[:] += dy

        pos_x[:] = (pos_x + dx) % limx
        pos_y[:] = (pos_y + dy) % limy

    for _ in range(n_estabilizar):
        paso_temporal()

    hist_posx_real = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_posy_real = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_posx_caja = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_posy_caja = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_velx = np.zeros((n_pasos, num_celulas), dtype=float)
    hist_vely = np.zeros((n_pasos, num_celulas), dtype=float)

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