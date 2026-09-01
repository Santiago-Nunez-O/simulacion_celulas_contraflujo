import numpy as np


def calcular_msd(hist_posx_real, hist_posy_real):
    T, N = hist_posx_real.shape
    msd = np.zeros(T, dtype=float)

    for k in range(1, T):
        dx = hist_posx_real[k:, :] - hist_posx_real[:-k, :]
        dy = hist_posy_real[k:, :] - hist_posy_real[:-k, :]
        msd[k] = np.mean(dx * dx + dy * dy)

    return msd


def calcular_distribucion_velocidades(hist_velx, hist_vely):
    vx_plano = hist_velx.flatten()
    vy_plano = hist_vely.flatten()
    rapidez = np.sqrt(vx_plano * vx_plano + vy_plano * vy_plano)
    return {
        "vx": vx_plano,
        "vy": vy_plano,
        "rapidez": rapidez,
    }


def calcular_autocorrelacion(hist_v):
    T, N = hist_v.shape
    auto_cor = np.zeros(T, dtype=float)
    var_v = np.sum(hist_v * hist_v) / (N * T)

    if var_v < 1e-12:
        return auto_cor

    for k in range(T):
        suma_auto = np.sum(hist_v[: T - k, :] * hist_v[k:, :])
        auto_cor[k] = (suma_auto / (N * (T - k))) / var_v

    return auto_cor


def calcular_correlacion_y_pares(
    hist_posx_caja,
    hist_posy_caja,
    hist_velx,
    hist_vely,
    limx=10.0,
    limy=10.0,
    bins=50,
):
    T, N = hist_posx_caja.shape
    rmax = min(limx, limy) / 2.0
    dr = rmax / bins
    radios = np.linspace(dr / 2.0, rmax - dr / 2.0, bins)

    numerador_x = np.zeros(bins, dtype=float)
    numerador_y = np.zeros(bins, dtype=float)
    denominador = np.zeros(bins, dtype=float)

    for t in range(T):
        vx = hist_velx[t]
        vy = hist_vely[t]
        px = hist_posx_caja[t]
        py = hist_posy_caja[t]

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                dx = px[i] - px[j]
                dy = py[i] - py[j]

                if dx > limx / 2.0:
                    dx -= limx
                elif dx < -limx / 2.0:
                    dx += limx

                if dy > limy / 2.0:
                    dy -= limy
                elif dy < -limy / 2.0:
                    dy += limy

                distancia = np.sqrt(dx * dx + dy * dy)
                if distancia < rmax:
                    b = int(distancia / dr)
                    if b < bins:
                        numerador_x[b] += vx[i] * vx[j]
                        numerador_y[b] += vy[i] * vy[j]
                        denominador[b] += 1.0

    cxx = np.zeros(bins, dtype=float)
    cyy = np.zeros(bins, dtype=float)
    gr = np.zeros(bins, dtype=float)
    densidad = N / (limx * limy)

    for b in range(bins):
        if denominador[b] > 0:
            cxx[b] = numerador_x[b] / denominador[b]
            cyy[b] = numerador_y[b] / denominador[b]

        r_actual = radios[b]
        area = 2.0 * np.pi * r_actual * dr
        gr[b] = denominador[b] / (T * N * densidad * area)

    cxx0 = np.sum(hist_velx * hist_velx) / (N * T)
    cyy0 = np.sum(hist_vely * hist_vely) / (N * T)

    cxx_norm = cxx / cxx0 if cxx0 > 1e-12 else cxx
    cyy_norm = cyy / cyy0 if cyy0 > 1e-12 else cyy

    return radios, cxx_norm, cyy_norm, gr


def calcular_todos_observables(resultado_sim):
    px_real = resultado_sim["posx_real"]
    py_real = resultado_sim["posy_real"]
    px_caja = resultado_sim["posx_caja"]
    py_caja = resultado_sim["posy_caja"]
    vx = resultado_sim["velx"]
    vy = resultado_sim["vely"]
    dt = resultado_sim["dt"]
    limx = resultado_sim["limx"]
    limy = resultado_sim["limy"]
    n_pasos = resultado_sim["n_pasos"]

    msd = calcular_msd(px_real, py_real)
    dist_vel = calcular_distribucion_velocidades(vx, vy)
    auto_vx = calcular_autocorrelacion(vx)
    auto_vy = calcular_autocorrelacion(vy)
    auto_prom = (auto_vx + auto_vy) / 2.0

    radios, cxx_r, cyy_r, gr = calcular_correlacion_y_pares(
        px_caja, py_caja, vx, vy, limx=limx, limy=limy
    )
    c_prom = (cxx_r + cyy_r) / 2.0
    tiempo = np.arange(n_pasos) * dt

    return {
        "msd": msd,
        "distribucion_vel": dist_vel,
        "autocor_vx": auto_vx,
        "autocor_vy": auto_vy,
        "autocor_prom": auto_prom,
        "radios": radios,
        "cxx_r": cxx_r,
        "cyy_r": cyy_r,
        "c_prom": c_prom,
        "gr": gr,
        "tiempo": tiempo,
    }
