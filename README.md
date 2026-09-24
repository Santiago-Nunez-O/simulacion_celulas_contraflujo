# Simulación células a contraflujo

Este repositorio contiene la simulacion del comportamiento de celulas a contraflujo, junto con el calculo de observables, analisis de sensibilidad y optimizacion del sistema.

## Estructura del Proyecto

| Archivo | Descripcion |
| --- | --- |
| simulador.py | Logica fisica de la simulacion. Contiene la funcion simulate(theta, seed) para correr sin interfaz grafica. |
| observables.py | Funciones de calculo de los 5 observables y la funcion calcular_todos_observables(resultado_simulacion). |
| test_simulador.py | Script para verificar la ejecucion y reproducibilidad del simulador por consola. |
| animacion_interactiva.py | Animacion visual interactiva con controles deslizantes en tiempo real. |
| simulacion_celulas.ipynb | Cuaderno original con analisis detallado y optimizadores. |
| server.py | Servidor Python que ejecuta `simulador.py` y transmite frames por SSE. |
| index.html | Dashboard HUD de pantalla completa para visualizar el campo activo. |

## Parametros del Modelo ($\theta$)

El vector $\theta$ se compone de cuatro parametros:

| Parametro | Descripcion |
| --- | --- |
| $V_0$ | Velocidad de propulsion propia de cada celula. |
| $u_0$ | Intensidad de la fuerza de repulsion al haber contacto entre celulas. |
| $w$ | Intensidad de la alineacion entre celulas cercanas. |
| $\sigma_{\text{angulo}}$ | Intensidad del ruido aleatorio en el angulo de movimiento. |

## Observables del Sistema

| Observable | Nombre | Descripcion |
| --- | --- | --- |
| 1 | MSD | Desplazamiento cuadratico medio a lo largo del tiempo. |
| 2 | Distribucion de velocidades | Histogramas de componentes $v_x$, $v_y$ y rapidez. |
| 3 | Autocorrelacion temporal | Decaimiento de la correlacion de velocidad en el tiempo. |
| 4 | Correlacion espacial | Correlacion de velocidades segun la distancia radial $r$. |
| 5 | Distribucion de pares $g(r)$ | Densidad radial de pares de celulas a distancia $r$. |

## Como Usar el Simulador

Para ejecutar una simulacion y obtener los observables desde otro script:

from simulador import simulate
from observables import calcular_todos_observables

theta = [1.0, 1.0, 1.0, 1.0]
semilla = 42

resultado = simulate(theta, seed=semilla, n_pasos=100, num_celulas=50)
metricas = calcular_todos_observables(resultado)

msd = metricas['msd']
gr = metricas['gr']
radios = metricas['radios']

## Validacion

Para correr la prueba basica en la terminal:

python test_simulador.py

## Dashboard web en Python

El dashboard usa directamente `simulador.py` y `observables.py`. Ejecuta el
servidor desde esta carpeta con el entorno virtual del proyecto:

	TID-SBI/bin/python server.py

Abre `http://localhost:8000` en el navegador. El endpoint `ws://localhost:8000/ws`
anterior de Deno ya no es necesario: el servidor Python transmite frames por
`GET /events` usando SSE y recibe cambios por `POST /api/config`. Tambien existe
`GET /health` para comprobar el estado.
El panel **Motion equations** permite agregar, activar, desactivar o quitar ecuaciones
del movimiento. Cada ecuacion se asigna a un termino seguro del modelo: `propulsion`,
`repulsion`, `alignment` o `noise`; los terminos activos se envian por HTTP y
modifican el campo de movimiento. El boton `CAMERA / SCAN WHITEBOARD` solicita la
camara, captura la pizarra y usa OCR para dejar el texto detectado en el editor antes
de incorporarlo con `ADD`.

En el panel de telemetria se pueden seleccionar los cinco resultados estadisticos
del notebook: `MSD`, distribucion de velocidades, autocorrelacion temporal,
correlacion espacial y `g(r)`.

Los graficos estadisticos del dashboard siguen ahora la misma estructura que
`observables.py` y el notebook: `Desplazamiento Cuadratico Medio (MSD)`,
`Distribucion de Velocidades` con `P(v_x)` y `P(v_y)`, `Autocorrelacion Temporal`
con `C_xx` y `C_yy`, `Correlacion Espacial de Velocidades` y la funcion de pares
`g(r)`. Las series se calculan en el servidor Python y se transmiten por SSE.

La interfaz incluye temas oscuro y claro, guardados en el navegador. En `MODEL CONTROL`
se pueden cambiar `N / nodes` y `L / canvas size`; ambos valores reinician la poblacion
con el nuevo dominio y se transmiten por HTTP. El servidor limita los nodos a
8-300 y el dominio a 40-300 para mantener la simulacion interactiva.
