# Simulacion de Celulas en Contraflujo

Este repositorio contiene la simulacion del comportamiento de celulas a contraflujo, junto con el calculo de observables, analisis de sensibilidad y optimizacion del sistema.

## Estructura del Proyecto

| Archivo | Descripcion |
| --- | --- |
| simulador.py | Logica fisica de la simulacion. Contiene la funcion simulate(theta, seed) para correr sin interfaz grafica. |
| observables.py | Funciones de calculo de los 5 observables y la funcion calcular_todos_observables(resultado_simulacion). |
| test_simulador.py | Script para verificar la ejecucion y reproducibilidad del simulador por consola. |
| animacion_interactiva.py | Animacion visual interactiva con controles deslizantes en tiempo real. |
| simulacion_celulas.ipynb | Cuaderno original con analisis detallado y optimizadores. |

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
