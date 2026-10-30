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
modifican el campo de movimiento. Al pulsar un valor numerico de `MODEL CONTROL` se
puede escribir el valor manualmente; `REINICIAR BASE` restaura las cuatro ecuaciones
originales despues de confirmar la advertencia. El boton `CAMERA / SCAN WHITEBOARD`
solicita la camara, captura la pizarra y usa OCR para dejar el texto detectado en el
editor antes de incorporarlo con `ADD`.

El panel de optimización cuenta con botón de acceso directo `⚡ OPTIMIZACIÓN` en la barra del simulador y un selector de métodos dentro del panel lateral de ajustes. Se incluyen las técnicas analizadas en el cuaderno:
1. **Descenso por Coordenadas (Coordinate Descent)**: Búsqueda acotada coordenada por coordenada, rápida y determinista.
2. **Downhill Simplex (Nelder-Mead / Amoeba)**: Método geométrico sin derivadas que deforma un simplex multidimensional (reflexión, expansión, contracción y contracción múltiple).
3. **Recocido Simulado (Simulated Annealing)**: Búsqueda estocástica basada en fluctuaciones térmicas decrecientes para evitar estancarse en mínimos locales.
4. **Scipy Nelder-Mead**: Versión estándar de Nelder-Mead de la librería Scipy (cuando está disponible).
5. **Scipy Dual Annealing**: Algoritmo global de Scipy con recocido dual y refinamiento local.

Para verificar la optimización en tiempo real ("Optimization Trace & Verification"), la interfaz muestra:
- Estado del optimizador en vivo (IDLE, OPTIMIZANDO, CONVERGIDO, CANCELADO).
- Contador de iteraciones y evaluaciones totales de la simulación.
- Error cuadrático normalizado en cada paso y barra de progreso porcentual.
- Panel de traza con registro detallado por paso (iteración, método, error y vector de parámetros evaluado).
- Posibilidad de cancelar en cualquier momento con el botón `CANCELAR`.
- Botón `PANEL DE TRAZA` para inspeccionar la evolución en cualquier momento.

## Despliegue en la Web con GitHub Pages

El proyecto está diseñado con **arquitectura dual**:
- **Modo Servidor Python**: Al ejecutar `python server.py`, el servidor realiza los cálculos y transmite los datos por SSE (`/events`).
- **Modo Web Autónomo (GitHub Pages / Standalone)**: Al abrir `index.html` en GitHub Pages (o sin servidor Python), el navegador ejecuta automáticamente el motor de física, el cálculo de los 5 observables y los algoritmos de optimización directamente en JavaScript a 60 FPS, sin requerir ningún backend.

### Cómo configurar GitHub Pages en GitHub

1. Sube los cambios al repositorio (`git push origin martin` o a tu rama principal).
2. Entra en tu repositorio en GitHub y ve a la pestaña **Settings** (Configuración).
3. En el menú lateral izquierdo, haz clic en **Pages**.
4. En la sección **Build and deployment**:
   - **Opción recomendada (Automática con GitHub Actions)**: En **Source**, selecciona **GitHub Actions**. El repositorio ya incluye el flujo `.github/workflows/deploy-pages.yml` que desplegará el proyecto automáticamente en cada push.
   - **Opción estándar**: En **Source**, selecciona **Deploy from a branch**, elige la rama (por ejemplo, `martin` o `main`) y la carpeta `/ (root)`. Haz clic en **Save**.
5. Espera un minuto a que finalice el despliegue y accede a la URL pública: `https://<usuario>.github.io/simulacion_celulas_contraflujo/`.

Con los cuatro parametros en cero las celulas permanecen quietas. La propulsion sola,
con ruido, repulsion y alineacion desactivados, produce rapidez constante. Con ruido o
fuerzas activos la direccion y la rapidez pueden variar intencionalmente.

En el panel de telemetria se pueden seleccionar los cinco resultados estadisticos
del notebook: `MSD`, distribucion de velocidades, autocorrelacion temporal,
correlacion espacial y `g(r)`.

Los graficos estadisticos del dashboard siguen la misma estructura que
`observables.py` y el notebook: `Desplazamiento Cuadratico Medio (MSD)`,
`Distribucion de Velocidades` con `P(v_x)` y `P(v_y)`, `Autocorrelacion Temporal`
con `C_xx` y `C_yy`, `Correlacion Espacial de Velocidades` y la funcion de pares
`g(r)`.

La interfaz incluye temas oscuro y claro, guardados en el navegador. En `MODEL CONTROL`
se pueden cambiar `N / nodes` y `L / canvas size`; ambos valores reinician la poblacion
con el nuevo dominio. En el modo servidor el backend limita los nodos a 8-300 y el dominio
a 40-300 para mantener la simulacion interactiva.
