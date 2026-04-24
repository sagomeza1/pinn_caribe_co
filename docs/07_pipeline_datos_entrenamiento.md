# Pipeline de datos para el entrenamiento de la PINN

**Proyecto:** Predicción de condiciones meteorológicas usando PINNs — Caribe de Colombia  
**Fecha:** Abril 2026  
**Contexto académico:** Trabajo de grado — Maestría en Analítica de Datos, Universidad Central

---

## 1. Origen de los datos

Los datos provienen de **Datos Abiertos Colombia** (<https://www.datos.gov.co/>), que publica
registros históricos de estaciones meteorológicas del IDEAM. Para este proyecto se utilizó el
archivo:

```
data/raw/em_caribe_20251201_20251231.parquet
```

Este archivo cubre el mes de **diciembre de 2025** y contiene registros horarios de estaciones
ubicadas en el Caribe colombiano. Cada fila corresponde a una observación de una estación en
un instante de tiempo. Las columnas relevantes son:

| Columna | Descripción |
|---|---|
| `codigo_estacion` | Identificador único de la estación |
| `segundos` | Registro temporal en segundos desde el inicio del periodo |
| `latitud` | Latitud geográfica (grados decimales) |
| `longitud` | Longitud geográfica (grados decimales) |
| `altura` | Altitud de la estación sobre el nivel del mar (m) |
| `presion` | Presión atmosférica medida (mbar) |
| `temperatura` | Temperatura del aire (°C) |
| `vel_u` | Componente X de la velocidad del viento (m/s) |
| `vel_v` | Componente Y de la velocidad del viento (m/s) |

> **Nota sobre la transformación del viento:** La dirección y magnitud originales del viento son
> descompuestas en componentes cartesianas `vel_u` (Este–Oeste) y `vel_v` (Norte–Sur) antes de
> almacenarse en el Parquet. Esta transformación es obligatoria e irrenunciable en el proyecto.

---

## 2. Carga de datos (`src/data/cargar_datos.py`)

La función `cargar_parquet()` lee el archivo Parquet y aplica las siguientes operaciones:

### 2.1 Filtrado de estaciones

Se excluyen estaciones con problemas detectados en análisis previos (lista centralizada en
`config/settings.py → ESTACIONES_EXCLUIDAS`). Para el experimento principal se excluyen cuatro
estaciones por presentar presión anómalamente alta o errores elevados en las componentes de viento.

### 2.2 Subset rectangular (opcional)

Si se especifica `registros_por_estacion`, el loader descarta estaciones con menos registros
disponibles que el umbral y recorta todas las demás al mismo número de observaciones, ordenadas
cronológicamente. Esto garantiza una estructura rectangular
$(N_{\text{estaciones}} \times N_{\text{tiempos}})$ antes de continuar.

### 2.3 Salida

La función retorna un diccionario con arrays NumPy planos (un valor por registro) y metadatos:
número de estaciones, códigos utilizados, total de registros y tiempos disponibles.

---

## 3. Preprocesamiento (`src/data/preprocesar_datos.py`)

El pipeline de preprocesamiento ejecuta cuatro transformaciones en secuencia:

### 3.1 Proyección cartesiana

Las coordenadas geográficas se convierten a metros usando la aproximación de seno esférico
sobre el radio terrestre $R_T = 6\,378\,000$ m:

$$X = R_T \cdot \sin(\text{longitud}^\circ) \qquad Y = R_T \cdot \sin(\text{latitud}^\circ)$$

La presión se convierte de mbar a Pa multiplicando por 100.

### 3.2 Reestructuración por estación

Los arrays planos de longitud $N_{\text{total}} = N_{\text{est}} \times N_{\text{t}}$ se
reorganizan en matrices de forma $(N_{\text{est}},\ N_t)$ usando orden Fortran (columna mayor).
Las estaciones sin coordenada espacial válida (NaN en X) se eliminan en este paso.

### 3.3 Filtrado temporal y ordenamiento

Se seleccionan únicamente los registros correspondientes a los primeros $N_{\text{días}} = 31$
días del periodo (diciembre 2025 completo). Opcionalmente se aplica un intervalo de submuestreo.
Finalmente, las estaciones se ordenan por su coordenada X en cada instante temporal.

### 3.4 Corrección de presión ISA

La presión medida en cada estación se corrige al nivel del mar usando el modelo de atmósfera
estándar internacional (ISA):

$$P_{\text{corr}} = P \cdot \left(1 - \frac{0.0065 \cdot Z}{T + 273.15 + 0.0065 \cdot Z}\right)^{-5.257}$$

donde $Z$ es la altitud y $T$ la temperatura en °C.

---

## 4. Normalización (`src/data/normalizar_datos.py`)

Antes del entrenamiento, todas las variables se adimensionalizan para que la red neuronal
opere con magnitudes comparables. El proceso define escalas de referencia a partir de los
propios datos de estaciones:

| Escala | Definición | Significado |
|---|---|---|
| $L$ | Diagonal del dominio espacial (m) | Longitud característica |
| $W$ | Magnitud máxima de viento observada (m/s) | Velocidad de referencia |
| $P_0$ | Presión media (Pa) | Presión de referencia |
| $Re$ | $W \cdot L / \nu$ | Número de Reynolds |

Las transformaciones aplicadas son:

$$X_{\text{norm}} = \frac{X - X_c}{L}, \quad Y_{\text{norm}} = \frac{Y - Y_c}{L}$$

$$T_{\text{norm}} = \frac{(T - T_{\min}) \cdot W}{L}$$

$$u_{\text{norm}} = \frac{u}{W}, \quad v_{\text{norm}} = \frac{v}{W}$$

$$P_{\text{norm}} = \frac{P - P_0}{\rho \cdot W^2}$$

donde $X_c$, $Y_c$ son los centros del dominio. Opcionalmente, si la presión normalizada tiene
un rango muy diferente al de las velocidades, se divide adicionalmente por su desviación
estándar $\sigma_P$ (reescalado de presión, activo en iteraciones avanzadas).

---

## 5. Construcción de la malla espacio-temporal (`src/data/construir_malla.py`)

Este paso es central en la arquitectura PINN: se genera una **malla de puntos de colocación**
donde se evaluarán las ecuaciones de Navier-Stokes, sin requerir observaciones reales.

### 5.1 Derivación de la malla a partir de las estaciones

La malla **no es arbitraria**: sus límites espaciales se derivan directamente de las posiciones
de las estaciones meteorológicas. Se toma el bounding box de las coordenadas cartesianas de
todas las estaciones y se extiende en cada dirección por una distancia equivalente a la
resolución de la malla en metros:

$$R_{\text{PINN}} = R_T \cdot \sin(R_{\text{grados}}^\circ) \approx R_T \cdot \frac{\pi \cdot R_{\text{grados}}}{180}$$

Con la resolución por defecto $R = 0.4°$, esto equivale aproximadamente a $44\,500$ m.

Los ejes de la malla se generan como:

$$x_{\text{pinn}} = \{x_{\min} - R_{\text{PINN}},\ x_{\min},\ \ldots,\ x_{\max} + R_{\text{PINN}}\} \text{ con paso } R_{\text{PINN}}$$

y análogamente para $y_{\text{pinn}}$.

### 5.2 Construcción del meshgrid espacio-temporal

A partir de los ejes se genera un `meshgrid` 2D que produce $N_{\text{pts}}$ pares $(x_i, y_j)$.
Este grid espacial se **replica para cada instante temporal** disponible en los datos de
estaciones, creando un tensor 3D:

$$\text{Malla} = \{(x_i,\ y_j,\ t_k)\} \quad \forall\, i,j \in \text{grid},\quad k \in \{1,\ldots,N_t\}$$

El resultado son tres matrices de forma $(N_{\text{pts}},\ N_t)$:

- `X_PINN`: posiciones X normalizadas, replicadas para todos los tiempos
- `Y_PINN`: posiciones Y normalizadas, replicadas para todos los tiempos
- `T_PINN`: instantes temporales normalizados, replicados para todos los puntos espaciales

La malla se normaliza con las mismas escalas $L$ y $W$ calculadas en el paso anterior.

### 5.3 Ejemplo de dimensiones (experimento con 300 registros/estación, R=0.4°)

| Componente | Dimensiones típicas |
|---|---|
| Estaciones activas | ~16 |
| Tiempos ($N_t$) | 300 |
| Puntos espaciales de la malla ($N_{\text{pts}}$) | ~30–50 |
| **Total puntos de colocación** | $N_{\text{pts}} \times N_t \approx 9{,}000–15{,}000$ |

---

## 6. Datasets de PyTorch (`src/data/datasets.py`)

A partir de los datos normalizados y la malla se construyen dos objetos `Dataset`:

### `EstacionesDataset`

Contiene los datos **observados** en las estaciones meteorológicas.  
Cada muestra es una tupla $(t,\ x,\ y,\ u,\ v,\ p)$ normalizada.  
Las filas con NaN en cualquier variable objetivo ($u$, $v$, $p$) se descartan en la
construcción del dataset (sin eliminar los datos del Parquet original).

### `ColocacionDataset`

Contiene los **puntos de la malla** donde se evaluará la física.  
Cada muestra es una tupla $(t,\ x,\ y)$ normalizada.  
No requiere etiquetas observadas: su único propósito es proporcionar coordenadas donde
calcular los residuos de Navier-Stokes.

---

## 7. Uso de los datasets en el entrenamiento (`src/training/entrenador.py`)

La función `entrenar()` consume ambos datasets mediante `DataLoader` con batches aleatorios
y ejecuta el ciclo de entrenamiento con una **función de pérdida combinada**:

$$\mathcal{L} = \frac{\lambda_{\text{NS}} \cdot \mathcal{L}_{\text{NS}}^2 + \mathcal{L}_u^2 + \mathcal{L}_v^2 + \mathcal{L}_p^2}{\lambda_{\text{NS}} \cdot \mathcal{L}_{\text{NS}} + \mathcal{L}_u + \mathcal{L}_v + \mathcal{L}_p}$$

donde:

- $\mathcal{L}_u,\ \mathcal{L}_v,\ \mathcal{L}_p$: MSE entre predicción y observación en estaciones
- $\mathcal{L}_{\text{NS}}$: residuo de Navier-Stokes evaluado en puntos de la malla y de estaciones
- $\lambda_{\text{NS}}$: peso de la pérdida física (configurable)

### 7.1 Pérdida física — Ecuaciones de Navier-Stokes (Euler invíscido)

La red calcula derivadas parciales de sus propias salidas mediante diferenciación automática
(`torch.autograd.grad`). Los residuos evaluados son:

$$\text{Continuidad:} \quad \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

$$\text{Momento X:} \quad \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \sigma_P \frac{\partial p}{\partial x} = 0$$

$$\text{Momento Y:} \quad \frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \sigma_P \frac{\partial p}{\partial y} = 0$$

donde $\sigma_P$ es el factor de reescalado de presión. Estos residuos se minimizan en los
puntos de la malla donde **no hay observaciones**.

### 7.2 Curriculum learning

El entrenamiento sigue tres fases para evitar que la física mal calibrada obstaculice el
ajuste inicial a los datos:

| Fase | Épocas | Descripción |
|---|---|---|
| 1 — Solo datos | 0 – 99 | $\lambda_{\text{NS}} = 0$ (solo observaciones) |
| 2 — Rampa | 100 – 199 | $\lambda_{\text{NS}}$ crece linealmente de 0 a $\lambda_{\max}$ |
| 3 — Completo | 200 – fin | $\lambda_{\text{NS}} = \lambda_{\max}$ (física activa al 100%) |

---

## 8. Diagrama del pipeline completo

```
Datos Abiertos Colombia
        │
        ▼
em_caribe_20251201_20251231.parquet
        │
        ▼ cargar_parquet()
  Arrays planos NumPy
  (código, t, lat, lon, altura, presión, temp, vel_u, vel_v)
        │
        ▼ preprocesar()
  Proyección cartesiana (X, Y en metros)
  Reestructuración → matrices (N_est × N_t)
  Filtrado temporal (31 días)
  Corrección presión ISA
        │
        ▼ normalizar()
  Centrado + adimensionalización
  Escalas L, W, P₀, Re
  Datos normalizados (N_est × N_t)
        │
  ┌─────┴──────────────────────────┐
  ▼                                ▼
EstacionesDataset            construir_malla()
(t, x, y, u, v, p)          Bounding box de estaciones
observado y normalizado      + extensión R_PINN
                             Meshgrid 2D × N_t timesteps
                             → ColocacionDataset (t, x, y)
  │                                │
  └───────────────┬────────────────┘
                  ▼
           DataLoader (batches aleatorios)
                  │
                  ▼
          PINN(t, x, y) → (û_x, û_y, p̂)
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
  L_u (MSE)   L_v (MSE)   L_p (MSE)    ← estaciones
  L_NS (Navier-Stokes autograd)         ← malla
     └────────────┼────────────┘
                  ▼
        L_total (combinada ponderada)
                  │
                  ▼
            Adam + ReduceLROnPlateau
            Gradient clipping (0.5)
            Curriculum learning
```

---

## 9. Resumen de parámetros clave (experimento de referencia)

| Parámetro | Valor |
|---|---|
| Fuente | `data/raw/em_caribe_20251201_20251231.parquet` |
| Periodo | Diciembre 2025 (31 días, horario) |
| Registros por estación | 300 |
| Estaciones excluidas | 4 |
| Resolución de la malla | 0.4° (~44.5 km) |
| $\lambda_{\text{NS}}$ máximo | 3.0 |
| Épocas totales | 2 000 |
| Arquitectura red | 12 capas × 1 200 neuronas, activación Tanh |
| Optimizador | Adam ($\alpha = 10^{-4}$) + ReduceLROnPlateau |
| Gradient clipping | norma máx. 0.5 |
