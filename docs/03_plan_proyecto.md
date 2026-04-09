# Plan de Proyecto — PINN Prediccion Meteorologica Caribe Colombia

**Ultima actualizacion:** 9 de abril de 2026
**Estado global:** Fase 0 (Planificacion) — aprobado

---

## Resumen

Construir desde cero un proyecto PINN en PyTorch que prediga u_x, u_y y presion sobre una malla espacial en el Caribe colombiano, replicando el pipeline de datos del proyecto base (`trabajo_base_pinns`), con una red de 12 capas x 1200 neuronas con Weight Normalization desacoplada, entrenamiento de 2000 epocas con CUDA, y validacion visual mediante GIFs de curvas de nivel.

---

## Decisiones confirmadas

| Decision | Valor |
|----------|-------|
| Entradas | X, Y, t (invariable) |
| Salidas | u_x, u_y, presion (invariable) |
| Datos Parquet | `em_caribe_20251201_20251231.parquet` — 16,368 filas, 9 columnas |
| Columnas Parquet | codigo_estacion, segundos, latitud, longitud, presion, temperatura, altura, vel_u, vel_v |
| vel_u/vel_v | Ya descompuestas en el Parquet (la transformacion dir+mag a XY fue aplicada en preprocesamiento del proyecto base) |
| Presion | Corregir a nivel del mar con ISA, luego adimensionalizar como P_norm = (P - P0) / (rho * W^2) |
| Normalizacion | Adimensionalizacion por escalas de referencia: L (longitud), W (velocidad), rho*W^2 (presion) |
| Malla | R=0.1 grados, derivada de posiciones de estaciones |
| Arquitectura | 12 capas ocultas x 1200 neuronas, GammaBiasLayer con Weight Normalization |
| Activaciones | 8 primeras capas: Tanh, 4 ultimas capas: Lineal |
| Navier-Stokes | Inviscido (sin termino de difusion) |
| Optimizador | Adam + ReduceLROnPlateau (factor=0.3162, patience=50) |
| Lambda (peso fisico) | 3.0 |
| Epocas | 2000, checkpoint cada 500 |
| GPU | CUDA-PyTorch |
| Entorno | venv + pip (requirements.txt) |
| Idioma codigo | Espanol (comentarios, docstrings, logs) |

---

## Datos del Parquet (verificado)

```
Filas: 16,368 | Columnas: 9
Esquema:
  codigo_estacion: string    (aprox. 22 estaciones x 744 horas)
  segundos: double           (0, 3600, 7200, ... → intervalo de 1 hora)
  latitud: double            (ejemplo: 9.071)
  longitud: double           (ejemplo: -76.224)
  presion: double            (mbar, ejemplo: 1006.9)
  temperatura: double        (°C, ejemplo: 27.9)
  altura: double             (m, ejemplo: 12.0)
  vel_u: double              (m/s, componente X)
  vel_v: double              (m/s, componente Y)
```

**Nota:** vel_u y vel_v ya estan descompuestas. La transformacion direccion+magnitud a X/Y se realizo en el preprocesamiento del proyecto base. Se documenta la trazabilidad pero no se repite la descomposicion.

---

## Fases del plan

### Fase 1 — Infraestructura y configuracion del proyecto

**Objetivo:** Tener la estructura del repositorio, entorno virtual, dependencias y git inicializado.

**Estado:** [x] Completada

**Pasos:**

1. [x] **Inicializar repositorio Git** con .gitignore adecuado (incluir `solutions/`, `models/`, `logs/`, `data/processed/`, `__pycache__/`, `.venv/`, `*.gif`)
2. [x] **Crear estructura de directorios:**
   ```
   pinn/
     src/
       __init__.py
       data/          <- __init__.py + modulos de carga y preprocesamiento
       model/         <- __init__.py + arquitectura PINN
       training/      <- __init__.py + bucle de entrenamiento
       evaluation/    <- __init__.py + metricas y GIFs
     config/
       __init__.py
       settings.py    <- constantes centralizadas
       logging_config.py
     notebooks/
     models/           <- checkpoints guardados
     logs/             <- logs de entrenamiento
     solutions/        <- documentacion dinamica
     main.py           <- punto de entrada
   ```
3. [x] **Crear requirements.txt** con: torch (CUDA), numpy, pandas, pyarrow, matplotlib, scipy, pillow
4. [x] **Crear entorno virtual venv** e instalar dependencias
5. [x] **Crear `config/settings.py`** con constantes centralizadas:
   - RADIO_TIERRA = 6_378_000 (m)
   - RHO_AIRE = 1.18 (kg/m3)
   - NU_AIRE = 1.382e-5 (m2/s)
   - R_MALLA = 0.1 (grados)
   - N_DIAS = 31
   - LAMBDA_FISICA = 3.0
   - NUM_EPOCAS = 2000
   - LR_INICIAL = 1e-4
   - MAX_NORM_GRAD = 0.5
   - HIDDEN_NEURONS = 1200
   - NUM_CAPAS_TANH = 8
   - NUM_CAPAS_LINEALES = 4
   - CHECKPOINT_INTERVALO = 500
6. [x] **Commit:** `feat: estructura inicial del proyecto y configuracion`

**Archivos a crear:**
- `.gitignore`
- `requirements.txt`
- `config/__init__.py`, `config/settings.py`, `config/logging_config.py`
- `src/__init__.py`, `src/data/__init__.py`, `src/model/__init__.py`, `src/training/__init__.py`, `src/evaluation/__init__.py`
- `main.py`
- `models/.gitkeep`, `logs/.gitkeep`, `solutions/.gitkeep`, `notebooks/.gitkeep`

---

### Fase 2 — Pipeline de datos

**Objetivo:** Cargar, preprocesar, normalizar y construir la malla, replicando el pipeline del proyecto base.
**Depende de:** Fase 1
**Estado:** [x] Completada

**Pasos:**

7. [x] **Crear `src/data/cargar_datos.py`** — Carga del Parquet con pyarrow
   - Funcion `cargar_parquet(ruta)` que retorne arrays NumPy
   - Manejar valores faltantes segun regla canonica (no eliminar automaticamente, documentar)
   - Referencia: `trabajo_base_pinns/src/process_data.py` clase ProcessDataColombia

8. [x] **Crear `src/data/preprocesar_datos.py`** — Transformaciones del proyecto base:
   - **Proyeccion geografica a cartesiano:**
     - X_WS = 6378000 * sin(rad(longitud))
     - Y_WS = 6378000 * sin(rad(latitud))
   - **Correccion de presion ISA a nivel del mar:**
     - P_corr = P * (1 - 0.0065 * Z / (T + 273.15 + 0.0065 * Z))^(-5.257)
     - P_pa = P_corr * 100 (mbar a Pa)
   - **Filtrado temporal y ordenamiento por coordenada X**
   - Referencia: `trabajo_base_pinns/src/process_data.py` metodos `_process_coordinates_and_projections()`, `_filter_by_time_and_sort_by_coor()`

9. [x] **Crear `src/data/normalizar_datos.py`** — Adimensionalizacion:
   - **Centrado espacial:**
     - X_c = X - (X_min + X_max) / 2
     - Y_c = Y - (Y_min + Y_max) / 2
     - T_c = T - T_min
   - **Escalas de referencia:**
     - L = sqrt((X_max - X_min)^2 + (Y_max - Y_min)^2)  (longitud caracteristica)
     - W = sqrt(max|U|^2 + max|V|^2)  (velocidad maxima)
     - P0 = mean(P)  (presion media)
   - **Normalizacion:**
     - X_norm = X_c / L
     - Y_norm = Y_c / L
     - T_norm = T_c * W / L
     - U_norm = U / W
     - V_norm = V / W
     - P_norm = (P - P0) / (rho * W^2)
   - **Calcular Reynolds:** Re = W * L / nu
   - Referencia: `trabajo_base_pinns/src/process_data.py` metodo `_centered_grid_adimensionalization()`

10. [x] **Crear `src/data/construir_malla.py`** — Malla de prediccion:
    - R_PINN = 6378000 * sin(rad(R)) con R=0.1
    - x_grid = arange(x_min - R_PINN, x_max + R_PINN, R_PINN) - centro_x
    - y_grid = arange(y_min - R_PINN, y_max + R_PINN, R_PINN) - centro_y
    - meshgrid con flatten('F') (orden columna mayor)
    - Replicar tiempo para cada punto de la malla
    - Normalizar la malla con las mismas escalas (L, W)
    - Referencia: `trabajo_base_pinns/src/process_data.py` metodo `_centered_grid_adimensionalization()` seccion de meshgrid

11. [x] **Crear `src/data/datasets.py`** — Datasets de PyTorch:
    - `EstacionesDataset`: (t, x, y, u, v, p) aplanado a N_estaciones x N_tiempos filas
    - `ColocationDataset`: (t, x, y) puntos de la malla para perdida fisica
    - Referencia: `trabajo_base_pinns/src/dataset.py`

12. [x] **Commit:** `feat: pipeline de datos — carga, preprocesamiento, normalizacion y malla`

**Archivos a crear:**
- `src/data/cargar_datos.py`
- `src/data/preprocesar_datos.py`
- `src/data/normalizar_datos.py`
- `src/data/construir_malla.py`
- `src/data/datasets.py`

---

### Fase 3 — Arquitectura de la red PINN

**Objetivo:** Implementar la red con Weight Normalization desacoplada segun especificacion.
**Depende de:** Fase 1
**Estado:** [x] Completada

**Pasos:**

13. [x] **Crear `src/model/capas.py`** — Capa personalizada GammaBiasLayer:
    - nn.Linear(in_features, out_features, bias=False)
    - weight_norm() de torch.nn.utils
    - gamma = nn.Parameter(torch.ones(out_features)) — escalado
    - beta = nn.Parameter(torch.zeros(out_features)) — desplazamiento
    - Inicializacion Xavier/Glorot Uniform en pesos
    - Forward: y = gamma * (W_norm * x) + beta

14. [x] **Crear `src/model/pinn.py`** — Red PINN completa:
    - Input: (t, x, y) con dim=3
    - Capa de entrada: GammaBiasLayer(3, 1200) + Tanh
    - 8 capas ocultas con Tanh: GammaBiasLayer(1200, 1200) + Tanh
    - 4 capas ocultas lineales: GammaBiasLayer(1200, 1200) sin activacion
    - Capa de salida: GammaBiasLayer(1200, 3)
    - Output: (u_x, u_y, p)
    - **Total:** 1 entrada + 8 tanh + 4 lineal + 1 salida = 14 capas (12 ocultas)

15. [x] **Crear `src/model/perdida_fisica.py`** — Navier-Stokes inviscido:
    - Derivadas parciales via torch.autograd.grad con create_graph=True
    - Ecuacion de continuidad: du/dx + dv/dy = 0
    - Momento X: du/dt + u*du/dx + v*du/dy + dp/dx = 0
    - Momento Y: dv/dt + u*dv/dx + v*dv/dy + dp/dy = 0
    - Residual = MSE(e1, 0) + MSE(e2, 0) + MSE(e3, 0)
    - Referencia: `trabajo_base_pinns/src/loss_functions.py`

16. [x] **Commit:** `feat: arquitectura PINN con Weight Normalization y perdida Navier-Stokes`

**Archivos a crear:**
- `src/model/capas.py`
- `src/model/pinn.py`
- `src/model/perdida_fisica.py`

---

### Fase 4 — Bucle de entrenamiento

**Objetivo:** Implementar el entrenamiento completo con logging, checkpoints y CUDA.
**Depende de:** Fases 2 y 3
**Estado:** [x] Completada

**Pasos:**

17. [x] **Crear `config/logging_config.py`** — Log dual:
    - Consola: nivel INFO
    - Archivo en `logs/`: nivel DEBUG con timestamp
    - Registrar en puntos clave: inicio, cada N epocas, cambio de LR, checkpoint, fin
    - Referencia: `trabajo_base_pinns/config/logging_config.py`

18. [x] **Crear `src/training/entrenador.py`** — Bucle de entrenamiento:
    - Deteccion automatica CUDA para seleccionar device
    - Modelo a GPU
    - DataLoaders para estaciones y puntos de colocacion
    - Optimizador Adam(lr=1e-4)
    - Scheduler ReduceLROnPlateau(factor=0.3162, patience=50, threshold=1e-3)
    - Gradient clipping: clip_grad_norm_(max_norm=0.5)
    - **Funcion de perdida combinada** (del proyecto base):
      - loss = (lambda*loss_ns^2 + loss_u^2 + loss_v^2 + loss_p^2) / (lambda*loss_ns + loss_u + loss_v + loss_p)
      - La perdida fisica se calcula en puntos de estaciones Y en puntos de colocacion
    - **Logging por epoca:**
      - loss total, loss_ns, loss_u, loss_v, loss_p, learning rate
    - **Checkpoint cada 500 epocas** (epocas 500, 1000, 1500, 2000):
      - Guardar model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, history
      - Formato: `models/PINN_caribe_epoca_{N}.pth`
    - **Historial como .mat** (scipy.io.savemat):
      - epoch, loss, ns_loss, p_loss, u_loss, v_loss, lr
    - Referencia: `trabajo_base_pinns/src/train_pinn.py`

19. [x] **Crear `main.py`** — Punto de entrada:
    - Cargar datos -> Preprocesar -> Normalizar -> Construir malla
    - Crear datasets y dataloaders
    - Instanciar modelo PINN en GPU
    - Ejecutar entrenamiento
    - Referencia: `trabajo_base_pinns/main.py`

20. [x] **Commit:** `feat: bucle de entrenamiento con CUDA, logging y checkpoints`

**Archivos a crear/modificar:**
- `src/training/entrenador.py`
- `config/logging_config.py`
- `main.py`

---

### Fase 5 — Evaluacion y visualizacion (GIFs)

**Objetivo:** Generar GIFs de curvas de nivel para u_x, u_y y presion.
**Depende de:** Fase 4
**Estado:** [x] Completada

**Pasos:**

21. [x] **Crear `src/evaluation/generar_gifs.py`** — Generacion de GIFs:
    - Cargar modelo entrenado desde checkpoint
    - Predecir sobre toda la malla (modelo en eval, torch.no_grad)
    - Para cada variable (u_x, u_y, p):
      - Por cada timestep: reshape a malla 2D, generar contourf
      - Overlay: scatter de las estaciones (observaciones reales)
      - FuncAnimation con PillowWriter para generar GIF
      - FPS: 60, DPI: 120
    - Guardar en directorio `reports/`
    - Referencia: `trabajo_base_pinns/cuadernos/elaboracion_gifs.ipynb` funcion generar_gif

22. [x] **Crear `src/evaluation/metricas.py`** — Metricas de evaluacion:
    - MSE por variable (u, v, p) en puntos de estaciones
    - Residual Navier-Stokes promedio

23. [x] **Commit:** `feat: evaluacion — GIFs de curvas de nivel y metricas`

**Archivos a crear:**
- `src/evaluation/generar_gifs.py`
- `src/evaluation/metricas.py`

---

### Fase 6 — Ejecucion del entrenamiento y validacion iterativa

**Objetivo:** Entrenar, evaluar y ajustar iterativamente hasta que la red aprenda.
**Depende de:** Fase 5
**Estado:** [ ] Pendiente

**Pasos:**

24. [ ] **Ejecutar entrenamiento inicial** (2000 epocas)
    - Monitorear logs: verificar que loss total, loss_ns, loss_u, loss_v, loss_p disminuyen
    - Verificar curva de aprendizaje
25. [ ] **Generar GIFs** con el mejor checkpoint
    - Evaluar visualmente si las predicciones tienen sentido fisico
26. [ ] **Si el entrenamiento se estanca** (las perdidas no bajan), explorar:
    - Ajustar lambda (peso de la perdida fisica)
    - Modificar learning rate o scheduler
    - Aumentar/reducir epocas
    - Ajustar tamano de batch
    - Learning rate warmup
    - Cambiar inicializacion de pesos
    - Normalizacion alternativa de datos
    - Curriculum learning (entrenar primero solo datos, luego agregar fisica)
    - Aumentar/reducir la complejidad de la red
27. [ ] **Registrar cada iteracion** en `solutions/INFORME_entrenamiento.md`
28. [ ] **Commit por cada iteracion significativa**

---

## Archivos relevantes del proyecto base (referencia)

| Archivo del proyecto base | Que reutilizar |
|--------------------------|----------------|
| `trabajo_base_pinns/src/process_data.py` | Pipeline de preprocesamiento: proyeccion cartesiana, correccion ISA, centrado, adimensionalizacion, construccion de malla |
| `trabajo_base_pinns/src/pre_process_data.py` | Descomposicion viento (ya aplicada), extraccion de datos |
| `trabajo_base_pinns/src/model_pinn.py` | Patron GammaBiasLayer + estructura PINN (adaptar a 12x1200 con 8 tanh + 4 lineal) |
| `trabajo_base_pinns/src/loss_functions.py` | compute_derivatives + loss_navier_stokes (Euler inviscido) |
| `trabajo_base_pinns/src/train_pinn.py` | Bucle de entrenamiento, funcion de perdida combinada, checkpointing |
| `trabajo_base_pinns/src/dataset.py` | StationDataset, CollocationDataset |
| `trabajo_base_pinns/cuadernos/elaboracion_gifs.ipynb` | Funcion generar_gif con contourf + scatter + FuncAnimation |
| `trabajo_base_pinns/config/logging_config.py` | Configuracion dual de logging |
| `trabajo_base_pinns/main.py` | Estructura del punto de entrada |

---

## Criterios de verificacion

| Fase | Verificacion |
|------|-------------|
| Fase 1 | `git log` muestra commit inicial; `pip install -r requirements.txt` sin errores; `import torch; print(torch.cuda.is_available())` retorna True |
| Fase 2 | Script de prueba que carga datos, preprocesa, normaliza y construye malla; verificar dimensiones (N_estaciones x N_tiempos); malla cubre el dominio |
| Fase 3 | Instanciar PINN(3, 3, 1200) y pasar tensor torch.randn(10, 3); verificar salida shape (10, 3); contar parametros totales |
| Fase 4 | Ejecutar 5 epocas de prueba; verificar que loss se imprime y checkpoints se guardan |
| Fase 5 | Generar GIF con modelo de prueba (random); verificar que el archivo .gif se crea |
| Fase 6 | Entrenamiento completo 2000 epocas; loss converge; GIFs muestran patrones coherentes |

---

## Diferencias con el proyecto base

| Aspecto | Proyecto base | Este proyecto |
|---------|--------------|---------------|
| Capas ocultas | 12 (11 tanh + 1 lineal) | 12 (8 tanh + 4 lineal) |
| Neuronas | 600 | 1200 |
| Epocas | 4000 | 2000 |
| R malla | 0.15 grados | 0.1 grados |
| Checkpoint | cada num_epochs/4 | cada 500 epocas |
| Estructura | monolitico | modularizado (src/data, src/model, src/training, src/evaluation) |

---

## Registro dinamico de cambios

| Fecha | Cambio | Estado |
|-------|--------|--------|
| 2026-04-09 | Plan inicial creado y aprobado | Completado |
| 2026-04-09 | Fase 1: estructura, git, venv, config, logging | Completado |
| 2026-04-09 | Fase 2: pipeline datos (carga, preproceso, normalizacion, malla, datasets) | Completado |
| 2026-04-09 | Fase 3: arquitectura PINN (GammaBiasLayer, red 12x1200, perdida NS) | Completado |
| 2026-04-09 | Fase 4: entrenamiento (Adam, ReduceLROnPlateau, checkpoints, main.py) | Completado |
| 2026-04-09 | Fase 5: evaluacion (GIFs contourf + scatter, metricas MSE + residual NS) | Completado |
| 2026-04-09 | PyTorch 2.6.0+cu124 instalado (driver NVIDIA 535 no compatible con cu130) | Nota |
| 2026-04-09 | Verificacion: 22 est, 744 timesteps, 320 pts malla, 238K colocacion, 15.9M params | Nota |
| 2026-04-09 | Test 2 epocas OK en CUDA: loss 2.80e+00 -> 2.60e+00 | Nota |
