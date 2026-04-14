# PINN Prediccion Meteorologica Caribe Colombia

Proyecto de red neuronal informada fisicamente (PINN) para predecir condiciones meteorologicas en una malla espacial del Caribe colombiano usando observaciones de estaciones y restricciones de Navier-Stokes.

## Objetivo

Predecir sobre una malla espacio-temporal:

- Velocidad del viento en X (u_x)
- Velocidad del viento en Y (u_y)
- Presion (p)

Entradas del modelo:

- Posicion X
- Posicion Y
- Registro temporal (t)

## Alcance y lineamientos del proyecto

- Dominio geografico: Caribe de Colombia.
- Fuente de datos: Datos Abiertos Colombia (trazabilidad preservada).
- Marco fisico obligatorio: ecuaciones de Navier-Stokes.
- Balance de aprendizaje: observaciones y perdida fisica se usan conjuntamente.
- Convencion de trabajo: comentarios, docstrings y logs en espanol.

## Estado actual (13 abril 2026)

- Fase 1: Infraestructura y configuracion - Completada
- Fase 2: Pipeline de datos - Completada
- Fase 3: Arquitectura PINN - Completada
- Fase 4: Entrenamiento base (2000 epocas) - Completada
- Fase 5: Evaluacion y visualizacion - Completada
- Fase 6: Iteracion y ajuste - Iteracion 1 completada, iteracion 2 pendiente

Metricas de referencia (iteracion 1):

- Loss total final: 1.001
- MSE u_x: 0.671820
- MSE u_y: 0.627913
- MSE presion: 1.423412
- Residual Navier-Stokes: 0.020416

Diagnostico de iteracion 1:

- Se observa convergencia de loss.
- El modelo presenta comportamiento cuasi-estacionario en las velocidades.
- La presion domina la perdida y sigue siendo el principal cuello de botella.

## Arquitectura del modelo

- PINN con 12 capas ocultas de 1200 neuronas.
- Capa personalizada GammaBiasLayer con Weight Normalization.
- Activaciones: 8 capas Tanh + 4 capas lineales.
- Salida: 3 variables (u_x, u_y, p).
- Perdida fisica: continuidad + momento X + momento Y (Navier-Stokes inviscido).

## Estructura del repositorio

```text
pinn/
  config/
    settings.py
    logging_config.py
  data/
    raw/
  docs/
  logs/
  models/
  notebooks/
  reports/
  solutions/
  src/
    data/
    model/
    training/
    evaluation/
  main.py
  evaluar.py
  requirements.txt
```

## Requisitos

- Linux (recomendado)
- Python 3.12
- Entorno virtual (venv)
- GPU CUDA (opcional, recomendado para entrenamiento)

Dependencias en [requirements.txt](requirements.txt).

## Instalacion

1. Crear y activar entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecucion

### Entrenamiento

```bash
source .venv/bin/activate
python main.py
```

### Evaluacion (GIFs + metricas)

```bash
source .venv/bin/activate
python evaluar.py
```

## Artefactos generados

- Checkpoints en [models](models):
  - PINN_caribe_epoca_500.pth
  - PINN_caribe_epoca_1000.pth
  - PINN_caribe_epoca_1500.pth
  - PINN_caribe_epoca_2000.pth
- Historial de entrenamiento: [models/historial_PINN_caribe.mat](models/historial_PINN_caribe.mat)
- Reportes visuales en [reports](reports):
  - GIFs por variable
  - curvas de perdida
  - graficas de frames de referencia

## Documentacion de seguimiento

- Definicion del proyecto: [docs/01_definicion_proyecto.md](docs/01_definicion_proyecto.md)
- Instrucciones canonicas: [docs/02_instrucciones_canonicas.md](docs/02_instrucciones_canonicas.md)
- Plan de proyecto: [docs/03_plan_proyecto.md](docs/03_plan_proyecto.md)
- Estado consolidado: [docs/04_estado_proyecto_2026-04-13.md](docs/04_estado_proyecto_2026-04-13.md)
- Informe iteracion 1: [solutions/INFORME_entrenamiento.md](solutions/INFORME_entrenamiento.md)

## Proximo paso recomendado

Implementar iteracion 2 de entrenamiento con curriculum learning y/o reescalado de presion, luego comparar metricas y comportamiento temporal frente a iteracion 1.
