"""
Constantes centralizadas del proyecto PINN — Caribe Colombia.

Todas las constantes de dominio, hiperparametros y rutas se definen aqui.
No se deben hardcodear valores en otros modulos.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Rutas del proyecto
# ──────────────────────────────────────────────────────────────────────
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS_RAW = RAIZ_PROYECTO / "data" / "raw"
RUTA_DATOS_PROCESSED = RAIZ_PROYECTO / "data" / "processed"
RUTA_MODELOS = RAIZ_PROYECTO / "models"
RUTA_LOGS = RAIZ_PROYECTO / "logs"
RUTA_REPORTES = RAIZ_PROYECTO / "reports"

ARCHIVO_PARQUET = RUTA_DATOS_RAW / "em_caribe_20251201_20251231.parquet"

# ──────────────────────────────────────────────────────────────────────
# Constantes fisicas
# ──────────────────────────────────────────────────────────────────────
RADIO_TIERRA = 6_378_000       # metros
RHO_AIRE = 1.18                # kg/m3
NU_AIRE = 1.382e-5             # m2/s (viscosidad cinematica)

# ──────────────────────────────────────────────────────────────────────
# Parametros de la malla
# ──────────────────────────────────────────────────────────────────────
R_MALLA = 0.1                  # grados (resolucion de la malla)

# ──────────────────────────────────────────────────────────────────────
# Parametros de datos
# ──────────────────────────────────────────────────────────────────────
N_DIAS = 31                    # dias de diciembre 2025
INTERVALO = 1                  # cada registro (1 h)

# ──────────────────────────────────────────────────────────────────────
# Hiperparametros de entrenamiento
# ──────────────────────────────────────────────────────────────────────
LAMBDA_FISICA = 3.0            # peso de la perdida fisica
NUM_EPOCAS = 2000
LR_INICIAL = 1e-4
MAX_NORM_GRAD = 0.5

# Scheduler ReduceLROnPlateau
SCHEDULER_FACTOR = 0.3162
SCHEDULER_PATIENCE = 50
SCHEDULER_THRESHOLD = 1e-3

# Checkpoints
CHECKPOINT_INTERVALO = 500     # guardar cada N epocas

# ──────────────────────────────────────────────────────────────────────
# Arquitectura de la red
# ──────────────────────────────────────────────────────────────────────
INPUT_DIM = 3                  # (t, x, y)
OUTPUT_DIM = 3                 # (u_x, u_y, p)
HIDDEN_NEURONS = 1200
NUM_CAPAS_TANH = 8             # capas ocultas con activacion Tanh
NUM_CAPAS_LINEALES = 4         # capas ocultas sin activacion
