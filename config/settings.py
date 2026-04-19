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
R_MALLA = 0.4                  # grados (resolucion de la malla)

# ──────────────────────────────────────────────────────────────────────
# Parametros de datos
# ──────────────────────────────────────────────────────────────────────
N_DIAS = 31                    # dias de diciembre 2025
INTERVALO = 1                  # cada registro (1 h)
ESTACIONES_EXCLUIDAS = [       # estaciones a excluir del entrenamiento
    "0025025380",              # presion anomalamente alta respecto al grupo
    "0025025030",              # alto error de u en evaluacion por estacion-tiempo
    "0025025190",              # alto error de u en evaluacion por estacion-tiempo
    "0015015100",              # alto error de u en evaluacion por estacion-tiempo
]

NOMBRE_MODELO_ENTRENAMIENTO = "PINN_caribe_excl_4est_R04_bg"
NOMBRE_LOG_ENTRENAMIENTO = "entrenamiento_excl_4est_R04_bg.log"

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

# Curriculum learning
EPOCAS_SOLO_DATOS = 0          # epocas iniciales solo con datos (lambda=0)
EPOCAS_RAMPA = 0               # epocas de rampa lineal 0 -> LAMBDA_FISICA
REESCALAR_PRESION = False       # dividir P_norm por sigma_p para equilibrar rangos

# ──────────────────────────────────────────────────────────────────────
# Arquitectura de la red
# ──────────────────────────────────────────────────────────────────────
INPUT_DIM = 3                  # (t, x, y)
OUTPUT_DIM = 3                 # (u_x, u_y, p)
HIDDEN_NEURONS = 1200
NUM_CAPAS_TANH = 8             # capas ocultas con activacion Tanh
NUM_CAPAS_LINEALES = 4         # capas ocultas sin activacion
