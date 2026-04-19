"""
Punto de entrada principal del proyecto PINN.

Ejecuta el pipeline completo:
    1. Carga de datos
    2. Preprocesamiento y proyeccion cartesiana
    3. Normalizacion (adimensionalizacion)
    4. Construccion de malla
    5. Entrenamiento de la red PINN
"""

import logging

import numpy as np
import torch

from config.logging_config import configurar_logging
from config.settings import (
    ARCHIVO_PARQUET, LAMBDA_FISICA, NUM_EPOCAS, LR_INICIAL,
    MAX_NORM_GRAD, HIDDEN_NEURONS, R_MALLA, N_DIAS, INTERVALO,
    CHECKPOINT_INTERVALO, REESCALAR_PRESION,
    EPOCAS_SOLO_DATOS, EPOCAS_RAMPA,
    NOMBRE_MODELO_ENTRENAMIENTO, NOMBRE_LOG_ENTRENAMIENTO,
)
from src.data.cargar_datos import cargar_parquet
from src.data.preprocesar_datos import preprocesar
from src.data.normalizar_datos import normalizar
from src.data.construir_malla import construir_malla
from src.data.datasets import EstacionesDataset, ColocacionDataset
from src.model.pinn import PINN
from src.training.entrenador import entrenar


def main():
    """Ejecuta el pipeline completo de la PINN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    # ── 1. Carga de datos ────────────────────────────────────────────
    datos = cargar_parquet(ARCHIVO_PARQUET)

    # ── 2. Preprocesamiento ──────────────────────────────────────────
    datos_prep = preprocesar(datos, n_dias=N_DIAS, intervalo=INTERVALO)

    # ── 3. Normalizacion ─────────────────────────────────────────────
    datos_norm, escalas = normalizar(datos_prep,
                                     reescalar_presion=REESCALAR_PRESION)

    # ── 4. Construccion de malla ─────────────────────────────────────
    malla = construir_malla(datos_norm, escalas, R=R_MALLA)

    # ── 5. Datasets ──────────────────────────────────────────────────
    dataset_estaciones = EstacionesDataset(datos_norm)
    dataset_colocacion = ColocacionDataset(malla)

    # ── 6. Modelo ────────────────────────────────────────────────────
    modelo = PINN(hidden_neurons=HIDDEN_NEURONS)
    total_params = sum(p.numel() for p in modelo.parameters())
    logger.info(f"Modelo PINN: {total_params:,} parametros")
    logger.debug(modelo)

    # ── 7. Entrenamiento ─────────────────────────────────────────────
    p_scale = escalas.get("P_scale", 1.0)
    entrenar(
        modelo=modelo,
        dataset_estaciones=dataset_estaciones,
        dataset_colocacion=dataset_colocacion,
        device=device,
        lr=LR_INICIAL,
        lamb=LAMBDA_FISICA,
        num_epocas=NUM_EPOCAS,
        max_norm=MAX_NORM_GRAD,
        checkpoint_intervalo=CHECKPOINT_INTERVALO,
        nombre_modelo=NOMBRE_MODELO_ENTRENAMIENTO,
        p_scale=p_scale,
        epocas_solo_datos=EPOCAS_SOLO_DATOS,
        epocas_rampa=EPOCAS_RAMPA,
    )


if __name__ == "__main__":
    configurar_logging(NOMBRE_LOG_ENTRENAMIENTO)
    logger = logging.getLogger(__name__)
    try:
        main()
    except KeyboardInterrupt:
        print()
        logger.info("Proceso interrumpido manualmente")
