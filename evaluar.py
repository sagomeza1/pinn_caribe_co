"""
Script de evaluacion: genera GIFs y calcula metricas desde un checkpoint.

Uso:
    python evaluar.py
    python evaluar.py --epoca 1500
"""

import argparse
import logging

import torch

from config.logging_config import configurar_logging
from config.settings import (
    ARCHIVO_PARQUET, HIDDEN_NEURONS, R_MALLA, N_DIAS, INTERVALO,
    RUTA_MODELOS,
)
from src.data.cargar_datos import cargar_parquet
from src.data.preprocesar_datos import preprocesar
from src.data.normalizar_datos import normalizar
from src.data.construir_malla import construir_malla
from src.data.datasets import EstacionesDataset
from src.model.pinn import PINN
from src.evaluation.generar_gifs import generar_todos_los_gifs
from src.evaluation.metricas import evaluar_metricas


def main(epoca: int = 2000, fps: int = 60):
    """
    Evalua el modelo PINN cargando un checkpoint especifico.

    :param epoca: epoca del checkpoint a cargar.
    :param fps: frames por segundo para los GIFs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    # ── 1. Cargar y preparar datos (mismo pipeline que main.py) ──────
    datos = cargar_parquet(ARCHIVO_PARQUET)
    datos_prep = preprocesar(datos, n_dias=N_DIAS, intervalo=INTERVALO)
    datos_norm, escalas = normalizar(datos_prep)
    malla = construir_malla(datos_norm, escalas, R=R_MALLA)
    dataset_estaciones = EstacionesDataset(datos_norm)

    logger.info(f"Datos: {dataset_estaciones.t.shape[0]:,} puntos de estaciones")
    logger.info(f"Malla: {malla['dim_N']} espaciales x {malla['dim_T']} timesteps")

    # ── 2. Cargar modelo desde checkpoint ────────────────────────────
    ruta_checkpoint = RUTA_MODELOS / f"PINN_caribe_epoca_{epoca}.pth"
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ruta_checkpoint}")

    modelo = PINN(hidden_neurons=HIDDEN_NEURONS)
    checkpoint = torch.load(ruta_checkpoint, map_location=device, weights_only=False)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(device)
    modelo.eval()

    logger.info(f"Modelo cargado desde {ruta_checkpoint.name} (epoca {checkpoint.get('epoch', epoca)})")

    # ── 3. Metricas ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUACION DE METRICAS")
    logger.info("=" * 60)
    metricas = evaluar_metricas(modelo, dataset_estaciones, device)

    logger.info("-" * 40)
    logger.info(f"MSE u_x:         {metricas['mse_u']:.6f}")
    logger.info(f"MSE u_y:         {metricas['mse_v']:.6f}")
    logger.info(f"MSE presion:     {metricas['mse_p']:.6f}")
    logger.info(f"Residual NS:     {metricas['residual_ns']:.6f}")
    logger.info("-" * 40)

    # ── 4. Generacion de GIFs ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GENERACION DE GIFs")
    logger.info("=" * 60)
    generar_todos_los_gifs(
        modelo=modelo,
        datos_norm=datos_norm,
        malla=malla,
        escalas=escalas,
        device=device,
        prefijo=f"pinn_epoca_{epoca}",
        fps=fps,
    )

    logger.info("Evaluacion completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluacion PINN")
    parser.add_argument("--epoca", type=int, default=2000,
                        help="Epoca del checkpoint a evaluar (default: 2000)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Frames por segundo de los GIFs (default: 60)")
    args = parser.parse_args()

    configurar_logging()
    logger = logging.getLogger(__name__)
    try:
        main(epoca=args.epoca, fps=args.fps)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error: {e}", exc_info=True)
        raise
