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
    RUTA_MODELOS, RUTA_REPORTES,
)
from src.data.cargar_datos import cargar_parquet
from src.data.preprocesar_datos import preprocesar
from src.data.normalizar_datos import normalizar
from src.data.construir_malla import construir_malla
from src.data.datasets import EstacionesDataset
from src.model.pinn import PINN
from src.evaluation.generar_gifs import generar_todos_los_gifs
from src.evaluation.metricas import (
    evaluar_metricas,
    graficar_error_promedio_temporal,
    graficar_mse_por_estacion_en_tiempo,
)


def main(epoca: int = 2000, fps: int = 60, prefijo: str = "PINN_caribe"):
    """
    Evalua el modelo PINN cargando un checkpoint especifico.

    :param epoca: epoca del checkpoint a cargar.
    :param fps: frames por segundo para los GIFs.
    :param prefijo: prefijo del nombre del checkpoint (PINN_caribe o PINN_caribe_v2).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    # ── 1. Cargar checkpoint primero para obtener p_scale ────────────
    ruta_checkpoint = RUTA_MODELOS / f"{prefijo}_epoca_{epoca}.pth"
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ruta_checkpoint}")

    checkpoint = torch.load(ruta_checkpoint, map_location=device, weights_only=False)
    p_scale = checkpoint.get("p_scale", 1.0)
    reescalar = abs(p_scale - 1.0) > 1e-6
    logger.info(f"Checkpoint: {ruta_checkpoint.name}, p_scale={p_scale:.4f}")

    # ── 2. Cargar y preparar datos ───────────────────────────────────
    datos = cargar_parquet(ARCHIVO_PARQUET)
    datos_prep = preprocesar(datos, n_dias=N_DIAS, intervalo=INTERVALO)
    datos_norm, escalas = normalizar(datos_prep, reescalar_presion=reescalar)
    malla = construir_malla(datos_norm, escalas, R=R_MALLA)
    dataset_estaciones = EstacionesDataset(datos_norm)

    logger.info(f"Datos: {dataset_estaciones.t.shape[0]:,} puntos de estaciones")
    logger.info(f"Malla: {malla['dim_N']} espaciales x {malla['dim_T']} timesteps")

    # ── 3. Cargar modelo ─────────────────────────────────────────────
    modelo = PINN(hidden_neurons=HIDDEN_NEURONS)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(device)
    modelo.eval()

    logger.info(f"Modelo cargado desde {ruta_checkpoint.name} (epoca {checkpoint.get('epoch', epoca)})")

    # ── 4. Metricas ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUACION DE METRICAS")
    logger.info("=" * 60)
    metricas = evaluar_metricas(modelo, dataset_estaciones, device,
                                p_scale=p_scale)

    logger.info("-" * 40)
    logger.info(f"MSE u_x:         {metricas['mse_u']:.6f}")
    logger.info(f"MSE u_y:         {metricas['mse_v']:.6f}")
    logger.info(f"MSE presion:     {metricas['mse_p']:.6f}")
    if reescalar:
        logger.info(f"MSE presion (escala original): {metricas['mse_p'] * p_scale**2:.6f}")
    logger.info(f"Residual NS:     {metricas['residual_ns']:.6f}")
    logger.info("-" * 40)

    rutas_graficas = graficar_error_promedio_temporal(
        modelo=modelo,
        dataset_estaciones=dataset_estaciones,
        device=device,
        ruta_reporte=RUTA_REPORTES,
        prefijo_archivo=f"{prefijo.lower()}_epoca_{epoca}",
        p_scale=p_scale,
        escalas=escalas,
    )
    logger.info("Graficas de error promedio temporal generadas:")
    logger.info(f"- u: {rutas_graficas['u'].name}")
    logger.info(f"- v: {rutas_graficas['v'].name}")
    logger.info(f"- p: {rutas_graficas['p'].name}")

    rutas_mse_estacion = graficar_mse_por_estacion_en_tiempo(
        modelo=modelo,
        datos_norm=datos_norm,
        device=device,
        ruta_reporte=RUTA_REPORTES,
        prefijo_archivo=f"{prefijo.lower()}_epoca_{epoca}",
        p_scale=p_scale,
        escalas=escalas,
    )
    logger.info("Graficas de MSE por estacion en el tiempo generadas:")
    logger.info(f"- u: {rutas_mse_estacion['u'].name}")
    logger.info(f"- v: {rutas_mse_estacion['v'].name}")
    logger.info(f"- p: {rutas_mse_estacion['p'].name}")

    # ── 5. Generacion de GIFs ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GENERACION DE GIFs")
    logger.info("=" * 60)
    generar_todos_los_gifs(
        modelo=modelo,
        datos_norm=datos_norm,
        malla=malla,
        escalas=escalas,
        device=device,
        prefijo=f"{prefijo.lower()}_epoca_{epoca}",
        fps=fps,
    )

    logger.info("Evaluacion completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluacion PINN")
    parser.add_argument("--epoca", type=int, default=2000,
                        help="Epoca del checkpoint a evaluar (default: 2000)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Frames por segundo de los GIFs (default: 60)")
    parser.add_argument("--prefijo", type=str, default="PINN_caribe_v2",
                        help="Prefijo del checkpoint (default: PINN_caribe_v2)")
    args = parser.parse_args()

    configurar_logging()
    logger = logging.getLogger(__name__)
    try:
        main(epoca=args.epoca, fps=args.fps, prefijo=args.prefijo)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error: {e}", exc_info=True)
        raise
