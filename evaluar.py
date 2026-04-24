"""Script de evaluacion reproducible desde checkpoint o manifiesto."""

import argparse
import json
import logging

from pathlib import Path

import torch

from config.logging_config import configurar_logging
from config.settings import (
    ARCHIVO_PARQUET, ESTACIONES_EXCLUIDAS, HIDDEN_NEURONS, INTERVALO,
    N_DIAS, R_MALLA, RUTA_MODELOS, RUTA_REPORTES,
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
from src.training.manifiestos import cargar_manifiesto_experimento


logger = logging.getLogger(__name__)


def _guardar_metricas_evaluacion(
    ruta_salida: Path,
    nombre_modelo: str,
    nombre_experimento: str | None,
    epoca: int,
    metricas: dict,
    ruta_checkpoint: Path,
    ruta_reporte: Path,
) -> Path:
    """Guarda las metricas finales de evaluacion en JSON.

    :param ruta_salida: archivo JSON destino.
    :param nombre_modelo: nombre del modelo evaluado.
    :param nombre_experimento: nombre logico del experimento.
    :param epoca: epoca evaluada.
    :param metricas: metricas calculadas.
    :param ruta_checkpoint: checkpoint usado en la evaluacion.
    :param ruta_reporte: carpeta de artefactos de evaluacion.
    :return: ruta final del JSON escrito.
    """
    contenido = {
        "nombre_modelo": nombre_modelo,
        "nombre_experimento": nombre_experimento,
        "epoca": epoca,
        "mse_u": float(metricas["mse_u"]),
        "mse_v": float(metricas["mse_v"]),
        "mse_p": float(metricas["mse_p"]),
        "residual_ns": float(metricas["residual_ns"]),
        "ruta_checkpoint": str(ruta_checkpoint),
        "ruta_reporte": str(ruta_reporte),
    }
    with ruta_salida.open("w", encoding="utf-8") as archivo_salida:
        json.dump(contenido, archivo_salida, indent=2)
    return ruta_salida


def _resolver_ruta_reporte(
    carpeta_reportes: str | None,
    nombre_experimento: str | None,
) -> Path:
    """Resuelve la subcarpeta de reports para la evaluacion.

    :param carpeta_reportes: carpeta explicita pedida por CLI.
    :param nombre_experimento: nombre del experimento si existe manifiesto.
    :return: ruta donde se guardaran los artefactos.
    """
    if carpeta_reportes:
        return RUTA_REPORTES / carpeta_reportes

    if nombre_experimento:
        return RUTA_REPORTES / nombre_experimento

    return RUTA_REPORTES / "sin_iteracion"


def _resolver_contexto_evaluacion(
    manifiesto: dict | None,
    prefijo: str,
    epoca: int | None,
) -> dict:
    """Consolida los parametros efectivos de evaluacion.

    :param manifiesto: diccionario del manifiesto o None.
    :param prefijo: prefijo de checkpoint en modo legado.
    :param epoca: epoca pedida por CLI o None.
    :return: diccionario con la configuracion a usar.
    :raises ValueError: si no se puede determinar la epoca.
    """
    if manifiesto is None:
        epoca_efectiva = 2000 if epoca is None else epoca
        return {
            "ruta_parquet": ARCHIVO_PARQUET,
            "r_malla": R_MALLA,
            "n_dias": N_DIAS,
            "intervalo": INTERVALO,
            "estaciones_excluidas": ESTACIONES_EXCLUIDAS,
            "registros_por_estacion": None,
            "nombre_modelo": prefijo,
            "nombre_experimento": None,
            "epoca": epoca_efectiva,
        }

    epoca_efectiva = manifiesto.get("num_epocas") if epoca is None else epoca
    if epoca_efectiva is None:
        raise ValueError("No fue posible determinar la epoca a evaluar")

    return {
        "ruta_parquet": Path(manifiesto["archivo_parquet"]),
        "r_malla": float(manifiesto["r_malla"]),
        "n_dias": int(manifiesto["n_dias"]),
        "intervalo": int(manifiesto["intervalo"]),
        "estaciones_excluidas": list(manifiesto.get("estaciones_excluidas", [])),
        "registros_por_estacion": manifiesto.get("registros_por_estacion"),
        "nombre_modelo": str(manifiesto["nombre_modelo"]),
        "nombre_experimento": str(manifiesto.get("nombre_experimento", "")),
        "epoca": int(epoca_efectiva),
    }


def main(
    epoca: int | None = None,
    fps: int = 60,
    prefijo: str = "PINN_caribe",
    ruta_manifiesto: Path | None = None,
    carpeta_reportes: str | None = None,
):
    """
    Evalua el modelo PINN cargando un checkpoint especifico.

    :param epoca: epoca del checkpoint a cargar.
    :param fps: frames por segundo para los GIFs.
    :param prefijo: prefijo del nombre del checkpoint (PINN_caribe o PINN_caribe_v2).
    :param ruta_manifiesto: ruta opcional del manifiesto del experimento.
    :param carpeta_reportes: subcarpeta destino dentro de reports/.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    manifiesto = None
    if ruta_manifiesto is not None:
        manifiesto = cargar_manifiesto_experimento(ruta_manifiesto)

    contexto = _resolver_contexto_evaluacion(manifiesto, prefijo, epoca)
    ruta_reporte = _resolver_ruta_reporte(
        carpeta_reportes=carpeta_reportes,
        nombre_experimento=contexto["nombre_experimento"],
    )
    ruta_reporte.mkdir(parents=True, exist_ok=True)
    logger.info(f"Reportes de evaluacion: {ruta_reporte}")

    # ── 1. Cargar checkpoint primero para obtener p_scale ────────────
    ruta_checkpoint = (
        RUTA_MODELOS / f"{contexto['nombre_modelo']}_epoca_{contexto['epoca']}.pth"
    )
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ruta_checkpoint}")

    checkpoint = torch.load(ruta_checkpoint, map_location=device, weights_only=False)
    p_scale = checkpoint.get("p_scale", 1.0)
    reescalar = abs(p_scale - 1.0) > 1e-6
    logger.info(f"Checkpoint: {ruta_checkpoint.name}, p_scale={p_scale:.4f}")

    # ── 2. Cargar y preparar datos ───────────────────────────────────
    datos = cargar_parquet(
        contexto["ruta_parquet"],
        estaciones_excluidas=contexto["estaciones_excluidas"],
        registros_por_estacion=contexto["registros_por_estacion"],
    )
    datos_prep = preprocesar(
        datos,
        n_dias=contexto["n_dias"],
        intervalo=contexto["intervalo"],
    )
    datos_norm, escalas = normalizar(datos_prep, reescalar_presion=reescalar)
    malla = construir_malla(datos_norm, escalas, R=contexto["r_malla"])
    dataset_estaciones = EstacionesDataset(datos_norm)

    logger.info(f"Datos: {dataset_estaciones.t.shape[0]:,} puntos de estaciones")
    logger.info(f"Malla: {malla['dim_N']} espaciales x {malla['dim_T']} timesteps")

    # ── 3. Cargar modelo ─────────────────────────────────────────────
    modelo = PINN(hidden_neurons=HIDDEN_NEURONS)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(device)
    modelo.eval()

    logger.info(
        "Modelo cargado desde %s (epoca %s)",
        ruta_checkpoint.name,
        checkpoint.get("epoch", contexto["epoca"]),
    )

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

    ruta_metricas = RUTA_MODELOS / (
        f"metricas_{contexto['nombre_modelo']}_epoca_{contexto['epoca']}.json"
    )
    ruta_metricas = _guardar_metricas_evaluacion(
        ruta_salida=ruta_metricas,
        nombre_modelo=contexto["nombre_modelo"],
        nombre_experimento=contexto["nombre_experimento"],
        epoca=contexto["epoca"],
        metricas=metricas,
        ruta_checkpoint=ruta_checkpoint,
        ruta_reporte=ruta_reporte,
    )
    logger.info(f"Metricas de evaluacion guardadas: {ruta_metricas.name}")

    rutas_graficas = graficar_error_promedio_temporal(
        modelo=modelo,
        dataset_estaciones=dataset_estaciones,
        device=device,
        ruta_reporte=ruta_reporte,
        prefijo_archivo=f"{contexto['nombre_modelo'].lower()}_epoca_{contexto['epoca']}",
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
        ruta_reporte=ruta_reporte,
        prefijo_archivo=f"{contexto['nombre_modelo'].lower()}_epoca_{contexto['epoca']}",
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
        prefijo=f"{contexto['nombre_modelo'].lower()}_epoca_{contexto['epoca']}",
        ruta_salida=ruta_reporte,
        fps=fps,
    )

    logger.info("Evaluacion completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluacion PINN")
    parser.add_argument("--epoca", type=int, default=None,
                        help="Epoca del checkpoint a evaluar; con manifiesto usa num_epocas por defecto")
    parser.add_argument("--fps", type=int, default=60,
                        help="Frames por segundo de los GIFs (default: 60)")
    parser.add_argument("--prefijo", type=str, default="PINN_caribe_v2",
                        help="Prefijo del checkpoint en modo legado (default: PINN_caribe_v2)")
    parser.add_argument("--manifiesto", type=Path, default=None,
                        help="Ruta al manifiesto JSON del experimento a reconstruir")
    parser.add_argument("--carpeta-reportes", type=str, default=None,
                        help="Subcarpeta dentro de reports/ para guardar artefactos")
    args = parser.parse_args()

    configurar_logging()
    try:
        main(
            epoca=args.epoca,
            fps=args.fps,
            prefijo=args.prefijo,
            ruta_manifiesto=args.manifiesto,
            carpeta_reportes=args.carpeta_reportes,
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"Error: {e}", exc_info=True)
        raise
