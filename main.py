"""Punto de entrada principal del proyecto PINN."""

import argparse
import logging

import torch

from config.experimentos import (
    ConfiguracionExperimento,
    construir_configuracion_experimento,
)
from config.logging_config import configurar_logging
from config.settings import (
    HIDDEN_NEURONS,
    LAMBDA_FISICA,
    NUM_EPOCAS,
    R_MALLA,
)
from src.data.cargar_datos import cargar_parquet
from src.data.preprocesar_datos import preprocesar
from src.data.normalizar_datos import normalizar
from src.data.construir_malla import construir_malla
from src.data.datasets import EstacionesDataset, ColocacionDataset
from src.model.pinn import PINN
from src.training.entrenador import entrenar
from src.training.manifiestos import guardar_manifiesto_experimento


logger = logging.getLogger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de linea de comandos.

    :return: parser del entrenamiento principal.
    """
    parser = argparse.ArgumentParser(description="Entrenamiento principal de la PINN")
    parser.add_argument("--nombre-experimento", type=str, default=None,
                        help="Nombre logico de la corrida")
    parser.add_argument("--nombre-modelo", type=str, default=None,
                        help="Prefijo de checkpoints e historial")
    parser.add_argument("--nombre-log", type=str, default=None,
                        help="Nombre del archivo de log")
    parser.add_argument("--r-malla", type=float, default=None,
                        help="Resolucion de la malla en grados")
    parser.add_argument("--lambda-fisica", type=float, default=None,
                        help="Peso fijo de la perdida fisica")
    parser.add_argument("--num-epocas", type=int, default=None,
                        help="Numero de epocas del entrenamiento")
    parser.add_argument("--registros-por-estacion", type=int, default=None,
                        help="Limite de registros por estacion")
    return parser


def construir_configuracion_desde_args(
    args: argparse.Namespace,
) -> ConfiguracionExperimento:
    """Traduce argumentos CLI a configuracion interna.

    :param args: argumentos parseados.
    :return: configuracion reusable del experimento.
    """
    return construir_configuracion_experimento(
        nombre_experimento=args.nombre_experimento,
        nombre_modelo=args.nombre_modelo,
        nombre_log=args.nombre_log,
        r_malla=args.r_malla if args.r_malla is not None else R_MALLA,
        lambda_fisica=(
            args.lambda_fisica if args.lambda_fisica is not None else LAMBDA_FISICA
        ),
        num_epocas=args.num_epocas if args.num_epocas is not None else NUM_EPOCAS,
        registros_por_estacion=args.registros_por_estacion,
    )


def main(configuracion: ConfiguracionExperimento):
    """Ejecuta el pipeline completo de la PINN.

    :param configuracion: parametros de la corrida actual.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Experimento: {configuracion.nombre_experimento}")

    # ── 1. Carga de datos ────────────────────────────────────────────
    datos = cargar_parquet(
        configuracion.archivo_parquet,
        estaciones_excluidas=configuracion.estaciones_excluidas,
        registros_por_estacion=configuracion.registros_por_estacion,
    )

    # ── 2. Preprocesamiento ──────────────────────────────────────────
    datos_prep = preprocesar(
        datos,
        n_dias=configuracion.n_dias,
        intervalo=configuracion.intervalo,
    )

    # ── 3. Normalizacion ─────────────────────────────────────────────
    datos_norm, escalas = normalizar(
        datos_prep,
        reescalar_presion=configuracion.reescalar_presion,
    )

    # ── 4. Construccion de malla ─────────────────────────────────────
    malla = construir_malla(datos_norm, escalas, R=configuracion.r_malla)
    ruta_manifiesto = guardar_manifiesto_experimento(configuracion, datos, malla)
    logger.info(f"Manifiesto guardado: {ruta_manifiesto.name}")

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
        lr=configuracion.lr_inicial,
        lamb=configuracion.lambda_fisica,
        num_epocas=configuracion.num_epocas,
        max_norm=configuracion.max_norm_grad,
        checkpoint_intervalo=configuracion.checkpoint_intervalo,
        nombre_modelo=configuracion.nombre_modelo,
        p_scale=p_scale,
        epocas_solo_datos=configuracion.epocas_solo_datos,
        epocas_rampa=configuracion.epocas_rampa,
        r_malla=configuracion.r_malla,
        n_dias=configuracion.n_dias,
    )


if __name__ == "__main__":
    argumentos = construir_parser().parse_args()
    configuracion = construir_configuracion_desde_args(argumentos)
    configurar_logging(configuracion.nombre_log)
    try:
        main(configuracion)
    except KeyboardInterrupt:
        print()
        logger.info("Proceso interrumpido manualmente")
