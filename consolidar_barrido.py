"""CLI para consolidar el resumen comparativo del barrido R-lambda."""

import argparse
import logging

from pathlib import Path

from config.logging_config import configurar_logging
from config.settings import RAIZ_PROYECTO
from src.evaluation.resumen_barrido import exportar_resumen_barrido_csv


logger = logging.getLogger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI del consolidado del barrido.

    :return: parser listo para ejecutarse.
    """
    parser = argparse.ArgumentParser(description="Consolidar resumen comparativo del barrido")
    parser.add_argument(
        "--salida",
        type=Path,
        default=RAIZ_PROYECTO / "solutions" / "resumen_barrido_r_lambda_300xest.csv",
        help="Ruta del CSV de salida",
    )
    parser.add_argument(
        "--nombre-log",
        type=str,
        default="consolidar_barrido.log",
        help="Nombre del log del consolidado",
    )
    return parser


def main() -> None:
    """Ejecuta la exportacion del resumen comparativo."""
    args = construir_parser().parse_args()
    configurar_logging(args.nombre_log)
    ruta_csv = exportar_resumen_barrido_csv(args.salida)
    logger.info("CSV generado en %s", ruta_csv)


if __name__ == "__main__":
    main()