"""CLI para listar o ejecutar el barrido secuencial R-lambda."""

import argparse
import logging

from config.logging_config import configurar_logging
from config.settings import RAIZ_PROYECTO
from src.evaluation.resumen_barrido import exportar_resumen_barrido_csv
from src.training.barrido import (
    ejecutar_barrido_secuencial,
    generar_configuraciones_barrido,
    VALORES_LAMBDA_BARRIDO,
    VALORES_R_BARRIDO,
)


logger = logging.getLogger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser del runner del barrido.

    :return: parser CLI.
    """
    parser = argparse.ArgumentParser(description="Barrido secuencial R-lambda")
    parser.add_argument("--num-epocas", type=int, default=2000,
                        help="Numero de epocas por corrida")
    parser.add_argument("--registros-por-estacion", type=int, default=300,
                        help="Cantidad fija de registros por estacion")
    parser.add_argument(
        "--valores-r",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Subconjunto de valores R a ejecutar en esta iteracion. "
            f"Opciones validas: {VALORES_R_BARRIDO}"
        ),
    )
    parser.add_argument(
        "--valores-lambda",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Subconjunto de valores lambda a ejecutar en esta iteracion. "
            f"Opciones validas: {VALORES_LAMBDA_BARRIDO}"
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo listar los comandos del barrido")
    parser.add_argument("--nombre-log", type=str, default="barrido_r_lambda.log",
                        help="Nombre del log del runner")
    parser.add_argument(
        "--consolidar-al-final",
        action="store_true",
        help="Regenera el CSV comparativo si el barrido termina sin errores",
    )
    parser.add_argument(
        "--ruta-csv",
        type=str,
        default=str(RAIZ_PROYECTO / "solutions" / "resumen_barrido_r_lambda_300xest.csv"),
        help="Ruta del CSV comparativo a regenerar",
    )
    return parser


def main() -> None:
    """Genera y ejecuta el barrido segun la configuracion CLI."""
    args = construir_parser().parse_args()
    configurar_logging(args.nombre_log)

    configuraciones = generar_configuraciones_barrido(
        registros_por_estacion=args.registros_por_estacion,
        num_epocas=args.num_epocas,
        valores_r=args.valores_r,
        valores_lambda=args.valores_lambda,
    )
    comandos = ejecutar_barrido_secuencial(
        configuraciones=configuraciones,
        dry_run=args.dry_run,
    )
    logger.info("Corridas preparadas: %s", len(comandos))

    if args.consolidar_al_final and not args.dry_run:
        ruta_csv = exportar_resumen_barrido_csv(args.ruta_csv)
        logger.info("CSV comparativo regenerado: %s", ruta_csv)


if __name__ == "__main__":
    main()