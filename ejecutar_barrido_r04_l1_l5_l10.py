"""Lanzador dedicado para el barrido con R=0.4 y lambda en {1, 5, 10}."""

import logging

from pathlib import Path

from config.logging_config import configurar_logging
from config.settings import RAIZ_PROYECTO
from src.evaluation.resumen_barrido import exportar_resumen_barrido_csv
from src.training.barrido import ejecutar_barrido_secuencial, generar_configuraciones_barrido


logger = logging.getLogger(__name__)


def main() -> None:
    """Ejecuta la iteracion dedicada del barrido R=0.4 con lambdas 1, 5 y 10."""
    configurar_logging("barrido_R04_l1_l5_l10.log")

    configuraciones = generar_configuraciones_barrido(
        registros_por_estacion=300,
        num_epocas=2000,
        valores_r=[0.4],
        valores_lambda=[1.0, 5.0, 10.0],
    )
    comandos = ejecutar_barrido_secuencial(configuraciones=configuraciones, dry_run=False)
    logger.info("Corridas ejecutadas: %s", len(comandos))

    ruta_csv = exportar_resumen_barrido_csv(
        Path(RAIZ_PROYECTO / "solutions" / "resumen_barrido_r_lambda_300xest.csv")
    )
    logger.info("CSV comparativo regenerado: %s", ruta_csv)


if __name__ == "__main__":
    main()