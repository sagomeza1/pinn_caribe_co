"""
Configuracion dual de logging para el proyecto PINN.

- Consola: nivel INFO con formato simple.
- Archivo: nivel DEBUG con timestamp completo.
"""

import sys
import logging
import datetime

from config.settings import RUTA_LOGS


def configurar_logging(nombre_archivo: str = None) -> None:
    """
    Configura el sistema de logging con salida dual (consola + archivo).

    :param nombre_archivo: nombre del archivo de log. Si es None, se genera
                           automaticamente con la fecha y hora actual.
    """
    if nombre_archivo is None:
        ahora = datetime.datetime.now()
        nombre_archivo = f"pinn_{ahora.strftime('%y%m%d_%H%M')}.log"

    RUTA_LOGS.mkdir(parents=True, exist_ok=True)
    ruta_log = RUTA_LOGS / nombre_archivo

    # Formato detallado para archivo
    formato_archivo = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] (%(lineno)d) %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Formato simple para consola
    formato_consola = logging.Formatter(
        fmt="[%(asctime)s] (%(levelname)s) %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handler de consola (INFO+)
    handler_consola = logging.StreamHandler(sys.stdout)
    handler_consola.setLevel(logging.INFO)
    handler_consola.setFormatter(formato_consola)

    # Handler de archivo (DEBUG+)
    handler_archivo = logging.FileHandler(ruta_log, mode="a", encoding="utf-8")
    handler_archivo.setLevel(logging.DEBUG)
    handler_archivo.setFormatter(formato_archivo)

    # Logger raiz
    logger_raiz = logging.getLogger()
    logger_raiz.setLevel(logging.DEBUG)

    # Limpiar handlers previos para evitar duplicados
    if logger_raiz.hasHandlers():
        logger_raiz.handlers.clear()

    logger_raiz.addHandler(handler_consola)
    logger_raiz.addHandler(handler_archivo)
