"""Runner secuencial para el barrido de hiperparametros R-lambda."""

import logging
import subprocess
import sys

from itertools import product
from pathlib import Path

from config.experimentos import (
    ConfiguracionExperimento,
    construir_configuracion_experimento,
)
from config.settings import NUM_EPOCAS, RAIZ_PROYECTO

logger = logging.getLogger(__name__)

VALORES_R_BARRIDO = (0.1, 0.2, 0.4)
VALORES_LAMBDA_BARRIDO = (1.0, 3.0, 5.0, 10.0)


def validar_valores_r_barrido(valores_r: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Valida que los valores de R pertenezcan al barrido permitido.

    :param valores_r: valores de R pedidos por el usuario.
    :return: tupla ordenada y sin duplicados.
    :raises ValueError: si algun valor no pertenece al barrido.
    """
    valores_normalizados = tuple(sorted(set(float(valor) for valor in valores_r)))
    invalidos = [valor for valor in valores_normalizados if valor not in VALORES_R_BARRIDO]
    if invalidos:
        raise ValueError(
            "Valores de R fuera del barrido permitido: "
            f"{invalidos}. Use solo {VALORES_R_BARRIDO}."
        )
    return valores_normalizados


def validar_valores_lambda_barrido(
    valores_lambda: list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    """Valida que los valores de lambda pertenezcan al barrido permitido.

    :param valores_lambda: valores de lambda pedidos por el usuario.
    :return: tupla ordenada y sin duplicados.
    :raises ValueError: si algun valor no pertenece al barrido.
    """
    valores_normalizados = tuple(sorted(set(float(valor) for valor in valores_lambda)))
    invalidos = [
        valor for valor in valores_normalizados if valor not in VALORES_LAMBDA_BARRIDO
    ]
    if invalidos:
        raise ValueError(
            "Valores de lambda fuera del barrido permitido: "
            f"{invalidos}. Use solo {VALORES_LAMBDA_BARRIDO}."
        )
    return valores_normalizados


def generar_configuraciones_barrido(
    registros_por_estacion: int = 300,
    num_epocas: int = NUM_EPOCAS,
    valores_r: list[float] | tuple[float, ...] | None = None,
    valores_lambda: list[float] | tuple[float, ...] | None = None,
) -> list[ConfiguracionExperimento]:
    """Genera las corridas definidas para el barrido.

    :param registros_por_estacion: tamaño fijo del subset por estacion.
    :param num_epocas: epocas por corrida.
    :param valores_r: subconjunto opcional de valores de R para una iteracion.
    :param valores_lambda: subconjunto opcional de valores de lambda para una iteracion.
    :return: lista ordenada de configuraciones del barrido.
    """
    valores_r_efectivos = VALORES_R_BARRIDO
    valores_lambda_efectivos = VALORES_LAMBDA_BARRIDO
    if valores_r is not None:
        valores_r_efectivos = validar_valores_r_barrido(valores_r)
    if valores_lambda is not None:
        valores_lambda_efectivos = validar_valores_lambda_barrido(valores_lambda)

    configuraciones = []
    for valor_r, valor_lambda in product(valores_r_efectivos, valores_lambda_efectivos):
        configuraciones.append(
            construir_configuracion_experimento(
                r_malla=valor_r,
                lambda_fisica=valor_lambda,
                num_epocas=num_epocas,
                registros_por_estacion=registros_por_estacion,
            )
        )
    return configuraciones


def construir_comando_entrenamiento(
    configuracion: ConfiguracionExperimento,
    python_executable: str | None = None,
    ruta_main: Path = RAIZ_PROYECTO / "main.py",
) -> list[str]:
    """Construye el comando CLI para una corrida del barrido.

    :param configuracion: corrida a ejecutar.
    :param python_executable: ejecutable de Python a usar.
    :param ruta_main: ruta del entrypoint principal.
    :return: comando listo para subprocess.
    """
    ejecutable = python_executable or sys.executable
    return [
        ejecutable,
        str(ruta_main),
        "--nombre-experimento",
        configuracion.nombre_experimento,
        "--nombre-modelo",
        configuracion.nombre_modelo,
        "--nombre-log",
        configuracion.nombre_log,
        "--r-malla",
        str(configuracion.r_malla),
        "--lambda-fisica",
        str(configuracion.lambda_fisica),
        "--num-epocas",
        str(configuracion.num_epocas),
        "--registros-por-estacion",
        str(configuracion.registros_por_estacion),
    ]


def ejecutar_barrido_secuencial(
    configuraciones: list[ConfiguracionExperimento],
    python_executable: str | None = None,
    dry_run: bool = False,
) -> list[list[str]]:
    """Ejecuta o lista el barrido de forma secuencial.

    :param configuraciones: corridas a procesar.
    :param python_executable: ejecutable de Python opcional.
    :param dry_run: si True, solo retorna los comandos.
    :return: lista de comandos construidos.
    :raises subprocess.CalledProcessError: si alguna corrida falla.
    """
    comandos = []
    for configuracion in configuraciones:
        comando = construir_comando_entrenamiento(
            configuracion=configuracion,
            python_executable=python_executable,
        )
        comandos.append(comando)
        if dry_run:
            logger.info("Dry run barrido: %s", " ".join(comando))
            continue

        logger.info(
            "Ejecutando barrido: %s (R=%s, lambda=%s)",
            configuracion.nombre_modelo,
            configuracion.r_malla,
            configuracion.lambda_fisica,
        )
        subprocess.run(
            comando,
            cwd=RAIZ_PROYECTO,
            check=True,
        )

    return comandos