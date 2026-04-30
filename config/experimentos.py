"""Configuracion reusable de experimentos para entrenamiento y evaluacion."""

from dataclasses import dataclass, field
from pathlib import Path

from config.settings import (
    ARCHIVO_PARQUET,
    CHECKPOINT_INTERVALO,
    CUANTIL_MAX,
    CUANTIL_MIN,
    EPOCAS_RAMPA_RANGO,
    EPOCAS_RAMPA,
    EPOCAS_SOLO_DATOS,
    ESTACIONES_EXCLUIDAS,
    INTERVALO,
    LAMBDA_RANGO_MAX,
    LR_INICIAL,
    LAMBDA_FISICA,
    MAX_NORM_GRAD,
    NOMBRE_LOG_ENTRENAMIENTO,
    NOMBRE_MODELO_ENTRENAMIENTO,
    N_DIAS,
    NUM_EPOCAS,
    R_MALLA,
    REESCALAR_PRESION,
    TAU_RANGO,
    USAR_CUANTILES_RANGO,
)


def _formatear_sufijo_r(valor_r: float) -> str:
    """Convierte la resolucion R a un sufijo corto estable."""
    return f"R{int(round(valor_r * 10)):02d}"


def _formatear_sufijo_lambda(valor_lambda: float) -> str:
    """Convierte lambda a un sufijo corto estable."""
    if float(valor_lambda).is_integer():
        return f"L{int(valor_lambda)}"
    return f"L{str(valor_lambda).replace('.', 'p')}"


@dataclass(slots=True)
class ConfiguracionExperimento:
    """Representa la configuracion de una corrida del proyecto PINN.

    :param nombre_experimento: nombre logico de la corrida.
    :param nombre_modelo: prefijo de checkpoints e historiales.
    :param nombre_log: nombre del archivo de log.
    :param archivo_parquet: fuente de datos trazable.
    :param r_malla: resolucion espacial de la malla en grados.
    :param lambda_fisica: peso de la perdida fisica.
    :param num_epocas: numero de epocas de entrenamiento.
    :param lr_inicial: learning rate inicial.
    :param max_norm_grad: maximo para gradient clipping.
    :param checkpoint_intervalo: frecuencia de checkpoints.
    :param n_dias: horizonte temporal del subset.
    :param intervalo: submuestreo temporal.
    :param reescalar_presion: activa reescalado de presion.
    :param epocas_solo_datos: epocas iniciales sin perdida fisica.
    :param epocas_rampa: epocas de rampa lineal para lambda.
    :param lambda_rango_max: peso maximo de la restriccion suave de rango.
    :param epocas_rampa_rango: epocas de rampa para lambda_rango.
    :param tau_rango: temperatura de suavizado de la penalizacion.
    :param usar_cuantiles_rango: si True usa cuantiles en lugar de min/max.
    :param cuantil_min: cuantil inferior para limites robustos.
    :param cuantil_max: cuantil superior para limites robustos.
    :param estaciones_excluidas: estaciones excluidas del entrenamiento.
    :param registros_por_estacion: limite por estacion; None conserva todo.
    """

    nombre_experimento: str
    nombre_modelo: str
    nombre_log: str
    archivo_parquet: Path = ARCHIVO_PARQUET
    r_malla: float = R_MALLA
    lambda_fisica: float = LAMBDA_FISICA
    num_epocas: int = NUM_EPOCAS
    lr_inicial: float = LR_INICIAL
    max_norm_grad: float = MAX_NORM_GRAD
    checkpoint_intervalo: int = CHECKPOINT_INTERVALO
    n_dias: int = N_DIAS
    intervalo: int = INTERVALO
    reescalar_presion: bool = REESCALAR_PRESION
    epocas_solo_datos: int = EPOCAS_SOLO_DATOS
    epocas_rampa: int = EPOCAS_RAMPA
    lambda_rango_max: float = LAMBDA_RANGO_MAX
    epocas_rampa_rango: int = EPOCAS_RAMPA_RANGO
    tau_rango: float = TAU_RANGO
    usar_cuantiles_rango: bool = USAR_CUANTILES_RANGO
    cuantil_min: float = CUANTIL_MIN
    cuantil_max: float = CUANTIL_MAX
    estaciones_excluidas: list[str] = field(
        default_factory=lambda: list(ESTACIONES_EXCLUIDAS)
    )
    registros_por_estacion: int | None = None


def construir_configuracion_experimento(
    nombre_experimento: str | None = None,
    r_malla: float | None = None,
    lambda_fisica: float | None = None,
    num_epocas: int | None = None,
    registros_por_estacion: int | None = None,
    nombre_modelo: str | None = None,
    nombre_log: str | None = None,
) -> ConfiguracionExperimento:
    """Construye una configuracion de experimento con nombres consistentes.

    :param nombre_experimento: nombre logico opcional de la corrida.
    :param r_malla: resolucion espacial de la malla.
    :param lambda_fisica: peso fijo de la perdida fisica.
    :param num_epocas: numero de epocas a ejecutar.
    :param registros_por_estacion: limite de registros por estacion.
    :param nombre_modelo: prefijo opcional para checkpoints.
    :param nombre_log: nombre opcional del log.
    :return: instancia lista para entrenamiento o evaluacion.
    """
    if r_malla is None:
        r_malla = R_MALLA

    if lambda_fisica is None:
        lambda_fisica = LAMBDA_FISICA

    if num_epocas is None:
        num_epocas = NUM_EPOCAS

    if nombre_experimento is None:
        if registros_por_estacion is None:
            nombre_experimento = NOMBRE_MODELO_ENTRENAMIENTO
        else:
            nombre_experimento = (
                "PINN_caribe_excl4_"
                f"{registros_por_estacion}xest_"
                f"{_formatear_sufijo_r(r_malla)}_"
                f"{_formatear_sufijo_lambda(lambda_fisica)}"
            )

    if nombre_modelo is None:
        nombre_modelo = nombre_experimento

    if nombre_log is None:
        if nombre_modelo == NOMBRE_MODELO_ENTRENAMIENTO:
            nombre_log = NOMBRE_LOG_ENTRENAMIENTO
        else:
            nombre_log = f"{nombre_modelo}.log"

    return ConfiguracionExperimento(
        nombre_experimento=nombre_experimento,
        nombre_modelo=nombre_modelo,
        nombre_log=nombre_log,
        r_malla=r_malla,
        lambda_fisica=lambda_fisica,
        num_epocas=num_epocas,
        registros_por_estacion=registros_por_estacion,
    )