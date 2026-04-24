"""
Modulo de carga de datos desde archivos Parquet.

Carga el archivo Parquet con datos de estaciones meteorologicas del Caribe
de Colombia y retorna arrays NumPy con las columnas relevantes.

Referencia: trabajo_base_pinns/src/process_data.py — ProcessDataColombia.load_data()
"""

import logging

import numpy as np
import pandas as pd

from pathlib import Path

from config.settings import ESTACIONES_EXCLUIDAS

logger = logging.getLogger(__name__)


def cargar_parquet(
    ruta: Path,
    excluir_estaciones: bool = True,
    estaciones_excluidas: list[str] | None = None,
    registros_por_estacion: int | None = None,
) -> dict:
    """
    Carga un archivo Parquet con registros de estaciones meteorologicas.

    Columnas esperadas del Parquet:
        codigo_estacion, segundos, latitud, longitud, presion,
        temperatura, altura, vel_u, vel_v

    :param ruta: ruta al archivo Parquet.
    :param excluir_estaciones: si True, elimina las estaciones listadas en
                               ESTACIONES_EXCLUIDAS.
    :param estaciones_excluidas: lista explicita de estaciones a excluir.
    :param registros_por_estacion: cantidad maxima de registros por estacion,
                                   ordenados por segundos ascendente.
    :return: diccionario con arrays NumPy para cada columna y metadatos.
    :raises FileNotFoundError: si el archivo no existe.
    :raises KeyError: si faltan columnas esperadas.
    :raises ValueError: si no quedan estaciones tras aplicar el subset.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {ruta}")

    logger.info(f"Cargando datos desde {ruta.name}")
    df = pd.read_parquet(ruta, engine="fastparquet")
    logger.info(f"Registros cargados: {len(df):,}")

    columnas_requeridas = [
        "codigo_estacion", "segundos", "latitud", "longitud",
        "presion", "temperatura", "altura", "vel_u", "vel_v",
    ]
    faltantes = set(columnas_requeridas) - set(df.columns)
    if faltantes:
        raise KeyError(f"Faltan columnas en el Parquet: {faltantes}")

    estaciones_a_excluir = ESTACIONES_EXCLUIDAS
    if estaciones_excluidas is not None:
        estaciones_a_excluir = estaciones_excluidas

    # Excluir estaciones configuradas
    if excluir_estaciones and estaciones_a_excluir:
        n_antes = len(df)
        df = df[~df["codigo_estacion"].isin(estaciones_a_excluir)]
        n_excluidas = n_antes - len(df)
        if n_excluidas > 0:
            logger.info(f"Estaciones excluidas: {estaciones_a_excluir} "
                        f"({n_excluidas:,} registros eliminados)")

    estaciones_descartadas_por_subset = []

    if registros_por_estacion is not None:
        logger.info(
            f"Aplicando subset de {registros_por_estacion} registros por estacion"
        )
        conteos_por_estacion = df.groupby("codigo_estacion").size()
        estaciones_insuficientes = conteos_por_estacion[
            conteos_por_estacion < registros_por_estacion
        ]
        if not estaciones_insuficientes.empty:
            estaciones_descartadas_por_subset = (
                estaciones_insuficientes.index.astype(str).tolist()
            )
            detalle = ", ".join(
                f"{codigo}:{cantidad}"
                for codigo, cantidad in estaciones_insuficientes.items()
            )
            logger.warning(
                "Descartando estaciones con menos registros de los requeridos: "
                f"{detalle}"
            )
            df = df[
                ~df["codigo_estacion"].isin(estaciones_descartadas_por_subset)
            ]

        if df.empty:
            raise ValueError(
                "No quedaron estaciones disponibles tras aplicar el criterio "
                f"de {registros_por_estacion} registros por estacion"
            )

        df = df.sort_values(["codigo_estacion", "segundos"], kind="mergesort")
        df = df.groupby("codigo_estacion", sort=False, group_keys=False).head(
            registros_por_estacion
        )

        conteos_subset = df.groupby("codigo_estacion").size()
        if not (conteos_subset == registros_por_estacion).all():
            raise ValueError(
                "El subset por estacion no quedo rectangular tras el recorte inicial"
            )

        logger.info(
            f"Subset rectangular validado: {len(conteos_subset)} estaciones x "
            f"{registros_por_estacion} registros"
        )

    numero_estaciones = df["codigo_estacion"].nunique()
    estaciones_utilizadas = sorted(df["codigo_estacion"].unique().tolist())
    tiempos_seleccionados = np.sort(df["segundos"].unique()).astype(np.float64)
    logger.info(f"Numero de estaciones: {numero_estaciones}")
    logger.debug(f"Segundos max: {np.nanmax(df['segundos']):,.0f} s "
                 f"({int(np.nanmax(df['segundos']) // 86400)} dias)")

    datos = {
        "codigo_estacion": df["codigo_estacion"].values,
        "segundos": df["segundos"].values.astype(np.float64),
        "latitud": df["latitud"].values.astype(np.float64),
        "longitud": df["longitud"].values.astype(np.float64),
        "presion": df["presion"].values.astype(np.float64),
        "temperatura": df["temperatura"].values.astype(np.float64),
        "altura": df["altura"].values.astype(np.float64),
        "vel_u": df["vel_u"].values.astype(np.float64),
        "vel_v": df["vel_v"].values.astype(np.float64),
        "numero_estaciones": numero_estaciones,
        "registros_por_estacion": registros_por_estacion,
        "total_registros": len(df),
        "estaciones_utilizadas": estaciones_utilizadas,
        "estaciones_descartadas_por_subset": estaciones_descartadas_por_subset,
        "tiempos_seleccionados": tiempos_seleccionados,
    }

    return datos
