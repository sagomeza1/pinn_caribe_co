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


def cargar_parquet(ruta: Path, excluir_estaciones: bool = True) -> dict:
    """
    Carga un archivo Parquet con registros de estaciones meteorologicas.

    Columnas esperadas del Parquet:
        codigo_estacion, segundos, latitud, longitud, presion,
        temperatura, altura, vel_u, vel_v

    :param ruta: ruta al archivo Parquet.
    :param excluir_estaciones: si True, elimina las estaciones listadas en
                               ESTACIONES_EXCLUIDAS.
    :return: diccionario con arrays NumPy para cada columna y metadatos.
    :raises FileNotFoundError: si el archivo no existe.
    :raises KeyError: si faltan columnas esperadas.
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

    # Excluir estaciones configuradas
    if excluir_estaciones and ESTACIONES_EXCLUIDAS:
        n_antes = len(df)
        df = df[~df["codigo_estacion"].isin(ESTACIONES_EXCLUIDAS)]
        n_excluidas = n_antes - len(df)
        if n_excluidas > 0:
            logger.info(f"Estaciones excluidas: {ESTACIONES_EXCLUIDAS} "
                        f"({n_excluidas:,} registros eliminados)")

    numero_estaciones = df["codigo_estacion"].nunique()
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
    }

    return datos
