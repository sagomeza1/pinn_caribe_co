"""
Modulo de preprocesamiento de datos.

Transforma los datos crudos del Parquet en matrices organizadas por estacion,
aplica la proyeccion cartesiana, correccion de presion ISA y ordena por coordenada X.

Referencia: trabajo_base_pinns/src/process_data.py — ProcessDataColombia
"""

import logging

import numpy as np

from config.settings import RADIO_TIERRA, N_DIAS, INTERVALO

logger = logging.getLogger(__name__)


def proyectar_a_cartesiano(datos: dict) -> dict:
    """
    Proyecta coordenadas geograficas a cartesianas y convierte presion a Pa.

    Replica la logica de ProcessDataColombia._process_coordinates_and_projections():
        - X_WS = 6378000 * sin(rad(longitud))
        - Y_WS = 6378000 * sin(rad(latitud))
        - P en mbar a Pa (*100)

    :param datos: diccionario retornado por cargar_parquet().
    :return: diccionario con arrays T, X, Y, Z, U, V, P, Temp y metadatos.
    """
    logger.info("Proyectando coordenadas a cartesianas")

    resultado = {
        "T": datos["segundos"].copy(),
        "X": RADIO_TIERRA * np.sin(np.radians(datos["longitud"])),
        "Y": RADIO_TIERRA * np.sin(np.radians(datos["latitud"])),
        "Z": datos["altura"].copy(),
        "Temp": datos["temperatura"].copy(),
        "U": datos["vel_u"].copy(),
        "V": datos["vel_v"].copy(),
        "P": datos["presion"] * 100.0,  # mbar a Pa
        "numero_estaciones": datos["numero_estaciones"],
    }

    logger.debug(f"Rango X: [{np.nanmin(resultado['X']):.0f}, {np.nanmax(resultado['X']):.0f}]")
    logger.debug(f"Rango Y: [{np.nanmin(resultado['Y']):.0f}, {np.nanmax(resultado['Y']):.0f}]")

    return resultado


def reestructurar_por_estacion(datos: dict) -> dict:
    """
    Reestructura arrays planos a matrices (N_estaciones x N_tiempos) y elimina NaNs.

    Replica la logica de ProcessDataColombia._reshape_delete_nans().

    :param datos: diccionario con arrays planos y 'numero_estaciones'.
    :return: diccionario con matrices (N_estaciones, N_tiempos).
    """
    n_est = datos["numero_estaciones"]
    logger.info(f"Reestructurando datos para {n_est} estaciones")

    campos = ["T", "X", "Y", "Z", "U", "V", "P", "Temp"]

    resultado = {"numero_estaciones": n_est}

    # Reestructurar: (N_registros,) -> (N_estaciones, N_tiempos) con orden Fortran
    for campo in campos:
        arr = datos[campo]
        n_tiempos = arr.shape[0] // n_est
        resultado[campo] = np.reshape(arr, (n_tiempos, n_est), order="F").T

    logger.debug(f"Dimension reestructurada: {resultado['T'].shape}")

    # Eliminar estaciones con NaN en ubicacion (X)
    nan_filas = np.where(np.isnan(resultado["X"][:, 0]))[0]
    if len(nan_filas) > 0:
        logger.warning(f"Eliminando {len(nan_filas)} estaciones con NaN en ubicacion")
        for campo in campos:
            resultado[campo] = np.delete(resultado[campo], nan_filas, axis=0)
        resultado["numero_estaciones"] = resultado["T"].shape[0]

    return resultado


def filtrar_tiempo_y_ordenar(datos: dict, n_dias: int = N_DIAS,
                              intervalo: int = INTERVALO) -> dict:
    """
    Filtra por cantidad de dias e intervalo temporal, y ordena por coordenada X.

    Replica la logica de ProcessDataColombia._filter_by_time_and_sort_by_coor().

    :param datos: diccionario con matrices (N_estaciones, N_tiempos).
    :param n_dias: cantidad de dias a conservar.
    :param intervalo: intervalo temporal (1 = cada registro, 2 = cada 2, etc.).
    :return: diccionario filtrado y ordenado.
    """
    logger.info(f"Filtrando {n_dias} dias con intervalo {intervalo}")

    campos = ["T", "X", "Y", "Z", "U", "V", "P", "Temp"]

    # Filtro por dias: seleccionar columnas cuyo tiempo <= n_dias * 86400
    un_dia = 24 * 3600
    segundos_limite = un_dia * n_dias
    mascara_tiempo = datos["T"][0, :] <= segundos_limite

    resultado = {"numero_estaciones": datos["numero_estaciones"]}
    for campo in campos:
        resultado[campo] = datos[campo][:, mascara_tiempo]

    # Filtro por intervalo
    for campo in campos:
        resultado[campo] = resultado[campo][:, ::intervalo]

    logger.debug(f"Intervalo de {resultado['T'][0, 1] / 60:.0f} min")
    logger.debug(f"Dimension tras filtro: {resultado['T'].shape}")

    # Ordenar por coordenada X en cada snapshot temporal
    n_snapshots = resultado["T"].shape[1]
    for snap in range(n_snapshots):
        idx_sort = np.argsort(resultado["X"][:, snap])
        for campo in campos:
            resultado[campo][:, snap] = resultado[campo][idx_sort, snap]

    return resultado


def corregir_presion_isa(datos: dict) -> dict:
    """
    Corrige la presion a nivel del mar usando el modelo ISA.

    Formula: P_corr = P * (1 - 0.0065 * Z / (T + 273.15 + 0.0065 * Z))^(-5.257)

    :param datos: diccionario con matrices P, Z y Temp.
    :return: diccionario con P corregida (en Pa).
    """
    logger.info("Aplicando correccion de presion ISA a nivel del mar")

    datos["P"] = datos["P"] * (
        1 - 0.0065 * datos["Z"] / (datos["Temp"] + 273.15 + 0.0065 * datos["Z"])
    ) ** (-5.257)

    logger.debug(f"Rango presion corregida: [{np.nanmin(datos['P']):.0f}, {np.nanmax(datos['P']):.0f}] Pa")

    return datos


def preprocesar(datos: dict, n_dias: int = N_DIAS,
                intervalo: int = INTERVALO) -> dict:
    """
    Ejecuta el pipeline completo de preprocesamiento.

    :param datos: diccionario retornado por cargar_parquet().
    :param n_dias: cantidad de dias a conservar.
    :param intervalo: intervalo temporal.
    :return: diccionario con datos preprocesados.
    """
    resultado = proyectar_a_cartesiano(datos)
    resultado = reestructurar_por_estacion(resultado)
    resultado = filtrar_tiempo_y_ordenar(resultado, n_dias=n_dias, intervalo=intervalo)
    resultado = corregir_presion_isa(resultado)

    logger.info("Preprocesamiento completado")
    return resultado
