"""
Modulo de normalizacion (adimensionalizacion) de datos.

Centra los datos espaciales y temporales, calcula escalas de referencia
y normaliza todas las variables.

Referencia: trabajo_base_pinns/src/process_data.py —
            ProcessDataColombia._centered_grid_adimensionalization()
"""

import logging

import numpy as np

from config.settings import RHO_AIRE, NU_AIRE

logger = logging.getLogger(__name__)


def calcular_escalas(datos: dict, rho: float = RHO_AIRE,
                     nu: float = NU_AIRE) -> dict:
    """
    Calcula las escalas de referencia para la adimensionalizacion.

    :param datos: diccionario con matrices X, Y, T, U, V, P (preprocesados).
    :param rho: densidad del aire (kg/m3).
    :param nu: viscosidad cinematica del aire (m2/s).
    :return: diccionario con escalas L, W, P0, rho, Re.
    """
    x_min, x_max = np.min(datos["X"]), np.max(datos["X"])
    y_min, y_max = np.min(datos["Y"]), np.max(datos["Y"])

    L = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    W = np.sqrt(np.nanmax(np.abs(datos["U"])) ** 2 + np.nanmax(np.abs(datos["V"])) ** 2)
    P0 = np.nanmean(datos["P"])
    Re = int(W * L / nu)

    logger.info(f"Escalas: L={L:.2f} m, W={W:.2f} m/s, P0={P0:.2f} Pa, Re={Re:,}")

    return {
        "L": L,
        "W": W,
        "P0": P0,
        "rho": rho,
        "nu": nu,
        "Re": Re,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def centrar_y_normalizar(datos: dict, escalas: dict,
                         reescalar_presion: bool = False) -> dict:
    """
    Centra los datos espaciales/temporales y adimensionaliza todas las variables.

    Centrado:
        X_c = X - (x_min + x_max) / 2
        Y_c = Y - (y_min + y_max) / 2
        T_c = T - T_min

    Normalizacion:
        X_norm = X_c / L
        Y_norm = Y_c / L
        T_norm = T_c * W / L
        U_norm = U / W
        V_norm = V / W
        P_norm = (P - P0) / (rho * W^2)

    Si reescalar_presion=True, divide P_norm por su desviacion estandar (sigma_p)
    para equilibrar el rango de presion con el de las velocidades.

    :param datos: diccionario con matrices preprocesadas.
    :param escalas: diccionario retornado por calcular_escalas().
    :param reescalar_presion: si True, reescala P_norm por sigma_p.
    :return: diccionario con datos normalizados y escalas incorporadas.
    """
    L = escalas["L"]
    W = escalas["W"]
    P0 = escalas["P0"]
    rho = escalas["rho"]
    x_min = escalas["x_min"]
    x_max = escalas["x_max"]
    y_min = escalas["y_min"]
    y_max = escalas["y_max"]

    centro_x = (x_min + x_max) / 2
    centro_y = (y_min + y_max) / 2
    t_min = np.min(datos["T"])

    logger.info("Centrando y normalizando datos de estaciones")

    resultado = {
        "X": (datos["X"] - centro_x) / L,
        "Y": (datos["Y"] - centro_y) / L,
        "T": (datos["T"] - t_min) * W / L,
        "U": datos["U"] / W,
        "V": datos["V"] / W,
        "P": (datos["P"] - P0) / (rho * W ** 2),
        "numero_estaciones": datos["numero_estaciones"],
    }

    if "CODIGO_ESTACION" in datos:
        resultado["CODIGO_ESTACION"] = datos["CODIGO_ESTACION"].copy()

    # Reescalado de presion (Iteracion 2): dividir por sigma_p
    if reescalar_presion:
        sigma_p = float(np.nanstd(resultado["P"]))
        resultado["P"] = resultado["P"] / sigma_p
        resultado["P_scale"] = sigma_p
        logger.info(f"Presion reescalada: sigma_p={sigma_p:.4f}, "
                    f"rango P_rescaled: [{np.nanmin(resultado['P']):.3f}, "
                    f"{np.nanmax(resultado['P']):.3f}]")
    else:
        resultado["P_scale"] = 1.0

    # Guardar parametros de centrado para uso posterior (malla)
    resultado["centro_x"] = centro_x
    resultado["centro_y"] = centro_y
    resultado["t_min"] = t_min

    logger.debug(f"Rango T_norm: [{np.nanmin(resultado['T']):.3e}, {np.nanmax(resultado['T']):.3e}]")
    logger.debug(f"Rango P_norm: [{np.nanmin(resultado['P']):.3e}, {np.nanmax(resultado['P']):.3e}]")
    logger.debug(f"Rango U_norm: [{np.nanmin(resultado['U']):.3e}, {np.nanmax(resultado['U']):.3e}]")
    logger.debug(f"Rango V_norm: [{np.nanmin(resultado['V']):.3e}, {np.nanmax(resultado['V']):.3e}]")
    logger.debug(f"Rango X_norm: [{np.nanmin(resultado['X']):.3e}, {np.nanmax(resultado['X']):.3e}]")
    logger.debug(f"Rango Y_norm: [{np.nanmin(resultado['Y']):.3e}, {np.nanmax(resultado['Y']):.3e}]")

    return resultado


def normalizar(datos: dict, rho: float = RHO_AIRE,
               nu: float = NU_AIRE,
               reescalar_presion: bool = False) -> tuple:
    """
    Pipeline completo de normalizacion.

    :param datos: diccionario con datos preprocesados.
    :param rho: densidad del aire.
    :param nu: viscosidad cinematica.
    :param reescalar_presion: si True, reescala P_norm por sigma_p.
    :return: tupla (datos_normalizados, escalas).
    """
    escalas = calcular_escalas(datos, rho=rho, nu=nu)
    datos_norm = centrar_y_normalizar(datos, escalas,
                                      reescalar_presion=reescalar_presion)

    # Propagar P_scale a escalas para uso posterior
    escalas["P_scale"] = datos_norm.get("P_scale", 1.0)

    logger.info("Normalizacion completada")
    return datos_norm, escalas
