"""
Modulo de construccion de la malla de prediccion (puntos de colocacion).

Genera una malla rectangular derivada de las posiciones de las estaciones
meteorologicas, con resolucion R_MALLA grados, y la normaliza con las mismas
escalas usadas para los datos de estaciones.

Referencia: trabajo_base_pinns/src/process_data.py —
            ProcessDataColombia._centered_grid_adimensionalization() (seccion meshgrid)
"""

import logging

import numpy as np

from config.settings import RADIO_TIERRA, R_MALLA

logger = logging.getLogger(__name__)


def construir_malla(datos_norm: dict, escalas: dict,
                    R: float = R_MALLA) -> dict:
    """
    Construye la malla PINN de puntos de colocacion.

    Pasos:
        1. Calcular R_PINN en metros: 6378000 * sin(rad(R))
        2. Generar ejes x, y extendidos por R_PINN mas alla de los extremos
        3. Crear meshgrid y aplanar con orden columna mayor ('F')
        4. Replicar cada punto espacial para todos los timesteps
        5. Normalizar con las mismas escalas (L, W)

    :param datos_norm: diccionario con datos normalizados (contiene T para los timesteps).
    :param escalas: diccionario con L, W, x_min, x_max, y_min, y_max.
    :param R: resolucion de la malla en grados.
    :return: diccionario con T_PINN, X_PINN, Y_PINN normalizados,
             y metadatos de la malla (x_pinn, y_pinn sin normalizar, dim_T, dim_N).
    """
    logger.info(f"Construyendo malla PINN con R={R} grados")

    L = escalas["L"]
    W = escalas["W"]
    x_min = escalas["x_min"]
    x_max = escalas["x_max"]
    y_min = escalas["y_min"]
    y_max = escalas["y_max"]

    centro_x = datos_norm["centro_x"]
    centro_y = datos_norm["centro_y"]
    t_min = datos_norm["t_min"]

    # Resolucion en metros
    R_PINN = RADIO_TIERRA * np.sin(np.radians(R))
    logger.debug(f"R_PINN = {R_PINN:.0f} m")

    # Ejes de la malla (en coordenadas cartesianas originales, centradas)
    x_pinn = np.arange(x_min - R_PINN, x_max + R_PINN, R_PINN) - centro_x
    y_pinn = np.arange(y_min - R_PINN, y_max + R_PINN, R_PINN) - centro_y

    logger.debug(f"Puntos en X: {len(x_pinn)}, Puntos en Y: {len(y_pinn)}")

    # Meshgrid y aplanar con orden columna mayor (Fortran)
    X_grid, Y_grid = np.meshgrid(x_pinn, y_pinn)
    X_flat = X_grid.flatten("F")[:, None]
    Y_flat = Y_grid.flatten("F")[:, None]

    dim_N = X_flat.shape[0]  # numero total de puntos espaciales

    # Timesteps de la primera estacion (ya centrados: T - t_min)
    # En datos_norm, T ya esta normalizado (T_c * W / L), necesitamos T centrado sin normalizar
    # Reconstruimos: T_centrado = T_norm * L / W
    T_estac = datos_norm["T"][0:1, :]  # (1, N_tiempos) — ya normalizado
    dim_T = T_estac.shape[1]

    logger.info(f"Malla: {dim_N} puntos espaciales x {dim_T} timesteps = {dim_N * dim_T:,} total")

    # Replicar para todos los timesteps
    # T_PINN: (dim_N, dim_T) — cada fila replica los timesteps
    T_PINN = np.tile(T_estac, (dim_N, 1))

    # X_PINN, Y_PINN: (dim_N, dim_T) — cada columna replica las posiciones
    X_PINN = np.tile(X_flat, dim_T) / L
    Y_PINN = np.tile(Y_flat, dim_T) / L

    logger.debug(f"Forma malla: X_PINN={X_PINN.shape}, Y_PINN={Y_PINN.shape}, T_PINN={T_PINN.shape}")

    malla = {
        "T": T_PINN,
        "X": X_PINN,
        "Y": Y_PINN,
        "dim_T": dim_T,
        "dim_N": dim_N,
        "x_pinn": x_pinn,
        "y_pinn": y_pinn,
    }

    return malla
