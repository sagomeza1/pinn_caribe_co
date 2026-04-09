"""
Metricas de evaluacion para la PINN.

Calcula:
    - MSE por variable (u, v, p) en puntos de estaciones
    - Residual promedio de Navier-Stokes

Referencia: trabajo_base_pinns/src/loss_functions.py
"""

import logging

import torch
import torch.nn as nn

from src.model.perdida_fisica import perdida_navier_stokes

logger = logging.getLogger(__name__)


def evaluar_metricas(modelo: nn.Module, dataset_estaciones,
                     device: torch.device) -> dict:
    """
    Evalua MSE por variable y residual NS en los datos de estaciones.

    :param modelo: red PINN entrenada.
    :param dataset_estaciones: EstacionesDataset con datos observados.
    :param device: dispositivo.
    :return: diccionario con mse_u, mse_v, mse_p, residual_ns.
    """
    modelo.eval()
    mse = nn.MSELoss()

    # Obtener todos los datos del dataset
    t = dataset_estaciones.t.to(device)
    x = dataset_estaciones.x.to(device)
    y = dataset_estaciones.y.to(device)
    u_true = dataset_estaciones.u.to(device)
    v_true = dataset_estaciones.v.to(device)
    p_true = dataset_estaciones.p.to(device)

    with torch.no_grad():
        salida = modelo(t, x, y)
        u_pred = salida[:, 0:1]
        v_pred = salida[:, 1:2]
        p_pred = salida[:, 2:3]

        mse_u = mse(u_pred, u_true).item()
        mse_v = mse(v_pred, v_true).item()
        mse_p = mse(p_pred, p_true).item()

    # Residual NS requiere gradientes (no se puede usar no_grad)
    t_ns = dataset_estaciones.t.to(device)
    x_ns = dataset_estaciones.x.to(device)
    y_ns = dataset_estaciones.y.to(device)
    residual_ns = perdida_navier_stokes(modelo, t_ns, x_ns, y_ns).item()

    logger.info(f"MSE u: {mse_u:.3e} | MSE v: {mse_v:.3e} | MSE p: {mse_p:.3e}")
    logger.info(f"Residual NS: {residual_ns:.3e}")

    return {
        "mse_u": mse_u,
        "mse_v": mse_v,
        "mse_p": mse_p,
        "residual_ns": residual_ns,
    }
