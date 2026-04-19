"""
Metricas de evaluacion para la PINN.

Calcula:
    - MSE por variable (u, v, p) en puntos de estaciones
    - Residual promedio de Navier-Stokes

Referencia: trabajo_base_pinns/src/loss_functions.py
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.model.perdida_fisica import perdida_navier_stokes

logger = logging.getLogger(__name__)


def _promedio_por_instante(t: torch.Tensor, error: torch.Tensor) -> tuple:
    """
    Calcula el promedio del error por cada instante unico de tiempo.

    :param t: tensor de tiempo con forma (N, 1) o (N,).
    :param error: tensor de error con forma (N, 1) o (N,).
    :return: tupla (tiempos_unicos, error_promedio_por_tiempo).
    """
    t_flat = t.squeeze()
    err_flat = error.squeeze()

    tiempos_unicos, indices = torch.unique(
        t_flat,
        sorted=True,
        return_inverse=True,
    )
    conteos = torch.bincount(indices).to(err_flat.dtype)
    suma_errores = torch.bincount(indices, weights=err_flat)
    promedio = suma_errores / conteos

    return tiempos_unicos, promedio


def graficar_error_promedio_temporal(
    modelo: nn.Module,
    dataset_estaciones,
    device: torch.device,
    ruta_reporte,
    prefijo_archivo: str,
    p_scale: float = 1.0,
    escalas: dict | None = None,
) -> dict:
    """
    Genera tres graficas de error absoluto promedio por variable a lo largo del tiempo.

    :param modelo: red PINN entrenada.
    :param dataset_estaciones: EstacionesDataset con datos observados.
    :param device: dispositivo.
    :param ruta_reporte: ruta del directorio donde se guardan las figuras.
    :param prefijo_archivo: prefijo base para nombrar las figuras.
    :param p_scale: factor de reescalado de presion usado en entrenamiento.
    :param escalas: diccionario de escalas fisicas (L, W, rho).
    :return: diccionario con rutas de salida de las tres graficas.
    :raises ValueError: si no hay datos de estaciones para graficar.
    """
    modelo.eval()

    t = dataset_estaciones.t.to(device)
    x = dataset_estaciones.x.to(device)
    y = dataset_estaciones.y.to(device)
    u_true = dataset_estaciones.u.to(device)
    v_true = dataset_estaciones.v.to(device)
    p_true = dataset_estaciones.p.to(device)

    if t.numel() == 0:
        raise ValueError("No hay datos de estaciones para calcular errores temporales")

    with torch.no_grad():
        salida = modelo(t, x, y)
        u_pred = salida[:, 0:1]
        v_pred = salida[:, 1:2]
        p_pred = salida[:, 2:3]

    escala_u = float(escalas.get("W", 1.0)) if escalas else 1.0
    escala_v = float(escalas.get("W", 1.0)) if escalas else 1.0
    if escalas:
        escala_p = float(escalas.get("rho", 1.0)) * float(escalas.get("W", 1.0)) ** 2 * float(p_scale)
    else:
        escala_p = float(p_scale)

    err_u = torch.abs(u_pred - u_true).squeeze() * escala_u
    err_v = torch.abs(v_pred - v_true).squeeze() * escala_v
    err_p = torch.abs(p_pred - p_true).squeeze() * escala_p

    tiempos_u, err_u_t = _promedio_por_instante(t, err_u)
    tiempos_v, err_v_t = _promedio_por_instante(t, err_v)
    tiempos_p, err_p_t = _promedio_por_instante(t, err_p)

    if escalas and "L" in escalas and "W" in escalas:
        factor_horas = float(escalas["L"]) / float(escalas["W"]) / 3600.0
        tiempos_u_x = tiempos_u.detach().cpu().numpy() * factor_horas
        tiempos_v_x = tiempos_v.detach().cpu().numpy() * factor_horas
        tiempos_p_x = tiempos_p.detach().cpu().numpy() * factor_horas
        etiqueta_tiempo = "Tiempo [h]"
    else:
        tiempos_u_x = tiempos_u.detach().cpu().numpy()
        tiempos_v_x = tiempos_v.detach().cpu().numpy()
        tiempos_p_x = tiempos_p.detach().cpu().numpy()
        etiqueta_tiempo = "Tiempo adimensional"

    ruta_reporte.mkdir(parents=True, exist_ok=True)
    salidas = {
        "u": ruta_reporte / f"{prefijo_archivo}_error_promedio_u_tiempo.png",
        "v": ruta_reporte / f"{prefijo_archivo}_error_promedio_v_tiempo.png",
        "p": ruta_reporte / f"{prefijo_archivo}_error_promedio_p_tiempo.png",
    }

    figuras = [
        ("u", tiempos_u_x, err_u_t.detach().cpu().numpy(), "Error absoluto promedio de u", "Error [m/s]"),
        ("v", tiempos_v_x, err_v_t.detach().cpu().numpy(), "Error absoluto promedio de v", "Error [m/s]"),
        ("p", tiempos_p_x, err_p_t.detach().cpu().numpy(), "Error absoluto promedio de p", "Error [Pa]"),
    ]

    for clave, eje_x, eje_y, titulo, etiqueta_y in figuras:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(eje_x, eje_y, color="#006D77", linewidth=2.0)
        ax.set_title(titulo)
        ax.set_xlabel(etiqueta_tiempo)
        ax.set_ylabel(etiqueta_y)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(salidas[clave], dpi=180)
        plt.close(fig)

        logger.info(f"Grafica temporal de error guardada: {salidas[clave].name}")

    return salidas


def graficar_mse_por_estacion_en_tiempo(
    modelo: nn.Module,
    datos_norm: dict,
    device: torch.device,
    ruta_reporte,
    prefijo_archivo: str,
    p_scale: float = 1.0,
    escalas: dict | None = None,
) -> dict:
    """
    Genera graficas de MSE por estacion a lo largo del tiempo para u, v y p.

    :param modelo: red PINN entrenada.
    :param datos_norm: diccionario de datos normalizados (matrices por estacion-tiempo).
    :param device: dispositivo de ejecucion.
    :param ruta_reporte: ruta del directorio donde se guardan las figuras.
    :param prefijo_archivo: prefijo base para nombrar las figuras.
    :param p_scale: factor de reescalado de presion usado en entrenamiento.
    :param escalas: diccionario de escalas fisicas (L, W, rho).
    :return: diccionario con rutas de salida de las tres graficas.
    :raises KeyError: si faltan variables requeridas en datos_norm.
    """
    claves_requeridas = ["T", "X", "Y", "U", "V", "P"]
    for clave in claves_requeridas:
        if clave not in datos_norm:
            raise KeyError(f"Falta la clave requerida en datos_norm: {clave}")

    t_mat = datos_norm["T"]
    x_mat = datos_norm["X"]
    y_mat = datos_norm["Y"]
    u_true_mat = datos_norm["U"]
    v_true_mat = datos_norm["V"]
    p_true_mat = datos_norm["P"]

    n_estaciones, n_tiempos = t_mat.shape
    codigos = datos_norm.get("CODIGO_ESTACION")
    if codigos is not None:
        etiquetas = [str(codigos[i, 0]) for i in range(n_estaciones)]
    else:
        etiquetas = [f"EST_{i + 1:03d}" for i in range(n_estaciones)]

    modelo.eval()
    t_flat = torch.tensor(t_mat.reshape(-1), dtype=torch.float32, device=device).unsqueeze(1)
    x_flat = torch.tensor(x_mat.reshape(-1), dtype=torch.float32, device=device).unsqueeze(1)
    y_flat = torch.tensor(y_mat.reshape(-1), dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        salida = modelo(t_flat, x_flat, y_flat)
        u_pred_mat = salida[:, 0].detach().cpu().numpy().reshape(n_estaciones, n_tiempos)
        v_pred_mat = salida[:, 1].detach().cpu().numpy().reshape(n_estaciones, n_tiempos)
        p_pred_mat = salida[:, 2].detach().cpu().numpy().reshape(n_estaciones, n_tiempos)

    escala_u = float(escalas.get("W", 1.0)) if escalas else 1.0
    escala_v = float(escalas.get("W", 1.0)) if escalas else 1.0
    if escalas:
        escala_p = float(escalas.get("rho", 1.0)) * float(escalas.get("W", 1.0)) ** 2 * float(p_scale)
    else:
        escala_p = float(p_scale)

    u_pred_f = u_pred_mat * escala_u
    v_pred_f = v_pred_mat * escala_v
    p_pred_f = p_pred_mat * escala_p

    u_true_f = u_true_mat * escala_u
    v_true_f = v_true_mat * escala_v
    p_true_f = p_true_mat * escala_p

    mse_u = (u_pred_f - u_true_f) ** 2
    mse_v = (v_pred_f - v_true_f) ** 2
    mse_p = (p_pred_f - p_true_f) ** 2

    mascara_u = np.isnan(u_true_f)
    mascara_v = np.isnan(v_true_f)
    mascara_p = np.isnan(p_true_f)
    mse_u[mascara_u] = np.nan
    mse_v[mascara_v] = np.nan
    mse_p[mascara_p] = np.nan

    t_ref = t_mat[0, :]
    if escalas and "L" in escalas and "W" in escalas:
        tiempo_h = t_ref * (float(escalas["L"]) / float(escalas["W"]) / 3600.0)
        etiqueta_tiempo = "Tiempo [h]"
    else:
        tiempo_h = t_ref
        etiqueta_tiempo = "Tiempo adimensional"

    ruta_reporte.mkdir(parents=True, exist_ok=True)
    salidas = {
        "u": ruta_reporte / f"{prefijo_archivo}_mse_por_estacion_u_tiempo.png",
        "v": ruta_reporte / f"{prefijo_archivo}_mse_por_estacion_v_tiempo.png",
        "p": ruta_reporte / f"{prefijo_archivo}_mse_por_estacion_p_tiempo.png",
    }

    series = [
        ("u", mse_u, salidas["u"], "MSE de u por estacion en el tiempo", "MSE [m2/s2]"),
        ("v", mse_v, salidas["v"], "MSE de v por estacion en el tiempo", "MSE [m2/s2]"),
        ("p", mse_p, salidas["p"], "MSE de p por estacion en el tiempo", "MSE [Pa2]"),
    ]

    for _, matriz_mse, ruta_figura, titulo, etiqueta_y in series:
        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(n_estaciones):
            ax.plot(
                tiempo_h,
                matriz_mse[i, :],
                linewidth=0.9,
                alpha=0.55,
                label=etiquetas[i],
            )

        ax.set_title(titulo)
        ax.set_xlabel(etiqueta_tiempo)
        ax.set_ylabel(etiqueta_y)
        ax.grid(alpha=0.25)

        if n_estaciones <= 18:
            ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)

        fig.tight_layout()
        fig.savefig(ruta_figura, dpi=180)
        plt.close(fig)
        logger.info(f"Grafica de MSE por estacion guardada: {ruta_figura.name}")

    return salidas


def evaluar_metricas(modelo: nn.Module, dataset_estaciones,
                     device: torch.device,
                     p_scale: float = 1.0) -> dict:
    """
    Evalua MSE por variable y residual NS en los datos de estaciones.

    :param modelo: red PINN entrenada.
    :param dataset_estaciones: EstacionesDataset con datos observados.
    :param device: dispositivo.
    :param p_scale: factor de reescalado de presion para las ecuaciones N-S.
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
    residual_ns = perdida_navier_stokes(modelo, t_ns, x_ns, y_ns,
                                        p_scale=p_scale).item()

    logger.info(f"MSE u: {mse_u:.3e} | MSE v: {mse_v:.3e} | MSE p: {mse_p:.3e}")
    logger.info(f"Residual NS: {residual_ns:.3e}")

    return {
        "mse_u": mse_u,
        "mse_v": mse_v,
        "mse_p": mse_p,
        "residual_ns": residual_ns,
    }
