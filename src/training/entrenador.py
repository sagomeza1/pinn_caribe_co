"""
Bucle de entrenamiento de la PINN.

Implementa el entrenamiento completo con:
    - Deteccion automatica de CUDA
    - DataLoaders para estaciones y puntos de colocacion
    - Optimizador Adam + ReduceLROnPlateau
    - Gradient clipping
    - Funcion de perdida combinada (del proyecto base)
    - Logging por epoca
    - Checkpoints periodicos
    - Historial en formato .mat

Referencia: trabajo_base_pinns/src/train_pinn.py — funcion train_pinn_brusselas
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

from pathlib import Path
from torch.utils.data import DataLoader

from config.settings import (
    CUANTIL_MAX,
    CUANTIL_MIN,
    EPOCAS_RAMPA_RANGO,
    LAMBDA_FISICA, NUM_EPOCAS, LR_INICIAL, MAX_NORM_GRAD,
    LAMBDA_RANGO_MAX,
    SCHEDULER_FACTOR, SCHEDULER_PATIENCE, SCHEDULER_THRESHOLD,
    CHECKPOINT_INTERVALO, RUTA_MODELOS, R_MALLA, N_DIAS,
    TAU_RANGO,
    EPOCAS_SOLO_DATOS, EPOCAS_RAMPA,
    USAR_CUANTILES_RANGO,
)
from src.model.perdida_fisica import perdida_navier_stokes, perdida_datos

logger = logging.getLogger(__name__)


def _calcular_limites_rango(
    dataset_estaciones,
    usar_cuantiles: bool,
    cuantil_min: float,
    cuantil_max: float,
) -> dict[str, torch.Tensor]:
    """Calcula limites inferior/superior por variable en datos observados.

    :param dataset_estaciones: dataset de estaciones con tensores u, v, p.
    :param usar_cuantiles: si True usa cuantiles robustos; si False usa min/max.
    :param cuantil_min: cuantil inferior en [0, 1].
    :param cuantil_max: cuantil superior en [0, 1].
    :return: diccionario con limites por variable (u, v, p).
    :raises ValueError: si los cuantiles no estan en rango valido.
    """
    if not (0.0 <= cuantil_min < cuantil_max <= 1.0):
        raise ValueError("Los cuantiles de rango deben cumplir 0 <= min < max <= 1")

    limites = {}
    for nombre, tensor in {
        "u": dataset_estaciones.u.squeeze(),
        "v": dataset_estaciones.v.squeeze(),
        "p": dataset_estaciones.p.squeeze(),
    }.items():
        if usar_cuantiles:
            lim_inf = torch.quantile(tensor, cuantil_min)
            lim_sup = torch.quantile(tensor, cuantil_max)
        else:
            lim_inf = torch.min(tensor)
            lim_sup = torch.max(tensor)

        limites[nombre] = {
            "inf": lim_inf.detach(),
            "sup": lim_sup.detach(),
        }

    return limites


def _penalizacion_rango_suave(
    pred: torch.Tensor,
    lim_inf: torch.Tensor,
    lim_sup: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula penalizacion suave y porcentaje fuera de rango.

    :param pred: predicciones de una variable con forma (N, 1).
    :param lim_inf: limite inferior escalar.
    :param lim_sup: limite superior escalar.
    :param tau: temperatura de suavizado (>0).
    :return: tupla (perdida_suave, porcentaje_fuera_de_rango).
    """
    tau_efectivo = max(float(tau), 1e-6)

    exceso_superior = torch.nn.functional.softplus((pred - lim_sup) / tau_efectivo) * tau_efectivo
    exceso_inferior = torch.nn.functional.softplus((lim_inf - pred) / tau_efectivo) * tau_efectivo
    perdida = torch.mean(exceso_superior + exceso_inferior)

    fuera = ((pred < lim_inf) | (pred > lim_sup)).float().mean()
    return perdida, fuera


def _guardar_estado_entrenamiento(
    nombre_modelo: str,
    epoca: int,
    modelo: nn.Module,
    optimizador: optim.Optimizer,
    scheduler,
    avg_loss: float,
    p_scale: float,
    historial: dict,
) -> None:
    """Guarda checkpoint e historial del entrenamiento.

    :param nombre_modelo: prefijo de archivos de salida.
    :param epoca: epoca actual base cero.
    :param modelo: red entrenada.
    :param optimizador: optimizador activo.
    :param scheduler: scheduler activo.
    :param avg_loss: loss promedio de la epoca.
    :param p_scale: factor de reescalado de presion.
    :param historial: historial acumulado.
    :return: None.
    """
    ruta_ckpt = RUTA_MODELOS / f"{nombre_modelo}_epoca_{epoca + 1}.pth"
    torch.save({
        "epoch": epoca,
        "model_state_dict": modelo.state_dict(),
        "optimizer_state_dict": optimizador.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": avg_loss,
        "p_scale": p_scale,
    }, ruta_ckpt)
    logger.info(f"Checkpoint guardado: {ruta_ckpt.name}")

    ruta_hist = RUTA_MODELOS / f"historial_{nombre_modelo}.mat"
    sio.savemat(str(ruta_hist), historial)
    logger.info(f"Historial guardado: {ruta_hist.name}")
    logger.info(f"Epoca {epoca + 1} con loss {avg_loss:.2e}")


def entrenar(
    modelo: nn.Module,
    dataset_estaciones,
    dataset_colocacion,
    device: torch.device,
    lr: float = LR_INICIAL,
    lamb: float = LAMBDA_FISICA,
    num_epocas: int = NUM_EPOCAS,
    max_norm: float = MAX_NORM_GRAD,
    checkpoint_intervalo: int = CHECKPOINT_INTERVALO,
    nombre_modelo: str = "PINN_caribe",
    p_scale: float = 1.0,
    epocas_solo_datos: int = EPOCAS_SOLO_DATOS,
    epocas_rampa: int = EPOCAS_RAMPA,
    lambda_rango_max: float = LAMBDA_RANGO_MAX,
    epocas_rampa_rango: int = EPOCAS_RAMPA_RANGO,
    tau_rango: float = TAU_RANGO,
    usar_cuantiles_rango: bool = USAR_CUANTILES_RANGO,
    cuantil_min: float = CUANTIL_MIN,
    cuantil_max: float = CUANTIL_MAX,
    r_malla: float = R_MALLA,
    n_dias: int = N_DIAS,
) -> dict:
    """
    Ejecuta el bucle de entrenamiento de la PINN.

    Funcion de perdida combinada (replica del proyecto base):
        loss = (lamb*loss_ns^2 + loss_u^2 + loss_v^2 + loss_p^2)
             / (lamb*loss_ns + loss_u + loss_v + loss_p)

    La perdida fisica se calcula tanto en puntos de estaciones como en puntos
    de colocacion (malla).

    Curriculum learning (Iteracion 2):
        - Fase 1 (epocas 0 a epocas_solo_datos-1): lambda=0 (solo datos)
        - Fase 2 (rampa lineal): lambda de 0 a lamb
        - Fase 3 (restantes): lambda=lamb completo

    :param modelo: red PINN a entrenar.
    :param dataset_estaciones: EstacionesDataset con datos observados.
    :param dataset_colocacion: ColocacionDataset con puntos de la malla.
    :param device: dispositivo (cuda o cpu).
    :param lr: learning rate inicial.
    :param lamb: peso maximo de la perdida fisica.
    :param num_epocas: numero de epocas.
    :param max_norm: norma maxima para gradient clipping.
    :param checkpoint_intervalo: guardar checkpoint cada N epocas.
    :param nombre_modelo: prefijo del nombre de archivos de checkpoint.
    :param p_scale: factor de reescalado de presion para las ecuaciones N-S.
    :param epocas_solo_datos: epocas iniciales con lambda=0 (curriculum).
    :param epocas_rampa: epocas de rampa lineal de lambda (curriculum).
    :param lambda_rango_max: peso maximo de la restriccion suave de rango.
    :param epocas_rampa_rango: epocas de rampa lineal de lambda_rango.
    :param tau_rango: temperatura de suavizado para penalizacion softplus.
    :param usar_cuantiles_rango: si True usa cuantiles para limites robustos.
    :param cuantil_min: cuantil inferior para definir limite robusto.
    :param cuantil_max: cuantil superior para definir limite robusto.
    :param r_malla: resolucion de la malla para el calculo de batches.
    :param n_dias: horizonte temporal usado para escalar el batch size.
    :return: diccionario con el historial de entrenamiento.
    """
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Epocas: {num_epocas:,}, LR: {lr:.1e}, Lambda max: {lamb}")
    logger.info(f"Configuracion corrida: R={r_malla}, n_dias={n_dias}")
    logger.info(f"Curriculum: {epocas_solo_datos} solo datos, "
                f"{epocas_rampa} rampa, "
                f"{num_epocas - epocas_solo_datos - epocas_rampa} lambda completo")
    logger.info(f"Range loss: lambda_max={lambda_rango_max}, "
                f"rampa={epocas_rampa_rango}, tau={tau_rango}")
    logger.info(f"Limites de rango por {'cuantiles' if usar_cuantiles_rango else 'min/max'}: "
                f"qmin={cuantil_min:.3f}, qmax={cuantil_max:.3f}")
    logger.info(f"P_scale (reescalado presion): {p_scale:.4f}")

    # Mover modelo a GPU
    modelo.to(device)

    # Tamanos de batch (replica logica del proyecto base)
    batch_estaciones = max(1, int(np.ceil(len(dataset_estaciones) / n_dias * r_malla)))
    batch_colocacion = max(1, int(np.ceil(len(dataset_colocacion) / n_dias * r_malla)))

    logger.info(f"Registros estaciones: {len(dataset_estaciones):,}")
    logger.info(f"Registros colocacion: {len(dataset_colocacion):,}")
    logger.info(f"Batch estaciones: {batch_estaciones}")
    logger.info(f"Batch colocacion: {batch_colocacion}")

    loader_estaciones = DataLoader(
        dataset_estaciones, batch_size=batch_estaciones, shuffle=True, drop_last=True,
    )
    loader_colocacion = DataLoader(
        dataset_colocacion, batch_size=batch_colocacion, shuffle=True, drop_last=True,
    )

    # Optimizador y scheduler
    optimizador = optim.Adam(modelo.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        threshold=SCHEDULER_THRESHOLD,
    )

    logger.debug(f"Scheduler: factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE}")
    logger.debug(f"Max norm gradient: {max_norm}")

    limites_rango = _calcular_limites_rango(
        dataset_estaciones=dataset_estaciones,
        usar_cuantiles=usar_cuantiles_rango,
        cuantil_min=cuantil_min,
        cuantil_max=cuantil_max,
    )
    logger.info(
        "Limites rango u:[%.3f, %.3f] v:[%.3f, %.3f] p:[%.3f, %.3f]",
        limites_rango["u"]["inf"].item(), limites_rango["u"]["sup"].item(),
        limites_rango["v"]["inf"].item(), limites_rango["v"]["sup"].item(),
        limites_rango["p"]["inf"].item(), limites_rango["p"]["sup"].item(),
    )

    # Historial
    historial = {
        "epoch": [], "loss": [], "ns_loss": [],
        "p_loss": [], "u_loss": [], "v_loss": [], "lr": [],
        "lambda_efecto": [],
        "range_loss": [], "range_u_loss": [], "range_v_loss": [], "range_p_loss": [],
        "for_u": [], "for_v": [], "for_p": [], "lambda_rango_efecto": [],
    }

    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    lr_anterior = 0.0

    logger.info("Iniciando entrenamiento")

    for epoca in range(num_epocas):
        modelo.train()

        # Curriculum learning: calcular lambda efectivo para esta epoca
        if epoca < epocas_solo_datos:
            lambda_efecto = 0.0
        elif epoca < epocas_solo_datos + epocas_rampa:
            progreso = (epoca - epocas_solo_datos) / epocas_rampa
            lambda_efecto = lamb * progreso
        else:
            lambda_efecto = lamb

        # Logging de transiciones de fase
        if epoca == 0:
            logger.info(f"Curriculum fase 1: solo datos (epocas 0-{epocas_solo_datos - 1})")
        elif epoca == epocas_solo_datos:
            logger.info(f"Curriculum fase 2: rampa lambda "
                        f"(epocas {epocas_solo_datos}-{epocas_solo_datos + epocas_rampa - 1})")
        elif epoca == epocas_solo_datos + epocas_rampa:
            logger.info(f"Curriculum fase 3: lambda completo = {lamb}")

        acum_loss = 0.0
        acum_ns = 0.0
        acum_u = 0.0
        acum_v = 0.0
        acum_p = 0.0
        acum_range = 0.0
        acum_range_u = 0.0
        acum_range_v = 0.0
        acum_range_p = 0.0
        acum_for_u = 0.0
        acum_for_v = 0.0
        acum_for_p = 0.0
        n_batches = 0

        if epocas_rampa_rango > 0 and epoca < epocas_rampa_rango:
            lambda_rango_efecto = lambda_rango_max * (epoca + 1) / epocas_rampa_rango
        else:
            lambda_rango_efecto = lambda_rango_max

        for batch_est, batch_col in zip(loader_estaciones, loader_colocacion):
            # Datos observados
            t_u, x_u, y_u, u_true, v_true, p_true = [b.to(device) for b in batch_est]

            # Puntos de colocacion (fisica)
            t_f, x_f, y_f = [b.to(device) for b in batch_col]

            optimizador.zero_grad()

            # 1. Perdida fisica (NS) — solo si lambda_efecto > 0
            if lambda_efecto > 0:
                loss_ns_malla = perdida_navier_stokes(modelo, t_f, x_f, y_f,
                                                      p_scale=p_scale)
                loss_ns_datos = perdida_navier_stokes(modelo, t_u, x_u, y_u,
                                                      p_scale=p_scale)
                loss_fisica = lambda_efecto * (loss_ns_malla + loss_ns_datos)
            else:
                loss_fisica = torch.tensor(0.0, device=device)

            # 2. Perdida de datos (prediccion vs observacion)
            salida = modelo(t_u, x_u, y_u)
            u_pred = salida[:, 0:1]
            v_pred = salida[:, 1:2]
            p_pred = salida[:, 2:3]

            loss_u = perdida_datos(u_pred, u_true)
            loss_v = perdida_datos(v_pred, v_true)
            loss_p = perdida_datos(p_pred, p_true)

            loss_rango_u, for_u = _penalizacion_rango_suave(
                u_pred,
                lim_inf=limites_rango["u"]["inf"].to(device),
                lim_sup=limites_rango["u"]["sup"].to(device),
                tau=tau_rango,
            )
            loss_rango_v, for_v = _penalizacion_rango_suave(
                v_pred,
                lim_inf=limites_rango["v"]["inf"].to(device),
                lim_sup=limites_rango["v"]["sup"].to(device),
                tau=tau_rango,
            )
            loss_rango_p, for_p = _penalizacion_rango_suave(
                p_pred,
                lim_inf=limites_rango["p"]["inf"].to(device),
                lim_sup=limites_rango["p"]["sup"].to(device),
                tau=tau_rango,
            )
            loss_rango = lambda_rango_efecto * (loss_rango_u + loss_rango_v + loss_rango_p)

            # 3. Perdida combinada (formula del proyecto base)
            denominador = loss_fisica + loss_u + loss_v + loss_p + loss_rango + 1e-12
            loss_total = (
                (loss_fisica ** 2 + loss_u ** 2 + loss_v ** 2 + loss_p ** 2 + loss_rango ** 2)
                / denominador
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=max_norm)
            optimizador.step()

            # Acumular metricas
            acum_loss += loss_total.item()
            acum_ns += loss_fisica.item()
            acum_u += loss_u.item()
            acum_v += loss_v.item()
            acum_p += loss_p.item()
            acum_range += loss_rango.item()
            acum_range_u += loss_rango_u.item()
            acum_range_v += loss_rango_v.item()
            acum_range_p += loss_rango_p.item()
            acum_for_u += for_u.item()
            acum_for_v += for_v.item()
            acum_for_p += for_p.item()
            n_batches += 1

        # Promedios de la epoca
        avg_loss = acum_loss / n_batches
        avg_ns = acum_ns / n_batches
        avg_u = acum_u / n_batches
        avg_v = acum_v / n_batches
        avg_p = acum_p / n_batches
        avg_range = acum_range / n_batches
        avg_range_u = acum_range_u / n_batches
        avg_range_v = acum_range_v / n_batches
        avg_range_p = acum_range_p / n_batches
        avg_for_u = acum_for_u / n_batches
        avg_for_v = acum_for_v / n_batches
        avg_for_p = acum_for_p / n_batches

        if avg_ns < 1e-5 and lambda_efecto > 0:
            logger.warning(f"Loss NS muy bajo: {avg_ns:.3e}")

        # Actualizar scheduler
        scheduler.step(avg_loss)
        lr_actual = optimizador.param_groups[0]["lr"]

        if lr_anterior != lr_actual:
            logger.info(f"Cambio de LR: {lr_anterior:.1e} -> {lr_actual:.1e}")
        lr_anterior = lr_actual

        # Guardar historial
        historial["epoch"].append(epoca)
        historial["loss"].append(avg_loss)
        historial["ns_loss"].append(avg_ns)
        historial["u_loss"].append(avg_u)
        historial["v_loss"].append(avg_v)
        historial["p_loss"].append(avg_p)
        historial["range_loss"].append(avg_range)
        historial["range_u_loss"].append(avg_range_u)
        historial["range_v_loss"].append(avg_range_v)
        historial["range_p_loss"].append(avg_range_p)
        historial["for_u"].append(avg_for_u)
        historial["for_v"].append(avg_for_v)
        historial["for_p"].append(avg_for_p)
        historial["lr"].append(lr_actual)
        historial["lambda_efecto"].append(lambda_efecto)
        historial["lambda_rango_efecto"].append(lambda_rango_efecto)

        # Logging
        if epoca % 10 == 0:
            logger.info(
                f"| Epoca: {epoca:4} | Loss: {avg_loss:.3e} | LR: {lr_actual:.1e} "
                f"| LambdaNS: {lambda_efecto:.2f} | LambdaR: {lambda_rango_efecto:.2f} "
                f"| FOR(u,v,p): {avg_for_u:.2%}, {avg_for_v:.2%}, {avg_for_p:.2%} |"
            )
        logger.debug(
            f"|Epoca: {epoca:4}|Loss: {avg_loss:.3e}|NS: {avg_ns:.3e}"
            f"|U: {avg_u:.3e}|V: {avg_v:.3e}|P: {avg_p:.3e}"
            f"|R: {avg_range:.3e}|FOR_u: {avg_for_u:.2%}|FOR_v: {avg_for_v:.2%}"
            f"|FOR_p: {avg_for_p:.2%}|LR: {lr_actual:.1e}|"
            f"LambdaNS: {lambda_efecto:.2f}|LambdaR: {lambda_rango_efecto:.2f}|"
        )

        # Checkpoint periodico
        if (epoca + 1) % checkpoint_intervalo == 0:
            _guardar_estado_entrenamiento(
                nombre_modelo=nombre_modelo,
                epoca=epoca,
                modelo=modelo,
                optimizador=optimizador,
                scheduler=scheduler,
                avg_loss=avg_loss,
                p_scale=p_scale,
                historial=historial,
            )

    if num_epocas % checkpoint_intervalo != 0:
        _guardar_estado_entrenamiento(
            nombre_modelo=nombre_modelo,
            epoca=num_epocas - 1,
            modelo=modelo,
            optimizador=optimizador,
            scheduler=scheduler,
            avg_loss=historial["loss"][-1],
            p_scale=p_scale,
            historial=historial,
        )

    logger.info("Entrenamiento completado")
    return historial
