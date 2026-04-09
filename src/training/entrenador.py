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
    LAMBDA_FISICA, NUM_EPOCAS, LR_INICIAL, MAX_NORM_GRAD,
    SCHEDULER_FACTOR, SCHEDULER_PATIENCE, SCHEDULER_THRESHOLD,
    CHECKPOINT_INTERVALO, RUTA_MODELOS, R_MALLA, N_DIAS,
)
from src.model.perdida_fisica import perdida_navier_stokes, perdida_datos

logger = logging.getLogger(__name__)


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
) -> dict:
    """
    Ejecuta el bucle de entrenamiento de la PINN.

    Funcion de perdida combinada (replica del proyecto base):
        loss = (lamb*loss_ns^2 + loss_u^2 + loss_v^2 + loss_p^2)
             / (lamb*loss_ns + loss_u + loss_v + loss_p)

    La perdida fisica se calcula tanto en puntos de estaciones como en puntos
    de colocacion (malla).

    :param modelo: red PINN a entrenar.
    :param dataset_estaciones: EstacionesDataset con datos observados.
    :param dataset_colocacion: ColocacionDataset con puntos de la malla.
    :param device: dispositivo (cuda o cpu).
    :param lr: learning rate inicial.
    :param lamb: peso de la perdida fisica.
    :param num_epocas: numero de epocas.
    :param max_norm: norma maxima para gradient clipping.
    :param checkpoint_intervalo: guardar checkpoint cada N epocas.
    :param nombre_modelo: prefijo del nombre de archivos de checkpoint.
    :return: diccionario con el historial de entrenamiento.
    """
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Epocas: {num_epocas:,}, LR: {lr:.1e}, Lambda: {lamb}")

    # Mover modelo a GPU
    modelo.to(device)

    # Tamanos de batch (replica logica del proyecto base)
    batch_estaciones = int(np.ceil(len(dataset_estaciones) / N_DIAS * R_MALLA))
    batch_colocacion = int(np.ceil(len(dataset_colocacion) / N_DIAS * R_MALLA))

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

    # Historial
    historial = {
        "epoch": [], "loss": [], "ns_loss": [],
        "p_loss": [], "u_loss": [], "v_loss": [], "lr": [],
    }

    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    lr_anterior = 0.0

    logger.info("Iniciando entrenamiento")

    for epoca in range(num_epocas):
        modelo.train()

        acum_loss = 0.0
        acum_ns = 0.0
        acum_u = 0.0
        acum_v = 0.0
        acum_p = 0.0
        n_batches = 0

        for batch_est, batch_col in zip(loader_estaciones, loader_colocacion):
            # Datos observados
            t_u, x_u, y_u, u_true, v_true, p_true = [b.to(device) for b in batch_est]

            # Puntos de colocacion (fisica)
            t_f, x_f, y_f = [b.to(device) for b in batch_col]

            optimizador.zero_grad()

            # 1. Perdida fisica (NS) en malla y en puntos de estaciones
            loss_ns_malla = perdida_navier_stokes(modelo, t_f, x_f, y_f)
            loss_ns_datos = perdida_navier_stokes(modelo, t_u, x_u, y_u)
            loss_fisica = lamb * (loss_ns_malla + loss_ns_datos)

            # 2. Perdida de datos (prediccion vs observacion)
            salida = modelo(t_u, x_u, y_u)
            u_pred = salida[:, 0:1]
            v_pred = salida[:, 1:2]
            p_pred = salida[:, 2:3]

            loss_u = perdida_datos(u_pred, u_true)
            loss_v = perdida_datos(v_pred, v_true)
            loss_p = perdida_datos(p_pred, p_true)

            # 3. Perdida combinada (formula del proyecto base)
            loss_total = (
                (loss_fisica ** 2 + loss_u ** 2 + loss_v ** 2 + loss_p ** 2)
                / (loss_fisica + loss_u + loss_v + loss_p)
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
            n_batches += 1

        # Promedios de la epoca
        avg_loss = acum_loss / n_batches
        avg_ns = acum_ns / n_batches
        avg_u = acum_u / n_batches
        avg_v = acum_v / n_batches
        avg_p = acum_p / n_batches

        if avg_ns < 1e-5:
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
        historial["lr"].append(lr_actual)

        # Logging
        if epoca % 10 == 0:
            logger.info(
                f"| Epoca: {epoca:4} | Loss: {avg_loss:.3e} | LR: {lr_actual:.1e} |"
            )
        logger.debug(
            f"|Epoca: {epoca:4}|Loss: {avg_loss:.3e}|NS: {avg_ns:.3e}"
            f"|U: {avg_u:.3e}|V: {avg_v:.3e}|P: {avg_p:.3e}|LR: {lr_actual:.1e}|"
        )

        # Checkpoint periodico
        if (epoca + 1) % checkpoint_intervalo == 0:
            ruta_ckpt = RUTA_MODELOS / f"{nombre_modelo}_epoca_{epoca + 1}.pth"
            torch.save({
                "epoch": epoca,
                "model_state_dict": modelo.state_dict(),
                "optimizer_state_dict": optimizador.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            }, ruta_ckpt)
            logger.info(f"Checkpoint guardado: {ruta_ckpt.name}")

            # Guardar historial como .mat
            ruta_hist = RUTA_MODELOS / f"historial_{nombre_modelo}.mat"
            sio.savemat(str(ruta_hist), historial)
            logger.info(f"Historial guardado: {ruta_hist.name}")
            logger.info(f"Epoca {epoca + 1} con loss {avg_loss:.2e}")

    logger.info("Entrenamiento completado")
    return historial
