"""
Funciones de perdida para la PINN.

Implementa:
    - Calculo de derivadas parciales via autograd
    - Residuos de Navier-Stokes inviscido (Euler)
    - Perdida por variable de datos (MSE)

Referencia: trabajo_base_pinns/src/loss_functions.py
"""

import torch
import torch.nn as nn


mse = nn.MSELoss()


def calcular_derivadas(modelo: nn.Module, t: torch.Tensor,
                       x: torch.Tensor, y: torch.Tensor) -> tuple:
    """
    Calcula las derivadas parciales de las salidas del modelo respecto a las entradas.

    Habilita gradientes en t, x, y y usa autograd con create_graph=True
    para permitir la retropropagacion a traves de las derivadas.

    :param modelo: red PINN.
    :param t: tensor de tiempo (batch, 1).
    :param x: tensor de posicion X (batch, 1).
    :param y: tensor de posicion Y (batch, 1).
    :return: tupla (u, v, p, u_t, u_x, u_y, v_t, v_x, v_y, p_x, p_y).
    """
    t.requires_grad_(True)
    x.requires_grad_(True)
    y.requires_grad_(True)

    salida = modelo(t, x, y)
    u = salida[:, 0:1]
    v = salida[:, 1:2]
    p = salida[:, 2:3]

    ones = torch.ones_like(u)

    # Gradientes de u
    grads_u = torch.autograd.grad(u, [t, x, y], grad_outputs=ones, create_graph=True)
    u_t, u_x, u_y = grads_u

    # Gradientes de v
    grads_v = torch.autograd.grad(v, [t, x, y], grad_outputs=ones, create_graph=True)
    v_t, v_x, v_y = grads_v

    # Gradientes de p (solo espaciales)
    grads_p = torch.autograd.grad(p, [x, y], grad_outputs=ones, create_graph=True)
    p_x, p_y = grads_p

    return u, v, p, u_t, u_x, u_y, v_t, v_x, v_y, p_x, p_y


def perdida_navier_stokes(modelo: nn.Module, t: torch.Tensor,
                          x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calcula el residuo de las ecuaciones de Navier-Stokes inviscido (Euler).

    Ecuaciones (adimensionalizadas):
        - Continuidad:  du/dx + dv/dy = 0
        - Momento X:    du/dt + u*du/dx + v*du/dy + dp/dx = 0
        - Momento Y:    dv/dt + u*dv/dx + v*dv/dy + dp/dy = 0

    :param modelo: red PINN.
    :param t: tensor de tiempo (batch, 1).
    :param x: tensor de posicion X (batch, 1).
    :param y: tensor de posicion Y (batch, 1).
    :return: perdida escalar (MSE de los tres residuos contra cero).
    """
    u, v, p, u_t, u_x, u_y, v_t, v_x, v_y, p_x, p_y = calcular_derivadas(modelo, t, x, y)

    ceros = torch.zeros_like(u)

    # Residuos
    e1 = u_x + v_y                              # Continuidad
    e2 = u_t + (u * u_x + v * u_y) + p_x        # Momento X
    e3 = v_t + (u * v_x + v * v_y) + p_y        # Momento Y

    return mse(e1, ceros) + mse(e2, ceros) + mse(e3, ceros)


def perdida_datos(pred: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
    """
    Perdida MSE entre prediccion y objetivo para una variable.

    :param pred: prediccion del modelo (batch, 1).
    :param objetivo: valor observado (batch, 1).
    :return: perdida escalar MSE.
    """
    return mse(pred, objetivo)
