"""
Arquitectura de la red PINN completa.

Red con 12 capas ocultas x 1200 neuronas usando GammaBiasLayer:
    - Capa de entrada: GammaBiasLayer(3, 1200) + Tanh
    - 8 capas ocultas con Tanh
    - 4 capas ocultas lineales (sin activacion)
    - Capa de salida: GammaBiasLayer(1200, 3)

Entradas: (t, x, y) — dim=3
Salidas: (u_x, u_y, p) — dim=3

Referencia: trabajo_base_pinns/src/model_pinn.py — clase PINN
"""

import torch
import torch.nn as nn

from config.settings import INPUT_DIM, OUTPUT_DIM, HIDDEN_NEURONS, NUM_CAPAS_TANH, NUM_CAPAS_LINEALES
from src.model.capas import GammaBiasLayer


class PINN(nn.Module):
    """
    Red neuronal informada fisicamente (PINN).

    :param input_dim: dimension de entrada (por defecto 3: t, x, y).
    :param output_dim: dimension de salida (por defecto 3: u_x, u_y, p).
    :param hidden_neurons: neuronas por capa oculta (por defecto 1200).
    :param num_capas_tanh: capas ocultas con activacion Tanh (por defecto 8).
    :param num_capas_lineales: capas ocultas sin activacion (por defecto 4).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_neurons: int = HIDDEN_NEURONS,
        num_capas_tanh: int = NUM_CAPAS_TANH,
        num_capas_lineales: int = NUM_CAPAS_LINEALES,
    ):
        super().__init__()

        # Capa de entrada
        self.capa_entrada = GammaBiasLayer(input_dim, hidden_neurons)

        # Capas ocultas con activacion Tanh (la capa de entrada ya tiene Tanh,
        # por lo que aqui van num_capas_tanh - 1 capas adicionales)
        self.capas_tanh = nn.ModuleList([
            GammaBiasLayer(hidden_neurons, hidden_neurons)
            for _ in range(num_capas_tanh - 1)
        ])

        # Capas ocultas lineales (sin activacion)
        self.capas_lineales = nn.ModuleList([
            GammaBiasLayer(hidden_neurons, hidden_neurons)
            for _ in range(num_capas_lineales)
        ])

        # Capa de salida
        self.capa_salida = GammaBiasLayer(hidden_neurons, output_dim)

        self.tanh = nn.Tanh()

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Propagacion hacia adelante.

        :param t: tensor de tiempo (batch, 1).
        :param x: tensor de posicion X (batch, 1).
        :param y: tensor de posicion Y (batch, 1).
        :return: tensor de salida (batch, 3) con [u_x, u_y, p].
        """
        # Concatenar entradas
        h = torch.cat([t, x, y], dim=1)

        # Capa de entrada + Tanh
        h = self.tanh(self.capa_entrada(h))

        # Capas ocultas con Tanh
        for capa in self.capas_tanh:
            h = self.tanh(capa(h))

        # Capas ocultas lineales
        for capa in self.capas_lineales:
            h = capa(h)

        # Capa de salida
        salida = self.capa_salida(h)

        return salida
