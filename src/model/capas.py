"""
Capa personalizada GammaBiasLayer con Weight Normalization desacoplada.

Implementa una capa lineal con:
    - Weight Normalization (torch.nn.utils.parametrizations.weight_norm)
    - Inicializacion Xavier Uniform
    - Parametros gamma (escalado) y beta (desplazamiento) entrenables

Referencia: trabajo_base_pinns/src/model_pinn.py — clase GammaBiasLayer
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class GammaBiasLayer(nn.Module):
    """
    Capa densa con normalizacion de pesos, factor gamma y sesgo explicito.

    Forward: y = gamma * (W_norm * x) + beta

    :param in_features: dimension de entrada.
    :param out_features: dimension de salida.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        # Capa lineal sin bias, con Weight Normalization
        self.linear = weight_norm(nn.Linear(in_features, out_features, bias=False))

        # Inicializacion Xavier Uniform
        nn.init.xavier_uniform_(self.linear.weight)

        # Parametros entrenables: gamma (escalado) y beta (desplazamiento)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor de entrada (batch, in_features).
        :return: tensor de salida (batch, out_features).
        """
        return self.gamma * self.linear(x) + self.beta
