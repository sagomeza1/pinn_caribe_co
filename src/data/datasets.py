"""
Datasets de PyTorch para el entrenamiento PINN.

Define dos datasets:
    - EstacionesDataset: datos observados (t, x, y, u, v, p) para la perdida de datos.
    - ColocacionDataset: puntos de la malla (t, x, y) para la perdida fisica.

Referencia: trabajo_base_pinns/src/dataset.py
"""

import torch
from torch.utils.data import Dataset


class EstacionesDataset(Dataset):
    """
    Dataset para los datos medidos en estaciones meteorologicas.

    Aplana las matrices (N_estaciones, N_tiempos) a vectores (N_total, 1)
    y filtra filas con NaN en cualquiera de las variables objetivo.

    :param datos: diccionario con claves T, X, Y, U, V, P (matrices numpy).
    """

    def __init__(self, datos: dict):
        self.t = torch.tensor(datos["T"].flatten(), dtype=torch.float32).unsqueeze(1)
        self.x = torch.tensor(datos["X"].flatten(), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(datos["Y"].flatten(), dtype=torch.float32).unsqueeze(1)
        self.u = torch.tensor(datos["U"].flatten(), dtype=torch.float32).unsqueeze(1)
        self.v = torch.tensor(datos["V"].flatten(), dtype=torch.float32).unsqueeze(1)
        self.p = torch.tensor(datos["P"].flatten(), dtype=torch.float32).unsqueeze(1)

        # Filtrar filas con NaN en cualquier variable objetivo
        mascara = (
            ~torch.isnan(self.u.squeeze())
            & ~torch.isnan(self.v.squeeze())
            & ~torch.isnan(self.p.squeeze())
        )
        self.t = self.t[mascara]
        self.x = self.x[mascara]
        self.y = self.y[mascara]
        self.u = self.u[mascara]
        self.v = self.v[mascara]
        self.p = self.p[mascara]

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx], self.y[idx], self.u[idx], self.v[idx], self.p[idx]


class ColocacionDataset(Dataset):
    """
    Dataset para puntos de colocacion (malla PINN) donde se evalua la fisica.

    :param malla: diccionario con claves T, X, Y (matrices numpy).
    """

    def __init__(self, malla: dict):
        self.t = torch.tensor(malla["T"].flatten(), dtype=torch.float32).unsqueeze(1)
        self.x = torch.tensor(malla["X"].flatten(), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(malla["Y"].flatten(), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx], self.y[idx]
