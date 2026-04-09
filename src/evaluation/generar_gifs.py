"""
Generacion de GIFs de curvas de nivel para u_x, u_y y presion.

Para cada variable, genera un GIF animado donde cada frame es un timestep,
mostrando contourf de la prediccion sobre la malla y scatter de las
observaciones reales en estaciones.

Referencia: trabajo_base_pinns/cuadernos/elaboracion_gifs.ipynb — funcion generar_gif
"""

import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path

from config.settings import RUTA_REPORTES

logger = logging.getLogger(__name__)


def generar_gif(
    modelo: torch.nn.Module,
    datos_norm: dict,
    malla: dict,
    escalas: dict,
    variable: str,
    nombre_archivo: str,
    device: torch.device,
    ruta_salida: Path = RUTA_REPORTES,
    fps: int = 60,
    dpi: int = 120,
) -> None:
    """
    Genera un GIF de curvas de nivel para una variable sobre la malla.

    :param modelo: red PINN entrenada (en modo eval).
    :param datos_norm: diccionario con datos normalizados de estaciones (T, X, Y, U, V, P).
    :param malla: diccionario con T, X, Y de la malla, mas x_pinn, y_pinn, dim_N, dim_T.
    :param escalas: diccionario con L, W, P0, rho.
    :param variable: 'u', 'v' o 'p'.
    :param nombre_archivo: nombre del archivo GIF (sin extension).
    :param device: dispositivo (cuda o cpu).
    :param ruta_salida: directorio donde guardar el GIF.
    :param fps: frames por segundo del GIF.
    :param dpi: resolucion del GIF.
    """
    ruta_salida.mkdir(parents=True, exist_ok=True)

    indice_var = {"u": 0, "v": 1, "p": 2}
    etiqueta_var = {"u": "u_x (m/s)", "v": "u_y (m/s)", "p": "Presion (Pa)"}

    idx = indice_var[variable]
    etiqueta = etiqueta_var[variable]

    logger.info(f"Generando GIF para variable '{variable}'")

    modelo.eval()

    # Dimensiones de la malla
    dim_N = malla["dim_N"]
    dim_T = malla["dim_T"]
    n_x = len(malla["x_pinn"])
    n_y = len(malla["y_pinn"])

    # Predecir sobre toda la malla
    T_flat = torch.tensor(malla["T"].flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    X_flat = torch.tensor(malla["X"].flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    Y_flat = torch.tensor(malla["Y"].flatten(), dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        # Predecir en lotes para no exceder memoria
        batch_size = 50_000
        predicciones = []
        for i in range(0, len(T_flat), batch_size):
            t_b = T_flat[i:i + batch_size]
            x_b = X_flat[i:i + batch_size]
            y_b = Y_flat[i:i + batch_size]
            pred = modelo(t_b, x_b, y_b)
            predicciones.append(pred[:, idx:idx + 1].cpu())
        pred_var = torch.cat(predicciones, dim=0).numpy().flatten()

    # Coordenadas X, Y de la malla para el contourf (normalizadas)
    x_mesh = malla["X"][:, 0].reshape((n_x, n_y), order="F")
    y_mesh = malla["Y"][:, 0].reshape((n_x, n_y), order="F")

    # Rango de la prediccion para colores consistentes
    val_min = np.nanmin(pred_var)
    val_max = np.nanmax(pred_var)
    niveles = np.linspace(val_min, val_max, 50)

    kwargs_contour = {
        "vmin": val_min, "vmax": val_max,
        "levels": niveles, "cmap": "viridis",
    }
    kwargs_scatter = {
        "vmin": val_min, "vmax": val_max,
        "cmap": "viridis", "edgecolors": "black",
        "linewidths": 0.5, "s": 50,
    }

    # Datos de estaciones
    n_est = datos_norm["numero_estaciones"]

    # Mapa variable de estaciones
    var_estaciones = {"u": "U", "v": "V", "p": "P"}
    clave_est = var_estaciones[variable]

    fig, ax = plt.subplots(figsize=(8, 3))
    cbar_creado = [False]

    def dibujar_frame(t_idx):
        ax.clear()

        # Prediccion PINN para este timestep
        inicio = t_idx * dim_N
        fin = (t_idx + 1) * dim_N
        valores = pred_var[inicio:fin].reshape((n_x, n_y), order="F")

        # Contourf
        cs = ax.contourf(x_mesh, y_mesh, valores, **kwargs_contour)

        if not cbar_creado[0]:
            fig.colorbar(cs, ax=ax, label=etiqueta)
            cbar_creado[0] = True

        # Scatter de estaciones
        x_est = datos_norm["X"][:, t_idx]
        y_est = datos_norm["Y"][:, t_idx]
        val_est = datos_norm[clave_est][:, t_idx]
        ax.scatter(x_est, y_est, c=val_est, **kwargs_scatter)

        ax.set_title(f"Timestep {t_idx}")
        ax.set_xlabel("X (norm)")
        ax.set_ylabel("Y (norm)")
        return []

    ani = animation.FuncAnimation(fig, dibujar_frame, frames=dim_T, interval=200, blit=False)

    ruta_gif = ruta_salida / f"{nombre_archivo}.gif"
    ani.save(str(ruta_gif), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)

    logger.info(f"GIF guardado: {ruta_gif}")


def generar_todos_los_gifs(
    modelo: torch.nn.Module,
    datos_norm: dict,
    malla: dict,
    escalas: dict,
    device: torch.device,
    prefijo: str = "pinn",
) -> None:
    """
    Genera GIFs para las tres variables de salida: u_x, u_y y presion.

    :param modelo: red PINN entrenada.
    :param datos_norm: diccionario con datos normalizados.
    :param malla: diccionario con la malla de prediccion.
    :param escalas: diccionario con escalas de referencia.
    :param device: dispositivo.
    :param prefijo: prefijo del nombre de archivos.
    """
    for var in ["u", "v", "p"]:
        generar_gif(
            modelo=modelo,
            datos_norm=datos_norm,
            malla=malla,
            escalas=escalas,
            variable=var,
            nombre_archivo=f"{prefijo}_{var}",
            device=device,
        )
