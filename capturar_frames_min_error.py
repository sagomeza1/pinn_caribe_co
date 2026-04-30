"""Captura frames en el instante de menor error promedio por variable.

Para un experimento (manifiesto + checkpoint), calcula el error promedio
absoluto (MAE) por timestep en estaciones para u, v y p. Luego identifica
el timestep con menor MAE para cada variable y exporta la imagen del frame
correspondiente sobre la malla.
"""

import argparse
import json
import logging

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config.logging_config import configurar_logging
from config.settings import HIDDEN_NEURONS, RUTA_MODELOS, RUTA_REPORTES
from src.data.cargar_datos import cargar_parquet
from src.data.construir_malla import construir_malla
from src.data.normalizar_datos import normalizar
from src.data.preprocesar_datos import preprocesar
from src.model.pinn import PINN
from src.training.manifiestos import cargar_manifiesto_experimento

logger = logging.getLogger(__name__)


def _predecir_en_lotes(modelo: torch.nn.Module, t: np.ndarray, x: np.ndarray,
                       y: np.ndarray, device: torch.device,
                       batch_size: int = 50_000) -> np.ndarray:
    """Predice en lotes para evitar desbordar memoria."""
    t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    predicciones = []
    with torch.no_grad():
        for i in range(0, len(t_t), batch_size):
            out = modelo(t_t[i:i + batch_size], x_t[i:i + batch_size], y_t[i:i + batch_size])
            predicciones.append(out.detach().cpu().numpy())

    return np.vstack(predicciones)


def _guardar_frame_variable(
    variable: str,
    idx_var: int,
    t_idx: int,
    mae_min: float,
    segundos_t: float,
    pred_mesh_flat: np.ndarray,
    datos_norm: dict,
    malla: dict,
    ruta_salida: Path,
    prefijo: str,
) -> Path:
    """Guarda una imagen del frame de menor error para una variable."""
    etiqueta_var = {
        "u": "u_x (norm)",
        "v": "u_y (norm)",
        "p": "presion (norm)",
    }
    clave_obs = {
        "u": "U",
        "v": "V",
        "p": "P",
    }

    dim_n = malla["dim_N"]
    n_x = len(malla["x_pinn"])
    n_y = len(malla["y_pinn"])

    x_mesh = malla["X"][:, 0].reshape((n_y, n_x), order="F")
    y_mesh = malla["Y"][:, 0].reshape((n_y, n_x), order="F")

    pred_var_flat = pred_mesh_flat[:, idx_var]
    inicio = t_idx * dim_n
    fin = (t_idx + 1) * dim_n
    valores = pred_var_flat[inicio:fin].reshape((n_y, n_x), order="F")

    vmin = np.nanmin(pred_var_flat)
    vmax = np.nanmax(pred_var_flat)
    niveles = np.linspace(vmin, vmax, 50)

    fig, ax = plt.subplots(figsize=(9, 4))
    cs = ax.contourf(x_mesh, y_mesh, valores, levels=niveles, cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(etiqueta_var[variable])

    x_est = datos_norm["X"][:, t_idx]
    y_est = datos_norm["Y"][:, t_idx]
    val_est = datos_norm[clave_obs[variable]][:, t_idx]
    ax.scatter(
        x_est,
        y_est,
        c=val_est,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.5,
        s=55,
    )

    # Se mantienen disponibles para resumen/depuracion, pero no se imprimen en la figura.
    _ = segundos_t
    _ = mae_min
    _ = t_idx
    ax.set_xlabel("X (norm)")
    ax.set_ylabel("Y (norm)")

    nombre = f"{prefijo}_frame_min_error_{variable}_t{t_idx:03d}.png"
    ruta_img = ruta_salida / nombre
    fig.savefig(ruta_img, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ruta_img


def main(manifiesto_path: Path, carpeta_reportes: str | None = None) -> None:
    """Ejecuta la captura de frames de minimo error por variable."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    manifiesto = cargar_manifiesto_experimento(manifiesto_path)
    nombre_modelo = str(manifiesto["nombre_modelo"])
    nombre_experimento = str(manifiesto.get("nombre_experimento", nombre_modelo))
    epoca = int(manifiesto["num_epocas"])

    ruta_checkpoint = RUTA_MODELOS / f"{nombre_modelo}_epoca_{epoca}.pth"
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ruta_checkpoint}")

    if carpeta_reportes:
        ruta_salida = RUTA_REPORTES / carpeta_reportes
    else:
        ruta_salida = RUTA_REPORTES / nombre_experimento
    ruta_salida.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ruta_checkpoint, map_location=device, weights_only=False)
    p_scale = float(checkpoint.get("p_scale", 1.0))
    reescalar = abs(p_scale - 1.0) > 1e-6

    datos_raw = cargar_parquet(
        Path(manifiesto["archivo_parquet"]),
        estaciones_excluidas=list(manifiesto.get("estaciones_excluidas", [])),
        registros_por_estacion=manifiesto.get("registros_por_estacion"),
    )
    datos_prep = preprocesar(
        datos_raw,
        n_dias=int(manifiesto["n_dias"]),
        intervalo=int(manifiesto["intervalo"]),
    )
    datos_norm, escalas = normalizar(datos_prep, reescalar_presion=reescalar)
    malla = construir_malla(datos_norm, escalas, R=float(manifiesto["r_malla"]))

    modelo = PINN(hidden_neurons=HIDDEN_NEURONS)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(device)
    modelo.eval()

    # 1) Predicciones en estaciones para calcular MAE por timestep.
    t_est = datos_norm["T"].flatten(order="F")
    x_est = datos_norm["X"].flatten(order="F")
    y_est = datos_norm["Y"].flatten(order="F")
    pred_est = _predecir_en_lotes(modelo, t_est, x_est, y_est, device=device)

    n_est, n_t = datos_norm["U"].shape
    pred_u = pred_est[:, 0].reshape((n_est, n_t), order="F")
    pred_v = pred_est[:, 1].reshape((n_est, n_t), order="F")
    pred_p = pred_est[:, 2].reshape((n_est, n_t), order="F")

    mae_u_t = np.nanmean(np.abs(pred_u - datos_norm["U"]), axis=0)
    mae_v_t = np.nanmean(np.abs(pred_v - datos_norm["V"]), axis=0)
    mae_p_t = np.nanmean(np.abs(pred_p - datos_norm["P"]), axis=0)

    idx_u = int(np.nanargmin(mae_u_t))
    idx_v = int(np.nanargmin(mae_v_t))
    idx_p = int(np.nanargmin(mae_p_t))

    # 2) Predicciones en malla para capturar el frame visual del GIF.
    t_mesh = malla["T"].flatten()
    x_mesh = malla["X"].flatten()
    y_mesh = malla["Y"].flatten()
    pred_mesh = _predecir_en_lotes(modelo, t_mesh, x_mesh, y_mesh, device=device)

    prefijo = f"{nombre_modelo.lower()}_epoca_{epoca}"

    ruta_u = _guardar_frame_variable(
        variable="u",
        idx_var=0,
        t_idx=idx_u,
        mae_min=float(mae_u_t[idx_u]),
        segundos_t=float(datos_prep["T"][0, idx_u]),
        pred_mesh_flat=pred_mesh,
        datos_norm=datos_norm,
        malla=malla,
        ruta_salida=ruta_salida,
        prefijo=prefijo,
    )
    ruta_v = _guardar_frame_variable(
        variable="v",
        idx_var=1,
        t_idx=idx_v,
        mae_min=float(mae_v_t[idx_v]),
        segundos_t=float(datos_prep["T"][0, idx_v]),
        pred_mesh_flat=pred_mesh,
        datos_norm=datos_norm,
        malla=malla,
        ruta_salida=ruta_salida,
        prefijo=prefijo,
    )
    ruta_p = _guardar_frame_variable(
        variable="p",
        idx_var=2,
        t_idx=idx_p,
        mae_min=float(mae_p_t[idx_p]),
        segundos_t=float(datos_prep["T"][0, idx_p]),
        pred_mesh_flat=pred_mesh,
        datos_norm=datos_norm,
        malla=malla,
        ruta_salida=ruta_salida,
        prefijo=prefijo,
    )

    resumen = {
        "experimento": nombre_experimento,
        "modelo": nombre_modelo,
        "epoca": epoca,
        "min_error_promedio": {
            "u": {
                "timestep": idx_u,
                "segundos": float(datos_prep["T"][0, idx_u]),
                "mae": float(mae_u_t[idx_u]),
                "imagen": str(ruta_u),
            },
            "v": {
                "timestep": idx_v,
                "segundos": float(datos_prep["T"][0, idx_v]),
                "mae": float(mae_v_t[idx_v]),
                "imagen": str(ruta_v),
            },
            "p": {
                "timestep": idx_p,
                "segundos": float(datos_prep["T"][0, idx_p]),
                "mae": float(mae_p_t[idx_p]),
                "imagen": str(ruta_p),
            },
        },
    }

    ruta_json = ruta_salida / f"{prefijo}_resumen_min_error.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    logger.info("Resumen guardado: %s", ruta_json)
    logger.info("u minimo: t=%d, mae=%.6e", idx_u, mae_u_t[idx_u])
    logger.info("v minimo: t=%d, mae=%.6e", idx_v, mae_v_t[idx_v])
    logger.info("p minimo: t=%d, mae=%.6e", idx_p, mae_p_t[idx_p])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Captura frames en el timestep de menor error promedio por variable"
    )
    parser.add_argument(
        "--manifiesto",
        type=Path,
        required=True,
        help="Ruta al manifiesto JSON del experimento",
    )
    parser.add_argument(
        "--carpeta-reportes",
        type=str,
        default=None,
        help="Subcarpeta dentro de reports/ para guardar imagenes",
    )
    args = parser.parse_args()

    configurar_logging()
    main(args.manifiesto, args.carpeta_reportes)
