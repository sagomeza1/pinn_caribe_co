"""
Visualizacion geografica simple de estaciones meteorologicas.

Genera un mapa en coordenadas longitud-latitud y resalta en otro color
las estaciones excluidas del entrenamiento.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

logger = logging.getLogger(__name__)


def graficar_mapa_estaciones(
    datos: dict,
    estaciones_excluidas: list[str],
    ruta_salida: Path,
) -> Path:
    """
    Genera un mapa de estaciones y resalta las estaciones excluidas.

    :param datos: diccionario con codigo_estacion, latitud y longitud.
    :param estaciones_excluidas: lista de codigos de estaciones excluidas.
    :param ruta_salida: ruta del archivo PNG a generar.
    :return: ruta del archivo generado.
    :raises KeyError: si faltan columnas requeridas en datos.
    """
    columnas_requeridas = ["codigo_estacion", "latitud", "longitud"]
    faltantes = [col for col in columnas_requeridas if col not in datos]
    if faltantes:
        raise KeyError(f"Faltan columnas requeridas para el mapa: {faltantes}")

    df = pd.DataFrame(
        {
            "codigo_estacion": datos["codigo_estacion"],
            "latitud": datos["latitud"],
            "longitud": datos["longitud"],
        }
    )

    estaciones = (
        df.groupby("codigo_estacion", as_index=False)[["latitud", "longitud"]]
        .first()
        .sort_values("codigo_estacion")
    )
    estaciones["estado"] = estaciones["codigo_estacion"].isin(estaciones_excluidas)

    activas = estaciones[~estaciones["estado"]]
    excluidas = estaciones[estaciones["estado"]]

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F4EA")

    ax.scatter(
        activas["longitud"],
        activas["latitud"],
        s=80,
        color="#2A9D8F",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
        label="Estaciones activas",
        zorder=3,
    )
    ax.scatter(
        excluidas["longitud"],
        excluidas["latitud"],
        s=120,
        color="#D1495B",
        marker="X",
        edgecolors="black",
        linewidths=0.7,
        alpha=0.95,
        label="Estaciones removidas",
        zorder=4,
    )

    for _, fila in estaciones.iterrows():
        color_texto = "#8C1C13" if fila["estado"] else "#264653"
        ax.annotate(
            fila["codigo_estacion"],
            (fila["longitud"], fila["latitud"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color_texto,
        )

    ax.set_title("Mapa de estaciones meteorologicas", fontsize=14)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True)

    margen_lon = 0.25
    margen_lat = 0.25
    ax.set_xlim(estaciones["longitud"].min() - margen_lon, estaciones["longitud"].max() + margen_lon)
    ax.set_ylim(estaciones["latitud"].min() - margen_lat, estaciones["latitud"].max() + margen_lat)

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=180)
    plt.close(fig)

    logger.info(f"Mapa de estaciones guardado: {ruta_salida.name}")
    return ruta_salida