"""
Genera un mapa con la ubicación de las estaciones meteorológicas del Caribe
de Colombia sobre un fondo cartográfico real (OpenStreetMap).

Produce dos salidas:
  - reports/mapa_estaciones/mapa_estaciones_caribe.html  (interactivo, Folium)
  - reports/mapa_estaciones/mapa_estaciones_caribe.png   (estático, matplotlib + contextily)
"""

import logging
import warnings

import folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.lines import Line2D

from config.settings import ARCHIVO_PARQUET, RUTA_REPORTES, ESTACIONES_EXCLUIDAS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Carpeta de salida ────────────────────────────────────────────────
CARPETA_SALIDA = RUTA_REPORTES / "mapa_estaciones"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)


def cargar_estaciones(ruta: Path) -> pd.DataFrame:
    """
    Carga y retorna las coordenadas únicas de cada estación.

    :param ruta: ruta al archivo Parquet.
    :return: DataFrame con columnas codigo_estacion, latitud, longitud.
    """
    df = pd.read_parquet(ruta, engine="fastparquet")
    estaciones = (
        df[["codigo_estacion", "latitud", "longitud"]]
        .drop_duplicates(subset="codigo_estacion")
        .sort_values("codigo_estacion")
        .reset_index(drop=True)
    )
    estaciones["excluida"] = estaciones["codigo_estacion"].isin(ESTACIONES_EXCLUIDAS)
    logger.info(f"Estaciones totales: {len(estaciones)}")
    logger.info(f"Estaciones excluidas del entrenamiento: {estaciones['excluida'].sum()}")
    return estaciones


def generar_mapa_interactivo(estaciones: pd.DataFrame, ruta_salida: Path) -> None:
    """
    Genera un mapa HTML interactivo con Folium sobre OpenStreetMap.

    :param estaciones: DataFrame con codigo_estacion, latitud, longitud, excluida.
    :param ruta_salida: ruta de salida del archivo HTML.
    """
    centro_lat = estaciones["latitud"].mean()
    centro_lon = estaciones["longitud"].mean()

    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=7,
        tiles="OpenStreetMap",
        width="100%",
        height="100%",
    )

    # Capa adicional: Esri Satellite
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satélite (Esri)",
        overlay=False,
        control=True,
    ).add_to(mapa)

    # Agregar marcadores
    for _, fila in estaciones.iterrows():
        color = "red" if fila["excluida"] else "blue"
        icono = "times" if fila["excluida"] else "cloud"
        tooltip = (
            f"<b>{fila['codigo_estacion']}</b><br>"
            f"Lat: {fila['latitud']:.3f}°<br>"
            f"Lon: {fila['longitud']:.3f}°"
        )
        if fila["excluida"]:
            tooltip += "<br><i>Excluida del entrenamiento</i>"

        folium.Marker(
            location=[fila["latitud"], fila["longitud"]],
            tooltip=folium.Tooltip(tooltip, sticky=True),
            popup=folium.Popup(tooltip, max_width=250),
            icon=folium.Icon(color=color, icon=icono, prefix="fa"),
        ).add_to(mapa)

    # Leyenda manual como HTML
    leyenda_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:white; padding:10px 14px; border:2px solid #555;
                border-radius:8px; font-size:13px; font-family:sans-serif;">
      <b>Estaciones meteorológicas</b><br>
      <span style="color:blue;">&#9679;</span> En entrenamiento<br>
      <span style="color:red;">&#9679;</span> Excluida del entrenamiento
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(leyenda_html))

    folium.LayerControl().add_to(mapa)
    mapa.save(str(ruta_salida))
    logger.info(f"Mapa interactivo guardado: {ruta_salida}")


def generar_mapa_estatico(estaciones: pd.DataFrame, ruta_salida: Path) -> None:
    """
    Genera una imagen PNG estática con matplotlib + contextily (OpenStreetMap).

    :param estaciones: DataFrame con codigo_estacion, latitud, longitud, excluida.
    :param ruta_salida: ruta de salida del archivo PNG.
    """
    # Convertir coordenadas a Web Mercator (EPSG:3857)
    lon = estaciones["longitud"].values
    lat = estaciones["latitud"].values

    # Proyección manual lon/lat → Web Mercator
    R = 6_378_137.0
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R

    # Márgenes
    margen_x = (x.max() - x.min()) * 0.12
    margen_y = (y.max() - y.min()) * 0.15

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(x.min() - margen_x, x.max() + margen_x)
    ax.set_ylim(y.min() - margen_y, y.max() + margen_y)

    # Fondo cartográfico real
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom="auto",
        )
    except Exception as e:
        logger.warning(f"No se pudo cargar el mapa base: {e}")
        ax.set_facecolor("#d0e8f0")

    # Graficar estaciones
    activas = estaciones[~estaciones["excluida"]]
    excluidas = estaciones[estaciones["excluida"]]

    x_act = np.radians(activas["longitud"].values) * R
    y_act = np.log(np.tan(np.pi / 4 + np.radians(activas["latitud"].values) / 2)) * R

    x_exc = np.radians(excluidas["longitud"].values) * R
    y_exc = np.log(np.tan(np.pi / 4 + np.radians(excluidas["latitud"].values) / 2)) * R

    ax.scatter(x_act, y_act, s=120, c="#1565C0", edgecolors="white",
               linewidths=1.5, zorder=5, label="En entrenamiento")
    ax.scatter(x_exc, y_exc, s=120, c="#C62828", marker="X", edgecolors="white",
               linewidths=1.5, zorder=5, label="Excluida del entrenamiento")

    # Etiquetas de código
    for _, fila in estaciones.iterrows():
        xi = np.radians(fila["longitud"]) * R
        yi = np.log(np.tan(np.pi / 4 + np.radians(fila["latitud"]) / 2)) * R
        ax.annotate(
            fila["codigo_estacion"],
            xy=(xi, yi),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            color="black",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"),
        )

    # Leyenda y títulos
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_title(
        "Estaciones meteorológicas — Caribe de Colombia\n"
        "Diciembre 2025 · Datos Abiertos Colombia",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Longitud", fontsize=10)
    ax.set_ylabel("Latitud", fontsize=10)

    # Convertir ticks de Web Mercator a grados para los ejes
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticklabels([f"{np.degrees(t / R):.2f}°" for t in xticks], fontsize=8)
    ax.set_yticklabels(
        [f"{np.degrees(2 * np.arctan(np.exp(t / R)) - np.pi / 2):.2f}°" for t in yticks],
        fontsize=8,
    )

    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Mapa estático guardado: {ruta_salida}")


def main():
    """Punto de entrada principal."""
    logger.info("Cargando coordenadas de estaciones...")
    estaciones = cargar_estaciones(ARCHIVO_PARQUET)

    logger.info("Generando mapa interactivo (HTML)...")
    generar_mapa_interactivo(
        estaciones,
        CARPETA_SALIDA / "mapa_estaciones_caribe.html",
    )

    logger.info("Generando mapa estático (PNG)...")
    generar_mapa_estatico(
        estaciones,
        CARPETA_SALIDA / "mapa_estaciones_caribe.png",
    )

    logger.info("Listo. Archivos en: reports/mapa_estaciones/")


if __name__ == "__main__":
    main()
