"""Genera graficas de perdidas para todos_R01_L5 y su comparacion con excl_4est_R04_bg."""

import logging

from pathlib import Path

import matplotlib.pyplot as plt
import scipy.io as sio

from config.settings import RUTA_MODELOS, RUTA_REPORTES


logger = logging.getLogger(__name__)

# ── Identificadores de los modelos involucrados ──────────────────────────────

MODELO_TODOS_R01_L5 = "PINN_caribe_excl4_todos_R01_L5"
MODELO_EXCL4EST_R04_BG = "PINN_caribe_excl_4est_R04_bg"

ETIQUETA_TODOS = "todos_R01_L5 (R=0.1, λ=5, todos registros)"
ETIQUETA_REF = "excl4est_R04_bg (R=0.4, λ=3, todos registros)"

# ── Series de perdida a graficar ─────────────────────────────────────────────

SERIES_PERDIDA = {
    "loss": "Loss total",
    "ns_loss": "Residual Navier-Stokes",
    "u_loss": "Loss u_x",
    "v_loss": "Loss u_y",
    "p_loss": "Loss presión",
}

# ── Carpetas de salida ────────────────────────────────────────────────────────

CARPETA_BASE = "iter_R01_todos_analisis_20260426"
CARPETA_INDIVIDUALES = f"{CARPETA_BASE}/perdidas_todos_R01_L5"
CARPETA_COMPARACION = f"{CARPETA_BASE}/comparacion_todos_R01L5_vs_excl4est_R04L3"


def _cargar_historial(nombre_modelo: str) -> dict[str, list[float]]:
    """Carga el historial MAT de un modelo y devuelve sus series como listas.

    :param nombre_modelo: nombre base del modelo (sin prefijo historial_).
    :return: diccionario con las series por clave.
    :raises FileNotFoundError: si el historial no existe.
    :raises KeyError: si falta alguna clave esperada.
    """
    ruta = RUTA_MODELOS / f"historial_{nombre_modelo}.mat"
    if not ruta.exists():
        raise FileNotFoundError(f"Historial no encontrado: {ruta}")
    datos = sio.loadmat(ruta)
    historial = {}
    for clave in ("epoch", *SERIES_PERDIDA.keys()):
        valor = datos.get(clave)
        if valor is None:
            raise KeyError(f"Clave '{clave}' no encontrada en {ruta.name}")
        historial[clave] = valor.flatten().tolist()
    return historial


# ── Graficas individuales de todos_R01_L5 ────────────────────────────────────

def _graficar_serie_individual(
    historial: dict[str, list[float]],
    clave_serie: str,
    titulo: str,
    ruta_salida: Path,
) -> Path:
    """Genera una grafica de una funcion de perdida para un solo modelo.

    :param historial: historial del modelo.
    :param clave_serie: clave de la serie a graficar.
    :param titulo: titulo legible para el eje y la figura.
    :param ruta_salida: archivo PNG de destino.
    :return: ruta del archivo generado.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(historial["epoch"], historial[clave_serie], linewidth=2, color="steelblue")
    plt.title(f"{titulo} — {ETIQUETA_TODOS}")
    plt.xlabel("Época")
    plt.ylabel(titulo)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=200)
    plt.close()
    return ruta_salida


def _graficar_panel_individual(
    historial: dict[str, list[float]],
    ruta_salida: Path,
) -> Path:
    """Genera un panel resumen con todas las funciones de perdida del modelo.

    :param historial: historial del modelo.
    :param ruta_salida: archivo PNG de destino.
    :return: ruta del archivo generado.
    """
    figura, ejes = plt.subplots(3, 2, figsize=(16, 12))
    ejes_planos = ejes.flatten()

    for indice, (clave, titulo) in enumerate(SERIES_PERDIDA.items()):
        eje = ejes_planos[indice]
        eje.plot(historial["epoch"], historial[clave], linewidth=1.8, color="steelblue")
        eje.set_title(titulo)
        eje.set_xlabel("Época")
        eje.set_ylabel(titulo)
        eje.grid(True, alpha=0.3)

    ejes_planos[-1].axis("off")
    figura.suptitle(f"Funciones de pérdida — {ETIQUETA_TODOS}", fontsize=13)
    figura.tight_layout(rect=(0, 0.02, 1, 0.96))
    figura.savefig(ruta_salida, dpi=200)
    plt.close(figura)
    return ruta_salida


def generar_graficas_individuales(historial: dict[str, list[float]]) -> list[Path]:
    """Genera todas las graficas de perdida para todos_R01_L5.

    :param historial: historial cargado del modelo.
    :return: lista de rutas de archivos generados.
    """
    ruta_salida = RUTA_REPORTES / CARPETA_INDIVIDUALES
    ruta_salida.mkdir(parents=True, exist_ok=True)

    rutas = []
    for clave, titulo in SERIES_PERDIDA.items():
        nombre = f"perdida_{clave}_todos_R01_L5.png"
        rutas.append(
            _graficar_serie_individual(historial, clave, titulo, ruta_salida / nombre)
        )
        logger.info("Grafica individual generada: %s", nombre)

    rutas.append(
        _graficar_panel_individual(historial, ruta_salida / "panel_perdidas_todos_R01_L5.png")
    )
    logger.info("Panel resumen individual generado")
    return rutas


# ── Graficas comparativas ─────────────────────────────────────────────────────

def _graficar_comparacion_serie(
    historial_a: dict[str, list[float]],
    historial_b: dict[str, list[float]],
    clave_serie: str,
    titulo: str,
    ruta_salida: Path,
) -> Path:
    """Genera una grafica comparativa de una serie de perdida entre dos modelos.

    :param historial_a: historial del modelo todos_R01_L5.
    :param historial_b: historial del modelo excl_4est_R04_bg.
    :param clave_serie: clave de la serie a comparar.
    :param titulo: titulo legible de la figura.
    :param ruta_salida: archivo PNG de destino.
    :return: ruta del archivo generado.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        historial_a["epoch"], historial_a[clave_serie],
        linewidth=2, color="steelblue", label=ETIQUETA_TODOS,
    )
    plt.plot(
        historial_b["epoch"], historial_b[clave_serie],
        linewidth=2, color="tomato", linestyle="--", label=ETIQUETA_REF,
    )
    plt.title(f"Comparación — {titulo}")
    plt.xlabel("Época")
    plt.ylabel(titulo)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=200)
    plt.close()
    return ruta_salida


def _graficar_panel_comparacion(
    historial_a: dict[str, list[float]],
    historial_b: dict[str, list[float]],
    ruta_salida: Path,
) -> Path:
    """Genera un panel resumen comparativo de todas las funciones de perdida.

    :param historial_a: historial del modelo todos_R01_L5.
    :param historial_b: historial del modelo excl_4est_R04_bg.
    :param ruta_salida: archivo PNG de destino.
    :return: ruta del archivo generado.
    """
    figura, ejes = plt.subplots(3, 2, figsize=(16, 12))
    ejes_planos = ejes.flatten()

    for indice, (clave, titulo) in enumerate(SERIES_PERDIDA.items()):
        eje = ejes_planos[indice]
        eje.plot(
            historial_a["epoch"], historial_a[clave],
            linewidth=1.8, color="steelblue", label=ETIQUETA_TODOS,
        )
        eje.plot(
            historial_b["epoch"], historial_b[clave],
            linewidth=1.8, color="tomato", linestyle="--", label=ETIQUETA_REF,
        )
        eje.set_title(titulo)
        eje.set_xlabel("Época")
        eje.set_ylabel(titulo)
        eje.grid(True, alpha=0.3)

    ejes_planos[-1].axis("off")
    manejadores, etiquetas = ejes_planos[0].get_legend_handles_labels()
    figura.legend(manejadores, etiquetas, loc="lower center", ncol=2, frameon=False, fontsize=10)
    figura.suptitle(
        "Comparación de funciones de pérdida: todos_R01_L5 vs excl4est_R04_bg",
        fontsize=13,
    )
    figura.tight_layout(rect=(0, 0.06, 1, 0.96))
    figura.savefig(ruta_salida, dpi=200)
    plt.close(figura)
    return ruta_salida


def generar_graficas_comparacion(
    historial_a: dict[str, list[float]],
    historial_b: dict[str, list[float]],
) -> list[Path]:
    """Genera todas las graficas de comparacion entre todos_R01_L5 y excl4est_R04_bg.

    :param historial_a: historial del modelo todos_R01_L5.
    :param historial_b: historial del modelo de referencia.
    :return: lista de rutas de archivos generados.
    """
    ruta_salida = RUTA_REPORTES / CARPETA_COMPARACION
    ruta_salida.mkdir(parents=True, exist_ok=True)

    rutas = []
    for clave, titulo in SERIES_PERDIDA.items():
        nombre = f"comparacion_{clave}.png"
        rutas.append(
            _graficar_comparacion_serie(historial_a, historial_b, clave, titulo, ruta_salida / nombre)
        )
        logger.info("Grafica comparacion generada: %s", nombre)

    rutas.append(
        _graficar_panel_comparacion(
            historial_a, historial_b,
            ruta_salida / "panel_comparacion_perdidas.png",
        )
    )
    logger.info("Panel comparacion generado")
    return rutas


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main() -> None:
    """Carga los historiales y genera todas las graficas."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Cargando historial: %s", MODELO_TODOS_R01_L5)
    historial_todos = _cargar_historial(MODELO_TODOS_R01_L5)

    logger.info("Cargando historial de referencia: %s", MODELO_EXCL4EST_R04_BG)
    historial_ref = _cargar_historial(MODELO_EXCL4EST_R04_BG)

    logger.info("Generando graficas individuales de todos_R01_L5 ...")
    rutas_ind = generar_graficas_individuales(historial_todos)

    logger.info("Generando graficas de comparacion ...")
    rutas_comp = generar_graficas_comparacion(historial_todos, historial_ref)

    logger.info("\nArchivos generados:")
    for ruta in rutas_ind + rutas_comp:
        logger.info("  %s", ruta)


if __name__ == "__main__":
    main()
