"""Genera graficas comparativas de perdidas por variacion de lambda."""

import argparse
import logging

from pathlib import Path

import matplotlib.pyplot as plt
import scipy.io as sio

from config.settings import RUTA_MODELOS, RUTA_REPORTES


logger = logging.getLogger(__name__)


CORRIDAS_POR_R = {
    0.1: {
        1.0: "PINN_caribe_excl4_300xest_R01_L1",
        3.0: "PINN_caribe_excl4_300xest_R01_L3",
        5.0: "PINN_caribe_excl4_300xest_R01_L5",
        10.0: "PINN_caribe_excl4_300xest_R01_L10",
    },
    0.4: {
        1.0: "PINN_caribe_excl4_300xest_R04_L1",
        5.0: "PINN_caribe_excl4_300xest_R04_L5",
        10.0: "PINN_caribe_excl4_300xest_R04_L10",
    },
}

SERIES_PERDIDA = {
    "loss": "Loss total",
    "ns_loss": "Residual Navier-Stokes",
    "u_loss": "Loss u_x",
    "v_loss": "Loss u_y",
    "p_loss": "Loss presion",
}


def _cargar_historial(ruta_historial: Path) -> dict[str, list[float]]:
    """Carga un historial MAT y devuelve sus series en listas planas.

    :param ruta_historial: ruta del historial del entrenamiento.
    :return: diccionario con las series planas por clave.
    :raises FileNotFoundError: si el historial no existe.
    """
    if not ruta_historial.exists():
        raise FileNotFoundError(f"No se encontro el historial: {ruta_historial}")

    datos = sio.loadmat(ruta_historial)
    historial = {}
    for clave in ("epoch", *SERIES_PERDIDA.keys(), "lambda_efecto"):
        valor = datos.get(clave)
        if valor is None:
            raise KeyError(f"La clave '{clave}' no existe en {ruta_historial.name}")
        historial[clave] = valor.flatten().tolist()
    return historial


def _cargar_historiales_corridas(corridas: dict[float, str]) -> dict[float, dict[str, list[float]]]:
    """Carga los historiales de todas las corridas a comparar.

    :param corridas: mapa lambda -> nombre de modelo.
    :return: mapa lambda -> historial cargado.
    """
    historiales = {}
    for valor_lambda, nombre_modelo in corridas.items():
        ruta_historial = RUTA_MODELOS / f"historial_{nombre_modelo}.mat"
        historiales[valor_lambda] = _cargar_historial(ruta_historial)
    return historiales


def _crear_figura_serie(
    historiales: dict[float, dict[str, list[float]]],
    clave_serie: str,
    titulo: str,
    valor_r: float,
    ruta_salida: Path,
) -> Path:
    """Genera una grafica comparativa para una serie de perdida.

    :param historiales: historiales indexados por lambda.
    :param clave_serie: clave de la serie a graficar.
    :param titulo: titulo legible de la figura.
    :param ruta_salida: archivo PNG destino.
    :return: ruta de la figura escrita.
    """
    plt.figure(figsize=(12, 7))
    for valor_lambda in sorted(historiales):
        historial = historiales[valor_lambda]
        plt.plot(
            historial["epoch"],
            historial[clave_serie],
            label=f"lambda={valor_lambda:g}",
            linewidth=2,
        )

    plt.title(f"{titulo} por epoca para R={valor_r:g}")
    plt.xlabel("Epoca")
    plt.ylabel(titulo)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=200)
    plt.close()
    return ruta_salida


def _crear_panel_resumen(
    historiales: dict[float, dict[str, list[float]]],
    valor_r: float,
    ruta_salida: Path,
) -> Path:
    """Genera un panel con todas las funciones de perdida comparadas.

    :param historiales: historiales indexados por lambda.
    :param ruta_salida: archivo PNG destino.
    :return: ruta de la figura escrita.
    """
    figura, ejes = plt.subplots(3, 2, figsize=(16, 12))
    ejes_planos = ejes.flatten()

    for indice, (clave_serie, titulo) in enumerate(SERIES_PERDIDA.items()):
        eje = ejes_planos[indice]
        for valor_lambda in sorted(historiales):
            historial = historiales[valor_lambda]
            eje.plot(
                historial["epoch"],
                historial[clave_serie],
                label=f"lambda={valor_lambda:g}",
                linewidth=1.8,
            )
        eje.set_title(titulo)
        eje.set_xlabel("Epoca")
        eje.set_ylabel(titulo)
        eje.grid(True, alpha=0.3)

    ejes_planos[-1].axis("off")
    manejadores, etiquetas = ejes_planos[0].get_legend_handles_labels()
    figura.legend(manejadores, etiquetas, loc="lower center", ncol=4, frameon=False)
    figura.suptitle(
        f"Comparacion de funciones de perdida por lambda para R={valor_r:g}"
    )
    figura.tight_layout(rect=(0, 0.04, 1, 0.96))
    figura.savefig(ruta_salida, dpi=200)
    plt.close(figura)
    return ruta_salida


def generar_graficas_perdidas_lambda(
    corridas: dict[float, str] | None = None,
    valor_r: float = 0.1,
    carpeta_salida: str | None = None,
) -> list[Path]:
    """Genera las graficas comparativas de perdidas por lambda.

    :param corridas: mapa lambda -> nombre de modelo. Si es None, usa R01.
    :param valor_r: valor de R asociado a las corridas comparadas.
    :param carpeta_salida: subcarpeta dentro de reports/ donde se guardan las figuras.
    :return: lista de archivos generados.
    :raises ValueError: si no hay una configuracion por defecto para el valor de R.
    """
    if corridas is None:
        corridas_efectivas = CORRIDAS_POR_R.get(valor_r)
        if corridas_efectivas is None:
            raise ValueError(f"No hay corridas configuradas para R={valor_r:g}")
    else:
        corridas_efectivas = corridas

    carpeta_salida_efectiva = (
        carpeta_salida
        if carpeta_salida is not None
        else f"iter_R{int(round(valor_r * 10)):02d}_comparacion_lambdas"
    )
    ruta_reporte = RUTA_REPORTES / carpeta_salida_efectiva
    ruta_reporte.mkdir(parents=True, exist_ok=True)

    historiales = _cargar_historiales_corridas(corridas_efectivas)
    rutas_generadas = []

    for clave_serie, titulo in SERIES_PERDIDA.items():
        nombre_archivo = f"comparacion_{clave_serie}_por_lambda.png"
        rutas_generadas.append(
            _crear_figura_serie(
                historiales=historiales,
                clave_serie=clave_serie,
                titulo=titulo,
                valor_r=valor_r,
                ruta_salida=ruta_reporte / nombre_archivo,
            )
        )

    rutas_generadas.append(
        _crear_panel_resumen(
            historiales=historiales,
            valor_r=valor_r,
            ruta_salida=ruta_reporte / "panel_resumen_perdidas_por_lambda.png",
        )
    )

    logger.info("Graficas generadas en: %s", ruta_reporte)
    return rutas_generadas


def main() -> None:
    """Punto de entrada CLI para generar las graficas comparativas."""
    parser = argparse.ArgumentParser(
        description="Genera graficas comparativas de funciones de perdida por lambda.",
    )
    parser.add_argument(
        "--r-malla",
        type=float,
        default=0.1,
        help="Valor de R asociado al bloque de corridas a comparar.",
    )
    parser.add_argument(
        "--carpeta-reportes",
        default=None,
        help="Subcarpeta destino dentro de reports/. Si se omite, se deriva de R.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rutas = generar_graficas_perdidas_lambda(
        valor_r=args.r_malla,
        carpeta_salida=args.carpeta_reportes,
    )
    for ruta in rutas:
        logger.info("- %s", ruta.name)


if __name__ == "__main__":
    main()