"""Script alternativo para generar GIFs de evaluacion a 10 fps."""

import argparse
import logging

from pathlib import Path

import torch

from config.logging_config import configurar_logging
from config.settings import (
    ARCHIVO_PARQUET,
    ESTACIONES_EXCLUIDAS,
    HIDDEN_NEURONS,
    INTERVALO,
    N_DIAS,
    R_MALLA,
    RUTA_MODELOS,
    RUTA_REPORTES,
)
from src.data.cargar_datos import cargar_parquet
from src.data.preprocesar_datos import preprocesar
from src.data.normalizar_datos import normalizar
from src.data.construir_malla import construir_malla
from src.evaluation.generar_gifs import generar_todos_los_gifs
from src.model.pinn import PINN
from src.training.manifiestos import cargar_manifiesto_experimento


logger = logging.getLogger(__name__)
FPS_GIFS = 10


def _resolver_ruta_reporte(
    carpeta_reportes: str | None,
    nombre_experimento: str | None,
) -> Path:
    """Resuelve la carpeta destino para artefactos de GIF.

    :param carpeta_reportes: subcarpeta explicita dentro de reports/.
    :param nombre_experimento: nombre del experimento si existe manifiesto.
    :return: ruta destino para guardar GIFs.
    """
    if carpeta_reportes:
        return RUTA_REPORTES / carpeta_reportes

    if nombre_experimento:
        return RUTA_REPORTES / nombre_experimento

    return RUTA_REPORTES / "sin_iteracion"


def _resolver_contexto(
    manifiesto: dict | None,
    prefijo: str,
    epoca: int | None,
) -> dict:
    """Construye el contexto efectivo para cargar datos y checkpoint.

    :param manifiesto: manifiesto del experimento o None.
    :param prefijo: prefijo de checkpoint en modo legado.
    :param epoca: epoca objetivo o None.
    :return: parametros efectivos de ejecucion.
    :raises ValueError: si no se logra determinar una epoca valida.
    """
    if manifiesto is None:
        epoca_efectiva = 2000 if epoca is None else epoca
        return {
            "ruta_parquet": ARCHIVO_PARQUET,
            "r_malla": R_MALLA,
            "n_dias": N_DIAS,
            "intervalo": INTERVALO,
            "estaciones_excluidas": ESTACIONES_EXCLUIDAS,
            "registros_por_estacion": None,
            "nombre_modelo": prefijo,
            "nombre_experimento": None,
            "epoca": epoca_efectiva,
        }

    epoca_efectiva = manifiesto.get("num_epocas") if epoca is None else epoca
    if epoca_efectiva is None:
        raise ValueError("No fue posible determinar la epoca del checkpoint")

    return {
        "ruta_parquet": Path(manifiesto["archivo_parquet"]),
        "r_malla": float(manifiesto["r_malla"]),
        "n_dias": int(manifiesto["n_dias"]),
        "intervalo": int(manifiesto["intervalo"]),
        "estaciones_excluidas": list(manifiesto.get("estaciones_excluidas", [])),
        "registros_por_estacion": manifiesto.get("registros_por_estacion"),
        "nombre_modelo": str(manifiesto["nombre_modelo"]),
        "nombre_experimento": str(manifiesto.get("nombre_experimento", "")),
        "epoca": int(epoca_efectiva),
    }


def main(
    epoca: int | None = None,
    prefijo: str = "PINN_caribe_v2",
    ruta_manifiesto: Path | None = None,
    carpeta_reportes: str | None = None,
) -> None:
    """Genera GIFs de u, v y p a 10 fps para un checkpoint entrenado.

    :param epoca: epoca del checkpoint; con manifiesto usa num_epocas por defecto.
    :param prefijo: prefijo del checkpoint en modo legado.
    :param ruta_manifiesto: ruta al manifiesto JSON del experimento.
    :param carpeta_reportes: subcarpeta destino dentro de reports/.
    :return: None.
    :raises FileNotFoundError: si el checkpoint no existe.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    manifiesto = None
    if ruta_manifiesto is not None:
        manifiesto = cargar_manifiesto_experimento(ruta_manifiesto)

    contexto = _resolver_contexto(manifiesto, prefijo, epoca)
    ruta_reporte = _resolver_ruta_reporte(
        carpeta_reportes=carpeta_reportes,
        nombre_experimento=contexto["nombre_experimento"],
    )
    ruta_reporte.mkdir(parents=True, exist_ok=True)
    logger.info("Reportes de GIFs: %s", ruta_reporte)

    ruta_checkpoint = RUTA_MODELOS / (
        f"{contexto['nombre_modelo']}_epoca_{contexto['epoca']}.pth"
    )
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ruta_checkpoint}")

    checkpoint = torch.load(ruta_checkpoint, map_location=device, weights_only=False)
    p_scale = checkpoint.get("p_scale", 1.0)
    reescalar = abs(p_scale - 1.0) > 1e-6

    logger.info("Checkpoint: %s", ruta_checkpoint.name)
    logger.info("FPS objetivo para GIFs: %d", FPS_GIFS)

    datos = cargar_parquet(
        contexto["ruta_parquet"],
        estaciones_excluidas=contexto["estaciones_excluidas"],
        registros_por_estacion=contexto["registros_por_estacion"],
    )
    datos_prep = preprocesar(
        datos,
        n_dias=contexto["n_dias"],
        intervalo=contexto["intervalo"],
    )
    datos_norm, escalas = normalizar(datos_prep, reescalar_presion=reescalar)
    malla = construir_malla(datos_norm, escalas, R=contexto["r_malla"])

    modelo = PINN(hidden_neurons=HIDDEN_NEURONS)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(device)
    modelo.eval()

    generar_todos_los_gifs(
        modelo=modelo,
        datos_norm=datos_norm,
        malla=malla,
        escalas=escalas,
        device=device,
        prefijo=f"{contexto['nombre_modelo'].lower()}_epoca_{contexto['epoca']}",
        ruta_salida=ruta_reporte,
        fps=FPS_GIFS,
    )

    logger.info("GIFs generados correctamente a %d fps", FPS_GIFS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generacion alternativa de GIFs a 10 fps"
    )
    parser.add_argument(
        "--epoca",
        type=int,
        default=None,
        help="Epoca del checkpoint; con manifiesto usa num_epocas por defecto",
    )
    parser.add_argument(
        "--prefijo",
        type=str,
        default="PINN_caribe_v2",
        help="Prefijo del checkpoint en modo legado",
    )
    parser.add_argument(
        "--manifiesto",
        type=Path,
        default=None,
        help="Ruta al manifiesto JSON del experimento",
    )
    parser.add_argument(
        "--carpeta-reportes",
        type=str,
        default=None,
        help="Subcarpeta dentro de reports/ para guardar GIFs",
    )
    args = parser.parse_args()

    configurar_logging()
    try:
        main(
            epoca=args.epoca,
            prefijo=args.prefijo,
            ruta_manifiesto=args.manifiesto,
            carpeta_reportes=args.carpeta_reportes,
        )
    except Exception as error:
        logger.error("Error durante la generacion de GIFs: %s", error, exc_info=True)
        raise
