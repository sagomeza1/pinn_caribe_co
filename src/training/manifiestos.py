"""Utilidades para guardar metadatos de corridas experimentales."""

import json

from pathlib import Path

from config.experimentos import ConfiguracionExperimento
from config.settings import RUTA_MODELOS


def cargar_manifiesto_experimento(ruta_manifiesto: Path) -> dict:
    """Carga un manifiesto JSON de experimento.

    :param ruta_manifiesto: ruta del manifiesto a leer.
    :return: diccionario con los metadatos del experimento.
    :raises FileNotFoundError: si el archivo no existe.
    :raises ValueError: si la ruta no apunta a un JSON.
    """
    if ruta_manifiesto.suffix.lower() != ".json":
        raise ValueError("El manifiesto del experimento debe ser un archivo JSON")

    if not ruta_manifiesto.exists():
        raise FileNotFoundError(f"No se encontro el manifiesto: {ruta_manifiesto}")

    with ruta_manifiesto.open("r", encoding="utf-8") as archivo_entrada:
        return json.load(archivo_entrada)


def guardar_manifiesto_experimento(
    configuracion: ConfiguracionExperimento,
    datos: dict,
    malla: dict,
    ruta_salida: Path | None = None,
) -> Path:
    """Guarda un manifiesto JSON con la configuracion y el subset usado.

    :param configuracion: configuracion de la corrida.
    :param datos: metadatos generados por la carga del dataset.
    :param malla: metadatos de la malla de colocacion.
    :param ruta_salida: ruta opcional del manifiesto.
    :return: ruta final del archivo escrito.
    """
    if ruta_salida is None:
        ruta_salida = RUTA_MODELOS / f"manifiesto_{configuracion.nombre_modelo}.json"

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    manifiesto = {
        "nombre_experimento": configuracion.nombre_experimento,
        "nombre_modelo": configuracion.nombre_modelo,
        "nombre_log": configuracion.nombre_log,
        "archivo_parquet": str(configuracion.archivo_parquet),
        "r_malla": configuracion.r_malla,
        "lambda_fisica": configuracion.lambda_fisica,
        "num_epocas": configuracion.num_epocas,
        "lr_inicial": configuracion.lr_inicial,
        "checkpoint_intervalo": configuracion.checkpoint_intervalo,
        "n_dias": configuracion.n_dias,
        "intervalo": configuracion.intervalo,
        "epocas_solo_datos": configuracion.epocas_solo_datos,
        "epocas_rampa": configuracion.epocas_rampa,
        "registros_por_estacion": configuracion.registros_por_estacion,
        "estaciones_excluidas": configuracion.estaciones_excluidas,
        "estaciones_utilizadas": datos.get("estaciones_utilizadas", []),
        "estaciones_descartadas_por_subset": datos.get(
            "estaciones_descartadas_por_subset", []
        ),
        "numero_estaciones": datos.get("numero_estaciones"),
        "total_registros": datos.get("total_registros"),
        "tiempos_seleccionados": datos.get("tiempos_seleccionados", []).tolist(),
        "malla": {
            "dim_N": malla.get("dim_N"),
            "dim_T": malla.get("dim_T"),
            "total_puntos": malla.get("dim_N", 0) * malla.get("dim_T", 0),
        },
    }

    with ruta_salida.open("w", encoding="utf-8") as archivo_salida:
        json.dump(manifiesto, archivo_salida, indent=2)

    return ruta_salida