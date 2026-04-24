"""Consolidacion comparativa del barrido R-lambda a CSV."""

import csv
import json
import logging
import re

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import scipy.io as sio

from config.settings import RUTA_LOGS, RUTA_MODELOS, RUTA_REPORTES
from src.training.barrido import VALORES_LAMBDA_BARRIDO, VALORES_R_BARRIDO

logger = logging.getLogger(__name__)

_PATRON_FECHA_LOG = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


@dataclass(slots=True)
class CorridaBarrido:
    """Representa una corrida candidata del barrido.

    :param ruta_manifiesto: ruta del manifiesto JSON.
    :param manifiesto: contenido cargado del manifiesto.
    :param ruta_historial: ruta del historial MAT si existe.
    :param historial: contenido resumido del historial.
    :param ruta_metricas: ruta del JSON de metricas si existe.
    :param metricas: metricas finales si existen.
    :param ruta_log: ruta del log de entrenamiento si existe.
    :param tiempo_total_segundos: duracion inferida desde el log.
    """

    ruta_manifiesto: Path
    manifiesto: dict
    ruta_historial: Path | None
    historial: dict | None
    ruta_metricas: Path | None
    metricas: dict | None
    ruta_log: Path | None
    tiempo_total_segundos: float | None


def _cargar_json(ruta: Path) -> dict:
    """Carga un archivo JSON.

    :param ruta: ruta del archivo.
    :return: contenido JSON como diccionario.
    """
    with ruta.open("r", encoding="utf-8") as archivo_entrada:
        return json.load(archivo_entrada)


def _cargar_historial(ruta_historial: Path | None) -> dict | None:
    """Extrae el resumen util de un historial MAT.

    :param ruta_historial: ruta del archivo MAT.
    :return: resumen del historial o None.
    """
    if ruta_historial is None or not ruta_historial.exists():
        return None

    datos = sio.loadmat(ruta_historial)

    def obtener_ultimo(nombre: str) -> float | None:
        valor = datos.get(nombre)
        if valor is None:
            return None
        plano = valor.flatten()
        if plano.size == 0:
            return None
        return float(plano[-1])

    return {
        "loss_final": obtener_ultimo("loss"),
        "ns_loss_final": obtener_ultimo("ns_loss"),
        "u_loss_final": obtener_ultimo("u_loss"),
        "v_loss_final": obtener_ultimo("v_loss"),
        "p_loss_final": obtener_ultimo("p_loss"),
        "lambda_efecto_final": obtener_ultimo("lambda_efecto"),
        "lr_final": obtener_ultimo("lr"),
    }


def _extraer_tiempo_total_segundos(ruta_log: Path | None) -> float | None:
    """Infiere la duracion total de una corrida a partir del log.

    :param ruta_log: archivo de log de entrenamiento.
    :return: duracion total en segundos o None.
    """
    if ruta_log is None or not ruta_log.exists():
        return None

    primera_fecha = None
    ultima_fecha = None

    with ruta_log.open("r", encoding="utf-8") as archivo_log:
        for linea in archivo_log:
            coincidencia = _PATRON_FECHA_LOG.match(linea)
            if coincidencia is None:
                continue
            fecha = datetime.strptime(coincidencia.group(1), "%Y-%m-%d %H:%M:%S")
            if primera_fecha is None:
                primera_fecha = fecha
            ultima_fecha = fecha

    if primera_fecha is None or ultima_fecha is None:
        return None

    return float((ultima_fecha - primera_fecha).total_seconds())


def _resolver_ruta_historial(nombre_modelo: str) -> Path | None:
    """Resuelve la ruta del historial asociado a un modelo.

    :param nombre_modelo: nombre base del modelo.
    :return: ruta del historial si existe.
    """
    ruta_historial = RUTA_MODELOS / f"historial_{nombre_modelo}.mat"
    if ruta_historial.exists():
        return ruta_historial
    return None


def _resolver_ruta_metricas(nombre_modelo: str, epoca: int | None) -> Path | None:
    """Resuelve la mejor ruta de metricas para una corrida.

    :param nombre_modelo: nombre base del modelo.
    :param epoca: epoca esperada segun manifiesto.
    :return: ruta del JSON de metricas si existe.
    """
    candidatas = []
    if epoca is not None:
        candidatas.append(
            RUTA_MODELOS / f"metricas_{nombre_modelo}_epoca_{epoca}.json"
        )

    candidatas.extend(sorted(RUTA_MODELOS.glob(f"metricas_{nombre_modelo}_epoca_*.json")))

    existentes = [ruta for ruta in candidatas if ruta.exists()]
    if not existentes:
        return None
    return existentes[-1]


def _resolver_ruta_log(nombre_log: str | None) -> Path | None:
    """Resuelve la ruta del log de entrenamiento.

    :param nombre_log: nombre del log segun manifiesto.
    :return: ruta del log si existe.
    """
    if not nombre_log:
        return None

    ruta_log = RUTA_LOGS / nombre_log
    if ruta_log.exists():
        return ruta_log
    return None


def _puntaje_corrida(corrida: CorridaBarrido) -> tuple:
    """Calcula un puntaje para priorizar la mejor evidencia por combinacion.

    :param corrida: corrida candidata.
    :return: tupla comparable.
    """
    nombre_modelo = str(corrida.manifiesto.get("nombre_modelo", ""))
    return (
        corrida.metricas is not None,
        corrida.historial is not None,
        int(corrida.manifiesto.get("num_epocas", 0)),
        "humo" not in nombre_modelo.lower(),
        nombre_modelo,
    )


def recolectar_corridas_barrido() -> dict[tuple[float, float], CorridaBarrido]:
    """Recolecta la mejor corrida disponible para cada par R-lambda.

    :return: diccionario indexado por (R, lambda).
    """
    mejores_corridas: dict[tuple[float, float], CorridaBarrido] = {}

    for ruta_manifiesto in sorted(RUTA_MODELOS.glob("manifiesto_PINN_caribe_excl4_300xest*.json")):
        manifiesto = _cargar_json(ruta_manifiesto)
        clave = (
            float(manifiesto["r_malla"]),
            float(manifiesto["lambda_fisica"]),
        )
        nombre_modelo = str(manifiesto["nombre_modelo"])
        ruta_historial = _resolver_ruta_historial(nombre_modelo)
        historial = _cargar_historial(ruta_historial)
        ruta_metricas = _resolver_ruta_metricas(
            nombre_modelo=nombre_modelo,
            epoca=manifiesto.get("num_epocas"),
        )
        metricas = _cargar_json(ruta_metricas) if ruta_metricas is not None else None
        ruta_log = _resolver_ruta_log(manifiesto.get("nombre_log"))
        corrida = CorridaBarrido(
            ruta_manifiesto=ruta_manifiesto,
            manifiesto=manifiesto,
            ruta_historial=ruta_historial,
            historial=historial,
            ruta_metricas=ruta_metricas,
            metricas=metricas,
            ruta_log=ruta_log,
            tiempo_total_segundos=_extraer_tiempo_total_segundos(ruta_log),
        )

        corrida_existente = mejores_corridas.get(clave)
        if corrida_existente is None or _puntaje_corrida(corrida) > _puntaje_corrida(corrida_existente):
            mejores_corridas[clave] = corrida

    return mejores_corridas


def _inferir_estado(corrida: CorridaBarrido | None) -> str:
    """Clasifica el estado de una combinacion del barrido.

    :param corrida: corrida encontrada o None.
    :return: estado textual.
    """
    if corrida is None:
        return "pendiente"

    nombre_modelo = str(corrida.manifiesto.get("nombre_modelo", ""))
    if corrida.metricas is not None:
        if "humo" in nombre_modelo.lower():
            return "humo_validado"
        return "evaluado"

    if corrida.historial is not None:
        return "entrenado_sin_metricas"

    return "manifiesto_generado"


def _construir_fila_resumen(
    valor_r: float,
    valor_lambda: float,
    corrida: CorridaBarrido | None,
) -> dict:
    """Construye una fila plana del resumen comparativo.

    :param valor_r: valor R de la combinacion esperada.
    :param valor_lambda: valor lambda de la combinacion esperada.
    :param corrida: mejor corrida encontrada para la combinacion.
    :return: fila lista para exportar.
    """
    if corrida is None:
        return {
            "r_malla": valor_r,
            "lambda_fisica": valor_lambda,
            "estado": "pendiente",
            "nombre_modelo": "",
            "nombre_experimento": "",
            "num_epocas": "",
            "numero_estaciones": "",
            "registros_por_estacion": "",
            "total_registros": "",
            "puntos_colocacion": "",
            "loss_final": "",
            "mse_u": "",
            "mse_v": "",
            "mse_p": "",
            "residual_ns": "",
            "tiempo_total_segundos": "",
            "ruta_artefactos": "",
            "ruta_manifiesto": "",
            "ruta_historial": "",
            "ruta_metricas": "",
            "ruta_log": "",
        }

    manifiesto = corrida.manifiesto
    historial = corrida.historial or {}
    metricas = corrida.metricas or {}
    ruta_artefactos = metricas.get("ruta_reporte")
    if not ruta_artefactos:
        nombre_experimento = manifiesto.get("nombre_experimento")
        if nombre_experimento:
            ruta_reporte = RUTA_REPORTES / str(nombre_experimento)
            ruta_artefactos = str(ruta_reporte) if ruta_reporte.exists() else ""

    return {
        "r_malla": valor_r,
        "lambda_fisica": valor_lambda,
        "estado": _inferir_estado(corrida),
        "nombre_modelo": manifiesto.get("nombre_modelo", ""),
        "nombre_experimento": manifiesto.get("nombre_experimento", ""),
        "num_epocas": manifiesto.get("num_epocas", ""),
        "numero_estaciones": manifiesto.get("numero_estaciones", ""),
        "registros_por_estacion": manifiesto.get("registros_por_estacion", ""),
        "total_registros": manifiesto.get("total_registros", ""),
        "puntos_colocacion": manifiesto.get("malla", {}).get("total_puntos", ""),
        "loss_final": historial.get("loss_final", ""),
        "mse_u": metricas.get("mse_u", ""),
        "mse_v": metricas.get("mse_v", ""),
        "mse_p": metricas.get("mse_p", ""),
        "residual_ns": metricas.get("residual_ns", ""),
        "tiempo_total_segundos": corrida.tiempo_total_segundos or "",
        "ruta_artefactos": ruta_artefactos or "",
        "ruta_manifiesto": str(corrida.ruta_manifiesto),
        "ruta_historial": str(corrida.ruta_historial) if corrida.ruta_historial else "",
        "ruta_metricas": str(corrida.ruta_metricas) if corrida.ruta_metricas else "",
        "ruta_log": str(corrida.ruta_log) if corrida.ruta_log else "",
    }


def construir_resumen_barrido() -> list[dict]:
    """Construye las filas esperadas del resumen del barrido.

    :return: lista de filas listas para exportar.
    """
    corridas = recolectar_corridas_barrido()
    filas = []
    for valor_r in VALORES_R_BARRIDO:
        for valor_lambda in VALORES_LAMBDA_BARRIDO:
            filas.append(
                _construir_fila_resumen(
                    valor_r=valor_r,
                    valor_lambda=valor_lambda,
                    corrida=corridas.get((float(valor_r), float(valor_lambda))),
                )
            )
    return filas


def exportar_resumen_barrido_csv(ruta_salida: Path) -> Path:
    """Exporta a CSV el resumen comparativo del barrido.

    :param ruta_salida: ruta del CSV de salida.
    :return: ruta final escrita.
    """
    filas = construir_resumen_barrido()
    campos = [
        "r_malla",
        "lambda_fisica",
        "estado",
        "nombre_modelo",
        "nombre_experimento",
        "num_epocas",
        "numero_estaciones",
        "registros_por_estacion",
        "total_registros",
        "puntos_colocacion",
        "loss_final",
        "mse_u",
        "mse_v",
        "mse_p",
        "residual_ns",
        "tiempo_total_segundos",
        "ruta_artefactos",
        "ruta_manifiesto",
        "ruta_historial",
        "ruta_metricas",
        "ruta_log",
    ]

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", encoding="utf-8", newline="") as archivo_csv:
        writer = csv.DictWriter(archivo_csv, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    logger.info("Resumen del barrido exportado: %s", ruta_salida)
    return ruta_salida