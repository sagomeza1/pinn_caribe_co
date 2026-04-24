# Checklist operativo del barrido R-lambda

**Fecha:** 19 de abril de 2026

## Preparacion

- [x] Crear la rama `experimento/barrido-r-lambda-300xestacion`.
- [x] Registrar en `solutions/` el inicio del experimento y su alcance.
- [x] Mantener las exclusiones actuales de estaciones.
- [x] Confirmar que el barrido usara `R = {0.1, 0.2, 0.4}`.
- [x] Confirmar que el barrido usara `lambda = {1, 3, 5, 10}`.
- [x] Confirmar que el barrido se fragmenta en 3 iteraciones por `R`.
- [x] Confirmar que `lambda` sera fijo por corrida.
- [x] Confirmar que no se usara curriculum learning en este barrido.

## Subset de datos

- [x] Excluir las estaciones configuradas antes de cualquier recorte.
- [x] Ordenar los registros por `codigo_estacion` y `segundos` ascendente.
- [x] Conservar los primeros 300 registros de cada estacion.
- [x] Verificar que cada estacion retenida tenga al menos 300 registros.
- [x] Verificar que el total final sea `300 x numero_estaciones_utilizadas`.
- [x] Registrar en metadatos el numero final de estaciones y de registros.

## Configuracion experimental

- [x] Crear una configuracion reusable por experimento.
- [x] Parametrizar por corrida: `R`, `lambda`, nombre, registros por estacion, epocas y rutas.
- [x] Evitar depender de un unico valor fijo en `config/settings.py`.
- [x] Guardar un manifiesto por corrida con parametros y criterios del subset.

## Compatibilidad del pipeline

- [x] Reutilizar el pipeline actual de preprocesamiento si el subset conserva forma rectangular.
- [x] Confirmar que cada estacion queda con exactamente 300 observaciones.
- [x] Reutilizar la normalizacion actual del proyecto.
- [x] Mantener intacto el flujo base de entrenamiento fuera de este barrido.

## Malla y entrenamiento

- [x] Construir la malla solo con los tiempos presentes en el subset recortado.
- [x] Hacer que `construir_malla()` reciba `R` por experimento.
- [x] Hacer que `entrenar()` reciba `R` por experimento.
- [x] Desacoplar el calculo de batch size de `R_MALLA` global.
- [ ] Registrar `R` y `lambda` en logs, historial y manifiesto.

## Runner del barrido

- [x] Implementar ejecucion por iteracion fija de `R`.
- [x] Preparar iteracion `R=0.1` con `lambda = {1, 3, 5, 10}`.
- [x] Preparar iteracion `R=0.2` con `lambda = {1, 3, 5, 10}`.
- [x] Preparar iteracion `R=0.4` con `lambda = {1, 3, 5, 10}`.
- [x] Usar nombres estables por corrida.
- [x] Convencion sugerida: `PINN_caribe_excl4_300xest_R01_L1`.
- [x] Separar logs, checkpoints e historiales por prefijo de experimento.

## Evaluacion

- [x] Hacer que la evaluacion reconstruya cada corrida desde su manifiesto.
- [x] Calcular `mse_u`, `mse_v`, `mse_p` y `residual_ns`.
- [ ] Exportar curvas de entrenamiento por experimento.
- [x] Guardar los visuales en una subcarpeta especifica dentro de `reports/`.

## Validacion previa al barrido completo

- [x] Ejecutar una corrida corta de humo con una combinacion representativa.
- [x] Confirmar que el subset tenga exactamente 300 registros por estacion.
- [x] Confirmar que la malla use solo los tiempos del subset.
- [x] Confirmar que `R` modifica correctamente la malla.
- [x] Confirmar que la evaluacion reproduce la misma corrida.

## Cierre del barrido

- [ ] Lanzar las 3 iteraciones completas del barrido.
- [x] Consolidar una tabla comparativa de 12 filas.
- [x] Incluir `R`, `lambda`, estaciones usadas, total de registros, loss final y metricas finales.
- [x] Guardar un resumen en CSV o JSON junto al informe.
- [ ] Redactar el informe final del barrido en `solutions/`.