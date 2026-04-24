# Plan de barrido de hiperparametros R y lambda

**Fecha:** 19 de abril de 2026

## Objetivo

Crear una rama dedicada para un estudio controlado de 12 experimentos organizados en 3 iteraciones por `R` con:

- `R` en `0.1`, `0.2`, `0.4`
- `lambda` en `1`, `3`, `5`, `10`

El entrenamiento se realizara usando, para cada estacion incluida, sus primeros 300 registros ordenados por `segundos` ascendente. Si despues de excluir las 4 estaciones actuales quedan 18 estaciones validas, el subset total esperado sera de `5400` registros.

La implementacion recomendada debe introducir configuracion por experimento y metadatos por corrida, pero reutilizando el pipeline rectangular actual porque este criterio de subset preserva una estructura estacion-tiempo consistente.

## Analisis de impacto

- Las entradas del modelo se mantienen exclusivamente en `X`, `Y`, `t`.
- Las salidas del modelo se mantienen exclusivamente en `u_x`, `u_y`, `p`.
- La perdida fisica de Navier-Stokes se mantiene activa en todos los experimentos. El caso `lambda = 0` queda fuera del barrido actual.
- Se conserva la trazabilidad al parquet de Datos Abiertos Colombia.
- El subconjunto queda definido explicitamente como las primeras 300 observaciones por estacion, despues de excluir estaciones y ordenar por `segundos` ascendente dentro de cada estacion.
- El impacto abarca `src/data/`, `src/training/`, `src/evaluation/`, `main.py` y configuracion, pero el riesgo de compatibilidad es menor porque el subset por estacion preserva la forma rectangular ya usada por el pipeline.

## Plan de trabajo

### Fase 1. Aislamiento del experimento

1. Crear una rama nueva sugerida: `experimento/barrido-r-lambda-300xestacion`.
2. Documentar en `solutions/` el objetivo del barrido, la matriz de 12 combinaciones y la division operativa en 3 iteraciones por `R`: `R=0.1`, `R=0.2` y `R=0.4`, manteniendo `lambda` fijo por corrida y el subset de 300 registros por estacion ordenados por tiempo.

### Fase 2. Configuracion reusable y metadatos

1. Introducir una estructura de configuracion de experimento consumible por `main.py` y `evaluar.py` para pasar al menos `R`, `lambda`, nombre de corrida, cantidad de registros por estacion, epocas y rutas de salida sin reescribir constantes globales por cada corrida.
2. Esta capa debe reemplazar el uso directo de `R_MALLA`, `LAMBDA_FISICA`, `NOMBRE_MODELO_ENTRENAMIENTO` y `NOMBRE_LOG_ENTRENAMIENTO` como unico origen para experimentos comparativos.
3. Definir un manifiesto por experimento que se guarde junto al checkpoint e historial con `R`, `lambda`, criterio de subset, estaciones excluidas, estaciones efectivamente usadas, cantidad real de registros por estacion, total de registros, tiempos seleccionados y nombre del experimento.

### Fase 3. Carga del subset por estacion

1. Extender `src/data/cargar_datos.py` para que, tras excluir estaciones, ordene los datos por `codigo_estacion` y `segundos`, y luego conserve los primeros 300 registros de cada estacion.
2. Antes de seguir, validar que cada estacion retenida tenga al menos 300 observaciones.
3. Si alguna estacion no cumple, el flujo debe fallar con un mensaje explicito o registrar de forma trazable la decision adoptada.

### Fase 4. Compatibilidad con el pipeline actual

1. Reutilizar `proyectar_a_cartesiano()`, `reestructurar_por_estacion()`, `filtrar_tiempo_y_ordenar()` y la normalizacion actual siempre que el subset resultante mantenga exactamente 300 observaciones por estacion.
2. La adaptacion necesaria es menor: el punto critico es garantizar que el recorte preserve una malla rectangular coherente antes de entrar a `preprocesar()`.

### Fase 5. Tiempos y malla fisica

1. Construir la malla de colocacion solo con los tiempos presentes en las 300 observaciones por estacion.
2. La parte fisica y la parte observacional deben operar sobre el mismo horizonte temporal reducido.
3. `construir_malla()` debe recibir `R` por experimento y no depender solo del valor global.

### Fase 6. Entrenamiento parametrico

1. Ajustar `entrenar()` para que reciba `R` desde la configuracion de experimento en lugar de depender de `R_MALLA` global para calcular batch sizes y nombres.
2. Mantener `lambda` fijo por corrida y registrar ese valor en logs e historial.
3. No activar curriculum learning en este barrido.

### Fase 7. Runner de barrido por iteraciones

1. Incorporar un runner de experimentos que permita ejecutar el barrido en 3 iteraciones, fijando un valor de `R` por lanzamiento:
   - Iteracion 1: `R = 0.1`, `lambda in {1, 3, 5, 10}`
   - Iteracion 2: `R = 0.2`, `lambda in {1, 3, 5, 10}`
   - Iteracion 3: `R = 0.4`, `lambda in {1, 3, 5, 10}`
2. Cada iteracion debe ejecutar 4 corridas secuenciales con logs y checkpoints aislados.
3. El total del estudio queda en 12 combinaciones.
4. Convencion sugerida de nombres:
   - `PINN_caribe_excl4_300xest_R01_L1`
   - `PINN_caribe_excl4_300xest_R02_L3`
   - `PINN_caribe_excl4_300xest_R04_L10`

### Fase 8. Politica de artefactos

1. Guardar logs en `logs/` por corrida.
2. Guardar checkpoints e historiales en `models/` por prefijo.
3. Guardar visuales comparativos en una carpeta de iteracion especifica dentro de `reports/`, nunca en la raiz.
4. En `solutions/`, preparar PLAN, CHECKPOINT e INFORME del barrido para dejar trazabilidad del estudio.

### Fase 9. Evaluacion reproducible

1. Adaptar `evaluar.py` o crear un evaluador del barrido que lea el manifiesto de cada corrida.
2. Reconstruir exactamente el subset de 300 registros por estacion y la malla temporal asociada.
3. Calcular al menos:
   - `mse_u`
   - `mse_v`
   - `mse_p`
   - `residual_ns`
4. Exportar curvas de entrenamiento por experimento.
5. Como el subset sigue siendo rectangular por estacion-tiempo, las metricas y graficas actuales son reutilizables con cambios menores.

### Fase 10. Consolidacion comparativa

1. Generar una tabla resumen de 12 filas con:
   - `R`
   - `lambda`
   - numero de estaciones usadas
   - registros por estacion
   - total de registros
   - cantidad de puntos de colocacion
   - loss final
   - `mse_u`
   - `mse_v`
   - `mse_p`
   - `residual_ns`
   - tiempo total
   - ruta de artefactos
2. Usar ese resumen como base del informe final y de la seleccion de combinaciones prometedoras.

### Fase 11. Ejecucion incremental

1. Antes de lanzar las 3 iteraciones completas, hacer una validacion corta con una combinacion representativa.
2. Confirmar que el subset contiene exactamente 300 registros por estacion.
3. Confirmar que el total coincide con `300 x numero_estaciones`.
4. Confirmar que la malla usa solo esos tiempos.
5. Confirmar que `R` afecta correctamente la malla.
6. Confirmar que la evaluacion reconstruye el mismo experimento.
7. Luego lanzar las 3 iteraciones del barrido por `R`.

## Archivos relevantes

- `config/settings.py`: hoy concentra `R_MALLA`, `LAMBDA_FISICA`, nombres de modelo y log; debe pasar de ser configuracion unica a defaults reutilizables.
- `main.py`: entrypoint actual del entrenamiento; debe aceptar configuracion por corrida y orquestar el flujo experimental.
- `src/training/entrenador.py`: `entrenar()` usa `R_MALLA` para batch size y registra `lambda`; es un punto critico para parametrizar el barrido.
- `src/data/cargar_datos.py`: punto correcto para materializar el recorte de 300 registros por estacion tras exclusiones y orden temporal.
- `src/data/preprocesar_datos.py`: puede reutilizarse casi completo si el subset por estacion preserva 300 tiempos por fila.
- `src/data/normalizar_datos.py`: reutilizable para escalas y adimensionalizacion del subset reducido.
- `src/data/datasets.py`: `EstacionesDataset` y `ColocacionDataset` seguiran siendo el punto natural de entrada al entrenamiento.
- `src/data/construir_malla.py`: debe aceptar tiempos seleccionados del subset y `R` por experimento, no solo defaults globales.
- `evaluar.py`: hoy reconstruye con `R_MALLA` global; necesita leer metadatos del experimento para evaluar corridas heterogeneas.
- `src/evaluation/metricas.py`: probablemente reutilizable con cambios menores porque el subset mantiene forma estacion-tiempo.
- `solutions/PLAN_entrenamiento_excl_4est.md`: referencia del estilo de documentacion dinamica a replicar para este barrido.
- `solutions/INFORME_entrenamiento.md`: referencia para estructura del informe comparativo final.

## Verificacion

1. Verificar en una corrida de humo que el filtro de exclusiones se aplica antes del recorte y que cada estacion retenida conserva exactamente 300 registros ordenados por `segundos` ascendente.
2. Confirmar que el total final del subset coincide con `300 x numero_estaciones_utilizadas` y registrar explicitamente ese numero en el manifiesto del experimento.
3. Confirmar que los tiempos usados por la malla coinciden exactamente con las 300 observaciones temporales retenidas por estacion y que no se cuelan snapshots externos.
4. Ejecutar una corrida corta con un par `(R, lambda)` y revisar en logs y manifiesto que `R`, `lambda`, cantidad de puntos de colocacion y nombres de artefactos son correctos.
5. Evaluar esa misma corrida leyendo solo su manifiesto para comprobar que la reconstruccion del experimento no depende de `config/settings.py` global.
6. Lanzar las 3 iteraciones del barrido, con 4 corridas por iteracion, y validar que cada una genera checkpoint, historial, log y registro en la tabla comparativa sin colisiones de nombres.
7. Revisar que todos los visuales nuevos queden dentro de una subcarpeta dedicada de `reports/` asociada a esta iteracion del barrido.

## Decisiones cerradas

- `lambda` se interpreta como peso fijo de la perdida fisica por corrida.
- El barrido se ejecuta en 3 iteraciones separadas por valor de `R`.
- El valor `lambda = 0` queda excluido del estudio comparativo actual.
- Los 300 registros se seleccionan por estacion usando `segundos` ascendente dentro de cada estacion.
- La malla fisica del experimento usa solo los tiempos presentes en el subset recortado.
- Quedan fuera del alcance cambios de arquitectura, nuevas variables, eliminacion de Navier-Stokes y curriculum learning para este barrido.

## Consideraciones adicionales

1. Si alguna estacion tiene menos de 300 observaciones validas despues de exclusiones o filtrados, debe tratarse como error de preparacion del experimento y registrarse antes de lanzar el barrido completo.
2. Conviene guardar un CSV o JSON resumen del barrido junto al informe para evitar depender de lectura manual de logs.
3. Aunque el subset reduzca el horizonte temporal, debe mantenerse intacto el flujo base completo para no degradar el entrenamiento productivo ya existente.