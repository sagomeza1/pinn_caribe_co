# Estado General del Proyecto — 13 de abril de 2026

## Resumen ejecutivo

El proyecto PINN para prediccion meteorologica del Caribe colombiano tiene las fases de infraestructura, datos, arquitectura, entrenamiento y evaluacion implementadas.

Se completo una primera iteracion de entrenamiento de 2000 epocas en GPU con checkpoints en epocas 500, 1000, 1500 y 2000. La evaluacion del modelo muestra convergencia de la funcion de perdida, pero con un comportamiento cuasi-estacionario en las componentes de velocidad, por lo que se requiere una segunda iteracion de entrenamiento para mejorar la dinamica temporal.

## Estado funcional actual

- Fase 1 (infraestructura): completada.
- Fase 2 (pipeline de datos): completada.
- Fase 3 (arquitectura PINN): completada.
- Fase 4 (entrenamiento base): completada (2000 epocas).
- Fase 5 (evaluacion y visualizacion): completada.
- Fase 6 (iteracion y ajuste): iteracion 1 completada, iteracion 2 pendiente.

## Metricas de referencia (iteracion 1)

- Loss total final: 1.001
- MSE u_x: 0.671820
- MSE u_y: 0.627913
- MSE presion: 1.423412
- Residual Navier-Stokes: 0.020416

## Hallazgos tecnicos principales

1. Se corrigio el reshape de malla en la generacion de GIFs para eliminar artefactos visuales.
2. La presion sigue siendo la variable dominante en la perdida global.
3. El modelo logra bajo residual fisico, pero muestra baja variacion temporal en u_x y u_y.

## Artefactos disponibles

- Checkpoints del modelo: `models/PINN_caribe_epoca_500.pth`, `models/PINN_caribe_epoca_1000.pth`, `models/PINN_caribe_epoca_1500.pth`, `models/PINN_caribe_epoca_2000.pth`.
- Historial de entrenamiento: `models/historial_PINN_caribe.mat`.
- Reportes de evaluacion y visualizacion en `reports/` (GIFs y curvas de perdida).

## Proximo paso recomendado

Ejecutar una iteracion 2 aplicando curriculum learning y/o reescalado de presion para mejorar la dinamica temporal, seguido de evaluacion comparativa frente a la iteracion 1.
