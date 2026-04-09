# Copilot Instructions — PINN Predicción Meteorológica Caribe Colombia

## Proyecto

Red neuronal informada físicamente (PINN) que predice condiciones meteorológicas sobre
una malla espacial en el Caribe de Colombia, a partir de registros de estaciones
meteorológicas y restricciones de las ecuaciones de Navier-Stokes.
Trabajo de grado — Maestría en Analítica de Datos, Universidad Central.

**Stack:** Python 3.x · PyTorch · pip (`requirements.txt`) · Datos Abiertos Colombia

## Identidad irrenunciable del proyecto

Estas reglas no pueden modificarse sin autorización explícita del responsable.
Si cualquiera se elimina, el proyecto deja de ser el mismo.

| Elemento | Definición oficial |
|----------|-------------------|
| **Entradas** | Posición X · Posición Y · Registro temporal |
| **Salidas** | Velocidad viento en X · Velocidad viento en Y · Presión |
| **Marco físico** | Ecuaciones de Navier-Stokes (obligatorio) |
| **Fuente de datos** | Datos Abiertos Colombia — trazabilidad no puede alterarse |
| **Dominio geográfico** | Caribe de Colombia |
| **Horizonte temporal** | Diciembre de 2025 |
| **Transformación wind** | Dirección + magnitud → componentes X/Y (obligatoria) |

## Anti-patterns — rechazar siempre

| ❌ No hacer | ✅ Hacer en su lugar |
|-------------|---------------------|
| Añadir entradas distintas a X, Y, t | Solo usar posición X, Y y registro temporal como entradas |
| Añadir salidas distintas a u_x, u_y, presión | No predecir temperatura, humedad u otras variables |
| Usar dirección/magnitud de viento como salida directa | Descomponer siempre en componentes X/Y |
| Eliminar automáticamente datos faltantes | Resolver su tratamiento según la variable objetivo involucrada |
| Completar datos ambiguos por inferencia implícita | Dejar explícito el dato como no interpretable |
| Dar prioridad fija a observaciones sobre física (o viceversa) | Mantener ponderadas ambas fuentes en el criterio de ajuste |
| Redefinir la fuente de datos o su origen | Conservar trazabilidad hacia Datos Abiertos Colombia |
| Omitir la pérdida de física (physics loss) | Usar observaciones + restricciones Navier-Stokes conjuntamente |
| Usar la malla sin relación con posiciones de estaciones | La malla se deriva de la posición de las estaciones meteorológicas |

## Estructura del proyecto

```
pinn/
  data/
    raw/           ← datos originales sin modificar (Datos Abiertos Colombia)
    processed/     ← datos transformados, componentes X/Y calculadas
  src/
    data/          ← carga, preprocesamiento, descomposición viento → X/Y
    model/         ← arquitectura PINN, physics loss (Navier-Stokes)
    training/      ← bucle de entrenamiento, optimizadores, callbacks
    evaluation/    ← métricas, visualizaciones sobre la malla
  notebooks/       ← exploración y prototipado (no producción)
  docs/            ← definición y reglas canónicas del proyecto
  requirements.txt
```

## Convenciones de código

- **Idioma:** Comentarios, docstrings y mensajes de log siempre en **español**.
- **Docstrings:** Documentar toda función pública (`:param`, `:return:`, `:raises:`).
- **Constantes:** Centralizar en un módulo de configuración; nunca hardcodear valores de dominio.
- **Datos de aprendizaje:** Siempre normalizados y centralizados antes de entrenar.
- **Información espacial/temporal:** No alterar su significado conceptual durante el preprocesamiento.

## Datos faltantes y ambigüedad

- Los datos faltantes no se eliminan automáticamente; su uso depende de la variable objetivo.
- Los datos ambiguos no se completan por inferencia implícita.
- El conflicto entre observaciones y restricciones físicas no tiene prioridad predefinida;
  ambas fuentes permanecen ponderadas en la función de pérdida.

## Consulta pública

El proyecto mantiene compatibilidad con el origen abierto de los datos.
No incluye infraestructura de despliegue operativo ni alertamiento automático.
