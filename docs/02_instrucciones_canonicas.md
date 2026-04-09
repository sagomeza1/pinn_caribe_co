## Informacion faltante para cerrar el documento
- No faltan datos criticos para emitir esta version depurada.

# Instrucciones Canonicas del Proyecto

## 1. Proposito de las instrucciones canonicas
- Estas instrucciones aplican a toda persona o agente de IA que participe en la definicion, preparacion de datos, entrenamiento, evaluacion, interpretacion o ajuste del proyecto.
- Regulan decisiones que afectan la identidad, validez y gobernanza del proyecto.
- Buscan evitar cambios no autorizados en el significado de las variables, en el uso de los datos y en las condiciones minimas que definen el proyecto.

## 2. Principios generales de comportamiento
- Toda decision debe preservar el objetivo oficial del proyecto.
- Ante incertidumbre, no se debe asumir informacion no confirmada explicitamente.
- Ningun actor puede redefinir por criterio propio las entradas, las salidas, la malla de referencia o las relaciones obligatorias entre variables.
- Toda excepcion a estas instrucciones debe ser autorizada explicitamente por el responsable del proyecto.

## 3. Reglas canonicas de dominio o negocio
- El proyecto tiene como finalidad predecir condiciones meteorologicas sobre una malla definida a partir de la posicion de las estaciones meteorologicas.
- Las entradas funcionales oficiales del proyecto son exclusivamente:
  - posicion en X,
  - posicion en Y,
  - registro temporal.
- Las salidas funcionales oficiales del proyecto son exclusivamente:
  - velocidad del viento en X,
  - velocidad del viento en Y,
  - presion.
- La direccion del viento y la velocidad del viento observadas en la fuente original deben interpretarse, para efectos del proyecto, mediante su descomposicion en componentes X y Y.
- El proyecto solo conserva su identidad si mantiene el mismo objetivo, las mismas entradas oficiales, las mismas salidas oficiales y la incorporacion conjunta de observaciones y restricciones fisicas.

## 4. Reglas canonicas sobre el uso de los datos
- La fuente primaria del proyecto es un conjunto de datos previamente trabajado con trazabilidad a Datos Abiertos Colombia.
- La procedencia de los datos debe conservarse de forma explicita y no puede redefinirse.
- La relacion entre direccion del viento, velocidad del viento y sus componentes en X y Y es obligatoria.
- Los datos destinados al aprendizaje deben ser normalizados y centralizados.
- La informacion espacial y temporal no debe alterarse conceptualmente.
- Las observaciones de estaciones meteorologicas y las restricciones derivadas de Navier-Stokes deben utilizarse conjuntamente en la formulacion del problema.

## 5. Reglas ante ambiguedad, ausencia o conflicto de informacion
- Cuando existan datos faltantes, estos no deben eliminarse automaticamente del conjunto total; su uso debe resolverse segun el dato faltante y la variable objetivo involucrada.
- Cuando un dato sea ambiguo o no interpretable con certeza, no debe completarse por inferencia implicita.
- Cuando exista conflicto entre observaciones y restricciones fisicas, no debe asumirse prioridad fija de una sobre la otra.
- En caso de conflicto, ambas fuentes deben mantenerse ponderadas dentro del criterio de ajuste del proyecto.
- Nunca debe quedar al criterio implicito de un agente cambiar las entradas oficiales, las salidas oficiales, la descomposicion del viento a componentes X/Y, la normalizacion y centralizacion de los datos, o el tratamiento conceptual de los datos faltantes.

## 6. Restricciones canonicas del proyecto
- El proyecto pertenece al trabajo de grado de la Maestria en Analitica de Datos de la Universidad Central.
- El dominio geografico oficial del proyecto es el Caribe de Colombia.
- El horizonte temporal de referencia confirmado es diciembre de 2025.
- El proyecto debe mantener su condicion de consulta publica en terminos compatibles con el origen abierto de los datos.

## 7. Dependencias conceptuales obligatorias
- Las ecuaciones de Navier-Stokes constituyen el marco fisico obligatorio del proyecto.
- Datos Abiertos Colombia constituye la referencia obligatoria de procedencia de la informacion observacional de base.
- La correspondencia entre estaciones meteorologicas, posicion espacial y malla de prediccion debe respetarse como definicion oficial del problema.

## 8. Invariantes del proyecto
- Debe existir siempre una prediccion de velocidad del viento en X, velocidad del viento en Y y presion a partir de X, Y y t.
- Debe mantenerse la transformacion de direccion y magnitud del viento a componentes X/Y para efectos de entrenamiento.
- Debe mantenerse el uso conjunto de evidencia observacional y restricciones fisicas.
- Debe mantenerse la malla derivada de la posicion de las estaciones meteorologicas.
- Debe mantenerse la trazabilidad de los datos hacia su origen abierto.
- Si cualquiera de estas condiciones se elimina, el proyecto deja de conservar su identidad actual.

## 9. Alcance de aplicacion de estas instrucciones
- Estas instrucciones aplican tanto a personas como a agentes de IA.
- Aplican a todo el ciclo de vida del proyecto en aquello que afecte su identidad, validez o gobernanza.
- No regulan decisiones operativas o de implementacion que no alteren las reglas canonicas aqui definidas.