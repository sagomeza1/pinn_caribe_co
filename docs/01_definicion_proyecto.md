# Definicion del Proyecto

## 1. Nombre del proyecto
- **Nombre oficial o tentativo:** Prediccion de condiciones meteorologicas usando PINNs.
- **Acronimos relevantes:** PINNs (redes neuronales informadas fisicamente).
- **Contexto institucional:** Trabajo de grado de la Maestria en Analitica de Datos de la Universidad Central.

## 2. Contexto y motivacion
- El proyecto surge de la necesidad de estimar condiciones meteorologicas sobre una malla espacial en la region Caribe de Colombia, a partir de registros observados en estaciones meteorologicas y de restricciones fisicas derivadas de las ecuaciones de Navier-Stokes.
- El resultado es pertinente para fines academicos en el marco del trabajo de grado y, adicionalmente, se concibe como un proyecto de consulta publica.
- Sin este proyecto, la disponibilidad de estimaciones espaciales de viento y presion queda limitada a observaciones puntuales de estaciones, sin una representacion integrada sobre la malla definida para el estudio.

## 3. Objetivo general del proyecto
- Desarrollar un modelo para predecir condiciones meteorologicas sobre una malla definida a partir de la posicion de estaciones meteorologicas en el Caribe de Colombia, utilizando registros observados y restricciones fisicas.
- El resultado esperado es la estimacion de velocidad del viento en X, velocidad del viento en Y y presion a partir de posicion y tiempo.

## 4. Alcance del proyecto

### 4.1 Que incluye el proyecto
- El uso de registros de estaciones meteorologicas con informacion de:
  - Direccion del viento
  - Velocidad del viento
  - Presion
  - Tiempo
  - Posicion geografica (longitud y latitud)
- La consideracion del dominio geografico correspondiente al Caribe de Colombia.
- El uso del periodo de datos correspondiente a diciembre de 2025.
- La definicion de una malla de prediccion basada en la posicion de las estaciones meteorologicas consideradas.
- La generacion de predicciones de:
  - Velocidad del viento en X
  - Velocidad del viento en Y
  - Presion
- La incorporacion de restricciones fisicas derivadas de las ecuaciones de Navier-Stokes como condicionante del problema.

### 4.2 Que NO incluye el proyecto
- La operacion, mantenimiento o administracion de estaciones meteorologicas.
- La captura de datos en campo.
- La prediccion de variables meteorologicas no confirmadas expresamente, como temperatura, humedad o precipitacion.
- La definicion de infraestructura de despliegue, servicios operativos o integraciones institucionales.
- La generacion de productos de alertamiento, automatizacion de decisiones o analitica sectorial adicional no confirmada.
- La formalizacion de un sistema institucional de produccion mas alla del resultado academico y de consulta publica descrito.

## 5. Insumos principales
- **Fuente principal:** Registros de estaciones meteorologicas disponibles en Datos Abiertos Colombia.
- **Origen institucional o logico:** Portal de datos abiertos del Estado colombiano: https://www.datos.gov.co/
- **Tipos de datos incluidos:**
  - Direccion del viento
  - Velocidad del viento
  - Presion
  - Tiempo
  - Posicion geografica de las estaciones
- **Rasgos relevantes:**
  - Datos observacionales de naturaleza meteorologica
  - Datos con componente espacial y temporal
  - Datos de acceso abierto
  - Datos correspondientes al Caribe de Colombia y al periodo de diciembre de 2025

## 6. Productos y resultados esperados
- Un resultado funcional de prediccion meteorologica sobre una malla previamente definida.
- Salidas principales:
  - Velocidad del viento en X
  - Velocidad del viento en Y
  - Presion
- Una representacion espacial de dichas variables sobre la malla establecida a partir de la ubicacion de las estaciones meteorologicas.
- Un resultado apto para consulta publica y util como producto academico de trabajo de grado.
- Un nivel de transformacion suficiente para pasar de observaciones puntuales a estimaciones espacialmente organizadas y fisicamente consistentes dentro del alcance definido.

## 7. Restricciones y supuestos explicitos
- **Restricciones:**
  - El proyecto se basa en datos obtenidos de Datos Abiertos Colombia.
  - El dominio geografico del estudio corresponde al Caribe de Colombia.
  - El horizonte temporal considerado corresponde a diciembre de 2025.
  - Las variables objetivo del proyecto son exclusivamente velocidad del viento en X, velocidad del viento en Y y presion.
- **Supuestos explicitos:**
  - La malla de prediccion se define a partir de la posicion de las estaciones meteorologicas disponibles.
  - Los registros abiertos disponibles contienen informacion suficiente para formular el problema dentro del alcance definido.
  - Las ecuaciones de Navier-Stokes constituyen el marco fisico de referencia para restringir el comportamiento esperado de las variables.

## 8. Dependencias y antecedentes relevantes
- **Antecedentes conceptuales principales:**
  - Registros de estaciones meteorologicas
  - Ecuaciones de Navier-Stokes
- **Antecedentes previos declarados:**
  - Se toma como referencia un proyecto previo no culminado.
- **Estandares, diccionarios o definiciones reutilizadas:**
  - No se ha confirmado un estandar o diccionario formal especifico.
- **Conocimiento previo asumido:**
  - Comprension basica de variables meteorologicas observadas en estaciones.
  - Comprension general del dominio geografico de estudio.

## 9. Glosario
- **PINNs:** Redes neuronales informadas fisicamente que incorporan restricciones derivadas de leyes fisicas junto con datos observados.
- **Estacion meteorologica:** Punto de observacion que registra variables atmosfericas en una ubicacion y tiempo determinados.
- **Malla:** Conjunto de puntos espaciales sobre los cuales se representan o predicen variables meteorologicas.
- **Caribe de Colombia:** Ambito geografico definido para el proyecto.
- **Velocidad del viento en X:** Componente horizontal del viento asociada al eje X de la representacion espacial adoptada.
- **Velocidad del viento en Y:** Componente horizontal del viento asociada al eje Y de la representacion espacial adoptada.
- **Presion:** Variable atmosferica incluida como salida del proyecto.
- **Navier-Stokes:** Ecuaciones fisicas usadas como referencia para restringir el comportamiento del flujo.
- **Datos Abiertos Colombia:** Portal oficial de acceso abierto desde el cual se obtienen los datos de entrada del proyecto.