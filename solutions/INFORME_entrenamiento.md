# INFORME DE ENTRENAMIENTO — Iteracion 1 e Iteracion 2

**Fecha:** 10 de abril de 2026
**Sesion:** 3 (evaluacion del entrenamiento ejecutado en sesion 2)

---

## 1. Configuracion del entrenamiento

| Parametro | Valor |
|-----------|-------|
| Epocas | 2000 |
| Arquitectura | 12 capas ocultas x 1200 neuronas (GammaBiasLayer + WN) |
| Activaciones | 8 Tanh + 4 Lineales |
| Parametros totales | 15,890,409 |
| Optimizador | Adam (lr=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.3162, patience=50) |
| Lambda (peso fisico) | 3.0 |
| Gradient clipping | max_norm=0.5 |
| Dispositivo | CUDA (GPU) |
| Duracion | ~12 horas |

---

## 2. Metricas de evaluacion — Epoca 2000

| Metrica | Valor |
|---------|-------|
| MSE u_x | 0.671820 |
| MSE u_y | 0.627913 |
| MSE presion | 1.423412 |
| Residual NS | 0.020416 |
| Loss total | 1.001 |

### Proporcion de componentes en la loss

| Componente | Valor | Proporcion |
|------------|------:|----------:|
| NS loss | 0.651 | 19.3% |
| U loss | 0.669 | 19.8% |
| V loss | 0.629 | 18.7% |
| **P loss** | **1.424** | **42.2%** |

---

## 3. Curva de convergencia

| Epoca | Loss total | NS | U | V | P | LR |
|------:|-----------:|----:|---:|---:|---:|----:|
| 1 | 2.813 | 1.159 | 1.292 | 1.288 | 3.852 | 1.0e-4 |
| 500 | 2.044 | 1.423 | 0.930 | 0.944 | 2.897 | 1.0e-4 |
| 1000 | 1.305 | 0.967 | 0.802 | 0.766 | 1.816 | 3.2e-5 |
| 1500 | 1.082 | 0.717 | 0.721 | 0.667 | 1.534 | 1.0e-5 |
| 2000 | 1.001 | 0.651 | 0.669 | 0.629 | 1.424 | 3.2e-6 |

**Reduccion total:** 64.4% (2.813 → 1.001)
**Mejora ultimas 200 epocas:** 1.63% — practicamente estancado.
**Learning rate final:** 3.2e-6 (3 reducciones aplicadas, margen limitado).

---

## 4. Analisis visual — GIFs de curvas de nivel

### 4.1 Bug corregido: reshape en generar_gifs.py

Durante la primera generacion de GIFs se detecto un artefacto de **franjas diagonales**
en las graficas de contourf. La causa fue un error en el orden de los ejes al hacer
reshape: se usaba `reshape((n_x, n_y), order="F")` cuando lo correcto para la malla
generada con `np.meshgrid` es `reshape((n_y, n_x), order="F")`.

- **Malla:** n_x=8, n_y=40 (dim_N=320)
- **meshgrid produce:** shape (n_y, n_x) = (40, 8)
- **flatten('F') requiere:** reshape de vuelta a (n_y, n_x) para recuperar la malla
- **Fix aplicado:** 3 llamadas a reshape en `generar_gifs.py` corregidas

### 4.2 Observaciones de los GIFs corregidos

**Velocidad u_x (viento en X):**
- El campo es mayoritariamente uniforme (valores cercanos a 0 m/s).
- Variacion espacial debil: gradiente suave desde bordes inferiores (valores ligeramente
  negativos) hacia la esquina superior derecha (valores positivos ~3 m/s).
- **Variacion temporal practicamente nula:** los frames t=0, t=372 y t=743 son
  visualmente indistinguibles. El modelo colapso a una solucion cuasi-estacionaria.

**Velocidad u_y (viento en Y):**
- Campo dominado por tonos uniformes (cercanos a -0.6 a 0 m/s).
- Variacion espacial minima — el campo es casi constante en todo el dominio.
- **Sin variacion temporal detectable.** Peor que u_x en resolucion espacial.

**Presion:**
- Muestra la mayor variacion espacial entre las tres variables.
- Gradiente claro de suroeste (valores negativos, ~-7.5 Pa normalizado) a noreste
  (valores positivos, ~6 Pa normalizado).
- Se observan estructuras espaciales coherentes con gradientes de presion.
- **Ligera variacion temporal visible**, aunque sutil. Es la variable con mayor
  dinamica entre los tres frames inspeccionados.

### 4.3 Diagnostico visual consolidado

| Criterio | u_x | u_y | p |
|----------|-----|-----|---|
| Variacion espacial | Debil | Minima | Significativa |
| Variacion temporal | Nula | Nula | Sutil |
| Ajuste a observaciones | Pobre (MSE=0.67) | Pobre (MSE=0.63) | Pobre (MSE=1.42) |
| Estructura fisica | Poco definida | Campo plano | Gradiente coherente |

---

## 5. Diagnostico general

### 5.1 Solucion cuasi-estacionaria

El modelo ha convergido a una solucion que es esencialmente **independiente del tiempo**.
Las tres variables muestran el mismo patron espacial sin importar el timestep. Esto es un
modo de fallo conocido en PINNs: la red satisface las ecuaciones de Navier-Stokes en
estado estacionario (residual NS bajo = 0.020) a costa de ignorar la dinamica temporal
de las observaciones.

### 5.2 Desbalance de la presion

La presion domina la funcion de perdida (42.2%) porque su rango normalizado es ~10x
mas amplio que el de las velocidades:
- P_norm: [-4.894, 7.999]
- U_norm: [-0.735, 0.613]
- V_norm: [-0.518, 0.678]

Esto dificulta el balance del optimizador.

### 5.3 Residual NS artificialmente bajo

El residual NS (0.020) es significativamente menor que las perdidas de datos (~0.63-1.42).
Esto indica que la red encontro una solucion trivial para la fisica (estado estacionario
satisface N-S con derivadas temporales cercanas a cero) sin ajustarse bien a los datos.

---

## 6. Estrategias propuestas para iteracion 2

Basado en el diagnostico, se proponen las siguientes estrategias (Paso 26 del plan):

| # | Estrategia | Justificacion | Prioridad |
|---|-----------|---------------|-----------|
| 1 | **Curriculum learning** — entrenar primero solo con datos (lambda=0), luego incorporar fisica gradualmente | Fuerza al modelo a aprender la dinamica temporal antes de imponerle la restriccion fisica que facilita el colapso estacionario | Alta |
| 2 | **Reescalar presion** — reducir el rango de P_norm para que sea comparable con U/V | Elimina el desbalance 10x que hace que la presion domine la loss | Alta |
| 3 | **Pesos por variable en la loss** — ponderar u, v, p por separado | Permite dar mas peso a las velocidades (actualmente sub-representadas) | Media |
| 4 | **Aumentar lambda gradualmente** — empezar con lambda=0.1 y subir a 3.0 | Variante de curriculum: la fisica no domina al inicio | Media |
| 5 | **Cosine annealing del learning rate** — reemplazar ReduceLROnPlateau | Evita el estancamiento temprano del LR (solo 3 reducciones en 2000 epocas) | Baja |

---

## 7. Archivos modificados en esta sesion

| Archivo | Cambio |
|---------|--------|
| `src/evaluation/generar_gifs.py` | Fix: reshape (n_x, n_y) → (n_y, n_x) en 3 ubicaciones |
| `evaluar.py` | Nuevo: script de evaluacion para cargar checkpoint y generar GIFs + metricas |

---

## 8. Artefactos generados

| Archivo | Descripcion |
|---------|-------------|
| `reports/pinn_epoca_2000_u.gif` | GIF curvas de nivel u_x (7.9 MB, 744 frames) |
| `reports/pinn_epoca_2000_v.gif` | GIF curvas de nivel u_y (8.4 MB, 744 frames) |
| `reports/pinn_epoca_2000_p.gif` | GIF curvas de nivel presion (13.8 MB, 744 frames) |

---

## 9. Conclusion — Iteracion 1

El entrenamiento de 2000 epocas logro una reduccion del 64.4% en la loss total, pero
el modelo convergio a una **solucion cuasi-estacionaria** que no captura la dinamica
temporal del viento ni la presion. La presion es el cuello de botella (42.2% de la loss),
influenciada por un rango normalizado 10 veces mayor que las velocidades.

Se requiere una **iteracion 2** centrada en curriculum learning y/o reescalado de la
presion para romper el colapso estacionario.

---
---

# ITERACION 2 — Curriculum Learning + Reescalado de Presion

**Fecha:** 13 de abril de 2026
**Sesion:** 4

---

## 10. Estrategias implementadas

### 10.1 Curriculum learning

Se implemento un esquema de tres fases para el peso de la perdida fisica (lambda):

| Fase | Epocas | Lambda | Objetivo |
|------|--------|--------|----------|
| 1 — Solo datos | 0-199 | 0 | Aprender la dinamica temporal sin restriccion fisica |
| 2 — Rampa | 200-499 | 0 → 3.0 (lineal) | Incorporar fisica gradualmente |
| 3 — Completo | 500-1999 | 3.0 | Entrenamiento con restriccion completa |

**Justificacion:** En la Iteracion 1, la red encontro una solucion estacionaria trivial
que minimiza los residuos N-S (derivadas temporales ≈ 0). El curriculum fuerza a la red
a aprender primero la variacion temporal de los datos, y solo despues impone la
consistencia fisica.

### 10.2 Reescalado de presion

La presion normalizada tenia un rango ~10x mayor que las velocidades:
- P_norm: [-4.894, 7.999] (rango ≈ 12.9)
- U_norm: [-0.735, 0.613] (rango ≈ 1.3)
- V_norm: [-0.518, 0.678] (rango ≈ 1.2)

Se aplico un reescalado por la desviacion estandar (sigma_p):
- **sigma_p = 1.9656**
- P_rescaled = P_norm / sigma_p
- Rango P_rescaled: [-2.490, 4.069] (rango ≈ 6.6)

Para mantener la consistencia de las ecuaciones de Navier-Stokes, los terminos
dp/dx y dp/dy se multiplican por sigma_p en la perdida fisica.

---

## 11. Configuracion del entrenamiento — Iteracion 2

| Parametro | Iter 1 | Iter 2 | Cambio |
|-----------|--------|--------|--------|
| Epocas totales | 2000 | 2000 | — |
| Curriculum | No | Si (200/300/1500) | Nuevo |
| Lambda | 3.0 (fijo) | 0 → 3.0 (rampa) | Cambiado |
| Reescalado presion | No | Si (sigma_p=1.9656) | Nuevo |
| Arquitectura | 12x1200 GBL | 12x1200 GBL | — |
| Optimizador | Adam 1e-4 | Adam 1e-4 | — |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau | — |
| Gradient clipping | 0.5 | 0.5 | — |
| Nombre modelo | PINN_caribe | PINN_caribe_v2 | Diferenciado |

---

## 12. Resultados del entrenamiento

### 12.1 Curva de convergencia

| Epoca | Loss | NS | U | V | P | LR | Lambda |
|------:|-----:|---:|--:|--:|--:|----:|-------:|
| 0 | 0.755 | 0.000 | — | — | — | 1.0e-4 | 0.00 |
| 50 | 0.672 | 0.000 | — | — | — | 1.0e-4 | 0.00 |
| 100 | 0.583 | 0.000 | — | — | — | 1.0e-4 | 0.00 |
| 150 | 0.446 | 0.000 | — | — | — | 3.2e-5 | 0.00 |
| 200 | 0.255 | — | — | — | — | 3.2e-5 | 0.00 |
| 250 | 0.588 | — | — | — | — | 1.0e-5 | 0.50 |
| 300 | 0.572 | — | — | — | — | 3.2e-6 | 1.00 |
| 350 | 0.565 | — | — | — | — | 1.0e-6 | 1.50 |
| 400 | 0.565 | — | — | — | — | 3.2e-7 | 2.00 |
| 500 | 0.568 | — | — | — | — | 1.0e-7 | 3.00 |

### 12.2 Analisis por fases

**Fase 1 (solo datos, epocas 0-199):**
- Loss: 0.755 → 0.255 (reduccion del 66.2%)
- Sin computo de NS → ~3.5 s/epoca (vs ~22 s con NS)
- El scheduler redujo LR una vez (1e-4 → 3.2e-5 en ~epoca 140)
- El modelo aprendio la distribucion temporal de los datos antes de imponer fisica

**Fase 2 (rampa lambda, epocas 200-499):**
- Loss: 0.255 → 0.568 (aumento por la incorporacion de fisica)
- Lambda crecio linealmente de 0 a 3.0
- El scheduler redujo LR 5 veces adicionales (3.2e-5 → 1e-7)
- **Problema critico:** ReduceLROnPlateau interpreto la meseta de loss (causada por
  el aumento de lambda, no por estancamiento real) como falta de progreso

**Fase 3 (lambda completo, epocas 500+):**
- Loss: ~0.568 (estancada)
- LR: 3.2e-8 (efectivamente cero)
- Sin capacidad de aprendizaje adicional
- El checkpoint efectivo es la epoca 500

---

## 13. Metricas de evaluacion — Comparacion

### 13.1 MSE por variable

| Metrica | Iter 1 (ep 2000) | Iter 2 (ep 500) | Mejora |
|---------|:----------------:|:---------------:|:------:|
| MSE u_x | 0.6718 | **0.2875** | **-57.2%** |
| MSE u_y | 0.6279 | **0.2775** | **-55.8%** |
| MSE p (rescaled) | — | 0.8282 | — |
| MSE p (escala orig) | 1.4234 | 3.1999 | +124.8% |
| Residual NS | 0.0204 | **0.0161** | **-21.1%** |

### 13.2 Interpretacion

- **Velocidades (u_x, u_y):** Mejora sustancial (>55%). El curriculum learning permitio al
  modelo aprender la estructura espacial y temporal de los campos de viento.
- **Presion:** En escala original, la MSE empeoro. Sin embargo, esto es esperado porque
  (a) el modelo solo tuvo ~300 epocas con restriccion de presion (vs 2000 en Iter 1), y
  (b) el reescalado cambio la escala del gradiente. En escala rescalada, MSE_p=0.83 es
  comparable a las velocidades (~0.28), indicando un entrenamiento mas balanceado.
- **Residual NS:** Menor que Iter 1 (0.016 vs 0.020), indicando que la restriccion fisica
  se satisface incluso mejor a pesar del curriculum.

### 13.3 Variacion temporal — Diagnostico de colapso estacionario

Se comparo la prediccion del modelo en dos timesteps extremos (t=0 y t=743):

| Variable | max_diff | mean_diff | Iter 1 status |
|----------|:--------:|:---------:|:-------------:|
| u_x | **1.2043** | **0.4118** | ~0 (estacionario) |
| u_y | **1.1456** | **0.4139** | ~0 (estacionario) |
| p | **0.4081** | **0.2097** | sutil |

**El modelo captura dinamica temporal.** Los rangos de prediccion cambian
significativamente entre timesteps:
- t=0: u=[0.40, 1.23], v=[-1.10, -0.44], p=[-0.30, -0.15]
- t=743: u=[-0.13, 1.09], v=[-1.57, 0.46], p=[-0.70, 0.25]

**El colapso estacionario de la Iteracion 1 fue resuelto.**

---

## 14. Problema detectado: interaccion scheduler-curriculum

### 14.1 Descripcion

ReduceLROnPlateau con patience=50 reduce el LR cuando la loss no mejora durante 50 epocas.
Durante la fase de rampa (epocas 200-499), la loss total se mantiene aproximadamente
constante porque el incremento gradual de lambda compensa cualquier mejora del modelo.
El scheduler interpreta esto como estancamiento y reduce el LR agresivamente:

| Epoca | LR | Reducciones acumuladas |
|------:|---:|:----------------------:|
| 0 | 1.0e-4 | 0 |
| 140 | 3.2e-5 | 1 |
| 245 | 1.0e-5 | 2 |
| 295 | 3.2e-6 | 3 |
| 345 | 1.0e-6 | 4 |
| 400 | 3.2e-7 | 5 |
| 455 | 1.0e-7 | 6 |
| 505 | 3.2e-8 | 7 |

Para la epoca 500, el LR es 3200x menor que el inicial — practicamente cero.

### 14.2 Impacto

- El modelo no puede mejorar despues de la epoca ~400
- 1500 epocas de la Fase 3 son computacionalmente desperdiciadas
- El entrenamiento efectivo fue de ~400 epocas

### 14.3 Solucion propuesta para Iteracion 3

| Opcion | Descripcion | Complejidad |
|--------|-------------|:-----------:|
| A | **Congelar scheduler durante la rampa** — solo activar en Fase 1 y Fase 3 | Baja |
| B | **Reiniciar scheduler** al inicio de cada fase | Baja |
| C | **CosineAnnealingWarmRestarts** — scheduler ciclico con T_0 adaptado | Media |
| D | **LR fijo durante rampa** — solo usar scheduler en Fase 1 y Fase 3 | Baja |

**Recomendacion:** Opcion A o D (congelar/ignorar scheduler durante Fase 2).

---

## 15. Artefactos generados — Iteracion 2

| Archivo | Descripcion |
|---------|-------------|
| `models/PINN_caribe_v2_epoca_500.pth` | Checkpoint principal |
| `models/PINN_caribe_v2_epoca_1000.pth` | Checkpoint (en curso, sin mejora esperada) |
| `models/historial_PINN_caribe_v2.mat` | Historial completo |
| `reports/pinn_caribe_v2_epoca_500_u.gif` | GIF u_x (12 MB, 744 frames) |
| `reports/pinn_caribe_v2_epoca_500_v.gif` | GIF u_y (11 MB, 744 frames) |
| `reports/pinn_caribe_v2_epoca_500_p.gif` | GIF presion (17 MB, 744 frames) |

---

## 16. Archivos modificados — Iteracion 2

| Archivo | Cambio |
|---------|--------|
| `config/settings.py` | Nuevos: EPOCAS_SOLO_DATOS, EPOCAS_RAMPA, REESCALAR_PRESION |
| `src/data/normalizar_datos.py` | Reescalado de presion por sigma_p |
| `src/model/perdida_fisica.py` | p_scale en ecuaciones de momento N-S |
| `src/training/entrenador.py` | Curriculum learning (3 fases), p_scale, lambda historial |
| `main.py` | Pasa parametros de Iteracion 2 |
| `evaluar.py` | Lee p_scale del checkpoint, argumento --prefijo |
| `src/evaluation/metricas.py` | Pasa p_scale a residual NS |

---

## 17. Conclusion general

### 17.1 Resumen de la Iteracion 2

El curriculum learning rompio exitosamente el colapso estacionario de la Iteracion 1.
Las velocidades mejoraron un **56-57%** en MSE, la restriccion fisica se satisface mejor
(residual NS -21%), y el modelo captura dinamica temporal significativa.

La presion en escala original empeoro, pero esto se atribuye al entrenamiento truncado
por el problema del scheduler (solo ~400 epocas efectivas, de las cuales ~300 con presion).

### 17.2 Proximos ajustes — Iteracion 3

| # | Ajuste | Prioridad | Justificacion |
|---|--------|:---------:|---------------|
| 1 | Congelar scheduler durante rampa de lambda | **Alta** | Evitar agotamiento de LR |
| 2 | Ampliar Fase 3 con LR fresco (1e-4 o 1e-5) | **Alta** | Permitir convergencia real |
| 3 | Evaluar pesos separados para u/v/p en la loss | Media | La presion aun domina |
| 4 | Considerar CosineAnnealingWarmRestarts | Media | Mejor exploracion del landscape |
| 5 | Aumentar paciencia del scheduler (100-200) | Baja | Alternativa conservadora |
