# INFORME DE ENTRENAMIENTO — Iteracion 1

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

## 9. Conclusion

El entrenamiento de 2000 epocas logro una reduccion del 64.4% en la loss total, pero
el modelo convergio a una **solucion cuasi-estacionaria** que no captura la dinamica
temporal del viento ni la presion. La presion es el cuello de botella (42.2% de la loss),
influenciada por un rango normalizado 10 veces mayor que las velocidades.

Se requiere una **iteracion 2** centrada en curriculum learning y/o reescalado de la
presion para romper el colapso estacionario.
