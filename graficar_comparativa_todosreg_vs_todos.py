"""
Gráficas comparativas de funciones de entrenamiento:
    - PINN_caribe_excl4_todosreg_R01_L5_rangeL05  (entrenamiento actual, con range loss)
    - PINN_caribe_excl4_todos_R01_L5              (entrenamiento previo, sin range loss)

Las imágenes se guardan en reports/iter_todosreg_vs_todos_R01_L5_20260430/.
"""

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
RUTA_MODELOS = Path("models")
RUTA_SALIDA  = Path("reports/iter_todosreg_vs_todos_R01_L5_20260430")
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# ── Carga de historiales ──────────────────────────────────────────────────────
h_actual = sio.loadmat(RUTA_MODELOS / "historial_PINN_caribe_excl4_todosreg_R01_L5_rangeL05.mat")
h_previo = sio.loadmat(RUTA_MODELOS / "historial_PINN_caribe_excl4_todos_R01_L5.mat")

def _arr(hist, clave):
    """Extrae y aplana un array de un historial .mat."""
    return np.array(hist[clave]).flatten()

ep_act  = _arr(h_actual, "epoch") + 1   # base 1 para graficar
ep_prev = _arr(h_previo, "epoch") + 1

# ── Paleta y estilos ──────────────────────────────────────────────────────────
COLOR_ACT  = "#1f77b4"   # azul  — actual  (con range loss)
COLOR_PREV = "#ff7f0e"   # naranja — previo (sin range loss)
ALPHA_LINE = 0.85
LW = 1.4
LW_SMOOTH = 2.2

def suavizar(y, ventana=30):
    """Media móvil simple."""
    kernel = np.ones(ventana) / ventana
    return np.convolve(y, kernel, mode="same")

def guardar(fig, nombre):
    ruta = RUTA_SALIDA / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {ruta}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Loss total
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

loss_act  = _arr(h_actual, "loss")
loss_prev = _arr(h_previo, "loss")

ax.plot(ep_prev, loss_prev, color=COLOR_PREV, alpha=0.25, lw=LW, label="_nolegend_")
ax.plot(ep_act,  loss_act,  color=COLOR_ACT,  alpha=0.25, lw=LW, label="_nolegend_")
ax.plot(ep_prev, suavizar(loss_prev), color=COLOR_PREV, lw=LW_SMOOTH,
        label="todos_R01_L5 (sin range loss)")
ax.plot(ep_act,  suavizar(loss_act),  color=COLOR_ACT,  lw=LW_SMOOTH,
        label="todosreg_R01_L5_rangeL05 (con range loss)")

ax.set_xlabel("Época")
ax.set_ylabel("Loss total")
ax.set_title("Comparativa — Loss total de entrenamiento")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1, max(ep_act.max(), ep_prev.max()))
fig.tight_layout()
guardar(fig, "01_loss_total_comparativa.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Componentes de pérdida individuales (NS, U, V, P)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
componentes = [
    ("ns_loss",  axes[0, 0], "Pérdida Navier-Stokes (NS)"),
    ("u_loss",   axes[0, 1], "Pérdida datos — u (viento X)"),
    ("v_loss",   axes[1, 0], "Pérdida datos — v (viento Y)"),
    ("p_loss",   axes[1, 1], "Pérdida datos — p (presión)"),
]

for clave, ax, titulo in componentes:
    y_act  = _arr(h_actual, clave)
    y_prev = _arr(h_previo, clave)

    ax.plot(ep_prev, y_prev, color=COLOR_PREV, alpha=0.2, lw=LW)
    ax.plot(ep_act,  y_act,  color=COLOR_ACT,  alpha=0.2, lw=LW)
    ax.plot(ep_prev, suavizar(y_prev), color=COLOR_PREV, lw=LW_SMOOTH,
            label="todos_R01_L5")
    ax.plot(ep_act,  suavizar(y_act),  color=COLOR_ACT,  lw=LW_SMOOTH,
            label="todosreg + rangeL05")

    ax.set_title(titulo)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(ep_act.max(), ep_prev.max()))

fig.suptitle("Comparativa — Componentes de pérdida individuales", fontsize=13, y=1.01)
fig.tight_layout()
guardar(fig, "02_componentes_loss_comparativa.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Learning rate
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))

lr_act  = _arr(h_actual, "lr")
lr_prev = _arr(h_previo, "lr")

ax.semilogy(ep_prev, lr_prev, color=COLOR_PREV, lw=LW_SMOOTH, label="todos_R01_L5")
ax.semilogy(ep_act,  lr_act,  color=COLOR_ACT,  lw=LW_SMOOTH, label="todosreg + rangeL05")

ax.set_xlabel("Época")
ax.set_ylabel("Learning rate (escala log)")
ax.set_title("Comparativa — Learning rate")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
ax.set_xlim(1, max(ep_act.max(), ep_prev.max()))
fig.tight_layout()
guardar(fig, "03_learning_rate_comparativa.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Range loss y LambdaR (solo entrenamiento actual)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 3, figure=fig)

ax_total  = fig.add_subplot(gs[0, 0])
ax_u      = fig.add_subplot(gs[0, 1])
ax_v      = fig.add_subplot(gs[0, 2])
ax_p      = fig.add_subplot(gs[1, 0])
ax_lambda = fig.add_subplot(gs[1, 1:])

COLOR_RANGO = "#2ca02c"

# Range loss total
rl = _arr(h_actual, "range_loss")
ax_total.plot(ep_act, rl, color=COLOR_RANGO, alpha=0.25, lw=LW)
ax_total.plot(ep_act, suavizar(rl), color=COLOR_RANGO, lw=LW_SMOOTH)
ax_total.set_title("Range loss total (ponderada)")
ax_total.set_xlabel("Época"); ax_total.set_ylabel("Loss"); ax_total.grid(True, alpha=0.3)

# Range loss por variable
for ax_var, clave, titulo, color in [
    (ax_u, "range_u_loss", "Range loss — u (viento X)", "#1f77b4"),
    (ax_v, "range_v_loss", "Range loss — v (viento Y)", "#9467bd"),
    (ax_p, "range_p_loss", "Range loss — p (presión)",  "#d62728"),
]:
    y = _arr(h_actual, clave)
    ax_var.plot(ep_act, y, color=color, alpha=0.25, lw=LW)
    ax_var.plot(ep_act, suavizar(y), color=color, lw=LW_SMOOTH)
    ax_var.set_title(titulo)
    ax_var.set_xlabel("Época"); ax_var.set_ylabel("Loss"); ax_var.grid(True, alpha=0.3)

# LambdaR (rampa)
lr_eff = _arr(h_actual, "lambda_rango_efecto")
ax_lambda.plot(ep_act, lr_eff, color="#8c564b", lw=LW_SMOOTH)
ax_lambda.axhline(0.5, ls="--", color="gray", alpha=0.6, label="λR máx = 0.50")
ax_lambda.fill_between(ep_act, lr_eff, alpha=0.15, color="#8c564b")
ax_lambda.set_title("Evolución de LambdaR (rampa → estabilización en 0.50)")
ax_lambda.set_xlabel("Época"); ax_lambda.set_ylabel("λR efectivo")
ax_lambda.legend(); ax_lambda.grid(True, alpha=0.3)
ax_lambda.set_xlim(1, ep_act.max())

fig.suptitle("todosreg_R01_L5_rangeL05 — Métricas de range loss", fontsize=13, y=1.01)
fig.tight_layout()
guardar(fig, "04_range_loss_detalle.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Fracción Fuera de Rango (FOR) — solo entrenamiento actual
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for_data = [
    ("for_u", axes[0], "FOR — u (viento X)", "#1f77b4"),
    ("for_v", axes[1], "FOR — v (viento Y)", "#9467bd"),
    ("for_p", axes[2], "FOR — p (presión)",  "#d62728"),
]

for clave, ax, titulo, color in for_data:
    y = _arr(h_actual, clave) * 100  # a porcentaje
    ax.plot(ep_act, y, color=color, alpha=0.25, lw=LW)
    ax.plot(ep_act, suavizar(y), color=color, lw=LW_SMOOTH)
    ax.set_title(titulo)
    ax.set_xlabel("Época")
    ax.set_ylabel("FOR (%)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, ep_act.max())
    ax.set_ylim(bottom=0)

fig.suptitle("todosreg_R01_L5_rangeL05 — Fracción Fuera de Rango (FOR)", fontsize=13)
fig.tight_layout()
guardar(fig, "05_for_metricas.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Panel resumen: loss total + componentes en una figura
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

ax_loss = fig.add_subplot(gs[0, :])
ax_ns   = fig.add_subplot(gs[1, 0])
ax_u2   = fig.add_subplot(gs[1, 1])
ax_v2   = fig.add_subplot(gs[1, 2])
ax_p2   = fig.add_subplot(gs[1, 3])
ax_lr2  = fig.add_subplot(gs[2, :2])
ax_lmr  = fig.add_subplot(gs[2, 2:])

# Loss total
ax_loss.plot(ep_prev, loss_prev, color=COLOR_PREV, alpha=0.2, lw=LW)
ax_loss.plot(ep_act,  loss_act,  color=COLOR_ACT,  alpha=0.2, lw=LW)
ax_loss.plot(ep_prev, suavizar(loss_prev), color=COLOR_PREV, lw=LW_SMOOTH,
             label="todos_R01_L5 (sin range loss)  — final: {:.3f}".format(loss_prev[-50:].mean()))
ax_loss.plot(ep_act,  suavizar(loss_act),  color=COLOR_ACT,  lw=LW_SMOOTH,
             label="todosreg_R01_L5_rangeL05 (con range loss) — final: {:.3f}".format(loss_act[-50:].mean()))
ax_loss.set_title("Loss total"); ax_loss.set_xlabel("Época"); ax_loss.set_ylabel("Loss")
ax_loss.legend(fontsize=9); ax_loss.grid(True, alpha=0.3)

# Componentes
for ax_c, clave, titulo, short in [
    (ax_ns, "ns_loss", "NS", "NS"),
    (ax_u2, "u_loss",  "u (viento X)", "u"),
    (ax_v2, "v_loss",  "v (viento Y)", "v"),
    (ax_p2, "p_loss",  "p (presión)",  "p"),
]:
    y_act  = _arr(h_actual, clave)
    y_prev = _arr(h_previo, clave)
    ax_c.plot(ep_prev, suavizar(y_prev), color=COLOR_PREV, lw=LW_SMOOTH, label="previo")
    ax_c.plot(ep_act,  suavizar(y_act),  color=COLOR_ACT,  lw=LW_SMOOTH, label="actual")
    ax_c.set_title(f"Loss {titulo}"); ax_c.set_xlabel("Época"); ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=7)

# LR
ax_lr2.semilogy(ep_prev, lr_prev, color=COLOR_PREV, lw=LW_SMOOTH, label="previo")
ax_lr2.semilogy(ep_act,  lr_act,  color=COLOR_ACT,  lw=LW_SMOOTH, label="actual")
ax_lr2.set_title("Learning rate"); ax_lr2.set_xlabel("Época")
ax_lr2.set_ylabel("LR (log)"); ax_lr2.grid(True, alpha=0.3, which="both")
ax_lr2.legend(fontsize=8)

# LambdaR
ax_lmr.plot(ep_act, _arr(h_actual, "lambda_rango_efecto"), color="#8c564b", lw=LW_SMOOTH)
ax_lmr.axhline(0.5, ls="--", color="gray", alpha=0.6, label="máx = 0.50")
ax_lmr.set_title("LambdaR — rampa y estabilización"); ax_lmr.set_xlabel("Época")
ax_lmr.set_ylabel("λR"); ax_lmr.grid(True, alpha=0.3); ax_lmr.legend(fontsize=8)

fig.suptitle(
    "Panel resumen — todosreg_R01_L5_rangeL05  vs  todos_R01_L5\n"
    "(Configuración: R=0.1 · λNS=5.0 · 2000 épocas · excl. 4 estaciones para test)",
    fontsize=12, y=1.01
)
guardar(fig, "00_panel_resumen.png")

print("\nTodas las gráficas generadas en:", RUTA_SALIDA)
