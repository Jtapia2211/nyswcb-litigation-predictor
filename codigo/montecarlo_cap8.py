"""
Monte Carlo simulation for Cap8 — Impacto Económico.
Extended version: includes p_success (intervention effectiveness) as random variable.

Generates:
  fig5_montecarlo_distribution.png  — histogram comparing upper-bound (p=1) vs. full MC
  fig6_montecarlo_tornado.png       — tornado chart (now with p_success as 4th driver)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
from pathlib import Path
from scipy import stats

# ── Seed and size ─────────────────────────────────────────────────────────────
np.random.seed(42)
N = 50_000

PLOTS_DIR = Path("/sessions/epic-intelligent-hawking/mnt/Tesis_ML/codigo/model_plots8")
JSON_PATH  = PLOTS_DIR / "economic_summary.json"

# ── Confusion matrix (fixed) ──────────────────────────────────────────────────
TP = 47_957
FP = 35_390
FN =  7_426
TN = 169_383

# ── Cost parameter distributions (triangular: low, mode, high) ───────────────
C_j_params  = dict(low=12_600, mode=18_000, high=23_400)   # ±30%
C_p_params  = dict(low=2_800,  mode=3_500,  high=4_200)    # ±20%
C_fp_params = dict(low=350,    mode=500,    high=650)       # ±30%

# ── NEW: Intervention success rate (p_success) ───────────────────────────────
# Literature: mediation success in NY workers' comp = 40–75%
# Mode = 0.60 (NYSWCB 2022 annual report: 58% resolution rate in early mediation)
p_success_params = dict(low=0.40, mode=0.60, high=0.75)

def triangular_sample(params, n):
    lo, mo, hi = params['low'], params['mode'], params['high']
    c = (mo - lo) / (hi - lo)
    return (hi - lo) * np.random.triangular(0, c, 1, n) + lo

# ── Sampling ──────────────────────────────────────────────────────────────────
C_j      = triangular_sample(C_j_params,      N)
C_p      = triangular_sample(C_p_params,      N)
C_fp     = triangular_sample(C_fp_params,     N)
p_suc    = triangular_sample(p_success_params, N)

# ── Savings formulae ──────────────────────────────────────────────────────────
# Baseline: all positives litigate
baseline_cost = (TP + FN) * C_j

# Full model (with p_success):
#   - TP detected: pay prevention cost C_p
#   - Of those, p_success fraction avoids litigation; (1-p_success) still litigates
#   - FP: pay audit cost C_fp
#   - FN: missed, still litigate (cost C_j)
# => net_savings = TP*(p_success*C_j - C_p) - FP*C_fp
model_cost_full  = TP*C_p + TP*(1 - p_suc)*C_j + FP*C_fp + FN*C_j
net_savings_full = (baseline_cost - model_cost_full) / 1e6   # millions

# Upper-bound model (p_success = 1.0, original assumption):
model_cost_ub  = TP*C_p + FP*C_fp + FN*C_j
net_savings_ub = (baseline_cost - model_cost_ub) / 1e6

# ── Stats — full (primary) ────────────────────────────────────────────────────
def pct(arr, q): return float(np.percentile(arr, q))

p5_f  = pct(net_savings_full, 5)
p25_f = pct(net_savings_full, 25)
p50_f = pct(net_savings_full, 50)
p75_f = pct(net_savings_full, 75)
p95_f = pct(net_savings_full, 95)
mean_f = float(np.mean(net_savings_full))
std_f  = float(np.std(net_savings_full))
prob_pos_f   = float(np.mean(net_savings_full > 0) * 100)
prob_500_f   = float(np.mean(net_savings_full > 500) * 100)
prob_200_f   = float(np.mean(net_savings_full > 200) * 100)

# Stats — upper bound (p_success=1)
p5_ub   = pct(net_savings_ub, 5)
p50_ub  = pct(net_savings_ub, 50)
p95_ub  = pct(net_savings_ub, 95)
mean_ub = float(np.mean(net_savings_ub))

print("── Monte Carlo Results (FULL: including p_success uncertainty) ──────────")
print(f"  P5   = ${p5_f:,.0f}M")
print(f"  P25  = ${p25_f:,.0f}M")
print(f"  Mean = ${mean_f:,.0f}M")
print(f"  P50  = ${p50_f:,.0f}M")
print(f"  P75  = ${p75_f:,.0f}M")
print(f"  P95  = ${p95_f:,.0f}M")
print(f"  Std  = ${std_f:,.0f}M")
print(f"  P(>0)    = {prob_pos_f:.1f}%")
print(f"  P(>200M) = {prob_200_f:.1f}%")
print(f"  P(>500M) = {prob_500_f:.1f}%")
print()
print("── Upper-bound (p_success=1, cota superior) ─────────────────────────────")
print(f"  P5={p5_ub:,.0f}M  P50={p50_ub:,.0f}M  P95={p95_ub:,.0f}M  Mean={mean_ub:,.0f}M")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Dual distribution: upper-bound vs. full MC
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))

# Common bin range
x_min = min(net_savings_full.min(), net_savings_ub.min()) - 20
x_max = max(net_savings_full.max(), net_savings_ub.max()) + 20
bins = np.linspace(x_min, x_max, 90)

# Upper-bound histogram (ghost)
ax.hist(net_savings_ub,   bins=bins, color='#AABDD6', alpha=0.45,
        label=f'Cota superior (p_éxito = 100%)  —  P50 = USD {p50_ub:,.0f}M',
        edgecolor='white', linewidth=0.2)

# Full MC histogram (primary)
ax.hist(net_savings_full, bins=bins, color='#2166AC', alpha=0.75,
        label=f'Estimación central (p_éxito ~ Triang[0.40, 0.60, 0.75])  —  P50 = USD {p50_f:,.0f}M',
        edgecolor='white', linewidth=0.2)

# KDE — full
kde_f = stats.gaussian_kde(net_savings_full, bw_method=0.12)
x_kde = np.linspace(x_min, x_max, 500)
scale = N * (bins[1] - bins[0])
ax.plot(x_kde, kde_f(x_kde) * scale, color='#1a1a2e', lw=2.0, zorder=6)

# Vertical lines — full MC
for val, col, lbl, ls in [
    (p5_f,   '#D73027', f'P5 = USD {p5_f:,.0f}M',  'dashed'),
    (p50_f,  '#222222', f'P50 = USD {p50_f:,.0f}M', 'solid'),
    (p95_f,  '#1A9641', f'P95 = USD {p95_f:,.0f}M', 'dashed'),
]:
    ax.axvline(val, color=col, lw=1.8, linestyle=ls, label=lbl, zorder=7)

ax.axvline(mean_f, color='#7B2D8B', lw=1.4, linestyle='dotted',
           label=f'Media = USD {mean_f:,.0f}M', zorder=7)

# Zero line
ax.axvline(0, color='black', lw=1.2, linestyle='solid', alpha=0.4)

ax.set_xlabel('Ahorro Neto (millones USD)', fontsize=12)
ax.set_ylabel('Frecuencia (simulaciones)', fontsize=12)
ax.set_title(
    f'Distribución del Ahorro Neto — Simulación Monte Carlo (N = {N:,})\n'
    f'P(ahorro > 0) = {prob_pos_f:.1f}%   |   P(ahorro > USD 200M) = {prob_200_f:.1f}%   |   Desv. estándar = USD {std_f:,.0f}M',
    fontsize=10.5, pad=10
)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
plt.tight_layout()

out5 = PLOTS_DIR / "fig5_montecarlo_distribution.png"
fig.savefig(out5, dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"\n  ✓ Saved {out5.name}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Tornado chart (one-at-a-time, now with p_success)
# ══════════════════════════════════════════════════════════════════════════════
def calc_savings_full(cj, cp, cfp, ps):
    base  = (TP + FN) * cj
    model = TP*cp + TP*(1-ps)*cj + FP*cfp + FN*cj
    return (base - model) / 1e6

# Base point (all modes)
base_sav = calc_savings_full(
    C_j_params['mode'], C_p_params['mode'],
    C_fp_params['mode'], p_success_params['mode']
)

tornado_data = []
configs = [
    ('p_éxito (efectividad mediación)',  p_success_params, 'ps',  '#D73027'),
    ('C_j (costo litigación)',           C_j_params,       'C_j', '#2166AC'),
    ('C_p (costo prevención)',           C_p_params,       'C_p', '#4DAC26'),
    ('C_fp (costo falsa alarma)',        C_fp_params,      'C_fp','#F4A82E'),
]

for label, params, var, color in configs:
    def _s(v, which):
        return calc_savings_full(
            params[which] if var == 'C_j'  else C_j_params['mode'],
            params[which] if var == 'C_p'  else C_p_params['mode'],
            params[which] if var == 'C_fp' else C_fp_params['mode'],
            params[which] if var == 'ps'   else p_success_params['mode'],
        )
    lo_sav = _s(params['low'],  'low')
    hi_sav = _s(params['high'], 'high')
    swing  = abs(hi_sav - lo_sav)
    tornado_data.append((label, lo_sav, hi_sav, swing, color))

tornado_data.sort(key=lambda x: x[3], reverse=True)

fig2, ax2 = plt.subplots(figsize=(9.5, 4.2))
bar_h = 0.48

for i, (label, lo_sav, hi_sav, swing, color) in enumerate(tornado_data):
    left  = min(lo_sav, hi_sav)
    right = max(lo_sav, hi_sav)
    ax2.barh(i, right - left, left=left, height=bar_h,
             color=color, alpha=0.82, edgecolor='white', linewidth=0.5)
    ax2.text(left  - 10, i, f'${lo_sav:,.0f}M', va='center', ha='right', fontsize=8.5, color='#333')
    ax2.text(right + 10, i, f'${hi_sav:,.0f}M', va='center', ha='left',  fontsize=8.5, color='#333')
    ax2.text((left+right)/2, i, f'Δ${swing:,.0f}M', va='center', ha='center',
             fontsize=8, color='white', fontweight='bold')

ax2.axvline(base_sav, color='black', lw=1.5, linestyle='--',
            label=f'Base = ${base_sav:,.0f}M')
ax2.axvline(0, color='red', lw=0.8, linestyle=':', alpha=0.6, label='Cero (sin ahorro)')
ax2.set_yticks(range(len(tornado_data)))
ax2.set_yticklabels([d[0] for d in tornado_data], fontsize=10)
ax2.set_xlabel('Ahorro Neto (millones USD)', fontsize=11)
ax2.set_title('Análisis de Sensibilidad (Tornado) — Impacto de Cada Parámetro sobre el Ahorro Neto',
              fontsize=10, pad=10)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))
ax2.legend(fontsize=9, loc='lower right')
all_vals = [d[1] for d in tornado_data] + [d[2] for d in tornado_data]
ax2.set_xlim(min(all_vals) - 70, max(all_vals) + 70)
plt.tight_layout()

out6 = PLOTS_DIR / "fig6_montecarlo_tornado.png"
fig2.savefig(out6, dpi=160, bbox_inches='tight')
plt.close(fig2)
print(f"  ✓ Saved {out6.name}")

# ── Update economic_summary.json ──────────────────────────────────────────────
with open(JSON_PATH) as f:
    summary = json.load(f)

summary['montecarlo'] = {
    'N': N,
    # Full MC (primary — includes p_success uncertainty)
    'p5_M':  round(p5_f, 1),
    'p25_M': round(p25_f, 1),
    'mean_M': round(mean_f, 1),
    'p50_M': round(p50_f, 1),
    'p75_M': round(p75_f, 1),
    'p95_M': round(p95_f, 1),
    'std_M': round(std_f, 1),
    'prob_positive_pct':    round(prob_pos_f, 1),
    'prob_above_200M_pct':  round(prob_200_f, 1),
    'prob_above_500M_pct':  round(prob_500_f, 1),
    # Upper bound (p_success=1)
    'ub_p5_M':  round(p5_ub, 1),
    'ub_p50_M': round(p50_ub, 1),
    'ub_p95_M': round(p95_ub, 1),
    'ub_mean_M': round(mean_ub, 1),
    # Parameter distributions
    'C_j_params':       C_j_params,
    'C_p_params':       C_p_params,
    'C_fp_params':      C_fp_params,
    'p_success_params': p_success_params,
}

with open(JSON_PATH, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Updated economic_summary.json")
print("\n✅ Monte Carlo (full) complete.")
