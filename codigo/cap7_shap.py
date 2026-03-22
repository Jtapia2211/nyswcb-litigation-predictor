"""
cap7_shap.py
============
Análisis de Interpretabilidad SHAP — CatBoost GPU (modelo tuneado Cap. 6)
Maestría en Ciencia de Datos — ITBA  |  Tapia, Julián

Genera 7 figuras para el Cap. 7 de la tesis:
  fig1_shap_bar.png          → Importancia global (mean |SHAP|)
  fig2_shap_beeswarm.png     → Beeswarm global (dirección + magnitud)
  fig3_shap_dep_aww.png      → Dependence plot: AWW
  fig4_shap_dep_c3.png       → Dependence plot: has_C3 × days_C3_to_accident
  fig5_shap_dep_nature.png   → Dependence plot: wcio_nature_code
  fig6_shap_waterfall.png    → Waterfall 3 casos (riesgo alto/medio/bajo)
  fig7_shap_force.png        → Force plot interactivo exportado como PNG

Requisitos:
    pip install shap catboost pandas numpy matplotlib

Uso:
    python cap7_shap.py
"""

import os, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(r"C:\Users\julia\Tesis_ML")
DATA_FILE  = BASE_DIR / "raw_data" / "dataset_tesis_clean.csv"
MODEL_FILE = BASE_DIR / "codigo" / "tuning_catboost" / "best_catboost.cbm"
OUT_DIR    = BASE_DIR / "codigo" / "model_plots7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
YEAR_COL   = "accident_year"
SHAP_SAMPLE = 8000   # registros del test 2022 para SHAP (TreeExplainer es exacto, no aproximado)
RANDOM_SEED = 42

# ── Nombres de display para las 23 features ──────────────────────────────────
FEAT_NAMES = [
    'days_to_assembly', 'days_C2_to_accident', 'days_C3_to_accident',
    'age_at_injury', 'aww', 'has_C2', 'has_C3', 'has_ANCR_early',
    'accident_year', 'accident_month', 'accident_dow',
    'gender', 'accident_type', 'occupational_disease',
    'county_of_injury', 'medical_fee_region',
    'wcio_cause_code', 'wcio_nature_code', 'wcio_body_code',
    'carrier_type', 'district_name', 'industry_code', 'industry_desc',
]
CAT_FEATURES = [
    'gender', 'accident_type', 'occupational_disease',
    'county_of_injury', 'medical_fee_region',
    'wcio_cause_code', 'wcio_nature_code', 'wcio_body_code',
    'carrier_type', 'district_name', 'industry_code', 'industry_desc',
]
DISPLAY_NAMES = {
    'aww'                 : 'Salario Semanal Promedio (AWW)',
    'has_C3'              : 'Presenta Formulario C-3',
    'days_C3_to_accident' : 'Días C-3 → Accidente',
    'wcio_nature_code'    : 'Naturaleza de la Lesión (WCIO)',
    'wcio_cause_code'     : 'Causa del Accidente (WCIO)',
    'wcio_body_code'      : 'Parte del Cuerpo (WCIO)',
    'county_of_injury'    : 'Condado del Accidente',
    'industry_code'       : 'Código de Industria',
    'days_to_assembly'    : 'Días hasta Asamblea',
    'age_at_injury'       : 'Edad al Momento del Accidente',
    'has_C2'              : 'Presenta Formulario C-2',
    'days_C2_to_accident' : 'Días C-2 → Accidente',
    'has_ANCR_early'      : 'ANCR Temprano',
    'accident_year'       : 'Año del Accidente',
    'accident_month'      : 'Mes del Accidente',
    'accident_dow'        : 'Día de la Semana',
    'gender'              : 'Género',
    'accident_type'       : 'Tipo de Accidente',
    'occupational_disease': 'Enfermedad Ocupacional',
    'medical_fee_region'  : 'Región de Honorarios Médicos',
    'carrier_type'        : 'Tipo de Asegurador',
    'district_name'       : 'Distrito NYWCB',
    'industry_desc'       : 'Descripción de Industria',
}
PALETTE = {
    'pos'  : '#C0392B',   # rojo — empuja hacia judicialización
    'neg'  : '#2E86C1',   # azul — protege
    'neu'  : '#888888',   # gris — neutro
    'gold' : '#F39C12',
    'dark' : '#2E4057',
}

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS Y MODELO
# ══════════════════════════════════════════════════════════════════════════════

def load_test():
    print("[data] Cargando CSV ...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan", "MISSING")
    num_feats = [f for f in FEAT_NAMES if f not in CAT_FEATURES]
    for c in num_feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1)

    test = df[df[YEAR_COL] == 2022].copy()
    feats = [f for f in FEAT_NAMES if f in test.columns]
    X_te = test[feats]
    y_te = test[TARGET_COL].values.astype(int)
    print(f"[data] Test 2022: {X_te.shape}  | positivos: {y_te.mean()*100:.1f}%")
    return X_te, y_te, feats

def load_model():
    print("[model] Cargando best_catboost.cbm ...")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_FILE))
    print(f"[model] Cargado OK — {model.tree_count_} árboles, profundidad={model.get_param('depth')}")
    return model

# ══════════════════════════════════════════════════════════════════════════════
#  SHAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(model, X_te, y_te, feats):
    print(f"[shap] Tomando muestra estratificada de {SHAP_SAMPLE} registros ...")
    rng = np.random.default_rng(RANDOM_SEED)

    # Muestra estratificada: mitad positivos, mitad negativos
    idx_pos = np.where(y_te == 1)[0]
    idx_neg = np.where(y_te == 0)[0]
    n_pos   = min(SHAP_SAMPLE // 2, len(idx_pos))
    n_neg   = SHAP_SAMPLE - n_pos
    idx     = np.concatenate([
        rng.choice(idx_pos, n_pos, replace=False),
        rng.choice(idx_neg, n_neg, replace=False),
    ])
    idx = rng.permutation(idx)

    X_s = X_te.iloc[idx].reset_index(drop=True)
    y_s = y_te[idx]

    print(f"[shap] Muestra: {len(X_s)} registros ({y_s.mean()*100:.1f}% positivos)")
    print("[shap] Ejecutando TreeExplainer ...")
    t0 = time.time()
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)
    expected    = float(explainer.expected_value)
    print(f"[shap] SHAP completado en {time.time()-t0:.1f}s  |  E[f(x)] = {expected:.4f}")

    # Guardar SHAP values para reutilizar
    np.save(str(OUT_DIR / "shap_values.npy"),   shap_values)
    np.save(str(OUT_DIR / "shap_X_sample.npy"), X_s.values)
    with open(OUT_DIR / "shap_meta.json", "w") as f:
        json.dump({"expected_value": expected, "feature_names": feats,
                   "n_sample": len(X_s), "n_pos": int(y_s.sum())}, f)
    print(f"[shap] Guardados shap_values.npy, shap_X_sample.npy, shap_meta.json")

    return shap_values, X_s, y_s, expected, explainer

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 1 — Bar importance global
# ══════════════════════════════════════════════════════════════════════════════

def fig1_bar(shap_values, X_s, feats):
    print("[fig1] Bar importance global ...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:15]   # top 15
    names    = [DISPLAY_NAMES.get(feats[i], feats[i]) for i in order]
    vals     = mean_abs[order]
    colors   = [PALETTE['pos'] if v > vals[0]*0.3 else PALETTE['dark'] for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white", height=0.7)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, color="#444")
    ax.set_xlabel("mean(|SHAP value|)  —  impacto promedio sobre la probabilidad de judicialización",
                  fontsize=10)
    ax.set_title("Importancia Global de Variables — Modelo CatBoost Tuneado\n"
                 "(SHAP TreeExplainer, muestra test 2022, n=8.000)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, vals.max() * 1.18)
    ax.axvline(0, color="#ccc", lw=0.8)
    ax.grid(axis="x", alpha=0.25, ls="--")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "fig1_shap_bar.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[fig1] {out}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 2 — Beeswarm global
# ══════════════════════════════════════════════════════════════════════════════

def fig2_beeswarm(shap_values, X_s, feats):
    print("[fig2] Beeswarm global ...")
    display = [DISPLAY_NAMES.get(f, f) for f in feats]
    explanation = shap.Explanation(
        values          = shap_values,
        base_values     = np.zeros(len(X_s)),
        data            = X_s.values,
        feature_names   = display,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=15, show=False, color_bar=True)
    plt.title("Dirección e Intensidad del Impacto por Variable\n"
              "(SHAP beeswarm — color = valor de la feature: rojo=alto, azul=bajo)",
              fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    out = OUT_DIR / "fig2_shap_beeswarm.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[fig2] {out}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 3 — Dependence plots top-4 features (2×2)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_dependence(shap_values, X_s, feats):
    print("[fig3] Dependence plots ...")
    top4 = ['aww', 'has_C3', 'days_C3_to_accident', 'wcio_nature_code']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, feat in zip(axes, top4):
        if feat not in feats:
            ax.set_visible(False)
            continue
        fi  = feats.index(feat)
        x_v = X_s.iloc[:, fi].values
        s_v = shap_values[:, fi]

        # Intentar color por feature de interacción natural
        interact = {
            'aww'                 : 'has_C3',
            'has_C3'              : 'aww',
            'days_C3_to_accident' : 'has_C3',
            'wcio_nature_code'    : 'aww',
        }
        ci  = feats.index(interact[feat]) if interact[feat] in feats else None
        c_v = X_s.iloc[:, ci].values if ci is not None else None

        # Para categóricas: convertir a numérico por frecuencia de judicialización
        if feat in CAT_FEATURES:
            cats  = np.unique(x_v)
            c_map = {c: s_v[x_v == c].mean() for c in cats}
            x_num = np.array([c_map[v] for v in x_v])
        else:
            x_num = x_v.astype(float)
            x_num[x_num < 0] = np.nan   # -1 era fillna

        if c_v is not None and feat not in CAT_FEATURES:
            sc = ax.scatter(x_num, s_v, c=c_v.astype(float),
                            cmap="RdYlBu_r", alpha=0.35, s=12,
                            vmin=np.nanpercentile(c_v.astype(float), 5),
                            vmax=np.nanpercentile(c_v.astype(float), 95))
            plt.colorbar(sc, ax=ax,
                         label=DISPLAY_NAMES.get(interact[feat], interact[feat]),
                         shrink=0.8)
        else:
            ax.scatter(x_num, s_v, alpha=0.3, s=10, color=PALETTE['dark'])

        ax.axhline(0, color="#888", lw=1, ls="--", alpha=0.6)
        ax.set_xlabel(DISPLAY_NAMES.get(feat, feat), fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.set_title(f"Efecto de: {DISPLAY_NAMES.get(feat, feat)}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Dependence Plots — Top 4 Variables Predictivas\n"
                 "(SHAP value vs. valor de la feature — color = feature de interacción)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "fig3_shap_dependence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig3] {out}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 4 — Waterfall para 3 casos representativos
# ══════════════════════════════════════════════════════════════════════════════

def fig4_waterfall(shap_values, X_s, y_s, feats, expected, model):
    print("[fig4] Waterfall 3 casos ...")
    # Obtener probabilidades predichas
    probs = model.predict_proba(X_s)[:, 1]
    display = [DISPLAY_NAMES.get(f, f) for f in feats]

    # Seleccionar 3 casos representativos
    pos_idx   = np.where(y_s == 1)[0]
    neg_idx   = np.where(y_s == 0)[0]

    # Alto riesgo: positivo con prob > 0.85
    high_cands = pos_idx[probs[pos_idx] > 0.85]
    high_cands = high_cands if len(high_cands) > 0 else pos_idx[np.argsort(probs[pos_idx])[-1:]]
    idx_high   = high_cands[np.argmax(probs[high_cands])]

    # Bajo riesgo: negativo con prob < 0.15
    low_cands  = neg_idx[probs[neg_idx] < 0.15]
    low_cands  = low_cands if len(low_cands) > 0 else neg_idx[np.argsort(probs[neg_idx])[:1]]
    idx_low    = low_cands[np.argmin(probs[low_cands])]

    # Riesgo medio: caso cerca del umbral 0.55
    mid_target = 0.55
    idx_mid    = np.argmin(np.abs(probs - mid_target))

    cases = [
        (idx_high, f"Caso de ALTO RIESGO  (prob={probs[idx_high]:.3f}, real={'Judicializa' if y_s[idx_high]==1 else 'No'})"),
        (idx_mid,  f"Caso de RIESGO MEDIO  (prob={probs[idx_mid]:.3f}, real={'Judicializa' if y_s[idx_mid]==1 else 'No'})"),
        (idx_low,  f"Caso de BAJO RIESGO  (prob={probs[idx_low]:.3f}, real={'Judicializa' if y_s[idx_low]==1 else 'No'})"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, (idx, title) in zip(axes, cases):
        sv   = shap_values[idx]
        # Top-10 contribuyentes por |SHAP|
        order     = np.argsort(np.abs(sv))[::-1][:10]
        sv_top    = sv[order]
        names_top = [display[i] for i in order]
        colors    = [PALETTE['pos'] if v > 0 else PALETTE['neg'] for v in sv_top]

        # Barras horizontales tipo waterfall
        ax.barh(names_top[::-1], sv_top[::-1], color=colors[::-1], edgecolor="white", height=0.7)
        ax.axvline(0, color="#333", lw=1)
        ax.set_xlabel("Contribución SHAP (log-odds)", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8, wrap=True)
        ax.grid(axis="x", alpha=0.25, ls="--")
        ax.spines[["top","right"]].set_visible(False)

        # Anotación de probabilidad final
        p = probs[idx]
        ax.text(0.98, 0.02, f"P(judicial)={p:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#222",
                bbox=dict(facecolor="lightyellow", alpha=0.8, boxstyle="round,pad=0.3"))

    fig.suptitle("Explicación Local (SHAP Waterfall) — 3 Casos Representativos del Test 2022\n"
                 "Contribuciones positivas (rojo) empujan hacia judicialización; negativas (azul) protegen",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig4_shap_waterfall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig4] {out}")

    # Guardar detalles de los 3 casos para el capítulo
    meta_cases = {}
    for idx, title in cases:
        row = X_s.iloc[idx]
        meta_cases[title] = {
            "prob"    : float(probs[idx]),
            "real"    : int(y_s[idx]),
            "top5_shap": [
                {"feature": display[i], "shap": float(shap_values[idx][i]),
                 "value": str(X_s.iloc[idx, i])}
                for i in np.argsort(np.abs(shap_values[idx]))[::-1][:5]
            ]
        }
    with open(OUT_DIR / "waterfall_cases.json", "w", encoding="utf-8") as f:
        json.dump(meta_cases, f, indent=2, ensure_ascii=False)
    print("[fig4] Casos guardados en waterfall_cases.json")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 5 — Distribución SHAP por segmento (AWW quartiles)
# ══════════════════════════════════════════════════════════════════════════════

def fig5_aww_segments(shap_values, X_s, y_s, feats):
    print("[fig5] AWW quartile segments ...")
    if 'aww' not in feats:
        print("[fig5] aww no encontrado, skip")
        return

    fi_aww   = feats.index('aww')
    fi_c3    = feats.index('has_C3') if 'has_C3' in feats else None
    aww_vals = X_s.iloc[:, fi_aww].values.astype(float)
    aww_vals[aww_vals < 0] = np.nan

    q25, q50, q75 = np.nanpercentile(aww_vals, [25, 50, 75])
    labels = [f'Q1\n(≤${q25:.0f})', f'Q2\n(${ q25:.0f}–${ q50:.0f})',
              f'Q3\n(${ q50:.0f}–${ q75:.0f})', f'Q4\n(>${ q75:.0f})']

    def quartile(v):
        if np.isnan(v): return -1
        if v <= q25:   return 0
        elif v <= q50: return 1
        elif v <= q75: return 2
        else:          return 3

    qts = np.array([quartile(v) for v in aww_vals])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: tasa de judicialización por cuartil de AWW
    ax1 = axes[0]
    rates = [y_s[qts == q].mean() * 100 for q in range(4)]
    colors = [PALETTE['neg'], PALETTE['neu'], PALETTE['gold'], PALETTE['pos']]
    bars = ax1.bar(labels, rates, color=colors, edgecolor="white", width=0.6)
    ax1.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    ax1.set_ylabel("Tasa de judicialización (%)", fontsize=10)
    ax1.set_title("Tasa de Judicialización por Cuartil de AWW", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(rates) * 1.25)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    # Panel 2: SHAP de AWW por cuartil
    ax2 = axes[1]
    shap_aww = shap_values[:, fi_aww]
    bp = ax2.boxplot(
        [shap_aww[qts == q] for q in range(4)],
        labels=labels, patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#888"),
        capprops=dict(color="#888"),
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax2.axhline(0, color="#555", lw=1, ls="--")
    ax2.set_ylabel("SHAP value de AWW", fontsize=10)
    ax2.set_title("Contribución SHAP de AWW por Cuartil", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("Impacto del Salario Semanal Promedio (AWW) en la Probabilidad de Judicialización",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig5_shap_aww_segments.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig5] {out}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 6 — Interacción AWW × has_C3
# ══════════════════════════════════════════════════════════════════════════════

def fig6_interaction(shap_values, X_s, y_s, feats):
    print("[fig6] AWW × has_C3 interaction ...")
    if 'aww' not in feats or 'has_C3' not in feats:
        print("[fig6] features no encontradas, skip")
        return

    fi_aww = feats.index('aww')
    fi_c3  = feats.index('has_C3')
    aww    = X_s.iloc[:, fi_aww].values.astype(float)
    c3     = X_s.iloc[:, fi_c3].values.astype(float)
    shap_aww = shap_values[:, fi_aww]
    shap_c3  = shap_values[:, fi_c3]
    probs_approx = shap_values.sum(axis=1)  # log-odds approx

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: AWW vs P(judicial) coloreado por has_C3
    ax1 = axes[0]
    mask_c3_yes = c3 > 0.5
    mask_c3_no  = c3 <= 0.5
    aww_plot = aww.copy(); aww_plot[aww_plot < 0] = np.nan
    # Limitar a percentil 99 para legibilidad
    aww_clip = np.nanpercentile(aww_plot, 99)
    m_yes = mask_c3_yes & (aww_plot < aww_clip)
    m_no  = mask_c3_no  & (aww_plot < aww_clip)
    ax1.scatter(aww_plot[m_no],  probs_approx[m_no],  alpha=0.2, s=8,
                color=PALETTE['neg'],  label='Sin C-3')
    ax1.scatter(aww_plot[m_yes], probs_approx[m_yes], alpha=0.2, s=8,
                color=PALETTE['pos'],  label='Con C-3')
    ax1.axhline(0, color="#888", ls="--", lw=1)
    ax1.set_xlabel("AWW (Salario Semanal Promedio, USD)", fontsize=10)
    ax1.set_ylabel("Suma SHAP (log-odds hacia judicialización)", fontsize=10)
    ax1.set_title("AWW × Formulario C-3\nImpacto combinado sobre la probabilidad", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.2)
    ax1.spines[["top","right"]].set_visible(False)

    # Panel 2: contribuciones promedio por grupo
    ax2 = axes[1]
    groups = ['AWW bajo\nSin C-3', 'AWW bajo\nCon C-3',
              'AWW alto\nSin C-3', 'AWW alto\nCon C-3']
    aww_med = np.nanmedian(aww_plot)
    masks_g = [
        (aww_plot < aww_med) & mask_c3_no,
        (aww_plot < aww_med) & mask_c3_yes,
        (aww_plot >= aww_med) & mask_c3_no,
        (aww_plot >= aww_med) & mask_c3_yes,
    ]
    colors_g = [PALETTE['neg'], PALETTE['gold'], PALETTE['gold'], PALETTE['pos']]
    rates_g  = [y_s[m].mean()*100 if m.sum() > 0 else 0.0 for m in masks_g]
    bars2 = ax2.bar(groups, rates_g, color=colors_g, edgecolor="white", width=0.6)
    ax2.bar_label(bars2, fmt="%.1f%%", padding=4, fontsize=9)
    ax2.set_ylabel("Tasa real de judicialización (%)", fontsize=10)
    ax2.set_title("Tasa de Judicialización por\nCombinación AWW × C-3", fontsize=10, fontweight="bold")
    max_rate = max(rates_g) if max(rates_g) > 0 else 1.0
    ax2.set_ylim(0, max_rate * 1.3)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("Interacción AWW × Formulario C-3: el efecto conjunto explica la mayor parte\n"
                 "de la separación entre casos que judicializan y los que no",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig6_shap_interaction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig6] {out}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  SHAP Interpretability — CatBoost Tuneado (Cap. 7)")
    print("=" * 65)

    X_te, y_te, feats = load_test()
    model             = load_model()

    shap_values, X_s, y_s, expected, explainer = compute_shap(model, X_te, y_te, feats)

    fig1_bar(shap_values, X_s, feats)
    fig2_beeswarm(shap_values, X_s, feats)
    fig3_dependence(shap_values, X_s, feats)
    fig4_waterfall(shap_values, X_s, y_s, feats, expected, model)
    fig5_aww_segments(shap_values, X_s, y_s, feats)
    fig6_interaction(shap_values, X_s, y_s, feats)

    print("\n" + "=" * 65)
    print("  SHAP COMPLETADO")
    print(f"  Figuras guardadas en: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("fig*.png")):
        print(f"    {f.name}")
    print("=" * 65)

if __name__ == "__main__":
    main()
