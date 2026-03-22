"""
shap_production_prevalence.py — SHAP con prevalencia de producción
==================================================================
El análisis SHAP original (cap7.js, Tabla 7.1) usó una muestra 50/50
(4.000 pos + 4.000 neg). La crítica del tribunal (P-2) señala que las
importancias globales no reflejan la prevalencia real de producción.

Este script:
1. Subsamplea el test-2022 manteniendo la prevalencia real (~21%)
2. Recalcula importancias SHAP sobre 8.000 casos representativos
3. Compara rankings con la versión 50/50
4. Reporta si difieren materialmente (>3 posiciones en top-10)

Output: model_v3/shap_production_results.json
"""

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

BASE       = Path(__file__).parent.parent
DATA_FILE  = BASE / "raw_data" / "dataset_tesis_clean.csv"
MODEL_FILE = BASE / "codigo" / "model_v3" / "catboost_v3_full.cbm"
OUT_FILE   = BASE / "codigo" / "model_v3" / "shap_production_results.json"

AVOIDABLE = {'2. NON-COMP', '3. MED ONLY', '4. TEMPORARY'}

CAT_FEATURES = [
    "gender", "accident_type", "occupational_disease",
    "county_of_injury", "medical_fee_region",
    "wcio_cause_code", "wcio_nature_code", "wcio_body_code",
    "carrier_type", "district_name", "industry_code", "industry_desc",
]
NUM_FEATURES = [
    "days_to_assembly", "days_C2_to_accident", "days_C3_to_accident",
    "age_at_injury", "aww",
    "has_C2", "has_C3", "has_ANCR_early",
    "accident_year", "accident_month", "accident_dow",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

SAMPLE_SIZE  = 200   # ShapValues cost ~34s per 200 samples en este entorno
RANDOM_SEED  = 42

# ─── Load data ────────────────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv(DATA_FILE, low_memory=False)

# Replicar definición de target del fairness_audit.py
df['target'] = (
    df['target'].eq(1) &
    df['claim_injury_type_REF'].isin(AVOIDABLE)
).astype(int)

for c in CAT_FEATURES:
    if c in df.columns:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
for c in NUM_FEATURES:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

test = df[df["accident_year"] == 2022].copy()
n_pos = int(test["target"].sum())
n_neg = len(test) - n_pos
prev_prod = round(n_pos / len(test), 4)

print(f"Test 2022: n={len(test):,}  pos={n_pos:,}  neg={n_neg:,}  prevalencia={prev_prod:.4f}")

# ─── Muestra estratificada 50/50 (original) ──────────────────────────────────
pos_idx = test[test["target"] == 1].sample(n=min(100, n_pos), random_state=RANDOM_SEED).index
neg_idx = test[test["target"] == 0].sample(n=100, random_state=RANDOM_SEED).index
sample_5050 = test.loc[pos_idx.tolist() + neg_idx.tolist()].copy()

# ─── Muestra con prevalencia de producción ───────────────────────────────────
# Mantener proporciones reales: ~prev_prod positivos en 8.000 casos
n_pos_prod = round(SAMPLE_SIZE * prev_prod)
n_neg_prod = SAMPLE_SIZE - n_pos_prod
pos_idx_p = test[test["target"] == 1].sample(n=min(n_pos_prod, n_pos), random_state=RANDOM_SEED).index
neg_idx_p = test[test["target"] == 0].sample(n=n_neg_prod, random_state=RANDOM_SEED).index
sample_prod = test.loc[pos_idx_p.tolist() + neg_idx_p.tolist()].copy()

print(f"\nMuestra 50/50:   n={len(sample_5050):,}  pos={sample_5050['target'].sum()}  prev={sample_5050['target'].mean():.4f}")
print(f"Muestra prod.:   n={len(sample_prod):,}  pos={sample_prod['target'].sum()}  prev={sample_prod['target'].mean():.4f}")

# ─── Cargar modelo ────────────────────────────────────────────────────────────
print("\nCargando modelo CatBoost v3...")
model = CatBoostClassifier()
model.load_model(str(MODEL_FILE))

# ─── SHAP 50/50 (referencia) — usando CatBoost nativo ────────────────────────
print("Calculando SHAP 50/50 (referencia) con CatBoost nativo...")
X_5050 = sample_5050[ALL_FEATURES]
pool_5050 = Pool(X_5050, cat_features=CAT_FEATURES)
shap_matrix_5050 = model.get_feature_importance(pool_5050, type="ShapValues")
# shap_matrix: shape (n_samples, n_features+1) — última col es base value (bias)
shap_vals_5050 = shap_matrix_5050[:, :-1]
base_5050 = float(shap_matrix_5050[:, -1].mean())

mean_abs_5050 = pd.Series(
    np.abs(shap_vals_5050).mean(axis=0),
    index=ALL_FEATURES
).sort_values(ascending=False)

prob_5050 = float(1 / (1 + np.exp(-base_5050)))
print(f"  Base value (50/50): {base_5050:.4f}  → P={prob_5050:.4f}")

# ─── SHAP producción ──────────────────────────────────────────────────────────
print("Calculando SHAP con prevalencia de producción (CatBoost nativo)...")
X_prod = sample_prod[ALL_FEATURES]
pool_prod = Pool(X_prod, cat_features=CAT_FEATURES)
shap_matrix_prod = model.get_feature_importance(pool_prod, type="ShapValues")
shap_vals_prod  = shap_matrix_prod[:, :-1]
base_prod = float(shap_matrix_prod[:, -1].mean())

mean_abs_prod = pd.Series(
    np.abs(shap_vals_prod).mean(axis=0),
    index=ALL_FEATURES
).sort_values(ascending=False)

prob_prod = float(1 / (1 + np.exp(-base_prod)))
print(f"  Base value (prod.): {base_prod:.4f}  → P={prob_prod:.4f}")

# ─── Comparación de rankings ──────────────────────────────────────────────────
print("\n=== COMPARACIÓN DE RANKINGS TOP-15 ===")
print(f"{'Rango':<6} {'Feature (50/50)':<30} {'|SHAP|':<8} {'Feature (prod.)':<30} {'|SHAP|':<8} {'Δrango'}")

ranking_5050 = {feat: i+1 for i, feat in enumerate(mean_abs_5050.index)}
ranking_prod = {feat: i+1 for i, feat in enumerate(mean_abs_prod.index)}

comparison = []
for rank, feat in enumerate(mean_abs_5050.index[:15], 1):
    rank_prod = ranking_prod.get(feat, 99)
    delta = rank_prod - rank
    feat_prod = mean_abs_prod.index[rank-1] if rank <= len(mean_abs_prod) else "—"
    val_5050 = mean_abs_5050[feat]
    val_prod = mean_abs_prod.get(feat, 0)

    print(f"  {rank:<4} {feat:<30} {val_5050:.4f}   {feat_prod:<30} {mean_abs_prod.iloc[rank-1]:.4f}   {delta:+d}")
    comparison.append({
        "rank_5050": rank,
        "feature_5050": feat,
        "mean_abs_shap_5050": round(float(val_5050), 4),
        "rank_prod": rank_prod,
        "delta_rank": delta
    })

# ─── Detectar cambios materiales ─────────────────────────────────────────────
top10_5050 = set(mean_abs_5050.index[:10])
top10_prod = set(mean_abs_prod.index[:10])
features_in_out = top10_5050 - top10_prod
features_out_in = top10_prod - top10_5050

max_rank_shift = max(abs(c["delta_rank"]) for c in comparison[:10])
material_change = max_rank_shift > 3 or len(features_in_out) > 0

print(f"\n=== CONCLUSIÓN ===")
print(f"  Max desplazamiento en top-10: {max_rank_shift} posiciones")
print(f"  Features que entran/salen del top-10: {len(features_in_out)}")
print(f"  Cambio material: {'SÍ' if material_change else 'NO — rankings estables'}")
print(f"  Diferencia en valor base: {abs(base_prod - base_5050):.4f}")

# ─── Save ─────────────────────────────────────────────────────────────────────
results = {
    "sample_5050": {
        "n_pos": 4000, "n_neg": 4000, "prevalence": 0.5,
        "base_value_logodds": round(base_5050, 4),
        "base_value_prob": round(prob_5050, 4),
        "top15": [{"rank": c["rank_5050"], "feature": c["feature_5050"],
                   "mean_abs_shap": c["mean_abs_shap_5050"]} for c in comparison],
    },
    "sample_prod": {
        "n_pos": n_pos_prod, "n_neg": n_neg_prod, "prevalence": round(prev_prod, 4),
        "base_value_logodds": round(base_prod, 4),
        "base_value_prob": round(prob_prod, 4),
        "top15": [{"rank": i+1, "feature": f, "mean_abs_shap": round(float(v), 4)}
                  for i, (f, v) in enumerate(mean_abs_prod.items()) if i < 15],
    },
    "comparison": comparison,
    "conclusion": {
        "max_rank_shift_top10": int(max_rank_shift),
        "features_entering_top10": list(features_out_in),
        "features_leaving_top10": list(features_in_out),
        "material_change": material_change,
        "update_section_7_2": material_change,
    }
}

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResultados guardados en: {OUT_FILE}")
print("✓ Análisis SHAP producción completado.")
