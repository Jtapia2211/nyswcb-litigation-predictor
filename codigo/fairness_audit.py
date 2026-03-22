"""
fairness_audit.py — Auditoría de equidad del modelo CatBoost v3
================================================================
Métricas requeridas por el tribunal (EO 13985 / NYWCB):
  - Equal Opportunity:  TPR igual por grupo protegido
  - Predictive Parity:  Precision igual por grupo protegido

Grupos analizados:
  1. Género (M / F) — variable directa en el dataset
  2. Quintiles de AWW — proxy de nivel socioeconómico

Umbral de producción: τ = 0.708 (máximo F1 sobre validación 2021, §5.5.2)

EEOC 4/5 rule: disparidad significativa si ratio < 0.80 o > 1.25
"""

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
DATA_FILE  = BASE / "raw_data" / "dataset_tesis_clean.csv"
MODEL_FILE = BASE / "codigo" / "model_v3" / "catboost_v3_full.cbm"
OUT_FILE   = BASE / "codigo" / "model_v3" / "fairness_audit_results.json"

TAU = 0.708   # umbral de producción

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

# ─── Métricas de equidad ──────────────────────────────────────────────────────
def fairness_metrics(y_true, y_pred, label=""):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred==1) & (y_true==1)).sum())
    fp = int(((y_pred==1) & (y_true==0)).sum())
    fn = int(((y_pred==0) & (y_true==1)).sum())
    tn = int(((y_pred==0) & (y_true==0)).sum())
    n  = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    tpr       = tp / (tp + fn) if (tp + fn) > 0 else float('nan')   # recall / Equal Opportunity
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')   # Predictive Parity
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else float('nan')   # False Positive Rate
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else float('nan')   # Miss rate
    fdr       = fp / (fp + tp) if (fp + tp) > 0 else float('nan')   # False Discovery Rate
    prev      = n_pos / n if n > 0 else float('nan')                 # prevalencia real

    return {
        "group":       label,
        "n":           n,
        "n_pos":       n_pos,
        "prevalence":  round(prev,   4),
        "TPR":         round(tpr,    4),   # Equal Opportunity metric
        "Precision":   round(precision, 4), # Predictive Parity metric
        "FPR":         round(fpr,    4),
        "FNR":         round(fnr,    4),
        "FDR":         round(fdr,    4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }

def disparity_ratio(val_a, val_b):
    """Ratio val_b / val_a. EEOC 4/5 rule: problema si < 0.80 o > 1.25"""
    if val_a == 0 or np.isnan(val_a) or np.isnan(val_b):
        return float('nan')
    return round(val_b / val_a, 4)

def flag(ratio, metric_name):
    if np.isnan(ratio): return "N/A"
    if ratio < 0.80:    return f"⚠ DISPARIDAD ({ratio:.2f} < 0.80)"
    if ratio > 1.25:    return f"⚠ DISPARIDAD ({ratio:.2f} > 1.25)"
    return f"OK ({ratio:.2f})"

# ─── [1] Carga ────────────────────────────────────────────────────────────────
print("=" * 65)
print("AUDITORÍA DE EQUIDAD — CatBoost v3 | Test 2022 | τ=0.708")
print("=" * 65)

print(f"\n[1] Cargando datos y modelo ...")
df = pd.read_csv(DATA_FILE, low_memory=False)

df['target'] = (
    df['target'].eq(1) &
    df['claim_injury_type_REF'].isin(AVOIDABLE)
).astype(int)

for c in CAT_FEATURES:
    df[c] = df[c].fillna("UNKNOWN").astype(str)
for c in NUM_FEATURES:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1)

test = df[df['accident_year'] == 2022].reset_index(drop=True)
feats   = [c for c in ALL_FEATURES if c in test.columns]
cat_idx = [feats.index(c) for c in CAT_FEATURES if c in feats]

model = CatBoostClassifier()
model.load_model(str(MODEL_FILE))
print(f"   Modelo cargado: {MODEL_FILE.name}")
print(f"   Test 2022: {len(test):,} reclamos  |  positivos: {test['target'].sum():,} ({100*test['target'].mean():.1f}%)")

# ─── [2] Predicciones ─────────────────────────────────────────────────────────
print(f"\n[2] Generando predicciones (τ={TAU}) ...")
pool_test = Pool(test[feats], cat_features=cat_idx)
proba     = model.predict_proba(pool_test)[:, 1]
pred      = (proba >= TAU).astype(int)
test      = test.copy()
test['score'] = proba
test['pred']  = pred

# ─── [3] Auditoría por GÉNERO ─────────────────────────────────────────────────
print(f"\n[3] Equidad por GÉNERO ...")
print("-" * 65)

gender_results = {}
for g in ['M', 'F']:
    mask = test['gender'] == g
    m    = fairness_metrics(test.loc[mask, 'target'], test.loc[mask, 'pred'], label=g)
    gender_results[g] = m
    print(f"   {g}  n={m['n']:>7,}  prev={m['prevalence']:.3f}  "
          f"TPR={m['TPR']:.4f}  Prec={m['Precision']:.4f}  FPR={m['FPR']:.4f}  FNR={m['FNR']:.4f}")

# Reference group: M (majority)
r_tpr  = disparity_ratio(gender_results['M']['TPR'],       gender_results['F']['TPR'])
r_prec = disparity_ratio(gender_results['M']['Precision'], gender_results['F']['Precision'])
r_fpr  = disparity_ratio(gender_results['M']['FPR'],       gender_results['F']['FPR'])

print(f"\n   Disparidad F vs M:")
print(f"   Equal Opportunity  (TPR_F / TPR_M)   = {r_tpr:.4f}  → {flag(r_tpr,  'TPR')}")
print(f"   Predictive Parity  (Prec_F / Prec_M) = {r_prec:.4f}  → {flag(r_prec, 'Precision')}")
print(f"   FPR ratio          (FPR_F / FPR_M)   = {r_fpr:.4f}  → {flag(r_fpr,  'FPR')}")

# ─── [4] Auditoría por QUINTILES AWW ─────────────────────────────────────────
print(f"\n[4] Equidad por QUINTILES de AWW (proxy socioeconómico) ...")
print("-" * 65)

# Solo reclamos con AWW conocido (> 0)
test_aww = test[test['aww'] > 0].copy()
test_aww['aww_q'] = pd.qcut(test_aww['aww'], q=5,
                             labels=['Q1 (bajo)', 'Q2', 'Q3', 'Q4', 'Q5 (alto)'])

# Calcular límites de quintiles
quintile_bounds = pd.qcut(test_aww['aww'], q=5, retbins=True)[1]

aww_results = {}
ref_q = None
for q_label in ['Q1 (bajo)', 'Q2', 'Q3', 'Q4', 'Q5 (alto)']:
    mask = test_aww['aww_q'] == q_label
    m    = fairness_metrics(test_aww.loc[mask, 'target'], test_aww.loc[mask, 'pred'], label=q_label)
    aww_results[q_label] = m
    if ref_q is None: ref_q = m   # Q1 como referencia
    print(f"   {q_label}  n={m['n']:>6,}  prev={m['prevalence']:.3f}  "
          f"TPR={m['TPR']:.4f}  Prec={m['Precision']:.4f}")

print(f"\n   Quintiles AWW (límites): {[round(b,0) for b in quintile_bounds]}")
print(f"\n   Disparidad Q5 vs Q1 (referencia = Q1):")
r5_tpr  = disparity_ratio(ref_q['TPR'],       aww_results['Q5 (alto)']['TPR'])
r5_prec = disparity_ratio(ref_q['Precision'], aww_results['Q5 (alto)']['Precision'])
print(f"   Equal Opportunity (TPR_Q5 / TPR_Q1)   = {r5_tpr:.4f}  → {flag(r5_tpr,  'TPR')}")
print(f"   Predictive Parity (Prec_Q5 / Prec_Q1) = {r5_prec:.4f}  → {flag(r5_prec, 'Precision')}")

# ─── [5] Interseccionalidad: género × AWW alto/bajo ─────────────────────────
print(f"\n[5] Interseccionalidad: género × segmento AWW ...")
print("-" * 65)
aww_median = test_aww['aww'].median()
test_aww['aww_seg'] = (test_aww['aww'] >= aww_median).map({True: 'AWW_alto', False: 'AWW_bajo'})

intersect_results = {}
for g in ['M', 'F']:
    for seg in ['AWW_bajo', 'AWW_alto']:
        mask  = (test_aww['gender'] == g) & (test_aww['aww_seg'] == seg)
        label = f"{g}_{seg}"
        m     = fairness_metrics(test_aww.loc[mask, 'target'], test_aww.loc[mask, 'pred'], label=label)
        intersect_results[label] = m
        print(f"   {label:<15}  n={m['n']:>6,}  prev={m['prevalence']:.3f}  "
              f"TPR={m['TPR']:.4f}  Prec={m['Precision']:.4f}")

# ─── [6] Resumen ejecutivo ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("RESUMEN EJECUTIVO")
print(f"{'='*65}")

issues = []
if r_tpr  < 0.80 or r_tpr  > 1.25: issues.append(f"TPR género (ratio={r_tpr:.3f})")
if r_prec < 0.80 or r_prec > 1.25: issues.append(f"Precision género (ratio={r_prec:.3f})")
if r5_tpr  < 0.80 or r5_tpr  > 1.25: issues.append(f"TPR AWW Q5/Q1 (ratio={r5_tpr:.3f})")
if r5_prec < 0.80 or r5_prec > 1.25: issues.append(f"Precision AWW Q5/Q1 (ratio={r5_prec:.3f})")

if issues:
    print(f"\n  ⚠  DISPARIDADES DETECTADAS (EEOC 4/5 rule):")
    for iss in issues: print(f"     - {iss}")
else:
    print(f"\n  ✓  Ninguna disparidad supera el umbral EEOC (4/5 rule) en las métricas primarias.")

print(f"\n  Métricas primarias (Equal Opportunity + Predictive Parity):")
print(f"  {'Grupo':<18} {'TPR':>7} {'Precision':>10} {'FNR':>7} {'n_pos':>7}")
print(f"  {'-'*52}")
for g in ['M', 'F']:
    m = gender_results[g]
    print(f"  {'Género '+g:<18} {m['TPR']:>7.4f} {m['Precision']:>10.4f} {m['FNR']:>7.4f} {m['n_pos']:>7,}")
print(f"  {'-'*52}")
for q in ['Q1 (bajo)', 'Q2', 'Q3', 'Q4', 'Q5 (alto)']:
    m = aww_results[q]
    print(f"  {'AWW '+q:<18} {m['TPR']:>7.4f} {m['Precision']:>10.4f} {m['FNR']:>7.4f} {m['n_pos']:>7,}")

# ─── [7] Guardar ──────────────────────────────────────────────────────────────
results = {
    "model":        "catboost_v3_full.cbm",
    "threshold":    TAU,
    "test_year":    2022,
    "test_n":       len(test),
    "eeoc_rule":    "4/5: ratio < 0.80 o > 1.25 indica disparidad",
    "disparidades_detectadas": issues,
    "gender": {
        "referencia": "M",
        "grupos":     {g: gender_results[g] for g in ['M', 'F']},
        "ratios_F_vs_M": {
            "Equal_Opportunity_TPR":  r_tpr,
            "Predictive_Parity_Prec": r_prec,
            "FPR_ratio":              r_fpr,
        },
        "flags": {
            "Equal_Opportunity":  flag(r_tpr,  "TPR"),
            "Predictive_Parity":  flag(r_prec, "Precision"),
        }
    },
    "aww_quintiles": {
        "n_con_aww_conocido":  len(test_aww),
        "mediana_aww":         round(float(aww_median), 2),
        "limites_quintiles":   [round(float(b), 2) for b in quintile_bounds],
        "grupos":              aww_results,
        "ratios_Q5_vs_Q1": {
            "Equal_Opportunity_TPR":  r5_tpr,
            "Predictive_Parity_Prec": r5_prec,
        },
        "flags": {
            "Equal_Opportunity":  flag(r5_tpr,  "TPR"),
            "Predictive_Parity":  flag(r5_prec, "Precision"),
        }
    },
    "interseccionalidad": intersect_results,
}

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Resultados guardados: {OUT_FILE}")
print(f"{'='*65}")
