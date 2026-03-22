"""
fairness_calibration.py — Calibración diferenciada del umbral por quintil de AWW
==================================================================================
Estrategia: umbral τ_q específico por quintil.
  1. Usar pd.qcut en CADA split (val / test) por separado para quintiles iguales
     (consistente con fairness_audit.py).
  2. Buscar τ en val-2021 tal que TPR_q / TPR_Q1 ≥ 0.80 (EEOC mínimo),
     maximizando F1 dentro del rango que cumple esa restricción.
  3. Evaluar en test-2022 con esos umbrales.
  4. Para Q1, la búsqueda es libre (maximizar F1).

Output: model_v3/calibration_results.json
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
OUT_FILE   = BASE / "codigo" / "model_v3" / "calibration_results.json"

TAU_GLOBAL = 0.708
EEOC_RATIO = 0.80   # mínimo ratio Q5/Q1 requerido

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

# ─── Helpers ──────────────────────────────────────────────────────────────────
def metrics_at_tau(y_true, proba, tau):
    y_pred = (proba >= tau).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * tpr * prec / (tpr + prec) if (tpr + prec) > 0 else 0.0
    return {"TPR": round(tpr,4), "Precision": round(prec,4), "F1": round(f1,4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}

def best_tau_for_eeoc(y_true, proba, tpr_q1, eeoc_ratio,
                      tau_range=None, label=""):
    """
    Encuentra τ que maximiza F1 con restricción: TPR / tpr_q1 >= eeoc_ratio.
    Si no hay τ que cumpla la restricción, elige el que maximiza el ratio TPR.
    """
    if tau_range is None:
        tau_range = np.arange(0.30, 0.75, 0.005)
    target_tpr = tpr_q1 * eeoc_ratio
    best_tau, best_f1, best_ratio = None, -1, 0
    candidates = []
    for t in tau_range:
        m = metrics_at_tau(y_true, proba, t)
        ratio = m["TPR"] / tpr_q1 if tpr_q1 > 0 else 0
        if ratio >= eeoc_ratio:
            candidates.append((t, m["F1"], ratio))
    if candidates:
        # Entre candidatos que cumplen EEOC, maximizar F1
        candidates.sort(key=lambda x: -x[1])
        best_tau, best_f1, best_ratio = candidates[0]
        strategy = "EEOC + max F1"
    else:
        # No hay τ que cumpla: elige el de mayor TPR (τ mínimo en rango)
        max_tpr, best_tau = 0, tau_range[0]
        for t in tau_range:
            m = metrics_at_tau(y_true, proba, t)
            if m["TPR"] > max_tpr:
                max_tpr = m["TPR"]
                best_tau = t
        best_ratio = max_tpr / tpr_q1
        strategy = "max TPR (EEOC no alcanzable)"
    return float(best_tau), strategy, best_ratio

# ─── Load data ────────────────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv(DATA_FILE, low_memory=False)

AVOIDABLE = {'2. NON-COMP', '3. MED ONLY', '4. TEMPORARY'}

# Replicar exactamente la definición de target del fairness_audit.py
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

df_val  = df[df["accident_year"] == 2021].copy()
df_test = df[df["accident_year"] == 2022].copy()
print(f"Val 2021:  {len(df_val):,}  |  Test 2022: {len(df_test):,}")

# ─── Load model & predict ─────────────────────────────────────────────────────
print("Cargando modelo y calculando probabilidades...")
model = CatBoostClassifier()
model.load_model(str(MODEL_FILE))

pool_val  = Pool(df_val[ALL_FEATURES],  cat_features=CAT_FEATURES)
pool_test = Pool(df_test[ALL_FEATURES], cat_features=CAT_FEATURES)

df_val["proba"]  = model.predict_proba(pool_val)[:, 1]
df_test["proba"] = model.predict_proba(pool_test)[:, 1]

# ─── Quintiles AWW (qcut igual que fairness_audit.py) ────────────────────────
# Solo registros con AWW > 0
Q_LABELS = ["Q1 (bajo)", "Q2", "Q3", "Q4", "Q5 (alto)"]

df_val_aww  = df_val[df_val["aww"] > 0].copy()
df_test_aww = df_test[df_test["aww"] > 0].copy()

# qcut en cada split por separado (misma lógica que fairness_audit)
df_val_aww["quintile"], val_bins = pd.qcut(
    df_val_aww["aww"], q=5, labels=Q_LABELS, retbins=True)
df_test_aww["quintile"], test_bins = pd.qcut(
    df_test_aww["aww"], q=5, labels=Q_LABELS, retbins=True, duplicates="drop")

print(f"\nVal AWW n={len(df_val_aww):,}  |  Test AWW n={len(df_test_aww):,}")
print(f"Val  quintile limits (truncadas): {[round(b,2) for b in val_bins.tolist()]}")
print(f"Test quintile limits (truncadas): {[round(b,2) for b in test_bins.tolist()]}")

# ─── Baseline con τ global ────────────────────────────────────────────────────
print("\n=== BASELINE TEST-2022 con τ=0.708 ===")
baseline = {}
for q in Q_LABELS:
    mask = df_test_aww["quintile"] == q
    m = metrics_at_tau(df_test_aww.loc[mask, "target"].values,
                       df_test_aww.loc[mask, "proba"].values, TAU_GLOBAL)
    baseline[q] = m
    print(f"  {q}: n={mask.sum():,}  TPR={m['TPR']:.4f}  Prec={m['Precision']:.4f}  F1={m['F1']:.4f}")

tpr_q1_baseline = baseline["Q1 (bajo)"]["TPR"]
ratio_before = round(baseline["Q5 (alto)"]["TPR"] / tpr_q1_baseline, 4)
print(f"\n  Ratio Q5/Q1 (baseline): {ratio_before}  "
      f"{'✓ EEOC OK' if ratio_before >= EEOC_RATIO else '⚠ EEOC VIOLACIÓN'}")

# ─── Calibración en VAL-2021 ─────────────────────────────────────────────────
print("\n=== CALIBRACIÓN EN VAL-2021 ===")
TAU_RANGE = np.arange(0.30, 0.75, 0.005)

# Q1 en val: obtener su TPR en función del umbral (para referencia)
mask_q1_val = df_val_aww["quintile"] == "Q1 (bajo)"
m_q1_val_global = metrics_at_tau(
    df_val_aww.loc[mask_q1_val, "target"].values,
    df_val_aww.loc[mask_q1_val, "proba"].values, TAU_GLOBAL)
tpr_q1_val = m_q1_val_global["TPR"]

calibrated_taus = {"Q1 (bajo)": TAU_GLOBAL}
cal_strategies = {"Q1 (bajo)": "sin cambio (referencia)"}

for q in Q_LABELS[1:]:  # Q2–Q5
    mask_val = df_val_aww["quintile"] == q
    y_q = df_val_aww.loc[mask_val, "target"].values
    p_q = df_val_aww.loc[mask_val, "proba"].values

    tau_q, strategy, ratio_q = best_tau_for_eeoc(
        y_q, p_q, tpr_q1_val, EEOC_RATIO, TAU_RANGE, label=q)
    m_val_q = metrics_at_tau(y_q, p_q, tau_q)
    calibrated_taus[q] = round(tau_q, 3)
    cal_strategies[q] = strategy
    print(f"  {q}: τ={tau_q:.3f} [{strategy}]  "
          f"TPR={m_val_q['TPR']:.4f}  Prec={m_val_q['Precision']:.4f}  "
          f"Ratio={ratio_q:.3f}")

print(f"\nUmbrales calibrados: {calibrated_taus}")

# ─── Evaluar TEST-2022 con τ calibrado ───────────────────────────────────────
print("\n=== RESULTADOS TEST-2022 con τ calibrado ===")
calibrated = {}
for q in Q_LABELS:
    tau_q = calibrated_taus[q]
    mask = df_test_aww["quintile"] == q
    m = metrics_at_tau(df_test_aww.loc[mask, "target"].values,
                       df_test_aww.loc[mask, "proba"].values, tau_q)
    calibrated[q] = {**m, "tau": tau_q}
    dt = m["TPR"]  - baseline[q]["TPR"]
    dp = m["Precision"] - baseline[q]["Precision"]
    print(f"  {q}: τ={tau_q:.3f}  TPR={m['TPR']:.4f} (Δ={dt:+.4f})  "
          f"Prec={m['Precision']:.4f} (Δ={dp:+.4f})  F1={m['F1']:.4f}")

tpr_q1_cal  = calibrated["Q1 (bajo)"]["TPR"]
tpr_q5_cal  = calibrated["Q5 (alto)"]["TPR"]
prec_q1_cal = calibrated["Q1 (bajo)"]["Precision"]
prec_q5_cal = calibrated["Q5 (alto)"]["Precision"]
ratio_tpr_after  = round(tpr_q5_cal / tpr_q1_cal, 4)
ratio_prec_after = round(prec_q5_cal / prec_q1_cal, 4)
eeoc_status = "✓ EEOC OK" if ratio_tpr_after >= EEOC_RATIO else "⚠ EEOC VIOLACIÓN"

print(f"\n  Ratio Q5/Q1 TPR post-calibración:   {ratio_tpr_after}  {eeoc_status}")
print(f"  Ratio Q5/Q1 Prec post-calibración:  {ratio_prec_after}")

# ─── Impacto global ───────────────────────────────────────────────────────────
y_all  = df_test_aww["target"].values
p_all  = df_test_aww["proba"].values
m_glob_before = metrics_at_tau(y_all, p_all, TAU_GLOBAL)

pred_after = np.zeros(len(df_test_aww), dtype=int)
for i, q in enumerate(df_test_aww["quintile"].values):
    tau_q = calibrated_taus[str(q)]
    pred_after[i] = 1 if df_test_aww["proba"].iloc[i] >= tau_q else 0

tp_a = int(((pred_after==1) & (y_all==1)).sum())
fp_a = int(((pred_after==1) & (y_all==0)).sum())
fn_a = int(((pred_after==0) & (y_all==1)).sum())
n_pos = int((y_all==1).sum())
m_glob_after = {
    "TPR":       round(tp_a / (tp_a + fn_a), 4) if (tp_a+fn_a) > 0 else 0,
    "Precision": round(tp_a / (tp_a + fp_a), 4) if (tp_a+fp_a) > 0 else 0,
}
m_glob_after["F1"] = round(
    2*m_glob_after["TPR"]*m_glob_after["Precision"] /
    (m_glob_after["TPR"] + m_glob_after["Precision"]), 4) \
    if (m_glob_after["TPR"] + m_glob_after["Precision"]) > 0 else 0

print(f"\n=== IMPACTO GLOBAL (AWW conocido, n={len(df_test_aww):,}) ===")
print(f"  Pre-calibración:  TPR={m_glob_before['TPR']:.4f}  "
      f"Prec={m_glob_before['Precision']:.4f}  F1={m_glob_before['F1']:.4f}")
print(f"  Post-calibración: TPR={m_glob_after['TPR']:.4f}  "
      f"Prec={m_glob_after['Precision']:.4f}  F1={m_glob_after['F1']:.4f}")

# ─── Save ─────────────────────────────────────────────────────────────────────
results = {
    "tau_global": TAU_GLOBAL,
    "eeoc_target_ratio": EEOC_RATIO,
    "calibrated_taus": calibrated_taus,
    "val_quintile_limits": [round(b,2) for b in val_bins.tolist()],
    "test_quintile_limits": [round(b,2) for b in test_bins.tolist()],
    "baseline":    {q: baseline[q]   for q in Q_LABELS},
    "calibrated":  {q: calibrated[q] for q in Q_LABELS},
    "ratios_before": {"TPR_Q5_Q1": ratio_before},
    "ratios_after":  {"TPR_Q5_Q1": ratio_tpr_after,
                      "Prec_Q5_Q1": ratio_prec_after,
                      "EEOC_status": eeoc_status},
    "global_impact": {
        "before": m_glob_before,
        "after":  m_glob_after,
        "delta_TPR":  round(m_glob_after["TPR"] - m_glob_before["TPR"], 4),
        "delta_Prec": round(m_glob_after["Precision"] - m_glob_before["Precision"], 4),
        "delta_F1":   round(m_glob_after["F1"] - m_glob_before["F1"], 4),
    }
}

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResultados guardados en: {OUT_FILE}")
print("✓ Calibración completada.")
