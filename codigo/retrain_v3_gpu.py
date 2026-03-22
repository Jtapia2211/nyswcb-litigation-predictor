"""
retrain_v3_gpu.py — Reentrenamiento COMPLETO para GPU (Windows)
================================================================
Target v3: Litigio Evitable
  Positivo = litigó Y claim_injury_type ∈ {TEMPORARY, NON-COMP, MED ONLY}
  Negativo = no litigó  +  litigó con PPD/PTD/DEATH (inevitable)

Requisitos:
  pip install catboost pandas numpy
  NVIDIA GPU recomendada (pero corre en CPU si no hay GPU)

Ejecutar desde la carpeta raíz del proyecto:
  python retrain_v3_gpu.py

Genera:
  modelo: codigo/model_v3/catboost_v3_full.cbm
  resultados: codigo/model_v3/results_v3_full.json

Maestría en Ciencia de Datos — ITBA  |  Tapia, Julián
"""

import json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
# Ajusta BASE al directorio raíz de Tesis_ML en tu máquina
BASE      = Path(__file__).parent.parent          # sube un nivel desde /codigo
DATA_FILE = BASE / "raw_data" / "dataset_tesis_clean.csv"
OUT_DIR   = BASE / "codigo" / "model_v3"
OUT_DIR.mkdir(exist_ok=True)

# ─── Target v3 ───────────────────────────────────────────────────────────────
AVOIDABLE = {'2. NON-COMP', '3. MED ONLY', '4. TEMPORARY'}

# ─── Features (idénticas a v2) ────────────────────────────────────────────────
CAT_FEATURES = [
    "gender", "accident_type", "occupational_disease",
    "county_of_injury", "medical_fee_region",
    "wcio_cause_code", "wcio_nature_code", "wcio_body_code",
    "carrier_type", "district_name",
    "industry_code", "industry_desc",
]
NUM_FEATURES = [
    "days_to_assembly", "days_C2_to_accident", "days_C3_to_accident",
    "age_at_injury", "aww",
    "has_C2", "has_C3", "has_ANCR_early",
    "accident_year", "accident_month", "accident_dow",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# ─── Hiperparámetros óptimos (Optuna trial #22, Cap. 6) ──────────────────────
BEST_PARAMS = dict(
    iterations            = 1511,
    depth                 = 10,
    learning_rate         = 0.061,
    l2_leaf_reg           = 7.62,
    bagging_temperature   = 0.792,
    border_count          = 82,
    task_type             = "GPU",     # cambiar a "CPU" si no hay GPU
    loss_function         = "Logloss",
    eval_metric           = "PRAUC",
    scale_pos_weight      = 5.631,     # neg/pos del train set completo
    random_seed           = 42,
    verbose               = 100,
    early_stopping_rounds = 100,
    allow_writing_files   = False,
)

# ─── Métricas (pure numpy, sin sklearn) ──────────────────────────────────────
def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    pos = np.sum(y_true==1); neg = np.sum(y_true==0)
    if pos==0 or neg==0: return 0.5
    order = np.argsort(-y_score); y_s = y_true[order]
    tpr = np.cumsum(y_s)/pos; fpr = np.cumsum(1-y_s)/neg
    return float(np.trapz(tpr, fpr))

def pr_auc(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    order = np.argsort(-y_score); y_s = y_true[order]
    tp_c = np.cumsum(y_s); n = np.arange(1, len(y_s)+1)
    prec = np.concatenate([[1.], tp_c/n])
    rec  = np.concatenate([[0.], tp_c/y_true.sum()])
    return float(np.trapz(prec, rec))

def ks_stat(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    order = np.argsort(-y_score); y_s = y_true[order]
    pos = y_true.sum(); neg = len(y_true)-pos
    return float(np.max(np.cumsum(y_s)/pos - np.cumsum(1-y_s)/neg))

def best_f1_threshold(y_true, y_score):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 150):
        yp = (y_score>=t).astype(int)
        tp = ((yp==1)&(y_true==1)).sum()
        fp = ((yp==1)&(y_true==0)).sum()
        fn = ((yp==0)&(y_true==1)).sum()
        d = 2*tp+fp+fn; f1 = 2*tp/d if d>0 else 0.
        if f1>best_f1: best_f1,best_t = f1,t
    return best_t

def eval_metrics(y_true, y_score, tau):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    yp = (y_score>=tau).astype(int)
    tp=int(((yp==1)&(y_true==1)).sum()); fp=int(((yp==1)&(y_true==0)).sum())
    fn=int(((yp==0)&(y_true==1)).sum()); tn=int(((yp==0)&(y_true==0)).sum())
    prec=tp/(tp+fp) if tp+fp>0 else 0.; rec=tp/(tp+fn) if tp+fn>0 else 0.
    f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0.
    return dict(
        auc_roc   = round(roc_auc(y_true, y_score), 4),
        pr_auc    = round(pr_auc(y_true, y_score),  4),
        ks        = round(ks_stat(y_true, y_score),  4),
        brier     = round(float(np.mean((y_score-y_true)**2)), 4),
        precision = round(prec, 4),
        recall    = round(rec, 4),
        f1        = round(f1, 4),
        threshold = round(float(tau), 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )

# ─── Main ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("RETRAIN v3 — TARGET: LITIGIO EVITABLE (GPU, dataset completo)")
print("=" * 70)

t0 = time.time()

print(f"\n[1] Cargando dataset: {DATA_FILE}")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"    Total registros: {len(df):,}")

# ─── Nuevo target ─────────────────────────────────────────────────────────────
df['target'] = (
    df['target'].eq(1) &
    df['claim_injury_type_REF'].isin(AVOIDABLE)
).astype(int)

n_pos = int(df['target'].sum())
n_neg = len(df) - n_pos
print(f"    Positivos (litigio evitable): {n_pos:,} ({100*n_pos/len(df):.1f}%)")
print(f"    Negativos:                    {n_neg:,} ({100*n_neg/len(df):.1f}%)")
print(f"    scale_pos_weight:             {n_neg/n_pos:.3f}")

# ─── Preparación de features ──────────────────────────────────────────────────
for c in CAT_FEATURES:
    df[c] = df[c].fillna("UNKNOWN").astype(str)
for c in NUM_FEATURES:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1)

feats   = [c for c in ALL_FEATURES if c in df.columns]
cat_idx = [feats.index(c) for c in CAT_FEATURES if c in feats]
print(f"    Features: {len(feats)} ({len(cat_idx)} categóricas)")

# ─── Split temporal ───────────────────────────────────────────────────────────
train_df = df[df['accident_year'].isin([2017, 2018, 2019, 2020])]
val_df   = df[df['accident_year'] == 2021]
test_df  = df[df['accident_year'] == 2022]

print(f"\n[2] Splits temporales:")
print(f"    Train 2017-2020: {len(train_df):,}  pos={train_df['target'].sum():,} ({100*train_df['target'].mean():.1f}%)")
print(f"    Val  2021:       {len(val_df):,}   pos={val_df['target'].sum():,} ({100*val_df['target'].mean():.1f}%)")
print(f"    Test 2022:       {len(test_df):,}   pos={test_df['target'].sum():,} ({100*test_df['target'].mean():.1f}%)")

X_tr, y_tr   = train_df[feats].values, train_df['target'].values
X_val, y_val = val_df[feats].values,   val_df['target'].values
X_te, y_te   = test_df[feats].values,  test_df['target'].values

pool_tr  = Pool(X_tr,  y_tr,  cat_features=cat_idx, feature_names=feats)
pool_val = Pool(X_val, y_val, cat_features=cat_idx, feature_names=feats)
pool_te  = Pool(X_te,  y_te,  cat_features=cat_idx, feature_names=feats)

# ─── Entrenamiento ────────────────────────────────────────────────────────────
print(f"\n[3] Entrenando CatBoost (GPU, depth=10, max {BEST_PARAMS['iterations']} iter)...")
print(f"    (si no hay GPU disponible, cambia task_type='GPU' a 'CPU' en BEST_PARAMS)")

model = CatBoostClassifier(**BEST_PARAMS)
model.fit(pool_tr, eval_set=pool_val, use_best_model=True)

t_train = time.time() - t0
print(f"    Entrenamiento completado en {t_train/60:.1f} minutos")

# ─── Evaluación ───────────────────────────────────────────────────────────────
print("\n[4] Evaluando...")
val_scores  = model.predict_proba(pool_val)[:, 1]
test_scores = model.predict_proba(pool_te)[:, 1]

tau_val  = best_f1_threshold(y_val, val_scores)
tau_test = best_f1_threshold(y_te,  test_scores)

val_metrics  = eval_metrics(y_val, val_scores,  tau_val)
test_metrics = eval_metrics(y_te,  test_scores, tau_test)

print(f"\n  VAL 2021  (τ={tau_val:.4f}):  PR-AUC={val_metrics['pr_auc']}  AUC-ROC={val_metrics['auc_roc']}  KS={val_metrics['ks']}")
print(f"  TEST 2022 (τ={tau_test:.4f}): PR-AUC={test_metrics['pr_auc']} AUC-ROC={test_metrics['auc_roc']} KS={test_metrics['ks']}")
print(f"  TEST confusion: TP={test_metrics['tp']:,} FP={test_metrics['fp']:,} FN={test_metrics['fn']:,} TN={test_metrics['tn']:,}")
print(f"  TEST Precision={test_metrics['precision']:.4f}  Recall={test_metrics['recall']:.4f}  F1={test_metrics['f1']:.4f}")

# Importancia de features
feat_imp = model.get_feature_importance(pool_tr)
fi_sorted = sorted(zip(feats, feat_imp), key=lambda x: -x[1])
print("\n  Top-10 features por importancia:")
for name, imp in fi_sorted[:10]:
    print(f"    {name:40s}: {imp:.4f}")

# ─── Guardar ──────────────────────────────────────────────────────────────────
print("\n[5] Guardando artefactos...")
model_path = OUT_DIR / "catboost_v3_full.cbm"
model.save_model(str(model_path))

results = {
    "model_version":   "v3_full",
    "target_definition": "Litigated AND claim_injury_type in {TEMPORARY, NON-COMP, MED ONLY}",
    "n_positive": n_pos,
    "n_negative": n_neg,
    "prevalence": round(n_pos / len(df), 4),
    "scale_pos_weight": round(n_neg / n_pos, 3),
    "best_params": {k: v for k, v in BEST_PARAMS.items() if k != "task_type"},
    "best_params_note": "Optuna trial #22, Cap.6 — applied to v3 target",
    "val_2021":   val_metrics,
    "test_2022":  test_metrics,
    "features":   feats,
    "cat_features": CAT_FEATURES,
    "top10_feature_importance": fi_sorted[:10],
    "training_time_min": round(t_train / 60, 1),
}

results_path = OUT_DIR / "results_v3_full.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Modelo:     {model_path}")
print(f"✅ Resultados: {results_path}")
print(f"\n{'='*70}")
print("RESUMEN FINAL")
print(f"{'='*70}")
print(f"Target:         Litigio Evitable (TEMPORARY | NON-COMP | MED ONLY)")
print(f"Prevalencia:    {100*n_pos/len(df):.1f}%  ({n_pos:,} positivos / {len(df):,} total)")
print(f"PR-AUC  val/test:  {val_metrics['pr_auc']} / {test_metrics['pr_auc']}")
print(f"AUC-ROC val/test:  {val_metrics['auc_roc']} / {test_metrics['auc_roc']}")
print(f"KS      val/test:  {val_metrics['ks']} / {test_metrics['ks']}")
print(f"Test Precision: {100*test_metrics['precision']:.1f}%   Recall: {100*test_metrics['recall']:.1f}%   F1: {test_metrics['f1']:.4f}")
print(f"Test TP: {test_metrics['tp']:,}  FP: {test_metrics['fp']:,}  FN: {test_metrics['fn']:,}  TN: {test_metrics['tn']:,}")
print(f"Tiempo de entrenamiento: {t_train/60:.1f} min")
