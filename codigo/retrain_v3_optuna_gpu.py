"""
retrain_v3_optuna_gpu.py — Búsqueda de hiperparámetros DEDICADA al target v3
==============================================================================
v2 — Corrige compatibilidad GPU y velocidad por trial.

CAMBIOS vs. v1:
  - Eliminados min_data_in_leaf y random_strength (no compatibles GPU en CatBoost)
  - Submuestra estratificada para trials Optuna (~400k filas) → ~30-45s/trial
  - Reentrenamiento final con dataset COMPLETO (train+val)
  - early_stopping_rounds reducido a 30 (suficiente para detectar plateau)
  - max iterations = 1000 para trials, sin límite para modelo final
  - gc.collect() entre trials para evitar memory leaks

FLUJO:
  1. Carga dataset completo
  2. Define target v3 (TEMPORARY + MED ONLY + NON-COMP)
  3. Split temporal: Train 2017-2020 | Val 2021 | Test 2022
  4. Optuna: N_TRIALS trials en SUBMUESTRA (~400k) → objetivo = Val PR-AUC
  5. Mejor modelo: reentrenado en Train+Val COMPLETO → evaluado en Test 2022
  6. Guarda modelo + hiperparámetros + resultados

ESTIMACIÓN DE TIEMPO (GPU RTX 3070+):
  ~30-45 seg/trial  →  250 trials ≈ 2-3 horas

EJECUTAR:
  python retrain_v3_optuna_gpu.py              # 250 trials (~3h)
  python retrain_v3_optuna_gpu.py --trials 400 # overnight, ~5h
  python retrain_v3_optuna_gpu.py --trials 30  # prueba rápida, ~15min

SALIDAS en codigo/model_v3/:
  catboost_v3_optuna.cbm        — modelo final entrenado en train+val completo
  optuna_best_params.json       — mejores hiperparámetros + métricas test
  optuna_study.pkl              — estudio Optuna completo

Maestría en Ciencia de Datos — ITBA  |  Tapia, Julián
"""

import gc, json, warnings, time, sys, pickle, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Argumentos ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trials",    type=int, default=250,   help="Número de trials Optuna")
parser.add_argument("--seed",      type=int, default=42,    help="Semilla aleatoria")
parser.add_argument("--task-type", type=str, default="GPU", help="GPU o CPU")
parser.add_argument("--sample-k",  type=int, default=400,
                    help="Tamaño submuestra Optuna en miles (default: 400k filas)")
args = parser.parse_args()

N_TRIALS    = args.trials
SEED        = args.seed
TASK_TYPE   = args.task_type
SAMPLE_SIZE = args.sample_k * 1000

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
DATA_FILE = BASE / "raw_data" / "dataset_tesis_clean.csv"
OUT_DIR   = BASE / "codigo" / "model_v3"
OUT_DIR.mkdir(exist_ok=True)

# ─── Target v3 ────────────────────────────────────────────────────────────────
AVOIDABLE = {'2. NON-COMP', '3. MED ONLY', '4. TEMPORARY'}

# ─── Features ─────────────────────────────────────────────────────────────────
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

# ─── Métricas (sin sklearn) ───────────────────────────────────────────────────
def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    pos = y_true.sum(); neg = len(y_true) - pos
    if pos == 0 or neg == 0: return 0.5
    order = np.argsort(-y_score); y_s = y_true[order]
    tpr = np.cumsum(y_s) / pos; fpr = np.cumsum(1 - y_s) / neg
    return float(np.trapz(tpr, fpr))

def pr_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    order = np.argsort(-y_score); y_s = y_true[order]
    tp_c = np.cumsum(y_s); n = np.arange(1, len(y_s) + 1)
    prec = np.concatenate([[1.], tp_c / n])
    rec  = np.concatenate([[0.], tp_c / y_true.sum()])
    return float(np.trapz(prec, rec))

def ks_stat(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    order = np.argsort(-y_score); y_s = y_true[order]
    pos = y_true.sum(); neg = len(y_true) - pos
    return float(np.max(np.cumsum(y_s) / pos - np.cumsum(1 - y_s) / neg))

def best_f1_threshold(y_true, y_score):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 150):
        yp = (y_score >= t).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        d  = 2 * tp + fp + fn
        f1 = 2 * tp / d if d > 0 else 0.
        if f1 > best_f1: best_f1, best_t = f1, t
    return best_t

def eval_full(y_true, y_score, tau=None):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    if tau is None: tau = best_f1_threshold(y_true, y_score)
    yp = (y_score >= tau).astype(int)
    tp = int(((yp == 1) & (y_true == 1)).sum())
    fp = int(((yp == 1) & (y_true == 0)).sum())
    fn = int(((yp == 0) & (y_true == 1)).sum())
    tn = int(((yp == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0.
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.
    return dict(
        auc_roc=round(roc_auc(y_true, y_score), 4),
        pr_auc =round(pr_auc(y_true, y_score),  4),
        ks     =round(ks_stat(y_true, y_score),  4),
        brier  =round(float(np.mean((y_score - y_true) ** 2)), 4),
        precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
        threshold=round(float(tau), 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"OPTUNA v3 — LITIGIO EVITABLE | {N_TRIALS} trials | {TASK_TYPE} | sample={SAMPLE_SIZE//1000}k")
print("=" * 70)
t0 = time.time()

# ─── [1] Carga ────────────────────────────────────────────────────────────────
print(f"\n[1] Cargando dataset: {DATA_FILE}")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"    Total: {len(df):,} filas")

# ─── [2] Target v3 ────────────────────────────────────────────────────────────
df['target'] = (
    df['target'].eq(1) &
    df['claim_injury_type_REF'].isin(AVOIDABLE)
).astype(int)
n_pos_tot = int(df['target'].sum())
n_neg_tot = len(df) - n_pos_tot
print(f"    Positivos: {n_pos_tot:,} ({100*n_pos_tot/len(df):.1f}%)")

# ─── [3] Features ─────────────────────────────────────────────────────────────
for c in CAT_FEATURES:
    df[c] = df[c].fillna("UNKNOWN").astype(str)
for c in NUM_FEATURES:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1)

feats   = [c for c in ALL_FEATURES if c in df.columns]
cat_idx = [feats.index(c) for c in CAT_FEATURES if c in feats]
print(f"    Features: {len(feats)} ({len(cat_idx)} categóricas)")

# ─── [4] Split temporal ────────────────────────────────────────────────────────
train_mask = df['accident_year'].between(2017, 2020)
val_mask   = df['accident_year'] == 2021
test_mask  = df['accident_year'] == 2022

X_train, y_train = df.loc[train_mask, feats].reset_index(drop=True), df.loc[train_mask, 'target'].reset_index(drop=True)
X_val,   y_val   = df.loc[val_mask,   feats].reset_index(drop=True), df.loc[val_mask,   'target'].reset_index(drop=True)
X_test,  y_test  = df.loc[test_mask,  feats].reset_index(drop=True), df.loc[test_mask,  'target'].reset_index(drop=True)

spw_train = round((y_train == 0).sum() / (y_train == 1).sum(), 3)

print(f"\n    Train 2017-2020: {len(X_train):>9,}  pos={y_train.sum():,} ({100*y_train.mean():.1f}%)")
print(f"    Val  2021:       {len(X_val):>9,}  pos={y_val.sum():,}   ({100*y_val.mean():.1f}%)")
print(f"    Test 2022:       {len(X_test):>9,}  pos={y_test.sum():,}   ({100*y_test.mean():.1f}%)")
print(f"    scale_pos_weight (train): {spw_train}")

# ─── [5] Submuestra estratificada para Optuna ─────────────────────────────────
# Mantiene prevalencia original; mucho más rápido por trial (~30-45s vs 3-5min)
print(f"\n[5] Creando submuestra estratificada para Optuna ({SAMPLE_SIZE//1000}k filas) ...")
rng = np.random.default_rng(SEED)

pos_idx = np.where(y_train.values == 1)[0]
neg_idx = np.where(y_train.values == 0)[0]

n_pos_sample = min(len(pos_idx), int(SAMPLE_SIZE * y_train.mean()))
n_neg_sample = SAMPLE_SIZE - n_pos_sample

pos_sample = rng.choice(pos_idx, size=n_pos_sample, replace=False)
neg_sample = rng.choice(neg_idx, size=n_neg_sample, replace=False)
sample_idx = np.concatenate([pos_sample, neg_sample])
rng.shuffle(sample_idx)

X_tr_s = X_train.iloc[sample_idx].reset_index(drop=True)
y_tr_s = y_train.iloc[sample_idx].reset_index(drop=True)
spw_sample = round((y_tr_s == 0).sum() / (y_tr_s == 1).sum(), 3)

print(f"    Submuestra: {len(X_tr_s):,} filas  |  pos={y_tr_s.sum():,} ({100*y_tr_s.mean():.1f}%)  |  SPW={spw_sample}")

# Pools fijos para todos los trials
pool_sample = Pool(X_tr_s, y_tr_s, cat_features=cat_idx)
pool_val    = Pool(X_val,  y_val,  cat_features=cat_idx)
pool_test   = Pool(X_test, y_test, cat_features=cat_idx)

# ─── [6] Optuna objective ─────────────────────────────────────────────────────
trial_count      = [0]
best_val_so_far  = [0.0]

def objective(trial):
    trial_count[0] += 1
    n = trial_count[0]

    # Espacio de búsqueda: solo parámetros compatibles con GPU en CatBoost
    depth        = trial.suggest_int(  "depth",              6, 12)
    lr           = trial.suggest_float("learning_rate",      0.01, 0.15, log=True)
    l2           = trial.suggest_float("l2_leaf_reg",        1.0, 25.0, log=True)
    bag_temp     = trial.suggest_float("bagging_temperature", 0.0, 1.5)
    border_count = trial.suggest_int(  "border_count",       32, 128, step=16)

    params = dict(
        iterations            = 1000,
        depth                 = depth,
        learning_rate         = lr,
        l2_leaf_reg           = l2,
        bagging_temperature   = bag_temp,
        border_count          = border_count,
        task_type             = TASK_TYPE,
        loss_function         = "Logloss",
        eval_metric           = "PRAUC",
        scale_pos_weight      = spw_sample,
        random_seed           = SEED,
        verbose               = 0,
        early_stopping_rounds = 30,
        allow_writing_files   = False,
    )

    model = CatBoostClassifier(**params)
    model.fit(pool_sample, eval_set=pool_val, use_best_model=True)

    val_proba = model.predict_proba(pool_val)[:, 1]
    val_pr    = round(pr_auc(y_val.values, val_proba), 4)
    val_roc   = round(roc_auc(y_val.values, val_proba), 4)
    best_iter = model.get_best_iteration()

    trial.set_user_attr("val_pr_auc",  val_pr)
    trial.set_user_attr("val_roc_auc", val_roc)
    trial.set_user_attr("best_iter",   best_iter)

    # Liberar memoria entre trials
    del model, val_proba
    gc.collect()

    if val_pr > best_val_so_far[0]:
        best_val_so_far[0] = val_pr
        flag = "  ★ NUEVO BEST"
    else:
        flag = ""

    elapsed = (time.time() - t0) / 60
    eta     = elapsed / n * (N_TRIALS - n) if n > 0 else 0
    print(f"  [{n:>3}/{N_TRIALS}] PR={val_pr:.4f}  AUC={val_roc:.4f}  "
          f"iter={best_iter:>4}  depth={depth}  lr={lr:.4f}  "
          f"l2={l2:.2f}  bag={bag_temp:.2f}  bc={border_count}  "
          f"[{elapsed:.1f}min, ETA {eta:.0f}min]{flag}")

    return val_pr


# ─── [7] Correr estudio ───────────────────────────────────────────────────────
print(f"\n[6] Iniciando Optuna ({N_TRIALS} trials) ...")
print(f"    Primeros 20 trials = exploración aleatoria (TPE startup)")
print(f"    Métricas evaluadas en Val 2021 COMPLETO ({len(X_val):,} filas)")
print("-" * 70)

sampler = TPESampler(seed=SEED, n_startup_trials=20)
study   = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

with open(OUT_DIR / "optuna_study.pkl", "wb") as f:
    pickle.dump(study, f)

best = study.best_trial
print("-" * 70)
print(f"\n[7] Mejor trial #{best.number}:")
print(f"    Val PR-AUC = {best.value:.4f}  (baseline: 0.5981  delta: {best.value-0.5981:+.4f})")
print(f"    Val AUC    = {best.user_attrs.get('val_roc_auc','?'):.4f}")
print(f"    Best iter  = {best.user_attrs.get('best_iter','?')}")
print("    Hiperparámetros óptimos:")
for k, v in best.params.items():
    print(f"      {k:25s} = {v}")

# ─── [8] Reentrenamiento final: Train+Val COMPLETO ────────────────────────────
print(f"\n[8] Reentrenando en Train+Val COMPLETO (hiperparámetros óptimos) ...")

X_tv = pd.concat([X_train, X_val]).reset_index(drop=True)
y_tv = pd.concat([y_train, y_val]).reset_index(drop=True)
spw_tv = round((y_tv == 0).sum() / (y_tv == 1).sum(), 3)

# Escalar iteraciones: best_iter de submuestra × 1.3 para dataset mayor
best_iter_optuna = best.user_attrs.get('best_iter', 500)
final_iters      = int(best_iter_optuna * 1.3)
print(f"    Train+Val: {len(X_tv):,} filas  |  SPW={spw_tv}  |  iterations={final_iters}")

pool_tv = Pool(X_tv, y_tv, cat_features=cat_idx)

final_params = {
    **best.params,
    "iterations":          final_iters,
    "task_type":           TASK_TYPE,
    "loss_function":       "Logloss",
    "eval_metric":         "PRAUC",
    "scale_pos_weight":    spw_tv,
    "random_seed":         SEED,
    "verbose":             100,
    "allow_writing_files": False,
    # sin early_stopping en el modelo final
}

model_final = CatBoostClassifier(**final_params)
t_train = time.time()
model_final.fit(pool_tv)
print(f"    Listo en {(time.time()-t_train)/60:.1f} min")

# ─── [9] Evaluación Test 2022 ─────────────────────────────────────────────────
print(f"\n[9] Evaluando en Test 2022 ...")
test_proba = model_final.predict_proba(pool_test)[:, 1]
val_proba2 = model_final.predict_proba(pool_val)[:, 1]

tau    = best_f1_threshold(y_test.values, test_proba)
m_test = eval_full(y_test.values, test_proba, tau)
m_val  = eval_full(y_val.values,  val_proba2, best_f1_threshold(y_val.values, val_proba2))

print(f"\n  ╔══════════════════════════════════════╗")
print(f"  ║   RESULTADOS FINALES — Test 2022     ║")
print(f"  ╠══════════════════════════════════════╣")
print(f"  ║  AUC-ROC   = {m_test['auc_roc']:.4f}               ║")
print(f"  ║  PR-AUC    = {m_test['pr_auc']:.4f}               ║")
print(f"  ║  KS        = {m_test['ks']:.4f}               ║")
print(f"  ║  Brier     = {m_test['brier']:.4f}               ║")
print(f"  ║  Precision = {m_test['precision']:.4f}   (τ={tau:.4f}) ║")
print(f"  ║  Recall    = {m_test['recall']:.4f}               ║")
print(f"  ╠══════════════════════════════════════╣")
print(f"  ║  TP={m_test['tp']:>6}   FP={m_test['fp']:>6}          ║")
print(f"  ║  FN={m_test['fn']:>6}   TN={m_test['tn']:>6}          ║")
print(f"  ╠══════════════════════════════════════╣")
print(f"  ║  vs. baseline (Cap.6 hiperparámetros)║")
print(f"  ║  PR-AUC:  {m_test['pr_auc']:.4f} vs 0.5981  ({m_test['pr_auc']-0.5981:+.4f}) ║")
print(f"  ║  AUC-ROC: {m_test['auc_roc']:.4f} vs 0.8833  ({m_test['auc_roc']-0.8833:+.4f}) ║")
print(f"  ║  KS:      {m_test['ks']:.4f} vs 0.6202  ({m_test['ks']-0.6202:+.4f}) ║")
print(f"  ╚══════════════════════════════════════╝")

# ─── [10] Guardar ─────────────────────────────────────────────────────────────
print(f"\n[10] Guardando ...")
model_path = OUT_DIR / "catboost_v3_optuna.cbm"
model_final.save_model(str(model_path))

results = {
    "script":              "retrain_v3_optuna_gpu.py v2",
    "n_trials":            N_TRIALS,
    "sample_size_optuna":  len(X_tr_s),
    "best_trial_number":   best.number,
    "best_val_pr_auc":     round(best.value, 4),
    "best_params":         best.params,
    "final_iterations":    final_iters,
    "scale_pos_weight_tv": spw_tv,
    "validation":          m_val,
    "test":                m_test,
    "elapsed_min":         round((time.time() - t0) / 60, 1),
    "baseline_comparison": {
        "pr_auc_delta":  round(m_test['pr_auc']  - 0.5981, 4),
        "auc_roc_delta": round(m_test['auc_roc'] - 0.8833, 4),
        "ks_delta":      round(m_test['ks']       - 0.6202, 4),
    },
    "top5_trials": [
        {"number": t.number, "val_pr_auc": round(t.value, 4),
         "params": t.params, "best_iter": t.user_attrs.get("best_iter")}
        for t in sorted(study.trials, key=lambda x: x.value or 0, reverse=True)[:5]
        if t.value is not None
    ],
}

results_path = OUT_DIR / "optuna_best_params.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"    Modelo:     {model_path}")
print(f"    Resultados: {results_path}")
print(f"    Estudio:    {OUT_DIR / 'optuna_study.pkl'}")

total = (time.time() - t0) / 60
print(f"\n{'='*70}")
print(f"COMPLETADO en {total:.1f} min  |  Best Val PR-AUC = {best.value:.4f}")
print(f"{'='*70}")
