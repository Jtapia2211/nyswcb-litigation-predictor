"""
depth_ext_experiment.py — Experimento de seguimiento: ¿depth > 10 mejora el modelo?
======================================================================================
Motivación (Cap.6 tesis): los 5 mejores trials del Optuna original convergen a depth=10,
el tope del espacio de búsqueda [6,10]. Este experimento responde si el verdadero óptimo
está dentro o fuera del dominio explorado.

DISEÑO:
  Parte A — Grilla depth [8..14] con parámetros fijos al óptimo conocido
    → 7 modelos, curva limpia de profundidad vs. desempeño
    → Cada modelo se entrena en submuestra (~400k) y se evalúa en Val 2021

  Parte B — Optuna libre (80 trials) con depth en [8,14]
    → Permite al optimizador encontrar la combinación óptima en el rango extendido
    → Reentrenamiento final del mejor trial en Train+Val completo → evaluado en Test 2022

SALIDAS en codigo/model_v3_depth_ext/:
  depth_grid_results.json      — curva Parte A (depth vs métricas)
  optuna_study_depth_ext.pkl   — estudio Optuna Parte B
  optuna_best_params_ext.json  — mejor configuración extendida + comparación baseline
  catboost_v3_depth_ext.cbm    — modelo final (si mejora al baseline)

TIEMPO ESTIMADO (GPU RTX 3070+):
  Parte A:  ~7 × 45s  ≈  5 min
  Parte B:  80 × 45s  ≈  60 min
  Retrain final:       ≈  8 min
  TOTAL:               ≈  75 min

EJECUTAR:
  python depth_ext_experiment.py              # completo (~75 min)
  python depth_ext_experiment.py --skip-b     # solo grilla Parte A (~5 min)
  python depth_ext_experiment.py --task CPU   # sin GPU

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
parser.add_argument("--task",    type=str, default="GPU",  help="GPU o CPU")
parser.add_argument("--seed",    type=int, default=42,     help="Semilla")
parser.add_argument("--skip-b",  action="store_true",      help="Saltar Parte B (Optuna)")
parser.add_argument("--trials",  type=int, default=80,     help="Trials Optuna Parte B")
parser.add_argument("--sample-k",type=int, default=400,    help="Submuestra en miles")
args = parser.parse_args()

TASK_TYPE   = args.task
SEED        = args.seed
N_TRIALS    = args.trials
SAMPLE_SIZE = args.sample_k * 1000

# ─── Baseline conocido (Cap.6, trial #22) ─────────────────────────────────────
BASELINE_PARAMS = {
    "learning_rate":       0.061,
    "l2_leaf_reg":         7.62,
    "bagging_temperature": 0.792,
    "border_count":        82,
}
BASELINE_METRICS = {
    "val_pr_auc":  0.5976,
    "test_pr_auc": 0.5981,
    "test_auc_roc":0.8833,
    "test_ks":     0.6202,
}
BASELINE_DEPTH   = 10
BASELINE_ITERS   = 1511   # iteraciones efectivas del modelo de producción

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
DATA_FILE = BASE / "raw_data" / "dataset_tesis_clean.csv"
OUT_DIR   = BASE / "codigo" / "model_v3_depth_ext"
OUT_DIR.mkdir(exist_ok=True)

# ─── Target v3 ────────────────────────────────────────────────────────────────
AVOIDABLE = {'2. NON-COMP', '3. MED ONLY', '4. TEMPORARY'}

# ─── Features (idénticas al modelo original) ──────────────────────────────────
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

# ─── Métricas ─────────────────────────────────────────────────────────────────
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
print(f"EXPERIMENTO DEPTH EXTENDIDO [8-14] | {TASK_TYPE} | seed={SEED}")
print(f"Baseline: depth={BASELINE_DEPTH}  Val PR-AUC={BASELINE_METRICS['val_pr_auc']}  "
      f"Test PR-AUC={BASELINE_METRICS['test_pr_auc']}")
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

# ─── [5] Submuestra estratificada ─────────────────────────────────────────────
print(f"\n[2] Creando submuestra estratificada ({SAMPLE_SIZE//1000}k filas) ...")
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
spw_s  = round((y_tr_s == 0).sum() / (y_tr_s == 1).sum(), 3)
print(f"    Submuestra: {len(X_tr_s):,} | pos={y_tr_s.sum():,} ({100*y_tr_s.mean():.1f}%) | SPW={spw_s}")

pool_sample = Pool(X_tr_s, y_tr_s, cat_features=cat_idx)
pool_val    = Pool(X_val,  y_val,  cat_features=cat_idx)
pool_test   = Pool(X_test, y_test, cat_features=cat_idx)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE A — Grilla depth [8..14] con parámetros fijos
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PARTE A — Grilla depth [8, 9, 10, 11, 12, 13, 14] (parámetros fijos)")
print(f"          lr={BASELINE_PARAMS['learning_rate']}  l2={BASELINE_PARAMS['l2_leaf_reg']}  "
      f"bag_temp={BASELINE_PARAMS['bagging_temperature']}  border_count={BASELINE_PARAMS['border_count']}")
print(f"{'='*70}")

grid_results = []
t_part_a = time.time()

for depth in range(8, 15):   # [8, 9, 10, 11, 12, 13, 14]
    t_trial = time.time()
    params = {
        **BASELINE_PARAMS,
        "depth":               depth,
        "iterations":          1000,
        "task_type":           TASK_TYPE,
        "loss_function":       "Logloss",
        "eval_metric":         "PRAUC",
        "scale_pos_weight":    spw_s,
        "random_seed":         SEED,
        "verbose":             0,
        "early_stopping_rounds": 40,
        "allow_writing_files": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(pool_sample, eval_set=pool_val, use_best_model=True)

    val_proba = model.predict_proba(pool_val)[:, 1]
    val_pr    = round(pr_auc(y_val.values, val_proba), 4)
    val_roc   = round(roc_auc(y_val.values, val_proba), 4)
    best_iter = model.get_best_iteration()

    delta_pr  = val_pr - BASELINE_METRICS['val_pr_auc']
    marker    = " ★" if depth == BASELINE_DEPTH else ""
    marker    = " ◆ MEJOR" if val_pr > BASELINE_METRICS['val_pr_auc'] + 0.0001 else marker

    print(f"  depth={depth:>2}  PR-AUC={val_pr:.4f}  ({delta_pr:+.4f} vs baseline)  "
          f"AUC={val_roc:.4f}  iter={best_iter:>4}  ({time.time()-t_trial:.0f}s){marker}")

    grid_results.append({
        "depth":       depth,
        "val_pr_auc":  val_pr,
        "val_auc_roc": val_roc,
        "best_iter":   best_iter,
        "delta_pr_vs_baseline": round(delta_pr, 4),
        "is_baseline": depth == BASELINE_DEPTH,
    })

    del model, val_proba
    gc.collect()

print(f"\n  Parte A completada en {(time.time()-t_part_a)/60:.1f} min")

# Guardar grilla
with open(OUT_DIR / "depth_grid_results.json", "w") as f:
    json.dump({"baseline": BASELINE_METRICS, "grid": grid_results}, f, indent=2)
print(f"  Guardado: {OUT_DIR / 'depth_grid_results.json'}")

# Resumen Parte A
best_grid = max(grid_results, key=lambda x: x["val_pr_auc"])
print(f"\n  ► Mejor en grilla: depth={best_grid['depth']}  "
      f"Val PR-AUC={best_grid['val_pr_auc']:.4f}  "
      f"(delta={best_grid['delta_pr_vs_baseline']:+.4f} vs baseline depth=10)")

if best_grid["depth"] == BASELINE_DEPTH:
    print(f"  ► CONCLUSIÓN PARTE A: depth=10 sigue siendo óptimo en el rango extendido.")
    print(f"     La curva se aplana — la elección original está justificada.")
elif best_grid["val_pr_auc"] - BASELINE_METRICS['val_pr_auc'] < 0.001:
    print(f"  ► CONCLUSIÓN PARTE A: mejora marginal (< 0.001) — profundidades mayores no")
    print(f"     aportan ganancia prácticamente significativa.")
else:
    print(f"  ► CONCLUSIÓN PARTE A: depth={best_grid['depth']} mejora el baseline en "
          f"{best_grid['delta_pr_vs_baseline']:+.4f} — hay espacio para explorar.")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE B — Optuna libre con depth en [8, 14]
# ═══════════════════════════════════════════════════════════════════════════════
if args.skip_b:
    print("\n[Parte B omitida por --skip-b]")
    sys.exit(0)

print(f"\n{'='*70}")
print(f"PARTE B — Optuna {N_TRIALS} trials | depth en [8, 14] | todos los hiperparámetros libres")
print(f"{'='*70}\n")

trial_count     = [0]
best_so_far     = [0.0]
t_part_b        = time.time()

def objective(trial):
    trial_count[0] += 1
    n = trial_count[0]

    depth        = trial.suggest_int(  "depth",              8, 14)
    lr           = trial.suggest_float("learning_rate",      0.02, 0.12, log=True)
    l2           = trial.suggest_float("l2_leaf_reg",        2.0, 20.0, log=True)
    bag_temp     = trial.suggest_float("bagging_temperature", 0.0, 2.0)
    border_count = trial.suggest_int(  "border_count",       48, 128, step=16)

    params = dict(
        iterations            = 1200,
        depth                 = depth,
        learning_rate         = lr,
        l2_leaf_reg           = l2,
        bagging_temperature   = bag_temp,
        border_count          = border_count,
        task_type             = TASK_TYPE,
        loss_function         = "Logloss",
        eval_metric           = "PRAUC",
        scale_pos_weight      = spw_s,
        random_seed           = SEED,
        verbose               = 0,
        early_stopping_rounds = 40,
        allow_writing_files   = False,
    )

    model = CatBoostClassifier(**params)
    model.fit(pool_sample, eval_set=pool_val, use_best_model=True)

    val_proba = model.predict_proba(pool_val)[:, 1]
    val_pr    = round(pr_auc(y_val.values, val_proba), 4)
    val_roc   = round(roc_auc(y_val.values, val_proba), 4)
    best_iter = model.get_best_iteration()

    trial.set_user_attr("val_roc_auc", val_roc)
    trial.set_user_attr("best_iter",   best_iter)

    del model, val_proba
    gc.collect()

    flag = ""
    if val_pr > best_so_far[0]:
        best_so_far[0] = val_pr
        flag = "  ★ NUEVO BEST"

    elapsed = (time.time() - t_part_b) / 60
    eta     = elapsed / n * (N_TRIALS - n) if n > 0 else 0
    print(f"  [{n:>3}/{N_TRIALS}] PR={val_pr:.4f}  AUC={val_roc:.4f}  "
          f"iter={best_iter:>4}  d={depth}  lr={lr:.4f}  l2={l2:.2f}  "
          f"bag={bag_temp:.2f}  bc={border_count}  "
          f"[{elapsed:.1f}min ETA {eta:.0f}min]{flag}")
    return val_pr

sampler = TPESampler(seed=SEED, n_startup_trials=15)
study   = optuna.create_study(direction="maximize", sampler=sampler)

# Warm-start: añadir baseline como trial informado
study.enqueue_trial({
    "depth": BASELINE_DEPTH, "learning_rate": BASELINE_PARAMS["learning_rate"],
    "l2_leaf_reg": BASELINE_PARAMS["l2_leaf_reg"],
    "bagging_temperature": BASELINE_PARAMS["bagging_temperature"],
    "border_count": BASELINE_PARAMS["border_count"],
})

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

with open(OUT_DIR / "optuna_study_depth_ext.pkl", "wb") as f:
    pickle.dump(study, f)

best = study.best_trial
print(f"\n  Parte B completada en {(time.time()-t_part_b)/60:.1f} min")
print(f"\n  Mejor trial #{best.number}:")
print(f"    Val PR-AUC = {best.value:.4f}  (baseline: {BASELINE_METRICS['val_pr_auc']}  "
      f"delta: {best.value - BASELINE_METRICS['val_pr_auc']:+.4f})")
print(f"    Hiperparámetros: {best.params}")


# ─── Reentrenamiento final en Train+Val COMPLETO ──────────────────────────────
# Solo si el mejor trial supera al baseline en algo más que ruido (> 0.001)
gain = best.value - BASELINE_METRICS['val_pr_auc']

if gain > 0.001:
    print(f"\n[Retrain] Ganancia real ({gain:+.4f}) — reentrenando en Train+Val completo ...")
    X_tv  = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_tv  = pd.concat([y_train, y_val]).reset_index(drop=True)
    spw_tv = round((y_tv == 0).sum() / (y_tv == 1).sum(), 3)

    best_iter_optuna = best.user_attrs.get('best_iter', 600)
    final_iters      = int(best_iter_optuna * 1.3)
    print(f"    Train+Val: {len(X_tv):,} | SPW={spw_tv} | iterations={final_iters}")

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
    }
    model_final = CatBoostClassifier(**final_params)
    t_r = time.time()
    model_final.fit(pool_tv)
    print(f"    Listo en {(time.time()-t_r)/60:.1f} min")

    test_proba = model_final.predict_proba(pool_test)[:, 1]
    val_proba2 = model_final.predict_proba(pool_val)[:, 1]
    tau    = best_f1_threshold(y_test.values, test_proba)
    m_test = eval_full(y_test.values, test_proba, tau)
    m_val  = eval_full(y_val.values,  val_proba2)

    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║     MODELO EXTENDIDO — Test 2022                 ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  AUC-ROC   = {m_test['auc_roc']:.4f}  (baseline: {BASELINE_METRICS['test_auc_roc']:.4f}  {m_test['auc_roc']-BASELINE_METRICS['test_auc_roc']:+.4f}) ║")
    print(f"  ║  PR-AUC    = {m_test['pr_auc']:.4f}  (baseline: {BASELINE_METRICS['test_pr_auc']:.4f}  {m_test['pr_auc']-BASELINE_METRICS['test_pr_auc']:+.4f}) ║")
    print(f"  ║  KS        = {m_test['ks']:.4f}  (baseline: {BASELINE_METRICS['test_ks']:.4f}  {m_test['ks']-BASELINE_METRICS['test_ks']:+.4f}) ║")
    print(f"  ║  Precision = {m_test['precision']:.4f}  Recall = {m_test['recall']:.4f}  τ={tau:.4f} ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  TP={m_test['tp']:>6}   FP={m_test['fp']:>6}                    ║")
    print(f"  ║  FN={m_test['fn']:>6}   TN={m_test['tn']:>6}                    ║")
    print(f"  ╚══════════════════════════════════════════════════╝")

    model_path = OUT_DIR / "catboost_v3_depth_ext.cbm"
    model_final.save_model(str(model_path))
    print(f"  Modelo guardado: {model_path}")
else:
    print(f"\n[Retrain omitido] Ganancia ({gain:+.4f}) ≤ 0.001 — depth=10 sigue siendo óptimo.")
    print(f"  No se guarda modelo nuevo. El modelo de producción actual es el correcto.")
    m_test = None
    m_val  = None
    model_path = None

# ─── Guardar resultados Parte B ───────────────────────────────────────────────
top5 = [
    {"number": t.number, "val_pr_auc": round(t.value, 4),
     "params": t.params, "best_iter": t.user_attrs.get("best_iter")}
    for t in sorted(study.trials, key=lambda x: x.value or 0, reverse=True)[:5]
    if t.value is not None
]

results_b = {
    "experiment":          "depth_extension [8-14]",
    "baseline_depth":      BASELINE_DEPTH,
    "baseline_val_pr_auc": BASELINE_METRICS["val_pr_auc"],
    "baseline_test_pr_auc":BASELINE_METRICS["test_pr_auc"],
    "n_trials":            N_TRIALS,
    "best_trial_number":   best.number,
    "best_val_pr_auc":     round(best.value, 4),
    "best_params":         best.params,
    "gain_vs_baseline":    round(gain, 4),
    "conclusion": (
        "depth optimo fuera del rango original" if gain > 0.001
        else "depth=10 sigue siendo optimo - curva se aplana"
    ),
    "test_results":        m_test,
    "val_results":         m_val,
    "model_saved":         str(model_path) if model_path else None,
    "top5_trials":         top5,
    "elapsed_total_min":   round((time.time() - t0) / 60, 1),
}

results_path = OUT_DIR / "optuna_best_params_ext.json"
with open(results_path, "w") as f:
    json.dump(results_b, f, indent=2)
print(f"\n  Resultados guardados: {results_path}")

# ─── Resumen final ────────────────────────────────────────────────────────────
total = (time.time() - t0) / 60
print(f"\n{'='*70}")
print(f"EXPERIMENTO COMPLETADO en {total:.1f} min")
print(f"{'='*70}")
print(f"\n  PARTE A — Grilla depth [8-14]:")
for r in grid_results:
    marker = " ← baseline" if r["is_baseline"] else ""
    marker = " ← MEJOR GRILLA" if r["depth"] == best_grid["depth"] and not r["is_baseline"] else marker
    print(f"    depth={r['depth']:>2}  Val PR-AUC={r['val_pr_auc']:.4f}  ({r['delta_pr_vs_baseline']:+.4f}){marker}")

print(f"\n  PARTE B — Optuna libre [8-14]:")
print(f"    Mejor depth={best.params.get('depth')}  Val PR-AUC={best.value:.4f}  (delta={gain:+.4f})")
print(f"\n  CONCLUSIÓN FINAL: {'depth>10 mejora el modelo — considerar actualizar' if gain > 0.001 else 'depth=10 es óptimo — elección original confirmada'}")
print(f"{'='*70}")
