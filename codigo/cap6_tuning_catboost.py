"""
cap6_tuning_catboost.py
=======================
Optimización Bayesiana de CatBoost GPU con Optuna (TPE Sampler)
Maestría en Ciencia de Datos — ITBA  |  Tapia, Julián

Estrategia de búsqueda:
  - Sampler     : TPE (Tree-structured Parzen Estimator)
  - Trials      : 60
  - Métrica obj : PR-AUC en validación temporal 2021
  - Early stop  : 100 rondas CatBoost sobre val PR-AUC (sin contar en trials)
  - Espacio     : 6 hiperparámetros clave
  - Hardware    : CUDA GPU (devices="0")

Uso:
    pip install catboost optuna pandas numpy pyarrow scikit-learn
    python cap6_tuning_catboost.py

Salidas (todas en OUT_DIR):
    best_catboost.cbm          → modelo final con mejores parámetros
    optuna_study.pkl           → estudio Optuna completo (reanudable)
    tuning_results.json        → métricas de val/test + params de todos los trials
    fig_optuna_history.png     → curva de optimización PR-AUC vs trial
    fig_optuna_importance.png  → importancia de hiperparámetros (fANOVA)
    fig_optuna_contour.png     → contour plots pares de parámetros clave
"""

import json, time, warnings, os, pickle, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from catboost import CatBoostClassifier, Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN  ← AJUSTAR ANTES DE CORRER
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = Path(r"C:\Users\julia\Tesis_ML")                    # ← carpeta raíz
DATA_FILE   = BASE_DIR / "raw_data" / "dataset_tesis_clean.csv"   # CSV benchmark
OUT_DIR     = BASE_DIR / "codigo" / "tuning_catboost"

YEAR_COL    = "accident_year"   # columna para el split temporal
TARGET_COL  = "target"

# Hiperparámetros fijos (no se optimizan)
FIXED_PARAMS = dict(
    task_type          = "GPU",
    devices            = "0",
    loss_function      = "Logloss",
    eval_metric        = "PRAUC",          # métrica de early stopping
    scale_pos_weight   = 3.406,
    random_seed        = 42,
    verbose            = False,
    allow_writing_files= False,
)

# Espacio de búsqueda Optuna
SEARCH_SPACE = {
    "iterations"        : ("int",   200,   2000, False),   # (tipo, min, max, log)
    "depth"             : ("int",   6,     10,   False),
    "learning_rate"     : ("float", 0.01,  0.30, True),    # escala log
    "l2_leaf_reg"       : ("float", 1.0,   15.0, True),    # escala log
    "bagging_temperature": ("float",0.0,   1.5,  False),
    "border_count"      : ("int",   64,    255,  False),
}

N_TRIALS      = 60
EARLY_STOP    = 100     # rondas CatBoost sin mejora en val PR-AUC
RANDOM_SEED   = 42
STUDY_NAME    = "catboost_prauc_opt"

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURES
# ══════════════════════════════════════════════════════════════════════════════

# Categóricas identificadas en la tesis (12 features)
CAT_FEATURES = [
    "gender", "accident_type", "occupational_disease",
    "county_of_injury", "medical_fee_region",
    "wcio_cause_code", "wcio_nature_code", "wcio_body_code",
    "carrier_type", "district_name",
    "industry_code", "industry_desc",
]

# Numéricas (11 features)
NUM_FEATURES = [
    "days_to_assembly", "days_C2_to_accident", "days_C3_to_accident",
    "age_at_injury", "aww",
    "has_C2", "has_C3", "has_ANCR_early",
    "accident_year", "accident_month", "accident_dow",
]

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES   # orden estable

# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS  (NumPy puro — consistentes con el benchmark Cap 5)
# ══════════════════════════════════════════════════════════════════════════════

def roc_auc(y_true, y_score):
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(-y_score)
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    tpr = tp / pos
    fpr = fp / neg
    tpr = np.concatenate([[0.], tpr])
    fpr = np.concatenate([[0.], fpr])
    return float(np.trapz(tpr, fpr))

def pr_auc(y_true, y_score):
    order = np.argsort(-y_score)
    yt    = y_true[order]
    tp    = np.cumsum(yt)
    fp    = np.cumsum(1 - yt)
    prec  = tp / (tp + fp + 1e-12)
    rec   = tp / (tp.max() + 1e-12)
    # agregar punto (0, 1)
    prec  = np.concatenate([[1.], prec])
    rec   = np.concatenate([[0.], rec])
    return float(np.trapz(prec, rec))

def ks_stat(y_true, y_score):
    from scipy import stats as sst
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    return float(sst.ks_2samp(pos_scores, neg_scores).statistic)

def brier(y_true, y_score):
    return float(np.mean((y_score - y_true) ** 2))

def youden_threshold(y_true, y_score, n_thr=500):
    thresholds = np.linspace(0, 1, n_thr)
    best_thr, best_j = 0.5, -1.0
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_thr = j, thr
    return best_thr

def confusion_at_thr(y_true, y_score, thr):
    pred = (y_score >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                precision=float(prec), recall=float(rec), f1=float(f1),
                threshold=float(thr))

def full_metrics(y_true, y_score, label=""):
    auc  = roc_auc(y_true, y_score)
    prauc= pr_auc(y_true, y_score)
    br   = brier(y_true, y_score)
    try:
        ks = ks_stat(y_true, y_score)
    except Exception:
        # fallback manual KS
        order = np.argsort(-y_score)
        yt = y_true[order]
        tp = np.cumsum(yt) / (y_true.sum() + 1e-12)
        fp = np.cumsum(1 - yt) / ((1 - y_true).sum() + 1e-12)
        ks = float(np.max(np.abs(tp - fp)))
    thr  = youden_threshold(y_true, y_score)
    conf = confusion_at_thr(y_true, y_score, thr)
    return dict(auc_roc=round(auc,4), pr_auc=round(prauc,4),
                ks=round(ks,4), brier=round(br,4), **conf)

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_splits():
    print(f"[data] Cargando {DATA_FILE} ...")
    t0 = time.time()
    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"[data] Shape total: {df.shape}  ({time.time()-t0:.1f}s)")

    # Asegurar tipos correctos en categóricas
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan", "MISSING")

    # Rellenar numéricas
    for c in NUM_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1)

    # Filtrar solo features + target + year (sin duplicados: accident_year ya está en NUM_FEATURES)
    cols_need = list(dict.fromkeys(ALL_FEATURES + [TARGET_COL, YEAR_COL]))
    df = df[[c for c in cols_need if c in df.columns]]

    train = df[df[YEAR_COL] <= 2020]
    val   = df[df[YEAR_COL] == 2021]
    test  = df[df[YEAR_COL] == 2022]

    feats = [c for c in ALL_FEATURES if c in df.columns]

    X_tr  = train[feats].values
    y_tr  = train[TARGET_COL].values.astype(np.float32)
    X_val = val[feats].values
    y_val = val[TARGET_COL].values.astype(np.float32)
    X_te  = test[feats].values
    y_te  = test[TARGET_COL].values.astype(np.float32)

    cat_idx = [feats.index(c) for c in CAT_FEATURES if c in feats]

    print(f"[data] Train: {X_tr.shape}  |  Val: {X_val.shape}  |  Test: {X_te.shape}")
    print(f"[data] Prev train: {y_tr.mean()*100:.1f}%  |  val: {y_val.mean()*100:.1f}%  |  test: {y_te.mean()*100:.1f}%")
    print(f"[data] Cat features ({len(cat_idx)}): idx {cat_idx}")
    return X_tr, y_tr, X_val, y_val, X_te, y_te, cat_idx, feats

# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(X_tr, y_tr, X_val, y_val, cat_idx, trial_log):

    pool_tr  = Pool(X_tr,  y_tr,  cat_features=cat_idx)
    pool_val = Pool(X_val, y_val, cat_features=cat_idx)

    def objective(trial):
        t0 = time.time()

        # Sugerir hiperparámetros desde el espacio de búsqueda
        params = {}
        for name, spec in SEARCH_SPACE.items():
            kind, lo, hi, log_ = spec
            if kind == "int":
                params[name] = trial.suggest_int(name, lo, hi, log=log_)
            else:
                params[name] = trial.suggest_float(name, lo, hi, log=log_)

        model = CatBoostClassifier(
            **params,
            **FIXED_PARAMS,
            early_stopping_rounds=EARLY_STOP,
        )

        model.fit(
            pool_tr,
            eval_set    = pool_val,
            use_best_model = True,
            plot        = False,
        )

        y_prob_val = model.predict_proba(X_val)[:, 1]
        val_prauc  = pr_auc(y_val, y_prob_val)
        val_auc    = roc_auc(y_val, y_prob_val)
        elapsed    = time.time() - t0
        best_iter  = model.get_best_iteration()

        entry = {
            "trial"      : trial.number,
            "val_prauc"  : round(val_prauc, 5),
            "val_auc"    : round(val_auc, 5),
            "best_iter"  : best_iter,
            "elapsed_s"  : round(elapsed, 1),
            "params"     : params,
        }
        trial_log.append(entry)

        # Guardar log incremental en cada trial
        log_path = OUT_DIR / "trial_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(trial_log, f, indent=2)

        print(f"  Trial {trial.number:>3d} | "
              f"PR-AUC val={val_prauc:.5f} | AUC={val_auc:.4f} | "
              f"iter={best_iter:>4d} | {elapsed:.0f}s | "
              f"lr={params['learning_rate']:.4f} depth={params['depth']} "
              f"l2={params['l2_leaf_reg']:.2f}")

        return val_prauc

    return objective

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURAS OPTUNA
# ══════════════════════════════════════════════════════════════════════════════

def plot_optimization(study, trial_log, out_dir):

    trials = [t for t in study.trials if t.value is not None]
    values = [t.value for t in trials]
    best_so_far = np.maximum.accumulate(values)
    nums   = [t.number for t in trials]

    # ── Fig 1: Historial de optimización ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    ax1.scatter(nums, values, c="steelblue", alpha=0.6, s=40, label="Trial PR-AUC")
    ax1.plot(nums, best_so_far, c="crimson", lw=2, label="Mejor acumulado")
    # Línea baseline Cap 5
    ax1.axhline(0.7407, color="darkorange", lw=1.5, ls="--", label="Baseline CB_GPU (0.7407)")
    ax1.set_xlabel("Trial", fontsize=11)
    ax1.set_ylabel("PR-AUC (val 2021)", fontsize=11)
    ax1.set_title("Historial de optimización Optuna", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Distribución de valores
    ax2.hist(values, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.axvline(np.max(values), color="crimson", lw=2, ls="--",
                label=f"Mejor: {np.max(values):.5f}")
    ax2.axvline(0.7407, color="darkorange", lw=1.5, ls="--",
                label=f"Baseline: 0.7407")
    ax2.set_xlabel("PR-AUC (val 2021)", fontsize=11)
    ax2.set_ylabel("Frecuencia", fontsize=11)
    ax2.set_title("Distribución de trials", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Optimización Bayesiana CatBoost GPU — Optuna TPE", fontsize=13, y=1.01)
    plt.tight_layout()
    out = out_dir / "fig_optuna_history.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Guardado: {out}")

    # ── Fig 2: Importancia de hiperparámetros (fANOVA) ────────────────────────
    try:
        importance = optuna.importance.get_param_importances(study)
        names = list(importance.keys())
        vals  = list(importance.values())
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))[::-1]

        fig2, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xlabel("Importancia (fANOVA)", fontsize=11)
        ax.set_title("Importancia de hiperparámetros — CatBoost GPU", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(vals) * 1.2)
        plt.tight_layout()
        out2 = out_dir / "fig_optuna_importance.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Guardado: {out2}")
    except Exception as e:
        print(f"[plot] fANOVA skipped: {e}")

    # ── Fig 3: Contour plots pares de parámetros clave ────────────────────────
    try:
        param_names = list(SEARCH_SPACE.keys())
        pairs = [
            ("learning_rate", "depth"),
            ("learning_rate", "iterations"),
            ("l2_leaf_reg",   "bagging_temperature"),
        ]
        fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
        for ax_, (px, py) in zip(axes3, pairs):
            xs = [t.params.get(px) for t in trials if px in t.params and py in t.params]
            ys = [t.params.get(py) for t in trials if px in t.params and py in t.params]
            zs = [t.value       for t in trials if px in t.params and py in t.params]
            if len(xs) < 5:
                ax_.set_visible(False)
                continue
            sc = ax_.scatter(xs, ys, c=zs, cmap="RdYlGn", s=60, alpha=0.85,
                             vmin=np.percentile(zs, 10), vmax=np.max(zs))
            plt.colorbar(sc, ax=ax_, label="PR-AUC val")
            ax_.set_xlabel(px, fontsize=10)
            ax_.set_ylabel(py, fontsize=10)
            ax_.set_title(f"{px} × {py}", fontsize=10, fontweight="bold")
            ax_.grid(alpha=0.3)

        fig3.suptitle("Contour plots — PR-AUC val 2021", fontsize=12, y=1.02)
        plt.tight_layout()
        out3 = out_dir / "fig_optuna_contour.png"
        plt.savefig(out3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Guardado: {out3}")
    except Exception as e:
        print(f"[plot] Contour skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  CatBoost GPU — Optimización Bayesiana con Optuna")
    print(f"  Trials: {N_TRIALS}  |  Early stop: {EARLY_STOP} rondas  |  Métrica: PR-AUC val 2021")
    print("=" * 70)

    # ── Cargar datos ──────────────────────────────────────────────────────────
    X_tr, y_tr, X_val, y_val, X_te, y_te, cat_idx, feats = load_splits()

    # ── Crear / cargar estudio Optuna ──────────────────────────────────────────
    study_pkl = OUT_DIR / "optuna_study.pkl"
    trial_log = []

    if study_pkl.exists():
        with open(study_pkl, "rb") as f:
            study = pickle.load(f)
        print(f"[optuna] Estudio cargado: {len(study.trials)} trials previos")
        # Cargar log existente si hay
        log_path = OUT_DIR / "trial_log.json"
        if log_path.exists():
            with open(log_path) as f:
                trial_log = json.load(f)
    else:
        study = optuna.create_study(
            direction  = "maximize",
            sampler    = TPESampler(seed=RANDOM_SEED, n_startup_trials=10),
            pruner     = MedianPruner(n_startup_trials=10, n_warmup_steps=0),
            study_name = STUDY_NAME,
        )
        print("[optuna] Nuevo estudio creado")

    n_remaining = N_TRIALS - len([t for t in study.trials if t.value is not None])
    print(f"[optuna] Trials restantes: {n_remaining}\n")

    objective = make_objective(X_tr, y_tr, X_val, y_val, cat_idx, trial_log)

    t_start = time.time()
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=False)
    t_total = time.time() - t_start

    # Guardar estudio completo
    with open(study_pkl, "wb") as f:
        pickle.dump(study, f)

    # ── Resultados del tuning ─────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"  MEJOR TRIAL: #{best.number}  |  Val PR-AUC = {best.value:.5f}")
    print(f"  Tiempo total tuning: {t_total/3600:.1f}h")
    print("  Parámetros óptimos:")
    for k, v in best.params.items():
        print(f"    {k:25s}: {v}")
    print("=" * 70)

    # ── Reentrenar modelo final con mejores parámetros ─────────────────────────
    print("\n[final] Reentrenando modelo final con mejores parámetros ...")

    # Para el modelo final, usar iter óptimo de early stopping (ya encontrado)
    # sin límite de iteraciones (dejamos que best_params["iterations"] mande)
    pool_tr  = Pool(X_tr,  y_tr,  cat_features=cat_idx)
    pool_val = Pool(X_val, y_val, cat_features=cat_idx)

    final_model = CatBoostClassifier(
        **best.params,
        **FIXED_PARAMS,
        early_stopping_rounds = EARLY_STOP,
    )
    t_fit = time.time()
    final_model.fit(pool_tr, eval_set=pool_val, use_best_model=True, plot=False)
    fit_time = time.time() - t_fit

    best_iter_final = final_model.get_best_iteration()
    print(f"[final] Entrenamiento completado: {fit_time:.0f}s  |  best_iter={best_iter_final}")

    # Guardar modelo
    final_model.save_model(str(OUT_DIR / "best_catboost.cbm"))
    print(f"[final] Modelo guardado: {OUT_DIR / 'best_catboost.cbm'}")

    # ── Evaluar en val 2021 y test 2022 ────────────────────────────────────────
    print("\n[eval] Calculando métricas finales ...")
    y_prob_val = final_model.predict_proba(X_val)[:, 1]
    y_prob_te  = final_model.predict_proba(X_te)[:, 1]

    metrics_val  = full_metrics(y_val, y_prob_val, "val 2021")
    metrics_test = full_metrics(y_te,  y_prob_te,  "test 2022")

    print("\n── Métricas val 2021 (optimización) ────────────────────────────────")
    print(f"  AUC-ROC : {metrics_val['auc_roc']:.4f}  (baseline CB_GPU val: ver JSON)")
    print(f"  PR-AUC  : {metrics_val['pr_auc']:.4f}")
    print(f"  KS      : {metrics_val['ks']:.4f}")
    print(f"  F1      : {metrics_val['f1']:.4f}")
    print(f"  Brier   : {metrics_val['brier']:.4f}")

    print("\n── Métricas test 2022 (evaluación definitiva) ──────────────────────")
    print(f"  AUC-ROC : {metrics_test['auc_roc']:.4f}  (baseline: 0.9116)")
    print(f"  PR-AUC  : {metrics_test['pr_auc']:.4f}  (baseline: 0.7407)")
    print(f"  KS      : {metrics_test['ks']:.4f}  (baseline: 0.6955)")
    print(f"  F1      : {metrics_test['f1']:.4f}  (baseline: 0.6974)")
    print(f"  Brier   : {metrics_test['brier']:.4f}  (baseline: 0.1313)")
    print(f"  Precisión: {metrics_test['precision']:.4f}")
    print(f"  Recall  : {metrics_test['recall']:.4f}")
    print(f"  TP/TN/FP/FN: {metrics_test['tp']}/{metrics_test['tn']}/{metrics_test['fp']}/{metrics_test['fn']}")
    print(f"  Umbral Youden: {metrics_test['threshold']:.4f}")

    # Delta vs baseline
    delta = {
        "auc_roc": metrics_test["auc_roc"] - 0.9116,
        "pr_auc" : metrics_test["pr_auc"]  - 0.7407,
        "ks"     : metrics_test["ks"]      - 0.6955,
        "f1"     : metrics_test["f1"]      - 0.6974,
        "brier"  : 0.1313 - metrics_test["brier"],   # positivo = mejora
    }
    print("\n── Delta vs. Baseline CB_GPU (Cap 5) ──────────────────────────────")
    for k, v in delta.items():
        sign = "+" if v >= 0 else ""
        print(f"  Δ {k:8s}: {sign}{v:+.4f}  {'↑ MEJORA' if v > 0 else '↓ DESMEJORA'}")

    # ── Guardar resultados JSON ────────────────────────────────────────────────
    results = {
        "timestamp"     : datetime.now().isoformat(),
        "study_name"    : STUDY_NAME,
        "n_trials"      : N_TRIALS,
        "total_time_h"  : round(t_total / 3600, 2),
        "best_trial"    : best.number,
        "best_val_prauc": best.value,
        "best_params"   : best.params,
        "best_iter_final": int(best_iter_final),
        "baseline_cap5" : {
            "auc_roc": 0.9116, "pr_auc": 0.7407,
            "ks": 0.6955, "f1": 0.6974, "brier": 0.1313
        },
        "val_2021"      : metrics_val,
        "test_2022"     : metrics_test,
        "delta_vs_baseline": delta,
        "trial_log"     : trial_log,
        "fixed_params"  : {k: v for k, v in FIXED_PARAMS.items()
                           if k not in ("task_type", "devices", "verbose",
                                        "allow_writing_files")},
    }
    results_path = OUT_DIR / "tuning_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[output] Resultados guardados: {results_path}")

    # ── Plots finales ─────────────────────────────────────────────────────────
    print("[plot] Generando figuras ...")
    plot_optimization(study, trial_log, OUT_DIR)

    print("\n" + "=" * 70)
    print("  TUNING COMPLETADO")
    print(f"  Archivos en: {OUT_DIR}")
    print("  - best_catboost.cbm       → modelo listo para Cap 6")
    print("  - tuning_results.json     → métricas + params de todos los trials")
    print("  - optuna_study.pkl        → estudio reanudable con pickle")
    print("  - trial_log.json          → log incremental trial a trial")
    print("  - fig_optuna_history.png  → curva optimización PR-AUC")
    print("  - fig_optuna_importance.png → importancia hiperparámetros")
    print("  - fig_optuna_contour.png  → contour plots")
    print("=" * 70)


if __name__ == "__main__":
    main()
