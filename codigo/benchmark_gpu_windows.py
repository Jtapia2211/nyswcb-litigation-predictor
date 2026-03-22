"""
Benchmark GPU — Windows — Tesis ITBA
LightGBM + XGBoost + CatBoost con RTX, dataset completo.

Uso:
    cd "C:\\Users\\julia\\Tesis_ML"
    venv\\Scripts\\activate
    python codigo\\benchmark_gpu_windows.py
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# ── Rutas ─────────────────────────────────────────────────────────────────
BASE_DIR  = r'C:\Users\julia\Tesis_ML'
DATA_PATH = os.path.join(BASE_DIR, 'raw_data', 'dataset_tesis_clean.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'codigo', 'model_plots5')
RES_IN    = os.path.join(BASE_DIR, 'codigo', 'benchmark8_results.json')
RES_OUT   = os.path.join(BASE_DIR, 'codigo', 'benchmark_gpu_results.json')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"BASE_DIR: {BASE_DIR}")

# ── Métricas ──────────────────────────────────────────────────────────────
def roc_auc(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    tps = np.cumsum(ys); fps = np.cumsum(1 - ys)
    tpr = np.concatenate([[0], tps / tps[-1]])
    fpr = np.concatenate([[0], fps / fps[-1]])
    return float(np.trapezoid(tpr, fpr))

def pr_auc(y_true, y_score):
    idx  = np.argsort(-y_score)
    ys   = y_true[idx]
    tps  = np.cumsum(ys)
    ns   = np.arange(1, len(ys) + 1)
    prec = np.concatenate([[1], tps / ns])
    rec  = np.concatenate([[0], tps / ys.sum()])
    return float(np.trapezoid(prec, rec))

def ks_stat(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    return float(np.max(np.abs(
        np.cumsum(ys) / ys.sum() - np.cumsum(1 - ys) / (1 - ys).sum()
    )))

def brier(y_true, y_score):
    return float(np.mean((y_true - y_score) ** 2))

def metrics_at_youden(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    tps = np.cumsum(ys); fps = np.cumsum(1 - ys)
    j   = np.argmax(tps / ys.sum() - fps / (1 - ys).sum())
    thr = float(y_score[idx[j]])
    yp  = (y_score >= thr).astype(int)
    tp = int(((yp==1)&(y_true==1)).sum()); tn = int(((yp==0)&(y_true==0)).sum())
    fp = int(((yp==1)&(y_true==0)).sum()); fn = int(((yp==0)&(y_true==1)).sum())
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return {'threshold':thr,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'precision':float(prec),'recall':float(rec),'f1':float(f1)}

def full_metrics(y_true, y_score, name):
    m = metrics_at_youden(y_true, y_score)
    return {'model':name,
            'auc_roc': round(roc_auc(y_true, y_score), 4),
            'pr_auc':  round(pr_auc(y_true, y_score),  4),
            'ks':      round(ks_stat(y_true, y_score),  4),
            'brier':   round(brier(y_true, y_score),    4),
            **m}

# ── Carga de datos ────────────────────────────────────────────────────────
print("\nCargando datos...")
t0 = time.time()

TARGET   = 'target'
NUM_COLS = ['days_to_assembly','days_C2_to_accident','days_C3_to_accident','age_at_injury','aww']
BIN_COLS = ['has_C2','has_C3','has_ANCR_early','accident_year','accident_month','accident_dow']
CAT_COLS = ['gender','accident_type','occupational_disease','county_of_injury',
            'medical_fee_region','wcio_cause_code','wcio_nature_code','wcio_body_code',
            'carrier_type','district_name','industry_code','industry_desc']
FEATURES = NUM_COLS + BIN_COLS + CAT_COLS

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  {len(df):,} filas en {time.time()-t0:.1f}s")

df_train = df[df['accident_year'].between(2017, 2020)].copy()
df_val   = df[df['accident_year'] == 2021].copy()
df_test  = df[df['accident_year'] == 2022].copy()
print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

for col in NUM_COLS:
    med = df_train[col].median()
    for d in [df_train, df_val, df_test]:
        d[col] = d[col].fillna(med)
for col in BIN_COLS:
    for d in [df_train, df_val, df_test]:
        d[col] = d[col].fillna(0)
for col in CAT_COLS:
    for d in [df_train, df_val, df_test]:
        d[col] = d[col].fillna('UNKNOWN').astype(str)

X_train = df_train[FEATURES]; y_train = df_train[TARGET].values.astype(int)
X_val   = df_val[FEATURES];   y_val   = df_val[TARGET].values.astype(int)
X_test  = df_test[FEATURES];  y_test  = df_test[TARGET].values.astype(int)

neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  scale_pos_weight: {neg_pos_ratio:.3f}")

# ── LightGBM GPU ──────────────────────────────────────────────────────────
print("\n[7] LightGBM CPU (1000 iters, num_leaves=127, lr=0.05)...")
t1 = time.time()

X_train_lgb = X_train.copy()
X_val_lgb   = X_val.copy()
X_test_lgb  = X_test.copy()
for col in CAT_COLS:
    cats = X_train_lgb[col].astype('category').cat.categories
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_val_lgb[col]   = pd.Categorical(X_val_lgb[col],  categories=cats)
    X_test_lgb[col]  = pd.Categorical(X_test_lgb[col], categories=cats)

dtrain_lgb = lgb.Dataset(X_train_lgb, label=y_train)
dval_lgb   = lgb.Dataset(X_val_lgb,   label=y_val, reference=dtrain_lgb)

lgb_params = {
    'objective':        'binary',
    'metric':           'auc',
    'num_leaves':       127,   # más que en Linux (63) — aprovecha más RAM/CPU
    'learning_rate':    0.05,
    'scale_pos_weight': neg_pos_ratio,
    'min_child_samples':200,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_lambda':       1.0,
    'random_state':     42,
    'n_jobs':           -1,    # todos los cores
    'verbose':         -1,
}

lgb_evals_result = {}
lgb_model = lgb.train(
    lgb_params,
    dtrain_lgb,
    num_boost_round = 1000,
    valid_sets      = [dval_lgb],
    valid_names     = ['val'],
    callbacks       = [
        lgb.record_evaluation(lgb_evals_result),
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100),
    ],
)

lgb_val_preds  = lgb_model.predict(X_val_lgb)
lgb_test_preds = lgb_model.predict(X_test_lgb)
lgb_val_m  = full_metrics(y_val,  lgb_val_preds,  'LightGBM-GPU')
lgb_test_m = full_metrics(y_test, lgb_test_preds, 'LightGBM-GPU')
t_lgb = time.time() - t1

print(f"  Tiempo: {t_lgb:.1f}s  |  best_iter: {lgb_model.best_iteration}")
print(f"  Val  — AUC={lgb_val_m['auc_roc']}  PR={lgb_val_m['pr_auc']}  F1={lgb_val_m['f1']:.4f}  Brier={lgb_val_m['brier']}")
print(f"  Test — AUC={lgb_test_m['auc_roc']}  PR={lgb_test_m['pr_auc']}  F1={lgb_test_m['f1']:.4f}  Brier={lgb_test_m['brier']}")

# ── XGBoost GPU ───────────────────────────────────────────────────────────
print("\n[8] XGBoost GPU (1000 iters, depth=6, lr=0.05)...")
t2 = time.time()

X_train_xgb = X_train.copy()
X_val_xgb   = X_val.copy()
X_test_xgb  = X_test.copy()
for col in CAT_COLS:
    cats = {v: i for i, v in enumerate(X_train_xgb[col].value_counts().index)}
    X_train_xgb[col] = X_train_xgb[col].map(cats).fillna(len(cats)).astype(int)
    X_val_xgb[col]   = X_val_xgb[col].map(cats).fillna(len(cats)).astype(int)
    X_test_xgb[col]  = X_test_xgb[col].map(cats).fillna(len(cats)).astype(int)

dtrain_xgb = xgb.DMatrix(X_train_xgb, label=y_train)
dval_xgb   = xgb.DMatrix(X_val_xgb,   label=y_val)
dtest_xgb  = xgb.DMatrix(X_test_xgb,  label=y_test)

xgb_params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'device':           'cuda',
    'max_depth':        6,
    'eta':              0.05,
    'scale_pos_weight': neg_pos_ratio,
    'min_child_weight': 200,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'lambda':           1.0,
    'alpha':            0.1,
    'seed':             42,
}

xgb_evals_result = {}
xgb_model = xgb.train(
    xgb_params,
    dtrain_xgb,
    num_boost_round       = 1000,
    evals                 = [(dval_xgb, 'validation')],
    early_stopping_rounds = 50,
    evals_result          = xgb_evals_result,
    verbose_eval          = 100,
)

xgb_val_preds  = xgb_model.predict(dval_xgb)
xgb_test_preds = xgb_model.predict(dtest_xgb)
xgb_val_m  = full_metrics(y_val,  xgb_val_preds,  'XGBoost-GPU')
xgb_test_m = full_metrics(y_test, xgb_test_preds, 'XGBoost-GPU')
t_xgb = time.time() - t2

print(f"  Tiempo: {t_xgb:.1f}s  |  best_iter: {xgb_model.best_iteration}")
print(f"  Val  — AUC={xgb_val_m['auc_roc']}  PR={xgb_val_m['pr_auc']}  F1={xgb_val_m['f1']:.4f}  Brier={xgb_val_m['brier']}")
print(f"  Test — AUC={xgb_test_m['auc_roc']}  PR={xgb_test_m['pr_auc']}  F1={xgb_test_m['f1']:.4f}  Brier={xgb_test_m['brier']}")

# ── CatBoost GPU — dataset completo ───────────────────────────────────────
print("\n[6] CatBoost GPU — dataset COMPLETO (500 iters, depth=8, lr=0.05)...")
t3 = time.time()

cat_feature_indices = [FEATURES.index(c) for c in CAT_COLS]
train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_feature_indices)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_feature_indices)

cb_model = CatBoostClassifier(
    iterations            = 500,
    depth                 = 8,
    learning_rate         = 0.05,
    scale_pos_weight      = neg_pos_ratio,
    eval_metric           = 'AUC',
    task_type             = 'GPU',
    random_seed           = 42,
    verbose               = 100,
    early_stopping_rounds = 50,
)
cb_model.fit(train_pool, eval_set=val_pool)

cb_val_preds  = cb_model.predict_proba(val_pool)[:,1]
cb_test_preds = cb_model.predict_proba(test_pool)[:,1]
cb_val_m  = full_metrics(y_val,  cb_val_preds,  'CatBoost-GPU')
cb_test_m = full_metrics(y_test, cb_test_preds, 'CatBoost-GPU')
t_cb = time.time() - t3

print(f"  Tiempo: {t_cb:.1f}s")
print(f"  Val  — AUC={cb_val_m['auc_roc']}  PR={cb_val_m['pr_auc']}  F1={cb_val_m['f1']:.4f}  Brier={cb_val_m['brier']}")
print(f"  Test — AUC={cb_test_m['auc_roc']}  PR={cb_test_m['pr_auc']}  F1={cb_test_m['f1']:.4f}  Brier={cb_test_m['brier']}")

# ── Guardar resultados ────────────────────────────────────────────────────
with open(RES_IN, 'r', encoding='utf-8') as f:
    existing = json.load(f)
existing['val']['LGB_GPU']  = lgb_val_m
existing['val']['XGB_GPU']  = xgb_val_m
existing['val']['CB_GPU']   = cb_val_m
existing['test']['LGB_GPU'] = lgb_test_m
existing['test']['XGB_GPU'] = xgb_test_m
existing['test']['CB_GPU']  = cb_test_m
existing.setdefault('timing', {})
existing['timing']['LightGBM-GPU'] = round(t_lgb, 1)
existing['timing']['XGBoost-GPU']  = round(t_xgb, 1)
existing['timing']['CatBoost-GPU'] = round(t_cb,  1)
existing['lgb_gpu_val_aucs'] = lgb_evals_result.get('val', {}).get('auc', [])
existing['xgb_gpu_val_aucs'] = xgb_evals_result.get('validation', {}).get('auc', [])
existing['cb_gpu_val_aucs']  = [float(v) for v in
    cb_model.get_evals_result()['validation']['AUC']]

with open(RES_OUT, 'w', encoding='utf-8') as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)
print(f"\nResultados guardados: {RES_OUT}")

# Verificar que se escribió correctamente
import os
size = os.path.getsize(RES_OUT)
print(f"Tamaño del archivo: {size:,} bytes")
if size < 1000:
    print("ERROR: archivo muy pequeño, algo salió mal en la escritura")
else:
    print("OK: archivo guardado correctamente")

# ── Resumen final ─────────────────────────────────────────────────────────
MODEL_ORDER = ['LR','NB','DT','RF','GB','CB','LGB','XGB','CB_GPU','LGB_GPU','XGB_GPU']
MODEL_NAMES = {
    'LR':     'Logistic Regression',
    'NB':     'Naive Bayes',
    'DT':     'Decision Tree',
    'RF':     'Random Forest',
    'GB':     'GBDT (custom)',
    'CB':     'CatBoost  (CPU 50%)',
    'LGB':    'LightGBM  (CPU)',
    'XGB':    'XGBoost   (CPU)',
    'CB_GPU': 'CatBoost  (GPU full)',
    'LGB_GPU':'LightGBM  (CPU full)',
    'XGB_GPU':'XGBoost   (GPU)',
}

test_data = existing['test']
print(f"\n{'='*72}")
print(f"{'RESUMEN COMPLETO — TEST 2022':^72}")
print(f"{'='*72}")
print(f"{'Modelo':28s} {'AUC-ROC':>8} {'PR-AUC':>8} {'KS':>8} {'F1':>8} {'Brier':>8}")
print("-"*72)
for key in MODEL_ORDER:
    if key not in test_data:
        continue
    m = test_data[key]
    gpu = ' ◄ GPU' if key in ('CB_GPU','LGB_GPU','XGB_GPU') else ''
    print(f"{MODEL_NAMES[key]:28s} {m['auc_roc']:>8.4f} {m['pr_auc']:>8.4f} "
          f"{m['ks']:>8.4f} {m['f1']:>8.4f} {m['brier']:>8.4f}{gpu}")

print(f"\nTiempos GPU: LightGBM={t_lgb:.0f}s | XGBoost={t_xgb:.0f}s | CatBoost={t_cb:.0f}s")
print("Listo. Copiá benchmark_gpu_results.json de vuelta al chat.")
