"""
Benchmark LightGBM + XGBoost — Tesis ITBA
Agrega modelos 7 y 8 al benchmark existente.
Mismos splits temporales y métricas.
"""

import numpy as np
import pandas as pd
import json, time, os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

import lightgbm as lgb
import xgboost as xgb

DATA_PATH = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/raw_data/dataset_tesis_clean.csv'
OUT_DIR   = '/sessions/epic-intelligent-hawking/model_plots5'
RES6_PATH = '/sessions/epic-intelligent-hawking/benchmark6_results.json'
RES8_PATH = '/sessions/epic-intelligent-hawking/benchmark8_results.json'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Métricas ──────────────────────────────────────────────────────────────
def roc_auc(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    tps = np.cumsum(ys); fps = np.cumsum(1-ys)
    tpr = np.concatenate([[0], tps/tps[-1]])
    fpr = np.concatenate([[0], fps/fps[-1]])
    return float(np.trapezoid(tpr, fpr))

def pr_auc(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    tps = np.cumsum(ys)
    ns  = np.arange(1, len(ys)+1)
    prec = np.concatenate([[1], tps/ns])
    rec  = np.concatenate([[0], tps/ys.sum()])
    return float(np.trapezoid(prec, rec))

def ks_stat(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    return float(np.max(np.abs(np.cumsum(ys)/ys.sum() - np.cumsum(1-ys)/(1-ys).sum())))

def brier(y_true, y_score):
    return float(np.mean((y_true - y_score)**2))

def metrics_at_youden(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    tps = np.cumsum(ys); fps = np.cumsum(1-ys)
    j   = np.argmax(tps/ys.sum() - fps/(1-ys).sum())
    thr = float(y_score[idx[j]])
    yp  = (y_score >= thr).astype(int)
    tp  = int(((yp==1)&(y_true==1)).sum()); tn = int(((yp==0)&(y_true==0)).sum())
    fp  = int(((yp==1)&(y_true==0)).sum()); fn = int(((yp==0)&(y_true==1)).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return {'threshold':thr,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'precision':float(prec),'recall':float(rec),'f1':float(f1)}

def full_metrics(y_true, y_score, name):
    m = metrics_at_youden(y_true, y_score)
    return {'model':name, 'auc_roc':round(roc_auc(y_true,y_score),4),
            'pr_auc':round(pr_auc(y_true,y_score),4),
            'ks':round(ks_stat(y_true,y_score),4),
            'brier':round(brier(y_true,y_score),4), **m}

# ── Carga de datos ────────────────────────────────────────────────────────
print("Cargando datos...")
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

df_train = df[df['accident_year'].between(2017,2020)].copy()
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

neg_pos_ratio = (y_train==0).sum() / (y_train==1).sum()
print(f"  scale_pos_weight: {neg_pos_ratio:.3f}")

# ── LightGBM ──────────────────────────────────────────────────────────────
print("\n[7] LightGBM (500 iters, depth=6, lr=0.05, leaf-wise)...")
t1 = time.time()

# LightGBM maneja categoricas nativas con pd.Categorical
X_train_lgb = X_train.copy()
X_val_lgb   = X_val.copy()
X_test_lgb  = X_test.copy()
for col in CAT_COLS:
    cats = X_train_lgb[col].astype('category').cat.categories
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_val_lgb[col]   = pd.Categorical(X_val_lgb[col],   categories=cats)
    X_test_lgb[col]  = pd.Categorical(X_test_lgb[col],  categories=cats)

dtrain_lgb = lgb.Dataset(X_train_lgb, label=y_train)
dval_lgb   = lgb.Dataset(X_val_lgb,   label=y_val, reference=dtrain_lgb)

lgb_params = {
    'objective':        'binary',
    'metric':           'auc',
    'num_leaves':       63,
    'max_depth':        6,
    'learning_rate':    0.05,
    'n_estimators':     500,
    'scale_pos_weight': neg_pos_ratio,
    'min_child_samples':200,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_lambda':       1.0,
    'random_state':     42,
    'n_jobs':           2,
    'verbose':         -1,
}

lgb_callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=True),
    lgb.log_evaluation(period=50)
]

lgb_model = lgb.train(
    lgb_params,
    dtrain_lgb,
    num_boost_round      = 500,
    valid_sets           = [dval_lgb],
    callbacks            = lgb_callbacks,
)

lgb_val_preds  = lgb_model.predict(X_val_lgb)
lgb_test_preds = lgb_model.predict(X_test_lgb)
lgb_val_m  = full_metrics(y_val,  lgb_val_preds,  'LightGBM')
lgb_test_m = full_metrics(y_test, lgb_test_preds, 'LightGBM')
t_lgb = time.time() - t1

print(f"  Tiempo: {t_lgb:.1f}s  |  best_iteration: {lgb_model.best_iteration}")
print(f"  Val  — AUC={lgb_val_m['auc_roc']}  PR={lgb_val_m['pr_auc']}  KS={lgb_val_m['ks']}  F1={lgb_val_m['f1']:.4f}  Brier={lgb_val_m['brier']}")
print(f"  Test — AUC={lgb_test_m['auc_roc']}  PR={lgb_test_m['pr_auc']}  KS={lgb_test_m['ks']}  F1={lgb_test_m['f1']:.4f}  Brier={lgb_test_m['brier']}")

lgb_fi = lgb_model.feature_importance(importance_type='gain')
lgb_fi_dict = dict(zip(FEATURES, lgb_fi.tolist()))
lgb_fi_sorted = sorted(lgb_fi_dict.items(), key=lambda x: -x[1])[:10]
print("  Top-10 LightGBM importances:")
for fname, fval in lgb_fi_sorted:
    print(f"    {fname:35s} {fval:.0f}")

# ── XGBoost ───────────────────────────────────────────────────────────────
print("\n[8] XGBoost (500 iters, depth=6, lr=0.05)...")
t2 = time.time()

# XGBoost: ordinal encoding para categóricas (igual que GBDT custom)
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
    'max_depth':        6,
    'eta':              0.05,
    'scale_pos_weight': neg_pos_ratio,
    'min_child_weight': 200,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'lambda':           1.0,
    'alpha':            0.1,
    'seed':             42,
    'nthread':          2,
}

xgb_evals_result = {}
xgb_model = xgb.train(
    xgb_params,
    dtrain_xgb,
    num_boost_round      = 500,
    evals                = [(dval_xgb, 'validation')],
    early_stopping_rounds= 30,
    evals_result         = xgb_evals_result,
    verbose_eval         = 50,
)

xgb_val_preds  = xgb_model.predict(dval_xgb)
xgb_test_preds = xgb_model.predict(dtest_xgb)
xgb_val_m  = full_metrics(y_val,  xgb_val_preds,  'XGBoost')
xgb_test_m = full_metrics(y_test, xgb_test_preds, 'XGBoost')
t_xgb = time.time() - t2

print(f"  Tiempo: {t_xgb:.1f}s  |  best_iteration: {xgb_model.best_iteration}")
print(f"  Val  — AUC={xgb_val_m['auc_roc']}  PR={xgb_val_m['pr_auc']}  KS={xgb_val_m['ks']}  F1={xgb_val_m['f1']:.4f}  Brier={xgb_val_m['brier']}")
print(f"  Test — AUC={xgb_test_m['auc_roc']}  PR={xgb_test_m['pr_auc']}  KS={xgb_test_m['ks']}  F1={xgb_test_m['f1']:.4f}  Brier={xgb_test_m['brier']}")

xgb_fi_raw = xgb_model.get_score(importance_type='gain')
xgb_fi_dict = {f: xgb_fi_raw.get(f, 0.0) for f in FEATURES}
xgb_fi_sorted = sorted(xgb_fi_dict.items(), key=lambda x: -x[1])[:10]
print("  Top-10 XGBoost importances:")
for fname, fval in xgb_fi_sorted:
    print(f"    {fname:35s} {fval:.0f}")

# ── Fusionar con resultados anteriores ────────────────────────────────────
existing = json.load(open(RES6_PATH))
existing['val']['LGB']  = lgb_val_m
existing['val']['XGB']  = xgb_val_m
existing['test']['LGB'] = lgb_test_m
existing['test']['XGB'] = xgb_test_m
existing['lgb_importances'] = lgb_fi_dict
existing['xgb_importances'] = xgb_fi_dict
existing['lgb_val_aucs'] = lgb_model.evals_result_['valid_0']['auc'] if hasattr(lgb_model, 'evals_result_') else []
existing['xgb_val_aucs'] = xgb_evals_result.get('validation', {}).get('auc', [])
existing.setdefault('timing', {})
existing['timing']['LightGBM'] = round(t_lgb, 1)
existing['timing']['XGBoost']  = round(t_xgb, 1)

json.dump(existing, open(RES8_PATH, 'w'), indent=2)
print(f"\nResultados guardados: {RES8_PATH}")

# ── Tabla resumen 8 modelos ───────────────────────────────────────────────
MODEL_ORDER = ['LR','NB','DT','RF','GB','CB','LGB','XGB']
MODEL_NAMES = {
    'LR':  'Logistic Regression',
    'NB':  'Naive Bayes',
    'DT':  'Decision Tree',
    'RF':  'Random Forest',
    'GB':  'Gradient Boosting',
    'CB':  'CatBoost',
    'LGB': 'LightGBM',
    'XGB': 'XGBoost',
}

test_data = existing['test']
print(f"\n{'='*72}")
print(f"{'RESUMEN FINAL 8 MODELOS — TEST 2022':^72}")
print(f"{'='*72}")
print(f"{'Modelo':22s} {'AUC-ROC':>8} {'PR-AUC':>8} {'KS':>8} {'F1':>8} {'Brier':>8}")
print("-"*72)
for key in MODEL_ORDER:
    if key not in test_data: continue
    m = test_data[key]
    print(f"{MODEL_NAMES[key]:22s} {m['auc_roc']:>8.4f} {m['pr_auc']:>8.4f} {m['ks']:>8.4f} {m['f1']:>8.4f} {m['brier']:>8.4f}")

# ── Figuras actualizadas 8 modelos ────────────────────────────────────────
COLORS8 = ['#2E4057','#8E44AD','#27AE60','#E67E22','#C0392B','#1ABC9C','#3498DB','#F39C12']

print("\nGenerando figuras 8 modelos...")

# Fig 1: Barras comparativas
metrics_list   = ['auc_roc','pr_auc','ks','f1','brier']
metrics_labels = ['AUC-ROC','PR-AUC','KS Statistic','F1 (Youden)','Brier Score']
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
for ax, metric, mlabel in zip(axes, metrics_list, metrics_labels):
    vals, names = [], []
    for key in MODEL_ORDER:
        if key not in test_data: continue
        v = test_data[key].get(metric, test_data[key].get('f1', 0))
        vals.append(v); names.append(MODEL_NAMES[key].replace(' ','\n'))
    best_idx = (np.argmin(vals) if metric=='brier' else np.argmax(vals))
    bars = ax.bar(names, vals, color=COLORS8[:len(vals)], edgecolor='white', linewidth=0.5)
    for j,(bar,val) in enumerate(zip(bars,vals)):
        fw = 'bold' if j==best_idx else 'normal'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight=fw)
    ax.set_title(mlabel, fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(vals)*1.18)
    ax.tick_params(axis='x', labelsize=6.5)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Comparación de Métricas — Test 2022 (8 Modelos)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_metrics_bar_8models.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig3_metrics_bar_8models.png OK")

# Fig 2: Heatmap 8 modelos
val_data = existing['val']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, split_data, split_name in [(axes[0],val_data,'Validación 2021'),(axes[1],test_data,'Test 2022')]:
    keys_p = [k for k in MODEL_ORDER if k in split_data]
    matrix = np.array([[split_data[k]['auc_roc'],split_data[k]['pr_auc'],
                        split_data[k]['ks'],split_data[k]['f1'],split_data[k]['brier']]
                       for k in keys_p])
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['AUC-ROC','PR-AUC','KS','F1','Brier'], fontsize=9)
    ax.set_yticks(range(len(keys_p)))
    ax.set_yticklabels([MODEL_NAMES[k] for k in keys_p], fontsize=9)
    for i in range(len(keys_p)):
        for j in range(5):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=8.5, fontweight='bold',
                    color='white' if matrix[i,j] > 0.7 else 'black')
    ax.set_title(split_name, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle('Métricas por Modelo — 8 Modelos', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_heatmap_8models.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig4_heatmap_8models.png OK")

# Fig 3: Matrices de confusión 8 modelos
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, key in enumerate(MODEL_ORDER):
    if key not in test_data: continue
    m   = test_data[key]
    cm  = np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
    ax  = axes[i//4][i%4]
    im  = ax.imshow(cm, cmap='Blues')
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f'{cm[r,c]:,}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if cm[r,c]>cm.max()*0.5 else 'black')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred Neg','Pred Pos'], fontsize=8)
    ax.set_yticklabels(['Real Neg','Real Pos'], fontsize=8)
    ax.set_title(f"{MODEL_NAMES[key]}\nAUC={m['auc_roc']:.4f}  F1={m['f1']:.4f}", fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.7)
plt.suptitle('Matrices de Confusión — Test Set 2022 (8 Modelos, umbral Youden)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_confusion_8models.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig5_confusion_8models.png OK")

# Fig 4: Curvas de aprendizaje LightGBM + XGBoost
lgb_aucs = existing.get('lgb_val_aucs', [])
xgb_aucs = existing.get('xgb_val_aucs', [])
cb_aucs  = existing.get('cb_val_aucs',  [])
gb_aucs  = existing.get('gb_val_aucs',  [])

fig, ax = plt.subplots(figsize=(10, 5))
for aucs, name, color in [
    (gb_aucs,  'GBDT (custom)',  '#C0392B'),
    (cb_aucs,  'CatBoost',       '#1ABC9C'),
    (lgb_aucs, 'LightGBM',       '#3498DB'),
    (xgb_aucs, 'XGBoost',        '#F39C12'),
]:
    if aucs:
        ax.plot(range(1, len(aucs)+1), aucs, lw=2, label=name, color=color)
        best = int(np.argmax(aucs))+1
        ax.axvline(best, color=color, linestyle='--', alpha=0.4, lw=1)
ax.set_xlabel('Iteración / Árbol', fontsize=11)
ax.set_ylabel('AUC-ROC (Validación 2021)', fontsize=11)
ax.set_title('Curvas de Aprendizaje — Modelos de Boosting', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_learning_curves_boosting.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig7_learning_curves_boosting.png OK")

# Fig 5: Feature importance top-8 comparación 4 boosting models
feat_names = existing.get('feature_names', FEATURES)
gb_imp_list = existing['importances'].get('GB', [])
cb_imp_d    = existing.get('cb_importances', {})
lgb_imp_d   = lgb_fi_dict
xgb_imp_d   = xgb_fi_dict

def norm_list(imp_list, names):
    total = sum(imp_list) or 1
    return {n: v/total*100 for n, v in zip(names, imp_list)}

def norm_dict(d):
    total = sum(d.values()) or 1
    return {k: v/total*100 for k,v in d.items()}

gb_n  = norm_list(gb_imp_list, feat_names) if gb_imp_list else {}
cb_n  = norm_dict(cb_imp_d)
lgb_n = norm_dict(lgb_imp_d)
xgb_n = norm_dict(xgb_imp_d)

all_f = list(set(list(gb_n)+list(cb_n)+list(lgb_n)+list(xgb_n)))
avg_imp = {f:(gb_n.get(f,0)+cb_n.get(f,0)+lgb_n.get(f,0)+xgb_n.get(f,0))/4 for f in all_f}
top15 = [f for f,_ in sorted(avg_imp.items(), key=lambda x:-x[1])[:12]]
top15_rev = list(reversed(top15))

x    = np.arange(len(top15_rev))
w    = 0.2
fig, ax = plt.subplots(figsize=(12, 7))
for j,(imp_n,name,color) in enumerate([
    (gb_n, 'GBDT',     '#C0392B'),
    (cb_n, 'CatBoost', '#1ABC9C'),
    (lgb_n,'LightGBM', '#3498DB'),
    (xgb_n,'XGBoost',  '#F39C12'),
]):
    vals = [imp_n.get(f,0) for f in top15_rev]
    ax.barh(x + (j-1.5)*w, vals, w, label=name, color=color, alpha=0.85)
ax.set_yticks(x); ax.set_yticklabels(top15_rev, fontsize=9)
ax.set_xlabel('Importancia relativa (%)', fontsize=11)
ax.set_title('Importancia de Features — 4 Modelos de Boosting (Top 12)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_importance_4boosting.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig6_importance_4boosting.png OK")

print("\nTodas las figuras generadas.")
print(f"\nLightGBM tiempo: {t_lgb:.0f}s  |  XGBoost tiempo: {t_xgb:.0f}s")
