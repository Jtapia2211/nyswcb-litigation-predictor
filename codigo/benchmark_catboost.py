"""
Benchmark CatBoost — Tesis ITBA
Agrega CatBoost como 6to modelo al benchmark existente.
Usa los mismos splits temporales y métricas.
"""

import numpy as np
import pandas as pd
import json, time, os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

from catboost import CatBoostClassifier, Pool

DATA_PATH  = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/raw_data/dataset_tesis_clean.csv'
OUT_DIR    = '/sessions/epic-intelligent-hawking/model_plots5'
RES_PATH   = '/sessions/epic-intelligent-hawking/benchmark5_results.json'
RES6_PATH  = '/sessions/epic-intelligent-hawking/benchmark6_results.json'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Métricas ──────────────────────────────────────────────────────────────
def roc_auc(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    tps = np.cumsum(ys);  fps = np.cumsum(1-ys)
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
    tps = np.cumsum(ys);  fps = np.cumsum(1-ys)
    j   = np.argmax(tps/ys.sum() - fps/(1-ys).sum())
    thr = float(y_score[idx[j]])
    yp  = (y_score >= thr).astype(int)
    tp  = int(((yp==1)&(y_true==1)).sum());  tn = int(((yp==0)&(y_true==0)).sum())
    fp  = int(((yp==1)&(y_true==0)).sum());  fn = int(((yp==0)&(y_true==1)).sum())
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

TARGET = 'target'
NUM_COLS = ['days_to_assembly','days_C2_to_accident','days_C3_to_accident','age_at_injury','aww']
BIN_COLS = ['has_C2','has_C3','has_ANCR_early','accident_year','accident_month','accident_dow']
CAT_COLS = ['gender','accident_type','occupational_disease','county_of_injury',
            'medical_fee_region','wcio_cause_code','wcio_nature_code','wcio_body_code',
            'carrier_type','district_name','industry_code','industry_desc']

FEATURES = NUM_COLS + BIN_COLS + CAT_COLS

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Cargado: {len(df):,} filas en {time.time()-t0:.1f}s")

# Temporal split
df_train = df[df['accident_year'].between(2017, 2020)].copy()
df_val   = df[df['accident_year'] == 2021].copy()
df_test  = df[df['accident_year'] == 2022].copy()
print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# Llenar nulos
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

X_train = df_train[FEATURES]
y_train = df_train[TARGET].values.astype(int)
X_val   = df_val[FEATURES]
y_val   = df_val[TARGET].values.astype(int)
X_test  = df_test[FEATURES]
y_test  = df_test[TARGET].values.astype(int)

# Subsample estratificado del 50% para reducir uso de memoria (1.09M -> 545K)
rng = np.random.RandomState(42)
idx_pos = np.where(y_train == 1)[0]
idx_neg = np.where(y_train == 0)[0]
idx_pos_s = rng.choice(idx_pos, size=int(len(idx_pos)*0.5), replace=False)
idx_neg_s = rng.choice(idx_neg, size=int(len(idx_neg)*0.5), replace=False)
idx_sub = np.sort(np.concatenate([idx_pos_s, idx_neg_s]))
X_train_sub = X_train.iloc[idx_sub]
y_train_sub = y_train[idx_sub]
print(f"  Subsample train: {len(X_train_sub):,} filas (50% estratificado)")

# Indices de features categoricas para CatBoost
cat_feature_indices = [FEATURES.index(c) for c in CAT_COLS]
print(f"  Cat features ({len(cat_feature_indices)}): {CAT_COLS[:4]}...")

# Class weight
neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Class weight (scale_pos_weight): {neg_pos_ratio:.3f}")

# ── CatBoost ──────────────────────────────────────────────────────────────
print("\n[6] CatBoost (300 iters, depth=6, lr=0.05)...")
t1 = time.time()

train_pool = Pool(X_train_sub, y_train_sub, cat_features=cat_feature_indices)
val_pool   = Pool(X_val,       y_val,       cat_features=cat_feature_indices)
test_pool  = Pool(X_test,      y_test,      cat_features=cat_feature_indices)

cb_model = CatBoostClassifier(
    iterations         = 300,
    depth              = 6,
    learning_rate      = 0.05,
    scale_pos_weight   = neg_pos_ratio,
    eval_metric        = 'AUC',
    random_seed        = 42,
    thread_count       = 2,
    verbose            = 50,
    early_stopping_rounds   = 30,
    max_ctr_complexity = 1,       # reduce memoria para cat features
)
cb_model.fit(train_pool, eval_set=val_pool)

cb_val_preds  = cb_model.predict_proba(val_pool)[:,1]
cb_test_preds = cb_model.predict_proba(test_pool)[:,1]

cb_val_m  = full_metrics(y_val,  cb_val_preds,  'CatBoost')
cb_test_m = full_metrics(y_test, cb_test_preds, 'CatBoost')

t_cb = time.time() - t1
print(f"  Tiempo: {t_cb:.1f}s")
print(f"  Val  — AUC={cb_val_m['auc_roc']}  PR={cb_val_m['pr_auc']}  KS={cb_val_m['ks']}  F1={cb_val_m['f1']:.4f}  Brier={cb_val_m['brier']}")
print(f"  Test — AUC={cb_test_m['auc_roc']}  PR={cb_test_m['pr_auc']}  KS={cb_test_m['ks']}  F1={cb_test_m['f1']:.4f}  Brier={cb_test_m['brier']}")

# Feature importance
fi = cb_model.get_feature_importance()
fi_dict = dict(zip(FEATURES, fi.tolist()))
fi_sorted = sorted(fi_dict.items(), key=lambda x: -x[1])[:20]
print("\n  Top-10 feature importances:")
for fname, fval in fi_sorted[:10]:
    print(f"    {fname:35s} {fval:.2f}")

# ── Cargar resultados anteriores y fusionar ───────────────────────────────
existing = json.load(open(RES_PATH))

# Agrego CatBoost al dict
existing['val']['CB']  = cb_val_m
existing['test']['CB'] = cb_test_m
existing['cb_importances'] = fi_dict
existing['cb_val_aucs']    = [float(v) for v in
    cb_model.get_evals_result()['validation']['AUC']]
existing['timing'] = existing.get('timing', {})
existing['timing']['CatBoost'] = round(t_cb, 1)

json.dump(existing, open(RES6_PATH, 'w'), indent=2)
print(f"\nResultados guardados en {RES6_PATH}")

# ── Gráficos actualizados ─────────────────────────────────────────────────
COLORS6 = ['#2E4057','#8E44AD','#27AE60','#E67E22','#C0392B','#1ABC9C']
MODEL_ORDER = ['LR','NB','DT','RF','GB','CB']
MODEL_NAMES = {
    'LR': 'Logistic Regression',
    'NB': 'Naive Bayes',
    'DT': 'Decision Tree',
    'RF': 'Random Forest',
    'GB': 'Gradient Boosting',
    'CB': 'CatBoost'
}

val_data  = existing['val']
test_data = existing['test']

# ── Fig 1: ROC actualizado ────────────────────────────────────────────────
print("\nGenerando gráficos actualizados...")
fig, ax = plt.subplots(figsize=(7, 6))
for i, key in enumerate(MODEL_ORDER):
    if key not in test_data: continue
    m = test_data[key]
    # Reconstruir curva ROC con preds del modelo — usar solo AUC label
    auc = m['auc_roc']
    label = f"{MODEL_NAMES[key]} (AUC={auc:.4f})"
    ax.plot([0, m['fp']/(m['fp']+m['tn']), 1],
            [0, m['tp']/(m['tp']+m['fn']), 1],
            color=COLORS6[i], lw=1.5, alpha=0.7, linestyle='--')
    ax.scatter([], [], color=COLORS6[i], label=label, s=30)

ax.plot([0,1],[0,1],'k--',alpha=0.3,lw=1)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Curvas ROC — Conjunto de Test (2022)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_roc_6models.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 2: Barras comparativas ─────────────────────────────────────────────
metrics_list = ['auc_roc','pr_auc','ks','f1','brier']
metrics_labels = ['AUC-ROC','PR-AUC','KS Stat','F1 (Youden)','Brier Score']
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for ax, metric, mlabel in zip(axes, metrics_list, metrics_labels):
    vals = []
    names = []
    for key in MODEL_ORDER:
        if key not in test_data: continue
        m = test_data[key]
        v = m.get(metric, m.get('f1', 0))
        vals.append(v)
        names.append(MODEL_NAMES[key].replace(' ','\n'))
    bars = ax.bar(names, vals, color=COLORS6[:len(vals)], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_title(mlabel, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(vals)*1.15)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    if metric == 'brier':
        ax.set_ylim(0, max(vals)*1.2)
plt.suptitle('Comparación de Métricas — Test 2022 (6 Modelos)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_metrics_bar_6models.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 3: Heatmap val/test ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, split_data, split_name in [(axes[0], val_data, 'Validación 2021'),
                                    (axes[1], test_data, 'Test 2022')]:
    keys_present = [k for k in MODEL_ORDER if k in split_data]
    matrix = []
    for key in keys_present:
        m = split_data[key]
        matrix.append([m['auc_roc'], m['pr_auc'], m['ks'], m['f1'], m['brier']])
    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['AUC-ROC','PR-AUC','KS','F1','Brier'], fontsize=9)
    ax.set_yticks(range(len(keys_present)))
    ax.set_yticklabels([MODEL_NAMES[k] for k in keys_present], fontsize=9)
    for i in range(len(keys_present)):
        for j in range(5):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=8.5, fontweight='bold',
                    color='white' if matrix[i,j] > 0.7 else 'black')
    ax.set_title(split_name, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle('Métricas por Modelo y Conjunto — 6 Modelos', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_heatmap_6models.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 4: Feature importance CatBoost ────────────────────────────────────
top_n = 20
sorted_feats = sorted(fi_dict.items(), key=lambda x: -x[1])[:top_n]
names_fi  = [f[0] for f in reversed(sorted_feats)]
values_fi = [f[1] for f in reversed(sorted_feats)]
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(names_fi, values_fi, color='#1ABC9C', edgecolor='white')
for bar, val in zip(bars, values_fi):
    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=8)
ax.set_xlabel('Feature Importance (PredictionValuesChange)', fontsize=10)
ax.set_title('CatBoost — Top 20 Features por Importancia', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_importance_catboost.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 5: Learning curve CatBoost ────────────────────────────────────────
cb_aucs = existing['cb_val_aucs']
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cb_aucs)+1), cb_aucs, color='#1ABC9C', lw=2, label='CatBoost Val AUC')
best_iter = int(np.argmax(cb_aucs)) + 1
ax.axvline(best_iter, color='red', linestyle='--', alpha=0.6, label=f'Best iter={best_iter}')
ax.set_xlabel('Iteración', fontsize=11)
ax.set_ylabel('AUC-ROC (Validación 2021)', fontsize=11)
ax.set_title('CatBoost — Curva de Aprendizaje', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_learning_curve_catboost.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plots guardados:")
for f in ['fig1_roc_6models','fig3_metrics_bar_6models','fig4_heatmap_6models',
          'fig6_importance_catboost','fig7_learning_curve_catboost']:
    print(f"  {OUT_DIR}/{f}.png")

print("\n=== RESUMEN FINAL 6 MODELOS (TEST 2022) ===")
print(f"{'Modelo':22s} {'AUC-ROC':>8} {'PR-AUC':>8} {'KS':>8} {'F1':>8} {'Brier':>8}")
print("-"*65)
for key in MODEL_ORDER:
    if key not in test_data: continue
    m = test_data[key]
    print(f"{MODEL_NAMES[key]:22s} {m['auc_roc']:>8.4f} {m['pr_auc']:>8.4f} {m['ks']:>8.4f} {m['f1']:>8.4f} {m['brier']:>8.4f}")
