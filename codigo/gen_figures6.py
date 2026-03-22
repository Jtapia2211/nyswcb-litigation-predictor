"""Genera figuras actualizadas para los 6 modelos del benchmark."""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = '/sessions/epic-intelligent-hawking/model_plots5'
d = json.load(open('/sessions/epic-intelligent-hawking/benchmark6_results.json'))

COLORS6 = ['#2E4057','#8E44AD','#27AE60','#E67E22','#C0392B','#1ABC9C']
MODEL_ORDER = ['LR','NB','DT','RF','GB','CB']
MODEL_NAMES = {
    'LR': 'Logistic Regression',
    'NB': 'Naive Bayes',
    'DT': 'Decision Tree',
    'RF': 'Random Forest',
    'GB': 'Grad. Boosting',
    'CB': 'CatBoost'
}

val_data  = d['val']
test_data = d['test']

# ── Fig 1: Confusion matrices 6 modelos ───────────────────────────────────
fig, axes = plt.subplots(1, 6, figsize=(22, 4))
for i, key in enumerate(MODEL_ORDER):
    if key not in test_data: continue
    m = test_data[key]
    cm = np.array([[m['tn'], m['fp']],
                   [m['fn'], m['tp']]])
    ax = axes[i]
    im = ax.imshow(cm, cmap='Blues')
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f'{cm[r,c]:,}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if cm[r,c] > cm.max()*0.5 else 'black')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred Neg','Pred Pos'], fontsize=8)
    ax.set_yticklabels(['Actual Neg','Actual Pos'], fontsize=8)
    ax.set_title(f"{MODEL_NAMES[key]}\nAUC={m['auc_roc']:.4f}", fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.7)
plt.suptitle('Matrices de Confusión — Test Set 2022 (6 Modelos, umbral Youden)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_confusion_6models.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig5_confusion_6models.png OK")

# ── Fig 2: Barras comparativas 6 modelos ─────────────────────────────────
metrics_list = ['auc_roc','pr_auc','ks','f1','brier']
metrics_labels = ['AUC-ROC','PR-AUC','KS Statistic','F1 (Youden)','Brier Score']
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, metric, mlabel in zip(axes, metrics_list, metrics_labels):
    vals, names = [], []
    for key in MODEL_ORDER:
        if key not in test_data: continue
        m = test_data[key]
        v = m.get(metric, m.get('f1', 0))
        vals.append(v)
        names.append(MODEL_NAMES[key].replace(' ', '\n'))
    best_idx = (np.argmin(vals) if metric == 'brier' else np.argmax(vals))
    bars = ax.bar(names, vals, color=COLORS6[:len(vals)],
                  edgecolor='white', linewidth=0.5)
    for j, (bar, val) in enumerate(zip(bars, vals)):
        fw = 'bold' if j == best_idx else 'normal'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight=fw)
    ax.set_title(mlabel, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(vals)*1.18)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Comparación de Métricas — Test 2022 (6 Modelos)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_metrics_bar_6models.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig3_metrics_bar_6models.png OK")

# ── Fig 3: Score distribution CatBoost vs GBDT ────────────────────────────
# Regenerate from CatBoost learning curve data
cb_aucs = d['cb_val_aucs']
gb_aucs = d.get('gb_val_aucs', [])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(range(1, len(cb_aucs)+1), cb_aucs, color='#1ABC9C', lw=2, label='CatBoost')
best_cb = int(np.argmax(cb_aucs)) + 1
ax.axvline(best_cb, color='#1ABC9C', linestyle='--', alpha=0.6, lw=1.5,
           label=f'CatBoost best={best_cb}')
if gb_aucs:
    ax.plot(range(1, len(gb_aucs)+1), gb_aucs, color='#C0392B', lw=2, label='GBDT')
    best_gb = int(np.argmax(gb_aucs)) + 1
    ax.axvline(best_gb, color='#C0392B', linestyle='--', alpha=0.6, lw=1.5,
               label=f'GBDT best={best_gb}')
ax.set_xlabel('Iteración / Árbol', fontsize=11)
ax.set_ylabel('AUC-ROC (Validación 2021)', fontsize=11)
ax.set_title('Curvas de Aprendizaje — CatBoost vs GBDT', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Radar chart - all 6 models
ax2 = axes[1]
categories = ['AUC-ROC', 'PR-AUC', 'KS', 'F1', '1-Brier']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax2 = plt.subplot(122, polar=True)
for i, key in enumerate(MODEL_ORDER):
    if key not in test_data: continue
    m = test_data[key]
    vals = [m['auc_roc'], m['pr_auc'], m['ks'], m['f1'], 1 - m['brier']]
    vals += vals[:1]
    ax2.plot(angles, vals, color=COLORS6[i], lw=2, label=MODEL_NAMES[key])
    ax2.fill(angles, vals, color=COLORS6[i], alpha=0.05)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylim(0.5, 1.0)
ax2.set_title('Comparación Multidimensional\n(Test 2022)', fontsize=11, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig8_learning_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig8_learning_radar.png OK")

# ── Fig 4: Feature importance comparison ──────────────────────────────────
# Top features across models from importances dict
feat_names = d.get('feature_names', [])
gb_imp_list = d['importances'].get('GB', []) or d['importances'].get('GBDT', [])
cb_imp_d = d.get('cb_importances', {})

# GBDT importances as dict
gb_imp = {}
if gb_imp_list and feat_names:
    gb_imp = dict(zip(feat_names, gb_imp_list))

# Normalize
def normalize(imp_dict):
    total = sum(imp_dict.values()) or 1
    return {k: v/total*100 for k, v in imp_dict.items()}

gb_n = normalize(gb_imp)
cb_n = normalize(cb_imp_d)

# Top 15 by average rank
all_features = list(set(list(gb_n.keys()) + list(cb_n.keys())))
avg_imp = {f: (gb_n.get(f, 0) + cb_n.get(f, 0)) / 2 for f in all_features}
top_feats = sorted(avg_imp.items(), key=lambda x: -x[1])[:12]
top_names = [f[0] for f in reversed(top_feats)]

x = np.arange(len(top_names))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 7))
bars1 = ax.barh(x - width/2, [gb_n.get(f, 0) for f in top_names], width,
                label='Gradient Boosting', color='#C0392B', alpha=0.85)
bars2 = ax.barh(x + width/2, [cb_n.get(f, 0) for f in top_names], width,
                label='CatBoost', color='#1ABC9C', alpha=0.85)
ax.set_yticks(x); ax.set_yticklabels(top_names, fontsize=9)
ax.set_xlabel('Importancia relativa (%)', fontsize=11)
ax.set_title('Importancia de Features: GBDT vs CatBoost\n(Top 12, normalizado)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig6_importance_comparison.png OK")

print("\nTodas las figuras generadas.")
