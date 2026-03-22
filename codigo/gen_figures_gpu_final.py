"""Genera figuras finales con 9 modelos GPU benchmark + MLP."""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

JSON_PATH = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/codigo/benchmark_gpu_results.json'
OUT_DIR   = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/codigo/model_plots5'
os.makedirs(OUT_DIR, exist_ok=True)

d = json.load(open(JSON_PATH, encoding='utf-8'))

# Final 9-model order: 5 base models + MLP + 3 GPU boosting
MODEL_ORDER  = ['LR','NB','DT','RF','GB','MLP','LGB_GPU','XGB_GPU','CB_GPU']
MODEL_LABELS = {
    'LR':     'Log. Regression',
    'NB':     'Naive Bayes',
    'DT':     'Decision Tree',
    'RF':     'Random Forest',
    'GB':     'GBDT (custom)',
    'MLP':    'MLP\n(256-128-64)',
    'LGB_GPU':'LightGBM',
    'XGB_GPU':'XGBoost',
    'CB_GPU': 'CatBoost',
}
COLORS9 = ['#2E4057','#8E44AD','#27AE60','#E67E22','#C0392B','#E74C3C','#3498DB','#F39C12','#1ABC9C']
COLORS8 = COLORS9  # alias

test_data = d['test']
val_data  = d['val']

# ── Fig 1: Bar chart 8 models ─────────────────────────────────────────────────
metrics_list   = ['auc_roc','pr_auc','ks','f1','brier']
metrics_labels = ['AUC-ROC','PR-AUC','KS Statistic','F1 (Youden)','Brier Score']
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
for ax, metric, mlabel in zip(axes, metrics_list, metrics_labels):
    vals, names = [], []
    for key in MODEL_ORDER:
        if key not in test_data: continue
        v = test_data[key].get(metric, 0)
        vals.append(v)
        names.append(MODEL_LABELS[key].replace(' ', '\n'))
    best_idx = (np.argmin(vals) if metric == 'brier' else np.argmax(vals))
    bars = ax.bar(names, vals, color=COLORS8[:len(vals)], edgecolor='white', linewidth=0.5)
    for j, (bar, val) in enumerate(zip(bars, vals)):
        fw = 'bold' if j == best_idx else 'normal'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight=fw)
    ax.set_title(mlabel, fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(vals)*1.18)
    ax.tick_params(axis='x', labelsize=6.5)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Comparación de Métricas — Test 2022 (9 Modelos)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_metrics_bar_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig3_metrics_bar_final.png OK")

# ── Fig 2: Heatmap val/test ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, split_data, split_name in [(axes[0], val_data, 'Validación 2021'),
                                    (axes[1], test_data, 'Test 2022')]:
    keys_p = [k for k in MODEL_ORDER if k in split_data]
    matrix = np.array([[split_data[k]['auc_roc'], split_data[k]['pr_auc'],
                        split_data[k]['ks'],      split_data[k]['f1'],
                        split_data[k]['brier']]   for k in keys_p])
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['AUC-ROC','PR-AUC','KS','F1','Brier'], fontsize=9)
    ax.set_yticks(range(len(keys_p)))
    ax.set_yticklabels([MODEL_LABELS[k] for k in keys_p], fontsize=9)
    for i in range(len(keys_p)):
        for j in range(5):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=8.5, fontweight='bold',
                    color='white' if matrix[i,j] > 0.7 else 'black')
    ax.set_title(split_name, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle('Métricas por Modelo — Benchmark Final (9 Modelos)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_heatmap_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig4_heatmap_final.png OK")

# ── Fig 3: Confusion matrices ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes_flat = axes.ravel()
for i, key in enumerate(MODEL_ORDER):
    if key not in test_data: continue
    m  = test_data[key]
    cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    ax = axes_flat[i]
    im = ax.imshow(cm, cmap='Blues')
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f'{cm[r,c]:,}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if cm[r,c] > cm.max()*0.5 else 'black')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred Neg','Pred Pos'], fontsize=8)
    ax.set_yticklabels(['Real Neg','Real Pos'], fontsize=8)
    ax.set_title(f"{MODEL_LABELS[key]}\nAUC={m['auc_roc']:.4f}  F1={m['f1']:.4f}",
                 fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.7)
plt.suptitle('Matrices de Confusión — Test Set 2022 (Benchmark Final: 9 Modelos)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_confusion_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig5_confusion_final.png OK")

# ── Fig 4: Feature importance (top-20 LightGBM GPU) ─────────────────────────
feat_names = d.get('feature_names', [])
lgb_imp    = d.get('lgb_importances', [])
xgb_imp    = d.get('xgb_importances', [])
cb_imp     = d.get('cb_importances', [])

def imp_to_array(imp, feat_names):
    """Handle dict {feat: val} or list format."""
    if isinstance(imp, dict):
        return np.array([imp.get(f, 0.0) for f in feat_names], dtype=float)
    return np.array(imp, dtype=float)

if feat_names and lgb_imp:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    imp_data = [
        (lgb_imp, 'LightGBM (Full, CPU)', '#3498DB'),
        (xgb_imp, 'XGBoost (GPU)',        '#F39C12'),
        (cb_imp,  'CatBoost (GPU)',        '#1ABC9C'),
    ]
    for ax, (imp, label, color) in zip(axes, imp_data):
        if not imp:
            ax.set_visible(False); continue
        imp_arr = imp_to_array(imp, feat_names)
        top20 = np.argsort(imp_arr)[-20:][::-1]
        feats_top = [feat_names[j] for j in top20]
        vals_top  = imp_arr[top20]
        vals_top  = vals_top / vals_top.sum() * 100
        ax.barh(range(20), vals_top[::-1], color=color, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(20))
        ax.set_yticklabels(feats_top[::-1], fontsize=7.5)
        ax.set_xlabel('Importancia relativa (%)', fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    plt.suptitle('Top-20 Variables Más Importantes — Modelos de Boosting',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_importance_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("fig6_importance_final.png OK")
else:
    print("fig6_importance_final.png SKIPPED (no feature importance data)")

# ── Fig 5a: Learning curves — boosting ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

ax = axes[0]
curve_map_boost = [
    ('gb_val_aucs',     'GBDT (custom)',  '#C0392B'),
    ('cb_gpu_val_aucs', 'CatBoost (GPU)', '#1ABC9C'),
    ('lgb_gpu_val_aucs','LightGBM',       '#3498DB'),
    ('xgb_gpu_val_aucs','XGBoost (GPU)',  '#F39C12'),
]
for key, label, color in curve_map_boost:
    aucs = d.get(key, [])
    if not aucs: continue
    ax.plot(range(1, len(aucs)+1), aucs, lw=2, label=label, color=color)
    best = int(np.argmax(aucs)) + 1
    ax.axvline(best, color=color, linestyle='--', alpha=0.4, lw=1.2)
ax.set_xlabel('Iteración', fontsize=11)
ax.set_ylabel('AUC-ROC (Validación 2021)', fontsize=11)
ax.set_title('Modelos de Boosting', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Fig 5b: MLP learning curve (epochs)
ax2 = axes[1]
mlp_aucs = d.get('mlp_val_aucs', [])
if mlp_aucs:
    ax2.plot(range(1, len(mlp_aucs)+1), mlp_aucs, lw=2.5, color='#E74C3C', label='MLP (256-128-64)')
    best_ep = int(np.argmax(mlp_aucs)) + 1
    ax2.axvline(best_ep, color='#E74C3C', linestyle='--', alpha=0.5, lw=1.5, label=f'best epoch={best_ep}')
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel('AUC-ROC (Validación 2021)', fontsize=11)
    ax2.set_title('MLP — Curva de Aprendizaje', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.suptitle('Curvas de Aprendizaje — Benchmark Final (9 Modelos)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_learning_final.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"fig7_learning_final.png OK")

print("\nFiguras finales generadas correctamente.")
