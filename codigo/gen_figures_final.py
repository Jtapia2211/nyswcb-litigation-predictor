"""Genera figuras finales con todos los modelos incluyendo GPU."""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/sessions/epic-intelligent-hawking/model_plots5'
d = json.load(open('/sessions/epic-intelligent-hawking/benchmark_gpu_results.json'))

# Orden final — usamos las versiones GPU/full para boosting
MODEL_ORDER  = ['LR','NB','DT','RF','GB','LGB_GPU','XGB_GPU','CB_GPU']
MODEL_LABELS = {
    'LR':     'Log. Regression',
    'NB':     'Naive Bayes',
    'DT':     'Decision Tree',
    'RF':     'Random Forest',
    'GB':     'GBDT (custom)',
    'LGB_GPU':'LightGBM',
    'XGB_GPU':'XGBoost',
    'CB_GPU': 'CatBoost',
}
COLORS8 = ['#2E4057','#8E44AD','#27AE60','#E67E22','#C0392B','#3498DB','#F39C12','#1ABC9C']

test_data = d['test']
val_data  = d['val']

# ── Fig 1: Barras comparativas 8 modelos ──────────────────────────────────
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
plt.suptitle('Comparación de Métricas — Test 2022 (8 Modelos)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_metrics_bar_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig3_metrics_bar_final.png OK")

# ── Fig 2: Heatmap val/test ────────────────────────────────────────────────
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
plt.suptitle('Métricas por Modelo — Benchmark Final (8 Modelos)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_heatmap_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig4_heatmap_final.png OK")

# ── Fig 3: Confusion matrices ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, key in enumerate(MODEL_ORDER):
    if key not in test_data: continue
    m  = test_data[key]
    cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    ax = axes[i//4][i%4]
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
plt.suptitle('Matrices de Confusión — Test Set 2022 (Benchmark Final)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_confusion_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig5_confusion_final.png OK")

# ── Fig 4: Curvas de aprendizaje boosting ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
curve_map = [
    ('gb_val_aucs',     'GBDT (custom)',  '#C0392B'),
    ('cb_gpu_val_aucs', 'CatBoost (GPU)', '#1ABC9C'),
    ('lgb_gpu_val_aucs','LightGBM',       '#3498DB'),
    ('xgb_gpu_val_aucs','XGBoost (GPU)',  '#F39C12'),
]
for key, label, color in curve_map:
    aucs = d.get(key, [])
    if not aucs: continue
    ax.plot(range(1, len(aucs)+1), aucs, lw=2, label=label, color=color)
    best = int(np.argmax(aucs)) + 1
    ax.axvline(best, color=color, linestyle='--', alpha=0.4, lw=1.2,
               label=f'best={best}')
ax.set_xlabel('Iteración', fontsize=11)
ax.set_ylabel('AUC-ROC (Validación 2021)', fontsize=11)
ax.set_title('Curvas de Aprendizaje — Modelos de Boosting (Benchmark Final)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_learning_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig7_learning_final.png OK")

print("\nFiguras finales generadas.")
