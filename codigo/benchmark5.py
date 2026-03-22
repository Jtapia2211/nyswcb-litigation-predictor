"""
Benchmark 5 Modelos — Tesis ITBA
Split temporal: train 2017-2020 | val 2021 | test 2022
Modelos:
  1. Logistic Regression (SGD + L2)
  2. Naive Bayes (Gaussiano/Categórico mixto)
  3. Decision Tree (CART, profundidad 7)
  4. Random Forest (50 árboles, subspace aleatorio)
  5. Gradient Boosted Decision Trees (200 árboles)
"""

import numpy as np
import pandas as pd
import json, time, os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/raw_data/dataset_tesis_clean.csv'
OUT_DIR   = '/sessions/epic-intelligent-hawking/model_plots5'
RES_PATH  = '/sessions/epic-intelligent-hawking/benchmark5_results.json'
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = ['#2E4057','#8E44AD','#27AE60','#E67E22','#C0392B']
BLUE, PURPLE, GREEN, ORANGE, RED = COLORS

# ── Métricas (numpy puro) ──────────────────────────────────────────────────
def sigmoid(x):
    return np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

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

def roc_curve(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    tps = np.cumsum(ys);  fps = np.cumsum(1-ys)
    return np.concatenate([[0],fps/fps[-1]]), np.concatenate([[0],tps/tps[-1]])

def pr_curve(y_true, y_score):
    idx = np.argsort(-y_score)
    ys  = y_true[idx]
    tps = np.cumsum(ys)
    return (np.concatenate([[0],tps/ys.sum()]),
            np.concatenate([[1],tps/np.arange(1,len(ys)+1)]))

# ── Carga y Split ──────────────────────────────────────────────────────────
print("Cargando dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

train = df[df['accident_year'].between(2017,2020)].copy()
val   = df[df['accident_year']==2021].copy()
test  = df[df['accident_year']==2022].copy()
print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

NUM_COLS = ['days_to_assembly','days_C2_to_accident','days_C3_to_accident',
            'age_at_injury','aww']
BIN_COLS = ['has_C2','has_C3','has_ANCR_early','accident_year','accident_month','accident_dow']
CAT_COLS = ['gender','accident_type','occupational_disease','county_of_injury',
            'medical_fee_region','wcio_cause_code','wcio_nature_code','wcio_body_code',
            'carrier_type','district_name','industry_code','industry_desc']
FEATURE_COLS = NUM_COLS + BIN_COLS + CAT_COLS

# Imputación
medians = {c: float(train[c].median()) for c in NUM_COLS}
for ds in [train, val, test]:
    for c in NUM_COLS:   ds[c] = ds[c].fillna(medians[c])
    for c in BIN_COLS:   ds[c] = ds[c].fillna(0)
    for c in CAT_COLS:   ds[c] = ds[c].fillna('UNKNOWN').astype(str)

# Encoding categórico por frecuencia
cat_maps = {}
for col in CAT_COLS:
    freq = train[col].value_counts()
    cat_maps[col] = {v:i for i,v in enumerate(freq.index)}
for ds in [train, val, test]:
    for col in CAT_COLS:
        ds[col] = ds[col].map(cat_maps[col]).fillna(len(cat_maps[col])).astype(np.float32)

X_tr = train[FEATURE_COLS].values.astype(np.float32)
y_tr = train['target'].values.astype(np.float32)
X_va = val[FEATURE_COLS].values.astype(np.float32)
y_va = val['target'].values.astype(np.float32)
X_te = test[FEATURE_COLS].values.astype(np.float32)
y_te = test['target'].values.astype(np.float32)

pos_w = float((y_tr==0).sum()/(y_tr==1).sum())  # ~3.41
print(f"pos_weight = {pos_w:.3f}")

# Normalización para LR
mu = X_tr.mean(0); sd = X_tr.std(0)+1e-8
X_tr_s = (X_tr - mu)/sd
X_va_s = (X_va - mu)/sd
X_te_s = (X_te - mu)/sd

# ── Histogramas para árboles ───────────────────────────────────────────────
N_BINS = 32
print("\nDiscretizando features para árboles...")
bin_edges = []
X_tr_b = np.zeros_like(X_tr, dtype=np.uint8)
X_va_b = np.zeros_like(X_va, dtype=np.uint8)
X_te_b = np.zeros_like(X_te, dtype=np.uint8)
for j in range(X_tr.shape[1]):
    q = np.unique(np.percentile(X_tr[:,j], np.linspace(0,100,N_BINS+1)))
    bin_edges.append(q)
    X_tr_b[:,j] = np.searchsorted(q[1:-1], X_tr[:,j]).astype(np.uint8)
    X_va_b[:,j] = np.searchsorted(q[1:-1], X_va[:,j]).astype(np.uint8)
    X_te_b[:,j] = np.searchsorted(q[1:-1], X_te[:,j]).astype(np.uint8)
n_bins_f = [len(e)-1 for e in bin_edges]

# ── Árbol CART (numpy histograma) ──────────────────────────────────────────
REG_LAMBDA = 1.0

class Node:
    __slots__ = ['feat','thr','left','right','value','n']
    def __init__(self): self.feat=self.thr=self.left=self.right=self.value=self.n=None

def build(X_b, g, h, depth, min_s, max_feat=None):
    nd = Node(); nd.n = len(g)
    nd.value = float(-g.sum()/(h.sum()+REG_LAMBDA))
    if depth==0 or nd.n < min_s*2: return nd
    G_t = g.sum(); H_t = h.sum()
    best = -1e9; bf = -1; bb = -1
    feats = (np.random.choice(X_b.shape[1], max_feat, replace=False)
             if max_feat else range(X_b.shape[1]))
    for j in feats:
        nb = n_bins_f[j]
        bins = X_b[:,j]
        gh = np.bincount(bins, weights=g, minlength=nb+1)
        hh = np.bincount(bins, weights=h, minlength=nb+1)
        gc = np.cumsum(gh); hc = np.cumsum(hh)
        def score(G,H): return G*G/(H+REG_LAMBDA) if H+REG_LAMBDA>0 else 0
        for b in range(nb):
            if hc[b]<1 or H_t-hc[b]<1: continue
            gain = 0.5*(score(gc[b],hc[b])+score(G_t-gc[b],H_t-hc[b])
                        -score(G_t,H_t))
            if gain>best: best=gain; bf=j; bb=b
    if best<=0: return nd
    nd.feat=bf; nd.thr=bb
    ml = X_b[:,bf]<=bb; mr = ~ml
    if ml.sum()<min_s or mr.sum()<min_s: return nd
    nd.left  = build(X_b[ml], g[ml], h[ml], depth-1, min_s, max_feat)
    nd.right = build(X_b[mr], g[mr], h[mr], depth-1, min_s, max_feat)
    return nd

def predict_tree(nd, X_b):
    out = np.zeros(len(X_b), dtype=np.float64)
    def walk(n, idx):
        if n.left is None: out[idx]=n.value; return
        ml = X_b[idx,n.feat]<=n.thr
        walk(n.left, idx[ml]); walk(n.right, idx[~ml])
    walk(nd, np.arange(len(X_b)))
    return out

def feat_imp(trees_list, n_feat):
    imp = np.zeros(n_feat)
    def walk(nd):
        if nd.feat is not None and nd.left is not None:
            imp[nd.feat]+=1; walk(nd.left); walk(nd.right)
    for t in trees_list: walk(t)
    s = imp.sum()
    return imp/s if s>0 else imp

results_val, results_test, importances = {}, {}, {}

# ══════════════════════════════════════════════════════════════════════════
# MODELO 1: Logistic Regression
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODELO 1: Logistic Regression (mini-batch SGD, L2)")
print("="*60)
t0 = time.time()

w = np.zeros(X_tr_s.shape[1], dtype=np.float64); b = 0.0
for epoch in range(3):
    idx = np.random.permutation(len(y_tr))
    Xs, ys = X_tr_s[idx].astype(np.float64), y_tr[idx].astype(np.float64)
    for s in range(0, len(ys), 4096):
        Xb=Xs[s:s+4096]; yb=ys[s:s+4096]
        p = sigmoid(Xb@w+b)
        w_s = np.where(yb==1, pos_w, 1.0)
        err = (p-yb)*w_s
        w -= 0.05*((Xb.T@err)/len(yb)+1e-4*w)
        b -= 0.05*err.mean()
    p_v = sigmoid(X_va_s.astype(np.float64)@w+b)
    print(f"  Epoch {epoch+1}/3 | val AUC: {roc_auc(y_va, p_v.astype(np.float32)):.4f}")

p_va_lr  = sigmoid(X_va_s.astype(np.float64)@w+b).astype(np.float32)
p_te_lr  = sigmoid(X_te_s.astype(np.float64)@w+b).astype(np.float32)
results_val['LR']  = full_metrics(y_va, p_va_lr, 'Logistic Regression')
results_test['LR'] = full_metrics(y_te, p_te_lr, 'Logistic Regression')
importances['LR']  = (np.abs(w)/np.abs(w).sum()).tolist()
print(f"  Test AUC: {results_test['LR']['auc_roc']:.4f} | KS: {results_test['LR']['ks']:.4f} | F1: {results_test['LR']['f1']:.4f} | {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# MODELO 2: Naive Bayes (Gaussiano + Bernoulli + Categórico)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODELO 2: Naive Bayes (Gaussiano / Bernoulli / Categórico)")
print("="*60)
t0 = time.time()

n_num = len(NUM_COLS); n_bin = len(BIN_COLS); n_cat = len(CAT_COLS)
n_features = X_tr.shape[1]
classes = [0.0, 1.0]
log_prior = np.zeros(2)
# Gaussian params para numéricas
mu_nb  = np.zeros((2, n_num)); var_nb = np.zeros((2, n_num))
# Bernoulli params para binarias
p_bin  = np.zeros((2, n_bin))
# Categórico params para categorías
cat_probs = []  # lista de arrays [2 x n_cats]

for ci, c in enumerate(classes):
    mask = y_tr == c
    Xc   = X_tr[mask]
    log_prior[ci] = np.log(mask.mean() + 1e-10)
    # Gaussiano
    mu_nb[ci]  = Xc[:, :n_num].mean(0)
    var_nb[ci] = Xc[:, :n_num].var(0) + 1e-6
    # Bernoulli
    p_bin[ci]  = np.clip(Xc[:, n_num:n_num+n_bin].mean(0), 1e-5, 1-1e-5)

# Categórico con Laplace smoothing
for ci, c in enumerate(classes):
    mask = y_tr == c
    Xc   = X_tr[mask][:, n_num+n_bin:].astype(int)
    if ci == 0:
        cat_probs = []
    for j in range(n_cat):
        n_cats = len(cat_maps[CAT_COLS[j]]) + 2
        if ci == 0:
            cat_probs.append(np.zeros((2, n_cats)))
        counts = np.bincount(np.clip(Xc[:,j], 0, n_cats-1), minlength=n_cats).astype(float)
        cat_probs[j][ci] = (counts + 1) / (mask.sum() + n_cats)  # Laplace

def nb_predict(X):
    n = len(X)
    log_p = np.tile(log_prior, (n,1))  # [n x 2]
    # Gaussiano
    for j in range(n_num):
        for ci in range(2):
            x = X[:,j]
            lp = -0.5*np.log(2*np.pi*var_nb[ci,j]) - (x-mu_nb[ci,j])**2/(2*var_nb[ci,j])
            log_p[:,ci] += lp
    # Bernoulli
    for j in range(n_bin):
        for ci in range(2):
            x = X[:, n_num+j]
            lp = x*np.log(p_bin[ci,j]) + (1-x)*np.log(1-p_bin[ci,j])
            log_p[:,ci] += lp
    # Categórico
    for j in range(n_cat):
        n_cats = cat_probs[j].shape[1]
        idx = np.clip(X[:, n_num+n_bin+j].astype(int), 0, n_cats-1)
        for ci in range(2):
            log_p[:,ci] += np.log(cat_probs[j][ci][idx] + 1e-10)
    # Softmax → probabilidad de clase 1
    log_p_max = log_p.max(1, keepdims=True)
    exp_p = np.exp(log_p - log_p_max)
    return (exp_p[:,1] / exp_p.sum(1)).astype(np.float32)

p_va_nb  = nb_predict(X_va)
p_te_nb  = nb_predict(X_te)
results_val['NB']  = full_metrics(y_va, p_va_nb, 'Naive Bayes')
results_test['NB'] = full_metrics(y_te, p_te_nb, 'Naive Bayes')
importances['NB']  = None
print(f"  Test AUC: {results_test['NB']['auc_roc']:.4f} | KS: {results_test['NB']['ks']:.4f} | F1: {results_test['NB']['f1']:.4f} | {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# MODELO 3: Decision Tree (CART, depth=7)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODELO 3: Decision Tree (CART, profundidad=7)")
print("="*60)
t0 = time.time()

# Usar gradientes de log-loss para CART puro (equivalente a árbol de clasificación)
p_init = y_tr.mean()
F0 = np.full(len(y_tr), np.log(p_init/(1-p_init)))
p_t  = sigmoid(F0).astype(np.float32)
w_s  = np.where(y_tr==1, pos_w, 1.0).astype(np.float64)
g_dt = ((p_t - y_tr) * w_s).astype(np.float64)
h_dt = (p_t * (1-p_t) * w_s + 1e-6).astype(np.float64)

dt_tree = build(X_tr_b, g_dt, h_dt, depth=7, min_s=200)
lr_dt = 1.0  # full weight single tree

def dt_score(X_b):
    F = np.full(len(X_b), np.log(p_init/(1-p_init)))
    F += predict_tree(dt_tree, X_b)
    return sigmoid(F).astype(np.float32)

p_va_dt  = dt_score(X_va_b)
p_te_dt  = dt_score(X_te_b)
results_val['DT']  = full_metrics(y_va, p_va_dt, 'Decision Tree')
results_test['DT'] = full_metrics(y_te, p_te_dt, 'Decision Tree')
importances['DT']  = feat_imp([dt_tree], X_tr.shape[1]).tolist()
print(f"  Test AUC: {results_test['DT']['auc_roc']:.4f} | KS: {results_test['DT']['ks']:.4f} | F1: {results_test['DT']['f1']:.4f} | {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# MODELO 4: Random Forest (50 árboles, bootstrap + feature subspace)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODELO 4: Random Forest (50 árboles, depth=5, √p features)")
print("="*60)
t0 = time.time()

N_RF   = 50
D_RF   = 5
MF_RF  = int(np.sqrt(X_tr.shape[1]))  # √23 ≈ 5
rf_trees = []
rf_preds_va = np.zeros((N_RF, len(y_va)), dtype=np.float32)
rf_preds_te = np.zeros((N_RF, len(y_te)), dtype=np.float32)

for t_i in range(N_RF):
    # Bootstrap
    boot_idx = np.random.choice(len(y_tr), len(y_tr), replace=True)
    Xb_b = X_tr_b[boot_idx]
    yb   = y_tr[boot_idx]
    # Gradientes de log-loss
    F0b  = np.log(yb.mean()/(1-yb.mean()+1e-9))
    pb   = sigmoid(np.full(len(yb), F0b)).astype(np.float32)
    wb   = np.where(yb==1, pos_w, 1.0).astype(np.float64)
    gb   = ((pb - yb) * wb).astype(np.float64)
    hb   = (pb * (1-pb) * wb + 1e-6).astype(np.float64)
    tree_i = build(Xb_b, gb, hb, depth=D_RF, min_s=200, max_feat=MF_RF)
    rf_trees.append(tree_i)
    # Score (shift + tree leaf)
    base = np.log(y_tr.mean()/(1-y_tr.mean()))
    rf_preds_va[t_i] = sigmoid(base + predict_tree(tree_i, X_va_b)).astype(np.float32)
    rf_preds_te[t_i] = sigmoid(base + predict_tree(tree_i, X_te_b)).astype(np.float32)
    if (t_i+1) % 10 == 0:
        p_avg_v = rf_preds_va[:t_i+1].mean(0)
        print(f"  Tree {t_i+1:3d}/{N_RF} | val AUC: {roc_auc(y_va, p_avg_v):.4f} | {time.time()-t0:.1f}s")

p_va_rf  = rf_preds_va.mean(0)
p_te_rf  = rf_preds_te.mean(0)
results_val['RF']  = full_metrics(y_va, p_va_rf, 'Random Forest')
results_test['RF'] = full_metrics(y_te, p_te_rf, 'Random Forest')
importances['RF']  = feat_imp(rf_trees, X_tr.shape[1]).tolist()
print(f"  Test AUC: {results_test['RF']['auc_roc']:.4f} | KS: {results_test['RF']['ks']:.4f} | F1: {results_test['RF']['f1']:.4f} | {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# MODELO 5: Gradient Boosted Decision Trees
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODELO 5: Gradient Boosted Decision Trees (200 árboles, depth=4)")
print("="*60)
t0 = time.time()

N_GB = 200; D_GB = 4; LR_GB = 0.05
F_tr = np.full(len(y_tr), np.log(y_tr.mean()/(1-y_tr.mean())), dtype=np.float64)
F_va = np.full(len(y_va), np.log(y_va.mean()/(1-y_va.mean())), dtype=np.float64)
F_te = np.full(len(y_te), np.log(y_te.mean()/(1-y_te.mean())), dtype=np.float64)
gb_trees = []; gb_val_aucs = []

for t_i in range(N_GB):
    pt = sigmoid(F_tr).astype(np.float32)
    ws = np.where(y_tr==1, pos_w, 1.0).astype(np.float64)
    g  = ((pt - y_tr) * ws).astype(np.float64)
    h  = (pt * (1-pt) * ws + 1e-6).astype(np.float64)
    tree_i = build(X_tr_b, g, h, depth=D_GB, min_s=500)
    F_tr += LR_GB * predict_tree(tree_i, X_tr_b)
    F_va += LR_GB * predict_tree(tree_i, X_va_b)
    F_te += LR_GB * predict_tree(tree_i, X_te_b)
    gb_trees.append(tree_i)
    if (t_i+1) % 20 == 0:
        auc_v = roc_auc(y_va, sigmoid(F_va).astype(np.float32))
        gb_val_aucs.append(auc_v)
        print(f"  Tree {t_i+1:3d}/{N_GB} | val AUC: {auc_v:.4f} | {time.time()-t0:.1f}s")

p_va_gb  = sigmoid(F_va).astype(np.float32)
p_te_gb  = sigmoid(F_te).astype(np.float32)
results_val['GB']  = full_metrics(y_va, p_va_gb, 'Gradient Boosted Trees')
results_test['GB'] = full_metrics(y_te, p_te_gb, 'Gradient Boosted Trees')
importances['GB']  = feat_imp(gb_trees, X_tr.shape[1]).tolist()
print(f"  Test AUC: {results_test['GB']['auc_roc']:.4f} | KS: {results_test['GB']['ks']:.4f} | F1: {results_test['GB']['f1']:.4f} | {time.time()-t0:.1f}s")

# ── Resumen ────────────────────────────────────────────────────────────────
ORDER = ['LR','NB','DT','RF','GB']
NAMES = {'LR':'Logistic Regression','NB':'Naive Bayes','DT':'Decision Tree',
         'RF':'Random Forest','GB':'Gradient Boosted Trees'}

print("\n" + "="*70)
print(f"{'Modelo':<28} {'AUC-ROC':>8} {'PR-AUC':>8} {'KS':>6} {'F1':>6} {'Brier':>7}")
print("-"*68)
for k in ORDER:
    m = results_test[k]
    print(f"{NAMES[k]:<28} {m['auc_roc']:>8.4f} {m['pr_auc']:>8.4f} {m['ks']:>6.4f} {m['f1']:>6.4f} {m['brier']:>7.4f}")

# ── Guardar ────────────────────────────────────────────────────────────────
save = {
    'val':  {k: results_val[k]  for k in ORDER},
    'test': {k: results_test[k] for k in ORDER},
    'gb_val_aucs': gb_val_aucs,
    'importances': {k: importances[k] for k in ORDER},
    'feature_names': FEATURE_COLS,
}
with open(RES_PATH,'w') as f: json.dump(save, f, indent=2)
print(f"\nResultados → {RES_PATH}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURAS
# ══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':12,
    'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
    'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

model_keys   = ORDER
model_labels = [NAMES[k] for k in ORDER]
model_colors = COLORS
p_va_all = {'LR':p_va_lr,'NB':p_va_nb,'DT':p_va_dt,'RF':p_va_rf,'GB':p_va_gb}
p_te_all = {'LR':p_te_lr,'NB':p_te_nb,'DT':p_te_dt,'RF':p_te_rf,'GB':p_te_gb}

# ── Fig 1: Curvas ROC (Test) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for k, lab, col in zip(ORDER, model_labels, COLORS):
    fpr, tpr = roc_curve(y_te, p_te_all[k])
    auc = results_test[k]['auc_roc']
    ax.plot(fpr, tpr, color=col, lw=2.2, label=f'{lab} (AUC={auc:.4f})')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random (0.5000)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Curvas ROC — Test Set 2022', fontweight='bold')
ax.legend(fontsize=8.5, loc='lower right')
ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_roc.png', bbox_inches='tight')
plt.close()
print("fig1_roc OK")

# ── Fig 2: Curvas Precision-Recall (Test) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for k, lab, col in zip(ORDER, model_labels, COLORS):
    rec, prec = pr_curve(y_te, p_te_all[k])
    prauc = results_test[k]['pr_auc']
    ax.plot(rec, prec, color=col, lw=2.2, label=f'{lab} (PR-AUC={prauc:.4f})')
ax.axhline(y_te.mean(), color='gray', linestyle=':', lw=1.2, label=f'Baseline ({y_te.mean():.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Curvas Precision-Recall — Test Set 2022', fontweight='bold')
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_pr.png', bbox_inches='tight')
plt.close()
print("fig2_pr OK")

# ── Fig 3: Comparación de métricas en barras ──────────────────────────────
metrics_show = ['auc_roc','pr_auc','ks','f1','brier']
metric_labels_show = ['AUC-ROC','PR-AUC','KS','F1','Brier↓']
fig, axes = plt.subplots(1, 5, figsize=(16, 4.5))
for ax_i, (mkey, mlabel) in enumerate(zip(metrics_show, metric_labels_show)):
    vals = [results_test[k][mkey] for k in ORDER]
    # Para Brier, mejor es más bajo → invertir color
    best_idx = np.argmin(vals) if mkey=='brier' else np.argmax(vals)
    bar_cols = [COLORS[i] if i==best_idx else '#BDC3C7' for i in range(5)]
    bars = axes[ax_i].bar(range(5), vals, color=bar_cols, edgecolor='white', linewidth=1.2)
    for b, v in zip(bars, vals):
        axes[ax_i].text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                         f'{v:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    axes[ax_i].set_xticks(range(5))
    axes[ax_i].set_xticklabels(['LR','NB','DT','RF','GB'], fontsize=9)
    axes[ax_i].set_title(mlabel, fontweight='bold', fontsize=11)
    axes[ax_i].set_ylim(min(vals)*0.9, max(vals)*1.12)
plt.suptitle('Comparación de Métricas — Test Set 2022\n(Resaltado: mejor modelo por métrica)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_metrics_bar.png', bbox_inches='tight')
plt.close()
print("fig3_metrics_bar OK")

# ── Fig 4: Heatmap de métricas (val vs test) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax_i, (split, res, split_label) in enumerate([
    ('val',  results_val,  'Validación 2021'),
    ('test', results_test, 'Test 2022'),
]):
    data = np.array([[res[k]['auc_roc'],res[k]['pr_auc'],res[k]['ks'],res[k]['f1']] for k in ORDER])
    im = axes[ax_i].imshow(data, cmap='Blues', vmin=0.5, vmax=1.0, aspect='auto')
    axes[ax_i].set_xticks(range(4)); axes[ax_i].set_xticklabels(['AUC-ROC','PR-AUC','KS','F1'], fontsize=9)
    axes[ax_i].set_yticks(range(5)); axes[ax_i].set_yticklabels(model_labels, fontsize=9)
    for r in range(5):
        for c in range(4):
            axes[ax_i].text(c, r, f'{data[r,c]:.4f}', ha='center', va='center',
                             fontsize=9, fontweight='bold',
                             color='white' if data[r,c]>0.85 else 'black')
    axes[ax_i].set_title(f'Métricas — {split_label}', fontweight='bold')
    plt.colorbar(im, ax=axes[ax_i], fraction=0.046, pad=0.04)
plt.suptitle('Heatmap de Desempeño — Validación y Test', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_heatmap.png', bbox_inches='tight')
plt.close()
print("fig4_heatmap OK")

# ── Fig 5: Matrices de confusión (todos los modelos, test) ────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax_i, (k, lab, col) in enumerate(zip(ORDER, model_labels, COLORS)):
    m = results_test[k]
    cm = np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
    cm_pct = cm/cm.sum(axis=1,keepdims=True)*100
    cmap = plt.cm.Blues
    axes[ax_i].imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)
    for r in range(2):
        for c in range(2):
            axes[ax_i].text(c, r, f'{cm[r,c]:,}\n{cm_pct[r,c]:.1f}%', ha='center',
                             va='center', fontsize=8,
                             color='white' if cm_pct[r,c]>60 else 'black', fontweight='bold')
    axes[ax_i].set_xticks([0,1]); axes[ax_i].set_xticklabels(['Pred\nNeg','Pred\nPos'], fontsize=8)
    axes[ax_i].set_yticks([0,1]); axes[ax_i].set_yticklabels(['Real\nNeg','Real\nPos'], fontsize=8)
    axes[ax_i].set_title(f'{lab}\nF1={m["f1"]:.4f} | thr={m["threshold"]:.3f}',
                          fontweight='bold', fontsize=8.5)
plt.suptitle('Matrices de Confusión — Test Set 2022 (umbral de Youden)', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_confusion.png', bbox_inches='tight')
plt.close()
print("fig5_confusion OK")

# ── Fig 6: Feature importance (top 10) ────────────────────────────────────
short_names = [f.replace('days_to_assembly','d→asm').replace('days_C2_to_accident','d→C2')
    .replace('days_C3_to_accident','d→C3').replace('age_at_injury','age')
    .replace('accident_','acc_').replace('wcio_cause_code','wcio_cause')
    .replace('wcio_nature_code','wcio_nat').replace('wcio_body_code','wcio_body')
    .replace('county_of_injury','county').replace('medical_fee_region','med_reg')
    .replace('carrier_type','carrier').replace('district_name','district')
    .replace('industry_desc','ind_desc').replace('industry_code','ind_code')
    .replace('occupational_disease','occ_dis').replace('has_ANCR_early','ANCR_early')
    for f in FEATURE_COLS]

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
imp_models = [('LR','LR'), ('DT','DT'), ('RF','RF'), ('GB','GBDT')]
for ax_i, (k, lab) in enumerate(imp_models):
    imp = np.array(importances[k])
    top10 = np.argsort(imp)[-10:][::-1]
    names = [short_names[i] for i in top10]
    vals  = [imp[i] for i in top10]
    axes[ax_i].barh(names[::-1], vals[::-1], color=COLORS[ORDER.index(k)], edgecolor='white', alpha=0.9)
    axes[ax_i].set_xlabel('Importancia relativa', fontsize=9)
    axes[ax_i].set_title(f'{NAMES[k]}\n(Top 10)', fontweight='bold', fontsize=10)
    for i, v in enumerate(vals[::-1]):
        axes[ax_i].text(v+0.0005, i, f'{v:.3f}', va='center', fontsize=7.5)
plt.suptitle('Importancia de Variables — Top 10 por Modelo', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_importance.png', bbox_inches='tight')
plt.close()
print("fig6_importance OK")

# ── Fig 7: Curva de aprendizaje GBDT ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
iters = [(i+1)*20 for i in range(len(gb_val_aucs))]
ax.plot(iters, gb_val_aucs, color=RED, lw=2.5, marker='o', markersize=5, label='GBDT (val AUC)')
ax.axhline(results_val['RF']['auc_roc'],  color=ORANGE, lw=1.5, linestyle='--', label=f'Random Forest val ({results_val["RF"]["auc_roc"]:.4f})')
ax.axhline(results_val['LR']['auc_roc'],  color=BLUE,   lw=1.5, linestyle='--', label=f'Logistic Reg val ({results_val["LR"]["auc_roc"]:.4f})')
ax.set_xlabel('Número de árboles'); ax.set_ylabel('AUC-ROC (Validación 2021)')
ax.set_title('Curva de aprendizaje — GBDT vs. modelos de referencia', fontweight='bold')
ax.legend(fontsize=9); ax.set_ylim(0.82, 0.94)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_learning_curve.png', bbox_inches='tight')
plt.close()
print("fig7_learning_curve OK")

# ── Fig 8: Score distributions GBDT ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax_i, (p, y, title) in enumerate([(p_va_gb, y_va, 'Val 2021'), (p_te_gb, y_te, 'Test 2022')]):
    axes[ax_i].hist(p[y==0], bins=80, color=BLUE, alpha=0.6, density=True, label='No judic.')
    axes[ax_i].hist(p[y==1], bins=80, color=RED,  alpha=0.6, density=True, label='Judic.')
    axes[ax_i].set_xlabel('Score GBDT'); axes[ax_i].set_ylabel('Densidad')
    axes[ax_i].set_title(f'GBDT — Distribución de scores ({title})', fontweight='bold')
    axes[ax_i].legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig8_score_dist.png', bbox_inches='tight')
plt.close()
print("fig8_score_dist OK")

print(f"\nTodo listo. Plots → {OUT_DIR}")
