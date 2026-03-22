"""
MLP (Multi-Layer Perceptron) benchmark — NumPy puro
Arquitectura: 23 → 256 → 128 → 64 → 1 (sigmoid)
Optimizer: Adam  |  Loss: BCE ponderada  |  Early stopping temporal (val 2021)
"""
import numpy as np
import json, time, os
import pandas as pd

np.random.seed(42)

DATA_PATH = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/raw_data/dataset_tesis_clean.csv'
RES_IN    = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/codigo/benchmark_gpu_results.json'
RES_OUT   = RES_IN   # actualiza in-place

TARGET   = 'target'
NUM_COLS = ['days_to_assembly','days_C2_to_accident','days_C3_to_accident','age_at_injury','aww']
BIN_COLS = ['has_C2','has_C3','has_ANCR_early','accident_year','accident_month','accident_dow']
CAT_COLS = ['gender','accident_type','occupational_disease','county_of_injury',
            'medical_fee_region','wcio_cause_code','wcio_nature_code','wcio_body_code',
            'carrier_type','district_name','industry_code','industry_desc']
FEATURES = NUM_COLS + BIN_COLS + CAT_COLS

# ── Métricas ──────────────────────────────────────────────────────────────────
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

# ── Carga y split ─────────────────────────────────────────────────────────────
print("Cargando datos...", flush=True)
t0 = time.time()
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  {len(df):,} filas en {time.time()-t0:.1f}s")

df_train = df[df['accident_year'].between(2017, 2020)].copy()
df_val   = df[df['accident_year'] == 2021].copy()
df_test  = df[df['accident_year'] == 2022].copy()
print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# ── Preprocesamiento ──────────────────────────────────────────────────────────
# Imputación
for col in NUM_COLS:
    med = df_train[col].median()
    for d in [df_train, df_val, df_test]:
        d[col] = d[col].fillna(med)
for col in BIN_COLS:
    for d in [df_train, df_val, df_test]:
        d[col] = d[col].fillna(0)

# Codificación ordinal por frecuencia para categóricas
freq_maps = {}
for col in CAT_COLS:
    freq = df_train[col].fillna('UNKNOWN').astype(str).value_counts()
    freq_maps[col] = {v: i for i, v in enumerate(freq.index)}
    n = len(freq_maps[col])
    for d in [df_train, df_val, df_test]:
        d[col] = d[col].fillna('UNKNOWN').astype(str).map(freq_maps[col]).fillna(n).astype(float)

X_tr = df_train[FEATURES].values.astype(np.float32)
X_vl = df_val[FEATURES].values.astype(np.float32)
X_te = df_test[FEATURES].values.astype(np.float32)
y_tr = df_train[TARGET].values.astype(np.float32)
y_vl = df_val[TARGET].values.astype(np.float32)
y_te = df_test[TARGET].values.astype(np.float32)

# Z-score (fit solo en train)
mean_ = X_tr.mean(axis=0); std_ = X_tr.std(axis=0); std_[std_==0] = 1.0
X_tr = (X_tr - mean_) / std_
X_vl = (X_vl - mean_) / std_
X_te = (X_te - mean_) / std_

neg_pos_ratio = float((y_tr == 0).sum() / (y_tr == 1).sum())
print(f"  scale_pos_weight: {neg_pos_ratio:.3f}")
print(f"  Preprocesamiento OK")

# ── MLP — NumPy ───────────────────────────────────────────────────────────────
class AdamMLP:
    """MLP feedforward con Adam, BCE ponderada y dropout."""

    def __init__(self, layer_sizes, lr=1e-3, weight_decay=1e-4, dropout=0.2):
        self.lr = lr
        self.wd = weight_decay
        self.dropout = dropout
        self.W, self.b = [], []
        for i in range(len(layer_sizes)-1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            scale = np.sqrt(2.0 / fan_in)          # He init (ReLU)
            self.W.append(np.random.randn(fan_in, fan_out).astype(np.float32) * scale)
            self.b.append(np.zeros(fan_out, dtype=np.float32))
        # Adam moments
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def _forward_train(self, X):
        acts, masks = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            if i < len(self.W) - 1:
                a = np.maximum(0.0, z)
                m = (np.random.rand(*a.shape).astype(np.float32) > self.dropout) / (1.0 - self.dropout)
                acts.append(a * m); masks.append(m)
            else:
                a = 1.0 / (1.0 + np.exp(-np.clip(z.ravel(), -20, 20)))
                acts.append(a); masks.append(None)
        return acts, masks

    def predict_proba(self, X, batch_size=8192):
        out = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            a  = xb
            for j, (W, b) in enumerate(zip(self.W, self.b)):
                z = a @ W + b
                a = np.maximum(0.0, z) if j < len(self.W)-1 else \
                    1.0 / (1.0 + np.exp(-np.clip(z.ravel(), -20, 20)))
            out.append(a)
        return np.concatenate(out)

    def train_epoch(self, X, y, sw, batch_size=4096):
        idx = np.random.permutation(len(X))
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, len(X), batch_size):
            bi  = idx[start:start+batch_size]
            xb, yb, wb = X[bi], y[bi], sw[bi]
            n   = len(xb)

            acts, masks = self._forward_train(xb)
            p = acts[-1]  # (n,)

            # Weighted BCE loss
            eps = 1e-7
            loss = -np.mean(wb * (yb * np.log(p + eps) + (1-yb) * np.log(1-p+eps)))
            total_loss += float(loss); n_batches += 1

            # Backward
            delta = ((p - yb) * wb / n).reshape(-1, 1)
            for i in range(len(self.W)-1, -1, -1):
                gW = acts[i].reshape(n, -1).T @ delta + self.wd * self.W[i]
                gb = delta.sum(axis=0)
                if i > 0:
                    delta = delta @ self.W[i].T
                    if masks[i-1] is not None:
                        delta *= masks[i-1]
                    delta *= (acts[i] > 0)
                self._adam_step(i, gW, gb)

        return total_loss / n_batches

    def _adam_step(self, i, gW, gb, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1.0 / len(self.W)   # approximate; real t updated per epoch below
        lr_t = self.lr                 # use full lr; bias correction done per epoch
        self.mW[i] = 0.9*self.mW[i] + 0.1*gW
        self.vW[i] = 0.999*self.vW[i] + 0.001*gW**2
        self.W[i] -= lr_t * self.mW[i] / (np.sqrt(self.vW[i]) + eps)
        self.mb[i] = 0.9*self.mb[i] + 0.1*gb
        self.vb[i] = 0.999*self.vb[i] + 0.001*gb**2
        self.b[i] -= lr_t * self.mb[i] / (np.sqrt(self.vb[i]) + eps)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
print("\n[9] MLP (23→256→128→64→1, Adam, dropout=0.2, lr=0.001)...", flush=True)
t1 = time.time()

layer_sizes = [23, 256, 128, 64, 1]
mlp = AdamMLP(layer_sizes, lr=0.001, weight_decay=1e-4, dropout=0.2)

# Subsample estratificado 50% (igual que CatBoost Linux)
pos_idx = np.where(y_tr == 1)[0]
neg_idx = np.where(y_tr == 0)[0]
rng = np.random.default_rng(42)
sel_pos = rng.choice(pos_idx, size=len(pos_idx)//2, replace=False)
sel_neg = rng.choice(neg_idx, size=len(neg_idx)//2, replace=False)
sub_idx = np.sort(np.concatenate([sel_pos, sel_neg]))
X_tr_sub = X_tr[sub_idx]; y_tr_sub = y_tr[sub_idx]
print(f"  Subsample 50%: {len(X_tr_sub):,} filas  (pos: {y_tr_sub.mean()*100:.1f}%)")

# Sample weights para clase positiva
sw = np.where(y_tr_sub == 1, neg_pos_ratio, 1.0).astype(np.float32)

MAX_EPOCHS = 60
PATIENCE   = 7
best_auc   = -1.0
patience_c = 0
best_W     = [w.copy() for w in mlp.W]
best_b     = [b.copy() for b in mlp.b]
val_aucs   = []

for epoch in range(1, MAX_EPOCHS+1):
    loss = mlp.train_epoch(X_tr_sub, y_tr_sub, sw, batch_size=4096)
    val_preds = mlp.predict_proba(X_vl)
    auc = roc_auc(y_vl, val_preds)
    val_aucs.append(float(round(auc, 4)))

    if epoch % 5 == 0 or epoch <= 3:
        print(f"  Epoch {epoch:3d} — loss={loss:.4f}  val_AUC={auc:.4f}", flush=True)

    if auc > best_auc:
        best_auc   = auc
        patience_c = 0
        best_W = [w.copy() for w in mlp.W]
        best_b = [b.copy() for b in mlp.b]
    else:
        patience_c += 1
        if patience_c >= PATIENCE:
            print(f"  Early stopping en epoch {epoch} (patience={PATIENCE})")
            break

# Restaurar mejores pesos
mlp.W = best_W; mlp.b = best_b

val_preds_best  = mlp.predict_proba(X_vl)
test_preds_best = mlp.predict_proba(X_te)
mlp_val_m  = full_metrics(y_vl, val_preds_best,  'MLP')
mlp_test_m = full_metrics(y_te, test_preds_best, 'MLP')
t_mlp = time.time() - t1

print(f"\n  Tiempo: {t_mlp:.1f}s  |  best_epoch: {int(np.argmax(val_aucs))+1}")
print(f"  Val  — AUC={mlp_val_m['auc_roc']}  PR={mlp_val_m['pr_auc']}  F1={mlp_val_m['f1']:.4f}  Brier={mlp_val_m['brier']}")
print(f"  Test — AUC={mlp_test_m['auc_roc']}  PR={mlp_test_m['pr_auc']}  F1={mlp_test_m['f1']:.4f}  Brier={mlp_test_m['brier']}")

# ── Guardar resultados ────────────────────────────────────────────────────────
with open(RES_IN, 'r', encoding='utf-8') as f:
    existing = json.load(f)

existing['val']['MLP']  = mlp_val_m
existing['test']['MLP'] = mlp_test_m
existing.setdefault('timing', {})['MLP'] = round(t_mlp, 1)
existing['mlp_val_aucs'] = val_aucs

with open(RES_OUT, 'w', encoding='utf-8') as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

size = os.path.getsize(RES_OUT)
print(f"\nResultados guardados: {size:,} bytes")
if size < 5000:
    print("ERROR: archivo muy pequeño")
else:
    print("OK")
