# Predicting Workers' Compensation Litigation — NYSWCB

**Thesis**: *Predicción de Judicialización de Reclamos de Compensación Laboral mediante Machine Learning*
**Author**: Julian Tapia · MIT Sloan / CDSS · 2025
**Model**: CatBoost v3 · AUC-ROC 0.8833 · PR-AUC 0.5976 · τ = 0.708

---

## Overview

This repository contains all Python scripts used to reproduce the results of the thesis. The model predicts whether a workers' compensation claim in New York State will escalate to litigation (assembly before the NYWCB), enabling preventive intervention before claims become adversarial.

**Key results (test 2022, n = 260,156):**

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8833 |
| PR-AUC (precision-recall) | 0.5976 |
| Recall @ τ=0.708 | 67.8% |
| Precision @ τ=0.708 | 51.9% |
| KS statistic | 0.6202 |
| Est. annual savings (upper bound) | $252M |

---

## Dataset

**Source**: [New York State Workers' Compensation Board (NYSWCB) Open Data](https://data.ny.gov/Government-Finance/Assembled-Workers-Compensation-Claims-Beginning-20/jshw-gkgu)

Download the raw dataset and place it at `raw_data/nyswcb_claims.csv`.
The preprocessing script (`build_dataset.py`) generates `raw_data/dataset_tesis_clean.csv`.

> ⚠️ Raw data and trained model weights are **not** included in this repository due to file size. All scripts reproduce the full pipeline from the raw NYSWCB file.

---

## Repository Structure

```
├── codigo/
│   ├── build_dataset.py           # Step 1: raw → clean dataset
│   ├── eda_script.py              # Step 2: exploratory analysis
│   ├── benchmark5.py              # Step 3: multi-model benchmark
│   ├── benchmark_catboost.py      # CatBoost benchmark (GPU)
│   ├── benchmark_lgbm_xgb.py     # LightGBM + XGBoost benchmark
│   ├── cap6_tuning_catboost.py    # Step 4: Optuna hyperparameter search
│   ├── retrain_v3_gpu.py          # Step 5: final model training
│   ├── retrain_v3_optuna_gpu.py   # Alternative: Optuna full search
│   ├── depth_ext_experiment.py    # Confirmatory depth experiment [8,14]
│   ├── cap7_shap.py               # Step 6: SHAP interpretability
│   ├── shap_production_prevalence.py  # SHAP robustness check (production prevalence)
│   ├── gen_figures_final.py       # Figure generation (Cap. 5–6)
│   ├── montecarlo_cap8.py         # Step 7: Monte Carlo economic analysis
│   ├── fairness_audit.py          # Step 8: Fairness audit (EEOC / EO 13985)
│   └── fairness_calibration.py    # Step 9: Per-quintile threshold calibration
│
├── requirements.txt
└── README.md
```

---

## Reproduction Steps

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU**: scripts use CatBoost with `task_type='GPU'`. CPU fallback works but is significantly slower for tuning.

### 2. Download and preprocess data

```bash
# Download from NYSWCB Open Data (link above) → raw_data/nyswcb_claims.csv
python codigo/build_dataset.py
```

### 3. Exploratory analysis

```bash
python codigo/eda_script.py
```

### 4. Model benchmark (Cap. 5)

```bash
python codigo/benchmark5.py           # all models
python codigo/benchmark_catboost.py   # CatBoost detailed
```

### 5. Hyperparameter tuning (Cap. 6)

```bash
python codigo/cap6_tuning_catboost.py   # ~200 Optuna trials
```

Confirmatory depth experiment (range [8,14]):

```bash
python codigo/depth_ext_experiment.py
```

### 6. Final model training

```bash
python codigo/retrain_v3_gpu.py
```

### 7. SHAP interpretability (Cap. 7)

```bash
python codigo/cap7_shap.py
python codigo/shap_production_prevalence.py   # robustness check
```

### 8. Economic analysis (Cap. 8)

```bash
python codigo/montecarlo_cap8.py
```

### 9. Fairness audit + calibration (Cap. 9)

```bash
python codigo/fairness_audit.py          # EEOC metrics by gender + AWW quintile
python codigo/fairness_calibration.py    # Per-quintile threshold calibration
```

---

## Fairness Results

| Group | TPR (τ=0.708) | EEOC ratio | Status |
|-------|--------------|------------|--------|
| Gender M (ref.) | 0.679 | — | ✓ |
| Gender F | 0.681 | 1.003 | ✓ |
| AWW Q1 <$657 (ref.) | 0.861 | — | ✓ |
| AWW Q5 >$1,767 (τ=0.708) | 0.639 | **0.742** | ⚠ EEOC violation |
| AWW Q5 >$1,767 (τ=0.645 calibrated) | 0.755 | **0.876** | ✓ |

Calibration cost: ΔF1 global = −0.002 (negligible).

---

## Citation

```bibtex
@mastersthesis{tapia2025nyswcb,
  author  = {Julian Tapia},
  title   = {Predicción de Judicialización de Reclamos de Compensación
             Laboral mediante Machine Learning},
  school  = {MIT Sloan School of Management / CDSS},
  year    = {2025},
  note    = {Dataset: NYSWCB Open Data. Model: CatBoost v3.
             AUC-ROC 0.8833, PR-AUC 0.5976.}
}
```

---

## Regulatory Compliance

This work complies with:
- **EO 13985** (US, 2021): Equal Opportunity audit conducted
- **DFS Circular Letter No. 1** (NY, 2019): Model documentation
- **NYC Local Law 144** (2023): Automated Decision System audit

> ⚠️ The NYSWCB public dataset does not include race/ethnicity variables. A complete EO 13985 audit would require access to supplementary administrative data under a data-sharing agreement with the NYSWCB.

---

*MIT Sloan / CDSS · 2025*
