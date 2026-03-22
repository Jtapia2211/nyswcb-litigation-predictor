"""
EDA Completo - Tesis ITBA
Dataset: dataset_tesis_clean.csv (1.59M registros)
Genera estadísticas + plots PNG para el capítulo de EDA
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/raw_data/dataset_tesis_clean.csv'
OUT_DIR    = '/sessions/epic-intelligent-hawking/eda_plots'
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE    = ['#2E4057', '#C0392B']   # azul ITBA / rojo
BLUE       = '#2E4057'
RED        = '#C0392B'
GRAY       = '#7F8C8D'
LIGHT      = '#EAF0FB'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9,  'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})

print("Cargando dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Shape: {df.shape}")
print(df.dtypes)

# ── Variables numéricas y categóricas ───────────────────────────────────────
NUM_COLS = ['accident_year','accident_month','accident_dow','days_to_assembly',
            'days_C2_to_accident','days_C3_to_accident','age_at_injury','aww']
BIN_COLS = ['has_C2','has_C3','has_ANCR_early','accident_type','occupational_disease']
CAT_COLS = ['gender','county_of_injury','medical_fee_region','wcio_cause_code',
            'wcio_nature_code','wcio_body_code','carrier_type','district_name',
            'industry_code','industry_desc']

# ── 1. DISTRIBUCIÓN DEL TARGET ───────────────────────────────────────────────
print("\n[1] Distribución del target...")
vc = df['target'].value_counts()
rates = df['target'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Pie chart
axes[0].pie(vc, labels=['No judicializado\n(77.5%)', 'Judicializado\n(22.5%)'],
            colors=[BLUE, RED], startangle=90, autopct='%1.1f%%',
            textprops={'fontsize': 10}, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Distribución del Target', fontweight='bold')

# Bar chart con conteos
bars = axes[1].bar(['Target = 0\n(No judicializado)', 'Target = 1\n(Judicializado)'],
                   vc.values, color=[BLUE, RED], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, vc.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8000,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Cantidad de registros')
axes[1].set_title('Conteo absoluto por clase', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig01_target_distribution.png', bbox_inches='tight')
plt.close()
print("  fig01 OK")

# ── 2. TASA DE JUDICIALIZACIÓN POR AÑO ──────────────────────────────────────
print("[2] Evolución temporal...")
yearly = df.groupby('accident_year')['target'].agg(['mean','count']).reset_index()
yearly['mean'] *= 100

fig, ax1 = plt.subplots(figsize=(9, 4))
bars = ax1.bar(yearly['accident_year'], yearly['count']/1e3, color=LIGHT,
               edgecolor=BLUE, linewidth=1.2, label='Volumen (miles)')
ax1.set_xlabel('Año de accidente')
ax1.set_ylabel('Registros (miles)', color=GRAY)
ax1.tick_params(axis='y', labelcolor=GRAY)

ax2 = ax1.twinx()
ax2.plot(yearly['accident_year'], yearly['mean'], color=RED, linewidth=2.5,
         marker='o', markersize=7, label='Tasa judicialización (%)')
for x, y in zip(yearly['accident_year'], yearly['mean']):
    ax2.annotate(f'{y:.1f}%', (x, y), textcoords='offset points', xytext=(0, 10),
                 ha='center', fontsize=8.5, color=RED, fontweight='bold')
ax2.set_ylabel('Tasa de judicialización (%)', color=RED)
ax2.tick_params(axis='y', labelcolor=RED)
ax2.set_ylim(0, 35)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
plt.title('Volumen de siniestros y tasa de judicialización por año', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig02_temporal_trend.png', bbox_inches='tight')
plt.close()
print("  fig02 OK")

# ── 3. HAS_C3 - Feature más predictiva ──────────────────────────────────────
print("[3] has_C3 análisis...")
c3_rates = df.groupby('has_C3')['target'].mean() * 100
c3_counts = df.groupby('has_C3')['target'].count()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

bars = axes[0].bar(['Sin C3 (has_C3=0)', 'Con C3 (has_C3=1)'],
                   c3_rates.values, color=[BLUE, RED], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, c3_rates.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Tasa de judicialización (%)')
axes[0].set_title('Tasa de judicialización según formulario C-3', fontweight='bold')
axes[0].set_ylim(0, 70)
axes[0].axhline(22.5, color=GRAY, linestyle='--', linewidth=1, label='Media global (22.5%)')
axes[0].legend(fontsize=9)

bars2 = axes[1].bar(['Sin C3', 'Con C3'],
                    c3_counts.values/1e3, color=[BLUE, RED], edgecolor='white')
for bar, val in zip(bars2, c3_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=10)
axes[1].set_ylabel('Registros (miles)')
axes[1].set_title('Distribución de registros C3', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}k'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig03_has_c3.png', bbox_inches='tight')
plt.close()
print("  fig03 OK")

# ── 4. VARIABLES BINARIAS - Tasas de judicialización ────────────────────────
print("[4] Variables binarias...")
bin_features = {
    'has_C2': ('Sin C2', 'Con C2'),
    'has_C3': ('Sin C3', 'Con C3'),
    'has_ANCR_early': ('Sin ANCR early', 'Con ANCR early'),
    'accident_type': ('Enfermedad ocp.', 'Accidente'),
    'occupational_disease': ('No enf. ocp.', 'Enf. ocp.'),
}

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for ax, (col, labels) in zip(axes, bin_features.items()):
    rates_col = df.groupby(col)['target'].mean() * 100
    vals = [rates_col.get(0, 0), rates_col.get(1, 0)]
    bars = ax.bar(labels, vals, color=[BLUE, RED], edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(col.replace('_', '\n'), fontweight='bold', fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25 + 5)
    ax.axhline(22.5, color=GRAY, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='x', labelsize=7.5)

plt.suptitle('Tasa de judicialización por variables binarias', fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig04_binary_features.png', bbox_inches='tight')
plt.close()
print("  fig04 OK")

# ── 5. EDAD Y AWW ────────────────────────────────────────────────────────────
print("[5] Edad y AWW...")
df['age_bin'] = pd.cut(df['age_at_injury'], bins=[0,25,35,45,55,65,100],
                        labels=['<25','25-35','35-45','45-55','55-65','>65'])
df['aww_bin'] = pd.cut(df['aww'], bins=[-1, 0, 300, 600, 900, 1500, 99999],
                        labels=['$0','$1-300','$301-600','$601-900','$901-1500','>$1500'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

age_rates = df.groupby('age_bin', observed=True)['target'].mean() * 100
axes[0].bar(age_rates.index.astype(str), age_rates.values, color=BLUE, edgecolor='white')
for i, (label, v) in enumerate(zip(age_rates.index, age_rates.values)):
    axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
axes[0].axhline(22.5, color=RED, linestyle='--', linewidth=1, label='Media global')
axes[0].set_xlabel('Rango etario')
axes[0].set_ylabel('Tasa de judicialización (%)')
axes[0].set_title('Judicialización por edad al momento del accidente', fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 35)

aww_rates = df.groupby('aww_bin', observed=True)['target'].mean() * 100
axes[1].bar(aww_rates.index.astype(str), aww_rates.values, color=BLUE, edgecolor='white')
for i, (label, v) in enumerate(zip(aww_rates.index, aww_rates.values)):
    axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
axes[1].axhline(22.5, color=RED, linestyle='--', linewidth=1, label='Media global')
axes[1].set_xlabel('Salario semanal promedio (AWW)')
axes[1].set_ylabel('Tasa de judicialización (%)')
axes[1].set_title('Judicialización por salario semanal (AWW)', fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 40)
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig05_age_aww.png', bbox_inches='tight')
plt.close()
print("  fig05 OK")

# ── 6. CARRIER TYPE y DISTRICT ───────────────────────────────────────────────
print("[6] Carrier type y District...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ct_rates = df.groupby('carrier_type')['target'].mean().sort_values(ascending=False) * 100
ct_counts = df.groupby('carrier_type')['target'].count()
colors_ct = [RED if v > 22.5 else BLUE for v in ct_rates.values]
bars = axes[0].barh(ct_rates.index, ct_rates.values, color=colors_ct, edgecolor='white')
for bar, v in zip(bars, ct_rates.values):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
axes[0].axvline(22.5, color=GRAY, linestyle='--', linewidth=1)
axes[0].set_xlabel('Tasa de judicialización (%)')
axes[0].set_title('Judicialización por tipo de aseguradora', fontweight='bold')

dist_rates = df.groupby('district_name')['target'].mean().sort_values(ascending=False) * 100
colors_d = [RED if v > 22.5 else BLUE for v in dist_rates.values]
bars2 = axes[1].barh(dist_rates.index, dist_rates.values, color=colors_d, edgecolor='white')
for bar, v in zip(bars2, dist_rates.values):
    axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
axes[1].axvline(22.5, color=GRAY, linestyle='--', linewidth=1)
axes[1].set_xlabel('Tasa de judicialización (%)')
axes[1].set_title('Judicialización por distrito del WCB', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig06_carrier_district.png', bbox_inches='tight')
plt.close()
print("  fig06 OK")

# ── 7. INDUSTRY CODE ─────────────────────────────────────────────────────────
print("[7] Industry code...")
ind_rates = df.groupby('industry_desc')['target'].mean().sort_values() * 100
ind_counts = df.groupby('industry_desc')['target'].count()
# Solo sectores con más de 5000 registros
valid_ind = ind_counts[ind_counts >= 5000].index
ind_rates = ind_rates[ind_rates.index.isin(valid_ind)]

fig, ax = plt.subplots(figsize=(10, max(5, len(ind_rates)*0.4)))
colors_ind = [RED if v > 22.5 else BLUE for v in ind_rates.values]
bars = ax.barh(ind_rates.index, ind_rates.values, color=colors_ind, edgecolor='white')
for bar, v in zip(bars, ind_rates.values):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=8.5, fontweight='bold')
ax.axvline(22.5, color=GRAY, linestyle='--', linewidth=1.2, label='Media global (22.5%)')
ax.set_xlabel('Tasa de judicialización (%)')
ax.set_title('Tasa de judicialización por sector industrial (NAICS)', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig07_industry.png', bbox_inches='tight')
plt.close()
print("  fig07 OK")

# ── 8. WCIO CODES - Nature of Injury ─────────────────────────────────────────
print("[8] WCIO Nature code...")
nature_map = {
    '10': 'Strain/Tear', '11': 'Strain/Tear-occ', '12': 'Laceration',
    '20': 'Contusion', '21': 'Fracture', '22': 'Dislocation',
    '30': 'Heat burn', '31': 'Infection', '40': 'Amputation',
    '41': 'Enucleation', '50': 'Occupational disease',
    '54': 'Mental stress', '55': 'Hearing loss',
    '59': 'All other occ.', '60': 'Hernia', '62': 'Back disorder',
    '70': 'Carpal tunnel', '80': 'Other trauma', '90': 'Multiple',
    '99': 'Unknown'
}
df['wcio_nature_str'] = df['wcio_nature_code'].astype(str).map(nature_map).fillna(df['wcio_nature_code'].astype(str))
nat_rates = df.groupby('wcio_nature_str')['target'].mean().sort_values() * 100
nat_counts = df.groupby('wcio_nature_str')['target'].count()
valid_nat = nat_counts[nat_counts >= 3000].index
nat_rates = nat_rates[nat_rates.index.isin(valid_nat)]

fig, ax = plt.subplots(figsize=(10, max(5, len(nat_rates)*0.45)))
colors_n = [RED if v > 22.5 else BLUE for v in nat_rates.values]
bars = ax.barh(nat_rates.index, nat_rates.values, color=colors_n, edgecolor='white')
for bar, v in zip(bars, nat_rates.values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=8.5, fontweight='bold')
ax.axvline(22.5, color=GRAY, linestyle='--', linewidth=1.2, label='Media global (22.5%)')
ax.set_xlabel('Tasa de judicialización (%)')
ax.set_title('Tasa de judicialización por naturaleza de la lesión (WCIO)', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig08_wcio_nature.png', bbox_inches='tight')
plt.close()
print("  fig08 OK")

# ── 9. DÍAS A ASSEMBLY Y DÍAS C2 ─────────────────────────────────────────────
print("[9] días a assembly / días C2...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# days_to_assembly binned
d2a = df['days_to_assembly'].dropna()
bins_asm = [0, 7, 14, 21, 30, 60, 90, 365, 9999]
labels_asm = ['≤7','8-14','15-21','22-30','31-60','61-90','91-365','>365']
df['d2a_bin'] = pd.cut(df['days_to_assembly'], bins=bins_asm, labels=labels_asm)
asm_rates = df.groupby('d2a_bin', observed=True)['target'].mean() * 100
axes[0].bar(asm_rates.index.astype(str), asm_rates.values, color=BLUE, edgecolor='white')
for i, v in enumerate(asm_rates.values):
    axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=8.5, fontweight='bold')
axes[0].axhline(22.5, color=RED, linestyle='--', linewidth=1)
axes[0].set_xlabel('Días accidente → ensamblado')
axes[0].set_ylabel('Tasa judicialización (%)')
axes[0].set_title('Judicialización según latencia de ensamblado', fontweight='bold')
axes[0].tick_params(axis='x', rotation=20)

# days_C2 binned
bins_c2 = [0, 7, 14, 21, 30, 60, 90, 365, 9999]
labels_c2 = ['≤7','8-14','15-21','22-30','31-60','61-90','91-365','>365']
df['c2_bin'] = pd.cut(df['days_C2_to_accident'].dropna().reindex(df.index), bins=bins_c2, labels=labels_c2)
c2_rates = df.groupby('c2_bin', observed=True)['target'].mean() * 100
axes[1].bar(c2_rates.index.astype(str), c2_rates.values, color=BLUE, edgecolor='white')
for i, v in enumerate(c2_rates.values):
    axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=8.5, fontweight='bold')
axes[1].axhline(22.5, color=RED, linestyle='--', linewidth=1)
axes[1].set_xlabel('Días accidente → formulario C2')
axes[1].set_ylabel('Tasa judicialización (%)')
axes[1].set_title('Judicialización según velocidad de respuesta del empleador', fontweight='bold')
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig09_assembly_c2_days.png', bbox_inches='tight')
plt.close()
print("  fig09 OK")

# ── 10. GENDER ───────────────────────────────────────────────────────────────
print("[10] Gender...")
gen_rates = df.groupby('gender')['target'].mean().sort_values(ascending=False) * 100
gen_counts = df.groupby('gender')['target'].count()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors_g = [RED if v > 22.5 else BLUE for v in gen_rates.values]
axes[0].bar(gen_rates.index.astype(str), gen_rates.values, color=colors_g, edgecolor='white')
for i, v in enumerate(gen_rates.values):
    axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[0].axhline(22.5, color=GRAY, linestyle='--', linewidth=1)
axes[0].set_ylabel('Tasa de judicialización (%)')
axes[0].set_title('Judicialización por género', fontweight='bold')

axes[1].bar(gen_counts.index.astype(str), gen_counts.values/1e3, color=BLUE, edgecolor='white')
for i, v in enumerate(gen_counts.values):
    axes[1].text(i, v/1e3 + 1, f'{v:,.0f}', ha='center', fontsize=9)
axes[1].set_ylabel('Registros (miles)')
axes[1].set_title('Distribución de registros por género', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig10_gender.png', bbox_inches='tight')
plt.close()
print("  fig10 OK")

# ── 11. COUNTY TOP 15 ────────────────────────────────────────────────────────
print("[11] County...")
county_data = df.groupby('county_of_injury').agg(
    rate=('target', 'mean'), count=('target', 'count')
).reset_index()
county_data['rate'] *= 100
top_county = county_data[county_data['count'] >= 5000].sort_values('rate', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 6))
colors_c = [RED if v > 22.5 else BLUE for v in top_county['rate'].values[::-1]]
bars = ax.barh(top_county['county_of_injury'].values[::-1], top_county['rate'].values[::-1],
               color=colors_c, edgecolor='white')
for bar, v in zip(bars, top_county['rate'].values[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=8.5, fontweight='bold')
ax.axvline(22.5, color=GRAY, linestyle='--', linewidth=1.2, label='Media global')
ax.set_xlabel('Tasa de judicialización (%)')
ax.set_title('Top 20 condados por tasa de judicialización (mín. 5.000 registros)', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig11_county.png', bbox_inches='tight')
plt.close()
print("  fig11 OK")

# ── 12. MEDICAL FEE REGION ───────────────────────────────────────────────────
print("[12] Medical fee region...")
mfr_rates = df.groupby('medical_fee_region')['target'].mean().sort_values(ascending=False) * 100
mfr_counts = df.groupby('medical_fee_region')['target'].count()

fig, ax = plt.subplots(figsize=(8, 4))
colors_r = [RED if v > 22.5 else BLUE for v in mfr_rates.values]
bars = ax.bar(mfr_rates.index.astype(str), mfr_rates.values, color=colors_r, edgecolor='white')
for bar, v in zip(bars, mfr_rates.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(22.5, color=GRAY, linestyle='--', linewidth=1)
ax.set_xlabel('Región médica')
ax.set_ylabel('Tasa de judicialización (%)')
ax.set_title('Judicialización por región médica del WCB', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig12_medical_region.png', bbox_inches='tight')
plt.close()
print("  fig12 OK")

# ── 13. CORRELACIÓN VARIABLES NUMÉRICAS ─────────────────────────────────────
print("[13] Correlation matrix...")
num_for_corr = df[['target','age_at_injury','aww','days_to_assembly',
                   'days_C2_to_accident','has_C2','has_C3','has_ANCR_early',
                   'accident_month','accident_dow']].copy()
corr = num_for_corr.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, vmin=-0.5, center=0,
            square=True, linewidths=0.5, annot=True, fmt='.2f',
            annot_kws={'size': 8.5}, ax=ax)
ax.set_title('Matriz de correlación (variables numéricas y binarias)', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig13_correlation_matrix.png', bbox_inches='tight')
plt.close()
print("  fig13 OK")

# ── 14. DISTRIBUCIONES NUMÉRICAS (boxplots por target) ───────────────────────
print("[14] Boxplots numéricos...")
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
pairs = [('age_at_injury', 'Edad al accidente'),
         ('aww', 'Salario semanal (AWW)'),
         ('days_to_assembly', 'Días hasta ensamblado'),
         ('days_C2_to_accident', 'Días accidente→C2'),
         ('days_C3_to_accident', 'Días accidente→C3'),
         ('accident_month', 'Mes del accidente')]

for ax, (col, title) in zip(axes.flatten(), pairs):
    data_0 = df[df['target']==0][col].dropna()
    data_1 = df[df['target']==1][col].dropna()
    # Clip outliers for visualization
    p99 = df[col].quantile(0.99) if df[col].notna().any() else 1
    data_0 = data_0.clip(upper=p99)
    data_1 = data_1.clip(upper=p99)
    bp = ax.boxplot([data_0, data_1], patch_artist=True,
                    medianprops={'color': 'white', 'linewidth': 2},
                    whiskerprops={'color': GRAY}, capprops={'color': GRAY},
                    flierprops={'marker': '.', 'markersize': 1, 'alpha': 0.3})
    bp['boxes'][0].set_facecolor(BLUE)
    bp['boxes'][1].set_facecolor(RED)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['No judic.', 'Judic.'])
    ax.set_title(title, fontweight='bold', fontsize=10)
    med0 = data_0.median()
    med1 = data_1.median()
    ax.text(0.98, 0.98, f'Med: {med0:.0f} vs {med1:.0f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=GRAY)

plt.suptitle('Distribución de variables numéricas por clase (target 0 vs 1) — p99 cap',
             fontweight='bold', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig14_boxplots_numeric.png', bbox_inches='tight')
plt.close()
print("  fig14 OK")

# ── 15. SEASONALITY - mes y día de semana ────────────────────────────────────
print("[15] Seasonality...")
month_data = df.groupby('accident_month').agg(
    rate=('target','mean'), count=('target','count')).reset_index()
month_data['rate'] *= 100
dow_data = df.groupby('accident_dow').agg(
    rate=('target','mean'), count=('target','count')).reset_index()
dow_data['rate'] *= 100

month_labels = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
dow_labels = ['Dom','Lun','Mar','Mié','Jue','Vie','Sáb']

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax_t1 = axes[0].twinx()
b = axes[0].bar(range(1,13), month_data['count']/1e3, color=LIGHT, edgecolor=BLUE, linewidth=0.8)
axes[0].set_ylabel('Registros (miles)', color=GRAY)
axes[0].tick_params(axis='y', labelcolor=GRAY)
ax_t1.plot(range(1,13), month_data['rate'], color=RED, linewidth=2.5, marker='o', markersize=6)
ax_t1.set_ylabel('Tasa judicialización (%)', color=RED)
ax_t1.tick_params(axis='y', labelcolor=RED)
ax_t1.set_ylim(18, 28)
axes[0].set_xticks(range(1,13))
axes[0].set_xticklabels(month_labels)
axes[0].set_title('Estacionalidad mensual', fontweight='bold')

ax_t2 = axes[1].twinx()
b2 = axes[1].bar(range(7), dow_data['count']/1e3, color=LIGHT, edgecolor=BLUE, linewidth=0.8)
axes[1].set_ylabel('Registros (miles)', color=GRAY)
axes[1].tick_params(axis='y', labelcolor=GRAY)
ax_t2.plot(range(7), dow_data['rate'], color=RED, linewidth=2.5, marker='o', markersize=6)
ax_t2.set_ylabel('Tasa judicialización (%)', color=RED)
ax_t2.tick_params(axis='y', labelcolor=RED)
ax_t2.set_ylim(15, 30)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(dow_labels)
axes[1].set_title('Variación por día de semana', fontweight='bold')

plt.suptitle('Estacionalidad temporal del target', fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig15_seasonality.png', bbox_inches='tight')
plt.close()
print("  fig15 OK")

# ── ESTADÍSTICAS DESCRIPTIVAS PARA EL DOCUMENTO ──────────────────────────────
print("\n=== ESTADÍSTICAS PARA EL DOCUMENTO ===")
print("\n-- Target --")
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True)*100)

print("\n-- Numéricas descriptivas --")
print(df[['age_at_injury','aww','days_to_assembly','days_C2_to_accident']].describe().round(1))

print("\n-- has_C3 rates --")
print(df.groupby('has_C3')['target'].agg(['mean','count']))

print("\n-- Carrier type rates --")
ct = df.groupby('carrier_type')['target'].agg(['mean','count'])
ct['mean'] = (ct['mean']*100).round(1)
print(ct.sort_values('mean', ascending=False))

print("\n-- District rates --")
dr = df.groupby('district_name')['target'].agg(['mean','count'])
dr['mean'] = (dr['mean']*100).round(1)
print(dr.sort_values('mean', ascending=False))

print("\n-- Industry top 10 --")
ind = df.groupby('industry_desc')['target'].agg(['mean','count'])
ind['mean'] = (ind['mean']*100).round(1)
print(ind[ind['count']>=5000].sort_values('mean', ascending=False).head(10))

print("\n-- Gender --")
gen = df.groupby('gender')['target'].agg(['mean','count'])
gen['mean'] = (gen['mean']*100).round(1)
print(gen)

print("\n-- Medical fee region --")
mfr = df.groupby('medical_fee_region')['target'].agg(['mean','count'])
mfr['mean'] = (mfr['mean']*100).round(1)
print(mfr.sort_values('mean', ascending=False))

print("\n-- Yearly trend --")
print(yearly[['accident_year','mean','count']])

print("\n-- Missing values --")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n-- Age stats by target --")
print(df.groupby('target')['age_at_injury'].describe().round(1))

print("\n-- AWW stats by target --")
print(df.groupby('target')['aww'].describe().round(1))

print("\nAll plots saved to:", OUT_DIR)
print("Files:", sorted(os.listdir(OUT_DIR)))
