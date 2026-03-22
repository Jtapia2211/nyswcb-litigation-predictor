"""
build_dataset.py - v2
=====================
Construye el dataset analítico limpio para la tesis de Julian Tapia.

TARGET: Highest Process ∈ {4A. HEARING - JUDGE, 4B. HEARING - APPEAL, 4C. HEARING - SETTLEMENT}
        = La claim requirió proceso judicial/cuasi-judicial formal.

VENTANA DE PREDICCIÓN: Información disponible en semanas 1-4 post-accidente.
FILTRO BASE: Claim Type = 'WORKERS COMPENSATION CLAIM', Accident Date 2017-2022.

LEAKAGE IDENTIFICADO Y CORREGIDO:
  - claim_injury_type: determinado POST-proceso (PPD NSL=99.7%, PTD=99.8%). EXCLUIDA.
  - has_ANCR (global): reemplazado por has_ANCR_early (ANCR en ≤28 días post-assembly).
  - Controverted Date, First Hearing Date, First Appeal Date, Section 32 Date: EXCLUIDAS.
  - Hearing Count, Closed Count, IME-4 Count: EXCLUIDAS.
  - Attorney/Representative: timing ambiguo, EXCLUIDA.
  - PPD dates, PTD Date, Current Claim Status: EXCLUIDAS.
"""

import csv
from datetime import datetime

INPUT_PATH  = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/raw_data/nyswcb_claims.csv'
OUTPUT_PATH = '/sessions/epic-intelligent-hawking/mnt/Tesis_ML/raw_data/dataset_tesis_clean.csv'

JUDICIAL_PROCESSES = {
    '4A. HEARING - JUDGE',
    '4B. HEARING - APPEAL',
    '4C. HEARING - SETTLEMENT',
}

# ── Helper functions ───────────────────────────────────────────────────────────
def parse_date(s):
    if not s or s.strip() == '':
        return None
    s = s.strip()
    for fmt in ('%m/%d/%Y', '%Y-%m-%d', '%m/%d/%y'):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    return None

def days_between(d1, d2):
    if d1 and d2:
        return (d2 - d1).days
    return None

def clean_aww(s):
    if not s or s.strip() == '':
        return None
    try:
        return float(s.strip().replace('$', '').replace(',', ''))
    except:
        return None

def clean_float(s):
    if not s or s.strip() == '':
        return None
    try:
        return float(s.strip())
    except:
        return None

def clean_str(s):
    v = (s or '').strip()
    return v if v else None

# ── Main build ─────────────────────────────────────────────────────────────────
def build_dataset():
    print("=" * 60)
    print("BUILD DATASET TESIS v2 - JULIAN TAPIA")
    print("=" * 60)

    out_cols = [
        # ── Target ──────────────────────────────────────────────────────────
        'target',
        # ── Temporal features ───────────────────────────────────────────────
        'accident_year',
        'accident_month',
        'accident_dow',           # day of week (0=Mon, 6=Sun)
        'days_to_assembly',       # latencia administrativa
        # ── Señales procesales tempranas (semanas 1-4) ─────────────────────
        'has_C2',                 # empleador presentó reporte
        'days_C2_to_accident',    # velocidad de respuesta del empleador
        'has_C3',                 # trabajador presentó formulario de reclamo
        'days_C3_to_accident',    # velocidad del trabajador en reclamar
        'has_ANCR_early',         # ANCR establecido en ≤28 días post-assembly
        # ── Perfil del trabajador ────────────────────────────────────────────
        'age_at_injury',
        'gender',
        # ── Características del accidente ────────────────────────────────────
        'accident_type',          # Y/N: accidente laboral (no enfermedad)
        'occupational_disease',   # Y/N
        'county_of_injury',
        'medical_fee_region',
        # ── Códigos de lesión (WCIO) ─────────────────────────────────────────
        'wcio_cause_code',
        'wcio_nature_code',
        'wcio_body_code',
        # ── Económico ────────────────────────────────────────────────────────
        'aww',
        # ── Aseguradora / cobertura ──────────────────────────────────────────
        'carrier_type',
        'district_name',
        # ── Actividad económica ──────────────────────────────────────────────
        'industry_code',
        'industry_desc',
        # ── REFERENCIA SOLO (no usar como feature - es leakage) ───────────────
        # claim_injury_type: determinado DESPUÉS del proceso judicial
        'claim_injury_type_REF',  # referencia analítica únicamente
    ]

    total = 0
    kept = 0
    skipped_type = 0
    skipped_year = 0
    target_counts = {0: 0, 1: 0}

    with open(INPUT_PATH, 'r', encoding='utf-8-sig') as fin, \
         open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=out_cols)
        writer.writeheader()

        for row in reader:
            total += 1
            if total % 500_000 == 0:
                rate = target_counts[1]/(kept+1)*100
                print(f"  Leídos: {total:>8,} | Guardados: {kept:>8,} | "
                      f"Judicialized: {rate:.1f}%", flush=True)

            # ── Filtro 1: Solo Workers Compensation Claims ──────────────────
            if row.get('Claim Type', '').strip() != 'WORKERS COMPENSATION CLAIM':
                skipped_type += 1
                continue

            # ── Parse dates ──────────────────────────────────────────────────
            accident_date = parse_date(row.get('Accident Date', ''))
            assembly_date = parse_date(row.get('Assembly Date', ''))
            c2_date       = parse_date(row.get('C-2 Date', ''))
            c3_date       = parse_date(row.get('C-3 Date', ''))

            # ── Filtro 2: 2017-2022 ──────────────────────────────────────────
            if accident_date is None:
                skipped_year += 1
                continue
            year = accident_date.year
            if year < 2017 or year > 2022:
                skipped_year += 1
                continue

            # ── Target ───────────────────────────────────────────────────────
            highest = row.get('Highest Process', '').strip()
            target = 1 if highest in JUDICIAL_PROCESSES else 0

            # ── Feature engineering ──────────────────────────────────────────
            days_to_asm = days_between(accident_date, assembly_date)
            days_c2     = days_between(accident_date, c2_date)
            days_c3     = days_between(accident_date, c3_date)

            # ANCR early: established within 28 days of assembly
            ancr_interval_raw = row.get('Interval Assembled to ANCR', '').strip()
            ancr_days = clean_float(ancr_interval_raw)
            if ancr_days is not None and 0 <= ancr_days <= 28:
                has_ancr_early = 1
            else:
                has_ancr_early = 0

            aww = clean_aww(row.get('Average Weekly Wage (AWW)', ''))
            age = clean_float(row.get('Age at Injury', ''))

            # Validations
            if days_to_asm is not None and not (0 <= days_to_asm <= 730):
                days_to_asm = None
            if days_c2 is not None and not (-30 <= days_c2 <= 365):
                days_c2 = None
            if days_c3 is not None and not (0 <= days_c3 <= 730):
                days_c3 = None
            if age is not None and not (15 <= age <= 85):
                age = None

            out = {
                'target':             target,
                'accident_year':      year,
                'accident_month':     accident_date.month,
                'accident_dow':       accident_date.weekday(),
                'days_to_assembly':   days_to_asm,
                'has_C2':             1 if c2_date else 0,
                'days_C2_to_accident': days_c2,
                'has_C3':             1 if c3_date else 0,
                'days_C3_to_accident': days_c3,
                'has_ANCR_early':     has_ancr_early,
                'age_at_injury':      age,
                'gender':             clean_str(row.get('Gender', '')),
                'accident_type':      clean_str(row.get('Accident', '')),
                'occupational_disease': clean_str(row.get('Occupational Disease', '')),
                'county_of_injury':   clean_str(row.get('County of Injury', '')),
                'medical_fee_region': clean_str(row.get('Medical Fee Region', '')),
                'wcio_cause_code':    clean_str(row.get('WCIO Cause of Injury Code', '')),
                'wcio_nature_code':   clean_str(row.get('WCIO Nature of Injury Code', '')),
                'wcio_body_code':     clean_str(row.get('WCIO Part Of Body Code', '')),
                'aww':                aww,
                'carrier_type':       clean_str(row.get('Carrier Type', '')),
                'district_name':      clean_str(row.get('District Name', '')),
                'industry_code':      clean_str(row.get('Industry Code', '')),
                'industry_desc':      clean_str(row.get('Industry Code Description', '')),
                'claim_injury_type_REF': clean_str(row.get('Claim Injury Type', '')),
            }

            writer.writerow(out)
            kept += 1
            target_counts[target] += 1

    print()
    print("=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Total registros leídos:             {total:>10,}")
    print(f"Excluidos (no WC Claim):            {skipped_type:>10,}")
    print(f"Excluidos (fuera rango 2017-2022):  {skipped_year:>10,}")
    print(f"Dataset final:                      {kept:>10,}")
    print()
    print(f"TARGET = 0 (No judicializado):      {target_counts[0]:>10,} ({target_counts[0]/kept*100:.1f}%)")
    print(f"TARGET = 1 (Judicializado):         {target_counts[1]:>10,} ({target_counts[1]/kept*100:.1f}%)")
    print()
    print(f"Variables de features: 23 (sin leakage)")
    print(f"Variable REF (no usar): claim_injury_type_REF")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == '__main__':
    build_dataset()
