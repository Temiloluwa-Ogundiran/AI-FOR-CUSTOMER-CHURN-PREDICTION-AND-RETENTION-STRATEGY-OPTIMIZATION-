"""Generate SYNTHETIC survey data (demo only) matching thesis codebook + Ch.5 target means.
Outputs: pilot_raw.csv (30), main_raw.csv (420), filled workbook copy.
SYNTHETIC DEMONSTRATION ONLY - not real respondents. For pipeline verification.
"""
import os, numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260810

# Target means from thesis Ch.5 (Tables 5.2-5.8)
TARGETS = {
 "CHD1":3.12,"CHD2":2.85,"CHD3":3.41,"CHD4":2.65,"CHD5":2.95,"CHD6":3.20,"CHD7":2.55,"CHD8":3.05,"CHD9":3.35,"CHD10":2.75,
 "AIA1":3.15,"AIA2":2.70,"AIA3":3.38,"AIA4":2.58,"AIA5":2.92,"AIA6":3.08,"AIA7":3.22,"AIA8":2.85,
 "PAC1":2.78,"PAC2":3.18,"PAC3":2.62,"PAC4":3.48,"PAC5":2.95,"PAC6":3.28,"PAC7":3.02,"PAC8":2.88,
 "MLU1":3.32,"MLU2":2.80,"MLU3":3.05,"MLU4":2.65,"MLU5":3.40,"MLU6":2.98,"MLU7":3.12,"MLU8":2.75,"MLU9":3.14,
 "RSO1":3.32,"RSO2":2.80,"RSO3":3.05,"RSO4":2.65,"RSO5":3.40,"RSO6":2.98,"RSO7":3.12,"RSO8":2.75,
 "CRE1":2.94,"CRE2":3.26,"CRE3":2.52,"CRE4":3.46,"CRE5":3.10,"CRE6":2.86,
 "IMP1":3.08,"IMP2":2.60,"IMP3":3.35,"IMP4":2.84,"IMP5":3.18,"IMP6":2.95,"IMP7":3.42,"IMP8":2.72,
}
LIKERT_COLS = list(TARGETS.keys())
CONSTRUCTS = {
 "CHD":["CHD1","CHD2","CHD3","CHD4","CHD5","CHD6","CHD7","CHD8","CHD9","CHD10"],
 "AIA":["AIA1","AIA2","AIA3","AIA4","AIA5","AIA6","AIA7","AIA8"],
 "PAC":["PAC1","PAC2","PAC3","PAC4","PAC5","PAC6","PAC7","PAC8"],
 "MLU":["MLU1","MLU2","MLU3","MLU4","MLU5","MLU6","MLU7","MLU8","MLU9"],
 "RSO":["RSO1","RSO2","RSO3","RSO4","RSO5","RSO6","RSO7","RSO8"],
 "CRE":["CRE1","CRE2","CRE3","CRE4","CRE5","CRE6"],
 "IMP":["IMP1","IMP2","IMP3","IMP4","IMP5","IMP6","IMP7","IMP8"],
}
THRESH = {"CHD":8,"AIA":7,"PAC":7,"MLU":8,"RSO":7,"CRE":5,"IMP":7}  # 80% rule

def draw_likert(n, rng, maturity):
    """Latent-factor model to create H01/H02/H03 structure, then shift to target means."""
    # latent traits
    L_ai = rng.normal(0, 1, n) + 0.35 * maturity  # AI exposure rises with maturity
    L_pac = 0.6 * L_ai + rng.normal(0, 0.8, n)    # H01: AIA<->PAC correlated
    L_mlu = 0.3 * L_ai + rng.normal(0, 0.9, n)
    L_rso = 0.45 * L_pac + 0.30 * L_ai + 0.20 * L_mlu + rng.normal(0, 0.6, n)  # H02 structure
    L_cre = 0.30 * maturity + 0.25 * L_rso + rng.normal(0, 0.7, n)  # H03: CRE rises with A8
    L_chd = rng.normal(0, 1, n)
    L_imp = rng.normal(0, 1, n)
    lat = {"CHD": L_chd, "AIA": L_ai, "PAC": L_pac, "MLU": L_mlu, "RSO": L_rso, "CRE": L_cre, "IMP": L_imp}
    out = pd.DataFrame(index=range(n))
    for con, items in CONSTRUCTS.items():
        for j, it in enumerate(items):
            # item-specific noise; scale ~0.6 gives alpha ~0.75-0.85
            v = 3.0 + 0.70 * lat[con] + rng.normal(0, 0.6, n)
            out[it] = v
    # shift each column to hit target mean (preserves correlations)
    for it in LIKERT_COLS:
        out[it] = out[it] - out[it].mean() + TARGETS[it]
        out[it] = np.clip(np.round(out[it]), 1, 5).astype(float)
    return out

def demographics(n, rng, pilot=False):
    d = pd.DataFrame(index=range(n))
    d["A1"] = rng.choice(["Male","Female","Prefer not to say"], n, p=[0.55,0.42,0.03])
    d["A2"] = rng.choice(["18-24","25-34","35-44","45-54","55 or above"], n, p=[0.1,0.35,0.3,0.18,0.07])
    d["A3"] = rng.choice(["Diploma/OND/NCE","Bachelor/HND","Postgraduate Diploma","Master's","Doctorate","Other"], n, p=[0.12,0.45,0.12,0.24,0.04,0.03])
    d["A4"] = rng.choice(["Commercial bank","Digital bank/neobank","Merchant bank","Microfinance bank","Fintech/payment provider","Insurance/investment/other licensed financial institution"], n, p=[0.4,0.15,0.08,0.12,0.18,0.07])
    d["A5"] = rng.choice(["Digital Banking/Product","Data/Analytics/AI","IT/Technology","Customer Experience/Service","Relationship/Marketing","Risk/Compliance","Management/Strategy","Other"], n, p=[0.18,0.14,0.16,0.16,0.12,0.08,0.08,0.08])
    d["A6"] = rng.choice(["Less than 2","2-5","6-10","11-15","16 or more"], n, p=[0.15,0.35,0.28,0.14,0.08])
    # A8 maturity distribution
    if pilot:
        d["A8"] = rng.choice([0,1,2,3,4], n, p=[0.25,0.35,0.25,0.10,0.05])
    else:
        d["A8"] = rng.choice([0,1,2,3,4], n, p=[0.12,0.22,0.28,0.24,0.14])
    d["A9"] = np.where(d["A8"] >= 2, rng.choice(["Yes","No","Do not know"], n, p=[0.7,0.2,0.1]),
                        rng.choice(["Yes","No","Do not know"], n, p=[0.25,0.55,0.2]))
    return d

def build(n, seed, pilot=False, prefix="R"):
    rng = np.random.default_rng(seed)
    dem = demographics(n, rng, pilot)
    lik = draw_likert(n, rng, dem["A8"].to_numpy())
    df = pd.DataFrame()
    df["Respondent_ID"] = [f"{prefix}{i:03d}" for i in range(1, n+1)]
    df["Consent"] = "Yes"
    df = pd.concat([df, dem[["A1","A2","A3","A4","A5","A6"]]], axis=1)
    df["A7"] = "Yes"
    df["A8"] = dem["A8"]
    df["A9"] = dem["A9"]
    df["A10"] = "No" if pilot else "Yes"
    df = pd.concat([df, lik], axis=1)
    # ~2% MCAR missing on Likert to exercise 80% rule
    for c in LIKERT_COLS:
        m = rng.random(n) < 0.02
        df.loc[m, c] = np.nan
    # a few invalid records in main (declined consent / outside LGA) to demo tracker
    if not pilot:
        df.loc[df.index[:5], "Consent"] = "No"
        df.loc[df.index[5:8], "A10"] = "No"
    # Valid_Record: Consent Yes + A10 Yes + >=80% overall answered (simplify: all constructs pass or nearly)
    df["Valid_Record"] = np.where((df["Consent"]=="Yes") & (df["A10"]=="Yes"), "Yes", "No")
    # composites with 80% rule
    for con, items in CONSTRUCTS.items():
        cnt = df[items].notna().sum(axis=1)
        df[f"{con}_Mean"] = np.where(cnt >= THRESH[con], df[items].mean(axis=1, skipna=True), np.nan)
    # column order to match template: Respondent_ID, Valid_Record, Consent, A1..A10, items..., means
    order = ["Respondent_ID","Valid_Record","Consent","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10"] + LIKERT_COLS + [f"{c}_Mean" for c in ["CHD","AIA","PAC","MLU","RSO","CRE","IMP"]]
    return df[order]

def main():
    pilot = build(30, SEED, pilot=True, prefix="P")
    main_df = build(420, SEED+1, pilot=False, prefix="R")
    pilot.to_csv(os.path.join(BASE, "pilot_raw.csv"), index=False)
    main_df.to_csv(os.path.join(BASE, "main_raw.csv"), index=False)
    print(f"pilot valid: {(pilot['Valid_Record']=='Yes').sum()}/{len(pilot)} | main valid: {(main_df['Valid_Record']=='Yes').sum()}/{len(main_df)}")
    # filled workbook copy preserving template formulas where possible
    try:
        import openpyxl
        tpl = os.path.join(os.path.dirname(BASE), "MIT_AI_Churn_Data_Coding_and_Analysis_Template.xlsx")
        wb = openpyxl.load_workbook(tpl)
        ws = wb["Raw_Data"]
        # clear old rows 2..501
        for r in range(2, 502):
            for c in range(1, 78):
                ws.cell(r, c).value = None
        hdr = [c.value for c in ws[1]]
        combined = pd.concat([pilot, main_df], ignore_index=True).head(500)
        for i, (_, row) in enumerate(combined.iterrows(), 2):
            for j, h in enumerate(hdr, 1):
                if h in combined.columns:
                    v = row[h]
                    # leave composite means as formulas (template rule)
                    if h.endswith("_Mean"):
                        continue
                    ws.cell(i, j).value = None if pd.isna(v) else (int(v) if h in ["A8"]+LIKERT_COLS and pd.notna(v) else v)
            # restore mean formulas row i (cols BT..BZ = 72..78)
            # CHD N:W (14:23), AIA X:AE (24:31), PAC AF:AM (32:39), MLU AN:AV (40:48), RSO AW:BD (49:56), CRE BE:BJ (57:62), IMP BK:BR (63:70), means BT:BZ (72:78)? check: actually means are cols 71-77
            # simpler: re-apply known formulas with row i
            ws.cell(i, 71).value = f"=IF(COUNT(N{i}:W{i})>=8,AVERAGE(N{i}:W{i}),\"\")"
            ws.cell(i, 72).value = f"=IF(COUNT(X{i}:AE{i})>=7,AVERAGE(X{i}:AE{i}),\"\")"
            ws.cell(i, 73).value = f"=IF(COUNT(AF{i}:AM{i})>=7,AVERAGE(AF{i}:AM{i}),\"\")"
            ws.cell(i, 74).value = f"=IF(COUNT(AN{i}:AV{i})>=8,AVERAGE(AN{i}:AV{i}),\"\")"
            ws.cell(i, 75).value = f"=IF(COUNT(AW{i}:BD{i})>=7,AVERAGE(AW{i}:BD{i}),\"\")"
            ws.cell(i, 76).value = f"=IF(COUNT(BE{i}:BJ{i})>=5,AVERAGE(BE{i}:BJ{i}),\"\")"
            ws.cell(i, 77).value = f"=IF(COUNT(BK{i}:BR{i})>=7,AVERAGE(BK{i}:BR{i}),\"\")"
        out = os.path.join(BASE, "MIT_AI_Churn_SYNTHETIC_filled.xlsx")
        wb.save(out)
        print("Filled workbook:", out)
    except Exception as e:
        print("Workbook fill skipped:", e)

if __name__ == "__main__":
    main()
