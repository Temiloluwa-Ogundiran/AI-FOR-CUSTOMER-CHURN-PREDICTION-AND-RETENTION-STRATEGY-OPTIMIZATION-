"""Analyze SYNTHETIC survey data: descriptives, reliability, H01/H02/H03 (demo only).
Uses pandas/numpy/scipy/sklearn only. SYNTHETIC - not real respondents.
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
CONSTRUCTS = {
 "CHD":["CHD1","CHD2","CHD3","CHD4","CHD5","CHD6","CHD7","CHD8","CHD9","CHD10"],
 "AIA":["AIA1","AIA2","AIA3","AIA4","AIA5","AIA6","AIA7","AIA8"],
 "PAC":["PAC1","PAC2","PAC3","PAC4","PAC5","PAC6","PAC7","PAC8"],
 "MLU":["MLU1","MLU2","MLU3","MLU4","MLU5","MLU6","MLU7","MLU8","MLU9"],
 "RSO":["RSO1","RSO2","RSO3","RSO4","RSO5","RSO6","RSO7","RSO8"],
 "CRE":["CRE1","CRE2","CRE3","CRE4","CRE5","CRE6"],
 "IMP":["IMP1","IMP2","IMP3","IMP4","IMP5","IMP6","IMP7","IMP8"],
}
ALL_ITEMS = [c for v in CONSTRUCTS.values() for c in v]

def cronbach(df):
    df = df.dropna()
    k = df.shape[1]
    if len(df) < 3 or k < 2: return np.nan, len(df)
    vt = df.sum(axis=1).var(ddof=1)
    vi = df.var(axis=0, ddof=1).sum()
    return (k/(k-1))*(1-vi/vt) if vt else np.nan, len(df)

def pearson_ci(r, n):
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1/np.sqrt(n-3)
    lo, hi = np.tanh(z-1.96*se), np.tanh(z+1.96*se)
    return float(lo), float(hi)

def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, res, rank, sv = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    e = y - yhat
    n, p = len(y), X1.shape[1]
    s2 = (e @ e)/(n-p)
    cov = s2*np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    t = beta/se
    pval = 2*(1-stats.t.cdf(np.abs(t), n-p))
    ss_tot = ((y-y.mean())**2).sum(); ss_res = (e**2).sum()
    r2 = 1-ss_res/ss_tot
    adj = 1-(1-r2)*(n-1)/(n-p)
    f = ((ss_tot-ss_res)/(p-1))/(ss_res/(n-p))
    fp = 1-stats.f.cdf(f, p-1, n-p)
    return beta, se, t, pval, r2, adj, f, fp, n, p

pilot = pd.read_csv(os.path.join(BASE, "pilot_raw.csv"))
main = pd.read_csv(os.path.join(BASE, "main_raw.csv"))
valid = main[main["Valid_Record"]=="Yes"].copy()
print(f"valid main: {len(valid)}/{len(main)}")

# 1 descriptives
rows = []
for it in ALL_ITEMS:
    s = valid[it].dropna()
    rows.append({"item": it, "N": int(s.count()), "mean": round(float(s.mean()),2), "sd": round(float(s.std()),2)})
desc = pd.DataFrame(rows)
desc.to_csv(os.path.join(BASE, "descriptives_items.csv"), index=False)
crows = []
for c, items in CONSTRUCTS.items():
    s = valid[f"{c}_Mean"].dropna()
    crows.append({"construct": c, "N": int(s.count()), "mean": round(float(s.mean()),3), "sd": round(float(s.std()),3)})
pd.DataFrame(crows).to_csv(os.path.join(BASE, "descriptives_constructs.csv"), index=False)

# 2 reliability (pilot + main-valid)
rel = []
for c, items in CONSTRUCTS.items():
    a_p, n_p = cronbach(pilot[items])
    a_m, n_m = cronbach(valid[items].dropna())
    rel.append({"construct": c, "k": len(items), "pilot_alpha": round(float(a_p),3) if pd.notna(a_p) else None,
                "pilot_n": n_p, "main_alpha": round(float(a_m),3) if pd.notna(a_m) else None, "main_n": n_m,
                "threshold": 0.7, "decision": "Accept" if pd.notna(a_p) and a_p >= 0.7 else "Review"})
pd.DataFrame(rel).to_csv(os.path.join(BASE, "reliability.csv"), index=False)

# 3 H01 Pearson AIA vs PAC
d = valid[["AIA_Mean","PAC_Mean"]].dropna()
r, p = stats.pearsonr(d["AIA_Mean"], d["PAC_Mean"])
lo, hi = pearson_ci(r, len(d))
H01 = {"test": "Pearson AIA_Mean vs PAC_Mean", "n": int(len(d)), "r": round(float(r),3),
       "ci95": [round(lo,3), round(hi,3)], "p": float(p), "alpha": 0.05,
       "decision": "Reject H01 (significant)" if p < 0.05 else "Retain H01"}

# 4 H02 OLS RSO on AIA,PAC,MLU
d2 = valid[["RSO_Mean","AIA_Mean","PAC_Mean","MLU_Mean"]].dropna()
Xn = d2[["AIA_Mean","PAC_Mean","MLU_Mean"]].to_numpy(); y = d2["RSO_Mean"].to_numpy()
beta, se, t, pv, r2, adj, f, fp, n, pp = ols(Xn, y)
H02 = {"test": "OLS RSO_Mean ~ AIA+PAC+MLU", "n": int(n),
       "coef": [{"var": v, "b": round(float(b),3), "se": round(float(s),3), "t": round(float(tt),3), "p": float(ppv)}
                for v, b, s, tt, ppv in zip(["Intercept","AIA_Mean","PAC_Mean","MLU_Mean"], beta, se, t, pv)],
       "R2": round(float(r2),3), "adjR2": round(float(adj),3), "F": round(float(f),2), "p_F": float(fp),
       "decision": "Reject H02 (model significant)" if fp < 0.05 else "Retain H02"}

# 5 H03 ANOVA CRE across A8
g = valid[["A8","CRE_Mean"]].dropna()
groups = [g[g.A8==k].CRE_Mean.to_numpy() for k in [0,1,2,3,4]]
F, p_an = stats.f_oneway(*groups)
W, p_lev = stats.levene(*groups)
Fw, pw = None, None
try:
    from scipy.stats import f as fdist
    # Welch ANOVA manual
    ks = len(groups); ns = np.array([len(x) for x in groups]); ms = np.array([x.mean() for x in groups]); vs = np.array([x.var(ddof=1) for x in groups])
    w = ns/vs; xbar = (w*ms).sum()/w.sum()
    num = ((w*(ms-xbar)**2).sum()/(ks-1))
    den = 1+(2*(ks-2)/(ks**2-1))*(((1-w/w.sum())**2/(ns-1)).sum())
    Fw = num/den
    df1, df2 = ks-1, 1/(((1-w/w.sum())**2/(ns-1)).sum()/( (3/((ks**2-1))) )) if False else None
except Exception: pass
grand = g.CRE_Mean.mean(); ssb = sum(len(x)*(x.mean()-grand)**2 for x in groups); sst = ((g.CRE_Mean-grand)**2).sum()
eta2 = ssb/sst if sst else 0
grow = [{"A8": int(k), "n": int((g.A8==k).sum()), "mean": round(float(g[g.A8==k].CRE_Mean.mean()),3),
         "sd": round(float(g[g.A8==k].CRE_Mean.std()),3)} for k in [0,1,2,3,4]]
# Bonferroni pairwise
pairs = []
import itertools
for a, b in itertools.combinations([0,1,2,3,4], 2):
    xa, xb = g[g.A8==a].CRE_Mean, g[g.A8==b].CRE_Mean
    tt, ppv = stats.ttest_ind(xa, xb, equal_var=False)
    pairs.append({"pair": f"{a} vs {b}", "p_uncorrected": round(float(ppv),4), "p_bonferroni": round(min(float(ppv)*10,1.0),4)})
H03 = {"test": "One-way ANOVA CRE_Mean by A8", "groups": grow, "F": round(float(F),3), "p": float(p_an),
       "eta2": round(float(eta2),3), "levene_W": round(float(W),3), "levene_p": float(p_lev),
       "pairwise_bonferroni": pairs, "decision": "Reject H03 (differs)" if p_an < 0.05 else "Retain H03",
       "note": "Perception-based unless audited churn-rate data added. No causality from survey alone."}

report = {"SYNTHETIC_NOTE": "SYNTHETIC DEMONSTRATION ONLY - pipeline verification, not real respondents or findings.",
          "n_pilot": len(pilot), "n_main": len(main), "n_valid": int(len(valid)),
          "H01": H01, "H02": H02, "H03": H03}
json.dump(report, open(os.path.join(BASE, "analysis_report.json"), "w"), indent=2)
print(json.dumps({k: report[k] for k in ["H01","H02","H03"]}, indent=2))

# SPSS syntax
sps = """* SYNTHETIC DEMO ONLY - MIT AI Churn survey.
GET DATA /TYPE=TXT /FILE="main_raw.csv" /DELCASE=LINE /DELIMITERS="," /QUALIFIER='"'
 /ARRANGEMENT=DELIMITED /FIRSTCASE=2 /VARIABLES=
 Respondent_ID A6 Valid_Record A3 Consent A3 A1 A3 A2 A3 A3 A3 A4 A3 A5 A3 A6 A3
 A7 A3 A8 F1.0 A9 A3 A10 A3
 CHD1 F1.0 CHD2 F1.0 CHD3 F1.0 CHD4 F1.0 CHD5 F1.0 CHD6 F1.0 CHD7 F1.0 CHD8 F1.0 CHD9 F1.0 CHD10 F1.0
 AIA1 F1.0 AIA2 F1.0 AIA3 F1.0 AIA4 F1.0 AIA5 F1.0 AIA6 F1.0 AIA7 F1.0 AIA8 F1.0
 PAC1 F1.0 PAC2 F1.0 PAC3 F1.0 PAC4 F1.0 PAC5 F1.0 PAC6 F1.0 PAC7 F1.0 PAC8 F1.0
 MLU1 F1.0 MLU2 F1.0 MLU3 F1.0 MLU4 F1.0 MLU5 F1.0 MLU6 F1.0 MLU7 F1.0 MLU8 F1.0 MLU9 F1.0
 RSO1 F1.0 RSO2 F1.0 RSO3 F1.0 RSO4 F1.0 RSO5 F1.0 RSO6 F1.0 RSO7 F1.0 RSO8 F1.0
 CRE1 F1.0 CRE2 F1.0 CRE3 F1.0 CRE4 F1.0 CRE5 F1.0 CRE6 F1.0
 IMP1 F1.0 IMP2 F1.0 IMP3 F1.0 IMP4 F1.0 IMP5 F1.0 IMP6 F1.0 IMP7 F1.0 IMP8 F1.0 .
* Composites (80% rule via MEAN.n).
COMPUTE CHD_Mean=MEAN.8(CHD1 TO CHD10).
COMPUTE AIA_Mean=MEAN.7(AIA1 TO AIA8).
COMPUTE PAC_Mean=MEAN.7(PAC1 TO PAC8).
COMPUTE MLU_Mean=MEAN.8(MLU1 TO MLU9).
COMPUTE RSO_Mean=MEAN.7(RSO1 TO RSO8).
COMPUTE CRE_Mean=MEAN.5(CRE1 TO CRE6).
COMPUTE IMP_Mean=MEAN.7(IMP1 TO IMP8).
EXECUTE.
* Reliability (pilot n=30, then main).
RELIABILITY /VARIABLES=CHD1 TO CHD10 /MODEL=ALPHA.
RELIABILITY /VARIABLES=AIA1 TO AIA8 /MODEL=ALPHA.
RELIABILITY /VARIABLES=PAC1 TO PAC8 /MODEL=ALPHA.
RELIABILITY /VARIABLES=MLU1 TO MLU9 /MODEL=ALPHA.
RELIABILITY /VARIABLES=RSO1 TO RSO8 /MODEL=ALPHA.
RELIABILITY /VARIABLES=CRE1 TO CRE6 /MODEL=ALPHA.
RELIABILITY /VARIABLES=IMP1 TO IMP8 /MODEL=ALPHA.
* Descriptives RQ1-RQ6.
FREQUENCIES A1 A2 A3 A4 A5 A6 A8 A9 A10 /STATISTICS=NONE.
DESCRIPTIVES CHD1 TO IMP8 CHD_Mean AIA_Mean PAC_Mean MLU_Mean RSO_Mean CRE_Mean IMP_Mean /STATISTICS=MEAN STDDEV MIN MAX.
* H01 Pearson.
CORRELATIONS /VARIABLES=AIA_Mean PAC_Mean /PRINT=TWOTAIL NOSIG /CI.
* H02 Multiple regression.
REGRESSION /MISSING LISTWISE /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL
 /DEPENDENT RSO_Mean /METHOD=ENTER AIA_Mean PAC_Mean MLU_Mean /RESIDUALS HISTOGRAM(ZRESID) NORMPROB(ZRESID).
* H03 ANOVA + Welch + post-hoc.
ONEWAY CRE_Mean BY A8 /STATISTICS DESCRIPTIVES HOMOGENEITY WELCH /POSTHOC=TUKEY BONFERRONI ALPHA(0.05).
"""
open(os.path.join(BASE, "SPSS_syntax.sps"), "w").write(sps)
print("Wrote descriptives, reliability, analysis_report.json, SPSS_syntax.sps")
