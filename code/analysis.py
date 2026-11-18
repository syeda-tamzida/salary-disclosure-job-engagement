"""DS502 Group 4 - Phase 3 progress report analysis.
Salary disclosure vs. job-posting engagement across career levels.
Usage: python3 -I analysis.py <data_dir> <out_dir>
"""
import sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2])
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
SEED = 502
res = {}

# ---------------------------------------------------------------- load
cols = ["job_id", "company_id", "title", "description", "views", "applies",
        "normalized_salary", "pay_period", "formatted_experience_level",
        "formatted_work_type", "remote_allowed", "application_type",
        "location", "original_listed_time", "listed_time"]
raw = pd.read_csv(DATA / "postings.csv", usecols=cols)
res["n_raw"] = len(raw)

df = raw.copy()
df["disclosed"] = df["normalized_salary"].notna().astype(int)
lvl_map = {"Internship": "Internship", "Entry level": "Entry", "Associate": "Associate",
           "Mid-Senior level": "Mid-Senior", "Director": "Director", "Executive": "Executive"}
df["level"] = df["formatted_experience_level"].map(lvl_map).fillna("Not specified")
LEVELS = ["Internship", "Entry", "Associate", "Mid-Senior", "Director", "Executive"]

# posting age (exposure) relative to the scrape snapshot
snap = df["listed_time"].max()
df["age_days"] = (snap - df["original_listed_time"]) / 864e5
df["log_age"] = np.log1p(df["age_days"].clip(lower=0))
df["remote"] = df["remote_allowed"].fillna(0).astype(int)
df["work_type"] = df["formatted_work_type"].replace({"Volunteer": "Other", "Temporary": "Other"})
df["app_type"] = df["application_type"].replace({"UnknownApply": "OffsiteApply"})
df["log_desc_len"] = np.log1p(df["description"].fillna("").str.len())

# US census region from "City, ST"
REG = {"Northeast": "CT ME MA NH RI VT NJ NY PA", "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
       "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
       "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA"}
st2reg = {s: r for r, ss in REG.items() for s in ss.split()}
st = df["location"].fillna("").str.extract(r",\s*([A-Z]{2})$")[0]
df["region"] = st.map(st2reg).fillna("Unknown/National")

# industry (first listed), top 12 + Other
ji = pd.read_csv(DATA / "jobs/job_industries.csv")
im = pd.read_csv(DATA / "mappings/industries.csv")
ji = ji.merge(im, on="industry_id", how="left").drop_duplicates("job_id")
df = df.merge(ji[["job_id", "industry_name"]], on="job_id", how="left")
top = df["industry_name"].value_counts().head(12).index
df["industry"] = df["industry_name"].where(df["industry_name"].isin(top), "Other")
df["industry"] = df["industry"].fillna("Other")

df["log_views"] = np.log1p(df["views"])
df["log_applies"] = np.log1p(df["applies"])

res["n_level_known"] = int((df.level != "Not specified").sum())
res["views_missing_pct"] = float(df["views"].isna().mean() * 100)
res["applies_missing_pct"] = float(df["applies"].isna().mean() * 100)
res["disclosure_rate"] = float(df["disclosed"].mean() * 100)
res["applies_obs_by_apptype"] = df.groupby("app_type")["applies"].apply(lambda s: round(s.notna().mean() * 100, 1)).to_dict()
res["pay_period"] = df["pay_period"].value_counts().to_dict()
res["views_q"] = df["views"].quantile([.5, .9, .99]).round(1).to_dict()
res["applies_q"] = df["applies"].quantile([.5, .9, .99]).round(1).to_dict()
res["views_skew_raw"] = float(df["views"].skew()); res["views_skew_log"] = float(df["log_views"].skew())
res["applies_skew_raw"] = float(df["applies"].skew()); res["applies_skew_log"] = float(df["log_applies"].skew())

main = df[df.level.isin(LEVELS)].copy()
main["level"] = pd.Categorical(main["level"], categories=["Entry"] + [l for l in LEVELS if l != "Entry"])
V = main.dropna(subset=["views"]).copy()
A = main.dropna(subset=["applies"]).copy()
res["n_views"] = len(V); res["n_applies"] = len(A)

# descriptive table by level
desc = main.groupby("level", observed=True).agg(
    n=("job_id", "size"), disc_rate=("disclosed", "mean"),
    med_views=("views", "median"), med_applies=("applies", "median")).reindex(LEVELS)
d2 = main.groupby(["level", "disclosed"], observed=True)[["views", "applies"]].median().unstack()
desc["med_views_nd"] = d2[("views", 0)]; desc["med_views_d"] = d2[("views", 1)]
desc["med_app_nd"] = d2[("applies", 0)]; desc["med_app_d"] = d2[("applies", 1)]
res["desc_by_level"] = desc.round(3).reset_index().to_dict(orient="records")

# ---------------------------------------------------------------- OLS (inference)
CTRL = "C(work_type) + remote + C(app_type) + log_age + log_desc_len + C(region) + C(industry)"
forms = {
    "M0": "{y} ~ disclosed",
    "M1": "{y} ~ disclosed + C(level) + " + CTRL,
    "M2": "{y} ~ disclosed * C(level) + " + CTRL,
}

def level_effects(fit, ref="Entry"):
    """Disclosure effect at each career level = main effect + interaction, with cluster-robust SE."""
    out = []
    b = fit.params; cov = fit.cov_params()
    for l in LEVELS:
        names = ["disclosed"] if l == ref else ["disclosed", f"disclosed:C(level)[T.{l}]"]
        w = pd.Series(1.0, index=names)
        est = float((b[names] * w).sum())
        se = float(np.sqrt(w @ cov.loc[names, names] @ w))
        out.append({"level": l, "beta": est, "se": se, "lo": est - 1.96 * se, "hi": est + 1.96 * se,
                    "pct": (np.exp(est) - 1) * 100, "pct_lo": (np.exp(est - 1.96 * se) - 1) * 100,
                    "pct_hi": (np.exp(est + 1.96 * se) - 1) * 100})
    return out

ols = {}
for tag, D, y in [("views", V, "log_views"), ("applies", A, "log_applies")]:
    ols[tag] = {}
    for m, f in forms.items():
        fit = smf.ols(f.format(y=y), data=D).fit(cov_type="cluster",
                                                 cov_kwds={"groups": D["company_id"].fillna(-1).astype(int)})
        ols[tag][m] = fit
        res[f"ols_{tag}_{m}"] = {"disclosed_b": float(fit.params["disclosed"]),
                                 "disclosed_se": float(fit.bse["disclosed"]),
                                 "disclosed_p": float(fit.pvalues["disclosed"]),
                                 "r2": float(fit.rsquared), "adj_r2": float(fit.rsquared_adj),
                                 "n": int(fit.nobs), "k": int(fit.df_model)}
    # joint Wald test of all interaction terms
    m2 = ols[tag]["M2"]
    inter = [p for p in m2.params.index if p.startswith("disclosed:")]
    R = np.zeros((len(inter), len(m2.params)))
    for i, p in enumerate(inter):
        R[i, list(m2.params.index).index(p)] = 1
    w = m2.wald_test(R, scalar=True)
    res[f"wald_{tag}"] = {"F": float(w.statistic), "p": float(w.pvalue), "df": len(inter)}
    res[f"inter_{tag}"] = {p: {"b": float(m2.params[p]), "se": float(m2.bse[p]), "p": float(m2.pvalues[p])} for p in inter}
    res[f"levels_{tag}"] = level_effects(m2)
    # selected control coefficients from M1
    m1 = ols[tag]["M1"]
    keep = [p for p in m1.params.index if p.startswith(("remote", "log_age", "log_desc", "C(app_type)", "C(level)"))]
    res[f"ctrl_{tag}"] = {p: {"b": float(m1.params[p]), "se": float(m1.bse[p])} for p in keep}

# ---------------------------------------------------------------- predictive CV: OLS vs Ridge vs Lasso
import patsy
def cv_compare(D, y):
    kf = KFold(5, shuffle=True, random_state=SEED)
    rows = {}
    specs = {"Mean baseline": None, "OLS M0 (disclosure only)": forms["M0"],
             "OLS M1 (+ level, controls)": forms["M1"], "OLS M2 (+ interaction)": forms["M2"],
             "Ridge (M2 design)": forms["M2"], "Lasso (M2 design)": forms["M2"]}
    for name, f in specs.items():
        r, a, q = [], [], []
        for tr, te in kf.split(D):
            Dtr, Dte = D.iloc[tr], D.iloc[te]
            ytr, yte = Dtr[y].values, Dte[y].values
            if f is None:
                pred = np.full(len(yte), ytr.mean())
            else:
                yd, Xtr = patsy.dmatrices(f.format(y=y), Dtr, return_type="dataframe")
                Xte = patsy.build_design_matrices([Xtr.design_info], Dte, return_type="dataframe")[0]
                Xtr, Xte = Xtr.drop(columns="Intercept"), Xte.drop(columns="Intercept")
                if name.startswith("OLS"):
                    mdl = LinearRegression()
                elif name.startswith("Ridge"):
                    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30)))
                else:
                    mdl = make_pipeline(StandardScaler(), LassoCV(alphas=40, cv=5, random_state=SEED, max_iter=5000))
                mdl.fit(Xtr, ytr); pred = mdl.predict(Xte)
            r.append(np.sqrt(mean_squared_error(yte, pred))); a.append(mean_absolute_error(yte, pred)); q.append(r2_score(yte, pred))
        rows[name] = {"RMSE": np.mean(r), "RMSE_sd": np.std(r), "MAE": np.mean(a), "R2": np.mean(q), "R2_sd": np.std(q)}
    return rows

for tag, D, y in [("views", V, "log_views"), ("applies", A, "log_applies")]:
    res[f"cv_{tag}"] = cv_compare(D, y)

# full-data Lasso: which disclosure terms survive, and chosen alphas
def fit_reg(D, y):
    yd, X = patsy.dmatrices(forms["M2"].format(y=y), D, return_type="dataframe")
    X = X.drop(columns="Intercept"); yv = yd.values.ravel()
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    las = LassoCV(alphas=40, cv=5, random_state=SEED, max_iter=5000).fit(Xs, yv)
    rid = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(Xs, yv)
    coef = pd.Series(las.coef_, index=X.columns)
    dterms = coef[[c for c in X.columns if c.startswith("disclosed")]]
    return {"lasso_alpha": float(las.alpha_), "ridge_alpha": float(rid.alpha_),
            "n_features": X.shape[1], "n_zero": int((coef == 0).sum()),
            "disclosure_terms": dterms.round(4).to_dict(),
            "top_abs": coef.abs().sort_values(ascending=False).head(10).index.tolist()}, las, X.columns
for tag, D, y in [("views", V, "log_views"), ("applies", A, "log_applies")]:
    res[f"lasso_{tag}"], _, _ = fit_reg(D, y)

# sensitivity: keep "Not specified" level as its own category (views)
S = df.dropna(subset=["views"]).copy()
S["level"] = pd.Categorical(S["level"], categories=["Entry"] + [l for l in LEVELS if l != "Entry"] + ["Not specified"])
fs = smf.ols(forms["M1"].format(y="log_views"), data=S).fit(cov_type="cluster", cov_kwds={"groups": S["company_id"].fillna(-1).astype(int)})
res["sens_views_with_unspecified"] = {"b": float(fs.params["disclosed"]), "se": float(fs.bse["disclosed"]), "n": int(fs.nobs)}
# sensitivity: annual-paid postings only vs non-disclosed
ann = df[(df.pay_period.isna()) | (df.pay_period == "YEARLY")]
ann = ann[ann.level.isin(LEVELS)].dropna(subset=["views"]).copy()
ann["level"] = pd.Categorical(ann["level"], categories=["Entry"] + [l for l in LEVELS if l != "Entry"])
fa = smf.ols(forms["M1"].format(y="log_views"), data=ann).fit(cov_type="cluster", cov_kwds={"groups": ann["company_id"].fillna(-1).astype(int)})
res["sens_views_yearly_only"] = {"b": float(fa.params["disclosed"]), "se": float(fa.bse["disclosed"]), "n": int(fa.nobs)}

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight"})
C_ND, C_D = "#eb6834", "#2a78d6"   # not disclosed / disclosed
INK = "#0b0b0b"

# Fig 1: skew + log transform
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9))
for i, (col, lab) in enumerate([("views", "Views"), ("applies", "Applications")]):
    x = df[col].dropna()
    ax[2 * i].hist(x[x <= x.quantile(.95)], bins=np.arange(0.5, x.quantile(.95) + 1.5, 1), color="#2a78d6", edgecolor="white", linewidth=0.3)
    ax[2 * i].set_title(f"{lab} (raw, $\\leq$p95)", fontsize=8); ax[2 * i].set_yticks([])
    ax[2 * i + 1].hist(np.log1p(x), bins=30, color="#2a78d6", edgecolor="white", linewidth=0.3)
    ax[2 * i + 1].set_title(f"log(1+{lab.lower()})", fontsize=8); ax[2 * i + 1].set_yticks([])
fig.savefig(FIG / "fig1_distributions.pdf"); plt.close(fig)

# Fig 2: disclosure rate + raw median gap by level
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
x = np.arange(len(LEVELS))
ax[0].bar(x, desc["disc_rate"] * 100, color="#2a78d6", width=0.6)
for xi, v, n in zip(x, desc["disc_rate"] * 100, desc["n"]):
    ax[0].text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=7, color=INK)
ax[0].set_xticks(x, LEVELS, rotation=25, fontsize=7.5); ax[0].set_ylabel("% postings disclosing salary")
ax[0].set_ylim(0, 50); ax[0].set_title("(a) Salary disclosure rate", fontsize=9)
g = main.dropna(subset=["views"]).groupby(["level", "disclosed"], observed=True)["log_views"]
m = g.mean().unstack().reindex(LEVELS); se = (g.std() / np.sqrt(g.count())).unstack().reindex(LEVELS)
ax[1].errorbar(x - 0.1, m[0], yerr=1.96 * se[0], fmt="o", color=C_ND, ms=4.5, capsize=2, lw=1.2, label="Not disclosed")
ax[1].errorbar(x + 0.1, m[1], yerr=1.96 * se[1], fmt="s", color=C_D, ms=4.5, capsize=2, lw=1.2, label="Disclosed")
ax[1].set_xticks(x, LEVELS, rotation=25, fontsize=7.5); ax[1].set_ylabel("Mean log(1+views)")
ax[1].set_title("(b) Unadjusted engagement (95% CI)", fontsize=9); ax[1].legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig2_levels.pdf"); plt.close(fig)

# Fig 3: adjusted disclosure effect by level (M2) for views and applies
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for i, (tag, lab) in enumerate([("views", "Views"), ("applies", "Applications")]):
    L = pd.DataFrame(res[f"levels_{tag}"])
    yy = np.arange(len(L))[::-1]
    ax[i].axvline(0, color="#52514e", lw=0.8)
    ax[i].errorbar(L["pct"], yy, xerr=[L["pct"] - L["pct_lo"], L["pct_hi"] - L["pct"]],
                   fmt="o", color="#2a78d6", ms=4.5, capsize=2, lw=1.2)
    for yv, p in zip(yy, L["pct"]):
        ax[i].text(p + 2, yv + 0.22, f"{p:+.0f}%", ha="left", fontsize=7, color=INK)
    ax[i].set_yticks(yy, L["level"]); ax[i].set_xlabel("Adjusted % difference, disclosed vs. not")
    ax[i].set_title(f"({'ab'[i]}) {lab}", fontsize=9); ax[i].set_ylim(-0.6, len(L) - 0.3)
fig.tight_layout(); fig.savefig(FIG / "fig3_level_effects.pdf"); plt.close(fig)

# Fig 4: CV comparison
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2), sharey=True)
for i, (tag, lab) in enumerate([("views", "log(1+views)"), ("applies", "log(1+applications)")]):
    cv = pd.DataFrame(res[f"cv_{tag}"]).T
    yy = np.arange(len(cv))[::-1]
    ax[i].barh(yy, cv["R2"], xerr=cv["R2_sd"], color="#2a78d6", height=0.55, error_kw={"lw": 0.8, "capsize": 2})
    for yv, v in zip(yy, cv["R2"]):
        ax[i].text(max(v, 0) + cv["R2_sd"].max() + 0.01, yv, f"{abs(v) if abs(v) < 5e-4 else v:.3f}", va="center", fontsize=7, color=INK)
    ax[i].set_yticks(yy, cv.index, fontsize=7.5); ax[i].set_xlabel("5-fold CV $R^2$")
    ax[i].set_title(f"({'ab'[i]}) {lab}", fontsize=9); ax[i].set_xlim(0, max(cv["R2"]) * 1.25)
fig.tight_layout(); fig.savefig(FIG / "fig4_cv.pdf"); plt.close(fig)

json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
