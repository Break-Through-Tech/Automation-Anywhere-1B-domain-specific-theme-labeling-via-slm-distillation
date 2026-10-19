"""
phase1/evaluation/stats.py

Statistical tests for comparing 2 or more groups (models or hyper-parameter configs).

Our data is PAIRED: every model/config is scored on the same (cluster_id, prompt_id)
items, so we use paired tests (they remove item-to-item difficulty variance).

Cheat-sheet
-----------
2 groups, paired      : paired t-test (normal diffs) / Wilcoxon signed-rank (non-parametric)
2 groups, independent : Welch t-test / Mann-Whitney U
3+ groups, paired     : repeated-measures ANOVA / Friedman test  (+ pairwise Wilcoxon, Holm-corrected)
3+ groups, independent: one-way ANOVA / Kruskal-Wallis           (+ pairwise Mann-Whitney, Holm)

Effect sizes (p-values only say "is there a difference", not "how big"):
  Cohen's d_z (paired), rank-biserial / Cliff-style effect, and a bootstrap 95% CI
  of the mean difference.

With very few test clusters (3-5) p-values are low-powered: always read the effect
size and CI next to the p-value, and consider evaluation.eval_all_splits: true.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def holm_correct(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (controls family-wise error)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def bootstrap_ci(diff: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


# ── Two-group paired comparison ───────────────────────────────────────────────

def paired_compare(a, b, name_a: str = "A", name_b: str = "B") -> dict:
    """
    Compare two paired samples (same items, two conditions). Reports B - A.

    Returns dict with n, means, mean_diff, bootstrap CI, paired t-test,
    Wilcoxon signed-rank, Cohen's d_z and rank-biserial correlation.
    """
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    out = {"group_a": name_a, "group_b": name_b, "n_pairs": n,
           "mean_a": float(a.mean()) if n else float("nan"),
           "mean_b": float(b.mean()) if n else float("nan")}
    if n < 2:
        out.update(mean_diff=float("nan"), note="need >= 2 paired observations")
        return out

    diff = b - a
    ci_lo, ci_hi = bootstrap_ci(diff)
    sd = diff.std(ddof=1)
    out.update(
        mean_diff=float(diff.mean()),
        ci95_low=ci_lo, ci95_high=ci_hi,
        cohens_dz=float(diff.mean() / sd) if sd > 0 else float("nan"),
    )

    # Paired t-test (assumes differences ~ normal)
    if sd > 0:
        t = stats.ttest_rel(b, a)
        out.update(t_stat=float(t.statistic), t_pvalue=float(t.pvalue))
    else:
        out.update(t_stat=float("nan"), t_pvalue=float("nan"))

    # Wilcoxon signed-rank (no normality assumption); undefined if all diffs are 0
    if np.any(diff != 0):
        try:
            w = stats.wilcoxon(b, a, zero_method="wilcox")
            out.update(wilcoxon_stat=float(w.statistic), wilcoxon_pvalue=float(w.pvalue))
            # matched-pairs rank-biserial correlation (-1..1)
            nz = diff[diff != 0]
            ranks = stats.rankdata(np.abs(nz))
            pos, neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
            out["rank_biserial"] = float((pos - neg) / (pos + neg))
        except ValueError as e:
            out.update(wilcoxon_pvalue=float("nan"), note=str(e))
    else:
        out.update(wilcoxon_pvalue=1.0, rank_biserial=0.0)
    return out


# ── Multi-group paired comparison ─────────────────────────────────────────────

def multi_group_compare(wide: pd.DataFrame, higher_is_better: bool = True) -> dict:
    """
    Compare 3+ paired groups.

    wide : DataFrame, one row per item (cluster x prompt), one column per group.
           Rows with any NaN are dropped (paired design).

    Returns
    -------
    {"omnibus": {...Friedman + RM-ANOVA...}, "pairwise": DataFrame (Holm-corrected),
     "means": Series, "n_items": int}
    """
    from scipy import stats

    wide = wide.dropna(axis=0, how="any")
    groups = list(wide.columns)
    n_items = len(wide)
    result: dict = {"n_items": n_items, "means": wide.mean().sort_values(
        ascending=not higher_is_better)}

    if len(groups) < 3 or n_items < 3:
        result["omnibus"] = {"note": "need >= 3 groups and >= 3 paired items"}
    else:
        omnibus = {}
        try:
            f = stats.friedmanchisquare(*[wide[g].values for g in groups])
            k = len(groups)
            omnibus["friedman_chi2"] = float(f.statistic)
            omnibus["friedman_pvalue"] = float(f.pvalue)
            omnibus["kendalls_W"] = float(f.statistic / (n_items * (k - 1)))  # effect size 0..1
        except ValueError as e:
            omnibus["friedman_note"] = str(e)
        try:  # repeated-measures one-way ANOVA computed by hand
            X = wide.values
            k = X.shape[1]
            grand = X.mean()
            ss_cond = n_items * ((X.mean(axis=0) - grand) ** 2).sum()
            ss_subj = k * ((X.mean(axis=1) - grand) ** 2).sum()
            ss_total = ((X - grand) ** 2).sum()
            ss_err = ss_total - ss_cond - ss_subj
            df_c, df_e = k - 1, (k - 1) * (n_items - 1)
            if ss_err > 0:
                F = (ss_cond / df_c) / (ss_err / df_e)
                omnibus["rm_anova_F"] = float(F)
                omnibus["rm_anova_pvalue"] = float(stats.f.sf(F, df_c, df_e))
                omnibus["partial_eta_sq"] = float(ss_cond / (ss_cond + ss_err))
        except Exception as e:  # noqa: BLE001
            omnibus["rm_anova_note"] = str(e)
        result["omnibus"] = omnibus

    # Pairwise Wilcoxon with Holm correction
    rows = []
    for ga, gb in itertools.combinations(groups, 2):
        r = paired_compare(wide[ga], wide[gb], ga, gb)
        rows.append(r)
    pw = pd.DataFrame(rows)
    if not pw.empty and "wilcoxon_pvalue" in pw.columns:
        p = pw["wilcoxon_pvalue"].fillna(1.0).tolist()
        pw["wilcoxon_p_holm"] = holm_correct(p)
        pw["significant_0.05"] = pw["wilcoxon_p_holm"] < 0.05
    result["pairwise"] = pw
    return result


# ── Independent groups (e.g. latency of different runs/hardware) ──────────────

def independent_compare(groups: dict[str, np.ndarray]) -> dict:
    """Kruskal-Wallis + one-way ANOVA across independent samples, plus Mann-Whitney pairs."""
    from scipy import stats

    clean = {k: np.asarray(v, dtype=float)[~np.isnan(np.asarray(v, dtype=float))]
             for k, v in groups.items()}
    clean = {k: v for k, v in clean.items() if len(v) >= 2}
    out: dict = {"n": {k: len(v) for k, v in clean.items()}}
    if len(clean) < 2:
        out["note"] = "need >= 2 groups with >= 2 observations"
        return out

    vals = list(clean.values())
    if len(clean) >= 3:
        kw = stats.kruskal(*vals)
        out["kruskal_H"], out["kruskal_pvalue"] = float(kw.statistic), float(kw.pvalue)
        an = stats.f_oneway(*vals)
        out["anova_F"], out["anova_pvalue"] = float(an.statistic), float(an.pvalue)

    rows = []
    for (na, a), (nb, b) in itertools.combinations(clean.items(), 2):
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        wt = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"group_a": na, "group_b": nb, "mean_a": a.mean(), "mean_b": b.mean(),
                     "mannwhitney_p": float(mw.pvalue), "welch_t_p": float(wt.pvalue)})
    pw = pd.DataFrame(rows)
    pw["mannwhitney_p_holm"] = holm_correct(pw["mannwhitney_p"].tolist())
    out["pairwise"] = pw
    return out


def format_pvalue(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
