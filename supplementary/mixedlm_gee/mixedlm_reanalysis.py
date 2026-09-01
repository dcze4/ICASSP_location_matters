"""
Mixed-effects analysis of voice/style disentanglement metrics.

1. Linear mixed-effects models (random intercept per prompt unit),
    which account for repeated measures of the same voice_id / style_id
    across model conditions and embedding models.
2. Holm-corrected pairwise model contrasts
3. A GEE robustness check with cluster-robust SEs.

Usage: edit PATHS below, then `python mixedlm_reanalysis.py`.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ----------------------------------------------------------------------
# PATHS - edit these
# ----------------------------------------------------------------------

# parent dir of <metric>/embedding_analysis_results_<metric>.pkl
DIR_PATH = "./templates_norm"
# the style_metric used when the pkl was created
DISTANCE_METRIC = "cosine"
DATASET_PATH = "./structured_prompts_all.csv"
OUT_DIR = os.path.join(DIR_PATH, DISTANCE_METRIC, "mixedlm")

ALPHA = 0.05


# ----------------------------------------------------------------------
# 1. Explode, keeping unit identity
# ----------------------------------------------------------------------
def explode_with_unit_ids(dir_path, metric, dataset_path):
    df = pd.read_pickle(
        os.path.join(dir_path, metric, f"embedding_analysis_results_{metric}.pkl")
    )
    data = pd.read_csv(dataset_path)
    # Order MUST match EmbeddingAnalyzer (it uses .unique() on the same CSV)
    voice_ids = data["voice_id"].unique().tolist()
    style_ids = data["style_id"].unique().tolist()

    style_cols = [
        "model",
        "voice_embedding_model",
        "emotion_embedding_model",
        "style_spreads",
        "voice_consistencies",
    ]
    voice_cols = [
        "model",
        "voice_embedding_model",
        "emotion_embedding_model",
        "voice_spreads",
        "style_consistencies",
    ]

    style_df = df[style_cols].copy()
    voice_df = df[voice_cols].copy()

    # sanity check: array lengths must match id list lengths
    assert all(len(x) == len(voice_ids) for x in style_df["style_spreads"]), (
        "style_spreads length != number of voice_ids -- check ID ordering"
    )
    assert all(len(x) == len(style_ids) for x in voice_df["voice_spreads"]), (
        "voice_spreads length != number of style_ids -- check ID ordering"
    )

    # style-disentanglement metrics are computed per voice_id;
    # voice-disentanglement metrics per style_id
    style_df["unit_id"] = [list(voice_ids)] * len(style_df)
    voice_df["unit_id"] = [list(style_ids)] * len(voice_df)

    style_df = style_df.explode(
        ["style_spreads", "voice_consistencies", "unit_id"]
    ).reset_index(drop=True)
    voice_df = voice_df.explode(
        ["voice_spreads", "style_consistencies", "unit_id"]
    ).reset_index(drop=True)

    for c in ("style_spreads", "voice_consistencies"):
        style_df[c] = style_df[c].astype(float)
    for c in ("voice_spreads", "style_consistencies"):
        voice_df[c] = voice_df[c].astype(float)

    # ratio metrics depend on BOTH embedding families -> keep full combo
    style_df["style_disentanglement"] = (
        style_df["style_spreads"] / style_df["voice_consistencies"]
    )
    voice_df["voice_disentanglement"] = (
        voice_df["voice_spreads"] / voice_df["style_consistencies"]
    )
    for d in (style_df, voice_df):
        d["embedding_combo"] = (
            d["voice_embedding_model"] + "-" + d["emotion_embedding_model"]
        )

    return style_df, voice_df


# ----------------------------------------------------------------------
# 2. Build one tidy frame per metric, mirroring the paper's convention:
#    fix the irrelevant embedding family to one level for base metrics;
#    keep the full 6-level combo for the ratio metrics.
# ----------------------------------------------------------------------
FIX_EMOTION = "audeering"  # level used when the metric ignores the style model
FIX_VOICE = "wavlm"  # level used when the metric ignores the voice model


def build_metric_frames(style_df, voice_df):
    """Returns dict: metric_name -> df with columns [model, embed, unit_id, value]."""
    vs = voice_df.loc[voice_df.emotion_embedding_model == FIX_EMOTION]
    vc = style_df.loc[style_df.emotion_embedding_model == FIX_EMOTION]
    ss = style_df.loc[style_df.voice_embedding_model == FIX_VOICE]
    sc = voice_df.loc[voice_df.voice_embedding_model == FIX_VOICE]

    spec = {
        # metric                  source     value col                embedding factor
        "style_spread": (ss, "style_spreads", "emotion_embedding_model"),
        "style_consistency": (sc, "style_consistencies", "emotion_embedding_model"),
        "style_disentanglement": (style_df, "style_disentanglement", "embedding_combo"),
        "voice_spread": (vs, "voice_spreads", "voice_embedding_model"),
        "voice_consistency": (vc, "voice_consistencies", "voice_embedding_model"),
        "voice_disentanglement": (voice_df, "voice_disentanglement", "embedding_combo"),
    }
    frames = {}
    for name, (src, col, embed_col) in spec.items():
        d = (
            src[["model", embed_col, "unit_id", col]]
            .rename(columns={embed_col: "embed", col: "value"})
            .reset_index(drop=True)
        )
        frames[name] = d
    return frames


# ----------------------------------------------------------------------
# 3. Mixed model per metric
#    Random structure: intercept per unit (voice/style prompt) PLUS a
#    variance component for unit x model -- the latter captures that the
#    same generated audio is measured by multiple embedding models, so
#    those observations share a wav-level idiosyncrasy. Omitting it
#    deflates p-values exactly as the reviewer described.
# ----------------------------------------------------------------------
WAV_VC = {"wav": "0 + C(model)"}  # random (unit x model) effects


def _fit_lmm(formula, d, reml=True):
    md = smf.mixedlm(
        formula, data=d, groups=d["unit_id"], re_formula="1", vc_formula=WAV_VC
    )
    try:
        return md.fit(reml=reml, method="lbfgs")
    except Exception:
        return md.fit(reml=reml, method="powell")


def fit_mixedlm(d, formula="value ~ C(model) * C(embed)"):
    return _fit_lmm(formula, d, reml=True)


def icc(mfit):
    """Variance decomposition: shares due to unit and wav (unit x model)."""
    var_unit = float(np.asarray(mfit.cov_re)[0, 0])
    var_wav = float(sum(mfit.vcomp)) if len(mfit.vcomp) else 0.0
    var_resid = float(mfit.scale)
    total = var_unit + var_wav + var_resid
    return {
        "unit": var_unit / total,
        "wav": var_wav / total,
        "resid": var_resid / total,
    }


def omnibus_tests(d):
    """LRT for the model factor (and interaction), fit by ML for valid LRTs."""
    full = _fit_lmm("value ~ C(model) * C(embed)", d, reml=False)
    no_int = _fit_lmm("value ~ C(model) + C(embed)", d, reml=False)
    no_model = _fit_lmm("value ~ C(embed)", d, reml=False)

    from scipy import stats

    def lrt(big, small, df_diff):
        stat = 2 * (big.llf - small.llf)
        return stat, stats.chi2.sf(stat, df_diff)

    k_model = d["model"].nunique() - 1
    k_embed = d["embed"].nunique() - 1
    lrt_int = lrt(full, no_int, k_model * k_embed)
    lrt_model = lrt(no_int, no_model, k_model)
    return {
        "interaction_chi2": lrt_int[0],
        "interaction_p": lrt_int[1],
        "model_chi2": lrt_model[0],
        "model_p": lrt_model[1],
    }


# ----------------------------------------------------------------------
# 4. Pairwise model contrasts (replacement for Tukey HSD)
#    Subset to each pair, fit MixedLM with additive embed adjustment,
#    Holm-correct across the 6 pairs.
# ----------------------------------------------------------------------
def pairwise_contrasts(d, alpha=ALPHA):
    models = sorted(d["model"].unique())
    rows = []
    for m1, m2 in combinations(models, 2):
        sub = d[d["model"].isin([m1, m2])].copy()
        fit = _fit_lmm("value ~ C(model) + C(embed)", sub, reml=True)
        coef = (
            f"C(model)[T.{m2}]"
            if f"C(model)[T.{m2}]" in fit.params.index
            else f"C(model)[T.{m1}]"
        )
        est = fit.params[coef]
        se = fit.bse[coef]
        p = fit.pvalues[coef]
        rows.append(
            {
                "group1": m1,
                "group2": m2,
                "meandiff": est,
                "se": se,
                "z": est / se,
                "p_raw": p,
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    out["reject"] = out["p_holm"] < alpha
    return out


# ----------------------------------------------------------------------
# 5. GEE robustness check (sandwich SEs, exchangeable working corr.)
# ----------------------------------------------------------------------
def fit_gee(d):
    gee = smf.gee(
        "value ~ C(model) * C(embed)",
        groups="unit_id",
        data=d,
        cov_struct=sm.cov_struct.Exchangeable(),
        family=sm.families.Gaussian(),
    )
    return gee.fit()


# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    style_df, voice_df = explode_with_unit_ids(DIR_PATH, DISTANCE_METRIC, DATASET_PATH)
    frames = build_metric_frames(style_df, voice_df)

    summary_rows = []
    for name, d in frames.items():
        print("=" * 80)
        print(
            f"METRIC: {name}   (n={len(d)}, units={d['unit_id'].nunique()}, "
            f"models={d['model'].nunique()}, embeds={d['embed'].nunique()})"
        )
        print("=" * 80)

        m = fit_mixedlm(d)
        print(m.summary())
        rho = icc(m)
        print(
            f"\nVariance shares -- unit: {rho['unit']:.3f}, wav (unit x model): {rho['wav']:.3f}, residual: {rho['resid']:.3f}"
        )

        om = omnibus_tests(d)
        print(
            f"LRT model effect:       chi2={om['model_chi2']:.2f}, p={om['model_p']:.4g}"
        )
        print(
            f"LRT model x embedding:  chi2={om['interaction_chi2']:.2f}, p={om['interaction_p']:.4g}"
        )

        pw = pairwise_contrasts(d)
        print("\nPairwise model contrasts (Holm-corrected):")
        print(pw.to_string(index=False, float_format=lambda x: f"{x: .4g}"))
        pw.to_csv(os.path.join(OUT_DIR, f"pairwise_{name}.csv"), index=False)

        g = fit_gee(d)
        print("\nGEE robustness check (model terms, robust SEs):")
        print(g.summary().tables[1])

        summary_rows.append(
            {
                "metric": name,
                "icc_unit": rho["unit"],
                "icc_wav": rho["wav"],
                "model_LRT_chi2": om["model_chi2"],
                "model_LRT_p": om["model_p"],
                "interaction_LRT_chi2": om["interaction_chi2"],
                "interaction_LRT_p": om["interaction_p"],
                "n_obs": len(d),
                "n_units": d["unit_id"].nunique(),
            }
        )
        print()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "mixedlm_summary.csv"), index=False)
    print("=" * 80)
    print("SUMMARY (saved to mixedlm_summary.csv)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
