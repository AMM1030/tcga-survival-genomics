"""
TCGA Pan-Cancer Survival & Biomarker Explorer
----------------------------------------------
Gradio app demonstrating ML survival prediction across 4 cancer types,
with per-cancer top biomarkers and Kaplan-Meier curves.

Run locally:
    python app/gradio_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "tcga_features.csv"
MODEL_DIR = ROOT / "models"
BIOMARKER_CSV = ROOT / "data" / "processed" / "biomarker_summary.csv"

CANCERS = ["BRCA", "COAD", "HNSC", "LUAD"]

# Common driver genes shown as checkboxes (subset; each cancer model uses
# the ones that appeared frequently enough during training)
DISPLAY_GENES = [
    "TP53", "KRAS", "PIK3CA", "EGFR", "BRAF", "APC", "PTEN",
    "ARID1A", "CDKN2A", "NOTCH1", "STK11", "ATM", "RB1", "GATA3",
]


# -----------------------------------------------------------------------------
# Load once at startup
# -----------------------------------------------------------------------------
print("Loading data and models...")
DF = pd.read_csv(DATA_PATH)
MODELS = {}
for cancer in CANCERS:
    path = MODEL_DIR / f"{cancer}_best_survival_model.joblib"
    if path.exists():
        MODELS[cancer] = joblib.load(path)
        print(f"  Loaded {cancer}: {MODELS[cancer]['model_type']}")
BIOMARKERS = pd.read_csv(BIOMARKER_CSV) if BIOMARKER_CSV.exists() else pd.DataFrame()
print(f"  Loaded {len(DF):,} patients, {len(MODELS)} models")


# -----------------------------------------------------------------------------
# Core prediction
# -----------------------------------------------------------------------------
def predict_survival(cancer: str, age: int, total_mutations: int,
                     mutated_genes: List[str]):
    """Take user inputs, return risk score + KM plot + biomarker table."""
    if cancer not in MODELS:
        return "Model not loaded.", None, pd.DataFrame()

    model_info = MODELS[cancer]
    features = model_info["features"]
    model = model_info["model"]

    # Build feature vector matching the model's expected column order
    row = {f: 0 for f in features}
    if "AGE" in features:
        row["AGE"] = age
    if "total_mutations" in features:
        row["total_mutations"] = total_mutations
    for gene in mutated_genes:
        col = f"mut_{gene}"
        if col in features:
            row[col] = 1
    X = pd.DataFrame([row])[features]

    # Predict raw risk score (higher = worse prognosis)
    raw_risk = float(model.predict(X)[0])

    # Convert raw risk into population percentile by comparing against cohort
    sub = DF[DF["cancer_type"] == cancer].copy()
    if len(sub) > 0:
        # Build the same feature row for every patient in the cohort to get
        # baseline scores for percentile calculation
        # (Use the cohort's median risk as a comparison anchor for simplicity)
        cohort_X = pd.DataFrame(0, index=sub.index, columns=features)
        for f in features:
            if f == "AGE" and "AGE" in sub.columns:
                cohort_X[f] = sub["AGE"].fillna(sub["AGE"].median()).values
            elif f == "total_mutations" and "total_mutations" in sub.columns:
                cohort_X[f] = sub["total_mutations"].values
            elif f in sub.columns:
                cohort_X[f] = sub[f].values
        cohort_scores = model.predict(cohort_X)
        percentile = (cohort_scores < raw_risk).mean() * 100
        risk_band = ("Low" if percentile < 33
                     else "Medium" if percentile < 66
                     else "High")
    else:
        percentile = 50.0
        risk_band = "Unknown"

    risk_summary = (
        f"**Predicted risk score:** {raw_risk:.3f}\n\n"
        f"**Population percentile:** {percentile:.0f}th\n\n"
        f"**Risk band:** {risk_band}\n\n"
        f"_Higher score = higher predicted mortality risk. "
        f"Patient ranks higher than {percentile:.0f}% of TCGA {cancer} cohort._"
    )

    # Kaplan-Meier curve for this cancer (stratified by risk band of all cohort)
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = DF[DF["cancer_type"] == cancer]
    if len(sub) > 0:
        kmf = KaplanMeierFitter()
        kmf.fit(sub["OS_MONTHS"], sub["event"], label=f"All {cancer} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, color="black", ci_show=False)
        ax.set_title(f"{cancer} — Kaplan-Meier survival curve\n"
                     f"Your patient: {risk_band} risk band")
        ax.set_xlabel("Months")
        ax.set_ylabel("Survival probability")
        ax.grid(alpha=0.3)
        # Mark a colored vertical band corresponding to the risk band
        colors = {"Low": "green", "Medium": "orange", "High": "red", "Unknown": "gray"}
        ax.axhline(0.5, color=colors[risk_band], ls="--", alpha=0.5,
                   label=f"{risk_band} risk")
        ax.legend()
    plt.tight_layout()

    # Top biomarkers for this cancer
    if not BIOMARKERS.empty:
        bm = (BIOMARKERS[BIOMARKERS["cancer"] == cancer]
              .sort_values("importance_mean", ascending=False)
              .head(10)
              [["feature", "importance_mean", "HR", "p", "direction"]]
              .round(3))
    else:
        bm = pd.DataFrame()

    return risk_summary, fig, bm


# -----------------------------------------------------------------------------
# Build the UI
# -----------------------------------------------------------------------------
with gr.Blocks(title="TCGA Survival Explorer", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# TCGA Pan-Cancer Survival & Biomarker Explorer\n"
        "Predict survival risk for breast, colorectal, head & neck, and lung "
        "cancers using clinical + genomic features from TCGA. Models are "
        "Random Survival Forests with permutation-importance biomarker "
        "discovery. Built by Ammulakshmi M.S."
    )
    gr.Markdown(
        "_Synthetic input → real model trained on public TCGA data._"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Patient features")
            cancer_in = gr.Dropdown(CANCERS, value="BRCA", label="Cancer type")
            age_in = gr.Slider(18, 90, value=60, step=1, label="Age (years)")
            total_mut_in = gr.Slider(0, 30, value=2, step=1,
                                      label="Total mutations across driver genes")
            genes_in = gr.CheckboxGroup(
                DISPLAY_GENES, value=["TP53"], label="Mutated driver genes",
            )
            btn = gr.Button("Predict survival risk", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Prediction")
            risk_out = gr.Markdown()
            plot_out = gr.Plot()
            gr.Markdown("### Top biomarkers for this cancer")
            biomarker_out = gr.Dataframe(
                headers=["feature", "importance_mean", "HR", "p", "direction"],
                wrap=True,
            )

    btn.click(
        predict_survival,
        inputs=[cancer_in, age_in, total_mut_in, genes_in],
        outputs=[risk_out, plot_out, biomarker_out],
    )

    gr.Markdown(
        "---\n"
        "**Data source:** [cBioPortal TCGA Pan-Cancer Atlas 2018](https://www.cbioportal.org/) · "
        "**Models:** Random Survival Forest (scikit-survival) · "
        "**Repository:** [github.com/AMM1030/tcga-survival-genomics](https://github.com/AMM1030/tcga-survival-genomics)"
    )


if __name__ == "__main__":
    demo.launch()