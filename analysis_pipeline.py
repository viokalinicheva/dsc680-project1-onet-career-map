"""DSC 680 Project 1, Milestone 2: reproducible O*NET analysis.

The pipeline deliberately separates intake/validation, cleaning, analysis, and
communication. It uses O*NET 31.0 Importance (IM) ratings for four domains and
treats O*NET's Recommend Suppress flag as a reliability warning rather than as
missing data in the raw source.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import sklearn
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import StandardScaler


# %% 1. Configuration and provenance
PROJECT_ROOT = Path(os.environ.get("DSC680_PROJECT_ROOT", Path.cwd())).resolve()
DATA_DIR = Path(
    os.environ.get(
        "ONET_DATA_DIR",
        PROJECT_ROOT / "work/dsc680_project_screen/onet31/db_31_0_excel",
    )
)
ARCHIVE_PATH = DATA_DIR.parent / "db_31_0_excel.zip"
RUN_DIR = Path(os.environ.get("DSC680_RUN_DIR", PROJECT_ROOT / "work/dsc680_milestone2"))
RESULTS_DIR = RUN_DIR / "results"
FIGURES_DIR = RUN_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CODE = "15-2051.00"
TARGET_TITLE = "Data Scientists"
RANDOM_STATE = 42
TOP_K = 10
EXPECTED_ARCHIVE_SHA256 = "b8d9bc5bcf3ed90fcacf386ad40171a46d63cebb47acc250c2ae88a9f3448a23"

DOMAIN_FILES = {
    "Essential Skills": "Essential Skills.xlsx",
    "Transferable Skills": "Transferable Skills.xlsx",
    "Knowledge": "Knowledge.xlsx",
    "Work Activities": "Work Activities.xlsx",
}
DOMAIN_PREFIX = {
    "Essential Skills": "ES",
    "Transferable Skills": "TS",
    "Knowledge": "KN",
    "Work Activities": "WA",
}
EXPECTED_FEATURE_COUNTS = {
    "Essential Skills": 10,
    "Transferable Skills": 25,
    "Knowledge": 33,
    "Work Activities": 41,
}

COLORS = {
    "navy": "#1B365D",
    "blue": "#2F6B9A",
    "teal": "#159A9C",
    "gold": "#E6A700",
    "orange": "#D55E00",
    "purple": "#7A5195",
    "gray": "#667085",
    "light": "#E8EEF4",
    "red": "#B33A3A",
}

mpl.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
)
sns.set_theme(style="whitegrid", palette="colorblind")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# %% 2. Intake, schema checks, and raw-data audit
REQUIRED_COLUMNS = {
    "O*NET-SOC Code",
    "Title",
    "Element ID",
    "Element Name",
    "Scale ID",
    "Scale Name",
    "Data Value",
    "Recommend Suppress",
    "Date",
}

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"O*NET 31.0 Excel directory not found: {DATA_DIR}. "
        "Download db_31_0_excel.zip from the O*NET Resource Center."
    )

archive_hash = sha256(ARCHIVE_PATH) if ARCHIVE_PATH.exists() else None
if archive_hash and archive_hash != EXPECTED_ARCHIVE_SHA256:
    raise ValueError("The local O*NET archive hash does not match the audited input.")

scales = pd.read_excel(DATA_DIR / "Scales Reference.xlsx")
im_scale = scales.loc[scales["Scale ID"].eq("IM")].iloc[0]
im_min, im_max = float(im_scale["Minimum"]), float(im_scale["Maximum"])

raw_tables: dict[str, pd.DataFrame] = {}
audit_rows: list[dict] = []

for domain, filename in DOMAIN_FILES.items():
    raw = pd.read_excel(DATA_DIR / filename)
    missing_columns = REQUIRED_COLUMNS.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"{filename} is missing columns: {sorted(missing_columns)}")

    table = raw.loc[raw["Scale ID"].eq("IM")].copy()
    table["Recommend Suppress"] = table["Recommend Suppress"].fillna("N").astype(str)
    table["Domain"] = domain
    table["Feature"] = (
        DOMAIN_PREFIX[domain]
        + " | "
        + table["Element ID"].astype(str)
        + " | "
        + table["Element Name"].astype(str)
    )

    if table["Data Value"].isna().any():
        raise ValueError(f"{domain} has missing IM Data Value entries.")
    if not table["Data Value"].between(im_min, im_max).all():
        raise ValueError(f"{domain} contains IM values outside [{im_min}, {im_max}].")
    duplicate_keys = int(table.duplicated(["O*NET-SOC Code", "Element ID"]).sum())
    if duplicate_keys:
        raise ValueError(f"{domain} has {duplicate_keys} duplicate occupation-element keys.")
    feature_count = int(table["Element ID"].nunique())
    if feature_count != EXPECTED_FEATURE_COUNTS[domain]:
        raise ValueError(
            f"{domain} has {feature_count} IM features; expected {EXPECTED_FEATURE_COUNTS[domain]}."
        )

    title_count = table.groupby("O*NET-SOC Code")["Title"].nunique().max()
    if title_count != 1:
        raise ValueError(f"{domain} has inconsistent titles within an occupation code.")

    raw_tables[domain] = table
    audit_rows.append(
        {
            "domain": domain,
            "raw_rows": len(raw),
            "im_rows": len(table),
            "occupations": table["O*NET-SOC Code"].nunique(),
            "features": feature_count,
            "missing_values": int(table["Data Value"].isna().sum()),
            "duplicate_keys": duplicate_keys,
            "suppressed_values": int(table["Recommend Suppress"].eq("Y").sum()),
            "affected_occupations": int(
                table.loc[table["Recommend Suppress"].eq("Y"), "O*NET-SOC Code"].nunique()
            ),
            "distinct_profile_dates": int(table["Date"].nunique()),
        }
    )

audit = pd.DataFrame(audit_rows)
audit.to_csv(RESULTS_DIR / "table_01_raw_data_audit.csv", index=False)


# %% 3. Cleaning rules and two analysis matrices
common_codes = set.intersection(
    *(set(table["O*NET-SOC Code"].unique()) for table in raw_tables.values())
)
if TARGET_CODE not in common_codes:
    raise ValueError(f"Target occupation {TARGET_CODE} is absent from the four-domain intersection.")

long = pd.concat(raw_tables.values(), ignore_index=True)
long = long.loc[long["O*NET-SOC Code"].isin(common_codes)].copy()

title_map = (
    long[["O*NET-SOC Code", "Title"]]
    .drop_duplicates()
    .set_index("O*NET-SOC Code")["Title"]
    .sort_index()
)
if title_map.index.duplicated().any():
    raise ValueError("An occupation code maps to multiple titles across domain tables.")
if title_map.loc[TARGET_CODE] != TARGET_TITLE:
    raise ValueError("The target O*NET-SOC code does not map to Data Scientists.")

feature_meta = (
    long[["Feature", "Domain", "Element ID", "Element Name"]]
    .drop_duplicates()
    .set_index("Feature")
)
feature_order = list(feature_meta.sort_values(["Domain", "Element ID"]).index)

full_matrix = long.pivot(
    index="O*NET-SOC Code", columns="Feature", values="Data Value"
).reindex(columns=feature_order)
flag_matrix = (
    long.assign(flag=long["Recommend Suppress"].eq("Y"))
    .pivot(index="O*NET-SOC Code", columns="Feature", values="flag")
    .reindex(index=full_matrix.index, columns=full_matrix.columns, fill_value=False)
    .astype(bool)
)

if full_matrix.shape[1] != sum(EXPECTED_FEATURE_COUNTS.values()):
    raise ValueError("The combined feature matrix does not contain 109 features.")
if full_matrix.isna().any().any():
    raise ValueError("The four-domain intersection unexpectedly contains missing feature values.")

# Primary conservative rule: convert recommended-suppression values to missing,
# then use complete occupations. Full matrix is retained for sensitivity analysis.
conservative_with_na = full_matrix.mask(flag_matrix)
conservative_matrix = conservative_with_na.dropna(axis=0, how="any")
excluded_codes = conservative_with_na.index[conservative_with_na.isna().any(axis=1)]

if TARGET_CODE not in conservative_matrix.index:
    raise ValueError("Data Scientists are excluded by the conservative suppression rule.")

cleaning_summary = {
    "source_release": "O*NET 31.0 (August 2026)",
    "archive_sha256": archive_hash,
    "common_occupations": int(len(common_codes)),
    "features": int(full_matrix.shape[1]),
    "raw_missing_values": int(full_matrix.isna().sum().sum()),
    "raw_duplicate_occupation_feature_keys": int(
        long.duplicated(["O*NET-SOC Code", "Feature"]).sum()
    ),
    "recommended_suppression_values": int(flag_matrix.sum().sum()),
    "occupations_with_any_suppression": int(flag_matrix.any(axis=1).sum()),
    "primary_complete_occupations": int(conservative_matrix.shape[0]),
    "full_sensitivity_occupations": int(full_matrix.shape[0]),
}

pd.DataFrame(
    {
        "O*NET-SOC Code": excluded_codes,
        "Title": title_map.reindex(excluded_codes).values,
        "Suppressed feature count": flag_matrix.loc[excluded_codes].sum(axis=1).values,
    }
).sort_values("Suppressed feature count", ascending=False).to_csv(
    RESULTS_DIR / "table_02_excluded_occupations.csv", index=False
)


# %% 4. Standardization and PCA
scaler = StandardScaler()
X = pd.DataFrame(
    scaler.fit_transform(conservative_matrix),
    index=conservative_matrix.index,
    columns=conservative_matrix.columns,
)

standardization_checks = {
    "maximum_absolute_column_mean": float(X.mean().abs().max()),
    "minimum_population_sd": float(X.std(ddof=0).min()),
    "maximum_population_sd": float(X.std(ddof=0).max()),
    "zero_variance_features": int((conservative_matrix.std(ddof=0) == 0).sum()),
    "finite_values": bool(np.isfinite(X.to_numpy()).all()),
}
if standardization_checks["zero_variance_features"]:
    raise ValueError("Zero-variance features remain after cleaning.")

pca = PCA()
scores_array = pca.fit_transform(X)
pc_names = [f"PC{i + 1}" for i in range(pca.n_components_)]
pca_scores = pd.DataFrame(scores_array, index=X.index, columns=pc_names)
pca_loadings = pd.DataFrame(
    pca.components_.T * np.sqrt(pca.explained_variance_),
    index=X.columns,
    columns=pc_names,
)
cum_variance = np.cumsum(pca.explained_variance_ratio_)
n_pc_80 = int(np.searchsorted(cum_variance, 0.80) + 1)
n_pc_90 = int(np.searchsorted(cum_variance, 0.90) + 1)

pca_variance = pd.DataFrame(
    {
        "component": pc_names,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": cum_variance,
    }
)
pca_variance.to_csv(RESULTS_DIR / "table_03_pca_variance.csv", index=False)


# %% 5. Nearest occupations and prespecified sensitivity analyses
def rank_distances(matrix: pd.DataFrame, metric: str = "euclidean") -> pd.DataFrame:
    target_position = matrix.index.get_loc(TARGET_CODE)
    if metric == "euclidean":
        values = euclidean_distances(matrix.iloc[[target_position]], matrix).ravel()
    elif metric == "cosine":
        values = cosine_distances(matrix.iloc[[target_position]], matrix).ravel()
    else:
        raise ValueError(metric)
    ranked = pd.DataFrame({"distance": values}, index=matrix.index)
    ranked = ranked.drop(index=TARGET_CODE).sort_values(["distance"], kind="stable")
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["Title"] = title_map.reindex(ranked.index)
    ranked.index.name = "O*NET-SOC Code"
    return ranked[["rank", "Title", "distance"]]


rank_primary = rank_distances(X, "euclidean")
rank_cosine = rank_distances(X, "cosine")

# Equal-domain weighting: each standardized domain contributes equally to the
# squared Euclidean distance, irrespective of its feature count.
X_equal_domain = X.copy()
for domain, count in EXPECTED_FEATURE_COUNTS.items():
    domain_cols = feature_meta.index[feature_meta["Domain"].eq(domain)]
    X_equal_domain.loc[:, domain_cols] /= math.sqrt(count)
rank_equal_domain = rank_distances(X_equal_domain, "euclidean")

domain_ranks: dict[str, pd.DataFrame] = {}
for domain in DOMAIN_FILES:
    cols = feature_meta.index[feature_meta["Domain"].eq(domain)]
    domain_ranks[domain] = rank_distances(X.loc[:, cols], "euclidean")

full_scaler = StandardScaler()
X_full = pd.DataFrame(
    full_scaler.fit_transform(full_matrix), index=full_matrix.index, columns=full_matrix.columns
)
rank_full = rank_distances(X_full, "euclidean")

rank_table = rank_primary.rename(columns={"rank": "primary_rank", "distance": "primary_distance"})
for name, table in {
    "cosine": rank_cosine,
    "equal_domain": rank_equal_domain,
    "full_sensitivity": rank_full,
    **{re.sub(r"\W+", "_", domain.lower()): value for domain, value in domain_ranks.items()},
}.items():
    rank_table = rank_table.join(
        table[["rank", "distance"]].rename(
            columns={"rank": f"{name}_rank", "distance": f"{name}_distance"}
        ),
        how="left",
    )
rank_table.reset_index().to_csv(RESULTS_DIR / "table_04_all_similarity_ranks.csv", index=False)


def compare_rankings(a: pd.DataFrame, b: pd.DataFrame, label: str) -> dict:
    common = a.index.intersection(b.index)
    a10, b10 = set(a.head(TOP_K).index), set(b.head(TOP_K).index)
    return {
        "comparison": label,
        "common_occupations": int(len(common)),
        "top10_overlap_count": int(len(a10 & b10)),
        "top10_jaccard": float(len(a10 & b10) / len(a10 | b10)),
        "spearman_rank_correlation": float(
            spearmanr(a.loc[common, "rank"], b.loc[common, "rank"]).statistic
        ),
        "kendall_tau": float(kendalltau(a.loc[common, "rank"], b.loc[common, "rank"]).statistic),
    }


ranking_comparisons = pd.DataFrame(
    [
        compare_rankings(rank_primary, rank_cosine, "Cosine distance"),
        compare_rankings(rank_primary, rank_equal_domain, "Equal domain weighting"),
        compare_rankings(rank_primary, rank_full, "Full-data suppression sensitivity"),
        *[
            compare_rankings(rank_primary, table, f"{domain} only")
            for domain, table in domain_ranks.items()
        ],
    ]
)
ranking_comparisons.to_csv(RESULTS_DIR / "table_05_ranking_robustness.csv", index=False)


# %% 6. Feature gaps and domain contributions
top_neighbors = rank_primary.head(TOP_K).index
target_vector = X.loc[TARGET_CODE]
gap_rows: list[dict] = []
domain_contribution_rows: list[dict] = []

for code in top_neighbors:
    delta = X.loc[code] - target_vector
    squared = delta.pow(2)
    for feature in X.columns:
        gap_rows.append(
            {
                "O*NET-SOC Code": code,
                "Title": title_map.loc[code],
                "Feature": feature,
                "Domain": feature_meta.loc[feature, "Domain"],
                "Element Name": feature_meta.loc[feature, "Element Name"],
                "neighbor_minus_data_scientists_z": float(delta[feature]),
                "absolute_gap_z": float(abs(delta[feature])),
                "squared_distance_contribution": float(squared[feature]),
            }
        )
    totals = squared.groupby(feature_meta["Domain"]).sum()
    for domain, value in totals.items():
        domain_contribution_rows.append(
            {
                "O*NET-SOC Code": code,
                "Title": title_map.loc[code],
                "Domain": domain,
                "squared_distance_contribution": float(value),
                "share_of_squared_distance": float(value / squared.sum()),
            }
        )

feature_gaps = pd.DataFrame(gap_rows)
domain_contributions = pd.DataFrame(domain_contribution_rows)
feature_gaps.to_csv(RESULTS_DIR / "table_06_neighbor_feature_gaps.csv", index=False)
domain_contributions.to_csv(RESULTS_DIR / "table_07_domain_contributions.csv", index=False)

mean_signed_gap = (
    feature_gaps.groupby(["Feature", "Domain", "Element Name"], as_index=False)
    .agg(
        mean_neighbor_minus_ds_z=("neighbor_minus_data_scientists_z", "mean"),
        mean_absolute_gap_z=("absolute_gap_z", "mean"),
    )
    .sort_values("mean_absolute_gap_z", ascending=False)
)
mean_signed_gap.to_csv(RESULTS_DIR / "table_08_mean_feature_gaps.csv", index=False)


# %% 7. Clustering selection, stability, and hierarchical comparison
cluster_rows: list[dict] = []
cluster_models: dict[int, KMeans] = {}

for k in range(2, 13):
    model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE, algorithm="lloyd")
    labels = model.fit_predict(X)
    seed_labels = [
        KMeans(n_clusters=k, n_init=1, random_state=seed, algorithm="lloyd").fit_predict(X)
        for seed in range(20)
    ]
    pairwise_ari = [
        adjusted_rand_score(seed_labels[i], seed_labels[j])
        for i, j in combinations(range(len(seed_labels)), 2)
    ]
    hierarchical_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    cluster_rows.append(
        {
            "k": k,
            "kmeans_silhouette": float(silhouette_score(X, labels)),
            "kmeans_calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "kmeans_davies_bouldin": float(davies_bouldin_score(X, labels)),
            "mean_seed_pairwise_ari": float(np.mean(pairwise_ari)),
            "minimum_seed_pairwise_ari": float(np.min(pairwise_ari)),
            "hierarchical_silhouette": float(silhouette_score(X, hierarchical_labels)),
            "kmeans_hierarchical_ari": float(adjusted_rand_score(labels, hierarchical_labels)),
        }
    )
    cluster_models[k] = model

cluster_validation = pd.DataFrame(cluster_rows)
# Prespecified selection: maximize silhouette among solutions with mean seed ARI >= .80;
# if none meet the threshold, maximize silhouette and report instability transparently.
stable_candidates = cluster_validation.loc[cluster_validation["mean_seed_pairwise_ari"] >= 0.80]
selection_pool = stable_candidates if not stable_candidates.empty else cluster_validation
selected_k = int(selection_pool.sort_values(["kmeans_silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
selected_model = cluster_models[selected_k]
selected_labels = pd.Series(selected_model.labels_, index=X.index, name="cluster")
target_cluster = int(selected_labels.loc[TARGET_CODE])

cluster_membership = pd.DataFrame(
    {
        "O*NET-SOC Code": X.index,
        "Title": title_map.reindex(X.index).values,
        "cluster": selected_labels.values,
        "is_data_scientists_cluster": selected_labels.values == target_cluster,
        "primary_distance_from_data_scientists": [
            0.0 if code == TARGET_CODE else float(rank_primary.loc[code, "distance"])
            for code in X.index
        ],
    }
).sort_values(["cluster", "primary_distance_from_data_scientists"])
cluster_membership.to_csv(RESULTS_DIR / "table_09_cluster_membership.csv", index=False)
cluster_validation.to_csv(RESULTS_DIR / "table_10_cluster_validation.csv", index=False)

cluster_profile = (
    pd.concat([X, selected_labels], axis=1)
    .groupby("cluster")
    .mean()
    .T
)
target_cluster_profile = cluster_profile[target_cluster].sort_values(key=abs, ascending=False)
target_cluster_profile.rename("target_cluster_mean_z").to_csv(
    RESULTS_DIR / "table_11_target_cluster_profile.csv"
)


# %% 8. O*NET software skills and transparent roadmap grouping
software = pd.read_excel(DATA_DIR / "Software Skills.xlsx")
required_software_columns = {
    "O*NET-SOC Code",
    "Title",
    "Workplace Example",
    "Element Name",
    "Hot Technology",
    "In Demand",
}
if required_software_columns.difference(software.columns):
    raise ValueError("Software Skills.xlsx does not match the expected schema.")

ds_software = software.loc[software["O*NET-SOC Code"].eq(TARGET_CODE)].copy()
ds_software["Hot Technology"] = ds_software["Hot Technology"].fillna("N").astype(str)
ds_software["In Demand"] = ds_software["In Demand"].fillna("N").astype(str)
if ds_software["Workplace Example"].duplicated().any():
    raise ValueError("Duplicate Data Scientists software examples were found.")

# These categories are an analyst-created communication layer. They do not imply
# that O*NET specifies a learning order. Every rule is exposed here for audit.
def roadmap_group(row: pd.Series) -> str:
    name = str(row["Element Name"]).lower()
    tool = str(row["Workplace Example"]).lower()
    # Explicit tool-family rules come first because O*NET's software taxonomy can
    # place a modern library under a broader legacy software category.
    if any(token in tool for token in ["tensorflow", "pytorch", "scikit", "numpy", "pandas", "sas", "spss", "matlab", "minitab", "rapidminer"]):
        return "Model & analyze"
    if any(token in tool for token in ["amazon web services", "aws", "microsoft azure", "apache hadoop", "apache spark", "apache kafka", "docker", "kubernetes", "snowflake"]):
        return "Scale & deploy"
    if any(token in tool for token in ["tableau", "power bi", "microsoft excel", "powerpoint", "qlik", "looker", "plotly", "d3.js"]):
        return "Communicate & visualize"
    if any(token in name for token in ["business intelligence", "reporting", "spreadsheet", "presentation", "office suite", "geographic"]):
        return "Communicate & visualize"
    if any(token in name for token in ["cloud", "application server", "operating system", "enterprise", "storage networking", "procedure management", "industrial control"]):
        return "Scale & deploy"
    if any(token in name for token in ["data base", "database", "query", "data mining"]):
        return "Store, query & engineer"
    if any(token in tool for token in ["r software"]):
        return "Model & analyze"
    return "Build & collaborate"


ds_software["Roadmap Group"] = ds_software.apply(roadmap_group, axis=1)
roadmap_order = [
    "Build & collaborate",
    "Store, query & engineer",
    "Model & analyze",
    "Communicate & visualize",
    "Scale & deploy",
]
software_summary = (
    ds_software.groupby("Roadmap Group")
    .agg(
        listed_tools=("Workplace Example", "nunique"),
        hot_tools=("Hot Technology", lambda s: int(s.eq("Y").sum())),
        in_demand_tools=("In Demand", lambda s: int(s.eq("Y").sum())),
    )
    .reindex(roadmap_order)
    .fillna(0)
    .astype(int)
    .reset_index()
)
ds_software.sort_values(["Roadmap Group", "In Demand", "Hot Technology", "Workplace Example"], ascending=[True, False, False, True]).to_csv(
    RESULTS_DIR / "table_12_data_scientist_software_skills.csv", index=False
)
software_summary.to_csv(RESULTS_DIR / "table_13_software_roadmap_summary.csv", index=False)


# %% 9. Publication-quality figures
# Figure 1: accountable sample flow and suppression distribution.
fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [1.05, 1.45]})
flow_labels = ["O*NET occupations\n(occupation file)", "Four-domain\nintersection", "Primary complete\nanalysis set"]
flow_values = [1016, len(common_codes), len(conservative_matrix)]
axes[0].barh(flow_labels[::-1], flow_values[::-1], color=[COLORS["teal"], COLORS["blue"], COLORS["navy"]])
for i, value in enumerate(flow_values[::-1]):
    axes[0].text(value + 10, i, f"{value:,}", va="center", fontweight="bold")
axes[0].set_xlim(0, max(flow_values) * 1.16)
axes[0].set_xlabel("Number of occupations")
axes[0].set_title("A. Sample construction")

supp = audit.set_index("domain")["suppressed_values"].reindex(DOMAIN_FILES)
bars = axes[1].bar(supp.index, supp.values, color=[COLORS["teal"], COLORS["blue"], COLORS["gold"], COLORS["purple"]])
for bar, value in zip(bars, supp.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value}", ha="center", fontweight="bold")
axes[1].set_ylabel("Values marked ‘Recommend Suppress’")
axes[1].set_title("B. Reliability flags are concentrated in Knowledge")
axes[1].tick_params(axis="x", rotation=18)
fig.suptitle("Figure 1. Cleaning changes the sample, not the measurements", x=0.04, ha="left", fontsize=16, fontweight="bold")
fig.text(0.04, 0.01, "Note. O*NET 31.0 IM ratings. The primary analysis excludes any occupation with ≥1 recommended-suppression value; the full 910-occupation matrix is retained for sensitivity analysis.", fontsize=8.5)
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
save_figure(fig, "figure_01_data_quality_and_sample_flow.png")

# Figure 2: PCA map with primary nearest neighbors and target cluster.
plot_codes = set(rank_primary.head(12).index) | {TARGET_CODE}
plot_frame = pca_scores[["PC1", "PC2"]].copy()
plot_frame["cluster"] = selected_labels
fig, ax = plt.subplots(figsize=(11.5, 7.2))
ax.scatter(plot_frame["PC1"], plot_frame["PC2"], s=20, c=COLORS["light"], edgecolors="none", alpha=0.72, label="Other occupations")
same_cluster = plot_frame.index[selected_labels.eq(target_cluster)]
ax.scatter(plot_frame.loc[same_cluster, "PC1"], plot_frame.loc[same_cluster, "PC2"], s=28, c=COLORS["teal"], alpha=0.55, edgecolors="white", linewidth=0.3, label=f"Data Scientists cluster (k={selected_k})")
label_texts = []
for code in plot_codes:
    is_target = code == TARGET_CODE
    ax.scatter(plot_frame.loc[code, "PC1"], plot_frame.loc[code, "PC2"], s=125 if is_target else 58, marker="*" if is_target else "o", c=COLORS["orange"] if is_target else COLORS["navy"], edgecolors="white", linewidth=0.8, zorder=5)
    label = title_map.loc[code]
    label_texts.append(
        ax.text(
            plot_frame.loc[code, "PC1"],
            plot_frame.loc[code, "PC2"],
            label,
            fontsize=7.8,
            fontweight="bold" if is_target else "normal",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
    )
adjust_text(
    label_texts,
    ax=ax,
    arrowprops={"arrowstyle": "-", "color": "#64748B", "lw": 0.45},
    expand=(1.08, 1.18),
    force_text=(0.4, 0.7),
    max_move=(80, 80),
)
ax.axhline(0, color="#CBD5E1", lw=0.7)
ax.axvline(0, color="#CBD5E1", lw=0.7)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)")
ax.set_title("Figure 2. Data Scientists occupy a neighborhood, not an isolated job title", loc="left", pad=14)
ax.legend(loc="best")
fig.text(0.08, 0.015, f"Note. PCA uses {X.shape[1]} standardized O*NET IM features for {X.shape[0]} complete occupations. Labels show Data Scientists and the 12 nearest standardized-Euclidean neighbors; the two axes preserve only {pca.explained_variance_ratio_[:2].sum():.1%} of total variance.", fontsize=8.5)
fig.tight_layout(rect=[0, 0.055, 1, 1])
save_figure(fig, "figure_02_pca_career_map.png")

# Figure 3: nearest-neighbor distances and rank stability across prespecified models.
neighbors = rank_primary.head(TOP_K).copy().sort_values("distance", ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={"width_ratios": [1.0, 1.25]})
axes[0].barh(neighbors["Title"], neighbors["distance"], color=COLORS["blue"])
axes[0].set_xlabel("Standardized Euclidean distance (lower = closer)")
axes[0].set_title("A. Closest occupations in the primary model")
axes[0].invert_yaxis()

specs = {
    "Primary": rank_primary,
    "Cosine": rank_cosine,
    "Equal-domain": rank_equal_domain,
    "Full-data": rank_full,
}
heat_codes = list(rank_primary.head(TOP_K).index)
heat = pd.DataFrame(
    {
        label: table.reindex(heat_codes)["rank"].to_numpy()
        for label, table in specs.items()
    },
    index=[title_map.loc[c] for c in heat_codes],
)
sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlGnBu_r", cbar_kws={"label": "Rank (lower = closer)"}, linewidths=0.5, linecolor="white", ax=axes[1])
axes[1].set_title("B. Primary neighbors remain visible across specifications")
axes[1].set_xlabel("Model specification")
axes[1].set_ylabel("")
fig.suptitle("Figure 3. Similarity is strongest when it survives reasonable analytical choices", x=0.04, ha="left", fontsize=16, fontweight="bold")
fig.text(0.04, 0.01, "Note. Equal-domain weighting prevents the 41 Work Activities fields from receiving more aggregate weight solely because that domain has more variables. Full-data retains all 910 occupations and treats flagged values as usable sensitivity inputs.", fontsize=8.5)
fig.tight_layout(rect=[0, 0.055, 1, 0.94])
save_figure(fig, "figure_03_nearest_occupations_and_stability.png")

# Figure 4: clustering evidence and the target cluster's strongest features.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
ax1 = axes[0]
ax1.plot(cluster_validation["k"], cluster_validation["kmeans_silhouette"], marker="o", color=COLORS["navy"], label="Silhouette")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Silhouette score", color=COLORS["navy"])
ax1.tick_params(axis="y", labelcolor=COLORS["navy"])
ax1.set_xticks(cluster_validation["k"])
ax1.axvline(selected_k, color=COLORS["orange"], ls="--", lw=1.5, label=f"Selected k={selected_k}")
ax1b = ax1.twinx()
ax1b.plot(cluster_validation["k"], cluster_validation["mean_seed_pairwise_ari"], marker="s", color=COLORS["teal"], label="Mean seed ARI")
ax1b.set_ylabel("Mean pairwise adjusted Rand index", color=COLORS["teal"])
ax1b.tick_params(axis="y", labelcolor=COLORS["teal"])
ax1.set_title("A. Separation and repeatability guide k")
handles = [
    Line2D([0], [0], color=COLORS["navy"], marker="o", label="Silhouette"),
    Line2D([0], [0], color=COLORS["teal"], marker="s", label="Mean seed ARI"),
    Line2D([0], [0], color=COLORS["orange"], ls="--", label=f"Selected k={selected_k}"),
]
ax1.legend(handles=handles, loc="best")

profile_plot = target_cluster_profile.head(10).sort_values()
profile_labels = [feature_meta.loc[f, "Element Name"] for f in profile_plot.index]
colors = [COLORS["orange"] if v < 0 else COLORS["teal"] for v in profile_plot.values]
axes[1].barh(profile_labels, profile_plot.values, color=colors)
axes[1].axvline(0, color="#475569", lw=0.8)
axes[1].set_xlabel("Mean standardized rating in Data Scientists cluster")
axes[1].set_title("B. Features that most define the target cluster")
fig.suptitle("Figure 4. The career map is useful only when its clusters are stable and interpretable", x=0.04, ha="left", fontsize=16, fontweight="bold")
fig.text(0.04, 0.01, "Note. K-means uses 50 initializations for the selected solution. Stability is the mean adjusted Rand index across every pair of 20 single-start seeds. Ward hierarchical clustering is reported separately in the supporting tables.", fontsize=8.5)
fig.tight_layout(rect=[0, 0.055, 1, 0.94])
save_figure(fig, "figure_04_cluster_validation_and_profile.png")

# Figure 5: evidence-based software roadmap.
fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), gridspec_kw={"width_ratios": [0.9, 1.35]})
plot_summary = software_summary.set_index("Roadmap Group").reindex(roadmap_order)
plot_summary[["hot_tools", "in_demand_tools"]].plot.barh(ax=axes[0], color=[COLORS["gold"], COLORS["teal"]], width=0.75)
axes[0].set_xlabel("Number of listed Data Scientists tools")
axes[0].set_ylabel("")
axes[0].set_title("A. Signals by analyst-created roadmap group")
axes[0].legend(["Hot technology", "In demand"])
axes[0].invert_yaxis()

priority_tools = ds_software.loc[
    ds_software["Hot Technology"].eq("Y") | ds_software["In Demand"].eq("Y")
].copy()
priority_tools["signal_score"] = priority_tools["Hot Technology"].eq("Y").astype(int) + 2 * priority_tools["In Demand"].eq("Y").astype(int)
priority_tools = priority_tools.sort_values(["signal_score", "Roadmap Group", "Workplace Example"], ascending=[False, True, True]).head(22)
priority_tools["label"] = priority_tools["Workplace Example"].str.replace(r"^.*?\bsoftware\b", "", regex=True).str.strip()
priority_tools.loc[priority_tools["label"].eq(""), "label"] = priority_tools["Workplace Example"]
priority_tools = priority_tools.sort_values(["Roadmap Group", "signal_score", "label"])
y = np.arange(len(priority_tools))
axes[1].scatter(priority_tools["signal_score"], y, s=65, c=priority_tools["Roadmap Group"].map(dict(zip(roadmap_order, [COLORS["navy"], COLORS["blue"], COLORS["purple"], COLORS["gold"], COLORS["teal"]]))), edgecolors="white", linewidth=0.6)
axes[1].set_yticks(y, priority_tools["label"])
axes[1].set_xticks([1, 2, 3], ["Hot only", "In demand only", "Hot + in demand"])
axes[1].set_xlim(0.6, 3.4)
axes[1].set_title("B. Priority signals in the O*NET software list")
axes[1].grid(axis="x", alpha=0.2)
fig.suptitle("Figure 5. The occupation profile points to a layered software-learning roadmap", x=0.04, ha="left", fontsize=16, fontweight="bold")
fig.text(0.04, 0.01, "Note. O*NET flags are descriptive labor-market signals, not causal proof or a prescribed curriculum. Roadmap groups are transparent analyst-created categories; tool flags come directly from O*NET 31.0 Software Skills.", fontsize=8.5)
fig.tight_layout(rect=[0, 0.055, 1, 0.94])
save_figure(fig, "figure_05_software_learning_roadmap.png")


# %% 10. Machine-readable summary and final assertions
top_gap_up = mean_signed_gap.sort_values("mean_neighbor_minus_ds_z", ascending=False).head(10)
top_gap_down = mean_signed_gap.sort_values("mean_neighbor_minus_ds_z", ascending=True).head(10)
selected_validation = cluster_validation.set_index("k").loc[selected_k]

summary = {
    "provenance": {
        "release": "O*NET 31.0",
        "release_month": "August 2026",
        "archive_sha256": archive_hash,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": mpl.__version__,
        "seaborn": sns.__version__,
    },
    "cleaning": cleaning_summary,
    "standardization_checks": standardization_checks,
    "pca": {
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]),
        "first_two_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "components_for_80_percent": n_pc_80,
        "components_for_90_percent": n_pc_90,
    },
    "primary_top_10": [
        {
            "rank": int(row["rank"]),
            "code": code,
            "title": row["Title"],
            "distance": float(row["distance"]),
        }
        for code, row in rank_primary.head(TOP_K).iterrows()
    ],
    "ranking_robustness": ranking_comparisons.to_dict(orient="records"),
    "clustering": {
        "selected_k": selected_k,
        "target_cluster": target_cluster,
        "target_cluster_size": int(selected_labels.eq(target_cluster).sum()),
        "silhouette": float(selected_validation["kmeans_silhouette"]),
        "mean_seed_pairwise_ari": float(selected_validation["mean_seed_pairwise_ari"]),
        "hierarchical_silhouette": float(selected_validation["hierarchical_silhouette"]),
        "kmeans_hierarchical_ari": float(selected_validation["kmeans_hierarchical_ari"]),
    },
    "software": {
        "listed_tools": int(ds_software["Workplace Example"].nunique()),
        "hot_tools": int(ds_software["Hot Technology"].eq("Y").sum()),
        "in_demand_tools": int(ds_software["In Demand"].eq("Y").sum()),
        "roadmap_summary": software_summary.to_dict(orient="records"),
    },
    "highest_neighbor_minus_ds_features": top_gap_up[["Domain", "Element Name", "mean_neighbor_minus_ds_z"]].to_dict(orient="records"),
    "highest_ds_minus_neighbor_features": top_gap_down[["Domain", "Element Name", "mean_neighbor_minus_ds_z"]].to_dict(orient="records"),
}

with (RESULTS_DIR / "analysis_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

assert X.shape == (882, 109), X.shape
assert full_matrix.shape == (910, 109), full_matrix.shape
assert int(flag_matrix.sum().sum()) == 69
assert int(flag_matrix.any(axis=1).sum()) == 28
assert standardization_checks["maximum_absolute_column_mean"] < 1e-12
assert abs(standardization_checks["minimum_population_sd"] - 1.0) < 1e-12
assert abs(standardization_checks["maximum_population_sd"] - 1.0) < 1e-12
assert len(list(FIGURES_DIR.glob("figure_*.png"))) == 5

print(json.dumps(summary, indent=2))
