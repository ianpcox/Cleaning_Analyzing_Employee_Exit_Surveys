"""
Elevated Employee Exit Survey — HR Analytics Pipeline
Project Elevate — Cleaning_Analyzing_Employee_Exit_Surveys

Performs:
  1. Deep data cleaning & harmonization of DETE + TAFE surveys
  2. Dissatisfaction flag engineering (unified attrition label)
  3. Attrition driver analysis by department, tenure, age, gender
  4. Satisfaction score analysis (TAFE Likert scale)
  5. Churn prediction model (Random Forest) with feature importance
  6. 12 static charts + interactive Plotly executive dashboard

Datasets:
  - dete_survey.csv  : 822 rows, 56 columns (DETE Queensland)
  - tafe_survey.csv  : 703 rows, 72 columns (TAFE Queensland)
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/ubuntu/exit_outputs")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

PALETTE = {"Dissatisfied": "#E74C3C", "Not Dissatisfied": "#2ECC71",
           "Unknown": "#BDC3C7"}


# ── Data Loading & Cleaning ───────────────────────────────────────────────────

def load_dete(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Rename columns to snake_case
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s/]+", "_", regex=True)
    # Key columns
    df = df.rename(columns={
        "separationtype": "separation_type",
        "gender": "gender",
        "currentage": "age",
        "lengthofservice": "service_years",
        "classification": "role",
    })
    # Dissatisfaction flag: resigned due to job dissatisfaction
    dissatisfied_reasons = [
        "job_dissatisfaction", "dissatisfaction_with_the_department",
        "physical_work_environment", "lack_of_recognition",
        "lack_of_job_security", "work_location", "employment_conditions",
        "work_life_balance", "workload"
    ]
    available = [c for c in dissatisfied_reasons if c in df.columns]
    if available:
        df["dissatisfied"] = df[available].any(axis=1)
    else:
        df["dissatisfied"] = False

    df["institute"] = "DETE"
    df["separation_type"] = df["separation_type"].str.strip()
    return df


def load_tafe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Rename to harmonized names
    df = df.rename(columns={
        "Record ID": "id",
        "Institute": "institute_name",
        "WorkArea": "work_area",
        "CESSATION YEAR": "year",
        "Reason for ceasing employment": "separation_type",
        "Contributing Factors. Dissatisfaction": "contributing_dissatisfaction",
        "Contributing Factors. Job Dissatisfaction": "job_dissatisfaction",
        "Gender. What is your Gender?": "gender",
        "CurrentAge. Current Age": "age",
        "LengthofServiceOverall. Overall Length of Service at Institute (in years)": "service_years",
        "Classification. Classification": "role",
        "Employment Type. Employment Type": "employment_type",
    })
    # Dissatisfaction flag
    dissatisfied_cols = [c for c in df.columns if "dissatisf" in c.lower() or "Dissatisf" in c]
    if dissatisfied_cols:
        df["dissatisfied"] = df[dissatisfied_cols].apply(
            lambda row: any(str(v).strip() not in ["-", "", "nan", "NaN"]
                            for v in row), axis=1
        )
    else:
        df["dissatisfied"] = False

    df["institute"] = "TAFE"
    df["separation_type"] = df["separation_type"].str.strip() if "separation_type" in df.columns else "Unknown"
    return df


def harmonize_age(val):
    val = str(val).strip()
    mapping = {
        "21  25": "21-25", "26  30": "26-30", "31  35": "31-35",
        "36  40": "36-40", "41  45": "41-45", "46  50": "46-50",
        "51  55": "51-55", "56  60": "56-60", "61  65": "61-65",
        "20 or younger": "Under 21", "66 or older": "66+",
        "21-25": "21-25", "26-30": "26-30", "31-35": "31-35",
        "36-40": "36-40", "41-45": "41-45", "46-50": "46-50",
        "51-55": "51-55", "56-60": "56-60", "61-65": "61-65",
    }
    return mapping.get(val, val)


def harmonize_service(val):
    val = str(val).strip()
    mapping = {
        "Less than 1 year": "<1", "1-2": "1-2", "3-4": "3-4",
        "5-6": "5-6", "7-10": "7-10", "11-20": "11-20",
        "More than 20 years": "20+",
        "1 year": "1-2", "2 years": "1-2", "3 years": "3-4",
    }
    return mapping.get(val, val)


def build_combined(dete_path: str, tafe_path: str) -> pd.DataFrame:
    print("Loading DETE survey...")
    dete = load_dete(dete_path)
    print(f"  DETE rows: {len(dete)}")

    print("Loading TAFE survey...")
    tafe = load_tafe(tafe_path)
    print(f"  TAFE rows: {len(tafe)}")

    # Select common columns
    common_cols = ["institute", "separation_type", "dissatisfied",
                   "gender", "age", "service_years", "role"]
    dete_sub = dete[[c for c in common_cols if c in dete.columns]].copy()
    tafe_sub = tafe[[c for c in common_cols if c in tafe.columns]].copy()

    combined = pd.concat([dete_sub, tafe_sub], ignore_index=True)

    # Harmonize
    combined["age"]          = combined["age"].apply(harmonize_age)
    combined["service_years"] = combined["service_years"].apply(harmonize_service)
    combined["gender"]       = combined["gender"].str.strip().str.title()
    combined["separation_type"] = combined["separation_type"].str.strip()

    # Resignation flag
    combined["resigned"] = combined["separation_type"].str.lower().str.contains(
        "resign", na=False
    )

    # Tenure bucket
    tenure_order = ["<1", "1-2", "3-4", "5-6", "7-10", "11-20", "20+"]
    combined["tenure_bucket"] = pd.Categorical(
        combined["service_years"], categories=tenure_order, ordered=True
    )

    print(f"Combined rows: {len(combined)}")
    print(f"Dissatisfied: {combined['dissatisfied'].sum()} ({combined['dissatisfied'].mean()*100:.1f}%)")
    return combined, dete, tafe


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_separation_types(combined):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, inst in zip(axes, ["DETE", "TAFE"]):
        sub = combined[combined["institute"] == inst]
        counts = sub["separation_type"].value_counts().head(8)
        colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
        ax.barh(counts.index, counts.values, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xlabel("Number of Employees")
        ax.set_title(f"{inst} — Separation Types")
        for i, v in enumerate(counts.values):
            ax.text(v + 1, i, str(v), va="center", fontsize=8)

    plt.suptitle("Why Are Employees Leaving?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_separation_types.png")
    plt.close()
    print("Saved: 01_separation_types.png")


def plot_dissatisfaction_overview(combined):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall dissatisfaction by institute
    rates = combined.groupby("institute")["dissatisfied"].mean() * 100
    colors = ["#E74C3C", "#3498DB"]
    bars = axes[0].bar(rates.index, rates.values, color=colors, alpha=0.85, edgecolor="white", width=0.4)
    for bar, val in zip(bars, rates.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Dissatisfaction Rate (%)")
    axes[0].set_title("Dissatisfaction Rate by Institute")
    axes[0].set_ylim(0, max(rates.values) * 1.3)

    # Dissatisfaction by gender
    gender_rate = (
        combined[combined["gender"].isin(["Male", "Female"])]
        .groupby(["institute", "gender"])["dissatisfied"]
        .mean() * 100
    ).unstack()
    x = np.arange(len(gender_rate))
    w = 0.35
    axes[1].bar(x - w/2, gender_rate.get("Female", [0]*len(x)),
                w, label="Female", color="#E91E8C", alpha=0.85, edgecolor="white")
    axes[1].bar(x + w/2, gender_rate.get("Male", [0]*len(x)),
                w, label="Male", color="#3498DB", alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gender_rate.index)
    axes[1].set_ylabel("Dissatisfaction Rate (%)")
    axes[1].set_title("Dissatisfaction Rate by Institute & Gender")
    axes[1].legend()

    plt.suptitle("Employee Dissatisfaction Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_dissatisfaction_overview.png")
    plt.close()
    print("Saved: 02_dissatisfaction_overview.png")


def plot_tenure_analysis(combined):
    tenure_order = ["<1", "1-2", "3-4", "5-6", "7-10", "11-20", "20+"]
    tenure_data = (
        combined[combined["service_years"].isin(tenure_order)]
        .groupby(["service_years", "institute"])["dissatisfied"]
        .mean() * 100
    ).unstack().reindex(tenure_order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dissatisfaction by tenure
    for inst, color in zip(["DETE", "TAFE"], ["#E74C3C", "#3498DB"]):
        if inst in tenure_data.columns:
            axes[0].plot(tenure_data.index, tenure_data[inst], "o-",
                         color=color, lw=2, markersize=7, label=inst)
    axes[0].set_xlabel("Years of Service")
    axes[0].set_ylabel("Dissatisfaction Rate (%)")
    axes[0].set_title("Dissatisfaction Rate by Tenure")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    # Headcount by tenure
    tenure_counts = (
        combined[combined["service_years"].isin(tenure_order)]
        .groupby("service_years")
        .size()
        .reindex(tenure_order)
    )
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(tenure_counts)))
    axes[1].bar(tenure_counts.index, tenure_counts.values,
                color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Years of Service")
    axes[1].set_ylabel("Number of Employees")
    axes[1].set_title("Headcount by Tenure")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Tenure Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_tenure_analysis.png")
    plt.close()
    print("Saved: 03_tenure_analysis.png")


def plot_age_analysis(combined):
    age_order = ["Under 21", "21-25", "26-30", "31-35", "36-40",
                 "41-45", "46-50", "51-55", "56-60", "61-65", "66+"]
    valid_ages = [a for a in age_order if a in combined["age"].values]

    age_diss = (
        combined[combined["age"].isin(valid_ages)]
        .groupby("age")["dissatisfied"]
        .agg(rate=lambda x: x.mean() * 100, count="count")
        .reindex(valid_ages)
        .dropna()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#E74C3C" if v > 30 else "#F39C12" if v > 20 else "#2ECC71"
              for v in age_diss["rate"].values]
    axes[0].bar(age_diss.index, age_diss["rate"], color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Age Group")
    axes[0].set_ylabel("Dissatisfaction Rate (%)")
    axes[0].set_title("Dissatisfaction Rate by Age Group")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(age_diss.index, age_diss["count"],
                color=plt.cm.Purples(np.linspace(0.3, 0.9, len(age_diss))),
                alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("Number of Employees")
    axes[1].set_title("Headcount by Age Group")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle("Age Group Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_age_analysis.png")
    plt.close()
    print("Saved: 04_age_analysis.png")


def plot_resignation_vs_dissatisfaction(combined):
    resigned = combined[combined["resigned"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Resignation rate by tenure
    tenure_order = ["<1", "1-2", "3-4", "5-6", "7-10", "11-20", "20+"]
    res_tenure = (
        combined[combined["service_years"].isin(tenure_order)]
        .groupby("service_years")["resigned"]
        .mean() * 100
    ).reindex(tenure_order)
    axes[0].bar(res_tenure.index, res_tenure.values,
                color=plt.cm.Oranges(np.linspace(0.3, 0.9, len(res_tenure))),
                alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Years of Service")
    axes[0].set_ylabel("Resignation Rate (%)")
    axes[0].set_title("Resignation Rate by Tenure")
    axes[0].tick_params(axis="x", rotation=30)

    # Dissatisfied vs. not dissatisfied among resignees
    diss_counts = resigned["dissatisfied"].value_counts()
    labels = ["Dissatisfied" if k else "Not Dissatisfied" for k in diss_counts.index]
    colors = [PALETTE.get(l, "#BDC3C7") for l in labels]
    axes[1].pie(diss_counts.values, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Resignees: Dissatisfied vs. Not Dissatisfied")

    plt.suptitle("Resignation & Dissatisfaction Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_resignation_dissatisfaction.png")
    plt.close()
    print("Saved: 05_resignation_dissatisfaction.png")


def plot_tafe_satisfaction_scores(tafe):
    """Analyze TAFE Likert scale satisfaction items."""
    likert_cols = [c for c in tafe.columns if "Topic:" in c][:15]
    if not likert_cols:
        print("  No TAFE Likert columns found — skipping satisfaction heatmap")
        return

    score_map = {
        "Strongly Agree": 5, "Agree": 4, "Neutral": 3,
        "Disagree": 2, "Strongly Disagree": 1,
    }
    scores = tafe[likert_cols].replace(score_map)
    scores = scores.apply(pd.to_numeric, errors="coerce")
    mean_scores = scores.mean().sort_values()

    # Shorten labels
    short_labels = [c.split("Topic:")[-1].split(";")[0].strip()[:50] for c in mean_scores.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E74C3C" if v < 3.5 else "#F39C12" if v < 4.0 else "#2ECC71"
              for v in mean_scores.values]
    ax.barh(short_labels, mean_scores.values, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(4.0, color="black", linestyle="--", lw=1, label="Agree threshold (4.0)")
    ax.set_xlabel("Mean Score (1=Strongly Disagree, 5=Strongly Agree)")
    ax.set_title("TAFE Workplace Satisfaction Scores\n(Mean Likert Rating per Topic)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_xlim(1, 5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_tafe_satisfaction_scores.png")
    plt.close()
    print("Saved: 06_tafe_satisfaction_scores.png")


def plot_heatmap_tenure_age(combined):
    tenure_order = ["<1", "1-2", "3-4", "5-6", "7-10", "11-20", "20+"]
    age_order    = ["21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60"]

    pivot = (
        combined[
            combined["service_years"].isin(tenure_order) &
            combined["age"].isin(age_order)
        ]
        .groupby(["age", "service_years"])["dissatisfied"]
        .mean() * 100
    ).unstack().reindex(index=age_order, columns=tenure_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, cmap="RdYlGn_r", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Dissatisfaction Rate (%)"},
                annot=True, fmt=".0f", annot_kws={"size": 8})
    ax.set_title("Dissatisfaction Rate Heatmap — Age vs. Tenure",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Years of Service")
    ax.set_ylabel("Age Group")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_heatmap_tenure_age.png")
    plt.close()
    print("Saved: 07_heatmap_tenure_age.png")


def build_churn_model(combined):
    """Build a Random Forest churn prediction model."""
    print("\nBuilding churn prediction model...")

    df = combined.copy()
    # Encode features
    feature_cols = ["institute", "gender", "age", "service_years", "separation_type"]
    available = [c for c in feature_cols if c in df.columns]
    df_model = df[available + ["dissatisfied"]].dropna()

    le = LabelEncoder()
    for col in available:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model[available]
    y = df_model["dissatisfied"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                 random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
    print(f"  Test ROC-AUC:  {auc:.3f}")
    print(f"  5-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Not Dissatisfied", "Dissatisfied"]))

    return rf, X_test, y_test, y_pred, y_proba, available


def plot_model_results(rf, X_test, y_test, y_pred, y_proba, feature_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Not Dissatisfied", "Dissatisfied"],
                yticklabels=["Not Dissatisfied", "Dissatisfied"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="#E74C3C", lw=2, label=f"ROC AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve — Churn Prediction Model")
    axes[1].legend()

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
    axes[2].barh(importances.index, importances.values, color=colors, alpha=0.85, edgecolor="white")
    axes[2].set_xlabel("Feature Importance")
    axes[2].set_title("Random Forest Feature Importance")

    plt.suptitle("Churn Prediction Model Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_model_results.png")
    plt.close()
    print("Saved: 08_model_results.png")


def plot_institute_comparison(combined):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Separation type comparison
    for ax, inst in zip(axes, ["DETE", "TAFE"]):
        sub = combined[combined["institute"] == inst]
        diss_by_sep = (
            sub.groupby("separation_type")["dissatisfied"]
            .agg(rate=lambda x: x.mean() * 100, count="count")
            .query("count >= 5")
            .sort_values("rate", ascending=True)
            .tail(8)
        )
        colors = ["#E74C3C" if v > 40 else "#F39C12" if v > 20 else "#2ECC71"
                  for v in diss_by_sep["rate"].values]
        ax.barh(diss_by_sep.index, diss_by_sep["rate"],
                color=colors, alpha=0.85, edgecolor="white")
        ax.set_xlabel("Dissatisfaction Rate (%)")
        ax.set_title(f"{inst} — Dissatisfaction by Separation Type")

    plt.suptitle("Dissatisfaction Rate by Separation Type", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_institute_comparison.png")
    plt.close()
    print("Saved: 09_institute_comparison.png")


def plot_employment_type(tafe):
    if "employment_type" not in tafe.columns:
        print("  employment_type not found — skipping")
        return
    emp_diss = (
        tafe.groupby("employment_type")["dissatisfied"]
        .agg(rate=lambda x: x.mean() * 100, count="count")
        .query("count >= 5")
        .sort_values("rate", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E74C3C" if v > 40 else "#F39C12" if v > 20 else "#2ECC71"
              for v in emp_diss["rate"].values]
    ax.barh(emp_diss.index, emp_diss["rate"],
            color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Dissatisfaction Rate (%)")
    ax.set_title("Dissatisfaction Rate by Employment Type (TAFE)",
                 fontsize=12, fontweight="bold")
    for i, (_, row) in enumerate(emp_diss.iterrows()):
        ax.text(row["rate"] + 0.3, i, f"{row['rate']:.1f}%  (n={row['count']})",
                va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_employment_type.png")
    plt.close()
    print("Saved: 10_employment_type.png")


def plot_long_tenure_risk(combined):
    """Highlight the 'long-tenure dissatisfaction' pattern."""
    tenure_order = ["<1", "1-2", "3-4", "5-6", "7-10", "11-20", "20+"]
    valid = combined[combined["service_years"].isin(tenure_order)]

    diss_rate = (
        valid.groupby(["service_years", "dissatisfied"])
        .size()
        .unstack(fill_value=0)
        .reindex(tenure_order)
    )
    diss_rate.columns = ["Not Dissatisfied", "Dissatisfied"]
    diss_rate_pct = diss_rate.div(diss_rate.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    diss_rate_pct.plot(kind="bar", stacked=True, ax=ax,
                       color=["#2ECC71", "#E74C3C"], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Years of Service")
    ax.set_ylabel("Percentage of Employees (%)")
    ax.set_title("Dissatisfaction Composition by Tenure\n(Stacked 100% Bar)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_tenure_dissatisfaction_stacked.png")
    plt.close()
    print("Saved: 11_tenure_dissatisfaction_stacked.png")


# ── Interactive Plotly Executive Dashboard ────────────────────────────────────

def build_dashboard(combined, tafe):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tenure_order = ["<1", "1-2", "3-4", "5-6", "7-10", "11-20", "20+"]
    age_order    = ["21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Dissatisfaction Rate by Institute",
            "Separation Types (DETE vs TAFE)",
            "Dissatisfaction Rate by Tenure",
            "Dissatisfaction Rate by Age Group",
            "Resignation Rate by Tenure",
            "TAFE: Dissatisfied vs. Not Dissatisfied (Resignees)",
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "pie"}],
        ],
        vertical_spacing=0.13,
        horizontal_spacing=0.1,
    )

    # 1. Dissatisfaction by institute
    rates = combined.groupby("institute")["dissatisfied"].mean() * 100
    fig.add_trace(go.Bar(
        x=rates.index, y=rates.values,
        marker_color=["#E74C3C", "#3498DB"], opacity=0.85,
        text=[f"{v:.1f}%" for v in rates.values], textposition="outside",
        hovertemplate="%{x}<br>Dissatisfaction: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    # 2. Separation types
    for inst, color in zip(["DETE", "TAFE"], ["#E74C3C", "#3498DB"]):
        sub = combined[combined["institute"] == inst]
        counts = sub["separation_type"].value_counts().head(6)
        fig.add_trace(go.Bar(
            name=inst, x=counts.index, y=counts.values,
            marker_color=color, opacity=0.8,
            hovertemplate=f"{inst}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ), row=1, col=2)

    # 3. Dissatisfaction by tenure
    for inst, color in zip(["DETE", "TAFE"], ["#E74C3C", "#3498DB"]):
        sub = combined[combined["institute"] == inst]
        tenure_diss = (
            sub[sub["service_years"].isin(tenure_order)]
            .groupby("service_years")["dissatisfied"]
            .mean() * 100
        ).reindex(tenure_order)
        fig.add_trace(go.Scatter(
            x=tenure_order, y=tenure_diss.values,
            name=inst, mode="lines+markers",
            line=dict(color=color, width=2),
            hovertemplate=f"{inst}: %{{x}}<br>Rate: %{{y:.1f}}%<extra></extra>",
        ), row=2, col=1)

    # 4. Dissatisfaction by age
    age_diss = (
        combined[combined["age"].isin(age_order)]
        .groupby("age")["dissatisfied"]
        .mean() * 100
    ).reindex(age_order)
    fig.add_trace(go.Bar(
        x=age_order, y=age_diss.values,
        marker_color=age_diss.values, marker_colorscale="RdYlGn_r",
        hovertemplate="Age: %{x}<br>Rate: %{y:.1f}%<extra></extra>",
    ), row=2, col=2)

    # 5. Resignation rate by tenure
    res_tenure = (
        combined[combined["service_years"].isin(tenure_order)]
        .groupby("service_years")["resigned"]
        .mean() * 100
    ).reindex(tenure_order)
    fig.add_trace(go.Bar(
        x=tenure_order, y=res_tenure.values,
        marker_color="#F39C12", opacity=0.85,
        hovertemplate="Tenure: %{x}<br>Resignation Rate: %{y:.1f}%<extra></extra>",
    ), row=3, col=1)

    # 6. Dissatisfied pie (resignees)
    resigned = combined[combined["resigned"]]
    diss_counts = resigned["dissatisfied"].value_counts()
    labels = ["Dissatisfied" if k else "Not Dissatisfied" for k in diss_counts.index]
    fig.add_trace(go.Pie(
        labels=labels, values=diss_counts.values,
        hole=0.35,
        marker_colors=["#E74C3C", "#2ECC71"],
    ), row=3, col=2)

    fig.update_layout(
        title=dict(
            text="Employee Exit Survey — HR Analytics Executive Dashboard",
            font=dict(size=18, family="Arial"), x=0.5,
        ),
        height=1100,
        template="plotly_white",
        showlegend=True,
        font=dict(family="Arial", size=11),
        barmode="group",
    )
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=2)
    fig.update_yaxes(ticksuffix="%", row=3, col=1)

    out_path = OUT_DIR / "dashboard.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print("Saved: dashboard.html")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  EMPLOYEE EXIT SURVEY — HR ANALYTICS PIPELINE")
    print("="*60)

    combined, dete, tafe = build_combined(
        "/home/ubuntu/dete_survey.csv",
        "/home/ubuntu/tafe_survey.csv"
    )

    print("\nGenerating visualizations...")
    plot_separation_types(combined)
    plot_dissatisfaction_overview(combined)
    plot_tenure_analysis(combined)
    plot_age_analysis(combined)
    plot_resignation_vs_dissatisfaction(combined)
    plot_tafe_satisfaction_scores(tafe)
    plot_heatmap_tenure_age(combined)
    plot_institute_comparison(combined)
    plot_employment_type(tafe)
    plot_long_tenure_risk(combined)

    rf, X_test, y_test, y_pred, y_proba, feature_names = build_churn_model(combined)
    plot_model_results(rf, X_test, y_test, y_pred, y_proba, feature_names)

    print("\nBuilding interactive Plotly dashboard...")
    build_dashboard(combined, tafe)

    # Save clean combined dataset
    combined.to_csv(OUT_DIR / "combined_exit_surveys.csv", index=False)

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"  Total respondents:       {len(combined):,}")
    print(f"  DETE respondents:        {(combined['institute']=='DETE').sum():,}")
    print(f"  TAFE respondents:        {(combined['institute']=='TAFE').sum():,}")
    print(f"  Overall dissatisfied:    {combined['dissatisfied'].mean()*100:.1f}%")
    print(f"  Overall resigned:        {combined['resigned'].mean()*100:.1f}%")
    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
