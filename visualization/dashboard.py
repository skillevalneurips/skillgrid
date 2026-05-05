#!/usr/bin/env python3
"""Interactive Streamlit dashboard for exploring experiment results.

Usage:
    streamlit run visualization/dashboard.py -- --results outputs/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        print("Install streamlit and pandas: pip install streamlit pandas")
        sys.exit(1)

    st.set_page_config(page_title="SkillEval-Bench Dashboard", layout="wide")
    st.title("SkillEval-Bench: Experiment Dashboard")

    results_path = st.sidebar.text_input("Results JSON path", value="outputs/results.json")
    try:
        with open(results_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        st.warning(f"Results file not found: {results_path}")
        st.info("Run experiments first, then reload this dashboard.")
        show_demo_dashboard(st)
        return

    df = pd.DataFrame(results)
    st.sidebar.markdown(f"**{len(df)} experiment results loaded**")

    # Filters
    datasets = st.sidebar.multiselect("Datasets", df["dataset_id"].unique(), default=df["dataset_id"].unique())
    models = st.sidebar.multiselect("Models", df["model_id"].unique(), default=df["model_id"].unique())
    origins = st.sidebar.multiselect("Skill Origins", df["skill_origin"].unique(), default=df["skill_origin"].unique())
    policies = st.sidebar.multiselect("Runtime Policies", df["runtime_policy"].unique(), default=df["runtime_policy"].unique())

    mask = (
        df["dataset_id"].isin(datasets) &
        df["model_id"].isin(models) &
        df["skill_origin"].isin(origins) &
        df["runtime_policy"].isin(policies)
    )
    filtered = df[mask]

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Success Rate", f"{filtered['success_rate'].mean():.1%}")
    col2.metric("Avg Steps", f"{filtered['avg_steps'].mean():.1f}")
    col3.metric("Avg Cost", f"${filtered['avg_cost'].mean():.4f}")
    col4.metric("Avg Recovery", f"{filtered['recovery_rate'].mean():.1%}")

    # Results table
    st.subheader("Results Table")
    display_cols = [
        "experiment_id", "dataset_id", "model_id",
        "skill_origin", "runtime_policy",
        "success_rate", "avg_steps", "avg_cost", "recovery_rate",
    ]
    st.dataframe(filtered[display_cols].sort_values("success_rate", ascending=False))

    # Charts
    st.subheader("Success Rate by Configuration")
    if len(filtered) > 0:
        pivot = filtered.pivot_table(
            values="success_rate",
            index="skill_origin",
            columns="runtime_policy",
            aggfunc="mean",
        )
        st.bar_chart(pivot)

    st.subheader("Cost vs Success")
    if len(filtered) > 0:
        chart_data = filtered[["avg_cost", "success_rate", "model_id"]].copy()
        st.scatter_chart(chart_data, x="avg_cost", y="success_rate", color="model_id")


def show_demo_dashboard(st: "module") -> None:
    """Show a demo dashboard with placeholder data."""
    st.subheader("Demo Mode (no results loaded)")
    st.markdown("""
    ### Quick Start
    1. Run an experiment: `python experiments/run_baseline.py --dataset gsm8k --model openai --config configs/models/gpt4o.yaml`
    2. Reload this dashboard with the results path.

    ### What this dashboard shows:
    - **Overview metrics**: Success rate, steps, cost, recovery rate
    - **Filterable results table**: Filter by dataset, model, skill origin, runtime policy
    - **Interactive charts**: Success by configuration, cost vs success
    """)


if __name__ == "__main__":
    main()
