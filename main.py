from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_DIR = Path(__file__).parent / "data"


def _load_csv(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _parse_assoc_text_row(text: str) -> dict[str, object]:
    # Handles rows like: {feature_level : L0_Coarse, mean_accuracy : 0.486, ...}
    raw = text.strip().strip('"').strip().strip("{").strip("}")
    fields = [item.strip() for item in raw.split(",")]
    out: dict[str, object] = {}
    for field in fields:
        if ":" not in field:
            continue
        key, value = field.split(":", 1)
        key = key.strip().strip('"')
        value = value.strip().strip('"')
        if value == "":
            out[key] = np.nan
        elif value in {"True", "False"}:
            out[key] = value == "True"
        else:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def _normalize_accuracy_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"feature_level", "mean_accuracy", "std", "n_reps", "is_control"}
    if required.issubset(df.columns):
        out = df.copy()
    elif len(df.columns) == 1:
        # Mathematica export fallback where each row is an association-like string.
        lone_col = df.columns[0]
        row_texts = [str(lone_col)] + df.iloc[:, 0].astype(str).tolist()
        parsed = [_parse_assoc_text_row(text) for text in row_texts]
        parsed = [row for row in parsed if row]
        out = pd.DataFrame(parsed)
    else:
        return _demo_accuracy()

    # Enforce expected dtypes and defaults.
    if "feature_level" not in out.columns:
        return _demo_accuracy()
    out["mean_accuracy"] = pd.to_numeric(out.get("mean_accuracy"), errors="coerce")
    out["std"] = pd.to_numeric(out.get("std"), errors="coerce")
    out["n_reps"] = pd.to_numeric(out.get("n_reps"), errors="coerce").fillna(1).astype(int)
    if "is_control" in out.columns:
        out["is_control"] = out["is_control"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    else:
        out["is_control"] = False
    out = out.dropna(subset=["feature_level", "mean_accuracy"])
    if out.empty:
        return _demo_accuracy()
    return out


def _demo_lifecycle() -> pd.DataFrame:
    generations = np.arange(0, 9)
    levels = ["L0_Coarse", "L1_Shallow", "L2_Hist", "L3_FullRich"]
    rows: list[dict[str, float | str | int]] = []
    for idx, level in enumerate(levels):
        amp = 0.8 + idx * 0.4
        radius = amp * (1 - np.exp(-0.9 * generations)) * np.exp(-0.08 * (generations - 3) ** 2)
        state_count = (1 + idx * 0.1) * (8 + 16 * generations - 1.8 * generations**2)
        for g, r, n in zip(generations, radius, state_count):
            rows.append(
                {
                    "generation": int(g),
                    "level": level,
                    "mean_radius": float(max(r, 0)),
                    "state_count": int(max(n, 1)),
                }
            )
    return pd.DataFrame(rows)


def _demo_accuracy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature_level": "L0_Coarse", "mean_accuracy": 0.486, "std": 0.172, "n_reps": 20, "is_control": False},
            {"feature_level": "L1_Shallow", "mean_accuracy": 0.659, "std": np.nan, "n_reps": 1, "is_control": False},
            {"feature_level": "L2_Shuffled", "mean_accuracy": 0.555, "std": 0.044, "n_reps": 20, "is_control": True},
            {"feature_level": "L2_Hist", "mean_accuracy": 0.636, "std": 0.083, "n_reps": 20, "is_control": False},
            {"feature_level": "L3_FullRich", "mean_accuracy": 0.612, "std": np.nan, "n_reps": 1, "is_control": False},
        ]
    )


def _demo_coarse_rich() -> pd.DataFrame:
    layers = np.arange(0, 9)
    expr = (4 + 5 * layers + 2 * layers**2).astype(int)
    coarse = np.maximum(1, (0.4 * expr - 2 * layers).astype(int))
    rich = np.maximum(coarse + 1, (0.9 * expr).astype(int))
    return pd.DataFrame(
        {
            "layer": layers,
            "distinct_expr": expr,
            "distinct_coarse": coarse,
            "distinct_rich": rich,
            "coarse_entropy_norm": np.clip(0.2 + 0.05 * layers, 0, 1),
            "rich_entropy_norm": np.clip(0.65 + 0.04 * layers, 0, 1),
        }
    )


def _demo_embedding() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    operators = ["ExpMinusLog", "ExpPlusLog", "XMinusLog"]
    points = []
    for idx, op in enumerate(operators):
        center = np.array([idx * 2.6, (idx - 1) * 1.8, idx * 1.2], dtype=float)
        cloud = rng.normal(loc=center, scale=0.45, size=(150, 3))
        for row in cloud:
            points.append({"x": row[0], "y": row[1], "z": row[2], "operator": op})
    return pd.DataFrame(points)


def load_data() -> tuple[dict[str, pd.DataFrame], dict[str, bool]]:
    lifecycle = _load_csv("lifecycle_by_level.csv")
    accuracy = _load_csv("predictive_accuracy.csv")
    coarse_rich = _load_csv("coarse_rich_divergence.csv")
    embedding = _load_csv("embedding_points.csv")
    sources = {
        "lifecycle": lifecycle is not None,
        "accuracy": accuracy is not None,
        "coarse_rich": coarse_rich is not None,
        "embedding": embedding is not None,
    }
    data = {
        "lifecycle": lifecycle if lifecycle is not None else _demo_lifecycle(),
        "accuracy": _normalize_accuracy_df(accuracy) if accuracy is not None else _demo_accuracy(),
        "coarse_rich": coarse_rich if coarse_rich is not None else _demo_coarse_rich(),
        "embedding": embedding if embedding is not None else _demo_embedding(),
    }
    return data, sources


def page_header() -> None:
    st.set_page_config(page_title="Heartfield Explorer", page_icon=":milky_way:", layout="wide")
    st.title("Heartfield Geometry Explorer")
    st.caption(
        "An interactive walkthrough from symbol rewrite idea to current experimental state."
    )


def sidebar_controls(source_count: int) -> tuple[int, str]:
    st.sidebar.header("Controls")
    depth = st.sidebar.slider("Generation depth", min_value=4, max_value=12, value=8, step=1)
    selector = st.sidebar.selectbox("Selector policy", ["All", "OneRandom", "TwoRandom"], index=0)
    st.sidebar.markdown("---")
    if source_count == 0:
        st.sidebar.info("Showing demo data. Add CSVs in `data/` to switch to real experiment outputs.")
    elif source_count < 4:
        st.sidebar.warning(f"Using mixed mode: {source_count}/4 datasets are real CSV exports.")
    else:
        st.sidebar.success("Using your CSV exports from `data/` for all visualizations.")
    return depth, selector


def render_story(depth: int, selector: str) -> None:
    st.subheader("From Idea to Current State")
    st.markdown(
        f"""
        - **Origin:** multiway symbolic rewrite inspired by Wolfram-style branchial dynamics and EML-style operator behavior.
        - **Mechanism intuition:** exponential diversification (`Exp[...]`) plus logarithmic reconvergence (`-Log[...]`).
        - **Current tested setting:** rewrite-anywhere trees, depth target `{depth}`, selector policy `{selector}`.
        - **Practical thesis:** choose the right observational resolution, not simply the richest one.
        """
    )


def render_lifecycle(lifecycle: pd.DataFrame, is_real: bool) -> None:
    st.subheader("Lifecycle and Geometric Spread")
    st.caption("Source: Mathematica export CSV" if is_real else "Source: demo placeholder until CSV is exported")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.line(
            lifecycle,
            x="generation",
            y="mean_radius",
            color="level",
            markers=True,
            title="Mean Radius by Generation and Feature Level",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pivot = lifecycle.pivot_table(index="generation", columns="level", values="state_count", aggfunc="mean")
        st.dataframe(pivot.round(2), use_container_width=True)


def render_accuracy(accuracy: pd.DataFrame, is_real: bool) -> None:
    st.subheader("Predictive Accuracy and Controls")
    st.caption("Source: Mathematica export CSV" if is_real else "Source: technical-note seeded/demo fallback")
    accuracy = accuracy.copy()
    accuracy["group"] = np.where(accuracy["is_control"], "Control", "Primary")
    fig = px.bar(
        accuracy,
        x="feature_level",
        y="mean_accuracy",
        color="group",
        error_y="std",
        title="Terminal Basin Prediction Accuracy",
        text="n_reps",
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(accuracy.drop(columns=["group"]), use_container_width=True)


def render_coarse_vs_rich(coarse_rich: pd.DataFrame, is_real: bool) -> None:
    st.subheader("Coarse vs Rich Divergence")
    st.caption("Source: Mathematica export CSV" if is_real else "Source: demo placeholder until CSV is exported")
    c1, c2 = st.columns(2)
    with c1:
        melt = coarse_rich.melt(
            id_vars=["layer"],
            value_vars=["distinct_expr", "distinct_coarse", "distinct_rich"],
            var_name="metric",
            value_name="count",
        )
        fig = px.line(
            melt,
            x="layer",
            y="count",
            color="metric",
            markers=True,
            title="Distinct-State Counts by Layer",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        em = coarse_rich.melt(
            id_vars=["layer"],
            value_vars=["coarse_entropy_norm", "rich_entropy_norm"],
            var_name="entropy_type",
            value_name="entropy",
        )
        fig2 = px.line(
            em,
            x="layer",
            y="entropy",
            color="entropy_type",
            markers=True,
            title="Normalized Entropy (Coarse vs Rich)",
        )
        fig2.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig2, use_container_width=True)


def render_embedding_3d(embedding: pd.DataFrame, is_real: bool) -> None:
    st.subheader("3D Branch-Space Render")
    st.caption("Source: Mathematica export CSV" if is_real else "Source: demo operator clouds")
    fig = px.scatter_3d(
        embedding,
        x="x",
        y="y",
        z="z",
        color="operator",
        opacity=0.75,
        title="Embedded State Cloud by Operator",
    )
    fig.update_traces(marker={"size": 4})
    st.plotly_chart(fig, use_container_width=True)


def render_resolution_selector(accuracy: pd.DataFrame) -> None:
    st.subheader("Resolution Recommendation Prototype")
    st.caption("Simple heuristic: maximize accuracy while penalizing complexity.")

    complexity_map = {
        "L0_Coarse": 1,
        "L1_Shallow": 2,
        "L2_Shuffled": 3,
        "L2_Hist": 3,
        "L3_FullRich": 4,
    }
    lambda_penalty = st.slider("Complexity penalty (lambda)", 0.0, 0.2, 0.06, 0.01)
    work = accuracy.copy()
    work["complexity"] = work["feature_level"].map(complexity_map).fillna(3)
    work["score"] = work["mean_accuracy"] - lambda_penalty * work["complexity"]
    work = work.sort_values("score", ascending=False)

    best = work.iloc[0]
    st.success(
        f"Recommended middle-line resolution: **{best['feature_level']}** "
        f"(score = {best['score']:.3f}, accuracy = {best['mean_accuracy']:.3f})."
    )
    st.dataframe(work[["feature_level", "mean_accuracy", "complexity", "score", "n_reps"]], use_container_width=True)


def render_data_contract() -> None:
    with st.expander("CSV schema expected in `data/`"):
        st.markdown(
            """
            - `lifecycle_by_level.csv`: `generation`, `level`, `mean_radius`, `state_count`
            - `predictive_accuracy.csv`: `feature_level`, `mean_accuracy`, `std`, `n_reps`, `is_control`
            - `coarse_rich_divergence.csv`: `layer`, `distinct_expr`, `distinct_coarse`, `distinct_rich`, `coarse_entropy_norm`, `rich_entropy_norm`
            - `embedding_points.csv`: `x`, `y`, `z`, `operator`
            """
        )


def main() -> None:
    page_header()
    data, sources = load_data()
    depth, selector = sidebar_controls(sum(sources.values()))
    render_story(depth, selector)
    render_lifecycle(data["lifecycle"], sources["lifecycle"])
    render_accuracy(data["accuracy"], sources["accuracy"])
    render_coarse_vs_rich(data["coarse_rich"], sources["coarse_rich"])
    render_embedding_3d(data["embedding"], sources["embedding"])
    render_resolution_selector(data["accuracy"])
    render_data_contract()


if __name__ == "__main__":
    main()
