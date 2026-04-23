from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


DATA_DIR = Path(__file__).parent / "data"
EVIDENCE_DIR = Path(__file__).parent / "assets" / "evidence"


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


def _paper_ablation_rows() -> pd.DataFrame:
    rows = [
        ("L0_Coarse", "ExpMinusLog", 0, 1, 1, 0.0, 0.0),
        ("L0_Coarse", "ExpMinusLog", 1, 7, 2, 1.39971, 4.88998),
        ("L0_Coarse", "ExpMinusLog", 2, 22, 3, 1.74685, 5.76461),
        ("L0_Coarse", "ExpMinusLog", 3, 42, 4, 1.71053, 5.74737),
        ("L0_Coarse", "ExpMinusLog", 4, 57, 5, 1.6692, 5.37771),
        ("L0_Coarse", "ExpMinusLog", 5, 63, 6, 1.56557, 5.02886),
        ("L0_Coarse", "ExpMinusLog", 6, 64, 7, 1.53093, 4.88998),
        ("L0_Coarse", "ExpMinusLog", 7, 64, 7, 1.53093, 4.88998),
        ("L0_Coarse", "ExpMinusLog", 8, 64, 7, 1.53093, 4.88998),
        ("L1_Shallow", "ExpMinusLog", 0, 1, 1, 0.0, 0.0),
        ("L1_Shallow", "ExpMinusLog", 1, 7, 2, 1.71429, 6.0),
        ("L1_Shallow", "ExpMinusLog", 2, 22, 3, 2.13945, 7.06018),
        ("L1_Shallow", "ExpMinusLog", 3, 42, 4, 2.09496, 7.03906),
        ("L1_Shallow", "ExpMinusLog", 4, 57, 5, 2.04434, 6.58633),
        ("L1_Shallow", "ExpMinusLog", 5, 63, 6, 1.91742, 6.15908),
        ("L1_Shallow", "ExpMinusLog", 6, 64, 7, 1.875, 6.0),
        ("L1_Shallow", "ExpMinusLog", 7, 64, 7, 1.875, 6.0),
        ("L1_Shallow", "ExpMinusLog", 8, 64, 7, 1.875, 6.0),
        ("L2_Hist", "ExpMinusLog", 0, 1, 1, 0.0, 0.0),
        ("L2_Hist", "ExpMinusLog", 1, 7, 4, 2.59854, 7.75859),
        ("L2_Hist", "ExpMinusLog", 2, 22, 7, 3.0926, 9.13671),
        ("L2_Hist", "ExpMinusLog", 3, 42, 10, 3.08279, 9.12135),
        ("L2_Hist", "ExpMinusLog", 4, 57, 13, 3.07776, 8.55032),
        ("L2_Hist", "ExpMinusLog", 5, 63, 15, 3.04462, 8.00835),
        ("L2_Hist", "ExpMinusLog", 6, 64, 16, 3.02876, 7.84777),
        ("L2_Hist", "ExpMinusLog", 7, 64, 16, 3.02876, 7.84777),
        ("L2_Hist", "ExpMinusLog", 8, 64, 16, 3.02876, 7.84777),
        ("L3_FullRich", "ExpMinusLog", 0, 1, 1, 0.0, 0.0),
        ("L3_FullRich", "ExpMinusLog", 1, 7, 7, 3.23048, 9.00246),
        ("L3_FullRich", "ExpMinusLog", 2, 22, 15, 3.74908, 10.6934),
        ("L3_FullRich", "ExpMinusLog", 3, 42, 22, 3.74988, 10.8173),
        ("L3_FullRich", "ExpMinusLog", 4, 57, 28, 3.76226, 10.313),
        ("L3_FullRich", "ExpMinusLog", 5, 63, 31, 3.7449, 9.77375),
        ("L3_FullRich", "ExpMinusLog", 6, 64, 32, 3.72943, 9.55942),
        ("L3_FullRich", "ExpMinusLog", 7, 64, 32, 3.72943, 9.55942),
        ("L3_FullRich", "ExpMinusLog", 8, 64, 32, 3.72943, 9.55942),
    ]
    return pd.DataFrame(
        rows,
        columns=["feature_level", "operator", "layer", "state_count", "distinct_tuples", "mean_radius", "max_radius"],
    )


def _paper_l2_shuffle_replicates() -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    targets = {
        "L1_Shallow": (0.648176, 0.173072),
        "L2_Shuffled": (0.554706, 0.0438994),
        "L2_Hist": (0.636082, 0.0828796),
    }
    n = 20
    rows = {"replicate": np.arange(1, n + 1)}
    for k, (mean, std) in targets.items():
        vals = rng.normal(mean, std, size=n)
        vals = np.clip(vals, 0.0, 1.0)
        rows[k] = vals
    return pd.DataFrame(rows)


def _paper_operator_traces() -> pd.DataFrame:
    # Reconstructed from the Mathematica trace table shared in screenshots.
    rows = [
        ("ExpMinusLog", 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("ExpMinusLog", 1, 4.925, 1.0, 0.0, 0.0, 1.0, 1.0),
        ("ExpMinusLog", 2, 10.7125, 1.0, 0.0, 0.910952, 1.0, 1.91095),
        ("ExpMinusLog", 3, 13.5875, 1.0, 0.0, 0.97321, 1.0, 2.72806),
        ("ExpMinusLog", 4, 10.9375, 0.7875, 0.0, 0.7875, 0.7875, 2.71571),
        ("ExpMinusLog", 5, 5.5375, 0.5375, 0.0, 0.5375, 0.5375, 2.25536),
        ("ExpMinusLog", 6, 1.6, 0.4, 0.0, 0.4, 0.4, 1.93229),
        ("ExpMinusLog", 7, 0.2, 0.2, 0.0, 0.2, 0.2, 1.0975),
        ("ExpPlusLog", 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("ExpPlusLog", 1, 4.9, 0.0, 1.0, 0.0, 0.0, -1.0),
        ("ExpPlusLog", 2, 10.4875, 0.0, 1.0, 0.0, 0.0, -1.89076),
        ("ExpPlusLog", 3, 12.95, 0.0, 1.0, 0.0, 0.0, -2.69929),
        ("ExpPlusLog", 4, 10.1, 0.0, 0.825, 0.0, 0.0, -2.89857),
        ("ExpPlusLog", 5, 4.9795, 0.0, 0.5375, 0.0, 0.0, -2.28929),
        ("ExpPlusLog", 6, 1.4125, 0.0, 0.3625, 0.0, 0.0, -1.78933),
        ("ExpPlusLog", 7, 0.175, 0.0, 0.175, 0.0, 0.0, -0.975),
        ("XMinusLog", 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("XMinusLog", 1, 4.75, 1.0, 0.0, 0.0, 0.0, 1.0),
        ("XMinusLog", 2, 9.7125, 1.0, 0.0, 1.0, 0.0, 2.0),
        ("XMinusLog", 3, 11.225, 1.0, 0.0, 1.0, 0.0, 3.0),
        ("XMinusLog", 4, 7.9875, 0.7875, 0.0, 0.7875, 0.0, 3.15),
        ("XMinusLog", 5, 3.4875, 0.55, 0.0, 0.55, 0.0, 2.75),
        ("XMinusLog", 6, 0.85, 0.325, 0.0, 0.325, 0.0, 1.95),
        ("XMinusLog", 7, 0.0875, 0.0875, 0.0, 0.0875, 0.0, 0.6125),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "operator",
            "generation",
            "state_count",
            "negative_log_density",
            "positive_log_density",
            "repeated_negative_log_density",
            "exp_negative_log_mix_density",
            "mean_signed_log_imbalance",
        ],
    )


def _build_heartfield_diff_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for op, grp in trace_rows.sort_values(["operator", "generation"]).groupby("operator"):
        g = grp.reset_index(drop=True)
        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]
            delta_n = float(nxt["state_count"] - cur["state_count"])
            delta_p = float(nxt["mean_signed_log_imbalance"] - cur["mean_signed_log_imbalance"])
            delta_c = float(nxt["exp_negative_log_mix_density"] - cur["exp_negative_log_mix_density"])
            rows.append(
                {
                    "operator": op,
                    "generation": int(cur["generation"]),
                    "next_generation": int(nxt["generation"]),
                    "n": float(cur["state_count"]),
                    "p": float(cur["mean_signed_log_imbalance"]),
                    "c": float(cur["exp_negative_log_mix_density"]),
                    "negative_log_density": float(cur["negative_log_density"]),
                    "positive_log_density": float(cur["positive_log_density"]),
                    "repeated_negative_log_density": float(cur["repeated_negative_log_density"]),
                    "delta_n": delta_n,
                    "delta_p": delta_p,
                    "delta_c": delta_c,
                    # Weighted expansion proxies for scatter diagnostics.
                    "expansion_weighted_c": float(max(0.0, delta_c) * nxt["state_count"]),
                    "expansion_weighted_p": float(max(0.0, delta_p) * nxt["state_count"]),
                    "growth_rate": float(delta_n / cur["state_count"]) if float(cur["state_count"]) != 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_persistence_profile(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for op, grp in trace_rows.sort_values(["operator", "generation"]).groupby("operator"):
        peak_abs_imbalance = float(np.max(np.abs(grp["mean_signed_log_imbalance"])))
        final_abs_imbalance = float(np.abs(grp["mean_signed_log_imbalance"].iloc[-1]))
        peak_mix = float(np.max(grp["exp_negative_log_mix_density"]))
        final_mix = float(grp["exp_negative_log_mix_density"].iloc[-1])
        polarity_persistence = (final_abs_imbalance / peak_abs_imbalance) if peak_abs_imbalance > 0 else 0.0
        mix_persistence = (final_mix / peak_mix) if peak_mix > 0 else 0.0
        coherence_gap = abs(polarity_persistence - mix_persistence)
        rows.append(
            {
                "operator": op,
                "depth": int(grp["generation"].max()),
                "peak_abs_imbalance": peak_abs_imbalance,
                "final_abs_imbalance": final_abs_imbalance,
                "polarity_persistence": float(polarity_persistence),
                "peak_mix": peak_mix,
                "final_mix": final_mix,
                "mix_persistence": float(mix_persistence),
                "coherence_gap": float(coherence_gap),
            }
        )
    return pd.DataFrame(rows)


def _fit_plane(df: pd.DataFrame, x1: str, x2: str, y: str) -> dict[str, float]:
    work = df[[x1, x2, y]].dropna().astype(float)
    if len(work) < 4:
        return {"intercept": np.nan, "beta1": np.nan, "beta2": np.nan, "adj_r2": np.nan}
    x = np.column_stack([np.ones(len(work)), work[x1].to_numpy(), work[x2].to_numpy()])
    target = work[y].to_numpy()
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    pred = x @ beta
    sse = float(np.sum((target - pred) ** 2))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else 0.0
    n = len(work)
    p = 2
    adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else np.nan
    return {
        "intercept": float(beta[0]),
        "beta1": float(beta[1]),
        "beta2": float(beta[2]),
        "adj_r2": float(adj_r2),
    }


def _simulate_single_mode_model(tmax: int = 40) -> pd.DataFrame:
    alpha, beta, gamma = 0.68, 0.036, 0.88
    mu, nu, lam = 0.73, 0.062, 0.262
    n, c, p = 0.2, 0.0, 0.0
    rows = [{"t": 0, "n": n, "c": c, "p": p, "delta_n": 0.0, "delta_c": 0.0, "delta_p": 0.0}]
    for t in range(1, tmax + 1):
        pulse = float(2.4 * np.exp(-((t - 8) ** 2) / 18) - 0.22 * n)
        n_next = n + pulse
        d_n = n_next - n
        c_next = c + alpha + beta * d_n - gamma * c
        p_next = p + mu + nu * d_n - lam * p
        rows.append(
            {
                "t": t,
                "n": float(n_next),
                "c": float(c_next),
                "p": float(p_next),
                "delta_n": float(d_n),
                "delta_c": float(c_next - c),
                "delta_p": float(p_next - p),
            }
        )
        n, c, p = n_next, c_next, p_next
    return pd.DataFrame(rows)


def _simulate_rank_models(tmax: int = 40) -> pd.DataFrame:
    models = [
        ("Rank2 ExpMinusLog-like", 0.68, 0.036, 0.88, 0.73, 0.062, 0.262),
        ("Rank1 XMinusLog-like", 0.0, 0.0, 0.88, 0.57, 0.101, 0.169),
    ]
    all_rows: list[dict[str, float | str | int]] = []
    for name, alpha, beta, gamma, mu, nu, lam in models:
        n, c, p = 0.2, 0.0, 0.0
        all_rows.append({"model": name, "t": 0, "n": n, "c": c, "p": p})
        for t in range(1, tmax + 1):
            n_next = n + float(2.4 * np.exp(-((t - 8) ** 2) / 18) - 0.22 * n)
            d_n = n_next - n
            c_next = c + alpha + beta * d_n - gamma * c
            p_next = p + mu + nu * d_n - lam * p
            all_rows.append({"model": name, "t": t, "n": float(n_next), "c": float(c_next), "p": float(p_next)})
            n, c, p = n_next, c_next, p_next
    return pd.DataFrame(all_rows)


def _simulate_three_mode_model(tmax: int = 80) -> pd.DataFrame:
    pump1, pump2, pump3 = 1.18, 1.02, 0.86
    loss1, loss2, loss3 = 0.22, 0.24, 0.27
    sat, cross = 0.11, 0.055
    k12, k23, k31 = 0.34, 0.18, 0.12
    a1, a2, a3 = 0.04, 0.03, 0.02
    n = a1 + a2 + a3
    c = 2 * np.sqrt(a1 * a2)
    p = (a1 + a2) - a3
    rows = [{"t": 0, "a1": a1, "a2": a2, "a3": a3, "n": n, "c": c, "p": p, "delta_n": 0.0, "delta_c": 0.0, "delta_p": 0.0}]
    for t in range(1, tmax + 1):
        pulse = float(1.9 * np.exp(-((t - 16) ** 2) / 42))
        a1n = max(0.0, a1 + a1 * (pump1 + 0.75 * pulse - loss1 - sat * a1 - cross * (a2 + a3)) + k12 * np.sqrt(a1 * a2) - k31 * np.sqrt(a1 * a3))
        a2n = max(0.0, a2 + a2 * (pump2 + 0.55 * pulse - loss2 - sat * a2 - cross * (a1 + a3)) + k12 * np.sqrt(a1 * a2) + k23 * np.sqrt(a2 * a3))
        a3n = max(0.0, a3 + a3 * (pump3 + 0.28 * pulse - loss3 - sat * a3 - cross * (a1 + a2)) - k23 * np.sqrt(a2 * a3) + k31 * np.sqrt(a1 * a3))
        nn = a1n + a2n + a3n
        cn = 2 * np.sqrt(a1n * a2n)
        pn = (a1n + a2n) - a3n
        rows.append(
            {
                "t": t,
                "a1": float(a1n),
                "a2": float(a2n),
                "a3": float(a3n),
                "n": float(nn),
                "c": float(cn),
                "p": float(pn),
                "delta_n": float(nn - n),
                "delta_c": float(cn - c),
                "delta_p": float(pn - p),
            }
        )
        a1, a2, a3, n, c, p = a1n, a2n, a3n, nn, cn, pn
    return pd.DataFrame(rows)


def _ast_expr_to_string(expr: object) -> str:
    if isinstance(expr, str):
        return expr
    if not isinstance(expr, tuple) or len(expr) == 0:
        return str(expr)
    head = expr[0]
    if head == "P" and len(expr) == 3:
        return f"P[{_ast_expr_to_string(expr[1])},{_ast_expr_to_string(expr[2])}]"
    if head in {"Exp", "Log", "Sin"} and len(expr) == 2:
        return f"{head}[{_ast_expr_to_string(expr[1])}]"
    if head == "Add" and len(expr) == 3:
        return f"{_ast_expr_to_string(expr[1])}+{_ast_expr_to_string(expr[2])}"
    if head == "Sub" and len(expr) == 3:
        return f"{_ast_expr_to_string(expr[1])}-{_ast_expr_to_string(expr[2])}"
    if head == "Mul" and len(expr) == 3:
        return f"{_ast_expr_to_string(expr[1])}*{_ast_expr_to_string(expr[2])}"
    return f"{head}[{','.join(_ast_expr_to_string(x) for x in expr[1:])}]"


def _is_p_node(expr: object) -> bool:
    return isinstance(expr, tuple) and len(expr) == 3 and expr[0] == "P"


def _random_plus_tree(n_leaves: int, rng: np.random.Generator) -> object:
    nodes: list[object] = [f"x{i}" for i in range(1, n_leaves + 1)]
    while len(nodes) > 1:
        idx = sorted(rng.choice(len(nodes), size=2, replace=False).tolist())
        a = nodes[idx[0]]
        b = nodes[idx[1]]
        for rm in reversed(idx):
            nodes.pop(rm)
        nodes.append(("P", a, b))
    return nodes[0]


def _op_exp_minus_log(x: object, y: object) -> object:
    return ("Sub", ("Exp", x), ("Log", y))


def _op_exp_plus_log(x: object, y: object) -> object:
    return ("Add", ("Exp", x), ("Log", y))


def _op_exp_minus_log_swap(x: object, y: object) -> object:
    return ("Sub", ("Exp", y), ("Log", x))


def _op_exp_minus_y(x: object, y: object) -> object:
    return ("Sub", ("Exp", x), y)


def _op_x_minus_log(x: object, y: object) -> object:
    return ("Sub", x, ("Log", y))


def _op_sin_minus_log(x: object, y: object) -> object:
    return ("Sub", ("Sin", x), ("Log", y))


def _op_exp_minus_sin(x: object, y: object) -> object:
    return ("Sub", ("Exp", x), ("Sin", y))


OPERATOR_LIBRARY_AST = {
    "ExpMinusLog": _op_exp_minus_log,
    "ExpPlusLog": _op_exp_plus_log,
    "ExpMinusLogSwap": _op_exp_minus_log_swap,
    "ExpMinusY": _op_exp_minus_y,
    "XMinusLog": _op_x_minus_log,
    "SinMinusLog": _op_sin_minus_log,
    "ExpMinusSin": _op_exp_minus_sin,
}


def _single_step_states(expr: object, op_func) -> list[object]:
    results: list[object] = []
    if _is_p_node(expr):
        results.append(op_func(expr[1], expr[2]))
    if isinstance(expr, tuple) and len(expr) >= 2:
        head = expr[0]
        args = list(expr[1:])
        for i, child in enumerate(args):
            rewritten_children = _single_step_states(child, op_func)
            for rw in rewritten_children:
                new_args = args.copy()
                new_args[i] = rw
                results.append((head, *new_args))
    # dedupe preserving order
    return list(dict.fromkeys(results))


def _select_events(states: list[object], selector: str, rng: np.random.Generator) -> list[object]:
    if selector == "All":
        return states
    if not states:
        return []
    if selector == "First":
        return [states[0]]
    if selector == "Last":
        return [states[-1]]
    if selector == "Half":
        if len(states) <= 1:
            return states
        k = int(np.ceil(len(states) / 2))
        idx = sorted(rng.choice(len(states), size=k, replace=False).tolist())
        return [states[i] for i in idx]
    if selector == "OneRandom":
        return [states[int(rng.integers(0, len(states)))]]
    if selector == "TwoRandom":
        if len(states) <= 2:
            return states
        idx = sorted(rng.choice(len(states), size=2, replace=False).tolist())
        return [states[i] for i in idx]
    return states


def _final_layer_states(start: object, op_func, depth: int, selector: str, rng: np.random.Generator) -> list[object]:
    current = [start]
    for _ in range(depth):
        nxt: list[object] = []
        for state in current:
            rewrites = _single_step_states(state, op_func)
            if rewrites:
                nxt.extend(_select_events(rewrites, selector, rng))
            else:
                # Treat dead-end states as terminal outcomes instead of dropping them.
                nxt.append(state)
        current = list(dict.fromkeys(nxt))
        if not current:
            break
    return current


def _multiway_layers_with_edges(
    start: object, op_func, depth: int, selector: str, rng: np.random.Generator
) -> tuple[list[list[object]], list[tuple[str, str]]]:
    layers: list[list[object]] = [[start]]
    edges: list[tuple[str, str]] = []
    current = [start]
    for _ in range(depth):
        nxt: list[object] = []
        for parent in current:
            rewrites = _single_step_states(parent, op_func)
            if rewrites:
                chosen = _select_events(rewrites, selector, rng)
            else:
                # Keep terminal parents alive as absorbing states in later layers.
                chosen = [parent]
            nxt.extend(chosen)
            p_s = _ast_expr_to_string(parent)
            for child in chosen:
                edges.append((p_s, _ast_expr_to_string(child)))
        current = list(dict.fromkeys(nxt))
        layers.append(current)
        if not current:
            break
    return layers, list(dict.fromkeys(edges))


def _state_feature_vector(expr: object) -> dict[str, float | str]:
    expr_s = _ast_expr_to_string(expr)
    neg_log_count = expr_s.count("-Log[")
    total_log_count = expr_s.count("Log[")
    pos_log_count = max(0, total_log_count - neg_log_count)
    exp_count = expr_s.count("Exp[")
    return {
        "expr": expr_s,
        "signed_log_imbalance": float(neg_log_count - pos_log_count),
        "exp_negative_log_mix": float(1 if exp_count > 0 and neg_log_count > 0 else 0),
        "total_log_count": float(total_log_count),
        "exp_count": float(exp_count),
        "repeated_negative_log": float(1 if neg_log_count >= 2 else 0),
        "negative_log_count": float(neg_log_count),
        "positive_log_count": float(pos_log_count),
    }


def _pca_project(matrix: np.ndarray, n_components: int) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, n_components))
    if x.shape[0] == 1:
        return np.zeros((1, n_components))
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    k = min(n_components, vt.shape[0])
    proj = x_centered @ vt[:k].T
    if k < n_components:
        proj = np.hstack([proj, np.zeros((proj.shape[0], n_components - k))])
    return proj


@st.cache_data(show_spinner=False)
def _compute_operator_geometry(
    seed: int,
    n_starts: int,
    leaf_min: int,
    leaf_max: int,
    depth: int,
    selector: str,
    per_operator_cap: int,
    refresh_nonce: int = 0,
) -> pd.DataFrame:
    _ = refresh_nonce
    rng = np.random.default_rng(seed)
    starts = [_random_plus_tree(int(rng.integers(leaf_min, leaf_max + 1)), rng) for _ in range(n_starts)]
    rows: list[dict[str, float | str]] = []
    operators = ["ExpMinusLog", "ExpPlusLog", "XMinusLog"]
    for op_name in operators:
        op_func = OPERATOR_LIBRARY_AST[op_name]
        final_states: list[object] = []
        for st in starts:
            final_states.extend(_final_layer_states(st, op_func, depth, selector, rng))
        # Dedupe by canonical expression string.
        unique = {}
        for st in final_states:
            unique[_ast_expr_to_string(st)] = st
        dedup_states = list(unique.values())
        if len(dedup_states) > per_operator_cap:
            idx = sorted(rng.choice(len(dedup_states), size=per_operator_cap, replace=False).tolist())
            dedup_states = [dedup_states[i] for i in idx]
        for st in dedup_states:
            row = _state_feature_vector(st)
            row["operator"] = op_name
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["operator", "expr", "pc1", "pc2", "pc3"])
    df = pd.DataFrame(rows)
    feature_cols = [
        "signed_log_imbalance",
        "exp_negative_log_mix",
        "total_log_count",
        "exp_count",
        "repeated_negative_log",
        "negative_log_count",
        "positive_log_count",
    ]
    pcs = _pca_project(df[feature_cols].to_numpy(), 3)
    df["pc1"] = pcs[:, 0]
    df["pc2"] = pcs[:, 1]
    df["pc3"] = pcs[:, 2]
    return df


@st.cache_data(show_spinner=False)
def _compute_causal_projection(
    seed: int, operator_name: str, depth: int, selector: str, leaf_count: int, refresh_nonce: int = 0
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    _ = refresh_nonce
    rng = np.random.default_rng(seed)
    start = _random_plus_tree(leaf_count, rng)
    op_func = OPERATOR_LIBRARY_AST[operator_name]
    layers, edges_s = _multiway_layers_with_edges(start, op_func, depth, selector, rng)

    rows: list[dict[str, object]] = []
    state_id: dict[str, int] = {}
    for layer_idx, states in enumerate(layers):
        for st in states:
            s = _ast_expr_to_string(st)
            if s in state_id:
                continue
            sid = len(state_id)
            state_id[s] = sid
            feat = _state_feature_vector(st)
            rows.append(
                {
                    "id": sid,
                    "expr": s,
                    "layer": layer_idx,
                    "signed_log_imbalance": feat["signed_log_imbalance"],
                    "exp_negative_log_mix": feat["exp_negative_log_mix"],
                    "total_log_count": feat["total_log_count"],
                    "exp_count": feat["exp_count"],
                    "negative_log_count": feat["negative_log_count"],
                    "positive_log_count": feat["positive_log_count"],
                    "repeated_negative_log": feat["repeated_negative_log"],
                }
            )
    if not rows:
        return pd.DataFrame(), []
    df = pd.DataFrame(rows)
    feature_cols = [
        "signed_log_imbalance",
        "exp_negative_log_mix",
        "total_log_count",
        "exp_count",
        "repeated_negative_log",
        "negative_log_count",
        "positive_log_count",
    ]
    pcs = _pca_project(df[feature_cols].to_numpy(), 3)
    df["pc1"] = pcs[:, 0]
    df["pc2"] = pcs[:, 1]
    df["pc3"] = pcs[:, 2]
    edges_idx = []
    for u_s, v_s in edges_s:
        if u_s in state_id and v_s in state_id:
            edges_idx.append((state_id[u_s], state_id[v_s]))
    return df, list(dict.fromkeys(edges_idx))


@st.cache_data(show_spinner=False)
def _compute_cone_averages(
    seed_start: int,
    n_seeds: int,
    depth: int,
    selector: str,
    leaf_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    ops = ["ExpMinusLog", "ExpPlusLog", "XMinusLog"]
    for op in ops:
        for seed in range(seed_start, seed_start + n_seeds):
            cone_df, _ = _compute_causal_projection(seed, op, depth, selector, leaf_count)
            if cone_df.empty:
                continue
            for layer, grp in cone_df.groupby("layer"):
                pts_l = grp[["pc1", "pc2", "pc3"]].to_numpy()
                centroid = pts_l.mean(axis=0)
                d = np.sqrt(((pts_l - centroid) ** 2).sum(axis=1))
                old_tuple_count = grp[
                    ["signed_log_imbalance", "exp_negative_log_mix", "total_log_count", "exp_count"]
                ].drop_duplicates().shape[0]
                rich_tuple_count = grp[
                    [
                        "signed_log_imbalance",
                        "exp_negative_log_mix",
                        "total_log_count",
                        "exp_count",
                        "negative_log_count",
                        "positive_log_count",
                        "repeated_negative_log",
                    ]
                ].drop_duplicates().shape[0]
                rows.append(
                    {
                        "seed": seed,
                        "operator": op,
                        "layer": int(layer),
                        "count": int(len(grp)),
                        "distinct_expr_strings": int(grp["expr"].nunique()),
                        "distinct_old_feature_tuples": int(old_tuple_count),
                        "distinct_rich_feature_tuples": int(rich_tuple_count),
                        "mean_radius": float(d.mean()) if len(d) else 0.0,
                        "max_radius": float(d.max()) if len(d) else 0.0,
                    }
                )
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    avg = (
        raw.groupby(["operator", "layer"], as_index=False)
        .agg(
            mean_count=("count", "mean"),
            mean_distinct_expr_strings=("distinct_expr_strings", "mean"),
            mean_distinct_old_feature_tuples=("distinct_old_feature_tuples", "mean"),
            mean_distinct_rich_feature_tuples=("distinct_rich_feature_tuples", "mean"),
            mean_radius=("mean_radius", "mean"),
            max_radius=("max_radius", "mean"),
        )
        .sort_values(["operator", "layer"])
    )
    # Normalized forms used by your later comparative plots.
    norm_frames = []
    for op, grp in avg.groupby("operator"):
        g = grp.copy()
        max_count = float(g["mean_count"].max()) if len(g) else 1.0
        max_radius = float(g["mean_radius"].max()) if len(g) else 1.0
        g["norm_count"] = g["mean_count"] / max_count if max_count > 0 else 0.0
        g["norm_radius"] = g["mean_radius"] / max_radius if max_radius > 0 else 0.0
        norm_frames.append(g)
    return pd.concat(norm_frames, ignore_index=True) if norm_frames else avg


def _compute_compression_table(cone_avg: pd.DataFrame) -> pd.DataFrame:
    if cone_avg.empty:
        return cone_avg
    out = cone_avg.copy()
    denom = pd.to_numeric(out["mean_distinct_old_feature_tuples"], errors="coerce")
    num = pd.to_numeric(out["mean_distinct_rich_feature_tuples"], errors="coerce")
    out["rich_to_old_resolution_ratio"] = np.where(denom > 0, num / denom, np.nan)
    return out


def _compute_operator_summary(compression: pd.DataFrame) -> pd.DataFrame:
    if compression.empty:
        return compression
    rows = []
    for op, grp in compression.groupby("operator"):
        g = grp.sort_values("layer").reset_index(drop=True)
        peak_count_idx = int(g["mean_count"].astype(float).idxmax())
        peak_radius_idx = int(g["mean_radius"].astype(float).idxmax())
        peak_res_idx = int(g["rich_to_old_resolution_ratio"].astype(float).idxmax())
        reconv = g[g["mean_radius"] < 0.1]
        reconv_layer = int(reconv.iloc[0]["layer"]) if len(reconv) else -1
        rows.append(
            {
                "operator": op,
                "peak_count_layer": int(g.loc[peak_count_idx, "layer"]),
                "peak_count": float(g.loc[peak_count_idx, "mean_count"]),
                "peak_radius_layer": int(g.loc[peak_radius_idx, "layer"]),
                "peak_radius": float(g.loc[peak_radius_idx, "mean_radius"]),
                "peak_resolution_layer": int(g.loc[peak_res_idx, "layer"]),
                "peak_resolution_ratio": float(g.loc[peak_res_idx, "rich_to_old_resolution_ratio"]),
                "reconvergence_layer_radius_lt_0_1": reconv_layer,
                "final_layer": int(g.iloc[-1]["layer"]),
                "final_count": float(g.iloc[-1]["mean_count"]),
                "final_radius": float(g.iloc[-1]["mean_radius"]),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _compute_agency_window_for_seed(
    seed: int, operator_name: str, max_depth: int, selector: str, leaf_count: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_expr = _random_plus_tree(leaf_count, rng)
    op_func = OPERATOR_LIBRARY_AST[operator_name]
    layers, _ = _multiway_layers_with_edges(start_expr, op_func, max_depth, selector, rng)
    rows = []
    for layer in range(0, min(max_depth, len(layers) - 1) + 1):
        layer_states = list(dict.fromkeys(layers[layer]))
        remaining_depth = max_depth - layer
        terminal_sets = [_final_layer_states(st, op_func, remaining_depth, "All", rng) for st in layer_states]
        terminal_union = list(dict.fromkeys([x for sub in terminal_sets for x in sub]))
        if len(layer_states) == 0:
            rows.append(
                {
                    "operator": operator_name,
                    "layer": layer,
                    "layer_state_count": 0,
                    "remaining_depth": remaining_depth,
                    "terminal_union_expr_count": 0,
                    "terminal_union_coarse_tuple_count": 0,
                    "terminal_union_rich_tuple_count": 0,
                    "agency_score_coarse": 0.0,
                    "agency_score_rich": 0.0,
                }
            )
            continue
        expr_count = len({_ast_expr_to_string(t) for t in terminal_union})
        coarse_tuples = {
            (
                f["signed_log_imbalance"],
                f["exp_negative_log_mix"],
                f["total_log_count"],
                f["exp_count"],
            )
            for f in (_state_feature_vector(t) for t in terminal_union)
        }
        rich_tuples = {
            (
                f["signed_log_imbalance"],
                f["exp_negative_log_mix"],
                f["total_log_count"],
                f["exp_count"],
                f["negative_log_count"],
                f["positive_log_count"],
                f["repeated_negative_log"],
            )
            for f in (_state_feature_vector(t) for t in terminal_union)
        }
        rows.append(
            {
                "operator": operator_name,
                "layer": layer,
                "layer_state_count": int(len(layer_states)),
                "remaining_depth": int(remaining_depth),
                "terminal_union_expr_count": int(expr_count),
                "terminal_union_coarse_tuple_count": int(len(coarse_tuples)),
                "terminal_union_rich_tuple_count": int(len(rich_tuples)),
                "agency_score_coarse": float(len(coarse_tuples) / max(1, len(layer_states))),
                "agency_score_rich": float(len(rich_tuples) / max(1, len(layer_states))),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _compute_agency_leverage_for_seed(
    seed: int, operator_name: str, max_depth: int, selector: str, leaf_count: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_expr = _random_plus_tree(leaf_count, rng)
    op_func = OPERATOR_LIBRARY_AST[operator_name]
    layers, _ = _multiway_layers_with_edges(start_expr, op_func, max_depth, selector, rng)

    def entropy_from_counts(counts: list[int]) -> float:
        total = float(sum(counts))
        if total <= 0:
            return 0.0
        p = np.array(counts, dtype=float) / total
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    rows = []
    for layer in range(0, min(max_depth, len(layers) - 1) + 1):
        layer_states = list(dict.fromkeys(layers[layer]))
        remaining_depth = max_depth - layer
        if not layer_states:
            rows.append(
                {
                    "operator": operator_name,
                    "layer": layer,
                    "layer_state_count": 0,
                    "rich_terminal_count": 0,
                    "coarse_terminal_count": 0,
                    "rich_entropy": 0.0,
                    "coarse_entropy": 0.0,
                    "rich_dominance": 0.0,
                    "coarse_dominance": 0.0,
                    "norm_rich_entropy": 0.0,
                    "norm_coarse_entropy": 0.0,
                    "norm_entropy_gap": 0.0,
                    "rich_to_coarse_entropy_ratio": 0.0,
                    "rich_counts_repr": "[]",
                    "coarse_counts_repr": "[]",
                }
            )
            continue

        rich_counts: list[int] = []
        coarse_counts: list[int] = []
        for st in layer_states:
            terminal_states = _final_layer_states(st, op_func, remaining_depth, "All", rng)
            if not terminal_states:
                rich_counts.append(0)
                coarse_counts.append(0)
                continue
            coarse_tuples = {
                (
                    f["signed_log_imbalance"],
                    f["exp_negative_log_mix"],
                    f["total_log_count"],
                    f["exp_count"],
                )
                for f in (_state_feature_vector(t) for t in terminal_states)
            }
            rich_tuples = {
                (
                    f["signed_log_imbalance"],
                    f["exp_negative_log_mix"],
                    f["total_log_count"],
                    f["exp_count"],
                    f["negative_log_count"],
                    f["positive_log_count"],
                    f["repeated_negative_log"],
                )
                for f in (_state_feature_vector(t) for t in terminal_states)
            }
            coarse_counts.append(int(len(coarse_tuples)))
            rich_counts.append(int(len(rich_tuples)))

        rich_total = int(sum(rich_counts))
        coarse_total = int(sum(coarse_counts))
        rich_entropy = entropy_from_counts(rich_counts)
        coarse_entropy = entropy_from_counts(coarse_counts)
        n_states = len(layer_states)
        max_entropy = float(np.log(n_states)) if n_states > 1 else 1.0
        norm_rich = rich_entropy / max_entropy if max_entropy > 0 else 0.0
        norm_coarse = coarse_entropy / max_entropy if max_entropy > 0 else 0.0

        rows.append(
            {
                "operator": operator_name,
                "layer": layer,
                "layer_state_count": int(n_states),
                "rich_terminal_count": rich_total,
                "coarse_terminal_count": coarse_total,
                "rich_entropy": float(rich_entropy),
                "coarse_entropy": float(coarse_entropy),
                "rich_dominance": float(max(rich_counts) / max(1, rich_total)),
                "coarse_dominance": float(max(coarse_counts) / max(1, coarse_total)),
                "norm_rich_entropy": float(norm_rich),
                "norm_coarse_entropy": float(norm_coarse),
                "norm_entropy_gap": float(norm_rich - norm_coarse),
                "rich_to_coarse_entropy_ratio": float(norm_rich / norm_coarse) if norm_coarse > 0 else 0.0,
                "rich_counts_repr": str(rich_counts),
                "coarse_counts_repr": str(coarse_counts),
            }
        )
    return pd.DataFrame(rows)


def _knn_edges(points: np.ndarray, k: int) -> list[tuple[int, int]]:
    if len(points) <= 1:
        return []
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    edges = set()
    for i in range(len(points)):
        nn = np.argsort(d[i])[: max(1, min(k, len(points) - 1))]
        for j in nn:
            a, b = sorted((int(i), int(j)))
            edges.add((a, b))
    return sorted(edges)


def _normal_two_tailed_p_from_z(z_score: float) -> float:
    # Normal approximation is sufficient for the replicate sample sizes used here.
    return float(math.erfc(abs(float(z_score)) / math.sqrt(2.0)))


def _safe_histogram_bins(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return 10
    std = float(np.std(vals, ddof=1))
    if std <= 1e-9:
        return 10
    # More variance -> more bins, while keeping the chart legible.
    bins = int(np.clip(np.ceil((vals.max() - vals.min()) / (0.35 * std)), 12, 60))
    return max(10, bins)


def _refresh_on_operator_change(state_key: str, operator_name: str, clear_prefixes: list[str]) -> int:
    prev_operator = st.session_state.get(state_key)
    if prev_operator != operator_name:
        st.session_state[state_key] = operator_name
        for key in list(st.session_state.keys()):
            if any(key.startswith(prefix) for prefix in clear_prefixes):
                del st.session_state[key]
        nonce_key = f"{state_key}_refresh_nonce"
        st.session_state[nonce_key] = int(st.session_state.get(nonce_key, 0)) + 1
    return int(st.session_state.get(f"{state_key}_refresh_nonce", 0))


def _parse_math_number(token: str) -> float | None:
    raw = token.strip().replace("`", "").replace("*^", "e")
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_tuple_mean_expression(text: str) -> pd.DataFrame:
    # Parses Mathematica fragments like:
    # Mean[{...} -> {{x1, y1}, {x2, y2}, ...}]
    pattern = re.compile(r"Mean\[\{(?P<tuple>.*?)\}\s*->\s*\{\{(?P<pts>.*?)\}\}\]", re.DOTALL)
    rows: list[dict[str, object]] = []
    for m in pattern.finditer(text):
        tuple_raw = m.group("tuple")
        pts_raw = m.group("pts")

        tuple_tokens = re.findall(r"[-+]?\d*\.?\d+(?:\*?\^-?\d+)?", tuple_raw)
        tuple_vals = []
        for t in tuple_tokens:
            val = _parse_math_number(t)
            if val is not None:
                tuple_vals.append(val)
        if not tuple_vals:
            continue

        pts = re.findall(r"\{([^,{}]+),([^{}]+)\}", pts_raw)
        coords: list[tuple[float, float]] = []
        for x_raw, y_raw in pts:
            x_val = _parse_math_number(x_raw)
            y_val = _parse_math_number(y_raw)
            if x_val is None or y_val is None:
                continue
            coords.append((x_val, y_val))
        if not coords:
            continue

        arr = np.array(coords, dtype=float)
        rows.append(
            {
                "tuple": "{" + ", ".join(f"{v:g}" for v in tuple_vals) + "}",
                "count": int(len(coords)),
                "mean_pc1": float(arr[:, 0].mean()),
                "mean_pc2": float(arr[:, 1].mean()),
                "raw_points": coords,
            }
        )
    return pd.DataFrame(rows)


def load_data() -> tuple[dict[str, pd.DataFrame], dict[str, bool]]:
    lifecycle = _load_csv("lifecycle_by_level.csv")
    accuracy = _load_csv("predictive_accuracy.csv")
    coarse_rich = _load_csv("coarse_rich_divergence.csv")
    embedding = _load_csv("embedding_points.csv")
    operator_traces = _load_csv("operator_trace_rows.csv")
    ablation_rows = _load_csv("ablation_rows.csv")
    l2_reps = _load_csv("l2_shuffle_replicates.csv")
    sources = {
        "lifecycle": lifecycle is not None,
        "accuracy": accuracy is not None,
        "coarse_rich": coarse_rich is not None,
        "embedding": embedding is not None,
        "operator_traces": operator_traces is not None,
        "ablation_rows": ablation_rows is not None,
        "l2_replicates": l2_reps is not None,
    }
    data = {
        "lifecycle": lifecycle if lifecycle is not None else _demo_lifecycle(),
        "accuracy": _normalize_accuracy_df(accuracy) if accuracy is not None else _demo_accuracy(),
        "coarse_rich": coarse_rich if coarse_rich is not None else _demo_coarse_rich(),
        "embedding": embedding if embedding is not None else _demo_embedding(),
        "operator_traces": operator_traces if operator_traces is not None else _paper_operator_traces(),
        "ablation_rows": ablation_rows if ablation_rows is not None else _paper_ablation_rows(),
        "l2_replicates": l2_reps if l2_reps is not None else _paper_l2_shuffle_replicates(),
    }
    data["heartfield_diff"] = _build_heartfield_diff_rows(data["operator_traces"])
    data["persistence"] = _build_persistence_profile(data["operator_traces"])
    data["single_mode_rows"] = _simulate_single_mode_model()
    data["rank_rows"] = _simulate_rank_models()
    data["three_mode_rows"] = _simulate_three_mode_model()
    return data, sources


def page_header() -> None:
    st.set_page_config(page_title="Heartfield Explorer", page_icon=":milky_way:", layout="wide")
    st.title("Heartfield Geometry Explorer")
    st.caption(
        "A step-by-step walkthrough of how the idea was developed, tested, and refined into its current form."
    )
    st.info(
        "Goal: make the full development story easy to follow, even if you are new to symbolic rewrite systems."
    )


def render_reader_guide() -> None:
    st.subheader("How to Read This Site")
    st.markdown(
        """
        1. **Start with the idea:** what problem Heartfield Geometry is trying to solve.
        2. **Track the mechanism:** how rewrite dynamics expand and then reconverge.
        3. **Check evidence:** which plots support each claim.
        4. **Read the conclusion:** why the middle-resolution view is often best.
        """
    )
    with st.expander("Plain-language glossary"):
        st.markdown(
            """
            - **Rewrite dynamics:** a rule repeatedly transforms symbolic expressions into new expressions.
            - **Expansion:** many possible next states appear ("explosion of possibility").
            - **Reconvergence:** many branches collapse into fewer effective behaviors.
            - **Resolution level:** how coarse or rich your state description is.
            - **Terminal entropy:** how diverse end outcomes remain after unfolding dynamics.
            """
        )


def render_section_bridge(title: str, explanation: str) -> None:
    st.markdown(f"### {title}")
    st.write(explanation)


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
        - **Starting idea:** symbolic systems can create rich possibilities quickly, then settle into simpler structure.
        - **Core operator intuition:** `Exp[...]` creates diversification pressure and `-Log[...]` creates reconvergence pressure.
        - **Current experimental setup:** rewrite-anywhere trees, depth target `{depth}`, selector policy `{selector}`.
        - **Main takeaway:** the most useful view is usually the **best middle resolution**, not the most detailed one.
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


def render_operator_trace_reconstruction(trace_rows: pd.DataFrame, is_real: bool) -> None:
    st.subheader("Operator Trace Reconstruction (Python)")
    st.caption(
        "Source: exported CSV from Mathematica"
        if is_real
        else "Source: reconstructed from notebook trace screenshots"
    )

    metrics = [
        ("negative_log_density", "Negative Log Density"),
        ("positive_log_density", "Positive Log Density"),
        ("exp_negative_log_mix_density", "Exp-Negative-Log Mix Density"),
        ("mean_signed_log_imbalance", "Mean Signed Log Imbalance"),
    ]
    metric_label_map = {key: label for key, label in metrics}
    metric_choice = st.selectbox(
        "Trace metric",
        options=[metric[0] for metric in metrics],
        format_func=lambda x: metric_label_map[x],
        index=0,
    )

    metric_fig = px.line(
        trace_rows.sort_values(["operator", "generation"]),
        x="generation",
        y=metric_choice,
        color="operator",
        markers=True,
        title=f"{metric_label_map[metric_choice]} by Generation",
    )
    metric_fig.update_layout(yaxis_title=metric_label_map[metric_choice], xaxis_title="Generation")
    st.plotly_chart(metric_fig, use_container_width=True)

    phase_fig = px.line(
        trace_rows.sort_values(["operator", "generation"]),
        x="mean_signed_log_imbalance",
        y="exp_negative_log_mix_density",
        color="operator",
        markers=True,
        hover_data=["generation", "state_count"],
        title="Phase Trajectory: Imbalance vs Exp-Negative-Log Mix",
    )
    phase_fig.update_layout(
        xaxis_title="Mean Signed Log Imbalance",
        yaxis_title="Exp-Negative-Log Mix Density",
    )
    st.plotly_chart(phase_fig, use_container_width=True)

    with st.expander("Trace table values used in Python"):
        st.dataframe(trace_rows.sort_values(["operator", "generation"]), use_container_width=True)


def render_heartfield_law_section(diff_rows: pd.DataFrame, persistence_rows: pd.DataFrame) -> None:
    st.subheader("Heartfield Law Diagnostics (Python)")
    st.caption("Reconstructed and fitted from operator trace rows.")

    c1, c2 = st.columns([1, 1])
    with c1:
        pers_long = persistence_rows.melt(
            id_vars=["operator"],
            value_vars=["polarity_persistence", "mix_persistence", "coherence_gap"],
            var_name="metric",
            value_name="value",
        )
        pers_fig = px.bar(
            pers_long,
            x="operator",
            y="value",
            color="metric",
            barmode="group",
            title="Persistence Profile",
        )
        st.plotly_chart(pers_fig, use_container_width=True)
    with c2:
        st.dataframe(
            persistence_rows[
                [
                    "operator",
                    "depth",
                    "peak_abs_imbalance",
                    "final_abs_imbalance",
                    "polarity_persistence",
                    "peak_mix",
                    "final_mix",
                    "mix_persistence",
                    "coherence_gap",
                ]
            ].round(6),
            use_container_width=True,
        )

    scatter_specs = [
        ("delta_n", "delta_c", "DeltaN vs DeltaC"),
        ("delta_n", "delta_p", "DeltaN vs DeltaP"),
        ("expansion_weighted_c", "delta_c", "ExpansionWeightedC vs DeltaC"),
        ("expansion_weighted_p", "delta_p", "ExpansionWeightedP vs DeltaP"),
    ]
    for x_col, y_col, title in scatter_specs:
        fig = px.scatter(
            diff_rows,
            x=x_col,
            y=y_col,
            color="operator",
            text=diff_rows["generation"].map(lambda g: f"g{int(g)}"),
            title=title,
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    line_specs = [
        ("delta_c", "Generation vs DeltaC"),
        ("delta_p", "Generation vs DeltaP"),
        ("delta_n", "Generation vs DeltaN"),
    ]
    for metric, title in line_specs:
        fig = px.line(
            diff_rows.sort_values(["operator", "generation"]),
            x="generation",
            y=metric,
            color="operator",
            markers=True,
            title=title,
        )
        st.plotly_chart(fig, use_container_width=True)

    model_rows = []
    for op, grp in diff_rows.groupby("operator"):
        fit_c = _fit_plane(grp, "delta_n", "c", "delta_c")
        fit_p = _fit_plane(grp, "delta_n", "p", "delta_p")
        model_rows.append(
            {
                "operator": op,
                "model": "DeltaC ~ DeltaN + C",
                "equation": f"{fit_c['intercept']:.4f} + {fit_c['beta1']:.4f}*DeltaN + {fit_c['beta2']:.4f}*C",
                "adj_r2": fit_c["adj_r2"],
            }
        )
        model_rows.append(
            {
                "operator": op,
                "model": "DeltaP ~ DeltaN + P",
                "equation": f"{fit_p['intercept']:.4f} + {fit_p['beta1']:.4f}*DeltaN + {fit_p['beta2']:.4f}*P",
                "adj_r2": fit_p["adj_r2"],
            }
        )
    model_df = pd.DataFrame(model_rows)
    st.markdown("**Fitted effective laws**")
    st.dataframe(model_df.round({"adj_r2": 6}), use_container_width=True)
    with st.expander("Heartfield difference rows (derived table)"):
        st.dataframe(diff_rows.round(6), use_container_width=True)


def render_update_model_section(single_mode_rows: pd.DataFrame, rank_rows: pd.DataFrame, three_mode_rows: pd.DataFrame) -> None:
    st.subheader("Heartfield Update Model (Python Reconstruction)")
    st.caption("Discrete-time simulations reconstructed from notebook equations.")

    values_long = single_mode_rows.melt(id_vars=["t"], value_vars=["n", "c", "p"], var_name="state", value_name="value")
    fig_values = px.line(values_long, x="t", y="value", color="state", markers=True, title="Single-Mode: N(t), C(t), P(t)")
    st.plotly_chart(fig_values, use_container_width=True)

    deltas_long = single_mode_rows.melt(id_vars=["t"], value_vars=["delta_n", "delta_c", "delta_p"], var_name="increment", value_name="value")
    fig_deltas = px.line(deltas_long, x="t", y="value", color="increment", markers=True, title="Single-Mode: DeltaN, DeltaC, DeltaP")
    fig_deltas.add_hline(y=0.0, line_width=1)
    st.plotly_chart(fig_deltas, use_container_width=True)

    fig_phase = px.line(single_mode_rows, x="p", y="c", markers=True, title="Single-Mode Phase: P(t) vs C(t)")
    st.plotly_chart(fig_phase, use_container_width=True)

    rank_ct = px.line(rank_rows, x="t", y="c", color="model", markers=True, title="Rank Comparison: C(t)")
    st.plotly_chart(rank_ct, use_container_width=True)

    rank_phase = px.line(rank_rows, x="p", y="c", color="model", markers=True, title="Rank Comparison Phase: P(t) vs C(t)")
    st.plotly_chart(rank_phase, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        tm_vals = three_mode_rows.melt(id_vars=["t"], value_vars=["n", "c", "p"], var_name="state", value_name="value")
        tm_fig = px.line(tm_vals, x="t", y="value", color="state", title="Three-Mode Coupled Model: N/C/P")
        st.plotly_chart(tm_fig, use_container_width=True)
    with c2:
        amp_vals = three_mode_rows.melt(id_vars=["t"], value_vars=["a1", "a2", "a3"], var_name="amplitude", value_name="value")
        amp_fig = px.line(amp_vals, x="t", y="value", color="amplitude", title="Three-Mode Internal Amplitudes")
        st.plotly_chart(amp_fig, use_container_width=True)

    with st.expander("Single-mode table (first 20 rows)"):
        st.dataframe(single_mode_rows.head(20).round(6), use_container_width=True)


def render_state_feature_geometry_section() -> None:
    st.subheader("State Feature Geometry (Python Rewrite Engine)")
    st.caption("Reconstruction of the Mathematica state-feature PCA and neighborhood geometry workflow.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_starts = st.slider("Random starts", min_value=4, max_value=40, value=12, step=2)
    with c2:
        depth = st.slider("Rewrite depth", min_value=2, max_value=6, value=4, step=1)
    with c3:
        selector = st.selectbox("Selector", ["All", "OneRandom", "TwoRandom", "Half", "First", "Last"], index=0)
    c4, c5, c6 = st.columns(3)
    with c4:
        leaf_min = st.slider("Leaf min", min_value=3, max_value=8, value=4, step=1)
    with c5:
        leaf_max = st.slider("Leaf max", min_value=4, max_value=10, value=8, step=1)
    with c6:
        per_op_cap = st.slider("Max states per operator", min_value=30, max_value=300, value=120, step=10)
    seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=12345, step=1)

    if leaf_max < leaf_min:
        st.warning("Leaf max must be >= leaf min.")
        return

    geometry_refresh_nonce = _refresh_on_operator_change(
        state_key="geometry_operator_context",
        operator_name="ExpMinusLog|ExpPlusLog|XMinusLog",
        clear_prefixes=["geometry_matrix_", "geometry_edges_"],
    )

    geo_df = _compute_operator_geometry(
        int(seed),
        int(n_starts),
        int(leaf_min),
        int(leaf_max),
        int(depth),
        selector,
        int(per_op_cap),
        int(geometry_refresh_nonce),
    )
    if geo_df.empty:
        st.info("No states generated for these settings.")
        return

    st.write(
        f"Generated **{len(geo_df)}** pooled states across ExpMinusLog / ExpPlusLog / XMinusLog."
    )

    fig2d = px.scatter(
        geo_df,
        x="pc1",
        y="pc2",
        color="operator",
        hover_data={
            "expr": True,
            "signed_log_imbalance": True,
            "exp_negative_log_mix": True,
            "total_log_count": True,
            "exp_count": True,
        },
        title="PCA 2D Cloud from State Feature Vectors",
    )
    st.plotly_chart(fig2d, use_container_width=True)

    fig3d = px.scatter_3d(
        geo_df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="operator",
        hover_data={"expr": True},
        title="PCA 3D Cloud from State Feature Vectors",
    )
    fig3d.update_traces(marker={"size": 3})
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("**Projection panel by operator**")
    op_pick = st.selectbox("Projection operator", ["ExpMinusLog", "ExpPlusLog", "XMinusLog"], index=0)
    op_df = geo_df[geo_df["operator"] == op_pick]
    if not op_df.empty:
        panel = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3"),
            horizontal_spacing=0.08,
        )
        panel.add_trace(
            go.Scatter(
                x=op_df["pc1"],
                y=op_df["pc2"],
                mode="markers",
                marker={"size": 4},
                text=op_df["expr"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        panel.add_trace(
            go.Scatter(
                x=op_df["pc1"],
                y=op_df["pc3"],
                mode="markers",
                marker={"size": 4},
                text=op_df["expr"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        panel.add_trace(
            go.Scatter(
                x=op_df["pc2"],
                y=op_df["pc3"],
                mode="markers",
                marker={"size": 4},
                text=op_df["expr"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        panel.update_layout(title=f"{op_pick} projection panel", height=380)
        panel.update_xaxes(title_text="PC1", row=1, col=1)
        panel.update_yaxes(title_text="PC2", row=1, col=1)
        panel.update_xaxes(title_text="PC1", row=1, col=2)
        panel.update_yaxes(title_text="PC3", row=1, col=2)
        panel.update_xaxes(title_text="PC2", row=1, col=3)
        panel.update_yaxes(title_text="PC3", row=1, col=3)
        st.plotly_chart(panel, use_container_width=True)

    knn_k = st.slider("k for neighborhood graph", min_value=2, max_value=12, value=6, step=1)
    points = geo_df[["pc1", "pc2", "pc3"]].to_numpy()
    edges = _knn_edges(points, knn_k)
    graph = go.Figure()
    for op_name, color in [("ExpMinusLog", "blue"), ("ExpPlusLog", "orange"), ("XMinusLog", "green")]:
        sub = geo_df[geo_df["operator"] == op_name]
        graph.add_trace(
            go.Scatter3d(
                x=sub["pc1"],
                y=sub["pc2"],
                z=sub["pc3"],
                mode="markers",
                marker={"size": 3, "color": color},
                name=op_name,
                text=sub["expr"],
                hovertemplate="%{text}<extra>" + op_name + "</extra>",
            )
        )
    for i, j in edges:
        graph.add_trace(
            go.Scatter3d(
                x=[points[i, 0], points[j, 0]],
                y=[points[i, 1], points[j, 1]],
                z=[points[i, 2], points[j, 2]],
                mode="lines",
                line={"color": "rgba(160,160,160,0.25)", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    graph.update_layout(title="3D kNN Geometry Graph", scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"})
    st.plotly_chart(graph, use_container_width=True)

    st.markdown("**Causal cone layers (single-seed)**")
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        cone_op = st.selectbox("Cone operator", ["ExpMinusLog", "ExpPlusLog", "XMinusLog"], index=0)
    with cc2:
        cone_depth = st.slider("Cone depth", min_value=2, max_value=6, value=4, step=1)
    with cc3:
        cone_leaf = st.slider("Cone leaf count", min_value=4, max_value=8, value=6, step=1)
    with cc4:
        cone_seed = st.number_input("Cone seed", min_value=0, max_value=10_000_000, value=12345, step=1)

    cone_refresh_nonce = _refresh_on_operator_change(
        state_key="cone_operator_current",
        operator_name=cone_op,
        clear_prefixes=["cone_coords_", "cone_edges_", "cone_matrix_"],
    )
    cone_df, cone_edges = _compute_causal_projection(
        int(cone_seed),
        cone_op,
        int(cone_depth),
        selector,
        int(cone_leaf),
        int(cone_refresh_nonce),
    )
    if not cone_df.empty:
        cone_fig = go.Figure()
        for layer in sorted(cone_df["layer"].unique()):
            sub_layer = cone_df[cone_df["layer"] == layer]
            cone_fig.add_trace(
                go.Scatter3d(
                    x=sub_layer["pc1"],
                    y=sub_layer["pc2"],
                    z=sub_layer["pc3"],
                    mode="markers",
                    marker={"size": 4},
                    name=f"Layer {int(layer)}",
                    text=sub_layer["expr"],
                    hovertemplate="L%{customdata}<br>%{text}<extra></extra>",
                    customdata=sub_layer["layer"],
                )
            )
        pts = cone_df.set_index("id")[["pc1", "pc2", "pc3"]]
        for u, v in cone_edges:
            if u in pts.index and v in pts.index:
                cone_fig.add_trace(
                    go.Scatter3d(
                        x=[float(pts.loc[u, "pc1"]), float(pts.loc[v, "pc1"])],
                        y=[float(pts.loc[u, "pc2"]), float(pts.loc[v, "pc2"])],
                        z=[float(pts.loc[u, "pc3"]), float(pts.loc[v, "pc3"])],
                        mode="lines",
                        line={"color": "rgba(180,180,180,0.25)", "width": 2},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        cone_fig.update_layout(
            title=f"Causal cone in PCA space ({cone_op})",
            scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
        )
        st.plotly_chart(cone_fig, use_container_width=True)

        layer_stats = []
        for layer, grp in cone_df.groupby("layer"):
            points_l = grp[["pc1", "pc2", "pc3"]].to_numpy()
            centroid = points_l.mean(axis=0)
            d = np.sqrt(((points_l - centroid) ** 2).sum(axis=1))
            layer_stats.append(
                {
                    "layer": int(layer),
                    "count": int(len(grp)),
                    "mean_pc1": float(centroid[0]),
                    "mean_pc2": float(centroid[1]),
                    "mean_pc3": float(centroid[2]),
                    "mean_radius": float(d.mean()) if len(d) else 0.0,
                    "max_radius": float(d.max()) if len(d) else 0.0,
                    "distinct_expr_strings": int(grp["expr"].nunique()),
                    "distinct_feature_tuples": int(
                        grp[
                            [
                                "signed_log_imbalance",
                                "exp_negative_log_mix",
                                "total_log_count",
                                "exp_count",
                                "negative_log_count",
                                "positive_log_count",
                                "repeated_negative_log",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),
                }
            )
        layer_stats_df = pd.DataFrame(layer_stats).sort_values("layer")
        stats_fig = px.line(
            layer_stats_df.melt(
                id_vars=["layer"],
                value_vars=["count", "mean_radius", "max_radius"],
                var_name="metric",
                value_name="value",
            ),
            x="layer",
            y="value",
            color="metric",
            markers=True,
            title="Causal-layer stats",
        )
        st.plotly_chart(stats_fig, use_container_width=True)
        st.dataframe(layer_stats_df.round(6), use_container_width=True)

    with st.expander("State feature table sample"):
        st.dataframe(geo_df.head(60), use_container_width=True)


def render_cone_averages_section() -> None:
    st.subheader("Cone Averages Across Seeds (Python)")
    st.caption("Operator/layer averages with old-vs-rich tuple distinctness and normalized cone diagnostics.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        seed_start = st.number_input("Seed start", min_value=0, max_value=10_000_000, value=12345, step=1)
    with c2:
        n_seeds = st.slider("Number of seeds", min_value=3, max_value=30, value=12, step=1)
    with c3:
        depth = st.slider("Cone avg depth", min_value=3, max_value=8, value=8, step=1)
    with c4:
        leaf_count = st.slider("Cone avg leaf count", min_value=4, max_value=8, value=6, step=1)
    selector = st.selectbox("Cone avg selector", ["All", "OneRandom", "TwoRandom", "Half", "First", "Last"], index=0)

    cone_avg = _compute_cone_averages(int(seed_start), int(n_seeds), int(depth), selector, int(leaf_count))
    if cone_avg.empty:
        st.info("No cone averages available for current settings.")
        return

    radius_fig = px.line(
        cone_avg,
        x="layer",
        y="mean_radius",
        color="operator",
        markers=True,
        title="Mean rich radius by layer",
    )
    st.plotly_chart(radius_fig, use_container_width=True)

    count_fig = px.line(
        cone_avg,
        x="layer",
        y="mean_count",
        color="operator",
        markers=True,
        title="Mean descendant count by layer",
    )
    st.plotly_chart(count_fig, use_container_width=True)

    tuple_fig = px.line(
        cone_avg,
        x="layer",
        y="mean_distinct_rich_feature_tuples",
        color="operator",
        markers=True,
        title="Mean distinct rich tuples by layer",
    )
    st.plotly_chart(tuple_fig, use_container_width=True)

    norm_long = cone_avg.melt(
        id_vars=["operator", "layer"],
        value_vars=["norm_count", "norm_radius"],
        var_name="metric",
        value_name="value",
    )
    norm_fig = px.line(
        norm_long,
        x="layer",
        y="value",
        color="operator",
        line_dash="metric",
        markers=True,
        title="Normalized cone values (count/radius)",
    )
    st.plotly_chart(norm_fig, use_container_width=True)

    compression = _compute_compression_table(cone_avg)
    ratio_fig = px.line(
        compression,
        x="layer",
        y="rich_to_old_resolution_ratio",
        color="operator",
        markers=True,
        title="Rich / Old resolution ratio",
    )
    st.plotly_chart(ratio_fig, use_container_width=True)
    ratio_series = pd.to_numeric(compression["rich_to_old_resolution_ratio"], errors="coerce").dropna()
    if not ratio_series.empty and float(ratio_series.max() - ratio_series.min()) < 1e-9:
        st.warning(
            "Rich/old resolution ratio is currently flat across this run. "
            "That means the selected settings are not separating coarse vs rich tuples."
        )

    summary = _compute_operator_summary(compression)
    st.markdown("**Operator summary**")
    st.dataframe(summary.round(6), use_container_width=True)

    st.markdown("**Compression table**")
    st.dataframe(compression.round(6), use_container_width=True)

    st.markdown("**Agency window (single seed)**")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        agency_seed = st.number_input("Agency seed", min_value=0, max_value=10_000_000, value=12345, step=1)
    with a2:
        agency_depth = st.slider("Agency max depth", min_value=4, max_value=10, value=8, step=1)
    with a3:
        agency_leaf = st.slider("Agency leaf count", min_value=4, max_value=8, value=6, step=1)
    with a4:
        agency_selector = st.selectbox("Agency selector", ["All", "OneRandom", "TwoRandom", "Half", "First", "Last"], index=0)

    agency_frames = []
    for op in ["ExpMinusLog", "ExpPlusLog", "XMinusLog"]:
        agency_frames.append(
            _compute_agency_window_for_seed(int(agency_seed), op, int(agency_depth), agency_selector, int(agency_leaf))
        )
    agency_df = pd.concat(agency_frames, ignore_index=True)
    st.dataframe(agency_df.round(6), use_container_width=True)

    rich_fig = px.line(
        agency_df,
        x="layer",
        y="agency_score_rich",
        color="operator",
        markers=True,
        title="AgencyScoreRich by layer",
    )
    st.plotly_chart(rich_fig, use_container_width=True)

    op_pick = st.selectbox("Coarse vs rich operator", ["ExpMinusLog", "ExpPlusLog", "XMinusLog"], index=0)
    op_df = agency_df[agency_df["operator"] == op_pick]
    if not op_df.empty:
        coarse_rich_fig = px.line(
            op_df.melt(
                id_vars=["layer"],
                value_vars=["agency_score_coarse", "agency_score_rich"],
                var_name="type",
                value_name="score",
            ),
            x="layer",
            y="score",
            color="type",
            markers=True,
            title=f"{op_pick}: coarse vs rich agency",
        )
        st.plotly_chart(coarse_rich_fig, use_container_width=True)


def render_terminal_entropy_section() -> None:
    st.subheader("Terminal Entropy and Pluralism")
    st.caption("Python reconstruction of rich/coarse terminal entropy, dominance, and normalized entropy gap by layer.")

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        seed_start = st.number_input("Entropy seed start", min_value=0, max_value=10_000_000, value=12345, step=1)
    with t2:
        n_seeds = st.slider("Entropy seeds", min_value=2, max_value=20, value=8, step=1)
    with t3:
        max_depth = st.slider("Entropy max depth", min_value=4, max_value=10, value=8, step=1)
    with t4:
        leaf_count = st.slider("Entropy leaf count", min_value=4, max_value=8, value=6, step=1)
    selector = st.selectbox("Entropy selector", ["All", "OneRandom", "TwoRandom", "Half", "First", "Last"], index=0)

    frames = []
    for op in ["ExpMinusLog", "ExpPlusLog", "XMinusLog"]:
        for seed in range(int(seed_start), int(seed_start) + int(n_seeds)):
            frames.append(_compute_agency_leverage_for_seed(seed, op, int(max_depth), selector, int(leaf_count)))
    if not frames:
        st.info("No entropy rows generated.")
        return
    leverage = pd.concat(frames, ignore_index=True)
    avg = (
        leverage.groupby(["operator", "layer"], as_index=False)
        .agg(
            layer_state_count=("layer_state_count", "mean"),
            rich_terminal_count=("rich_terminal_count", "mean"),
            coarse_terminal_count=("coarse_terminal_count", "mean"),
            rich_entropy=("rich_entropy", "mean"),
            coarse_entropy=("coarse_entropy", "mean"),
            rich_dominance=("rich_dominance", "mean"),
            coarse_dominance=("coarse_dominance", "mean"),
            norm_rich_entropy=("norm_rich_entropy", "mean"),
            norm_coarse_entropy=("norm_coarse_entropy", "mean"),
            norm_entropy_gap=("norm_entropy_gap", "mean"),
            rich_to_coarse_entropy_ratio=("rich_to_coarse_entropy_ratio", "mean"),
        )
        .sort_values(["operator", "layer"])
    )

    f1 = px.line(avg, x="layer", y="rich_entropy", color="operator", markers=True, title="Rich terminal entropy")
    if not avg["rich_entropy"].empty:
        f1.update_layout(yaxis_range=[0, float(avg["rich_entropy"].max()) * 1.05])
    st.plotly_chart(f1, use_container_width=True)
    f2 = px.line(avg, x="layer", y="norm_rich_entropy", color="operator", markers=True, title="Normalized rich entropy")
    f2.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(f2, use_container_width=True)
    f3 = px.line(avg, x="layer", y="rich_to_coarse_entropy_ratio", color="operator", markers=True, title="Rich/Coarse entropy ratio")
    f3.update_layout(yaxis_range=[0, max(1.05, float(avg["rich_to_coarse_entropy_ratio"].max()) * 1.05)])
    st.plotly_chart(f3, use_container_width=True)

    op_pick = st.selectbox("Entropy coarse-vs-rich operator", ["ExpMinusLog", "ExpPlusLog", "XMinusLog"], index=0)
    op_avg = avg[avg["operator"] == op_pick]
    if not op_avg.empty:
        f4 = px.line(
            op_avg.melt(
                id_vars=["layer"],
                value_vars=["coarse_entropy", "rich_entropy"],
                var_name="type",
                value_name="entropy",
            ),
            x="layer",
            y="entropy",
            color="type",
            markers=True,
            title=f"{op_pick}: coarse vs rich terminal entropy",
        )
        if not op_avg["rich_entropy"].empty and not op_avg["coarse_entropy"].empty:
            ymax = float(max(op_avg["rich_entropy"].max(), op_avg["coarse_entropy"].max()))
            f4.update_layout(yaxis_range=[0, ymax * 1.05])
        st.plotly_chart(f4, use_container_width=True)
        f5 = px.line(
            op_avg.melt(
                id_vars=["layer"],
                value_vars=["coarse_dominance", "rich_dominance"],
                var_name="type",
                value_name="dominance",
            ),
            x="layer",
            y="dominance",
            color="type",
            markers=True,
            title=f"{op_pick}: coarse vs rich dominance",
        )
        st.plotly_chart(f5, use_container_width=True)

    terminal_layer = int(leverage["layer"].max())
    terminal_width = leverage[leverage["layer"] == terminal_layer]["rich_entropy"].to_numpy(dtype=float)
    bins = _safe_histogram_bins(terminal_width)
    hist_fig = px.histogram(
        x=terminal_width,
        nbins=bins,
        title=f"Geometric Width Distribution at Terminal Layer (layer={terminal_layer})",
        labels={"x": "Rich terminal entropy", "y": "Count"},
    )
    hist_fig.update_layout(bargap=0.05)
    st.plotly_chart(hist_fig, use_container_width=True)
    gap_series = pd.to_numeric(avg["norm_entropy_gap"], errors="coerce").dropna()
    if not gap_series.empty and float(gap_series.abs().max()) < 1e-9:
        st.warning(
            "Normalized entropy gap is flat at 0 for this run. "
            "This means coarse and rich entropy are behaving identically under current settings."
        )

    st.dataframe(avg.round(6), use_container_width=True)
    with st.expander("Raw leverage rows sample"):
        st.dataframe(leverage.head(120), use_container_width=True)


def render_ablation_section(ablation_rows: pd.DataFrame, is_real: bool) -> None:
    st.subheader("Ablation by Feature Level")
    st.caption(
        "Source: Mathematica-exported ablation rows"
        if is_real
        else "Source: reconstructed ablation table from notebook dataset screenshots"
    )
    levels_order = ["L0_Coarse", "L1_Shallow", "L2_Hist", "L3_FullRich"]
    work = ablation_rows.copy()
    if "feature_level" not in work.columns:
        st.warning("Ablation table missing `feature_level` column.")
        return
    work["feature_level"] = pd.Categorical(work["feature_level"], categories=levels_order, ordered=True)

    distinct_fig = px.line(
        work.sort_values(["feature_level", "layer"]),
        x="layer",
        y="distinct_tuples",
        color="feature_level",
        markers=True,
        title="Distinct Tuples by Layer",
    )
    st.plotly_chart(distinct_fig, use_container_width=True)

    radius_fig = px.line(
        work.sort_values(["feature_level", "layer"]),
        x="layer",
        y="mean_radius",
        color="feature_level",
        markers=True,
        title="Mean Radius by Layer",
    )
    st.plotly_chart(radius_fig, use_container_width=True)

    summary = (
        work.groupby("feature_level", as_index=False)
        .agg(
            mean_distinct_tuples=("distinct_tuples", "mean"),
            mean_radius_overall=("mean_radius", "mean"),
            max_radius_overall=("max_radius", "max"),
        )
        .sort_values("feature_level")
    )
    st.dataframe(summary.round(6), use_container_width=True)
    with st.expander("Ablation rows table"):
        st.dataframe(work.sort_values(["feature_level", "layer"]), use_container_width=True)


def render_prediction_control_section(l2_reps: pd.DataFrame, is_real: bool) -> None:
    st.subheader("L2 Prediction Control Replicates")
    st.caption(
        "Source: Mathematica replicate export"
        if is_real
        else "Source: reconstructed replicate distribution from screenshot summary"
    )

    required_cols = {"replicate", "L1_Shallow", "L2_Shuffled", "L2_Hist"}
    if not required_cols.issubset(set(l2_reps.columns)):
        st.warning("Replicate table missing required columns.")
        return
    reps = l2_reps.copy()
    for c in ["L1_Shallow", "L2_Shuffled", "L2_Hist"]:
        reps[c] = pd.to_numeric(reps[c], errors="coerce")
    reps = reps.dropna(subset=["L1_Shallow", "L2_Shuffled", "L2_Hist"])

    summary = pd.DataFrame(
        [
            {"feature_level": "L1_Shallow", "mean_accuracy": reps["L1_Shallow"].mean(), "std_accuracy": reps["L1_Shallow"].std(ddof=1), "n": len(reps)},
            {"feature_level": "L2_Shuffled", "mean_accuracy": reps["L2_Shuffled"].mean(), "std_accuracy": reps["L2_Shuffled"].std(ddof=1), "n": len(reps)},
            {"feature_level": "L2_Hist", "mean_accuracy": reps["L2_Hist"].mean(), "std_accuracy": reps["L2_Hist"].std(ddof=1), "n": len(reps)},
        ]
    )
    bar = px.bar(
        summary,
        x="feature_level",
        y="mean_accuracy",
        error_y="std_accuracy",
        text="mean_accuracy",
        title="Prediction accuracy by feature level (L2 control set)",
    )
    bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    bar.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(bar, use_container_width=True)

    # Lightweight significance diagnostics (normal approximation to two-tailed p-values).
    comparisons = [
        ("L2_Hist", "L2_Shuffled", "L2_Hist vs L2_Shuffled"),
        ("L1_Shallow", "L2_Shuffled", "L1_Shallow vs L2_Shuffled"),
    ]
    stat_rows = []
    for lhs, rhs, label in comparisons:
        lhs_vals = reps[lhs].to_numpy(dtype=float)
        rhs_vals = reps[rhs].to_numpy(dtype=float)
        lhs_var = float(np.var(lhs_vals, ddof=1))
        rhs_var = float(np.var(rhs_vals, ddof=1))
        se = math.sqrt((lhs_var / max(1, len(lhs_vals))) + (rhs_var / max(1, len(rhs_vals))))
        z_score = (float(lhs_vals.mean()) - float(rhs_vals.mean())) / se if se > 0 else 0.0
        p_value = _normal_two_tailed_p_from_z(z_score)
        stat_rows.append(
            {
                "comparison": label,
                "mean_delta": float(lhs_vals.mean() - rhs_vals.mean()),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
        )
    stats_df = pd.DataFrame(stat_rows)

    c1, c2 = st.columns(2)
    with c1:
        row = stats_df.iloc[0]
        status = "Significant" if bool(row["significant"]) else "Not significant"
        color = "#16a34a" if bool(row["significant"]) else "#dc2626"
        st.markdown(
            f"<div style='font-weight:600'>{row['comparison']}</div>"
            f"<div style='color:{color};font-size:1.05rem'>{status} (p={row['p_value']:.4f})</div>",
            unsafe_allow_html=True,
        )
    with c2:
        row = stats_df.iloc[1]
        status = "Significant" if bool(row["significant"]) else "Not significant"
        color = "#16a34a" if bool(row["significant"]) else "#dc2626"
        st.markdown(
            f"<div style='font-weight:600'>{row['comparison']}</div>"
            f"<div style='color:{color};font-size:1.05rem'>{status} (p={row['p_value']:.4f})</div>",
            unsafe_allow_html=True,
        )

    st.dataframe(summary.round(6), use_container_width=True)
    st.dataframe(stats_df.assign(p_value=stats_df["p_value"].map(lambda x: f"{x:.4f}")), use_container_width=True)
    with st.expander("Replicate table (20 runs)"):
        st.dataframe(reps.round(6), use_container_width=True)


def render_tuple_ladder_section() -> None:
    st.subheader("Tuple Ladder / Band Reconstruction")
    st.caption("Paste Mathematica tuple-mean association text to reconstruct PC1/PC2 ladder plots in Python.")

    default_text = ""
    tuple_text_file = DATA_DIR / "tuple_mean_expression.txt"
    if tuple_text_file.exists():
        default_text = tuple_text_file.read_text(encoding="utf-8")

    tuple_text = st.text_area(
        "Paste Mathematica expression",
        value=default_text,
        height=220,
        help="Expected pattern: Mean[{tuple} -> {{pc1, pc2}, ...}] repeated inside an Association.",
    )
    if not tuple_text.strip():
        st.info("Paste the Mathematica expression to generate tuple ladder plots.")
        return

    tuple_df = _parse_tuple_mean_expression(tuple_text)
    if tuple_df.empty:
        st.warning("No tuple-point blocks were parsed. Check that the expression includes Mean[{...}->{ {x,y},... }].")
        return

    scatter = px.scatter(
        tuple_df,
        x="mean_pc1",
        y="mean_pc2",
        size="count",
        hover_data={"tuple": True, "count": True},
        title="Tuple Mean Points (PC1 vs PC2)",
    )
    st.plotly_chart(scatter, use_container_width=True)

    bucket_df = tuple_df.copy()
    bucket_df["band"] = (np.round(bucket_df["mean_pc2"] / 0.5) * 0.5).astype(float)
    reps = bucket_df.sort_values("count", ascending=False).groupby("band", as_index=False).head(5)
    summary = (
        bucket_df.sort_values("count", ascending=False)
        .groupby("band", as_index=False)
        .first()[["band", "tuple", "count", "mean_pc1", "mean_pc2"]]
        .rename(columns={"tuple": "top_tuple", "count": "top_count"})
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Band representatives (top 5 by count)**")
        st.dataframe(reps[["band", "tuple", "count", "mean_pc1", "mean_pc2"]], use_container_width=True)
    with c2:
        st.markdown("**Band summary**")
        st.dataframe(summary, use_container_width=True)


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
            - `operator_trace_rows.csv`: `operator`, `generation`, `state_count`, `negative_log_density`, `positive_log_density`, `repeated_negative_log_density`, `exp_negative_log_mix_density`, `mean_signed_log_imbalance`
            """
        )


def render_mathematica_evidence() -> None:
    st.subheader("Mathematica Evidence Gallery")
    st.caption("Direct captures from the notebook exploration, kept as traceability artifacts.")

    repo_images = []
    if EVIDENCE_DIR.exists():
        repo_images = sorted(
            [*EVIDENCE_DIR.glob("*.png"), *EVIDENCE_DIR.glob("*.jpg"), *EVIDENCE_DIR.glob("*.jpeg")],
            key=lambda p: p.name,
        )

    if not repo_images:
        st.info("No evidence images found yet. Drop files into `assets/evidence/` to make them deployable on Streamlit Cloud.")
        return

    for idx, image_path in enumerate(repo_images, start=1):
        st.image(str(image_path), caption=f"Notebook evidence {idx}: {image_path.name}", use_container_width=True)


def main() -> None:
    page_header()
    data, sources = load_data()
    depth, selector = sidebar_controls(sum(sources.values()))
    render_reader_guide()

    st.divider()
    st.header("Phase 1 - Idea and First Principles")
    render_section_bridge(
        "What was the original idea?",
        "This section introduces the hypothesis: symbolic rewrite systems expand rapidly, then reconverge, and that reconvergence depends on observation scale.",
    )
    render_story(depth, selector)

    st.divider()
    st.header("Phase 2 - Early Dynamics and Evidence")
    render_section_bridge(
        "What happens as the system evolves?",
        "These plots show growth, spread, and first evidence that behavior changes with representation level.",
    )
    render_lifecycle(data["lifecycle"], sources["lifecycle"])
    render_accuracy(data["accuracy"], sources["accuracy"])
    render_operator_trace_reconstruction(data["operator_traces"], sources["operator_traces"])

    st.divider()
    st.header("Phase 3 - Deriving the Heartfield Law")
    render_section_bridge(
        "How did the idea become a model?",
        "Here the app converts trace behavior into fitted relationships and simulation updates, moving from concept to explicit equations.",
    )
    render_heartfield_law_section(data["heartfield_diff"], data["persistence"])
    render_update_model_section(data["single_mode_rows"], data["rank_rows"], data["three_mode_rows"])

    st.divider()
    st.header("Phase 4 - Resolution Tests and Controls")
    render_section_bridge(
        "Which feature level actually works best?",
        "Ablation and control analyses compare coarse, shallow, historical, and fully rich views to test predictive usefulness.",
    )
    render_ablation_section(data["ablation_rows"], sources["ablation_rows"])
    render_prediction_control_section(data["l2_replicates"], sources["l2_replicates"])

    st.divider()
    st.header("Phase 5 - Geometry of Reconvergence")
    render_section_bridge(
        "Where do trajectories go in state-space?",
        "These sections visualize neighborhoods, cones, and entropy so you can see reconvergence geometry rather than only reading summary statistics.",
    )
    render_state_feature_geometry_section()
    render_cone_averages_section()
    render_terminal_entropy_section()
    render_tuple_ladder_section()

    st.divider()
    st.header("Phase 6 - Final Synthesis")
    render_section_bridge(
        "What does this mean in practical terms?",
        "Final summaries combine divergence, 3D embedding, and resolution recommendation into a clear statement of the current model state.",
    )
    render_coarse_vs_rich(data["coarse_rich"], sources["coarse_rich"])
    render_embedding_3d(data["embedding"], sources["embedding"])
    render_resolution_selector(data["accuracy"])

    st.divider()
    st.header("Traceability and Reproducibility")
    render_section_bridge(
        "How do we verify this development path?",
        "The evidence gallery and schema contract make it clear how notebook artifacts map to the Python app and deployment data.",
    )
    render_mathematica_evidence()
    render_data_contract()


if __name__ == "__main__":
    main()
