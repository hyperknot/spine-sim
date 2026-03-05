#!/usr/bin/env -S uv run
"""
sictest.py

Purpose
-------
Find whether any sic_* parameterization has a relationship with peak_T12L1_kN,
and generate scatter plots:

  - Rank SIC columns by cross-validated prediction of peak_T12L1_kN from SIC
    (simple linear regression).
  - Plot relationship for each SIC column:
      x-axis: peak_T12L1_kN (kN)
      y-axis: SIC value (that column)

Outputs (default out-dir: <subdir>/sic_analysis_out/)
-------------------------------------------
- sic_ranked_by_cv.csv                       : ranking of unique SIC columns
- sic_exact_duplicate_groups.json            : exact duplicates (often indicates window saturation)
- plots_per_sic/*.png                        : one plot per SIC column (including duplicates)
- best_sic_plot.png                          : highlighted plot for best-ranked SIC column
- input_files_sorted_by_sic_asc.csv          : input files with kN + best-SIC value, sorted by SIC asc

Dependencies
------------
pip install pandas numpy scikit-learn scipy statsmodels matplotlib
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# Utilities
# -----------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def choose_cv(n_rows: int, loo_threshold: int, kfold_splits: int, seed: int):
    if n_rows < loo_threshold:
        return LeaveOneOut(), f"LeaveOneOut(n={n_rows})"
    return KFold(n_splits=kfold_splits, shuffle=True, random_state=seed), f"KFold(n_splits={kfold_splits})"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def safe_filename(name: str) -> str:
    """
    Make a filesystem-safe filename fragment.
    """
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def to_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def resolve_subdir_path(subdir: str) -> str:
    """
    Resolve a user-provided subdir to an existing directory.
    Accepts either:
      - a direct path (e.g. output/advance), or
      - a short name under output/ (e.g. advance).
    """
    direct = subdir
    under_output = os.path.join("output", subdir)
    if os.path.isdir(direct):
        return direct
    return under_output


def describe_column_mismatch(expected: Sequence[str], actual: Sequence[str]) -> Optional[str]:
    """
    Return a concise mismatch description if columns are not exactly equal.
    Exact equality means same names, same count, and same order.
    """
    expected_cols = list(expected)
    actual_cols = list(actual)
    if actual_cols == expected_cols:
        return None

    problems: List[str] = [
        f"expected {len(expected_cols)} columns, got {len(actual_cols)}",
    ]

    expected_set = set(expected_cols)
    actual_set = set(actual_cols)

    missing = [c for c in expected_cols if c not in actual_set]
    extra = [c for c in actual_cols if c not in expected_set]

    if missing:
        preview = missing[:10]
        suffix = " ..." if len(missing) > 10 else ""
        problems.append(f"missing columns: {preview}{suffix}")
    if extra:
        preview = extra[:10]
        suffix = " ..." if len(extra) > 10 else ""
        problems.append(f"extra columns: {preview}{suffix}")

    # If names are the same set, report where the order first diverges.
    if not missing and not extra:
        for idx, (exp, got) in enumerate(zip(expected_cols, actual_cols)):
            if exp != got:
                problems.append(
                    f"column order mismatch at index {idx}: expected '{exp}', got '{got}'"
                )
                break

    return "; ".join(problems)


def write_input_files_sorted_by_sic(
    df_num: pd.DataFrame,
    subdir_paths: Sequence[str],
    kn_col: str,
    sic_col: str,
    out_dir: str,
    filename_col: str = "filename",
    source_subdir_col: Optional[str] = None,
) -> Optional[str]:
    """
    Save input files with kN + SIC values, sorted by SIC ascending.
    In single-subdir mode output columns are: input_file, kn_value, sic_column, sic_value.
    In mix mode output adds a leading subdir column.
    Returns output CSV path, or None if required paths/columns are missing.
    """
    if filename_col not in df_num.columns:
        eprint(f"Filename column not found (skipping input-file SIC export): {filename_col}")
        return None
    if kn_col not in df_num.columns:
        eprint(f"kN column not found (skipping input-file SIC export): {kn_col}")
        return None
    if sic_col not in df_num.columns:
        eprint(f"SIC column not found (skipping input-file SIC export): {sic_col}")
        return None

    subdir_names = [os.path.basename(os.path.normpath(p)) for p in subdir_paths]
    if not subdir_names:
        eprint("No subdirs provided (skipping input-file SIC export).")
        return None

    lookup_cols = [filename_col, kn_col, sic_col]
    if source_subdir_col is not None:
        if source_subdir_col not in df_num.columns:
            eprint(f"Source subdir column not found (skipping input-file SIC export): {source_subdir_col}")
            return None
        lookup_cols = [source_subdir_col, *lookup_cols]

    lookup = df_num[lookup_cols].copy()
    if source_subdir_col is not None:
        lookup["subdir"] = lookup[source_subdir_col].astype(str)
    else:
        if len(set(subdir_names)) != 1:
            eprint("Multiple subdirs provided but no source-subdir column available (skipping input-file SIC export).")
            return None
        lookup["subdir"] = subdir_names[0]

    lookup[filename_col] = lookup[filename_col].astype(str)
    lookup = lookup.rename(
        columns={
            filename_col: "input_file",
            kn_col: "kn_value",
            sic_col: "sic_value",
        }
    )
    lookup["kn_value"] = pd.to_numeric(lookup["kn_value"], errors="coerce")
    lookup["sic_value"] = pd.to_numeric(lookup["sic_value"], errors="coerce")
    lookup = lookup.drop_duplicates(subset=["subdir", "input_file"], keep="first")
    lookup = lookup[["subdir", "input_file", "kn_value", "sic_value"]]

    input_rows: List[dict] = []
    for subdir_name in sorted(set(subdir_names)):
        input_dir = os.path.join("input", subdir_name)
        if not os.path.isdir(input_dir):
            eprint(f"Input directory not found (skipping this subdir): {input_dir}")
            continue
        input_files = sorted(
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith(".")
        )
        input_rows.extend({"subdir": subdir_name, "input_file": f} for f in input_files)

    if not input_rows:
        eprint("No input files found (skipping input-file SIC export).")
        return None

    out = pd.DataFrame(input_rows)
    out = out.merge(lookup, on=["subdir", "input_file"], how="left")
    out["sic_column"] = sic_col

    is_mix = len(set(subdir_names)) > 1
    if is_mix:
        out = out[["subdir", "input_file", "kn_value", "sic_column", "sic_value"]]
        out = out.sort_values(
            ["sic_value", "subdir", "input_file"], ascending=[True, True, True], na_position="last"
        ).reset_index(drop=True)
    else:
        out = out[["input_file", "kn_value", "sic_column", "sic_value"]]
        out = out.sort_values(["sic_value", "input_file"], ascending=[True, True], na_position="last").reset_index(drop=True)

    out_path = os.path.join(out_dir, "input_files_sorted_by_sic_asc.csv")
    out.to_csv(out_path, index=False)
    return out_path


# -----------------------------
# Duplicate detection
# -----------------------------

def _hash_series(s: pd.Series) -> int:
    """
    Fast hash for grouping potential duplicates; verify equality afterwards.
    """
    h = pd.util.hash_pandas_object(s, index=False)
    return int(h.sum())


def find_exact_duplicate_columns(df_num: pd.DataFrame, cols: Sequence[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Returns:
      - dup_groups: representative -> [all columns identical], including rep
      - rep_for_col: column -> representative
    """
    hashes: Dict[int, List[str]] = {}
    for c in cols:
        hashes.setdefault(_hash_series(df_num[c]), []).append(c)

    dup_groups: Dict[str, List[str]] = {}
    rep_for_col: Dict[str, str] = {}

    for _, bucket in hashes.items():
        if len(bucket) == 1:
            c = bucket[0]
            dup_groups[c] = [c]
            rep_for_col[c] = c
            continue

        groups: List[List[str]] = []
        for c in bucket:
            placed = False
            for g in groups:
                if df_num[c].equals(df_num[g[0]]):
                    g.append(c)
                    placed = True
                    break
            if not placed:
                groups.append([c])

        for g in groups:
            rep = sorted(g)[0]
            g_sorted = sorted(g)
            dup_groups[rep] = g_sorted
            for c in g_sorted:
                rep_for_col[c] = rep

    return dup_groups, rep_for_col


def collapse_to_unique_columns(cols: Sequence[str], dup_groups: Dict[str, List[str]]) -> List[str]:
    reps = sorted(dup_groups.keys())
    colset = set(cols)
    return [r for r in reps if r in colset]


# -----------------------------
# Scoring / ranking
# -----------------------------

@dataclass(frozen=True)
class ColumnScore:
    col: str
    n: int
    rmse_cv: float
    mae_cv: float
    r2_cv: float
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float


def evaluate_single_predictor(
    df_num: pd.DataFrame,
    target: str,
    xcol: str,
    cv,
    min_n: int,
) -> Optional[ColumnScore]:
    sub = df_num[[target, xcol]].dropna()
    n = len(sub)
    if n < min_n:
        return None

    y = sub[target].to_numpy()
    X = sub[[xcol]].to_numpy()

    pear_r, pear_p = pearsonr(sub[xcol], sub[target])
    spear_r, spear_p = spearmanr(sub[xcol], sub[target])

    model = make_pipeline(StandardScaler(), LinearRegression())
    yhat = cross_val_predict(model, X, y, cv=cv)

    return ColumnScore(
        col=xcol,
        n=n,
        rmse_cv=rmse(y, yhat),
        mae_cv=float(mean_absolute_error(y, yhat)),
        r2_cv=float(r2_score(y, yhat)),
        pearson_r=float(pear_r),
        pearson_p=float(pear_p),
        spearman_r=float(spear_r),
        spearman_p=float(spear_p),
    )


def rank_sic_columns(
    df_num: pd.DataFrame,
    target: str,
    sic_cols_unique: Sequence[str],
    cv,
    min_n: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for c in sic_cols_unique:
        score = evaluate_single_predictor(df_num, target, c, cv=cv, min_n=min_n)
        if score is None:
            continue
        rows.append(score.__dict__)

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    res["pearson_p_fdr"] = multipletests(res["pearson_p"].to_numpy(), method="fdr_bh")[1]
    res["spearman_p_fdr"] = multipletests(res["spearman_p"].to_numpy(), method="fdr_bh")[1]

    res = res.sort_values(["rmse_cv", "r2_cv"], ascending=[True, False]).reset_index(drop=True)
    return res


# -----------------------------
# Plotting
# -----------------------------

def _fit_line_y_on_x(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Fit y = a + b*x using least squares. Returns (a, b) or None.
    """
    if len(x) < 2:
        return None
    # polyfit can warn on degenerate cases; handle exceptions
    try:
        b, a = np.polyfit(x, y, deg=1)  # returns slope, intercept
        return float(a), float(b)
    except Exception:
        return None


def plot_one_sic(
    df_num: pd.DataFrame,
    target: str,
    sic_col: str,
    out_path: str,
    title_extra: Optional[str] = None,
    highlight: bool = False,
    dpi: int = 160,
) -> bool:
    """
    Scatter plot: x = target (kN), y = sic_col (SIC).
    Returns True if plot created, else False.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as ex:
        eprint(f"[plot] matplotlib not available, cannot plot. ({ex})")
        return False

    sub = df_num[[target, sic_col]].dropna()
    if len(sub) < 2:
        return False

    x = sub[target].to_numpy()
    y = sub[sic_col].to_numpy()

    # Correlation for annotation
    try:
        r, p = pearsonr(sub[target], sub[sic_col])
    except Exception:
        r, p = np.nan, np.nan

    # Fit line (SIC as function of kN)
    fit = _fit_line_y_on_x(x, y)

    plt.figure(figsize=(6.2, 4.4))
    if highlight:
        plt.scatter(x, y, s=30, c="crimson", alpha=0.9, edgecolors="none")
    else:
        plt.scatter(x, y, s=24, alpha=0.85, edgecolors="none")

    if fit is not None:
        a, b = fit
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        ys = a + b * xs
        plt.plot(xs, ys, linewidth=2.0, alpha=0.9)

    plt.xlabel(f"{target} (kN)")
    plt.ylabel(sic_col)

    t = f"{sic_col} vs {target} (n={len(sub)}, Pearson r={r:.4f})"
    if title_extra:
        t += f"\n{title_extra}"
    plt.title(t)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return True


def plot_all_sic_columns(
    df_num: pd.DataFrame,
    target: str,
    sic_cols_all: Sequence[str],
    out_dir: str,
    best_col: Optional[str],
    scores_by_rep: Dict[str, dict],
    rep_for_col: Dict[str, str],
    dpi: int,
) -> Tuple[int, int]:
    """
    Creates one PNG per SIC column:
      plots_per_sic/peak_vs_<sic_col>.png

    Returns (created, skipped).
    """
    ensure_dir(out_dir)
    created = 0
    skipped = 0

    for c in sic_cols_all:
        rep = rep_for_col.get(c, c)
        score = scores_by_rep.get(rep)

        title_extra = None
        if score is not None:
            # These scores are for predicting peak from SIC (reverse direction from plot),
            # but are still useful as a relationship strength indicator.
            title_extra = f"CV(RMSE)={score['rmse_cv']:.6g}, CV(R2)={score['r2_cv']:.6g}"

        out_path = os.path.join(out_dir, f"peak_vs_{safe_filename(c)}.png")
        ok = plot_one_sic(
            df_num=df_num,
            target=target,
            sic_col=c,
            out_path=out_path,
            title_extra=title_extra,
            highlight=(best_col == c),
            dpi=dpi,
        )
        if ok:
            created += 1
        else:
            skipped += 1

    return created, skipped


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Rank sic_* columns for relationship with peak_T12L1_kN and plot x=kN, y=SIC for each SIC column."
    )
    ap.add_argument(
        "subdir",
        nargs="?",
        help="Subdir containing summary.csv, like 'advance' or 'output/advance'.",
    )
    ap.add_argument(
        "--mix",
        nargs="+",
        metavar="SUBDIR",
        default=None,
        help=(
            "Mix rows from multiple subdirs by concatenating each <subdir>/summary.csv. "
            "All columns must match exactly across files."
        ),
    )
    ap.add_argument(
        "--mix-all",
        action="store_true",
        help="Mix all subdirs under output/ that contain summary.csv.",
    )
    ap.add_argument("--csv", default=None, help="Path to CSV file (default: <subdir>/summary.csv).")
    ap.add_argument("--target", default="peak_T12L1_kN", help="Target column (default: peak_T12L1_kN).")
    ap.add_argument(
        "--min-force-kn",
        type=float,
        default=None,
        help=(
            "If set, keep only rows where force column >= this kN value. "
            "Uses peak_T12L1_kN when present, otherwise uses --target."
        ),
    )
    ap.add_argument("--sic-prefix", default="sic_", help="Prefix for SIC columns (default: sic_).")
    ap.add_argument("--min-n", type=int, default=5, help="Minimum non-NaN rows required to score a column (default: 5).")

    ap.add_argument("--loo-threshold", type=int, default=30, help="Use LOO-CV if n < this (default: 30).")
    ap.add_argument("--kfold-splits", type=int, default=5, help="KFold splits if not using LOO (default: 5).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for KFold shuffle (default: 0).")

    ap.add_argument("--top", type=int, default=15, help="How many top SIC columns to print (default: 15).")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <subdir>/sic_analysis_out).")

    # Plotting behavior: enabled by default; allow disabling
    ap.add_argument("--no-plots", action="store_true", help="Disable generating plots per SIC column.")
    ap.add_argument("--plot-dpi", type=int, default=160, help="DPI for saved PNGs (default: 160).")

    args = ap.parse_args()

    mix_subdirs: List[str] = list(args.mix or [])
    if args.subdir and mix_subdirs:
        eprint("Use either <subdir> or --mix SUBDIR [SUBDIR ...], not both.")
        sys.exit(2)
    if args.subdir and args.mix_all:
        eprint("Use either <subdir> or --mix-all, not both.")
        sys.exit(2)
    if mix_subdirs and args.mix_all:
        eprint("Use either --mix SUBDIR [SUBDIR ...] or --mix-all, not both.")
        sys.exit(2)

    if args.mix_all:
        output_root = "output"
        if not os.path.isdir(output_root):
            eprint("Directory not found: output")
            sys.exit(2)
        mix_subdirs = sorted(
            d
            for d in os.listdir(output_root)
            if os.path.isdir(os.path.join(output_root, d))
            and os.path.exists(os.path.join(output_root, d, "summary.csv"))
        )
        if not mix_subdirs:
            eprint("No subdirs with summary.csv found under output/.")
            sys.exit(2)
        print(f"Mix-all mode: found {len(mix_subdirs)} subdirs under output/")

    if not args.subdir and not mix_subdirs:
        eprint("Please provide <subdir>, --mix SUBDIR [SUBDIR ...], or --mix-all.")
        sys.exit(2)

    if mix_subdirs and args.csv:
        eprint("--csv cannot be used with --mix/--mix-all. Mixed mode reads <subdir>/summary.csv for each subdir.")
        sys.exit(2)

    source_label: str
    source_csv_label: str
    out_dir: str
    input_export_subdir_paths: List[str] = []

    if mix_subdirs:
        resolved_subdirs: List[str] = []
        csv_paths: List[str] = []
        frames: List[pd.DataFrame] = []
        expected_columns: Optional[List[str]] = None
        row_counts: List[int] = []

        for subdir in mix_subdirs:
            subdir_path = resolve_subdir_path(subdir)
            if not os.path.isdir(subdir_path):
                eprint(f"Subdir not found: {subdir}")
                sys.exit(2)

            csv_path = os.path.join(subdir_path, "summary.csv")
            if not os.path.exists(csv_path):
                eprint(f"CSV not found: {csv_path}")
                sys.exit(2)

            part = pd.read_csv(csv_path).copy()
            part_cols = list(part.columns)

            if expected_columns is None:
                expected_columns = part_cols
            else:
                mismatch = describe_column_mismatch(expected_columns, part_cols)
                if mismatch is not None:
                    eprint("Column mismatch in --mix inputs.")
                    eprint(f"Reference CSV: {csv_paths[0]}")
                    eprint(f"Current CSV  : {csv_path}")
                    eprint(f"Details      : {mismatch}")
                    sys.exit(2)

            resolved_subdirs.append(subdir_path)
            csv_paths.append(csv_path)
            part["__source_subdir"] = os.path.basename(os.path.normpath(subdir_path))
            frames.append(part)
            row_counts.append(len(part))

        df = pd.concat(frames, axis=0, ignore_index=True)
        mix_name = "__".join(safe_filename(os.path.basename(p.rstrip(os.sep)) or p) for p in resolved_subdirs)
        default_out_base = os.path.join("output", f"mix_{mix_name}")
        out_dir = args.out_dir or os.path.join(default_out_base, "sic_analysis_out")
        source_label = " + ".join(resolved_subdirs)
        source_csv_label = " + ".join(csv_paths)
        input_export_subdir_paths = resolved_subdirs

        print("Mix mode: enabled")
        for subdir_path, csv_path, rows in zip(resolved_subdirs, csv_paths, row_counts):
            print(f"  - {subdir_path}: {rows} rows ({csv_path})")
    else:
        assert args.subdir is not None
        subdir_path = resolve_subdir_path(args.subdir)
        if not os.path.isdir(subdir_path):
            eprint(f"Subdir not found: {args.subdir}")
            sys.exit(2)

        csv_path = args.csv or os.path.join(subdir_path, "summary.csv")
        if not os.path.exists(csv_path):
            eprint(f"CSV not found: {csv_path}")
            sys.exit(2)

        df = pd.read_csv(csv_path)
        out_dir = args.out_dir or os.path.join(subdir_path, "sic_analysis_out")
        source_label = subdir_path
        source_csv_label = csv_path
        input_export_subdir_paths = [subdir_path]

    if args.target not in df.columns:
        eprint(f"Target column not found: {args.target}")
        sys.exit(2)

    force_col = "peak_T12L1_kN" if "peak_T12L1_kN" in df.columns else args.target

    sic_cols_all = [c for c in df.columns if c.startswith(args.sic_prefix)]
    if not sic_cols_all:
        eprint(f"No SIC columns found with prefix '{args.sic_prefix}'.")
        sys.exit(2)

    ensure_dir(out_dir)

    # Numeric coercion for target + force column + all SIC cols
    numeric_cols = list(dict.fromkeys([args.target, force_col, *sic_cols_all]))
    df_num = to_numeric_frame(df, numeric_cols)

    n_total = len(df_num)
    n_target_non_nan = int(df_num[args.target].notna().sum())

    if args.min_force_kn is not None:
        keep_mask = df_num[force_col] >= args.min_force_kn
        keep_count = int(keep_mask.fillna(False).sum())
        drop_count = n_total - keep_count
        df_num = df_num.loc[keep_mask.fillna(False)].reset_index(drop=True)
        n_total = len(df_num)
        n_target_non_nan = int(df_num[args.target].notna().sum())
        print(
            f"Applied force filter: {force_col} >= {args.min_force_kn:g} kN "
            f"(kept {keep_count}, dropped {drop_count})"
        )
    print(f"Source: {source_label}")
    print(f"Loaded {source_csv_label}")
    print(f"Rows total: {n_total}")
    print(f"Rows with non-NaN {args.target}: {n_target_non_nan}")
    print(f"SIC columns found: {len(sic_cols_all)}")

    # Choose CV strategy based on available target rows
    cv, cv_name = choose_cv(n_target_non_nan, loo_threshold=args.loo_threshold, kfold_splits=args.kfold_splits, seed=args.seed)
    print(f"CV strategy: {cv_name}")

    # Duplicate detection (useful debug info; doesn't change plotting unless you want it to)
    dup_groups, rep_for_col = find_exact_duplicate_columns(df_num, sic_cols_all)
    unique_sic_cols = collapse_to_unique_columns(sic_cols_all, dup_groups)

    dup_only = {rep: cols for rep, cols in dup_groups.items() if len(cols) > 1}
    with open(os.path.join(out_dir, "sic_exact_duplicate_groups.json"), "w", encoding="utf-8") as f:
        json.dump(dup_only, f, indent=2)

    print(f"Unique SIC columns after removing exact duplicates: {len(unique_sic_cols)}")
    if dup_only:
        largest = sorted(dup_only.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
        print("Largest exact-duplicate SIC groups (showing up to 10):")
        for rep, cols in largest:
            preview = cols[:10]
            suffix = " ..." if len(cols) > 10 else ""
            print(f"  {rep} -> {len(cols)} columns: {preview}{suffix}")
    else:
        print("No exact duplicate SIC columns detected.")

    # Rank unique SIC columns by predicting peak from SIC
    ranked = rank_sic_columns(df_num, args.target, unique_sic_cols, cv=cv, min_n=args.min_n)
    ranked_path = os.path.join(out_dir, "sic_ranked_by_cv.csv")
    ranked.to_csv(ranked_path, index=False)
    print(f"Saved: {ranked_path}")

    if ranked.empty:
        print("No SIC columns could be scored (possibly too few non-NaN rows).")
        sys.exit(0)

    # Print top ranking table
    show_cols = ["col", "n", "rmse_cv", "mae_cv", "r2_cv", "pearson_r", "pearson_p_fdr"]
    print(f"Top {min(args.top, len(ranked))} SIC columns by CV RMSE:")
    print(ranked[show_cols].head(args.top).to_string(index=False))

    best_rep_col = str(ranked.loc[0, "col"])
    print("\nBest SIC column (by CV RMSE, unique representatives):")
    print({"col": best_rep_col, "rmse_cv": float(ranked.loc[0, 'rmse_cv']), "r2_cv": float(ranked.loc[0, 'r2_cv'])})

    # For plotting: if the "best rep" has duplicates, pick the rep itself for the main best plot,
    # but we also generate per-column plots anyway.
    best_plot_col = best_rep_col

    # Build score lookup by representative col (so duplicates can reuse text annotation)
    scores_by_rep = {str(row["col"]): row for _, row in ranked.iterrows()}

    # Plots
    if not args.no_plots:
        plots_dir = os.path.join(out_dir, "plots_per_sic")
        created, skipped = plot_all_sic_columns(
            df_num=df_num,
            target=args.target,
            sic_cols_all=sic_cols_all,
            out_dir=plots_dir,
            best_col=None,  # we highlight only the dedicated best plot below
            scores_by_rep=scores_by_rep,
            rep_for_col=rep_for_col,
            dpi=args.plot_dpi,
        )

        print(f"\nSaved per-SIC plots to: {plots_dir}")
        print(f"Plots created: {created}, skipped (insufficient data): {skipped}")

        # Dedicated "best performance" plot (highlighted)
        best_score = scores_by_rep.get(best_rep_col)
        title_extra = None
        if best_score is not None:
            title_extra = f"BEST by CV: {best_rep_col} | CV(RMSE)={best_score['rmse_cv']:.6g}, CV(R2)={best_score['r2_cv']:.6g}"
        best_out_path = os.path.join(out_dir, "best_sic_plot.png")
        ok = plot_one_sic(
            df_num=df_num,
            target=args.target,
            sic_col=best_plot_col,
            out_path=best_out_path,
            title_extra=title_extra,
            highlight=True,
            dpi=args.plot_dpi,
        )
        if ok:
            print(f"Saved best plot: {best_out_path}")
        else:
            print("Could not create best plot (insufficient data).")

    kn_col = force_col
    input_sorted_path = write_input_files_sorted_by_sic(
        df_num=df_num,
        subdir_paths=input_export_subdir_paths,
        kn_col=kn_col,
        sic_col=best_rep_col,
        out_dir=out_dir,
        source_subdir_col="__source_subdir" if "__source_subdir" in df_num.columns else None,
    )
    if input_sorted_path is not None:
        print(f"Saved input-file SIC list: {input_sorted_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
