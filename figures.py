#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce figures for the roadside-tree wind-lodging paper
from tree_wind_latest_56160.csv.

Run:
    python figures.py
    python figures.py /path/to/tree_wind_latest_56160.csv

Outputs (PNG 300 dpi + PDF) are written to ./outputs/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
DATA_NAME = "tree_wind_latest_56160.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GUST_8LEVEL = 17.2
PAPER_FAIL_RATE = 73.52          # caption value; paper text says 74.22
OMEGA_ORDER = [0.03, 0.075, 0.15, 0.30, 0.50]

# Rainfall mapping (input may be Chinese or English)
RAIN_MAP = {
    "干燥": "Dry", "dry": "Dry",
    "小雨": "Light rain", "light rain": "Light rain", "light": "Light rain",
    "暴雨": "Heavy rain", "heavy rain": "Heavy rain", "heavy": "Heavy rain",
}
RAIN_ORDER = ["Dry", "Light rain", "Heavy rain"]

# Colourblind-friendly palette
CB = sns.color_palette("colorblind", 4)
COLOR_STEM = CB[3]              # red-orange
COLOR_ROOT = CB[1]              # orange
COLOR_FAIL = CB[0]              # blue
COLOR_STEM_PROP = COLOR_STEM
COLOR_SITE0 = CB[2]             # green
COLOR_SITE1 = CB[0]             # blue

COLOR_BOX = "#4C72B0"
COLOR_BOX_RISK = "#1F3A66"
COLOR_RISK_LINE = "#B22222"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def setup_font() -> str:
    """Prefer English sans-serif; keep CJK fallbacks if present."""
    candidates = [
        "Arial", "Helvetica", "DejaVu Sans", "Arial Unicode MS",
        "PingFang SC", "Hiragino Sans GB", "Heiti TC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen] + candidates
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[Font] Using: {chosen}")
    return chosen


def locate_data() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p
    base = Path(__file__).resolve().parent
    p = base / DATA_NAME
    if p.exists():
        return p
    hits = [q for q in sorted(base.glob("tree_wind*"))
            if q.suffix.lower() in (".csv", ".xlsx", ".xls")]
    if hits:
        print(f"[Info] Using: {hits[0].name}")
        return hits[0]
    raise FileNotFoundError(f"Data file not found: {DATA_NAME}")


def canon_rain(x) -> str:
    try:
        s = str(x).strip().lower()
    except (TypeError, ValueError, AttributeError):
        return "Other"
    if s in ("", "nan", "nat", "none"):
        return "Other"
    return RAIN_MAP.get(s, s.capitalize() if s else "Other")


def canon_mode(x) -> str:
    s = str(x).strip().lower()
    if s in ("stem_fracture", "stem fracture", "trunk fracture", "干折"):
        return "stem_fracture"
    if s in ("root_overturn", "root overturn", "倒伏"):
        return "root_overturn"
    return s


def load_data(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(data_path, encoding="utf-8-sig")
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    if "tree" not in df.columns and "树种" in df.columns:
        df = df.rename(columns={"树种": "tree"})

    df["tree"] = df["tree"].astype(str).str.strip()
    df["rain_label"] = df["rain_label"].map(canon_rain)
    df["failure_mode"] = df["failure_mode"].map(canon_mode)
    df["risk"] = df["risk_status_8gust"].astype(str).str.strip().str.lower()

    for c in ["omega", "site", "vcrit_gust", "klive", "kdefect", "kbase",
              "H_m", "DBH_m", "Cw_m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def fail_rate_pct(sub: pd.DataFrame) -> float:
    if len(sub) == 0:
        return 0.0
    return 100.0 * (sub["risk"] == "fail").mean()


def print_table(df: pd.DataFrame) -> None:
    with pd.option_context("display.unicode.east_asian_width", True,
                           "display.width", 160):
        print(df.to_string(index=False,
                           float_format=lambda v: f"{v:,.2f}"))


# ---------------------------------------------------------------------------
# Overall statistics
# ---------------------------------------------------------------------------
def overall_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    fails = df[df["risk"] == "fail"]
    n_fail = len(fails)
    n_stem = int((fails["failure_mode"] == "stem_fracture").sum())
    n_root = n_fail - n_stem
    return {
        "total": total,
        "n_fail": n_fail,
        "fail_rate": 100.0 * n_fail / total if total else 0.0,
        "n_stem": n_stem,
        "n_root": n_root,
        "stem_pct": 100.0 * n_stem / n_fail if n_fail else 0.0,
        "root_pct": 100.0 * n_root / n_fail if n_fail else 0.0,
    }


# ---------------------------------------------------------------------------
# Species breakdown
# ---------------------------------------------------------------------------
def species_table(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for sp, grp in df.groupby("tree"):
        fails = grp[grp["risk"] == "fail"]
        n_fail = len(fails)
        stem_p = (100.0 * (fails["failure_mode"] == "stem_fracture").mean()
                  if n_fail else 0.0)
        recs.append({
            "Species": sp,
            "Total scenarios": len(grp),
            "Failures": n_fail,
            "Failure rate (%)": 100.0 * n_fail / len(grp),
            "Stem fracture share (%)": stem_p,
            "Root overturn share (%)": 100.0 - stem_p,
        })
    t = pd.DataFrame(recs)
    return t.sort_values("Failure rate (%)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 1 — Species failure rate with stacked failure modes
# ---------------------------------------------------------------------------
def fig1_species_failure(df: pd.DataFrame, overall_rate: float,
                         out_dir: Path) -> Path:
    t = species_table(df).sort_values("Failure rate (%)").reset_index(drop=True)
    species = t["Species"].tolist()
    fr = t["Failure rate (%)"].to_numpy(dtype=float)
    stem_abs = t["Stem fracture share (%)"].to_numpy(dtype=float) / 100.0 * fr
    root_abs = fr - stem_abs

    fig, ax = plt.subplots(figsize=(13, 8))
    x = np.arange(len(species))
    ax.bar(x, root_abs, width=0.72, color=COLOR_ROOT,
           edgecolor="black", linewidth=0.4, alpha=0.85,
           label="Root overturn")
    ax.bar(x, stem_abs, bottom=root_abs, width=0.72, color=COLOR_STEM,
           edgecolor="black", linewidth=0.4, alpha=0.85,
           label="Stem fracture")
    for xi, yi in zip(x, fr):
        if yi > 0:
            ax.text(xi, yi + 1.6, f"{yi:.1f}", ha="center", va="bottom",
                    fontsize=10, color="#222222")

    ax.axhline(overall_rate, color="#555555", ls="--", lw=1.2,
               label=f"Overall mean failure rate ({overall_rate:.2f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sp}\n(n={n:,})"
                        for sp, n in zip(species, t["Total scenarios"])],
                       fontsize=10)
    ax.set_xlabel("Tree species", fontsize=12)
    ax.set_ylabel("Failure rate (%)", fontsize=12)
    ax.set_ylim(0, max(fr.max() + 5.0, overall_rate + 8.5))
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Figure 1. Failure rate and failure-mode composition "
                 "by species under 8-level gusts (17.2 m/s)",
                 fontsize=13)
    fig.tight_layout(pad=2.0)

    path = out_dir / "Fig1_Species_failure_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig1_Species_failure_rate.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2 — Decay effect (omega)
# ---------------------------------------------------------------------------
def omega_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for om, grp in df.groupby("omega", sort=True):
        fails = grp[grp["risk"] == "fail"]
        fr = 100.0 * len(fails) / len(grp)
        stem_p = (100.0 * (fails["failure_mode"] == "stem_fracture").mean()
                  if len(fails) else 0.0)
        rows.append((float(om), fr, stem_p))
    g = pd.DataFrame(rows, columns=["omega", "failure_rate", "stem_share"])
    return g.sort_values("omega").reset_index(drop=True)


def fig2_omega_effect(g: pd.DataFrame, out_dir: Path) -> Path:
    x = g["omega"].to_numpy(dtype=float)
    fr = g["failure_rate"].to_numpy(dtype=float)
    sp = g["stem_share"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(11.2, 7.4))
    x_fit = np.linspace(x.min(), x.max(), 100)
    fr_fit = np.polyval(np.polyfit(x, fr, 2), x_fit)
    sp_fit = np.polyval(np.polyfit(x, sp, 2), x_fit)

    ax1.plot(x_fit, fr_fit, "-", color=COLOR_FAIL, lw=1.5, alpha=0.6, zorder=1,
             label="Failure rate (quadratic fit)")
    l1 = ax1.scatter(x, fr, s=100, facecolor="white", edgecolor=COLOR_FAIL,
                     linewidths=2.5, zorder=3,
                     label="Overall failure rate (%)")
    ax1.scatter(x, fr, s=30, facecolor=COLOR_FAIL, zorder=4)

    ax1.set_xlabel("Wood decay severity, omega", fontsize=12)
    ax1.set_ylabel("Overall failure rate (%)", color=COLOR_FAIL, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLOR_FAIL)
    ax1.set_xticks(OMEGA_ORDER)
    ax1.set_xticklabels([f"{v:g}" for v in OMEGA_ORDER])
    ax1.set_xlim(0.0, 0.53)
    ax1.set_ylim(0, max(fr.max() * 1.30, 1.0))

    ax1.axvline(0.30, color="#2CA02C", ls="--", lw=1.2, zorder=1.5)
    ax1.text(0.315, ax1.get_ylim()[1] * 0.86,
             "omega = 0.30\n(recommended monitoring threshold)",
             color="#2CA02C", fontsize=10, ha="left", va="top")

    ax2 = ax1.twinx()
    ax2.plot(x_fit, sp_fit, "-", color=COLOR_STEM_PROP, lw=1.5,
             alpha=0.6, zorder=1,
             label="Stem fracture share (quadratic fit)")
    l2 = ax2.scatter(x, sp, s=100, facecolor="white", edgecolor=COLOR_STEM_PROP,
                     linewidths=2.5, zorder=3,
                     label="Stem fracture share of failures (%)")
    ax2.scatter(x, sp, s=30, facecolor=COLOR_STEM_PROP, zorder=4)
    ax2.set_ylabel("Stem fracture share of failures (%)",
                   color=COLOR_STEM_PROP, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=COLOR_STEM_PROP)
    ax2.set_ylim(0, max(sp.max() * 1.30, 1.0))
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#333333")
    ax2.spines["right"].set_linewidth(1.0)

    for xi in [0.03, 0.15, 0.30, 0.50]:
        i = int(np.argmin(np.abs(x - xi)))
        if xi in (0.03, 0.15):
            ax1.text(xi, fr[i] + ax1.get_ylim()[1] * 0.045, f"{fr[i]:.1f}",
                     ha="center", va="bottom", fontsize=10, color=COLOR_FAIL)
        else:
            ax1.text(xi, fr[i] - ax1.get_ylim()[1] * 0.055, f"{fr[i]:.1f}",
                     ha="center", va="top", fontsize=10, color=COLOR_FAIL)
        ax2.text(xi, sp[i] - ax2.get_ylim()[1] * 0.125, f"{sp[i]:.1f}",
                 ha="center", va="top", fontsize=10, color=COLOR_STEM_PROP)

    ax1.legend(handles=[l1, l2], loc="lower right", fontsize=10)
    ax1.set_title("Figure 2. Effect of wood decay severity (omega) "
                  "on failure rate and failure mode", fontsize=13)
    fig.tight_layout(pad=2.0)
    path = out_dir / "Fig2_Decay_effect.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig2_Decay_effect.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Healthy subset
# ---------------------------------------------------------------------------
def healthy_subset_stats(df: pd.DataFrame) -> None:
    mask = (np.isclose(df["klive"], 0.775) & np.isclose(df["kdefect"], 0.915)
            & np.isclose(df["omega"], 0.03) & np.isclose(df["kbase"], 1.0))
    sub = df[mask]
    fails = sub[sub["risk"] == "fail"]
    n_fail = len(fails)
    n_stem = int((fails["failure_mode"] == "stem_fracture").sum())
    n_root = n_fail - n_stem
    stem_p = 100.0 * n_stem / n_fail if n_fail else 0.0

    print("Filter: klive=0.775, kdefect=0.915, omega=0.03, kbase=1.0")
    print(f"Healthy subset total: {len(sub):,}")
    print(f"Healthy subset failure rate: {fail_rate_pct(sub):.2f}%  "
          f"(n_fail={n_fail:,})")
    print(f"Stem fracture: {n_stem:,} ({stem_p:.2f}%)")
    print(f"Root overturn: {n_root:,} ({100.0 - stem_p:.2f}%)")


# ---------------------------------------------------------------------------
# Figure 3 — Rainfall × site interaction
# ---------------------------------------------------------------------------
def fig3_rain_site(df: pd.DataFrame, out_dir: Path) -> Path:
    rain_cats = [r for r in RAIN_ORDER if r in set(df["rain_label"])]
    grand_mean = fail_rate_pct(df)
    site_labels_en = {0: "Open green space (site=0)",
                      1: "Narrow paved tree pit (site=1)"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7.6))

    rain_fail = df.groupby("rain_label")["risk"].apply(
        lambda s: 100.0 * (s == "fail").mean()).reindex(rain_cats)

    site_vals = {
        0: [fail_rate_pct(df[(df["rain_label"] == r) & (df["site"] == 0)])
            for r in rain_cats],
        1: [fail_rate_pct(df[(df["rain_label"] == r) & (df["site"] == 1)])
            for r in rain_cats],
    }
    ymax = max(rain_fail.max(),
               max(max(v) for v in site_vals.values()),
               grand_mean) * 1.18

    bars = axL.bar(rain_cats, rain_fail, width=0.65, color=COLOR_FAIL,
                   edgecolor="black", linewidth=0.4, alpha=0.85)
    for b, v in zip(bars, rain_fail):
        cx = b.get_x() + b.get_width() / 2
        if v >= ymax * 0.25:
            axL.text(cx, v * 0.5, f"{v:.1f}", ha="center", va="center",
                     color="white", fontsize=10)
        else:
            axL.text(cx, v + ymax * 0.02, f"{v:.1f}", ha="center",
                     va="bottom", color="#222222", fontsize=10)
    axL.set_ylim(0, ymax)
    axL.set_ylabel("Failure rate (%)", fontsize=12)
    axL.set_xlabel("Rainfall condition", fontsize=12)
    axL.set_xticklabels(rain_cats, fontsize=10)
    axL.set_title("(a) Overall failure rate by rainfall", fontsize=12)

    xpos = np.arange(len(rain_cats))
    w = 0.48
    for site, color, lab in [(0, COLOR_SITE0, site_labels_en[0]),
                             (1, COLOR_SITE1, site_labels_en[1])]:
        vals = site_vals[site]
        off = -w / 2 if site == 0 else w / 2
        bars = axR.bar(xpos + off, vals, width=w, color=color,
                       edgecolor="black", linewidth=0.4, alpha=0.85,
                       label=lab)
        for b, v in zip(bars, vals):
            cx = b.get_x() + b.get_width() / 2
            if v >= ymax * 0.25:
                axR.text(cx, v * 0.5, f"{v:.1f}", ha="center", va="center",
                         color="white", fontsize=10)
            else:
                axR.text(cx, v + ymax * 0.015, f"{v:.1f}", ha="center",
                         va="bottom", color="#222222", fontsize=10)
    axR.axhline(grand_mean, color="#555555", ls="--", lw=1.2,
                label=f"Overall mean failure rate ({grand_mean:.1f}%)")
    axR.set_xticks(xpos)
    axR.set_xticklabels(rain_cats, fontsize=10)
    axR.set_ylim(0, ymax)
    axR.set_ylabel("Failure rate (%)", fontsize=12)
    axR.set_xlabel("Rainfall condition", fontsize=12)
    axR.set_title("(b) Failure rate by rainfall and planting site",
                  fontsize=12)
    axR.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=10)

    fig.suptitle("Figure 3. Effects of rainfall condition and planting "
                 "site on failure rate", fontsize=13, y=1.02)
    fig.tight_layout(pad=2.0)
    path = out_dir / "Fig3_Rain_site_effect.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig3_Rain_site_effect.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4 — Critical gust speed distribution
# ---------------------------------------------------------------------------
def fig4_vcrit_gust(df: pd.DataFrame, out_dir: Path) -> Path:
    order = (df.groupby("tree")["vcrit_gust"].median()
             .sort_values().index.tolist())
    med_by_sp = df.groupby("tree")["vcrit_gust"].median()
    box_colors = [COLOR_BOX_RISK if med_by_sp[sp] < GUST_8LEVEL else COLOR_BOX
                  for sp in order]

    fig, ax = plt.subplots(figsize=(13.6, 7.8))
    sns.boxplot(data=df, x="tree", y="vcrit_gust", order=order, ax=ax,
                width=0.55, fliersize=0, linewidth=1.2, saturation=0.75,
                palette=box_colors, zorder=2)

    for patch in ax.artists:
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.2)
    for line in ax.lines:
        xd = line.get_xdata()
        yd = line.get_ydata()
        if (len(xd) == 2 and len(yd) == 2
                and abs(xd[1] - xd[0]) > 0.25 and yd[0] == yd[1]):
            line.set_linewidth(2.0)
            line.set_color("#111111")

    rng = np.random.default_rng(42)
    for xi, sp in enumerate(order):
        vals = df.loc[df["tree"] == sp, "vcrit_gust"].to_numpy()
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(xi + jitter, vals, s=2, alpha=0.3, color="#2C3E50",
                   linewidths=0, zorder=3)

    ylo, yhi = ax.get_ylim()
    if GUST_8LEVEL > ylo:
        ax.axhspan(ylo, GUST_8LEVEL, color="#D95F02", alpha=0.08, zorder=0)
        ax.text(0.012, 0.035, "Below 17.2 m/s: high risk",
                transform=ax.transAxes, color=COLOR_RISK_LINE,
                fontsize=10, va="bottom")
    ax.axhline(GUST_8LEVEL, color=COLOR_RISK_LINE, ls="--", lw=1.8, zorder=1)
    ax.annotate("8-level gust = 17.2 m/s",
                xy=(len(order) - 0.55, GUST_8LEVEL + 0.05),
                color=COLOR_RISK_LINE, fontsize=10, ha="right", va="bottom")

    for xi, sp in enumerate(order):
        vals = df.loc[df["tree"] == sp, "vcrit_gust"].dropna()
        q1, q3 = vals.quantile([0.25, 0.75])
        whisker_hi = min(vals.max(), q3 + 1.5 * (q3 - q1))
        ax.text(xi, whisker_hi + (yhi - ylo) * 0.015,
                f"{med_by_sp[sp]:.1f}",
                ha="center", va="bottom", fontsize=10, color="#222222")
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.06)
    ax.set_ylabel("Critical gust wind speed, vcrit_gust (m/s)", fontsize=12)
    ax.set_xlabel("Tree species (ordered by ascending median)", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title("Figure 4. Distribution of gust-equivalent critical "
                 "wind speed by species", fontsize=13)
    fig.tight_layout(pad=2.0)
    path = out_dir / "Fig4_Critical_wind_speed.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig4_Critical_wind_speed.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 5 — Sensitivity tornado (OAT)
# ---------------------------------------------------------------------------
def _query_level(df, base_tree, base_params, vary_param, vary_value):
    m = df["tree"] == base_tree
    for k, v in base_params.items():
        if k == vary_param:
            continue
        if isinstance(v, float):
            m &= np.isclose(df[k].astype(float), v)
        else:
            m &= (df[k] == v)
    if isinstance(vary_value, float):
        m &= np.isclose(df[vary_param].astype(float), vary_value)
    else:
        m &= (df[vary_param] == vary_value)
    return df[m]


def _nearest_level(df, base_tree, base_params, vary_param, target):
    m = df["tree"] == base_tree
    for k, v in base_params.items():
        if k == vary_param:
            continue
        if isinstance(v, float):
            m &= np.isclose(df[k].astype(float), v)
        else:
            m &= (df[k] == v)
    avail = sorted(df.loc[m, vary_param].dropna().unique().tolist())
    if not avail:
        return None
    if isinstance(target, str):
        return target if target in avail else None
    return min(avail, key=lambda a: abs(float(a) - float(target)))


def sensitivity_tornado(df: pd.DataFrame, out_dir: Path) -> Path:
    base_tree = "Chinese scholar tree"
    if base_tree not in set(df["tree"].unique()):
        raise ValueError(
            "Chinese scholar tree not found in the simulation dataset."
        )

    base_params = {
        "site": 0,
        "rain_label": "Dry",
        "omega": 0.03,
        "klive": 0.775,
        "kdefect": 0.915,
        "kbase": 1.0,
    }

    base_sub = df[df["tree"] == base_tree]
    for k, v in base_params.items():
        if isinstance(v, float):
            base_sub = base_sub[np.isclose(base_sub[k].astype(float), v)]
        else:
            base_sub = base_sub[base_sub[k] == v]

    if len(base_sub) == 0:
        print(f"[Warning] Exact baseline not found for '{base_tree}'; "
              f"using dataset medians.")
        base_params = {
            "site": int(round(float(df["site"].median()))),
            "rain_label": str(df["rain_label"].mode().iloc[0]),
            "omega": float(df["omega"].median()),
            "klive": float(df["klive"].median()),
            "kdefect": float(df["kdefect"].median()),
            "kbase": float(df["kbase"].median()),
        }
        baseline_vcrit = float(df["vcrit_gust"].median())
    else:
        baseline_vcrit = float(base_sub["vcrit_gust"].mean())

    print(f"Baseline species: {base_tree}")
    print(f"Baseline params : {base_params}")
    print(f"Baseline mean vcrit_gust = {baseline_vcrit:.2f} m/s")

    variations = [
        ("omega", "Decay severity (omega)",
         [0.03, 0.075, 0.15, 0.30, 0.50]),

        ("kdefect", "Defect factor (kdefect)",
         [0.50, 0.70, 0.915]),

        ("klive", "Live-condition factor (klive)",
         [0.45, 0.50, 0.70, 0.775]),

        ("kbase", "Base-condition factor (kbase)",
         [0.60, 0.80, 1.00]),

        ("site", "Planting site", [0, 1]),

        ("rain_label", "Rainfall condition",
         ["Dry", "Light rain", "Heavy rain"]),
    ]

    rows = []
    for param, label, levels in variations:
        means, used = [], []

        for lv in levels:
            sub = _query_level(df, base_tree, base_params, param, lv)

            if sub.empty:
                raise ValueError(
                    f"No matching scenarios for {label}={lv!r}. "
                    "Check the parameter levels and baseline settings."
                )

            means.append(float(sub["vcrit_gust"].mean()))
            used.append(lv)

        rng = max(means) - min(means)
        rows.append({
            "param": label,
            "range": rng,
            "levels": used,
            "means": means
        })

    if not rows:
        print("[Warning] No sensitivity data available.")
        return None

    res = sorted(rows, key=lambda r: r["range"])
    names = [r["param"] for r in res]
    ranges = [r["range"] for r in res]

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    y = np.arange(len(res))
    ax.barh(y, ranges, height=0.62, color=COLOR_FAIL, alpha=0.85,
            edgecolor="black", linewidth=0.4)
    for yi, r in zip(y, ranges):
        ax.text(r + max(ranges) * 0.015, yi, f"{r:.2f}",
                va="center", ha="left", fontsize=10, color="#222222")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, max(ranges) * 1.18)
    ax.set_xlabel("Range of mean vcrit_gust (m/s)", fontsize=12)
    ax.set_ylabel("Parameter", fontsize=12)
    ax.text(0.98, 0.96,
            f"Baseline: {base_tree} (site=0, Dry, omega=0.03)\n"
            f"mean vcrit_gust = {baseline_vcrit:.2f} m/s",
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            color="#555555")
    ax.set_title("Figure 5. Sensitivity of gust-equivalent critical "
                 "wind speed under baseline conditions", fontsize=13)
    fig.tight_layout(pad=2.0)
    path = out_dir / "Fig5_Tornado_Sensitivity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig5_Tornado_Sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 6 — ML feature importance (optional)
# ---------------------------------------------------------------------------
def sensitivity_ml_feature_importance(df: pd.DataFrame, out_dir: Path) -> Path:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("[Warning] scikit-learn not installed; skipping ML importance.")
        return None

    feats = ["omega", "site", "klive", "kdefect", "kbase",
             "tree", "rain_label"]
    sub = df[feats + ["vcrit_gust"]].dropna().copy()
    le_tree = LabelEncoder()
    sub["tree_code"] = le_tree.fit_transform(sub["tree"])
    le_rain = LabelEncoder()
    sub["rain_code"] = le_rain.fit_transform(sub["rain_label"])

    X = sub[["omega", "site", "klive", "kdefect", "kbase",
             "tree_code", "rain_code"]]
    y = sub["vcrit_gust"]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns)

    display_names = {
        "omega": "Decay severity (omega)",
        "site": "Planting site",
        "klive": "Live-wood moisture (klive)",
        "kdefect": "Defect factor (kdefect)",
        "kbase": "Base decay (kbase)",
        "tree_code": "Tree species",
        "rain_code": "Rainfall condition",
    }
    imp.index = [display_names.get(c, c) for c in imp.index]
    imp = imp.sort_values()

    fig, ax = plt.subplots(figsize=(10.5, 7))
    ypos = np.arange(len(imp))
    ax.barh(ypos, imp.values, height=0.62, color=COLOR_FAIL, alpha=0.85,
            edgecolor="black", linewidth=0.4)
    for yi, v in zip(ypos, imp.values):
        ax.text(v + imp.max() * 0.02, yi, f"{v:.3f}", va="center",
                ha="left", fontsize=10, color="#222222")
    ax.set_yticks(ypos)
    ax.set_yticklabels(imp.index.tolist(), fontsize=10)
    ax.set_xlim(0, imp.max() * 1.18)
    ax.set_xlabel("Feature importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Figure 6. Random-Forest feature importance "
                 "for vcrit_gust", fontsize=13)
    fig.tight_layout(pad=2.0)
    path = out_dir / "Fig6_ML_Feature_Importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig6_ML_Feature_Importance.pdf",
                bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 7 — Interaction heatmap
# ---------------------------------------------------------------------------
def sensitivity_interaction_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    sub = df.dropna(subset=["rain_label", "site", "risk"])
    pivot = (sub.assign(is_fail=(sub["risk"] == "fail"))
             .groupby(["rain_label", "site"])["is_fail"]
             .mean().mul(100.0)
             .unstack("site"))
    pivot = pivot.reindex([r for r in RAIN_ORDER if r in pivot.index], axis=0)
    pivot = pivot.reindex([c for c in (0, 1) if c in pivot.columns], axis=1)

    site_ticks = ["Open green space (site=0)", "Narrow paved tree pit (site=1)"]
    xticklabels = [site_ticks[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Failure rate (%)"},
                xticklabels=xticklabels,
                yticklabels=pivot.index.tolist(),
                annot_kws={"fontsize": 11},
                linewidths=0)
    ax.set_xlabel("Planting site", fontsize=12)
    ax.set_ylabel("Rainfall condition", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title("Figure 7. Joint effect of rainfall and planting site "
                 "on failure rate", fontsize=13)
    fig.tight_layout(pad=2.0)
    path = out_dir / "Fig7_Interaction_Heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig7_Interaction_Heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Key summary
# ---------------------------------------------------------------------------
def print_key_summary(df: pd.DataFrame, t: pd.DataFrame, g: pd.DataFrame) -> None:
    t_sorted = t.sort_values("Failure rate (%)", ascending=False).reset_index(drop=True)
    hi = t_sorted.iloc[0]
    lo = t_sorted.iloc[-1]
    print(f"Highest failure rate: {hi['Species']}  ({hi['Failure rate (%)']:.2f}%)")
    print(f"Lowest  failure rate: {lo['Species']}  ({lo['Failure rate (%)']:.2f}%)")

    o03 = g.loc[np.isclose(g["omega"], 0.03)].iloc[0]
    o50 = g.loc[np.isclose(g["omega"], 0.50)].iloc[0]
    print(f"omega 0.03 -> 0.50: failure rate "
          f"{o03['failure_rate']:.2f}% -> {o50['failure_rate']:.2f}% "
          f"({o50['failure_rate'] - o03['failure_rate']:+.2f} pp)")
    print(f"omega 0.03 -> 0.50: stem share "
          f"{o03['stem_share']:.2f}% -> {o50['stem_share']:.2f}% "
          f"({o50['stem_share'] - o03['stem_share']:+.2f} pp)")

    fr_dry = fail_rate_pct(df[df["rain_label"] == "Dry"])
    fr_heavy = fail_rate_pct(df[df["rain_label"] == "Heavy rain"])
    print(f"Rainfall: Dry {fr_dry:.2f}% -> Heavy rain {fr_heavy:.2f}% "
          f"({fr_heavy - fr_dry:+.2f} pp)")

    fr_h0 = fail_rate_pct(df[(df["rain_label"] == "Heavy rain") & (df["site"] == 0)])
    fr_h1 = fail_rate_pct(df[(df["rain_label"] == "Heavy rain") & (df["site"] == 1)])
    print(f"Heavy rain: narrow pit (site=1) {fr_h1:.2f}% vs "
          f"open site (site=0) {fr_h0:.2f}% ({fr_h1 - fr_h0:+.2f} pp)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.15)
    setup_font()

    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.labelcolor"] = "#222222"
    plt.rcParams["xtick.color"] = "#222222"
    plt.rcParams["ytick.color"] = "#222222"
    plt.rcParams["axes.labelpad"] = 8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.size"] = 4.0
    plt.rcParams["ytick.major.size"] = 4.0
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["legend.edgecolor"] = "#E0E0E0"
    plt.rcParams["legend.framealpha"] = 0.85
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["savefig.bbox"] = "tight"

    data_path = locate_data()
    print(f"[Data] Reading: {data_path}  ({data_path.resolve()})")
    df = load_data(data_path)

    # ---- 1. Overall stats ----------------------------------------------
    section("1. Overall statistics")
    ov = overall_stats(df)
    print(f"Total scenarios: {ov['total']:,}")
    print(f"Overall failure rate: {ov['fail_rate']:.2f}%  "
          f"(failed {ov['n_fail']:,})")
    print(f"Failure mode: stem_fracture {ov['n_stem']:,} "
          f"({ov['stem_pct']:.2f}%) | "
          f"root_overturn {ov['n_root']:,} ({ov['root_pct']:.2f}%)")
    if abs(ov["fail_rate"] - PAPER_FAIL_RATE) < 0.2:
        print(f"[Match] Paper reported {PAPER_FAIL_RATE}% ✓")
    else:
        print(f"[Note] Paper reported {PAPER_FAIL_RATE}%, "
              f"this data gives {ov['fail_rate']:.2f}%")

    # ---- 2. Species breakdown -------------------------------------------
    section("2. Species breakdown (sorted by failure rate)")
    t_species = species_table(df)
    print_table(t_species)

    # ---- 3. Figure 1 ----------------------------------------------------
    print("\n[Plot] Figure 1 ...")
    p1 = fig1_species_failure(df, ov["fail_rate"], OUT_DIR)
    print(f"Saved: {p1}")

    # ---- 4. Figure 2 ----------------------------------------------------
    section("4.5 Decay effect (omega)")
    g_omega = omega_table(df)
    print(g_omega.to_string(index=False,
                            float_format=lambda v: f"{v:.2f}"))
    print("\n[Plot] Figure 2 ...")
    p2 = fig2_omega_effect(g_omega, OUT_DIR)
    print(f"Saved: {p2}")

    # ---- 5. Healthy subset ---------------------------------------------
    section("4.6 Healthy subset")
    healthy_subset_stats(df)

    # ---- 6. Figure 3 ----------------------------------------------------
    print("\n[Plot] Figure 3 ...")
    p3 = fig3_rain_site(df, OUT_DIR)
    print(f"Saved: {p3}")

    # ---- 7. Figure 4 ----------------------------------------------------
    print("\n[Plot] Figure 4 ...")
    p4 = fig4_vcrit_gust(df, OUT_DIR)
    print(f"Saved: {p4}")

    # ---- 8. Summary -----------------------------------------------------
    section("8. Key numerical findings")
    print_key_summary(df, t_species, g_omega)

    # ---- 9. Sensitivity -------------------------------------------------
    section("9. Sensitivity Analysis")
    p5 = sensitivity_tornado(df, OUT_DIR)
    print(f"Saved Tornado: {p5}")
    p6 = sensitivity_ml_feature_importance(df, OUT_DIR)
    print(f"Saved ML Importance: {p6}")
    p7 = sensitivity_interaction_heatmap(df, OUT_DIR)
    print(f"Saved Interaction: {p7}")

    section("Done")
    print(f"All figures saved to: {OUT_DIR.resolve()}  (PNG 300 dpi + PDF)")
    for p in sorted(OUT_DIR.glob("Fig*.png")):
        print(f"  - {p.name}")
    for p in sorted(OUT_DIR.glob("Fig*.pdf")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
