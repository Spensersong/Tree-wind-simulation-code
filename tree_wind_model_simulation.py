#!/usr/bin/env python3
"""Deterministic 56,160-scenario wind model; see README_CN.md for assumptions.

This revision corrects willow/Chinese scholar tree geometry and searches for
the FIRST load/capacity crossing across the discontinuous wind-load branches.
It retains the supplied mechanical model and exposes geometry fallbacks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path


RHO_AIR = 1.225
GRAVITY = 9.8
ETA_I0 = 0.80
SIGMA_B0 = 110.0e6

GUST_AMPLIFICATION_FACTOR = 1.5
GUST_PEAK_8LEVEL = 17.2
THRESHOLD_MEAN = GUST_PEAK_8LEVEL / GUST_AMPLIFICATION_FACTOR

KSHEAR = 1.10
KSYN_OPEN = 1.45
KSYN_PAVED = 1.00
LATERAL_ROOT_DIAMETER_MIN_MM = 20.0
LATERAL_ROOT_DIAMETER_MAX_MM = 40.0
CROWN_MASS_ALLOMETRY_A = 0.12
CROWN_MASS_ALLOMETRY_B = 2.18
SP_LENGTH_NORMALIZER_M = 1.0  # assumed reference length, not a measurement
NUMERICAL_BENDING_SCALE_NM2 = 1.0e9  # supplied numerical scale, not measured EI
MAX_SEARCH_MEAN_MPS = 160.0
SOLVER_VERSION = "first_crossing_piecewise_v1"


def original_species(
    dh: dict,
    dc: dict,
    lai: float,
    kcrown: float,
    beta: float,
    rho_g: float,
    tree_type: str,
    heights: list[float],
    root_depth: float,
    n_side: int,
    root_radius: float,
) -> dict:
    return {
        "dh": dh,
        "dc": dc,
        "LAI": lai,
        "kcrown": kcrown,
        "beta": beta,
        "rho_g": rho_g,
        "type": tree_type,
        "H_list": heights,
        "root": {
            "root_depth": root_depth,
            "n_side": n_side,
            "R_root": root_radius,
            "L_eff_ratio": 0.60,
        },
    }


# The supplied mechanical inputs are retained. Willow and Chinese scholar
# tree regressions are assigned using Wei et al. (2013), Tables 5 and 7.
# Chinese ash geometry uses the published 95th-quantile
# Fraxinus chinensis equations ln(H)=0.3157+0.5624ln(DBH) and
# ln(CW)=-0.0204+0.7253ln(DBH). Its remaining mechanical/root inputs are
# supplied representative assumptions (some based on broad-leaved medians),
# not measurements of Chinese ash; all values are recorded in the audit.
TREE_PARAMS = OrderedDict(
    [
        (
            "Cedar",
            original_species(
                {"type": "power", "a": 4.874, "b": 0.814},
                {"type": "power", "a": 3.320, "b": 1.067},
                3.0,
                0.70,
                0.012,
                420,
                "coniferous",
                [5, 8, 11, 14],
                0.60,
                8,
                1.35,
            ),
        ),
        (
            "Chinese pine",
            original_species(
                {"type": "linear", "a": 0.773, "b": 9.606},
                {"type": "quadratic", "a": -0.151, "b": 2.595, "c": 8.604},
                2.8,
                0.68,
                0.012,
                470,
                "coniferous",
                [3, 5, 7, 8],
                0.65,
                9,
                1.40,
            ),
        ),
        (
            "Juniper",
            original_species(
                {"type": "linear", "a": 1.682, "b": 3.452},
                {"type": "log", "a": 5.017, "b": 10.444},
                3.2,
                0.65,
                0.012,
                490,
                "coniferous",
                [5, 7, 9, 11],
                0.58,
                8,
                1.30,
            ),
        ),
        (
            "Walnut",
            original_species(
                {"type": "power", "a": 12.078, "b": 0.167},
                {"type": "quadratic", "a": 0.089, "b": 1.007, "c": 8.737},
                2.1,
                0.72,
                0.009,
                580,
                "broadleaf",
                [5, 8, 11, 14],
                0.70,
                9,
                1.50,
            ),
        ),
        (
            "Tree of heaven",
            original_species(
                {"type": "power", "a": 1.023, "b": 1.262},
                {"type": "linear", "a": 2.694, "b": 1.825},
                1.8,
                0.70,
                0.009,
                540,
                "broadleaf",
                [5, 8, 11, 14],
                0.62,
                8,
                1.45,
            ),
        ),
        (
            "Torch tree",
            original_species(
                {"type": "linear", "a": 0.937, "b": 4.038},
                {"type": "quadratic", "a": -0.681, "b": 6.472, "c": -3.341},
                1.7,
                0.71,
                0.009,
                510,
                "broadleaf",
                [5, 7, 9, 11],
                0.40,
                6,
                1.10,
            ),
        ),
        (
            "Oriental oak",
            original_species(
                {"type": "power", "a": 2.449, "b": 0.932},
                {"type": "linear", "a": 3.334, "b": 4.779},
                2.2,
                0.70,
                0.009,
                570,
                "broadleaf",
                [5, 8, 11, 14],
                0.68,
                9,
                1.48,
            ),
        ),
        (
            "Goldenrain tree",
            original_species(
                {"type": "power", "a": 1.704, "b": 1.132},
                {"type": "power", "a": 2.167, "b": 1.241},
                2.0,
                0.70,
                0.009,
                530,
                "broadleaf",
                [5, 7, 9, 11],
                0.58,
                8,
                1.32,
            ),
        ),
        (
            "Canadian poplar",
            original_species(
                {"type": "linear", "a": 1.161, "b": -0.216},
                {"type": "linear", "a": 4.521, "b": 2.312},
                2.3,
                0.73,
                0.009,
                520,
                "broadleaf",
                [6, 10, 14, 18],
                0.52,
                10,
                1.42,
            ),
        ),
        (
            "Willow",
            original_species(
                {"type": "power", "a": 6.1479, "b": 0.439},
                {"type": "quadratic", "a": -0.058, "b": 2.826, "c": 6.078},
                2.4,
                0.74,
                0.009,
                500,
                "broadleaf",
                [4, 7, 10, 13],
                0.48,
                10,
                1.35,
            ),
        ),
        (
            "Ash",
            original_species(
                {"type": "inverse_log_power", "a": 0.3157, "b": 0.5624},
                {"type": "log_power", "a": -0.0204, "b": 0.7253},
                2.1,
                0.70,
                0.009,
                540,
                "broadleaf",
                [4, 7, 10, 13],
                0.60,
                9,
                1.42,
            ),
        ),
        (
            "Chinese arborvitae",
            original_species(
                {"type": "linear", "a": 1.377, "b": 2.307},
                {"type": "quadratic", "a": 0.267, "b": 1.370, "c": 5.240},
                2.9,
                0.66,
                0.012,
                460,
                "coniferous",
                [4, 7, 10, 13],
                0.50,
                7,
                1.20,
            ),
        ),
        (
            "Chinese scholar tree",
            original_species(
                {"type": "power", "a": 4.524, "b": 0.593},
                {"type": "quadratic", "a": 0.437, "b": -2.446, "c": 17.945},
                2.0,
                0.70,
                0.009,
                550,
                "broadleaf",
                [5, 8, 11, 14],
                0.64,
                9,
                1.44,
            ),
        ),
    ]
)

KLIVE_LIST = [0.775, 0.700, 0.500, 0.450]
KDEFECT_LIST = [0.915, 0.700, 0.500]
OMEGA_LIST = [0.03, 0.075, 0.15, 0.30, 0.50]
KBASE_LIST = [1.0, 0.8, 0.6]
SITE_LIST = [0, 1]
RAIN_CONFIG_LIST = [
    {"kwet": 1.00,  "label": "干燥"},
    {"kwet": 0.80,  "label": "小雨"},
    {"kwet": 0.55,  "label": "暴雨"},
]

FIELDNAMES = [
    "tree",
    "H_m",
    "klive",
    "kdefect",
    "omega",
    "kbase",
    "site",
    "kwet",
    "rain_label",
    "Cw_m",
    "DBH_m",
    "vcrit_mean",
    "vcrit_gust",
    "M_stem_resist_Nm",
    "M_root_resist_Nm",
    "failure_mode",
    "gust_peak_8level",
    "threshold_mean_for_8gust",
    "eta_I0",
    "gust_amplify",
    "risk_status_8gust",
    "geometry_status",
    "vcrit_status",
    "solver_version",
    "M_wind_ref_Nm",
    "load_capacity_ratio_ref",
]


def solve_quadratic(a: float, b: float, c: float) -> float:
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return 2.0
    sqrt_d = math.sqrt(disc)
    x1 = (-b + sqrt_d) / (2.0 * a)
    x2 = (-b - sqrt_d) / (2.0 * a)
    candidates = [x for x in (x1, x2) if 0.1 < x < 30.0]
    if not candidates:
        return max(x1, x2) if max(x1, x2) > 0 else 2.0
    return min(candidates) if a < 0 else max(candidates)


def calc_crown_param(height: float, tree_para: dict) -> tuple[float, float]:
    dh = tree_para["dh"]
    dc = tree_para["dc"]
    if dh["type"] == "power":
        dbh_cm = dh["a"] * (height ** dh["b"])
    elif dh["type"] == "inverse_log_power":
        dbh_cm = math.exp((math.log(height) - dh["a"]) / dh["b"])
    else:
        dbh_cm = dh["a"] * height + dh["b"]

    if dc["type"] == "linear":
        crown_width = (dbh_cm - dc["b"]) / dc["a"]
    elif dc["type"] == "power":
        crown_width = (dbh_cm / dc["a"]) ** (1.0 / dc["b"])
    elif dc["type"] == "log":
        crown_width = math.exp((dbh_cm - dc["b"]) / dc["a"])
    elif dc["type"] == "log_power":
        crown_width = math.exp(dc["a"] + dc["b"] * math.log(dbh_cm))
    else:
        crown_width = solve_quadratic(dc["a"], dc["b"], dc["c"] - dbh_cm)

    crown_width = max(0.5, min(25.0, crown_width))
    return crown_width, dbh_cm / 100.0


def calc_wind_moment(
    v_mean: float,
    height: float,
    crown_width: float,
    dbh_m: float,
    tree_para: dict,
) -> float:
    area = crown_width * 0.70 * height * tree_para["kcrown"] * 1.05
    crown_depth = crown_width 
    h_cg = 0.66 * height
    base_sp = 1.0 - math.exp(
        -2.0 * SP_LENGTH_NORMALIZER_M * crown_depth * tree_para["LAI"] / area
    )
    if v_mean < 15.0:
        eta_vent = 1.0
    elif v_mean < 25.0:
        eta_vent = 0.5 + 0.5 * base_sp
    else:
        eta_vent = base_sp
    force = 0.5 * RHO_AIR * eta_vent * area * (v_mean**2)

    dbh_cm = dbh_m * 100.0
    n_iter = 3 if dbh_cm < 20.0 else (2 if dbh_cm < 35.0 else 1)
    fi, hi = force, h_cg
    # A declared numerical scale in N*m^2, not a measured modulus or EI.
    e_dummy = NUMERICAL_BENDING_SCALE_NM2
    for _ in range(n_iter):
        delta_i = fi * (hi**3) / (3.0 * e_dummy)
        alpha_i = math.atan(delta_i / hi)
        fi = fi * math.cos(alpha_i)
        hi = hi + delta_i

    eta_crown = math.exp(-tree_para["beta"] * v_mean) if v_mean >= 25.0 else 1.0
    return fi * eta_crown * hi


def calc_stem_resistance(
    dbh_m: float,
    klive: float,
    kdefect: float,
    omega: float,
    kbase: float,
) -> float:
    dbh_cm = dbh_m * 100.0
    k_size = 0.85 if dbh_cm < 15.0 else (0.90 if dbh_cm < 30.0 else 0.95)
    k_m = math.exp(-2.8 * omega)
    sigma_eff = klive * kdefect * k_size * k_m * SIGMA_B0
    dbh_base_m = 1.05 * dbh_m
    eta_z = ETA_I0 * (1.0 - omega**2)
    z = eta_z * math.pi * (dbh_m**3) / 32.0
    z_base = eta_z * math.pi * (dbh_base_m**3) / 32.0
    m_stem = sigma_eff * z
    # kbase was present in the old factorial table but omitted from the old
    # code path.  It applies only to the base check, matching the supplied
    # model's base-effective-strength definition.
    m_stem_base = sigma_eff * kbase * z_base
    return min(m_stem, m_stem_base)


def calc_root_resistance(
    crown_width: float,
    dbh_m: float,
    site: int,
    rain_config: dict,
    tree_para: dict,
) -> float:
    root = tree_para["root"]
    root_depth = root["root_depth"]
    le = root_depth * root["L_eff_ratio"]
    if site == 0:
        ksyn = KSYN_OPEN
        h_root = 0.5 * root_depth
    else:
        ksyn = KSYN_PAVED
        h_root = 0.365 * root_depth

    gamma_eff = 17000.0
    mu_eff = 0.32
    taproot_diameter = dbh_m
    f_tap_side = (
        0.5
        * math.pi
        * taproot_diameter
        * gamma_eff
        * mu_eff
        * (le**2)
    )
    f_end = 0.25 * math.pi * (taproot_diameter**2) * gamma_eff * le
    f_tap_total = f_tap_side + f_end

    dbh_cm = dbh_m * 100.0
    lateral_diameter_mm = max(
        LATERAL_ROOT_DIAMETER_MIN_MM,
        min(LATERAL_ROOT_DIAMETER_MAX_MM, dbh_cm * 2.2),
    )
    f_single_lateral = 15.22 * (lateral_diameter_mm**1.61)
    f_lateral = ksyn * KSHEAR * root["n_side"] * f_single_lateral

    # kwet is applied once to root force. fm is a recorded reference value;
    # this code does not execute an fm-based sensitivity calculation.
    f_root = (f_tap_total + f_lateral) * rain_config["kwet"]
    m_root = f_root * h_root

    # Supplied assumption: m_kg=0.12*DBH_cm^2.18. Species-specific empirical
    # support for these coefficients has not been established in this audit.
    crown_mass = CROWN_MASS_ALLOMETRY_A * (dbh_cm**CROWN_MASS_ALLOMETRY_B)
    m_weight = crown_mass * GRAVITY * h_root
    m_equation_33 = m_root + m_weight
    return m_equation_33



def wind_branches() -> list[tuple[float, float]]:
    # nextafter evaluates the representable left limits with the correct
    # branch (<15 or <25), without crossing the discontinuity during bisection.
    return [
        (0.0, math.nextafter(15.0, -math.inf)),
        (15.0, math.nextafter(25.0, -math.inf)),
        (25.0, MAX_SEARCH_MEAN_MPS),
    ]


def first_crossing_v(moment, resistance: float) -> float:
    """First crossing on increasing continuous branches; inf means censored.

    main() checks each geometry on 2,001 points per branch before generation.
    That is a numerical grid check, not a mathematical monotonicity proof.
    Modified parameters require rerunning this check.
    """
    if not math.isfinite(resistance) or resistance <= 0:
        raise ValueError("Resistance must be finite and positive")
    for low, high in wind_branches():
        if moment(low) >= resistance:
            return low
        if moment(high) < resistance:
            continue
        for _ in range(60):
            middle = (low + high) / 2.0
            if moment(middle) < resistance:
                low = middle
            else:
                high = middle
        return high
    return math.inf


def geometry_status(height: float, tree_para: dict) -> str:
    crown, dbh = calc_crown_param(height, tree_para)
    dc = tree_para["dc"]
    flags = []
    if dc["type"] == "quadratic":
        discriminant = dc["b"]**2 - 4*dc["a"]*(dc["c"]-dbh*100)
        if discriminant < 0:
            flags.append("assumed_2m_no_real_root")
    if crown in (0.5, 25.0):
        flags.append("crown_clipped_to_bound")
    return ";".join(flags) if flags else "regression"


def check_wind_branches(points: int = 2001) -> dict:
    violations = []
    geometries = []
    for name, para in TREE_PARAMS.items():
        for height in para["H_list"]:
            crown, dbh = calc_crown_param(height, para)
            status = geometry_status(height, para)
            geometries.append({"tree": name, "H_m": height, "Cw_m": crown,
                               "DBH_m": dbh, "geometry_status": status})
            for low, high in wind_branches():
                previous = -math.inf
                for i in range(points):
                    v = low + (high-low)*i/(points-1)
                    # Force the endpoint exactly onto the selected branch.
                    if i == points-1:
                        v = high
                    value = calc_wind_moment(v, height, crown, dbh, para)
                    if not math.isfinite(value) or value < previous-1e-7:
                        violations.append({"tree":name,"H_m":height,
                                           "branch":[low,high],"at_mean_mps":v})
                        break
                    previous = value
    if violations:
        raise RuntimeError("Non-increasing/nonfinite wind branch: " + str(violations))
    return {"points_per_branch": points, "geometries_checked": len(geometries),
            "branch_nonmonotonic": violations, "geometry": geometries,
            "scope": "numerical grid check on the specified 52 geometries, 0..160 m/s"}

def solve_critical_v(
    height: float,
    tree_para: dict,
    klive: float,
    kdefect: float,
    omega: float,
    kbase: float,
    site: int,
    rain_config: dict,
) -> tuple[float, float, str, float, float, float, float]:
    crown_width, dbh_m = calc_crown_param(height, tree_para)
    stem_r = calc_stem_resistance(dbh_m, klive, kdefect, omega, kbase)
    root_r = calc_root_resistance(crown_width, dbh_m, site, rain_config, tree_para)
    resistance = min(stem_r, root_r)

    # Load drops at 15 and 25 m/s. Search the branches in increasing order;
    # a global bisection can return a later crossing instead of the first.
    v_mean = first_crossing_v(
        lambda v: calc_wind_moment(v, height, crown_width, dbh_m, tree_para),
        resistance,
    )
    v_gust = v_mean * GUST_AMPLIFICATION_FACTOR
    failure_mode = "stem_fracture" if stem_r < root_r else "root_overturn"
    return v_mean, v_gust, failure_mode, stem_r, root_r, crown_width, dbh_m


def generate_rows() -> list[dict]:
    rows: list[dict] = []
    for tree_name, tree_para in TREE_PARAMS.items():
        for height in tree_para["H_list"]:
            for klive in KLIVE_LIST:
                for kdefect in KDEFECT_LIST:
                    for omega in OMEGA_LIST:
                        for kbase in KBASE_LIST:
                            for site in SITE_LIST:
                                for rain in RAIN_CONFIG_LIST:
                                    (
                                        v_mean,
                                        v_gust,
                                        failure_mode,
                                        stem_r,
                                        root_r,
                                        crown_width,
                                        dbh_m,
                                    ) = solve_critical_v(
                                        height,
                                        tree_para,
                                        klive,
                                        kdefect,
                                        omega,
                                        kbase,
                                        site,
                                        rain,
                                    )
                                    rows.append(
                                        {
                                            "tree": tree_name,
                                            "H_m": height,
                                            "klive": klive,
                                            "kdefect": kdefect,
                                            "omega": omega,
                                            "kbase": kbase,
                                            "site": site,
                                            "kwet": rain["kwet"],
                                            "rain_label": rain["label"],
                                            "Cw_m": round(crown_width, 4),
                                            "DBH_m": round(dbh_m, 6),
                                            "vcrit_mean": round(v_mean, 8),
                                            "vcrit_gust": round(v_gust, 8),
                                            "M_stem_resist_Nm": round(stem_r, 2),
                                            "M_root_resist_Nm": round(root_r, 2),
                                            "failure_mode": failure_mode,
                                            "gust_peak_8level": GUST_PEAK_8LEVEL,
                                            "threshold_mean_for_8gust": round(
                                                THRESHOLD_MEAN, 4
                                            ),
                                            "eta_I0": ETA_I0,
                                            "gust_amplify": GUST_AMPLIFICATION_FACTOR,
                                            "risk_status_8gust": (
                                                "fail" if calc_wind_moment(
                                                    THRESHOLD_MEAN, height,
                                                    crown_width, dbh_m, tree_para
                                                ) >= min(stem_r, root_r) else "safe"
                                            ),
                                            "geometry_status": geometry_status(height, tree_para),
                                            "vcrit_status": "crossing_found" if math.isfinite(v_mean) else "not_reached_by_160_mean_mps",
                                            "solver_version": SOLVER_VERSION,
                                            "M_wind_ref_Nm": round(calc_wind_moment(
                                                THRESHOLD_MEAN, height, crown_width,
                                                dbh_m, tree_para), 6),
                                            "load_capacity_ratio_ref": calc_wind_moment(
                                                THRESHOLD_MEAN, height, crown_width,
                                                dbh_m, tree_para)/min(stem_r, root_r),
                                        }
                                    )
    return rows



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent/"data")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    numerical = check_wind_branches()
    rows = generate_rows()
    expected = len(TREE_PARAMS)*4*len(KLIVE_LIST)*len(KDEFECT_LIST)*len(OMEGA_LIST)*len(KBASE_LIST)*len(SITE_LIST)*len(RAIN_CONFIG_LIST)
    keys = {(r["tree"], r["H_m"], r["klive"], r["kdefect"], r["omega"],
             r["kbase"], r["site"], r["kwet"]) for r in rows}
    if expected != 56160 or len(rows) != expected or len(keys) != expected:
        raise RuntimeError("The full 56,160-scenario grid is incomplete or duplicated")
    failures = [r for r in rows if r["risk_status_8gust"] == "fail"]
    stem_n = sum(r["failure_mode"] == "stem_fracture" for r in failures)
    root_n = sum(r["failure_mode"] == "root_overturn" for r in failures)
    if stem_n+root_n != len(failures):
        raise RuntimeError("Unknown failure mode")
    threshold_mismatches = sum(
        (r["vcrit_mean"] <= THRESHOLD_MEAN) != (r["risk_status_8gust"] == "fail")
        for r in rows
    )
    if threshold_mismatches:
        raise RuntimeError("Fixed-wind load check disagrees with first-crossing classification")
    out_csv = out_dir/"tree_wind_latest_56160.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir/"geometry_52.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(numerical["geometry"][0]))
        writer.writeheader()
        writer.writerows(numerical["geometry"])
    audit = {
        "model_version": SOLVER_VERSION,
        "simulation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": len(rows), "unique_scenarios": len(keys), "columns": FIELDNAMES,
        "expected_rows_formula": "13*4*4*3*5*3*2*3",
        "species_parameters": TREE_PARAMS,
        "factor_levels": {"klive": KLIVE_LIST,"kdefect": KDEFECT_LIST,
                          "omega": OMEGA_LIST,"kbase": KBASE_LIST,
                          "site": SITE_LIST,"rain": RAIN_CONFIG_LIST},
        "geometry_assignment_source": {
            "reference":"Wei et al. (2013), Journal of Beijing Forestry University 35(5):56-63",
            "url":"https://j.bjfu.edu.cn/cn/article/pdf/preview/9945.pdf",
            "willow":"Table 7 D_cm=6.1479*H_m^0.439; Table 5 D_cm=-0.058*Cw_m^2+2.826*Cw_m+6.078",
            "Chinese_scholar_tree":"Table 7 D_cm=4.524*H_m^0.593; Table 5 D_cm=0.437*Cw_m^2-2.446*Cw_m+17.945",
            "height_scenarios":"restored original user-selected levels; not literature measurements",
        },
        "mechanical_model": {
            "RHO_AIR_kg_m3":RHO_AIR,"gravity_m_s2":GRAVITY,
            "sigma_b0_Pa":SIGMA_B0,"eta_I0":ETA_I0,
            "gust_amplification":GUST_AMPLIFICATION_FACTOR,
            "gust_equivalent_reference_m_s":GUST_PEAK_8LEVEL,
            "mean_reference_m_s":THRESHOLD_MEAN,
            "SP_length_normalizer_m":SP_LENGTH_NORMALIZER_M,
            "numerical_bending_scale_N_m2":NUMERICAL_BENDING_SCALE_NM2,
            "kshear":KSHEAR,"ksyn_open":KSYN_OPEN,"ksyn_paved":KSYN_PAVED,
            "lateral_diameter_mm_bounds":[LATERAL_ROOT_DIAMETER_MIN_MM,LATERAL_ROOT_DIAMETER_MAX_MM],
            "root_sequence":"(Ftap+Flateral)*kwet*h_root + 0.12*DBH_cm^2.18*g*h_root",
            "root_crown_mass":"supplied assumed coefficients; species-specific source not established",
            "inactive_inputs":["fm", "rho_g", "type", "R_root"],
            "fm_role":"recorded only; no fm-based sensitivity calculation is performed",
            "omega_role":"stem strength/section only; no decay reduction of root capacity",
            "white_ash":"95th-quantile Qingdao geometry, not mean geometry; other inputs are supplied representative assumptions",
            "scope":"remaining mechanical parameters retained; this is not empirical calibration or validation",
        },
        "solver": {"definition":"first mean wind speed at which load reaches min(stem capacity, root capacity)",
                   "branches":wind_branches(),"iterations_per_bisection":60,
                   "max_mean_m_s":MAX_SEARCH_MEAN_MPS,
                   "no_crossing_representation":"inf with explicit status; never report ceiling as a root",
                   "mode_tie_rule":"root_overturn if root and stem capacities are exactly equal",
                   "fixed_wind_failure_rule":"wind moment at 17.2/1.5 m/s >= minimum capacity",
                   "csv_threshold_classification_mismatches":threshold_mismatches},
        "numerical_check": numerical,
        "summary": {"total":len(rows),"n_fail":len(failures),
                    "fail_rate_pct":100*len(failures)/len(rows),
                    "n_stem":stem_n,"n_root":root_n,
                    "root_share_of_failures_pct":100*root_n/len(failures) if failures else None,
                    "non_regression_geometry_rows":sum(r["geometry_status"]!="regression" for r in rows),
                    "no_crossing_rows":sum(r["vcrit_status"]!="crossing_found" for r in rows)},
        "interpretation":"scenario proportions, not measured population probabilities; no tuning to a target failure rate",
    }
    audit_path = out_dir/"tree_wind_latest_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"csv":str(out_csv),"audit":str(audit_path),**audit["summary"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
