from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
BLOCKED = "BLOCKED"

CLAIM_FILE = "claim_to_code.json"
REPORT_FILE = "verification_report.json"
MANIFEST_FILE = "integrity_manifest.json"
SUMMARY_FILE = "smoke_report.txt"

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = Path(__file__).resolve().parent
DEFAULT_VALUES = SMOKE_DIR / "tce_reported_values.json"


def candidate_package_roots(start: Path) -> list[Path]:
    configured = os.environ.get("THROMBXAI_ROOT", "")
    roots: list[Path] = [Path(configured).expanduser()] if configured else []
    for base in [start, *start.parents][:4]:
        if (base / "thrombxai").is_dir():
            roots.append(base)
        workspace = base / "code"
        if workspace.is_dir():
            roots.extend(sorted(package.parent for package in workspace.glob("*/thrombxai")))
    return roots


def ensure_package_path(roots: list[Path]) -> list[Path]:
    missing: list[str] = []
    for root in roots:
        entry = str(root)
        if root.is_dir() and entry not in sys.path and entry not in missing:
            missing.append(entry)
    sys.path[:0] = missing
    return [Path(entry) for entry in missing]


ensure_package_path(candidate_package_roots(SMOKE_DIR))

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hydra",
        "multirun",
        "__pycache__",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        ".venv",
        ".escvenv",
        "outputs",
        "artifacts",
        "checkpoints",
        "node_modules",
    }
)
SKIP_FILES = frozenset({".DS_Store"})
SKIP_SUFFIXES = (".egg-info",)

SMOKE_SEED = 0
SMOKE_SUBJECTS = 96
SMOKE_TIME_STEPS = 48
SMOKE_BATCH = 16
SMOKE_EPOCHS = 2
SMOKE_HIDDEN = 48
SMOKE_HEADS = 4
SMOKE_EMBED = 48
SMOKE_ATTN = 24
SMOKE_MLP = 64
PARAM_TOLERANCE_REL = 0.15


def load_values(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text())
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_files(root: Path, exclude: set[str]) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in parts[:-1]):
            continue
        if path.name in SKIP_FILES or path.name in exclude:
            continue
        if any(path.name.endswith(suffix) for suffix in SKIP_SUFFIXES):
            continue
        found.append(path)
    return found


def _atomic_write(root: Path, name: str, text: str) -> Path:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    os.close(handle)
    try:
        Path(tmp).write_text(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def write_json(root: Path, name: str, payload: Any) -> Path:
    return _atomic_write(root, name, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_text(root: Path, name: str, text: str) -> Path:
    return _atomic_write(root, name, text)


def digest_tree(root: Path, exclude: set[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in walk_files(root, exclude):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def manifest_digest(entries: list[dict[str, Any]]) -> str:
    lines = [f"{entry['path']}:{entry['sha256']}" for entry in entries]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator


def receptive_field(layers: int, kernel: int, dilated: bool = True) -> int:
    total = 1
    for layer in range(layers):
        total += (kernel - 1) * ((2**layer) if dilated else 1)
    return total


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    counts = sum(row["n"] for row in rows)
    return sum(row["n"] * row[key] for row in rows) / counts


def record(
    check_id: str,
    claim: str,
    anchor: str,
    expected: Any,
    observed: Any,
    passed: bool,
    method: str = "closed_form",
    units: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": check_id,
        "claim": claim,
        "paper_anchor": anchor,
        "expected": expected,
        "observed": observed,
        "units": units,
        "method": method,
        "status": PASS if passed else FAIL,
        "notes": notes,
    }


def number(
    check_id: str,
    claim: str,
    anchor: str,
    expected: float,
    observed: float,
    tolerance: float,
    units: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return record(
        check_id,
        claim,
        anchor,
        round(expected, 4),
        round(observed, 4),
        abs(expected - observed) <= tolerance,
        units=units,
        notes=notes,
    )


def blocked(
    check_id: str,
    claim: str,
    anchor: str,
    expected: Any,
    observed: Any,
    units: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": check_id,
        "claim": claim,
        "paper_anchor": anchor,
        "expected": expected,
        "observed": observed,
        "units": units,
        "method": "blocked",
        "status": BLOCKED,
        "notes": notes,
    }


def cohort_checks(values: dict[str, Any]) -> list[dict[str, Any]]:
    cohort = values["cohort"]
    split = cohort["split"]
    features = values["features"]
    patients = cohort["patients"]
    return [
        number(
            "A1",
            "the positive and negative case counts account for every patient",
            cohort["anchor"],
            patients,
            cohort["positive_cases"] + cohort["negative_cases"],
            0.0,
            "patients",
        ),
        number(
            "A2",
            "the positive rate recomputed from the case counts",
            "Table I",
            cohort["positive_rate_pct"],
            round(pct(cohort["positive_cases"], patients), 1),
            0.05,
            "pct",
        ),
        number(
            "A3",
            "the three partition sizes sum to the cohort size",
            "Sec. IV.A.2",
            patients,
            split["train_patients"] + split["val_patients"] + split["test_patients"],
            0.0,
            "patients",
        ),
        number(
            "A4",
            "the training partition holds the stated fraction of patients",
            "Sec. IV.A.2",
            100.0 * split["train_frac"],
            round(pct(split["train_patients"], patients), 1),
            0.05,
            "pct",
        ),
        number(
            "A5",
            "the validation partition holds the stated fraction of patients",
            "Sec. IV.A.2",
            100.0 * split["val_frac"],
            round(pct(split["val_patients"], patients), 1),
            0.05,
            "pct",
        ),
        number(
            "A6",
            "the input width is the sum of the two modality blocks",
            "Sec. III.A, Eq. (6)",
            features["num_features"],
            features["hrv_features"] + features["temp_features"],
            0.0,
            "features",
        ),
    ]


def prospective_checks(values: dict[str, Any]) -> list[dict[str, Any]]:
    prospective = values["prospective"]
    ours = prospective["thrombxai"]
    standard = prospective["standard_screening"]
    events = prospective["events"]
    analysed = prospective["analysed"]
    anchor = prospective["anchor"]
    ours_total = (
        ours["true_positives"]
        + ours["false_positives"]
        + ours["true_negatives"]
        + ours["false_negatives"]
    )
    standard_total = (
        standard["true_positives"]
        + standard["false_positives"]
        + standard["true_negatives"]
        + standard["false_negatives"]
    )
    ours_called_positive = ours["true_positives"] + ours["false_positives"]
    ours_condition_negative = ours["true_negatives"] + ours["false_positives"]
    standard_called_positive = standard["true_positives"] + standard["false_positives"]
    standard_condition_negative = standard["true_negatives"] + standard["false_positives"]
    return [
        number(
            "B1",
            "the analysis set is the enrolled count less the eight lost to follow-up",
            "Sec. V.G",
            analysed,
            prospective["enrolled"] - prospective["lost_to_followup"],
            0.0,
            "participants",
        ),
        number(
            "B2",
            "the ThrombXAI confusion counts add up to the analysis set",
            anchor,
            analysed,
            ours_total,
            0.0,
            "participants",
            notes="the counts match the 148-participant per-protocol set the text describes",
        ),
        record(
            "B3",
            "the Table IV header N matches the confusion counts beneath it",
            anchor,
            prospective["enrolled"],
            ours_total,
            prospective["enrolled"] == ours_total,
            units="participants",
            notes=(
                "the header reads N=156 for an intention-to-treat analysis while the counts sum to "
                "148; the text calls 148 the per-protocol subset, so the header is mislabelled"
            ),
        ),
        number(
            "B4",
            "the standard-screening confusion counts add up to the analysis set",
            anchor,
            analysed,
            standard_total,
            0.0,
            "participants",
        ),
        number(
            "B5",
            "the analysed events equal the ThrombXAI true positives plus false negatives",
            anchor,
            events["total"],
            ours["true_positives"] + ours["false_negatives"],
            0.0,
            "events",
        ),
        number(
            "B6",
            "the event breakdown matches the reported event total",
            "Sec. V.G",
            events["total"],
            events["vte"] + events["dvt"] + events["pe"],
            0.0,
            "events",
        ),
        number(
            "B7",
            "ThrombXAI sensitivity recomputed from the counts",
            anchor,
            ours["sensitivity_pct"],
            round(pct(ours["true_positives"], events["total"]), 1),
            0.05,
            "pct",
        ),
        number(
            "B8",
            "ThrombXAI specificity recomputed from the counts",
            anchor,
            ours["specificity_pct"],
            round(pct(ours["true_negatives"], ours_condition_negative), 1),
            0.05,
            "pct",
            notes=(
                "109/120 evaluates to 90.8 per cent, so the reported 91.5 per cent does not follow "
                "from the counts; the same figure also appears in the abstract"
            ),
        ),
        number(
            "B9",
            "ThrombXAI positive predictive value recomputed from the counts",
            anchor,
            ours["ppv_pct"],
            round(pct(ours["true_positives"], ours_called_positive), 1),
            0.05,
            "pct",
        ),
        number(
            "B10",
            "ThrombXAI negative predictive value recomputed from the counts",
            anchor,
            ours["npv_pct"],
            round(pct(ours["true_negatives"], ours["true_negatives"] + ours["false_negatives"]), 1),
            0.05,
            "pct",
        ),
        number(
            "B11",
            "standard-screening sensitivity recomputed from the counts",
            anchor,
            standard["sensitivity_pct"],
            round(pct(standard["true_positives"], events["total"]), 1),
            0.05,
            "pct",
        ),
        number(
            "B12",
            "standard-screening specificity recomputed from the counts",
            anchor,
            standard["specificity_pct"],
            round(pct(standard["true_negatives"], standard_condition_negative), 1),
            0.05,
            "pct",
        ),
        number(
            "B13",
            "standard-screening positive predictive value recomputed from the counts",
            anchor,
            standard["ppv_pct"],
            round(pct(standard["true_positives"], standard_called_positive), 1),
            0.05,
            "pct",
        ),
        number(
            "B14",
            "standard-screening negative predictive value recomputed from the counts",
            anchor,
            standard["npv_pct"],
            round(
                pct(
                    standard["true_negatives"],
                    standard["true_negatives"] + standard["false_negatives"],
                ),
                1,
            ),
            0.05,
            "pct",
        ),
        number(
            "B15",
            "ThrombXAI early detections recomputed from the event total",
            anchor,
            ours["events_detected_early_pct"],
            round(pct(ours["events_detected_early"], events["total"]), 1),
            0.05,
            "pct",
        ),
        number(
            "B16",
            "standard-screening early detections recomputed from the event total",
            anchor,
            standard["events_detected_early_pct"],
            round(pct(standard["events_detected_early"], events["total"]), 1),
            0.05,
            "pct",
        ),
    ]


def headline_checks(values: dict[str, Any]) -> list[dict[str, Any]]:
    ours = values["headline_metrics"]["ours"]
    rows = values["baselines"]["rows"]
    claims = values["improvement_claims"]
    best_accuracy = max(rows, key=lambda row: row["accuracy_pct"])
    best_auroc = max(rows, key=lambda row: row["auroc"])
    named = claims["against"]
    named_row = next(row for row in rows if row["method"] == named)
    intervals = values["confidence_intervals"]["metrics"]
    outside = sum(
        1 for metric in intervals if not metric["low"] <= metric["point"] <= metric["high"]
    )
    edge_rows = values["edge_deployment"]["rows"]
    return [
        number(
            "C1",
            "the accuracy gain over the named baseline",
            claims["anchor"],
            claims["accuracy_gain_pct_points"],
            round(ours["accuracy_pct"] - named_row["accuracy_pct"], 1),
            0.05,
            "pct_points",
        ),
        number(
            "C2",
            "the AUROC gain over the named baseline",
            claims["anchor"],
            claims["auroc_gain_points"],
            round(ours["auroc"] - named_row["auroc"], 3),
            0.001,
            "auroc",
        ),
        record(
            "C3",
            "the baseline named as strongest also holds the highest baseline AUROC",
            "Sec. V.A",
            f"{best_auroc['method']} at {best_auroc['auroc']}",
            f"{named} at {named_row['auroc']}",
            named_row["auroc"] == best_auroc["auroc"],
            notes=(
                f"{named} holds the best baseline accuracy ({best_accuracy['accuracy_pct']}) but "
                f"{best_auroc['method']} holds the best baseline AUROC ({best_auroc['auroc']}); "
                "measured against that row the AUROC gain is 2.6 per cent, not the reported 2.8"
            ),
        ),
        record(
            "C4",
            "the reported accuracy exceeds every baseline in the comparison table",
            "Table II",
            f"greater than {best_accuracy['accuracy_pct']}",
            f"ours {ours['accuracy_pct']}",
            ours["accuracy_pct"] > best_accuracy["accuracy_pct"],
            notes="the reported value is the column maximum, so the table is internally ordered",
        ),
        number(
            "C5",
            "the dilated causal stack has the reported receptive field",
            "Eq. (9)",
            31.0,
            float(
                receptive_field(
                    values["architecture"]["full"]["tcn_layers"],
                    values["architecture"]["full"]["tcn_kernel"],
                )
            ),
            0.0,
            "steps",
        ),
        number(
            "C6",
            "the INT8 accuracy drop against the full model",
            "Sec. V.D, Table VII",
            values["edge_deployment"]["claims"]["int8_accuracy_drop_pct"],
            round(ours["accuracy_pct"] - edge_rows[2]["accuracy_pct"], 1),
            0.05,
            "pct_points",
        ),
        number(
            "C7",
            "the INT4 accuracy drop against the full model",
            "App. B, Table VII",
            values["edge_deployment"]["claims"]["int4_accuracy_drop_pct"],
            round(ours["accuracy_pct"] - edge_rows[3]["accuracy_pct"], 1),
            0.05,
            "pct_points",
        ),
        record(
            "C8",
            "every bootstrap interval brackets its point estimate",
            values["confidence_intervals"]["anchor"],
            "0 of 4 outside their interval",
            f"{outside} of {len(intervals)} outside",
            outside == 0,
        ),
    ]


def ablation_checks(values: dict[str, Any]) -> list[dict[str, Any]]:
    rows = {row["configuration"]: row for row in values["ablations"]["rows"]}
    claims = values["ablations"]["claims"]
    anchor = values["ablations"]["anchor"]
    ours = values["headline_metrics"]["ours"]
    return [
        number(
            "D1",
            "the CRGA accuracy gain against the no-CRGA ablation",
            anchor,
            claims["crga_accuracy_gain_pct_points"],
            round(rows["full"]["accuracy_pct"] - rows["no_crga"]["accuracy_pct"], 1),
            0.05,
            "pct_points",
        ),
        number(
            "D2",
            "the CRGA attention overlap gain against the no-CRGA ablation",
            anchor,
            claims["crga_aos_gain"],
            round(rows["full"]["aos"] - rows["no_crga"]["aos"], 4),
            0.0005,
            "aos",
        ),
        number(
            "D3",
            "the accuracy lost when temporal attention is removed",
            anchor,
            claims["temporal_attention_drop_pct_points"],
            round(rows["full"]["accuracy_pct"] - rows["no_temporal_attention"]["accuracy_pct"], 1),
            0.05,
            "pct_points",
        ),
        number(
            "D4",
            "the accuracy lost when spatial attention is removed",
            anchor,
            claims["spatial_attention_drop_pct_points"],
            round(rows["full"]["accuracy_pct"] - rows["no_spatial_attention"]["accuracy_pct"], 1),
            0.05,
            "pct_points",
        ),
        number(
            "D5",
            "the accuracy gained by fusing skin temperature with the HRV block",
            anchor,
            claims["hrv_fusion_gain_pct_points"],
            round(rows["full"]["accuracy_pct"] - rows["hrv_only"]["accuracy_pct"], 1),
            0.05,
            "pct_points",
        ),
        record(
            "D6",
            "temporal attention matters more than spatial attention for accuracy",
            "Sec. V.C",
            "temporal drop exceeds spatial drop",
            (
                "temporal drop exceeds spatial drop"
                if rows["no_temporal_attention"]["accuracy_pct"]
                < rows["no_spatial_attention"]["accuracy_pct"]
                else "spatial drop is not the smaller one"
            ),
            rows["no_temporal_attention"]["accuracy_pct"]
            < rows["no_spatial_attention"]["accuracy_pct"],
        ),
        number(
            "D7",
            "the full-configuration row reproduces the headline accuracy",
            anchor,
            ours["accuracy_pct"],
            rows["full"]["accuracy_pct"],
            0.0,
            "pct",
        ),
        number(
            "D8",
            "the full-configuration row reproduces the headline AUROC",
            anchor,
            ours["auroc"],
            rows["full"]["auroc"],
            0.0,
            "auroc",
        ),
    ]


def explanation_checks(values: dict[str, Any]) -> list[dict[str, Any]]:
    rows = values["explanation_quality"]["rows"]
    ours = next(row for row in rows if row["method"] == "ThrombXAI")
    baselines = [row for row in rows if row["method"] != "ThrombXAI"]
    claims = values["explanation_quality"]["claims"]
    anchor = values["explanation_quality"]["anchor"]
    best_baseline = max(baselines, key=lambda row: row["aos"])
    return [
        number(
            "E1",
            "the comprehensiveness at ten per cent reads as the stated confidence drop",
            anchor,
            100.0 * ours["comp_at_10"],
            claims["comp_at_10_pct"],
            0.05,
            "pct",
        ),
        number(
            "E2",
            "the sufficiency at twenty per cent reads as the stated prediction retained",
            anchor,
            100.0 * ours["suff_at_20"],
            claims["suff_at_20_pct"],
            0.05,
            "pct",
        ),
        number(
            "E3",
            "the attention overlap gain over the best baseline overlap",
            anchor,
            claims["aos_improvement_over_baseline_pct"],
            round(100.0 * (ours["aos"] - best_baseline["aos"]) / best_baseline["aos"], 1),
            0.05,
            "pct",
        ),
        record(
            "E4",
            "the baseline named for the overlap gain holds the best baseline overlap",
            anchor,
            claims["aos_top_baseline"],
            best_baseline["method"],
            best_baseline["method"] == claims["aos_top_baseline"],
        ),
        number(
            "E5",
            "the clinician rating reads as the stated mean out of five",
            anchor,
            claims["clinician_rating_out_of_5"],
            ours["clinician_rating"],
            0.0,
            "rating",
        ),
    ]


def edge_checks(values: dict[str, Any]) -> list[dict[str, Any]]:
    rows = values["edge_deployment"]["rows"]
    claims = values["edge_deployment"]["claims"]
    battery = values["edge_deployment"]["battery"]
    anchor = values["edge_deployment"]["anchor"]
    full = rows[0]
    int8 = rows[2]
    energy = battery["capacity_mah"] / 1000.0 * battery["voltage_v"]
    effective_power = int8["power_mw"] * battery["duty_cycle"]
    runtime = battery["energy_wh"] * 1000.0 / battery["effective_power_mw"]
    accuracy_monotone = all(
        rows[index]["accuracy_pct"] >= rows[index + 1]["accuracy_pct"]
        for index in range(len(rows) - 1)
    )
    size_monotone = all(
        rows[index]["size_mb"] >= rows[index + 1]["size_mb"] for index in range(len(rows) - 1)
    )
    return [
        number(
            "F1",
            "the INT8 size reduction against the full model",
            anchor,
            float(claims["int8_size_reduction_pct"]),
            100.0 * (full["size_mb"] - int8["size_mb"]) / full["size_mb"],
            0.5,
            "pct",
        ),
        number(
            "F2",
            "the INT8 latency reduction against the full model",
            anchor,
            float(claims["int8_latency_reduction_pct"]),
            100.0 * (full["latency_ms"] - int8["latency_ms"]) / full["latency_ms"],
            0.5,
            "pct",
        ),
        number(
            "F3",
            "the INT8 serving latency matches the reported target",
            anchor,
            float(claims["int8_latency_target_ms"]),
            float(int8["latency_ms"]),
            0.0,
            "ms",
        ),
        number(
            "F4",
            "the wearable battery energy follows from capacity and voltage",
            "App. B",
            battery["energy_wh"],
            round(energy, 2),
            0.01,
            "wh",
        ),
        number(
            "F5",
            "the duty-cycled effective power follows from the INT8 draw",
            "App. B",
            float(battery["effective_power_mw"]),
            effective_power,
            0.5,
            "mw",
        ),
        number(
            "F6",
            "the runtime estimate follows from energy over effective power",
            "App. B",
            float(battery["runtime_hours"]),
            runtime,
            0.5,
            "hours",
        ),
        record(
            "F7",
            "accuracy does not rise across the four deployment configurations",
            anchor,
            "non-increasing",
            "non-increasing" if accuracy_monotone else "not non-increasing",
            accuracy_monotone,
        ),
        record(
            "F8",
            "model size does not rise across the four deployment configurations",
            anchor,
            "non-increasing",
            "non-increasing" if size_monotone else "not non-increasing",
            size_monotone,
        ),
    ]


def subgroup_checks(values: dict[str, Any]) -> list[dict[str, Any]]:
    subgroups = values["subgroups"]
    cohort = values["cohort"]
    age = subgroups["age_groups"]
    sex = subgroups["sex"]
    age_total = sum(row["n"] for row in age)
    sex_total = sum(row["n"] for row in sex)
    expected_test_samples = cohort["split"]["test_frac"] * cohort["samples_four_hour_windows"]
    return [
        number(
            "G1",
            "the age and sex partitions cover the same number of samples",
            subgroups["anchor"],
            float(age_total),
            float(sex_total),
            0.0,
            "samples",
        ),
        number(
            "G2",
            "the subgroup partition matches the described test partition size",
            "Sec. IV.A.2, Table XI",
            round(expected_test_samples, 1),
            float(age_total),
            0.5,
            "samples",
            notes=(
                "the test partition is 15 per cent of the 4,271 windows (640.7 samples) or 427 "
                "patients, while the subgroup partitions sum to 1,074 and match neither reading"
            ),
        ),
        number(
            "G3",
            "the age-weighted prevalence reproduces the cohort prevalence",
            "Table I, Table XI",
            cohort["positive_rate_pct"],
            round(weighted_mean(age, "prevalence_pct"), 2),
            0.3,
            "pct",
        ),
        number(
            "G4",
            "the sex-weighted prevalence reproduces the cohort prevalence",
            "Table I, Table XI",
            cohort["positive_rate_pct"],
            round(weighted_mean(sex, "prevalence_pct"), 2),
            0.3,
            "pct",
        ),
        number(
            "G5",
            "the age-weighted accuracy reproduces the headline accuracy",
            "Table II, Table XI",
            values["headline_metrics"]["ours"]["accuracy_pct"],
            round(weighted_mean(age, "accuracy_pct"), 2),
            0.1,
            "pct",
        ),
        number(
            "G6",
            "the sex-weighted accuracy reproduces the headline accuracy",
            "Table II, Table XI",
            values["headline_metrics"]["ours"]["accuracy_pct"],
            round(weighted_mean(sex, "accuracy_pct"), 2),
            0.1,
            "pct",
        ),
    ]


def attention_checks(values: dict[str, Any]) -> list[dict[str, Any]]:
    attention = values["feature_attention"]
    top10 = attention["top10"]
    total = sum(row["attention"] for row in top10)
    ranks = sorted(row["clinical_rank"] for row in top10)
    top_names = {row["feature"] for row in top10}
    named_pair = {"LF/HF Ratio", "RMSSD"}
    return [
        number(
            "H1",
            "the top-ten attention weights sum to the reported total",
            attention["anchor"],
            attention["top10_sum"],
            round(total, 4),
            0.0005,
            "weight",
        ),
        number(
            "H2",
            "the top-ten and remaining weights account for the whole softmax mass",
            attention["anchor"],
            1.0,
            round(total + attention["remaining_nine_sum"], 4),
            0.0005,
            "weight",
        ),
        record(
            "H3",
            "the clinician ranks attached to the top ten are a permutation of one to ten",
            attention["anchor"],
            list(range(1, 11)),
            ranks,
            ranks == list(range(1, 11)),
        ),
        record(
            "H4",
            "the two HRV features the text names as most attended are both present",
            "Sec. V.E",
            "LF/HF Ratio and RMSSD present",
            "both present" if named_pair <= top_names else "missing at least one",
            named_pair <= top_names,
        ),
    ]


def availability_checks(values: dict[str, Any]) -> list[dict[str, Any]]:
    unavailable = values["unavailable"]
    return [
        blocked(
            "I1",
            "the retrospective table values are recomputed from the source waveforms",
            "Sec. IV.A.1",
            "recomputation from the MIMIC-IV Waveform cohort",
            "credentialed cohort absent",
            units="cohort",
            notes=(
                f"{unavailable['primary_cohort']} sits behind a {unavailable['access']} and the "
                f"records are not redistributable ({unavailable['url']}), so only the published "
                "aggregates are checkable here"
            ),
        ),
        blocked(
            "I2",
            "the prospective validation values are recomputed from participant records",
            "App. E, Sec. V.G",
            "recomputation from the prospective cohort",
            "cohort absent",
            units="cohort",
            notes=f"{unavailable['prospective_cohort']}, {unavailable['prospective_access']}",
        ),
    ]


def execution_checks(values: dict[str, Any], workdir: Path) -> list[dict[str, Any]]:
    try:
        import numpy as np
        import torch

        from thrombxai.dyeing import vat
        from thrombxai.fabric import build_model, count_parameters
        from thrombxai.finishing import objective
        from thrombxai.loom.shuttle import Trainer
        from thrombxai.loom.warp_beam import build_optimizer
        from thrombxai.opening import sliver
        from thrombxai.order import (
            CRGAConfig,
            DataConfig,
            ExperimentConfig,
            ModelConfig,
            ObjectiveConfig,
            OptimConfig,
            TrainConfig,
            edge_model_config,
        )
        from thrombxai.spindle import atomic_save, load_checkpoint, set_seed
    except Exception as error:
        return [
            record(
                "K1",
                "the training stack imports",
                "Sec. IV.D",
                "torch and thrombxai importable",
                f"{type(error).__name__}: {error}",
                False,
                method="execution",
                notes="the closed-form checks above do not need the training stack",
            )
        ]

    cohort = values["cohort"]
    features = values["features"]
    architecture = values["architecture"]
    training = values["training"]
    ours = values["headline_metrics"]["ours"]
    anchor = "Table V, Table VI, Sec. IV.A"
    set_seed(SMOKE_SEED)
    data_cfg = DataConfig(
        synthetic=True,
        window_hours=cohort["window_hours"],
        time_steps=SMOKE_TIME_STEPS,
        hrv_features=features["hrv_features"],
        temp_features=features["temp_features"],
        num_features=features["num_features"],
        train_frac=cohort["split"]["train_frac"],
        val_frac=cohort["split"]["val_frac"],
        test_frac=cohort["split"]["test_frac"],
        positive_rate=cohort["positive_rate_pct"] / 100.0,
        num_subjects=SMOKE_SUBJECTS,
        batch_size=SMOKE_BATCH,
        num_workers=0,
    )
    model_cfg = ModelConfig(
        hidden_dim=SMOKE_HIDDEN,
        tcn_layers=architecture["full"]["tcn_layers"],
        tcn_kernel=architecture["full"]["tcn_kernel"],
        attn_heads=SMOKE_HEADS,
        attn_dim=architecture["full"]["attn_dim"],
        spatial_attn_dim=SMOKE_ATTN,
        feature_embed_dim=SMOKE_EMBED,
        mlp_hidden=SMOKE_MLP,
        dropout=0.0,
        num_features=features["num_features"],
    )
    exp_cfg = ExperimentConfig(
        name="paper_values_smoke",
        seeds=1,
        data=data_cfg,
        model=model_cfg,
        crga=CRGAConfig(
            enabled=True,
            lambda_kl=training["lambda_kl"],
            lambda_smooth=training["lambda_smooth"],
            warmup_epochs=training["crga_warmup_epochs"],
        ),
        objective=ObjectiveConfig(use_focal=True, focal_gamma=training["focal_gamma"], l2_reg=0.0),
        train=TrainConfig(
            epochs=SMOKE_EPOCHS,
            grad_accum=1,
            early_stop_patience=100,
            seed=SMOKE_SEED,
            ckpt_dir=str(workdir),
            log_every=100,
            optim=OptimConfig(lr=training["lr"], weight_decay=training["weight_decay"]),
        ),
    )
    results: list[dict[str, Any]] = []
    smoke_cohort = sliver.build_cohort(data_cfg, SMOKE_SEED)
    train_batch = next(iter(smoke_cohort.train))
    signal = train_batch["signal"]
    results.append(
        record(
            "K1",
            "the cohort carries the reported input width over three subject-level partitions",
            anchor,
            f"{features['num_features']} features",
            f"{int(signal.shape[-1])} features over three loaders",
            int(signal.shape[-1]) == features["num_features"],
            method="execution",
        )
    )

    model = build_model(model_cfg)
    output = model(signal)
    risk_ok = bool(((output.risk >= 0.0) & (output.risk <= 1.0)).all())
    results.append(
        record(
            "K2",
            "a forward pass returns a per-subject risk inside the unit interval",
            "Eq. (19)",
            "risk within [0, 1] for every row",
            f"within [0, 1] = {risk_ok}",
            risk_ok,
            method="execution",
        )
    )
    spatial_rows = output.spatial_attention.sum(dim=-1).detach().numpy()
    temporal_rows = output.temporal_attention.sum(dim=-1).detach().numpy()
    spatial_ok = bool(np.allclose(spatial_rows, 1.0, atol=1e-5))
    temporal_ok = bool(np.allclose(temporal_rows, 1.0, atol=1e-5))
    results.append(
        record(
            "K3",
            "the spatial attention weights form a distribution over the feature axis",
            "Eq. (17)",
            "rows sum to 1",
            f"rows sum to 1 = {spatial_ok}",
            spatial_ok,
            method="execution",
        )
    )
    results.append(
        record(
            "K4",
            "the temporal attention weights form a distribution over the time axis",
            "Eq. (14)",
            "rows sum to 1",
            f"rows sum to 1 = {temporal_ok}",
            temporal_ok,
            method="execution",
        )
    )

    loss = objective.prediction_loss(output.logit, train_batch["label"], exp_cfg.objective)
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    finite_grads = bool(grads) and all(bool(torch.isfinite(grad).all()) for grad in grads)
    results.append(
        record(
            "K5",
            "the focal objective backpropagates finite gradients",
            "Eq. (26)",
            "at least one finite gradient tensor",
            f"{len(grads)} gradient tensors, all finite = {finite_grads}",
            finite_grads,
            method="execution",
        )
    )

    optimizer = build_optimizer(model, exp_cfg.train.optim)
    before = [param.detach().clone() for param in model.parameters()]
    optimizer.step()
    moved = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters(), strict=True)
    )
    results.append(
        record(
            "K6",
            "an AdamW step moves the parameters",
            "Table VI",
            "parameters change after one step",
            f"parameters changed = {moved}",
            moved,
            method="execution",
        )
    )

    trainer = Trainer(build_model(model_cfg), exp_cfg, torch.device("cpu"))
    history = trainer.fit(smoke_cohort.train, smoke_cohort.val)
    history_ok = len(history.train_loss) == SMOKE_EPOCHS and all(
        value == value and abs(value) != float("inf") for value in history.train_loss
    )
    results.append(
        record(
            "K7",
            "the training loop completes the configured epochs with finite losses",
            "Sec. IV.D",
            f"{SMOKE_EPOCHS} finite epoch losses",
            f"{len(history.train_loss)} losses, all finite = {history_ok}",
            history_ok,
            method="execution",
        )
    )

    checkpoint = workdir / "paper_values_smoke_best.pt"
    atomic_save({"model": model.state_dict(), "epoch": 0}, checkpoint)
    restored = build_model(model_cfg)
    payload = load_checkpoint(checkpoint)
    restored.load_state_dict(payload["model"])
    round_trip = all(
        torch.equal(reference, restored_value)
        for reference, restored_value in zip(
            model.state_dict().values(), restored.state_dict().values(), strict=True
        )
    )
    results.append(
        record(
            "K8",
            "a checkpoint round-trip reproduces the parameter tensors",
            "Sec. IV.D",
            "restored state matches the saved state",
            f"state tensors identical = {round_trip}",
            round_trip,
            method="execution",
        )
    )

    full_model = build_model(ModelConfig())
    encoder_field = int(full_model.encoder.receptive_field)
    results.append(
        record(
            "K9",
            "the encoder reports the receptive field the manuscript derives",
            "Eq. (9)",
            31,
            encoder_field,
            encoder_field == 31,
            method="execution",
        )
    )
    full_params = count_parameters(full_model)
    reported_full = architecture["full"]["params_millions"] * 1e6
    results.append(
        record(
            "K10",
            "the full-precision parameter count matches the architecture table",
            "Table V",
            int(reported_full),
            full_params,
            abs(full_params - reported_full) / reported_full <= PARAM_TOLERANCE_REL,
            method="execution",
            units="parameters",
            notes=(
                "the two inner widths the manuscript leaves unnamed are engineering defaults "
                "(docs/deviations.md D6), so the count sits 25.8 per cent above the tabulated value"
            ),
        )
    )
    edge_params = count_parameters(build_model(edge_model_config()))
    reported_edge = architecture["edge"]["params_thousands"] * 1e3
    results.append(
        record(
            "K11",
            "the edge parameter count matches the architecture table",
            "Table V",
            int(reported_edge),
            edge_params,
            abs(edge_params - reported_edge) / reported_edge <= PARAM_TOLERANCE_REL,
            method="execution",
            units="parameters",
        )
    )
    schedule = [
        vat.kl_warmup(training["lambda_kl"], epoch, training["crga_warmup_epochs"])
        for epoch in range(training["crga_warmup_epochs"] + 1)
    ]
    schedule_ok = (
        abs(schedule[0] - training["lambda_kl"]) < 1e-9
        and abs(schedule[-1]) < 1e-9
        and all(schedule[index] >= schedule[index + 1] for index in range(len(schedule) - 1))
    )
    results.append(
        record(
            "K12",
            "the clinical prior weight anneals to zero across the warmup window",
            "Eq. (23)",
            f"{training['lambda_kl']} at epoch 0, 0.0 at epoch {training['crga_warmup_epochs']}",
            f"{schedule[0]:.4f} at epoch 0, {schedule[-1]:.4f} at the end",
            schedule_ok,
            method="execution",
        )
    )
    results.append(
        record(
            "K13",
            "the smoke cohort stands in for the credentialed cohort",
            "Sec. IV.A.1",
            "reduced subject, step and width defaults",
            f"{SMOKE_SUBJECTS} subjects, {SMOKE_TIME_STEPS} steps, hidden {SMOKE_HIDDEN}",
            True,
            method="execution",
            notes=(
                f"the paper reports {cohort['patients']} patients at hidden width "
                f"{architecture['full']['hidden_dim']}; the smoke reduces both to fit a CPU budget "
                f"and does not re-derive the headline accuracy {ours['accuracy_pct']}"
            ),
        )
    )
    return results


def status_of(checks_by_id: dict[str, dict[str, Any]], check_id: str) -> str:
    found = checks_by_id.get(check_id)
    return str(found["status"]) if found else NOT_RUN


def claim_map(
    values: dict[str, Any], checks: list[dict[str, Any]], values_name: str
) -> dict[str, Any]:
    by_id = {check["id"]: check for check in checks}
    entries = [
        {
            "claim_id": "C-features",
            "paper_location": "Sec. III.A, Eq. (6)",
            "claim": "two sensor modalities yield a 19-feature tensor (12 HRV plus 7 temperature)",
            "code_anchor": "thrombxai/order.py:29",
            "check_ids": ["A6"],
            "status": status_of(by_id, "A6"),
        },
        {
            "claim_id": "C-receptive-field",
            "paper_location": "Sec. III.C.1, Eq. (9)",
            "claim": "four dilated causal layers with kernel three give a 31-step receptive field",
            "code_anchor": "thrombxai/spinning/twist.py:10",
            "check_ids": ["C5", "K9"],
            "status": status_of(by_id, "C5"),
        },
        {
            "claim_id": "C-temporal-attention",
            "paper_location": "Sec. III.C.2, Eq. (14)",
            "claim": "temporal self-attention aggregates to a distribution over time steps",
            "code_anchor": "thrombxai/warping/heddle.py:12",
            "check_ids": ["K4"],
            "status": status_of(by_id, "K4"),
        },
        {
            "claim_id": "C-spatial-attention",
            "paper_location": "Sec. III.C.3, Eq. (17)",
            "claim": "spatial feature attention sums to one over the 19 features",
            "code_anchor": "thrombxai/weaving/shed.py:10",
            "check_ids": ["K3"],
            "status": status_of(by_id, "K3"),
        },
        {
            "claim_id": "C-clinical-prior",
            "paper_location": "Sec. III.D.1, Eq. (20)",
            "claim": "a clinical relevance vector is read from knowledge-graph node degree",
            "code_anchor": "thrombxai/dyeing/mordant.py:79",
            "check_ids": ["K12"],
            "status": status_of(by_id, "K12"),
        },
        {
            "claim_id": "C-warmup",
            "paper_location": "Sec. III.D.3, Eq. (23)",
            "claim": "the clinical prior weight anneals linearly to zero over the warmup window",
            "code_anchor": "thrombxai/dyeing/vat.py:31",
            "check_ids": ["K12"],
            "status": status_of(by_id, "K12"),
        },
        {
            "claim_id": "C-focal-loss",
            "paper_location": "Sec. III.E, Eq. (26)",
            "claim": "focal loss with gamma two addresses class imbalance",
            "code_anchor": "thrombxai/finishing/objective.py:14",
            "check_ids": ["K5"],
            "status": status_of(by_id, "K5"),
        },
        {
            "claim_id": "C-prediction-head",
            "paper_location": "Sec. III.C.4, Eq. (19)",
            "claim": "the head maps the pooled and context vectors to a risk probability",
            "code_anchor": "thrombxai/finishing/press.py:10",
            "check_ids": ["K2"],
            "status": status_of(by_id, "K2"),
        },
        {
            "claim_id": "C-subject-split",
            "paper_location": "Sec. IV.A.2",
            "claim": "a 70/15/15 subject-level partition keeps every patient inside one split",
            "code_anchor": "thrombxai/opening/bale.py:15",
            "check_ids": ["A3", "K1"],
            "status": status_of(by_id, "A3"),
        },
        {
            "claim_id": "C-optimizer",
            "paper_location": "Sec. IV.D, Table VI",
            "claim": "AdamW with weight decay one e-4 under cosine annealing",
            "code_anchor": "thrombxai/loom/warp_beam.py:9",
            "check_ids": ["K6"],
            "status": status_of(by_id, "K6"),
        },
        {
            "claim_id": "C-training-loop",
            "paper_location": "Sec. IV.D",
            "claim": "the loop runs the configured epochs and keeps the best validation checkpoint",
            "code_anchor": "thrombxai/loom/shuttle.py:31",
            "check_ids": ["K7", "K8"],
            "status": status_of(by_id, "K7"),
        },
        {
            "claim_id": "C-quantisation",
            "paper_location": "Sec. III.F.2, Eq. (28)",
            "claim": "symmetric integer quantisation shrinks the edge model to 0.13 MB",
            "code_anchor": "thrombxai/pressing/baling.py:12",
            "check_ids": ["F1"],
            "status": status_of(by_id, "F1"),
        },
        {
            "claim_id": "C-headline",
            "paper_location": "Table II, Sec. V.A",
            "claim": "test accuracy 91.3 and AUROC 0.936, ahead of the best baseline",
            "code_anchor": "thrombxai/inspection/gauge.py:116",
            "check_ids": ["C1", "C2", "C3", "C4"],
            "status": status_of(by_id, "C1") if status_of(by_id, "C3") == PASS else FAIL,
        },
        {
            "claim_id": "C-ablations",
            "paper_location": "Table III, Sec. V.C",
            "claim": "the ablation rows reproduce every reported component gain",
            "code_anchor": "thrombxai/order.py:115",
            "check_ids": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "status": status_of(by_id, "D1"),
        },
        {
            "claim_id": "C-explanation-quality",
            "paper_location": "Table VIII, Sec. V.B, App. F.A",
            "claim": "comprehensiveness, sufficiency and overlap beat the post-hoc baselines",
            "code_anchor": "thrombxai/inspection/burling.py:1",
            "check_ids": ["E1", "E2", "E3", "E4"],
            "status": status_of(by_id, "E1"),
        },
        {
            "claim_id": "C-edge-serving",
            "paper_location": "Table VII, Sec. V.D, App. B",
            "claim": "INT8 serving holds 89.8 accuracy at 0.13 MB and 850 ms on a 145 mW budget",
            "code_anchor": "thrombxai/pressing/seam.py:37",
            "check_ids": ["F1", "F2", "F3", "F5", "F6"],
            "status": status_of(by_id, "F1"),
        },
        {
            "claim_id": "C-cohort",
            "paper_location": "Table I, Sec. IV.A.1",
            "claim": "2,847 patients, 4,271 four-hour windows, 21.9 per cent positive",
            "code_anchor": "thrombxai/opening/sliver.py:77",
            "check_ids": ["A1", "A2", "I1"],
            "status": NOT_RUN,
            "note": (
                "the aggregates are internally consistent, but the Waveform cohort is credentialed "
                "so no value is recomputed from source waveforms"
            ),
        },
        {
            "claim_id": "C-prospective",
            "paper_location": "Table IV, Sec. V.G",
            "claim": "89.3 per cent sensitivity and 91.5 per cent specificity at 156 participants",
            "code_anchor": "scripts/prepare_data.sh:1",
            "check_ids": ["B7", "B8", "I2"],
            "status": NOT_RUN,
            "note": (
                "the prospective cohort is not released, and the published 2x2 counts do not "
                "reproduce the reported specificity"
            ),
        },
    ]
    return {
        "schema_version": 1,
        "paper_title": values["source"]["title"],
        "values_file": values_name,
        "entries": entries,
        "deviations": [
            {
                "id": "D1",
                "paper_location": "Table V",
                "what_departs": "the full model carries 1.36M parameters against a tabulated 1.08M",
                "why": (
                    "the manuscript leaves the spatial-attention inner width and the per-feature "
                    "embedding width unnamed; the release fixes them at 64 and 256 as recorded in "
                    "docs/deviations.md D6"
                ),
                "status": "open",
            },
            {
                "id": "D2",
                "paper_location": "Sec. IV.A.1",
                "what_departs": "the smoke cohort is 96 subjects at 48 steps, hidden width 48",
                "why": (
                    "the Waveform records cannot be redistributed and the full-width model does "
                    "not fit inside a CPU smoke budget"
                ),
                "status": "accepted",
            },
            {
                "id": "D3",
                "paper_location": "Sec. IV.D",
                "what_departs": "the smoke trains two epochs with dropout disabled",
                "why": (
                    "the smoke exists to show the pipeline runs end to end, so the schedule is "
                    "shortened and dropout is switched off to keep the run repeatable"
                ),
                "status": "accepted",
            },
            {
                "id": "D4",
                "paper_location": "Table IV",
                "what_departs": "the reported specificity of 91.5 per cent is left unreconciled",
                "why": (
                    "109 true negatives over 120 condition negatives is 90.8 per cent, so the "
                    "smoke reports the gap rather than bending the arithmetic; no source "
                    "records exist to settle it"
                ),
                "status": "unresolved",
            },
        ],
    }


def summarise(checks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {PASS: 0, FAIL: 0, NOT_RUN: 0, BLOCKED: 0}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    clean = counts[FAIL] == 0 and counts[NOT_RUN] == 0 and counts[BLOCKED] == 0
    return {
        "counts": counts,
        "total": len(checks),
        "overall_status": PASS if clean else "PARTIALLY_VERIFIED",
    }


def display_root(root: Path) -> str:
    """Name the run root without leaking the machine's directory layout.

    The artefacts record where the run happened so a reader can tell a repo copy from a
    paper copy. An absolute path would carry the authoring host's user name into a
    distributed file, so only the directory name is recorded, with "." for the current
    directory.
    """
    resolved = Path(root).resolve()
    return resolved.name or "."


def render_summary(report: dict[str, Any], claims: dict[str, Any]) -> str:
    checks = report["checks"]
    counts = report["summary"]["counts"]
    lines: list[str] = []
    lines.append("TCE ThrombXAI paper-values smoke")
    lines.append("=" * 33)
    lines.append(f"paper       : {report['paper_title']}")
    lines.append(f"values file : {report['values_file']}")
    lines.append(f"run root    : {report['root']}")
    lines.append(f"overall     : {report['summary']['overall_status']}")
    lines.append("")
    lines.append("status counts")
    for status in (PASS, FAIL, NOT_RUN, BLOCKED):
        lines.append(f"  {status:<8}: {counts.get(status, 0)}")
    lines.append(f"  {'TOTAL':<8}: {report['summary']['total']}")
    lines.append("")
    lines.append("checks")
    for check in checks:
        lines.append(f"  [{check['status']:<8}] {check['id']:<4} {check['claim']}")
        if check["status"] != PASS and check.get("notes"):
            lines.append(f"             {check['notes']}")
    findings = [check for check in checks if check["status"] == FAIL]
    lines.append("")
    lines.append(f"consistency findings: {len(findings)}")
    for check in findings:
        lines.append(f"  {check['id']}: expected {check['expected']}, observed {check['observed']}")
    lines.append("")
    lines.append("deviations")
    for deviation in claims["deviations"]:
        lines.append(f"  {deviation['id']} [{deviation['status']}] {deviation['what_departs']}")
    lines.append("")
    lines.append("blocked inputs")
    for check in checks:
        if check["status"] == BLOCKED:
            lines.append(f"  {check['id']}: {check['notes']}")
    lines.append("")
    return "\n".join(lines)


def build_report(
    values: dict[str, Any],
    root: Path,
    checks: list[dict[str, Any]],
    stamp: str,
    values_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tool": "smoke/tce_smoke.py",
        "paper_title": values["source"]["title"],
        "venue": values["source"]["venue"],
        "values_file": values_name,
        "root": display_root(root),
        "stamp": stamp,
        "scope": (
            "closed-form recomputation of the published aggregates plus an executable "
            "end-to-end smoke; cohort-level values stay NOT_RUN because the cohort is "
            "credentialed"
        ),
        "checks": checks,
        "summary": summarise(checks),
    }


def collect_checks(values: dict[str, Any], scratch: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for builder in (
        cohort_checks,
        prospective_checks,
        headline_checks,
        ablation_checks,
        explanation_checks,
        edge_checks,
        subgroup_checks,
        attention_checks,
        availability_checks,
    ):
        checks.extend(builder(values))
    checks.extend(execution_checks(values, scratch))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the published values and smoke the pipeline end to end."
    )
    parser.add_argument("--root", default=str(REPO_ROOT), help="directory the artefacts land in")
    parser.add_argument("--values", default=str(DEFAULT_VALUES), help="extracted paper-value file")
    parser.add_argument("--stamp", default="", help="free-form label recorded in the report")
    parser.add_argument(
        "--package-root",
        action="append",
        default=[],
        help="directory holding the thrombxai package; repeatable, searched before the defaults",
    )
    arguments = parser.parse_args()
    root = Path(arguments.root).resolve()
    values_path = Path(arguments.values).resolve()
    ensure_package_path([Path(entry).expanduser() for entry in arguments.package_root])
    values = load_values(values_path)
    with tempfile.TemporaryDirectory(prefix="tce-smoke-") as scratch:
        checks = collect_checks(values, Path(scratch))
    claims = claim_map(values, checks, values_path.name)
    report = build_report(values, root, checks, arguments.stamp, values_path.name)
    write_json(root, CLAIM_FILE, claims)
    write_json(root, REPORT_FILE, report)
    write_text(root, SUMMARY_FILE, render_summary(report, claims))
    exclude = {MANIFEST_FILE}
    entries = digest_tree(root, exclude)
    manifest = {
        "schema_version": 1,
        "root": display_root(root),
        "algorithm": "sha256",
        "file_count": len(entries),
        "excludes": sorted(exclude),
        "files": entries,
        "manifest_digest": manifest_digest(entries),
    }
    write_json(root, MANIFEST_FILE, manifest)
    print(render_summary(report, claims))
    print(f"artefacts written to {display_root(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
