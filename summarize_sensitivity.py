#!/usr/bin/env python3
"""Validate and aggregate ARC-BPO hyperparameter-sensitivity results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from sensitivity.common import TASKS, read_json, read_manifest, write_csv, write_json


ALL_RUN_FIELDS = (
    "sweep",
    "parameter",
    "value",
    "numeric_value",
    "seed",
    *TASKS,
    "average",
    "noise_rate",
    "checkpoint",
    "config_hash",
    "scientific_hash",
    "evaluation_protocol_hash",
    "result_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluated sensitivity runs into CSV, JSON, PDF, LaTeX, and Markdown."
    )
    parser.add_argument("--manifest", default="outputs/sensitivity/run_manifest.csv")
    parser.add_argument(
        "--main_result",
        help=(
            "Optional main-run JSON containing average and expected_variance (or pass "
            "--expected_variance) for the mandatory default reproduction check."
        ),
    )
    parser.add_argument("--expected_variance", type=float)
    parser.add_argument(
        "--published_anchors",
        help=(
            "JSON containing the fixed clean and noise20 task scores from the current "
            "spec. These rows are added only at the exact default points."
        ),
    )
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def _float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def load_run_results(
    manifest_path: Path,
    allow_missing: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    protocol_hashes = set()
    for manifest_row in read_manifest(manifest_path):
        result_path = Path(manifest_row["run_dir"]) / "evaluation_metrics.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        payload = read_json(result_path)
        task_values = {
            task: _float(payload.get("tasks", {}).get(task), f"{payload.get('run_name')}:{task}")
            for task in TASKS
        }
        recomputed_average = sum(task_values.values()) / len(task_values)
        stored_average = _float(payload.get("average"), f"{payload.get('run_name')}:average")
        if not math.isclose(recomputed_average, stored_average, rel_tol=0, abs_tol=1e-9):
            raise ValueError(
                f"Stored average mismatch in {result_path}: "
                f"stored={stored_average}, recomputed={recomputed_average}."
            )
        for key in ("run_name", "sweep", "value", "seed", "noise_rate", "config_hash"):
            if str(payload.get(key)) != str(manifest_row.get(key)):
                raise ValueError(
                    f"Result/manifest mismatch for {key} in {result_path}: "
                    f"{payload.get(key)!r} != {manifest_row.get(key)!r}."
                )
        protocol_hash = str(payload.get("evaluation_protocol_hash", ""))
        if not protocol_hash:
            raise ValueError(f"Missing evaluation protocol hash: {result_path}")
        protocol_hashes.add(protocol_hash)
        rows.append(
            {
                "sweep": manifest_row["sweep"],
                "parameter": manifest_row["parameter"],
                "value": manifest_row["value"],
                "numeric_value": manifest_row["numeric_value"],
                "seed": int(manifest_row["seed"]),
                **task_values,
                "average": recomputed_average,
                "noise_rate": float(manifest_row["noise_rate"]),
                "checkpoint": manifest_row["checkpoint"],
                "config_hash": manifest_row["config_hash"],
                "scientific_hash": manifest_row["scientific_hash"],
                "evaluation_protocol_hash": protocol_hash,
                "result_path": str(result_path),
            }
        )
    if missing and not allow_missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} evaluation result(s). Use --allow_missing only for "
            f"an explicitly partial summary. First paths:\n{preview}"
        )
    if not rows:
        raise ValueError("No completed sensitivity evaluations were found.")
    if len(protocol_hashes) != 1:
        raise ValueError(f"Mixed evaluation protocols detected: {sorted(protocol_hashes)}")
    return rows, missing


def aggregate(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, float], List[Mapping[str, Any]]] = defaultdict(list)
    order: List[Tuple[str, str, str, float]] = []
    for row in rows:
        key = (row["sweep"], row["parameter"], row["value"], row["noise_rate"])
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)

    summary: List[Dict[str, Any]] = []
    for sweep, parameter, value, noise_rate in order:
        group = grouped[(sweep, parameter, value, noise_rate)]
        output: Dict[str, Any] = {
            "sweep": sweep,
            "parameter": parameter,
            "value": value,
            "numeric_value": group[0]["numeric_value"],
            "noise_rate": noise_rate,
            "n_seeds": len(group),
            "seeds": ",".join(str(row["seed"]) for row in group),
        }
        for metric in (*TASKS, "average"):
            values = [float(row[metric]) for row in group]
            output[f"{metric}_mean"] = statistics.fmean(values)
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(output)
    return summary


def sort_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    sweep_order = {"T": 0, "delta0": 1, "kappa": 2, "lambda": 3}
    value_order = {
        "T": {"4": 0, "2": 1, "1": 2, "0.5": 3},
        "delta0": {"0.5": 0, "1": 1, "2": 2, "4": 3},
        "kappa": {"3": 0, "2": 1, "1.5": 2, "1": 3},
        "lambda": {"0.5": 0, "1": 1, "2": 2},
    }
    return sorted(
        rows,
        key=lambda row: (
            sweep_order[str(row["sweep"])],
            float(row["noise_rate"]),
            value_order[str(row["sweep"])][str(row["value"])],
        ),
    )


def load_published_anchor_rows(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    clean = {task: _float(payload["clean"][task], f"clean anchor:{task}") for task in TASKS}
    noise = {task: _float(payload["noise20"][task], f"noise20 anchor:{task}") for task in TASKS}
    definitions = (
        ("T", "T", "2", 2.0, 0.0, clean),
        ("delta0", "delta0", "2", 2.0, 0.0, clean),
        ("kappa", "kappa", "2", 2.0, 0.0, clean),
        ("kappa", "kappa", "2", 2.0, 0.2, noise),
        ("lambda", "lambda", "1", 1.0, 0.0, clean),
    )
    rows: List[Dict[str, Any]] = []
    for sweep, parameter, value, numeric_value, noise_rate, tasks in definitions:
        rows.append(
            {
                "sweep": sweep,
                "parameter": parameter,
                "value": value,
                "numeric_value": numeric_value,
                "seed": "published",
                **tasks,
                "average": sum(tasks.values()) / len(TASKS),
                "noise_rate": noise_rate,
                "checkpoint": "published_anchor",
                "config_hash": "published_anchor",
                "scientific_hash": "published_anchor",
                "evaluation_protocol_hash": "published_anchor_from_spec",
                "result_path": str(path.resolve()),
            }
        )
    return rows


def _summary_fields() -> List[str]:
    fields = ["sweep", "parameter", "value", "numeric_value", "noise_rate", "n_seeds", "seeds"]
    for metric in (*TASKS, "average"):
        fields.extend((f"{metric}_mean", f"{metric}_std"))
    return fields


def _condition_label(noise_rate: float) -> str:
    return "Clean" if noise_rate == 0 else f"{noise_rate:.0%} noise"


def make_plots(summary: Sequence[Mapping[str, Any]], output_root: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plot generation requires matplotlib. Install sensitivity/requirements.txt.") from error

    for sweep in ("T", "kappa", "delta0", "lambda"):
        sweep_rows = [row for row in summary if row["sweep"] == sweep]
        if not sweep_rows:
            continue
        by_noise: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
        for row in sweep_rows:
            by_noise[float(row["noise_rate"])].append(row)
        fig, axis = plt.subplots(figsize=(5.4, 3.6))
        for noise_rate, condition in by_noise.items():
            x = list(range(len(condition)))
            axis.errorbar(
                x,
                [row["average_mean"] for row in condition],
                yerr=[row["average_std"] for row in condition],
                marker="o",
                capsize=3,
                label=_condition_label(noise_rate),
            )
            axis.set_xticks(x, [str(row["value"]) for row in condition])
        axis.set_xlabel({"T": "T", "kappa": r"$\kappa$", "delta0": r"$\Delta_0$", "lambda": r"$\lambda$"}[sweep])
        axis.set_ylabel("Six-task average")
        axis.grid(alpha=0.25)
        if len(by_noise) > 1:
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_root / f"sensitivity_{sweep}.pdf", bbox_inches="tight")
        plt.close(fig)


def _latex_value(value: str) -> str:
    return f"${value}$"


def _mean_std(row: Mapping[str, Any]) -> str:
    return f"{float(row['average_mean']):.2f} $\\pm$ {float(row['average_std']):.2f}"


def _latex_score(row: Mapping[str, Any], metric: str, *, bold: bool = False) -> str:
    value = f"{float(row[f'{metric}_mean']):.2f}"
    return f"\\textbf{{{value}}}" if bold else value


def make_latex(summary: Sequence[Mapping[str, Any]], output_root: Path):
    filenames = {
        "T": "table_temperature.tex",
        "delta0": "table_margin.tex",
        "kappa": "table_kappa.tex",
        "lambda": "table_lambda.tex",
    }
    defaults = {"T": "2", "delta0": "2", "kappa": "2", "lambda": "1"}
    combined: List[str] = []
    for sweep in ("T", "delta0", "kappa", "lambda"):
        sweep_rows = [row for row in summary if row["sweep"] == sweep]
        if not sweep_rows:
            continue
        title = {"T": "$T$", "delta0": "$\\Delta_0$", "kappa": "$\\kappa$", "lambda": "$\\lambda$"}[sweep]
        lines = [
            "% Generated by summarize_sensitivity.py. Do not edit values manually.",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llrrrrrrr}",
            r"\toprule",
            f"Condition & {title} & HellaSwag & ARC & MMLU & TruthfulQA & WinoGrande & GSM8K & Avg. \\",
            r"\midrule",
        ]
        previous_noise = None
        for row in sweep_rows:
            noise_rate = float(row["noise_rate"])
            if previous_noise is not None and noise_rate != previous_noise:
                lines.append(r"\midrule")
            previous_noise = noise_rate
            is_default = str(row["value"]) == defaults[sweep]
            if is_default:
                lines.append(r"\rowcolor{gray!20}")
            cells = [
                _condition_label(noise_rate),
                _latex_value(str(row["value"])),
                *[_latex_score(row, task) for task in TASKS],
                _latex_score(row, "average", bold=is_default),
            ]
            lines.append(" & ".join(cells) + r" \\")
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                f"\\caption{{ARC-BPO sensitivity to {title} on Llama-3-8B.}}",
                f"\\label{{tab:arc_bpo_sensitivity_{sweep}}}",
                r"\end{table*}",
            ]
        )
        rendered = "\n".join(lines) + "\n"
        (output_root / filenames[sweep]).write_text(rendered, encoding="utf-8")
        combined.append(rendered.rstrip())
    (output_root / "sensitivity_tables.tex").write_text(
        "\n\n".join(combined) + "\n", encoding="utf-8"
    )


def _extract_main_average(payload: Mapping[str, Any]) -> float:
    if "average" in payload:
        return _float(payload["average"], "main average")
    if isinstance(payload.get("metrics"), Mapping) and "average" in payload["metrics"]:
        return _float(payload["metrics"]["average"], "main average")
    raise KeyError("Main result JSON must contain 'average' or 'metrics.average'.")


def default_reproduction_check(
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    main_result: str | None,
    expected_variance_arg: float | None,
) -> Dict[str, Any]:
    if not main_result:
        return {
            "status": "not_checked",
            "reason": "Pass --main_result and --expected_variance before interpreting results.",
        }
    main_payload = read_json(Path(main_result))
    expected_variance = expected_variance_arg
    if expected_variance is None:
        value = main_payload.get("expected_variance")
        if value is None:
            raise ValueError(
                "Default reproduction requires --expected_variance or expected_variance in main JSON."
            )
        expected_variance = _float(value, "expected_variance")
    if expected_variance < 0:
        raise ValueError("expected_variance cannot be negative.")

    default = read_json(output_root / "default_config.json")
    defaults = {
        "T": float(default["loss"]["T"]),
        "kappa": float(default["loss"]["kappa"]),
        "delta0": float(default["loss"]["delta_star"]),
        "lambda": float(default["loss"]["sba_lambda"]),
    }
    candidates = []
    seen_seeds = set()
    for row in rows:
        if float(row["noise_rate"]) != 0 or not row["numeric_value"]:
            continue
        if not math.isclose(float(row["numeric_value"]), defaults[row["sweep"]], abs_tol=1e-12):
            continue
        if row["seed"] not in seen_seeds:
            candidates.append(row)
            seen_seeds.add(row["seed"])
    if not candidates:
        raise ValueError("No clean exact-default sensitivity point was evaluated.")

    main_average = _extract_main_average(main_payload)
    sensitivity_average = statistics.fmean(float(row["average"]) for row in candidates)
    difference = abs(main_average - sensitivity_average)
    status = "passed" if difference <= expected_variance else "failed"
    return {
        "status": status,
        "main_reported_average": main_average,
        "sensitivity_default_average": sensitivity_average,
        "absolute_difference": difference,
        "expected_variance": expected_variance,
        "sensitivity_seeds": sorted(seen_seeds),
        "main_result": str(Path(main_result).resolve()),
    }


def write_markdown(
    summary: Sequence[Mapping[str, Any]],
    missing: Sequence[str],
    reproduction: Mapping[str, Any],
    output_root: Path,
):
    lines = [
        "# ARC-BPO Sensitivity Summary",
        "",
        "This file reports computed facts only; it does not infer stability or robustness.",
        "",
        f"Evaluated settings: {len(summary)}. Missing run results: {len(missing)}.",
        "",
        "## Default-point reproduction",
        "",
        f"Status: **{reproduction['status']}**.",
    ]
    if reproduction["status"] == "not_checked":
        lines.append(str(reproduction["reason"]))
    else:
        lines.extend(
            [
                f"Main average: {reproduction['main_reported_average']:.4f}.",
                f"Sensitivity default average: {reproduction['sensitivity_default_average']:.4f}.",
                f"Absolute difference: {reproduction['absolute_difference']:.4f}; "
                f"allowed variance: {reproduction['expected_variance']:.4f}.",
            ]
        )
    for sweep in ("T", "kappa", "delta0", "lambda"):
        sweep_rows = [row for row in summary if row["sweep"] == sweep]
        if not sweep_rows:
            continue
        lines.extend(("", f"## {sweep}", ""))
        by_noise: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
        for row in sweep_rows:
            by_noise[float(row["noise_rate"])].append(row)
        for noise_rate, condition in by_noise.items():
            best = max(condition, key=lambda row: float(row["average_mean"]))
            scores = [float(row["average_mean"]) for row in condition]
            lines.append(
                f"{_condition_label(noise_rate)}: best observed value `{best['value']}` "
                f"with {best['average_mean']:.2f} ± {best['average_std']:.2f}; "
                f"range across settings {min(scores):.2f}–{max(scores):.2f}."
            )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_root = manifest_path.parent
    rows, missing = load_run_results(manifest_path, args.allow_missing)
    if args.published_anchors:
        rows.extend(load_published_anchor_rows(Path(args.published_anchors)))
    summary = sort_summary(aggregate(rows))
    reproduction = default_reproduction_check(
        rows,
        output_root,
        args.main_result,
        args.expected_variance,
    )

    write_csv(output_root / "sensitivity_all_runs.csv", rows, ALL_RUN_FIELDS)
    write_csv(output_root / "sensitivity_results.csv", rows, ALL_RUN_FIELDS)
    write_csv(output_root / "sensitivity_summary.csv", summary, _summary_fields())
    write_json(
        output_root / "sensitivity_summary.json",
        {
            "version": 1,
            "partial": bool(missing),
            "missing_results": list(missing),
            "evaluation_protocol_hash": rows[0]["evaluation_protocol_hash"],
            "default_reproduction": reproduction,
            "settings": summary,
        },
    )
    write_json(output_root / "sensitivity_results.json", rows)
    write_json(output_root / "default_reproduction_check.json", reproduction)
    make_plots(summary, output_root)
    make_latex(summary, output_root)
    write_markdown(summary, missing, reproduction, output_root)
    print(f"Wrote sensitivity summary artifacts to {output_root}")
    if reproduction["status"] == "failed":
        raise RuntimeError(
            "Default sensitivity result does not reproduce the main result within the "
            "declared variance. Stop interpretation and identify the protocol/run mismatch."
        )


if __name__ == "__main__":
    main()
