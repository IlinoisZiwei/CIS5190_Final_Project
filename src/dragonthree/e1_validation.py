"""Explicit-input continuation of one unfinished V3 nested fold.

Only the approved historical data/cache copies named in a local configuration
are read. All output goes to a new directory. No historical module is imported.
The caller's supervisor owns the cumulative CPU ledger and process hard limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from .v3_validation import (
    C_VALUES, FIXED_THRESHOLD, LOCKED_SETTINGS, POLICIES, REPRESENTATIONS,
    SCORE_SCALE, SEEDS, THRESHOLDS, ProtocolError, RunLog, TrainingData,
    assert_group_disjoint, build_representations, candidate_records, fit_svm,
    inner_selection, select_policies, summarize,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def recorded_path(value):
    p = Path(value)
    if p.is_absolute():
        try: key = p.relative_to(REPOSITORY).as_posix()
        except ValueError: raise PermissionError("Historical input must belong to this repository")
    else: key = p.as_posix()
    mapping = json.loads((REPOSITORY / "configs/path_migration.json").read_text())["files"]
    result = (REPOSITORY / mapping.get(key, {}).get("destination", key)).resolve()
    if not result.is_relative_to(REPOSITORY): raise PermissionError("Input escaped the repository")
    return result
EXPECTED_INPUT_NAMES = {
    "audited_training": {"audited_training.jsonl", "audited_training_rows.jsonl"},
    "folds": {"folds.json", "new_folds.json"},
    "plan": {"latest_plan.json", "preregistered_plan.json"},
    "locked_selection": {"latest_selection.json", "final_selection.json"},
    "fixed_grid": {"fixed_grid_complete.npz"},
    "nested_oof": {"nested_oof.npz"},
    "outer_choices": {"outer_choices.json"},
    "historical_training_scores": {"training_scores.json"},
    "historical_nested_results": {"nested_results.json", "latest_nested.json"},
}
REQUIRED_INPUTS = {
    "audited_training", "folds", "plan", "locked_selection",
    "fixed_grid", "nested_oof", "outer_choices",
}
EXPECTED_KINDS = {
    key: ("jsonl" if key == "audited_training" else
          "npz" if key in {"fixed_grid", "nested_oof"} else "json")
    for key in EXPECTED_INPUT_NAMES
}


class BudgetStop(RuntimeError):
    """The caller-approved stage budget is exhausted."""


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value):
    """Atomic writes only to this new run's output directory."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


class InputCatalog:
    """Validate exact repo-local input files and log all reads."""
    def __init__(self, entries: dict, output: Path):
        if not REQUIRED_INPUTS <= set(entries) or not set(entries) <= set(EXPECTED_INPUT_NAMES):
            raise ProtocolError("E1 configuration has missing or unknown input roles")
        self.entries, self.output = entries, output
        self.paths, self.reads, self.hashes = {}, [], {}
        for key, entry in entries.items():
            path = recorded_path(entry["path"]).resolve(strict=True)
            if REPOSITORY not in path.parents:
                raise ProtocolError("All E1 inputs must be explicit copies inside the repository")
            if path.name not in EXPECTED_INPUT_NAMES[key] or path.suffix != "." + EXPECTED_KINDS[key]:
                raise ProtocolError(f"Unexpected filename/type for input role {key}: {path.name}")
            if not path.is_file() or "holdout" in path.name.lower() or path.name == "external_metadata.csv":
                raise ProtocolError("Sealed, mixed, or non-file input is not allowed")
            digest = entry.get("sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                raise ProtocolError(f"An approved SHA-256 is required for {key}")
            if path == output or output in path.parents:
                raise ProtocolError("Inputs and the new output directory overlap")
            self.paths[key] = path
        if len(set(self.paths.values())) != len(self.paths):
            raise ProtocolError("Different input roles cannot alias one file")

    def install_guard(self):
        allowed, output = set(self.paths.values()), self.output
        def guard(event, args):
            if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            p = Path(os.fsdecode(args[0])).absolute()
            mode, flags = args[1], args[2]
            writing = bool((mode and any(c in mode for c in "wax")) or
                           (flags and flags & (os.O_WRONLY | os.O_RDWR)))
            if "holdout" in p.name.lower() or p.name == "external_metadata.csv":
                raise PermissionError("Sealed or mixed metadata access is forbidden")
            if writing:
                return
            if p.suffix.lower() in {".csv", ".jsonl", ".npy", ".npz", ".pt", ".pth"}:
                resolved = p.resolve()
                if resolved not in allowed and output not in resolved.parents:
                    raise PermissionError("Dataset/cache is not on the explicit E1 allowlist")
        sys.addaudithook(guard)

    def bytes(self, key: str) -> bytes:
        path = self.paths[key]
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != self.entries[key]["sha256"].lower():
            raise ProtocolError(f"Approved input digest mismatch for {key}")
        self.hashes[key] = digest
        self.reads.append({"role": key, "path": str(path.relative_to(REPOSITORY)), "bytes": len(content)})
        return content

    def json(self, key: str):
        if EXPECTED_KINDS[key] != "json":
            raise ProtocolError(f"{key} is not a JSON input")
        return json.loads(self.bytes(key))

    def jsonl(self, key: str):
        if EXPECTED_KINDS[key] != "jsonl":
            raise ProtocolError(f"{key} is not a JSONL input")
        return [json.loads(line) for line in self.bytes(key).decode("utf-8").splitlines() if line.strip()]

    def npz(self, key: str):
        if EXPECTED_KINDS[key] != "npz":
            raise ProtocolError(f"{key} is not an NPZ input")
        from io import BytesIO
        with np.load(BytesIO(self.bytes(key)), allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}


def verify_plan(plan, locked, catalog: InputCatalog):
    if tuple(plan["C_values"]) != C_VALUES or tuple(plan["thresholds"]) != THRESHOLDS:
        raise ProtocolError("Historical search grid changed")
    if plan["score_scale"] != SCORE_SCALE or plan["fixed_C"] != 4 or plan["fixed_threshold"] != FIXED_THRESHOLD:
        raise ProtocolError("Historical scale or fixed control changed")
    for key, expected in LOCKED_SETTINGS.items():
        actual = locked[key]
        if any(actual[k] != v for k, v in expected.items()):
            raise ProtocolError(f"Locked final settings changed: {key}")
    if locked.get("development_used_for_selection") is not False:
        raise ProtocolError("Historical locked selection lacks the required development boundary")
    # The originals' recorded digests refer to approved training/fold files only.
    # Do not follow or hash any other paths embedded in historical documents.
    for key, basename in (("audited_training", "audited_training_rows.jsonl"), ("folds", "folds.json")):
        matches = [digest for path, digest in plan["inputs"].items() if Path(path).name == basename]
        if len(matches) != 1 or matches[0] != catalog.hashes[key]:
            raise ProtocolError(
                f"{key} is not a byte-identical copy of the historical approved input; "
                "do not combine old cached folds with changed data"
            )


def verify_folds(data: TrainingData, folds):
    if [r["seed"] for r in folds] != list(SEEDS):
        raise ProtocolError("Unexpected outer partition seeds/order")
    n = len(data.y)
    for partition in folds:
        if len(partition["parts"]) != 3:
            raise ProtocolError("Each saved partition must have exactly three folds")
        validation_union = []
        for part in partition["parts"]:
            tr, va = np.asarray(part["train"], dtype=int), np.asarray(part["validation"], dtype=int)
            if not len(tr) or not len(va) or np.any(tr < 0) or np.any(va < 0) or np.any(tr >= n) or np.any(va >= n):
                raise ProtocolError("Invalid saved fold indices")
            if len(np.unique(tr)) != len(tr) or len(np.unique(va)) != len(va):
                raise ProtocolError("A saved fold contains duplicate row indices")
            if not np.array_equal(np.sort(np.r_[tr, va]), np.arange(n)):
                raise ProtocolError("A saved train/validation fold does not partition the approved ledger")
            assert_group_disjoint(data, tr, va)
            validation_union.extend(va.tolist())
        if sorted(validation_union) != list(range(n)):
            raise ProtocolError("Saved outer validation folds do not cover every row exactly once")


def verify_caches(data, folds, fixed_cache, nested_cache, choices):
    n = len(data.y)
    raw = fixed_cache["raw_scores"]
    old_raw, scores = nested_cache["raw_scores"], nested_cache["policy_scores"]
    if raw.shape != (3, 4, 4, n) or old_raw.shape != raw.shape or scores.shape != (3, 17, n):
        raise ProtocolError("Historical cache dimensions do not match this protocol")
    if raw.dtype.kind != "f" or scores.dtype.kind != "f" or not np.isfinite(raw).all():
        raise ProtocolError("Fixed grid must contain complete finite floating-point scores")
    for cache in (fixed_cache, nested_cache):
        for key, expected in (("y", data.y), ("origin", data.origin), ("groups", data.groups)):
            if not np.array_equal(cache[key], expected):
                raise ProtocolError(f"Cached {key} differs from the approved record order")
    if nested_cache["policies"].tolist() != list(POLICIES):
        raise ProtocolError("Cached policy ordering differs")
    by_fold = {(int(row["seed"]), int(row["fold"])): row for row in choices}
    expected_done = {(seed, fold) for seed in SEEDS for fold in range(3)} - {(5192, 2)}
    if len(choices) != 8 or set(by_fold) != expected_done:
        raise ProtocolError("Expected exactly the eight preserved outer policy choices")
    for si, partition in enumerate(folds):
        for fi, part in enumerate(partition["parts"]):
            va = np.asarray(part["validation"], dtype=int)
            prior_block, policy_block = old_raw[si][:, :, va], scores[si][:, va]
            if (partition["seed"], fi) == (5192, 2):
                if not np.isnan(prior_block).all() or not np.isnan(policy_block).all():
                    raise ProtocolError("Missing fold is not entirely missing in the old nested cache")
                continue
            if not np.isfinite(prior_block).all() or not np.isfinite(policy_block).all():
                raise ProtocolError("A supposedly complete cached outer fold is partial")
            np.testing.assert_allclose(prior_block, raw[si][:, :, va], atol=0, rtol=0)
            selected = by_fold[(partition["seed"], fi)]["choices"]
            if set(selected) != set(POLICIES):
                raise ProtocolError("A cached fold has missing policy choices")
            for pi, policy in enumerate(POLICIES):
                spec = selected[policy]
                reference = raw[si, REPRESENTATIONS.index(spec["rep"]), C_VALUES.index(spec["C"])][va] - spec["threshold"]
                np.testing.assert_allclose(policy_block[pi], reference, atol=1e-12, rtol=0)
    return raw.copy(), scores.copy()


def compatibility_anchor(data, folds, raw, config, log: RunLog):
    """One course-only C=4 fit checks cached/new numeric compatibility."""
    options = config.get("compatibility_anchor", {})
    if options.get("enabled", True) is not True:
        raise ProtocolError("This cache continuation requires one compatibility anchor")
    seed, fi = int(options.get("partition_seed", 5190)), int(options.get("outer_fold", 0))
    if (seed, fi) != (5190, 0):
        raise ProtocolError("The predeclared compatibility anchor is seed5190_outer0")
    tolerance = float(options.get("atol", 1e-5))
    if tolerance <= 0 or tolerance > 1e-5:
        raise ProtocolError("Compatibility tolerance cannot exceed the approved 1e-5")
    part = folds[0]["parts"][0]
    tr, va = np.asarray(part["train"], dtype=int), np.asarray(part["validation"], dtype=int)
    _, matrices = build_representations(data, tr, "compatibility_anchor_seed5190_outer0", log)
    model = fit_svm(data, matrices["A"], tr, "A", 4.0, "compatibility_anchor_seed5190_outer0", log)
    current = model.decision_function(matrices["A"][va]) / SCORE_SCALE
    reference = raw[0, 0, C_VALUES.index(4.0)][va]
    max_error = float(np.max(np.abs(current - reference)))
    fixed_equal = bool(np.array_equal(current > FIXED_THRESHOLD, reference > FIXED_THRESHOLD))
    grid_equal = bool(np.array_equal(
        current[:, None] > np.asarray(THRESHOLDS),
        reference[:, None] > np.asarray(THRESHOLDS),
    ))
    record = {
        "partition_seed": seed, "outer_fold": fi, "representation": "A", "C": 4.0,
        "max_absolute_score_difference": max_error, "absolute_tolerance": tolerance,
        "fixed_threshold_prediction_equal": fixed_equal,
        "all_grid_threshold_predictions_equal": grid_equal,
        "passed": max_error <= tolerance and fixed_equal and grid_equal,
        "limitation": "One fixed-anchor comparison supports cache compatibility; it is not a complete environment equivalence proof.",
    }
    return record


def _verify_prior_summary(prior, reproduced):
    """Require agreement on the historical primary results before fresh fitting."""
    for panel in ("course", "new", "old"):
        expected = prior["primary_paired_deltas_vs_tuned_A"][panel]
        current = reproduced["primary_paired_deltas_vs_tuned_A"][panel]
        np.testing.assert_allclose(current["delta"], expected["delta"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(current["ci95"], expected["ci95"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        reproduced["primary_new_gain_by_seed"], prior["primary_new_gain_by_seed"], atol=1e-12, rtol=0,
    )


def make_summary(results: dict, catalog: InputCatalog, fit_count: int):
    """Keep complete OOF results separate from incomplete historical fit diagnostics."""
    by = {row["policy"]: row for row in results["policies"]}
    fixed_features = [dict(row) for row in results["regularization_curves"] if row["C"] == 4]
    diagnostics = (catalog.json("historical_training_scores")
                   if "historical_training_scores" in catalog.paths else [])
    for row in fixed_features:
        logs = [r for r in diagnostics
                if r["representation"] == row["representation"] and r["C"] == 4]
        if len(logs) > 9 or len({r["tag"] for r in logs}) != len(logs):
            raise ProtocolError("Historical fitting diagnostics contain duplicate or extra outer fits")
        row["course_train_accuracy"] = (
            float(np.mean([r["course_train"]["accuracy"] for r in logs])) if logs else None
        )
        row["new_train_accuracy"] = (
            float(np.mean([r["new_fit_rows"]["accuracy"] for r in logs]))
            if logs and row["representation"] != "A" else None
        )
        row["in_sample_diagnostics_available_folds"] = len(logs)
        row["in_sample_diagnostics_expected_folds"] = 9
        row["in_sample_diagnostics_complete"] = len(logs) == 9
        row["in_sample_diagnostics_tags"] = [r["tag"] for r in logs]
        row["in_sample_diagnostics_source"] = "preserved historical fixed-grid fitting logs only"
        row["in_sample_diagnostics_note"] = (
            f"In-sample means use {len(logs)} available historical fits, whereas OOF metrics "
            "use all nine fixed-grid folds. Missing fit diagnostics are not inferred from "
            "validation scores and no model is refitted to manufacture them."
        )
    return {
        "status": "complete", "fixed_grid_complete_folds": 9, "nested_completed_folds": 9,
        "nested_aggregated_complete_partitions": 3, "final_model_exported": False,
        "development_evaluated": False, "final_selection": LOCKED_SETTINGS,
        "fixed_feature_results": fixed_features, "nested_primary": by["augmented_joint"],
        "nested_tuned_course": by["A_joint"], "internal_gate_passed": results["internal_gate_passed"],
        "recommend_for_future_v4": False,
        "recommendation_note": (
            "E1 does not itself satisfy E4 prerequisites. E4 is conditionally authorized "
            "under the current approved scope and requires the E2 development gates "
            "and the applicable CPU cap before it can start."
        ),
        "final_test_accessed": False, "new_fits": fit_count,
    }


def write_completed_reports(output: Path, results: dict, catalog: InputCatalog, fit_count: int):
    summary = make_summary(results, catalog, fit_count)
    write_json(output / "nested_results.json", results)
    write_json(output / "summary.json", summary)
    write_json(output / "gates.json", {
        "internal_gate_passed": results["internal_gate_passed"],
        "checks": results["gate_checks"], "development_gate": "not evaluated by E1",
        "E4_authorized_by_this_stage": False,
        "E4_conditional_authorization_from_scope": True,
        "E4_prerequisite_note": (
            "Consult the Approved experimental scope section in Notebook 03 and E2/results.json. A passed E1 "
            "internal gate alone cannot start E4; the E2 development gates and "
            "current approved CPU limits must also be satisfied."
        ),
    })
    return summary


def finalize_completed_run(config: dict, output: Path, cpu_budget: float = 30):
    """Repair reporting from a saved completed run without any estimator fitting.

    This route deliberately contains no calls to feature construction, SVM
    fitting, compatibility_anchor(), or inner_selection(). The old failure and
    existing fit logs are preserved; the supervisor must account for this new
    finalization process in the same E1 budget.
    """
    output = output.resolve(strict=True)
    if REPOSITORY not in output.parents or not output.is_dir():
        raise ProtocolError("Finalization requires an existing repository-local E1 directory")
    if config.get("stage_id") != "E1" or config.get("authorization_scope") != "base1800":
        raise ProtocolError("Explicit E1/base1800 authorization is required")
    if not 0 < cpu_budget <= 60:
        raise ProtocolError("Reporting-only finalization must have a CPU cap of at most 60 seconds")
    required = ("status.json", "configuration.json", "compatibility_anchor.json",
                "fit_log.json", "outer_choices.json", "nested_oof_complete.npz",
                "locked_selection.json")
    if any(not (output / name).is_file() for name in required):
        raise ProtocolError("A completed score/choice checkpoint is required; do not rerun training")
    previous = json.loads((output / "status.json").read_text())
    if previous.get("status") == "complete":
        raise ProtocolError("E1 is already complete; reporting-only finalization is unnecessary")
    if json.loads((output / "configuration.json").read_text()) != config:
        raise ProtocolError("Finalization configuration differs from the original execution")
    fit_log = json.loads((output / "fit_log.json").read_text())
    if len(fit_log) != 49 or any(row["status"] != "complete" for row in fit_log):
        raise ProtocolError("Expected the successful anchor and all 48 completed inner SVM fits")
    anchor = json.loads((output / "compatibility_anchor.json").read_text())
    if not anchor.get("passed"):
        raise ProtocolError("The saved compatibility anchor did not pass")
    history_path = output / "status_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(previous)
    write_json(history_path, history)
    snapshot = output / "failure_snapshot.json"
    if not snapshot.exists():
        write_json(snapshot, {
            "status": previous,
            "reason_for_reporting_repair": (
                "The original runner required nine historical in-sample diagnostics. "
                "Only eight were preserved; the nine fixed-grid OOF folds and all nine "
                "nested policy folds had already been completed successfully."
            ),
            "training_will_not_be_repeated": True,
        })
    wall_start = time.monotonic()
    status = {k: v for k, v in previous.items() if k not in {"error", "reason"}}
    status.update(status="finalizing", finalization_only=True, new_fits_during_finalization=0)
    write_json(output / "status.json", status)
    catalog = None
    def check():
        if time.process_time() >= cpu_budget - min(2., cpu_budget / 4):
            raise BudgetStop("Reporting-only finalization reached its CPU limit")
    try:
        catalog = InputCatalog(config["inputs"], output)
        catalog.install_guard()
        data = TrainingData.from_rows(catalog.jsonl("audited_training"))
        folds, plan, locked = catalog.json("folds"), catalog.json("plan"), catalog.json("locked_selection")
        verify_plan(plan, locked, catalog)
        verify_folds(data, folds)
        cached = catalog.npz("fixed_grid")
        with np.load(output / "nested_oof_complete.npz", allow_pickle=False) as saved:
            complete = {key: saved[key].copy() for key in saved.files}
        raw, policy_scores = complete["raw_scores"], complete["policy_scores"]
        if raw.shape != (3, 4, 4, len(data.y)) or policy_scores.shape != (3, 17, len(data.y)):
            raise ProtocolError("Saved completed arrays have unexpected dimensions")
        if not np.isfinite(raw).all() or not np.isfinite(policy_scores).all():
            raise ProtocolError("Saved arrays are incomplete; reporting-only repair cannot train missing folds")
        for key, expected in (("y", data.y), ("origin", data.origin), ("groups", data.groups)):
            if not np.array_equal(complete[key], expected) or not np.array_equal(cached[key], expected):
                raise ProtocolError(f"Completed cache {key} does not match approved input order")
        if complete["record_ids"].tolist() != data.ids or complete["policies"].tolist() != list(POLICIES):
            raise ProtocolError("Completed record IDs or policy names changed")
        np.testing.assert_allclose(raw, cached["raw_scores"], atol=0, rtol=0)
        choices = json.loads((output / "outer_choices.json").read_text())
        mapping = {(int(r["seed"]), int(r["fold"])): r["choices"] for r in choices}
        expected_folds = {(seed, fi) for seed in SEEDS for fi in range(3)}
        if len(choices) != 9 or set(mapping) != expected_folds:
            raise ProtocolError("Nine distinct completed policy choices are required")
        for si, partition in enumerate(folds):
            for fi, part in enumerate(partition["parts"]):
                va = np.asarray(part["validation"], dtype=int)
                selected = mapping[(partition["seed"], fi)]
                if set(selected) != set(POLICIES):
                    raise ProtocolError("Incomplete saved policy selection")
                for pi, policy in enumerate(POLICIES):
                    spec = selected[policy]
                    expected = (raw[si, REPRESENTATIONS.index(spec["rep"]),
                                    C_VALUES.index(spec["C"]), va] - spec["threshold"])
                    np.testing.assert_allclose(policy_scores[si, pi, va], expected, atol=1e-12, rtol=0)
        saved_locked = json.loads((output / "locked_selection.json").read_text())
        for name, spec in LOCKED_SETTINGS.items():
            if any(saved_locked[name][key] != value for key, value in spec.items()):
                raise ProtocolError("Locked final settings changed during the original run")
        check()
        results = summarize(data, raw, policy_scores, choices, list(SEEDS), 9)
        check()
        summary = write_completed_reports(output, results, catalog, len(fit_log))
        write_json(output / "finalization_verification.json", {
            "saved_complete_npz_verified": True, "all_nine_policy_choices_verified": True,
            "saved_successful_anchor_reused": True, "new_estimator_fits": 0,
            "historical_files_changed": False, "final_settings_changed": False,
            "in_sample_diagnostics_not_imputed": True, "final_test_accessed": False,
            "approved_input_reads": catalog.reads,
        })
        status.update(status="complete", nested_completed_folds=9,
                      internal_gate_passed=results["internal_gate_passed"])
        print("E1_FINALIZATION_COMPLETE", json.dumps({
            "status": "complete", "nested_completed_folds": 9,
            "new_fits_during_finalization": 0,
            "internal_gate_passed": results["internal_gate_passed"],
        }), flush=True)
        return summary
    except BaseException as exc:
        status.update(status="partial" if isinstance(exc, BudgetStop) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["finalization_process_cpu_seconds"] = time.process_time()
        status["finalization_wall_seconds"] = time.monotonic() - wall_start
        status["cpu_accounting_note"] = "Original process diagnostics retained; cumulative CPU is owned by the supervisor ledger."
        write_json(output / "status.json", status)


def run(config: dict, output: Path):
    output = output.resolve()
    if REPOSITORY not in output.parents:
        raise ProtocolError("Output must be a new directory inside the repository")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ProtocolError("Output is not empty; historical and existing results will not be overwritten")
    if config.get("stage_id") != "E1" or config.get("authorization_scope") != "base1800":
        raise ProtocolError("Explicit stage_id=E1 and authorization_scope=base1800 are required")
    cpu_budget = float(config.get("cpu_budget_seconds", 600))
    reserve = float(config.get("stop_reserve_seconds", 8))
    if not 0 < cpu_budget <= 600 or not 2 <= reserve < cpu_budget:
        raise ProtocolError("E1 must fit within its approved 600 CPU-second cap")
    max_fits = int(config.get("max_fits", 65))
    if not 49 <= max_fits <= 65:
        raise ProtocolError("E1 fit limit must cover its 49 expected fits and cannot exceed 65")
    output.mkdir(parents=True, exist_ok=True)
    wall_start = time.monotonic()
    status = {
        "stage_id": "E1", "status": "running", "parent_experiment": "v3_feature_regularization_2026-10-29",
        "cpu_budget_seconds": cpu_budget, "new_fits": 0, "final_selection_changed": False,
        "development_evaluated": False, "final_test_accessed": False, "v4_trained": False,
    }
    write_json(output / "status.json", status)
    write_json(output / "configuration.json", config)
    write_json(output / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "thread_limit": 1,
    })
    catalog = None
    log = RunLog(max_fits=max_fits)
    def check_budget():
        if time.process_time() >= cpu_budget - reserve:
            raise BudgetStop("E1 CPU limit reached; preserving completed work without automatic continuation")
    def persist():
        write_json(output / "fit_log.json", log.fits)
        write_json(output / "feature_log.json", log.features)
        write_json(output / "inner_splits.json", log.splits)
        if catalog is not None:
            write_json(output / "input_reads.json", catalog.reads)
    log.check_budget, log.persist = check_budget, persist
    try:
        catalog = InputCatalog(config["inputs"], output)
        catalog.install_guard()
        data = TrainingData.from_rows(catalog.jsonl("audited_training"))
        folds, plan, locked = catalog.json("folds"), catalog.json("plan"), catalog.json("locked_selection")
        verify_plan(plan, locked, catalog)
        verify_folds(data, folds)
        choices = catalog.json("outer_choices")
        raw, policy_scores = verify_caches(
            data, folds, catalog.npz("fixed_grid"), catalog.npz("nested_oof"), choices,
        )
        # A reread of historical selection outcomes must reproduce the already
        # locked settings; this is verification, never a fresh choice of settings.
        reconstructed = select_policies(candidate_records(data, raw, np.arange(len(data.y))))
        for name, policy in (("tuned_course", "A_joint"), ("augmented", "augmented_joint")):
            for key, expected in LOCKED_SETTINGS[name].items():
                if reconstructed[policy][key] != expected:
                    raise ProtocolError("Cached fixed-grid selection no longer matches the locked settings")
        verification = {
            "approved_inputs_byte_identical": True, "outer_groups_disjoint": True,
            "cache_record_order_verified": True, "cached_eight_policy_folds_verified": True,
            "fixed_grid_complete_folds": 9, "missing_nested_fold": "seed5192_outer2",
            "locked_selection_reproduced_without_reselection": True,
            "input_sha256": catalog.hashes.copy(),
        }
        if "historical_nested_results" in catalog.paths:
            prior = catalog.json("historical_nested_results")
            reproduced = summarize(data, raw, policy_scores[:2], choices[:6], list(SEEDS[:2]), 6)
            _verify_prior_summary(prior, reproduced)
            verification["historical_two_partition_primary_summary_reproduced"] = True
        write_json(output / "input_verification.json", verification)
        persist()
        check_budget()

        anchor = compatibility_anchor(data, folds, raw, config, log)
        write_json(output / "compatibility_anchor.json", anchor)
        if not anchor["passed"]:
            raise ProtocolError("Compatibility anchor failed; no missing-fold fits were started")
        print("E1_COMPATIBILITY_ANCHOR_PASSED", json.dumps(anchor), flush=True)

        part = folds[2]["parts"][2]
        tr, va = np.asarray(part["train"], dtype=int), np.asarray(part["validation"], dtype=int)
        def checkpoint_inner(fi, inner_raw):
            np.savez_compressed(output / "missing_fold_inner_partial.npz", raw_scores=inner_raw,
                                completed_inner_folds=np.arange(fi + 1), y=data.y,
                                origin=data.origin, groups=data.groups)
            print(f"E1_INNER_FOLD_COMPLETE {fi + 1}/3 cpu={time.process_time():.3f}", flush=True)
            check_budget()
        selected, inner_raw = inner_selection(
            data, tr, 51922, "seed5192_outer2", log, checkpoint_inner,
        )
        for pi, policy in enumerate(POLICIES):
            spec = selected[policy]
            policy_scores[2, pi, va] = (
                raw[2, REPRESENTATIONS.index(spec["rep"]), C_VALUES.index(spec["C"]), va]
                - spec["threshold"]
            )
        choices = choices + [{"seed": 5192, "fold": 2, "choices": selected}]
        if not np.isfinite(policy_scores).all():
            raise ProtocolError("Completed policy array still contains missing scores")
        np.savez_compressed(
            output / "nested_oof_complete.npz", raw_scores=raw, policy_scores=policy_scores,
            y=data.y, origin=data.origin, groups=data.groups, policies=np.asarray(POLICIES),
            record_ids=np.asarray(data.ids),
        )
        write_json(output / "outer_choices.json", choices)
        write_json(output / "locked_selection.json", {
            **locked, "continuation_internal_gate_not_yet_applied_to_original_file": True,
            "settings_changed": False,
        })
        check_budget()
        results = summarize(data, raw, policy_scores, choices, list(SEEDS), 9)
        summary = write_completed_reports(output, results, catalog, len(log.fits))
        status.update(status="complete", nested_completed_folds=9,
                      internal_gate_passed=results["internal_gate_passed"])
        print("E1_COMPLETE", json.dumps(summary, default=_json_default), flush=True)
        return summary
    except BudgetStop as exc:
        status.update(status="partial", reason=str(exc))
        raise
    except BaseException as exc:
        status.update(status="blocked" if isinstance(exc, ProtocolError) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status.update(
            new_fits=len(log.fits), completed_new_fits=sum(r["status"] == "complete" for r in log.fits),
            process_cpu_seconds=time.process_time(), wall_seconds=time.monotonic() - wall_start,
        )
        persist()
        write_json(output / "status.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true",
                        help="Repair reports from a completed new E1 score checkpoint; never refit")
    parser.add_argument("--finalize-cpu-budget", type=float, default=30,
                        help="Reporting-only process CPU allowance; counted by the E1 supervisor")
    args = parser.parse_args(argv)
    config_path = args.config.resolve(strict=True)
    if REPOSITORY not in config_path.parents or config_path.suffix != ".json":
        parser.error("Configuration must be an explicit repository-local JSON file")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with threadpool_limits(limits=1):
        if args.finalize_only:
            finalize_completed_run(config, args.output, args.finalize_cpu_budget)
        else:
            run(config, args.output)


if __name__ == "__main__":
    main()
