"""E2: fit exactly the two previously locked candidates and confirm on approved dev.

Run only through the project CPU-budget supervisor. This module is passive on
import, has no default data path, and refuses a nonempty output directory. No
legacy experiment module is imported. Development results cannot change any
hyperparameter or reverse a failed E1 internal gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import platform
import re
import time
import warnings
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.svm import LinearSVC
import torch

SCALE = 0.776760627876633
SPECS = {
    "tuned_course": {"rep": "A", "C": 0.25, "threshold": -0.15},
    "augmented": {"rep": "C", "C": 1.0, "threshold": -0.20},
}
APPROVED = {
    "audited_training": ("data/approved/audited_training.jsonl", 8096),
    "course_dev": ("data/approved/course_dev.csv", 951),
    "legacy_dev": ("data/approved/legacy_dev.csv", 306),
    "paired_dev": ("data/approved/paired_dev.csv", 200),
}
DEV_KEYS = {"course_dev": "course_dev", "external_dev": "legacy_dev", "paired_dev": "paired_dev"}
LABELS = ("FoxNews", "NBC")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def url_to_text(url):
    path = urlparse(str(url).strip()).path
    segments = [part for part in path.split("/") if part]
    tail = unquote(segments[-1]) if segments else ""
    tail = re.sub(r"\.(html?|shtml|php)$", "", tail, flags=re.IGNORECASE)
    return " ".join(word for word in re.split(r"[-_]+", tail)
                    if word and not re.search(r"\d", word))


def label_from_url(url):
    host = (urlparse(str(url).strip()).hostname or "").lower()
    for domain, label in (("foxnews.com", "FoxNews"), ("nbcnews.com", "NBC")):
        if host == domain or host.endswith("." + domain):
            return label
    return ""


def text_group(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


class ApprovedInputs:
    """Only four literal, role-checked paths can be read through this loader."""

    def __init__(self, package):
        self.root = Path(package).resolve(strict=True)
        repository = Path(__file__).resolve().parents[2]
        if self.root != repository:
            raise PermissionError("E2 only accepts the repository root with its explicit approved-input manifest.")
        self.manifest = json.loads((self.root / "data/manifests/approved_inputs.json").read_text())
        frozen_path = repository / "configs" / "data_manifest.json"
        if frozen_path.resolve(strict=True) != frozen_path:
            raise PermissionError("The frozen E0 data manifest cannot be redirected.")
        self.frozen = json.loads(frozen_path.read_text())
        self.reads = []

    def load(self, key):
        relative, expected = APPROVED[key]
        record = self.manifest["data"][key]
        if record["path"] != relative or record["rows"] != expected:
            raise ValueError("Approved manifest differs from the frozen E2 input contract: " + key)
        frozen = self.frozen[key]
        if frozen["path"] != relative or frozen["rows"] != expected:
            raise ValueError("Frozen E0 manifest differs from the E2 input contract: " + key)
        path = (self.root / relative).resolve(strict=True)
        if path != self.root / relative:
            raise PermissionError("Symlinked or redirected data are not approved.")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != frozen["sha256"]:
            raise ValueError("Approved input bytes differ from the frozen E0 digest: " + key)
        self.reads.append({"key": key, "path": relative, "rows": expected,
                           "sha256": digest})
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        else:
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        if len(rows) != expected:
            raise ValueError(f"Unexpected row count for {key}: {len(rows)}")
        return rows


def make_vectorizer(name):
    params = dict(lowercase=True, strip_accents="unicode", dtype=np.float64,
                  sublinear_tf=True)
    if name == "word12":
        params.update(analyzer="word", ngram_range=(1, 2), min_df=1,
                      token_pattern=r"(?u)\b\w+\b")
    elif name == "char25":
        params.update(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=120000)
    else:
        raise ValueError(name)
    return TfidfVectorizer(**params)


def metric_values(truth, scores):
    predictions = scores > 0
    p, r, f, n = precision_recall_fscore_support(
        truth, predictions, labels=[0, 1], zero_division=0
    )
    return {
        "n": len(truth), "correct": int((truth == predictions).sum()),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]),
                               "f1": float(f[i]), "support": int(n[i])}
                      for i, label in enumerate(LABELS)},
    }


PREPROCESS_SOURCE = '''"""Course-compatible URL-slug preprocessing; preserves every input row."""
import csv
import re
from urllib.parse import unquote, urlparse

LABEL_TO_ID = {"FoxNews": 0, "NBC": 1}

def url_to_text(url):
    parts = [part for part in urlparse(str(url).strip()).path.split("/") if part]
    tail = unquote(parts[-1]) if parts else ""
    tail = re.sub(r"\\.(html?|shtml|php)$", "", tail, flags=re.IGNORECASE)
    return " ".join(word for word in re.split(r"[-_]+", tail)
                    if word and not re.search(r"\\d", word))

def label_from_url(url):
    host = (urlparse(str(url).strip()).hostname or "").lower()
    for domain, label in (("foxnews.com", "FoxNews"), ("nbcnews.com", "NBC")):
        if host == domain or host.endswith("." + domain):
            return label
    return ""

def prepare_data(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "url" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain a url column.")
        urls = [(row.get("url") or "").strip() for row in reader]
    return [url_to_text(url) for url in urls], [label_from_url(url) for url in urls]
'''


def export_model(vectors, classifier, threshold, folder):
    """Export tensor-only state and load the newly generated implementation."""
    folder.mkdir(parents=True, exist_ok=False)
    state, layout, offset = {}, [], 0
    for index, (name, vectorizer) in enumerate(vectors.items()):
        terms = vectorizer.get_feature_names_out().tolist()
        metadata = {"terms": terms, "analyzer": vectorizer.analyzer,
                    "ngram_range": list(vectorizer.ngram_range), "norm": vectorizer.norm,
                    "binary": vectorizer.binary, "sublinear_tf": vectorizer.sublinear_tf}
        raw = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        size = len(terms)
        layout.append({"metadata_bytes": len(raw), "features": size})
        state[f"metadata_{index}"] = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
        state[f"idf_{index}"] = torch.from_numpy(vectorizer.idf_.copy())
        state[f"weight_{index}"] = torch.from_numpy(classifier.coef_[0, offset:offset + size].copy() / SCALE)
        offset += size
    if offset != classifier.coef_.shape[1]:
        raise ValueError("Export coefficient dimensions do not match vocabulary.")
    state["bias"] = torch.tensor([float(classifier.intercept_[0]) / SCALE - threshold], dtype=torch.float64)
    template = Path(__file__).with_name("lexical_model.py").read_text()
    marker = "LAYOUT = []  # EXPORT_LAYOUT"
    if template.count(marker) != 1:
        raise ValueError("Invalid export layout marker.")
    (folder / "model.py").write_text(template.replace(marker, "LAYOUT = " + repr(layout)))
    (folder / "preprocess.py").write_text(PREPROCESS_SOURCE)
    torch.save(state, folder / "model.pt")
    spec = importlib.util.spec_from_file_location("e2_generated_" + folder.name, folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.get_model()
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model, layout


def write_predictions(path, rows):
    fields = ["model", "dataset", "record_id", "group_id", "dataset_role", "url", "text",
              "actual", "raw_margin", "normalized_margin", "threshold", "score",
              "prediction", "correct", "words"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(package, output):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("E2 output must be new or empty; no implicit retry or overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    inputs = ApprovedInputs(package)
    results, fit_log, predictions = {}, [], []
    dump(output / "status.json", {"status": "running", "stage": "E2", "model_fits_planned": 2})
    dump(output / "locked_specifications.json", {
        "score_scale": SCALE, "models": SPECS, "threshold_rule": "normalized_margin > threshold",
        "class_order": list(LABELS), "development_used_for_selection": False,
        "new_loss_mass": 200.0, "course_loss_mass": 3730.0,
        "feature_fit_rows": {"A": "course only", "C": "course vocabulary; course+new IDF"},
        "legacy_and_paired_training_pools_used_for_fitting": False,
    })
    try:
        rows = inputs.load("audited_training")
        origins = np.array([row["origin"] for row in rows])
        counts = {origin: int((origins == origin).sum()) for origin in np.unique(origins)}
        if counts != {"course": 3730, "legacy": 1566, "paired": 800, "new": 2000}:
            raise ValueError("Training-origin counts differ from the locked protocol.")
        for row in rows:
            if row["text"] != url_to_text(row["url"]) or row["label"] != label_from_url(row["url"]):
                raise ValueError("Audited training text/label differs from the frozen URL contract: " + row["id"])
        course_rows = [row for row in rows if row["origin"] == "course"]
        new_rows = [row for row in rows if row["origin"] == "new"]
        if {label: sum(row["label"] == label for row in new_rows) for label in LABELS} != {"FoxNews": 1000, "NBC": 1000}:
            raise ValueError("New training pool must contain 1000 records from each publisher.")
        course_text = [row["text"] for row in course_rows]
        joint_rows = course_rows + new_rows
        joint_text = [row["text"] for row in joint_rows]
        known_urls = {row["url"] for row in rows}
        known_text = {text_group(row["text"]) for row in rows}
        dev = {}
        for name, key in DEV_KEYS.items():
            panel = inputs.load(key)
            texts = [url_to_text(row["url"]) for row in panel]
            labels = [label_from_url(row["url"]) for row in panel]
            if any(label not in LABELS for label in labels):
                raise ValueError("Unknown publisher in approved development panel: " + name)
            if known_urls & {row["url"] for row in panel} or known_text & set(map(text_group, texts)):
                raise ValueError("Approved training/development exact overlap: " + name)
            dev[name] = (panel, texts, np.array([label == "NBC" for label in labels], dtype=int))
        dump(output / "input_access_log.json", inputs.reads)
        a = {name: make_vectorizer(name).fit(course_text) for name in ("word12", "char25")}
        c = {name: clone(vectorizer).set_params(vocabulary=dict(vectorizer.vocabulary_)).fit(joint_text)
             for name, vectorizer in a.items()}
        for name in a:
            if a[name].vocabulary_ != c[name].vocabulary_:
                raise ValueError("C must retain exactly the course vocabulary.")
        vectors_by_name = {"tuned_course": a, "augmented": c}
        for name, spec in SPECS.items():
            vectors = vectors_by_name[name]
            fitting_rows = course_rows if spec["rep"] == "A" else joint_rows
            texts = [row["text"] for row in fitting_rows]
            truth = np.array([row["label"] == "NBC" for row in fitting_rows], dtype=int)
            X = sparse.hstack([vectorizer.transform(texts) for vectorizer in vectors.values()], format="csr")
            weights = np.ones(len(fitting_rows), dtype=np.float64)
            if spec["rep"] == "C":
                weights[len(course_rows):] = 0.1
            classifier = LinearSVC(C=spec["C"], dual=True, max_iter=20000, random_state=42)
            fit_start = time.process_time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                classifier.fit(X, truth, sample_weight=weights)
            converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
            record = {"model": name, **spec, "course_rows": len(course_rows),
                      "new_rows": len(new_rows) if spec["rep"] == "C" else 0,
                      "new_loss_mass": float(weights[len(course_rows):].sum()),
                      "iterations": int(classifier.n_iter_), "converged": converged,
                      "cpu_seconds": time.process_time() - fit_start,
                      "warnings": [str(item.message) for item in caught]}
            fit_log.append(record)
            dump(output / "fit_log.json", fit_log)
            if not converged:
                raise RuntimeError("Locked SVM did not converge; no retry or retuning is permitted.")
            train_raw = classifier.decision_function(X)
            training = metric_values(truth, train_raw / SCALE - spec["threshold"])
            model, layout = export_model(vectors, classifier, spec["threshold"], output / "models" / name)
            checks, parity = {}, {}
            for panel_name, (panel, panel_text, panel_y) in dev.items():
                V = sparse.hstack([vectorizer.transform(panel_text) for vectorizer in vectors.values()], format="csr")
                raw = classifier.decision_function(V)
                reference = raw / SCALE - spec["threshold"]
                actual = model.decision_function(panel_text)
                max_error = float(np.max(np.abs(reference - actual)))
                equivalent = bool(np.array_equal(reference > 0, actual > 0))
                parity[panel_name] = {"max_abs_score_error": max_error, "prediction_identity": equivalent,
                                      "atol": 1e-9, "rtol": 1e-9}
                dump(output / (name + "_parity.json"), parity)
                np.testing.assert_allclose(reference, actual, atol=1e-9, rtol=1e-9)
                if not equivalent:
                    raise AssertionError("Export changes predicted classes.")
                checks[panel_name] = metric_values(panel_y, actual)
                for i, (row, text, label, margin, score) in enumerate(zip(panel, panel_text, panel_y, raw, actual)):
                    group = row.get("event_group") or row.get("group_id") or row.get("group") or text_group(text)
                    predictions.append({
                        "model": name, "dataset": panel_name,
                        "record_id": row.get("id") or row.get("record_id") or f"{panel_name}:{i}",
                        "group_id": group, "dataset_role": "reused_development",
                        "url": row["url"], "text": text, "actual": int(label),
                        "raw_margin": float(margin), "normalized_margin": float(margin / SCALE),
                        "threshold": spec["threshold"], "score": float(score),
                        "prediction": int(score > 0), "correct": bool((score > 0) == label),
                        "words": len(text.split()),
                    })
            results[name] = {"specification": spec, "development": checks,
                             "in_sample_diagnostics": {"fitted_rows": training},
                             "export_parity": parity, "feature_layout": layout,
                             "checkpoint_bytes": (output / "models" / name / "model.pt").stat().st_size}
            dump(output / "models_completed.json", results)
            write_predictions(output / "development_predictions.csv", predictions)
            print(json.dumps({"stage": "E2", "model": name, "development": checks}), flush=True)
        baseline = results["tuned_course"]["development"]
        candidate = results["augmented"]["development"]
        gates = {
            "G5_course_accuracy": candidate["course_dev"]["accuracy"] >= baseline["course_dev"]["accuracy"] - .005,
            "G5_course_nbc_recall": candidate["course_dev"]["per_class"]["NBC"]["recall"] >= baseline["course_dev"]["per_class"]["NBC"]["recall"] - .01,
            "G6_both_external_panels_non_decreasing": all(candidate[key]["correct"] >= baseline[key]["correct"] for key in ("external_dev", "paired_dev")),
            "G6_at_least_one_external_panel_improved": any(candidate[key]["correct"] > baseline[key]["correct"] for key in ("external_dev", "paired_dev")),
        }
        result = {"status": "complete", "stage": "E2", "models": results, "gates": gates,
                  "development_gate_passed": bool(all(gates.values())), "model_fits": len(fit_log),
                  "development_used_for_selection": False,
                  "internal_gate_evaluated_in": "E1/nested_results.json",
                  "promotion_policy": "Descriptive confirmation only. E1 failure cannot be reversed by E2.",
                  "v4_trained": False, "existing_models_replaced": False, "final_test_accessed": False,
                  "new_leaderboard_accuracy": None, "dataset_reads": inputs.reads,
                  "cpu_seconds": time.process_time() - start_cpu,
                  "wall_seconds": time.perf_counter() - start_wall,
                  "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                  "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                                  "torch": str(torch.__version__)}}
        dump(output / "results.json", result)
        dump(output / "status.json", {"status": "complete", "stage": "E2", "model_fits": len(fit_log)})
        return result
    except BaseException as exc:
        dump(output / "input_access_log.json", inputs.reads)
        dump(output / "status.json", {"status": "failed_or_interrupted", "stage": "E2",
                                      "error_type": type(exc).__name__, "error": str(exc),
                                      "model_fits": len(fit_log), "completed_models": list(results),
                                      "cpu_seconds": time.process_time() - start_cpu})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Repository root containing data/manifests/approved_inputs.json.")
    parser.add_argument("--output", type=Path, required=True, help="New E2 output directory.")
    args = parser.parse_args()
    run(args.package, args.output)


if __name__ == "__main__":
    main()
