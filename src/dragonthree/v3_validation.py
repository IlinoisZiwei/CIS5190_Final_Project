"""Passive, fold-local implementation of the locked historical V3 study.

Importing this module does not read data, fit models, set process limits, or run
historical scripts. The protocol is transcribed from the approved 2026-10-29
study, including its fixed margin scale, sample weights and selection rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import warnings

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


REPRESENTATIONS = ("A", "B", "C", "D")
C_VALUES = (0.25, 1.0, 4.0, 16.0)
THRESHOLDS = tuple(round(-0.5 + 0.05 * i, 2) for i in range(21))
SCORE_SCALE = 0.776760627876633
FIXED_THRESHOLD = -0.05
SEEDS = (5190, 5191, 5192)
POLICIES = tuple(
    f"{rep}_{mode}"
    for mode in ("fixed", "C", "threshold", "joint")
    for rep in REPRESENTATIONS
) + ("augmented_joint",)
LOCKED_SETTINGS = {
    "tuned_course": {"rep": "A", "C": 0.25, "threshold": -0.15},
    "augmented": {"rep": "C", "C": 1.0, "threshold": -0.20},
}


class ProtocolError(RuntimeError):
    """A required historical invariant could not be verified."""


@dataclass
class TrainingData:
    texts: list[str]
    y: np.ndarray
    groups: np.ndarray
    origin: np.ndarray
    words: np.ndarray
    ids: list[str]

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "TrainingData":
        if len(rows) != 8096:
            raise ProtocolError(f"Expected 8096 approved training records, found {len(rows)}")
        if any(row["label"] not in {"FoxNews", "NBC"} for row in rows):
            raise ProtocolError("Unknown publisher label in approved training records")
        obj = cls(
            texts=[str(row["text"]) for row in rows],
            y=np.asarray([int(row["label"] == "NBC") for row in rows], dtype=int),
            groups=np.asarray([str(row["group"]) for row in rows]),
            origin=np.asarray([str(row["origin"]) for row in rows]),
            words=np.asarray([int(row["words"]) for row in rows], dtype=int),
            ids=[str(row["id"]) for row in rows],
        )
        expected = {"course": 3730, "new": 2000, "legacy": 1566, "paired": 800}
        observed = {k: int((obj.origin == k).sum()) for k in np.unique(obj.origin)}
        if observed != expected:
            raise ProtocolError(f"Training pool counts changed: {observed}")
        if len(set(obj.ids)) != len(obj.ids) or any(not g for g in obj.groups):
            raise ProtocolError("Record IDs must be unique and groups nonempty")
        if not np.array_equal(obj.words, np.asarray([len(t.split()) for t in obj.texts])):
            raise ProtocolError("Stored word counts do not match approved model text")
        return obj

    @property
    def course(self):
        return self.origin == "course"

    @property
    def new(self):
        return self.origin == "new"

    @property
    def old(self):
        return np.isin(self.origin, ["legacy", "paired"])

    @property
    def strata(self):
        codes = {"course": 0, "legacy": 1, "paired": 2, "new": 3}
        return np.asarray([codes[o] * 2 + int(y) for o, y in zip(self.origin, self.y)])


@dataclass
class RunLog:
    """Shared counters and a caller-supplied persistence/budget callback."""
    max_fits: int = 65
    check_budget: Callable[[], None] = lambda: None
    persist: Callable[[], None] = lambda: None
    fits: list[dict] = field(default_factory=list)
    features: list[dict] = field(default_factory=list)
    splits: list[dict] = field(default_factory=list)

    def before_fit(self):
        self.check_budget()
        if len(self.fits) >= self.max_fits:
            raise ProtocolError(f"Fit cap of {self.max_fits} reached")


def make_vectorizer(name: str) -> TfidfVectorizer:
    shared = {
        "lowercase": True, "strip_accents": "unicode",
        "dtype": np.float64, "sublinear_tf": True,
    }
    if name == "word12":
        return TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1,
            token_pattern=r"(?u)\b\w+\b", **shared,
        )
    if name == "char25":
        return TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=2,
            max_features=120000, **shared,
        )
    raise ValueError(f"Unknown feature block: {name}")


def assert_group_disjoint(data: TrainingData, tr: np.ndarray, va: np.ndarray):
    if np.intersect1d(tr, va).size:
        raise ProtocolError("A row occurs in both fitting and validation indices")
    if set(data.groups[tr]) & set(data.groups[va]):
        raise ProtocolError("Dependency groups cross fitting and validation partitions")


def build_representations(data: TrainingData, tr: np.ndarray, tag: str, log: RunLog):
    """Fit only course/new rows of tr, then transform the approved training ledger.

    A/B share course-fitted vocabulary and IDF. C retains A vocabulary with joint
    unweighted IDF. D expands vocabulary and uses the same joint IDF. The direct
    C transform is checked against the historical restrict-D-and-renormalize
    optimization. Old-panel rows are transformed but never fitted.
    """
    log.check_budget()
    ct = tr[data.course[tr]]
    nt = tr[data.new[tr]]
    joint = np.r_[ct, nt]
    a = {name: make_vectorizer(name).fit([data.texts[i] for i in ct])
         for name in ("word12", "char25")}
    c = {
        name: clone(v).set_params(vocabulary=dict(v.vocabulary_)).fit(
            [data.texts[i] for i in joint]
        ) for name, v in a.items()
    }
    d = {name: make_vectorizer(name).fit([data.texts[i] for i in joint])
         for name in a}
    for name in a:
        if a[name].vocabulary_ != c[name].vocabulary_:
            raise ProtocolError("C vocabulary differs from A")
        if not set(a[name].vocabulary_) <= set(d[name].vocabulary_):
            raise ProtocolError("D feature cap removed an A term; protocol cannot continue")
        common = np.asarray([d[name].vocabulary_[t] for t in a[name].get_feature_names_out()])
        np.testing.assert_allclose(c[name].idf_, d[name].idf_[common], rtol=0, atol=1e-12)

    ablocks = {name: v.transform(data.texts) for name, v in a.items()}
    dblocks = {name: v.transform(data.texts) for name, v in d.items()}
    cblocks = {
        name: normalize(
            dblocks[name][:, [d[name].vocabulary_[t] for t in a[name].get_feature_names_out()]],
            norm="l2", copy=True,
        ) for name in a
    }
    check_rows = tr[:min(32, len(tr))]
    for name in a:
        np.testing.assert_allclose(
            cblocks[name][check_rows].toarray(),
            c[name].transform([data.texts[i] for i in check_rows]).toarray(),
            atol=1e-12, rtol=1e-12,
        )
    matrices = {
        rep: sparse.hstack(list(blocks.values()), format="csr")
        for rep, blocks in (("A", ablocks), ("C", cblocks), ("D", dblocks))
    }
    matrices["B"] = matrices["A"]
    vectors = {"A": a, "B": a, "C": c, "D": d}
    log.features.append({
        "tag": tag, "course_fit_rows": len(ct), "new_fit_rows": len(nt),
        "course_fit_indices": ct.tolist(), "new_fit_indices": nt.tolist(),
        "feature_counts": {rep: {name: len(v.vocabulary_) for name, v in vectors[rep].items()}
                           for rep in ("A", "C", "D")},
        "C_vocabulary_equals_A": True, "D_retains_A": True,
        "C_D_common_idf_equal": True, "optimized_C_transform_parity": True,
        "old_panel_fitted": False,
    })
    log.persist()
    return vectors, matrices


def fit_svm(
    data: TrainingData, X, tr: np.ndarray, rep: str, C: float,
    tag: str, log: RunLog,
) -> LinearSVC:
    log.before_fit()
    ct = tr[data.course[tr]]
    nt = tr[data.new[tr]] if rep != "A" else np.asarray([], dtype=int)
    ix = np.r_[ct, nt]
    weights = np.ones(len(ix))
    if len(nt):
        for label in (0, 1):
            where = np.flatnonzero(data.y[nt] == label)
            if not len(where):
                raise ProtocolError("Both publishers must occur in each new fitting pool")
            weights[len(ct) + where] = 100 * len(ct) / 3730 / len(where)
    record = {
        "tag": tag, "representation": rep, "C": float(C), "status": "started",
        "course_rows": len(ct), "new_rows": len(nt),
        "new_loss_mass": float(weights[len(ct):].sum()), "fitting_indices": ix.tolist(),
    }
    log.fits.append(record)
    log.persist()
    model = LinearSVC(C=C, dual=True, max_iter=20000, random_state=42)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X[ix], data.y[ix], sample_weight=weights)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise ProtocolError("LinearSVC did not converge")
        record.update(status="complete", iterations=int(model.n_iter_))
        log.persist()
        log.check_budget()
        return model
    except BaseException as exc:
        if record["status"] != "complete":
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        log.persist()
        raise


def grid_fit(data: TrainingData, tr: np.ndarray, va: np.ndarray, tag: str, log: RunLog):
    assert_group_disjoint(data, tr, va)
    _, matrices = build_representations(data, tr, tag, log)
    result = np.empty((4, 4, len(va)))
    for ri, rep in enumerate(REPRESENTATIONS):
        for ci, C in enumerate(C_VALUES):
            model = fit_svm(data, matrices[rep], tr, rep, C, tag, log)
            result[ri, ci] = model.decision_function(matrices[rep][va]) / SCORE_SCALE
    if not np.isfinite(result).all():
        raise ProtocolError("A fitted grid produced non-finite scores")
    return result


def candidate_records(data: TrainingData, raw: np.ndarray, scope: np.ndarray):
    """raw has repeat x representation x C x all-approved-record dimensions."""
    masks = {
        "course": scope[data.course[scope]],
        "new": scope[data.new[scope]],
        "old": scope[data.old[scope]],
        "course_nbc": scope[data.course[scope] & (data.y[scope] == 1)],
    }
    records = []
    for ri, rep in enumerate(REPRESENTATIONS):
        for ci, C in enumerate(C_VALUES):
            block = raw[:, ri, ci]
            if not np.isfinite(block[:, scope]).all():
                raise ProtocolError("Inner selection received incomplete held-out scores")
            values = {}
            for name, ix in masks.items():
                if not len(ix):
                    raise ProtocolError(f"Empty required selection panel: {name}")
                predicted = block[:, ix, None] > np.asarray(THRESHOLDS)
                values[name] = np.mean(predicted == data.y[ix][None, :, None], axis=(0, 1))
            for ti, threshold in enumerate(THRESHOLDS):
                records.append({
                    "rep": rep, "C": C, "threshold": threshold,
                    **{name: float(v[ti]) for name, v in values.items()},
                })
    return records


def fixed(records, rep="A"):
    return next(r for r in records
                if r["rep"] == rep and r["C"] == 4 and r["threshold"] == FIXED_THRESHOLD)


def choose(records, reps, mode, reference):
    pool = [
        r for r in records if r["rep"] in reps
        and (mode not in ("fixed", "threshold") or r["C"] == 4)
        and (mode not in ("fixed", "C") or r["threshold"] == FIXED_THRESHOLD)
    ]
    eligible = [
        r for r in pool
        if r["course"] >= reference["course"] - .005 - 1e-12
        and r["course_nbc"] >= reference["course_nbc"] - .01 - 1e-12
        and r["old"] >= reference["old"] - 1e-12
    ]
    if not eligible:
        return dict(fixed(records, reps[0]), eligible=False, fallback=True)
    best = sorted(eligible, key=lambda r: (
        -(r["new"] + r["old"]) / 2, -r["course"], r["C"],
        abs(r["threshold"] + .05), r["rep"],
    ))[0]
    return dict(best, eligible=True, fallback=False)


def select_policies(records):
    baseline, selected = fixed(records), {}
    for mode in ("fixed", "C", "threshold", "joint"):
        for rep in REPRESENTATIONS:
            selected[f"{rep}_{mode}"] = (
                dict(fixed(records, rep), eligible=True, fallback=False)
                if mode == "fixed" else choose(records, [rep], mode, baseline)
            )
    tuned = selected["A_joint"]
    reference = {k: max(baseline[k], tuned[k]) for k in ("course", "course_nbc", "old")}
    selected["augmented_joint"] = choose(records, ["B", "C", "D"], "joint", reference)
    return selected


def inner_selection(
    data: TrainingData, tr: np.ndarray, seed: int, tag: str,
    log: RunLog, fold_callback: Callable[[int, np.ndarray], None] | None = None,
):
    raw = np.full((1, 4, 4, len(data.y)), np.nan)
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    for fi, (a, b) in enumerate(splitter.split(np.zeros(len(tr)), data.strata[tr], data.groups[tr])):
        it, iv = tr[a], tr[b]
        assert_group_disjoint(data, it, iv)
        log.splits.append({
            "tag": tag, "fold": fi, "inner_seed": seed,
            "train": it.tolist(), "validation": iv.tolist(),
        })
        log.persist()
        block = grid_fit(data, it, iv, f"{tag}_inner{fi}", log)
        raw[0, :, :, iv] = np.moveaxis(block, -1, 0)
        if fold_callback is not None:
            fold_callback(fi, raw)
    return select_policies(candidate_records(data, raw, tr)), raw


def metric_values(y, scores):
    """NBC-positive metrics for margins already adjusted by the locked threshold."""
    y, scores = np.asarray(y), np.asarray(scores)
    prediction = (scores > 0).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=[0, 1], zero_division=0,
    )
    return {
        "n": len(y), "correct": int((prediction == y).sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(("FoxNews", "NBC"))
        },
    }


def metric_repeats(data: TrainingData, scores: np.ndarray, mask: np.ndarray):
    values = [metric_values(data.y[mask], s[mask]) for s in scores]
    return {
        "n": int(mask.sum()),
        "accuracy": float(np.mean([v["accuracy"] for v in values])),
        "accuracy_by_seed": [v["accuracy"] for v in values],
        "correct_by_seed": [v["correct"] for v in values],
        "balanced_accuracy": float(np.mean([v["balanced_accuracy"] for v in values])),
        "macro_f1": float(np.mean([v["macro_f1"] for v in values])),
        "roc_auc": float(np.mean([v["roc_auc"] for v in values])),
        "fox_recall": float(np.mean([v["per_class"]["FoxNews"]["recall"] for v in values])),
        "nbc_recall": float(np.mean([v["per_class"]["NBC"]["recall"] for v in values])),
    }


def group_bootstrap(data: TrainingData, diff: np.ndarray, mask: np.ndarray):
    """Historical paired group bootstrap: 20 x 250 resamples, seed 5190."""
    _, inverse = np.unique(data.groups[mask], return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=diff[mask])
    rng, draws = np.random.default_rng(5190), []
    for _ in range(20):
        ix = rng.integers(len(counts), size=(250, len(counts)))
        draws.extend((sums[ix].sum(1) / counts[ix].sum(1)).tolist())
    return np.quantile(draws, [.025, .975]).tolist()


def summarize(
    data: TrainingData, raw: np.ndarray, policy_scores: np.ndarray,
    choices_log: list[dict], complete_seeds: list[int], complete_folds: int,
):
    if not np.isfinite(raw).all() or not np.isfinite(policy_scores).all():
        raise ProtocolError("Only complete partitions can be summarized")
    summaries = []
    for pi, policy in enumerate(POLICIES):
        score = policy_scores[:, pi]
        short_fox = data.new & (data.y == 0) & (data.words <= 5)
        short_nbc = data.new & (data.y == 1) & (data.words <= 5)
        summaries.append({
            "policy": policy,
            "course": metric_repeats(data, score, data.course),
            "new": metric_repeats(data, score, data.new),
            "old": metric_repeats(data, score, data.old),
            "new_short_fox_recall": float((score[:, short_fox] <= 0).mean()),
            "new_short_nbc_recall": float((score[:, short_nbc] > 0).mean()),
        })
    by = {r["policy"]: r for r in summaries}
    primary, baseline = POLICIES.index("augmented_joint"), POLICIES.index("A_joint")
    correct = (policy_scores > 0) == data.y
    diff = correct[:, primary].mean(0) - correct[:, baseline].mean(0)
    deltas = {
        key: {"delta": float(diff[mask].mean()), "ci95": group_bootstrap(data, diff, mask)}
        for key, mask in (("course", data.course), ("new", data.new), ("old", data.old))
    }
    gains = correct[:, primary, data.new].mean(1) - correct[:, baseline, data.new].mean(1)
    candidate, tuned, fixed_a = by["augmented_joint"], by["A_joint"], by["A_fixed"]
    checks = {
        "all_nine_nested_folds_complete": complete_folds == 9 and complete_seeds == list(SEEDS),
        "course_accuracy_protected": candidate["course"]["accuracy"] >=
            max(tuned["course"]["accuracy"], fixed_a["course"]["accuracy"]) - .005,
        "course_nbc_recall_protected": candidate["course"]["nbc_recall"] >=
            max(tuned["course"]["nbc_recall"], fixed_a["course"]["nbc_recall"]) - .01,
        "old_panel_accuracy_protected": candidate["old"]["accuracy"] >=
            max(tuned["old"]["accuracy"], fixed_a["old"]["accuracy"]),
        "new_gain_positive_at_least_two_partitions": int((gains > 0).sum()) >= 2,
        "new_group_bootstrap_lower95_positive": deltas["new"]["ci95"][0] > 0,
    }
    curves = []
    for ri, rep in enumerate(REPRESENTATIONS):
        for ci, C in enumerate(C_VALUES):
            score = raw[:, ri, ci] - FIXED_THRESHOLD
            curves.append({
                "representation": rep, "C": C, "threshold": FIXED_THRESHOLD,
                "course": metric_repeats(data, score, data.course),
                "new": metric_repeats(data, score, data.new),
                "old": metric_repeats(data, score, data.old),
            })
    return {
        "policies": summaries, "regularization_curves": curves,
        "primary_paired_deltas_vs_tuned_A": deltas,
        "primary_new_gain_by_seed": gains.tolist(),
        "internal_gate_passed": bool(all(checks.values())),
        "gate_checks": {k: bool(v) for k, v in checks.items()},
        "primary_augmented_inner_eligible_folds": sum(
            bool(r["choices"]["augmented_joint"]["eligible"]) for r in choices_log
        ),
        "nested_complete_partitions": complete_seeds,
        "nested_completed_outer_folds": complete_folds,
        "fixed_grid_complete_partitions": list(SEEDS),
        "final_selection_changed": False,
        "final_test_accessed": False,
        "development_evaluated": False,
        "uncertainty": (
            "Paired dependency-group bootstrap, conditional on reused training records "
            "and folds. Repeated folds are not independent samples. Secondary policies "
            "are exploratory and are not multiplicity-adjusted."
        ),
    }
