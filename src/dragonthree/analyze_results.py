"""E3: descriptive analysis of approved saved results; never fits or predicts.

Inputs are explicit E0/E1/E2 result files and five named historical aggregate
files. Neither raw candidate pools nor sealed data are part of this interface.
Execute through the CPU ledger because bootstrap and plotting are experiment
analysis. Development uncertainty is conditional on reused panels and groups.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_INPUTS = {
    "baseline": "E0/baseline_metrics.json",
    "baseline_predictions": "E0/baseline_predictions.jsonl",
    "audit": "E0/data_audit.json",
    "nested": "E1/nested_results.json",
    "nested_summary": "E1/summary.json",
    "locked": "E2/results.json",
    "locked_predictions": "E2/development_predictions.csv",
}
HISTORICAL_INPUTS = {
    "v1": "evidence/results/v1_metrics.json",
    "v3": "evidence/results/v3_comparison.json",
    "v4": "evidence/results/v4_manifest.json",
    "v5": "evidence/results/v5_results.json",
}
OPTIONAL_STAGE_INPUTS = {
    "v4_gate": "E4/gate_decision.json",
    "freeze": "E5/final_decision.json",
    "runtime": "E6/runtime.json",
}
PANELS = ("course_dev", "external_dev", "paired_dev")
PANEL_NAMES = {"course_dev": "Course dev (951)", "external_dev": "Historical dev (306)",
               "paired_dev": "Paired dev (200)"}
MODELS = ("tuned_course", "augmented", "original_v4_validated")
MODEL_NAMES = {"tuned_course": "Locked A", "augmented": "Locked C", "original_v4_validated": "Original V4"}
LENGTH_BINS = ("0", "1-3", "4-5", "6-8", "9-12", "13+")
GROUP_ASSUMPTIONS = {
    "course_dev": "Normalized cleaned-text groups only; broader event dependence is unknown.",
    "external_dev": "Normalized cleaned-text groups only; historical event dependence is unknown.",
    "paired_dev": "Saved paired event_group identifiers; both publishers in an event are resampled together.",
}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_named(root, mapping, key, log):
    expected = root / mapping[key]
    path = expected.resolve(strict=True)
    if path != expected:
        raise PermissionError("Redirected result inputs are not approved: " + key)
    text = path.read_text(encoding="utf-8-sig")
    log.append({"key": key, "root_role": "historical_aggregates" if mapping is HISTORICAL_INPUTS else "run",
                "path": mapping[key]})
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if path.suffix == ".csv":
        return list(csv.DictReader(io.StringIO(text)))
    return json.loads(text)


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def length_bin(words):
    return "0" if words == 0 else "1-3" if words <= 3 else "4-5" if words <= 5 else "6-8" if words <= 8 else "9-12" if words <= 12 else "13+"


def bootstrap_difference(diff, groups, seed=5190, draws=5000):
    """Paired cluster bootstrap: resample clusters, retain all member records."""
    unique, inv = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=diff)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for start in range(0, draws, 250):
        indices = rng.integers(len(unique), size=(min(250, draws - start), len(unique)))
        bootstrap.extend((sums[indices].sum(1) / counts[indices].sum(1)).tolist())
    return {"n": len(diff), "groups": len(unique), "delta_accuracy": float(np.mean(diff)),
            "ci95": np.quantile(bootstrap, [.025, .975]).tolist(), "draws": draws, "seed": seed}


def metric_row(experiment, model, level, panel, metric, source, training_rows=None):
    per_class = metric.get("per_class", {})
    return {
        "experiment": experiment, "model": model, "evaluation_level": level,
        "panel": panel, "n_per_partition_or_panel": metric["n"],
        "correct": metric.get("correct"),
        "correct_by_partition": json.dumps(metric["correct_by_seed"]) if "correct_by_seed" in metric else "",
        "accuracy": metric["accuracy"], "balanced_accuracy": metric.get("balanced_accuracy"),
        "macro_f1": metric.get("macro_f1"), "roc_auc": metric.get("roc_auc", metric.get("auc")),
        "fox_recall": metric.get("fox_recall", per_class.get("FoxNews", {}).get("recall")),
        "nbc_recall": metric.get("nbc_recall", per_class.get("NBC", {}).get("recall")),
        "classifier_fit_rows_full_model": training_rows, "source": source,
    }


def historical_progression(history):
    original_n = int(history["v1"]["validation_rows"])
    records = [
        ("V1 baseline", history["v1"]["validation_accuracy"], HISTORICAL_INPUTS["v1"]),
        ("V2", history["v3"]["v2_accuracy"], HISTORICAL_INPUTS["v3"]),
        ("V3", history["v3"]["v3_accuracy"], HISTORICAL_INPUTS["v3"]),
        ("V4", history["v4"]["validation_accuracy"], HISTORICAL_INPUTS["v4"]),
        ("V5", history["v5"]["candidate_validation"]["accuracy"], HISTORICAL_INPUTS["v5"]),
    ]
    if original_n != 951 or history["v3"]["validation_rows"] != original_n or history["v4"]["validation_rows"] != original_n:
        raise ValueError("Historical progression requires the same 951-row development split.")
    return [metric_row("historical", name, "reused development", "course_dev",
                       {"n": original_n, "correct": round(accuracy * original_n), "accuracy": accuracy},
                       source, 3849) for name, accuracy, source in records]


def align_predictions(locked_rows, baseline_rows):
    grouped = {(model, panel): [] for model in MODELS for panel in PANELS}
    for source in locked_rows:
        model, panel = source["model"], source["dataset"]
        if model not in MODELS[:2] or panel not in PANELS:
            raise ValueError("Unexpected model/panel in E2 predictions.")
        row = dict(source)
        for key in ("actual", "prediction", "words"):
            row[key] = int(row[key])
        row["correct"] = row["actual"] == row["prediction"]
        grouped[model, panel].append(row)
    for source in baseline_rows:
        panel = "external_dev" if source["panel"] == "legacy_dev" else source["panel"]
        if panel not in PANELS:
            raise ValueError("Unexpected E0 panel.")
        reference = grouped["tuned_course", panel][source["row"]]
        if source["actual"] != reference["actual"] or source["length"] != reference["words"]:
            raise ValueError("E0/E2 row order or label mismatch.")
        row = dict(reference, prediction=int(source["prediction"]), correct=bool(source["correct"]),
                   score=float(source["score"]), model="original_v4_validated")
        grouped["original_v4_validated", panel].append(row)
    for panel, expected in zip(PANELS, (951, 306, 200)):
        for model in MODELS:
            if len(grouped[model, panel]) != expected:
                raise ValueError("Incomplete saved predictions: " + model + "/" + panel)
        a, c = grouped["tuned_course", panel], grouped["augmented", panel]
        if any(left["url"] != right["url"] or left["actual"] != right["actual"] or left["group_id"] != right["group_id"]
               for left, right in zip(a, c)):
            raise ValueError("Locked A/C predictions do not align.")
    return grouped


def error_analysis(grouped):
    summary, examples, intervals = {}, [], {}
    for panel in PANELS:
        a, c = grouped["tuned_course", panel], grouped["augmented", panel]
        counts = {"corrected_errors": 0, "new_errors": 0, "persistent_errors": 0, "both_correct": 0}
        for left, right in zip(a, c):
            category = ("both_correct" if left["correct"] and right["correct"] else
                        "new_errors" if left["correct"] else
                        "corrected_errors" if right["correct"] else "persistent_errors")
            counts[category] += 1
            if category != "both_correct":
                examples.append({"panel": panel, "category": category, "record_id": left["record_id"],
                                 "group_id": left["group_id"], "url": left["url"], "text": left["text"],
                                 "actual": left["actual"], "A_prediction": left["prediction"],
                                 "C_prediction": right["prediction"], "words": left["words"],
                                 "manual_cause_annotation": "not performed"})
        counts["n"] = len(a)
        counts["net_correct_gain"] = counts["corrected_errors"] - counts["new_errors"]
        summary[panel] = counts
        diff = np.array([int(right["correct"]) - int(left["correct"]) for left, right in zip(a, c)])
        intervals[panel] = {**bootstrap_difference(diff, [row["group_id"] for row in a]),
                            "group_assumption": GROUP_ASSUMPTIONS[panel],
                            "scope": "Conditional descriptive interval on reused development; no multiple-comparison adjustment."}
    return summary, examples, intervals


def slice_analysis(grouped):
    rows = []
    for panel in PANELS:
        for model in MODELS:
            for label, publisher in enumerate(("FoxNews", "NBC")):
                for size in LENGTH_BINS:
                    records = [row for row in grouped[model, panel] if row["actual"] == label and length_bin(row["words"]) == size]
                    correct = sum(row["correct"] for row in records)
                    rows.append({"panel": panel, "model": model, "publisher": publisher,
                                 "length_bin": size, "n": len(records), "correct": correct,
                                 "accuracy": correct / len(records) if records else None,
                                 "interpretation": "descriptive; small cells are not reliable improvement evidence"})
    return rows


def save_figure(fig, output, name, figure_log, source, scope):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / (name + "." + extension), dpi=170, bbox_inches="tight")
    plt.close(fig)
    figure_log.append({"figure": name, "files": [name + ".png", name + ".pdf"],
                       "source": source, "evaluation_level": scope})


def make_figures(output, history, nested, locked, baseline, slices, intervals):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    log = []
    fig, ax = plt.subplots(figsize=(6.7, 3.2))
    x = np.arange(len(history)); values = np.array([row["accuracy"] for row in history]) * 100
    ax.plot(x, values, "o-", color="#285f82", lw=2)
    for i, row in enumerate(history):
        ax.annotate(f'{row["correct"]}/951\n{row["accuracy"]:.2%}', (i, values[i]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=[row["model"] for row in history], ylabel="Accuracy (%)",
           ylim=(74, 88), title="Historical model progression: reused course development")
    ax.text(.01, .02, "All fitted on the original 3,849 training rows; V1 is our trained course baseline.", transform=ax.transAxes, fontsize=8)
    save_figure(fig, output, "model_progression", log, list(HISTORICAL_INPUTS.values()), "reused development")

    fixed = [row for row in nested["regularization_curves"] if row["C"] == 4 and abs(row["threshold"] + .05) < 1e-12]
    fixed = sorted(fixed, key=lambda row: row["representation"])
    if [row["representation"] for row in fixed] != list("ABCD"):
        raise ValueError("Expected all four fixed C=4, threshold=-.05 comparisons.")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, panel, title in zip(axes, ("course", "new", "old"), ("Course (3,730)", "New (2,000)", "Old (2,366)")):
        values = [100 * row[panel]["accuracy"] for row in fixed]
        ax.plot(list("ABCD"), values, "o-", color="#285f82")
        for i, value in enumerate(values):
            ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=title, ylabel="Accuracy (%)", xlabel="Representation")
        ax.margins(y=.4)
    fig.suptitle("Controlled training-only OOF comparison: C=4, threshold=-0.05", y=1.03)
    save_figure(fig, output, "fixed_representations", log, "E1/nested_results.json:regularization_curves", "fixed training-only OOF, mean over complete partitions")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, panel, title in zip(axes, ("course", "new", "old"), ("Course", "New", "Old")):
        for representation in "ABCD":
            points = sorted((row for row in nested["regularization_curves"] if row["representation"] == representation), key=lambda row: row["C"])
            ax.plot([row["C"] for row in points], [100 * row[panel]["accuracy"] for row in points], "o-", label=representation)
        ax.set(xscale="log", xticks=[.25, 1, 4, 16], xticklabels=[".25", "1", "4", "16"], title=title, xlabel="SVM C", ylabel="Accuracy (%)")
    axes[-1].legend(title="Representation", fontsize=8)
    fig.suptitle("Training-only regularization curves, fixed threshold -0.05", y=1.03)
    save_figure(fig, output, "regularization_curves", log, "E1/nested_results.json:regularization_curves", "fixed training-only OOF; exploratory secondary comparisons")

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for i, panel in enumerate(("course", "new", "old")):
        row = nested["primary_paired_deltas_vs_tuned_A"][panel]
        delta = 100 * row["delta"]; low, high = np.array(row["ci95"]) * 100
        ax.errorbar(delta, i, xerr=[[delta - low], [high - delta]], fmt="o", capsize=4, color="#285f82")
        ax.annotate(f"{delta:+.2f} [{low:+.2f}, {high:+.2f}] pp", (delta, i), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set(yticks=[0, 1, 2], yticklabels=["Course", "New", "Old"], ylim=(-.5, 2.6), xlabel="Accuracy difference (percentage points)",
           title="Nested selected augmentation minus equally tuned A")
    save_figure(fig, output, "nested_group_intervals", log, "E1/nested_results.json:primary_paired_deltas_vs_tuned_A", "nested training-only OOF; paired group bootstrap conditional on partitions")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, panel in zip(axes, PANELS):
        baseline_key = "legacy_dev" if panel == "external_dev" else panel
        metrics = [locked["models"][name]["development"][panel] for name in MODELS[:2]] + [baseline["panels"][baseline_key]]
        bars = ax.bar(["A", "C", "V4"], [100 * row["accuracy"] for row in metrics], color=["#739bb1", "#d98652", "#355d73"])
        for bar, row in zip(bars, metrics):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{row["correct"]}/{row["n"]}', ha="center", fontsize=8)
        ax.set(title=PANEL_NAMES[panel], ylabel="Accuracy (%)", ylim=(0, 100))
    fig.suptitle("Locked development confirmation; V4 is a separate architecture/reference", y=1.03)
    save_figure(fig, output, "locked_development", log, ["E2/results.json", "E0/baseline_metrics.json"], "reused development; differing model families/training pools")

    fig, axes = plt.subplots(3, 2, figsize=(9, 8.1), sharex=True, sharey=True)
    colors = {"tuned_course": "#739bb1", "augmented": "#d98652", "original_v4_validated": "#355d73"}
    for row_index, panel in enumerate(PANELS):
        for col_index, publisher in enumerate(("FoxNews", "NBC")):
            ax = axes[row_index, col_index]
            for model in MODELS:
                points = [next(row for row in slices if row["panel"] == panel and row["model"] == model and row["publisher"] == publisher and row["length_bin"] == size) for size in LENGTH_BINS]
                ax.plot(range(len(LENGTH_BINS)), [100 * point["accuracy"] if point["n"] else np.nan for point in points], "o-", color=colors[model], label=MODEL_NAMES[model])
            counts = [point["n"] for point in points]
            ax.set(title=PANEL_NAMES[panel] + ": " + publisher, ylim=(-2, 102), ylabel="Accuracy (%)",
                   xticks=range(len(LENGTH_BINS)), xticklabels=[size + "\nn=" + str(n) for size, n in zip(LENGTH_BINS, counts)])
            ax.tick_params(labelbottom=True)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Publisher × cleaned-text length: descriptive reused-dev slices", y=1.01)
    save_figure(fig, output, "publisher_length_slices", log, "E3/publisher_length_slices.csv", "reused development; unadjusted exploratory small-cell summaries")
    dump(output / "figure_sources.json", log)
    return log


def stages_table(audit, baseline, nested, locked, errors, intervals, later_stages):
    e0 = "; ".join(f'{panel}: {row["correct"]}/{row["n"]}' for panel, row in baseline["panels"].items())
    internal = bool(nested["internal_gate_passed"])
    folds = nested.get("nested_completed_outer_folds", "unknown")
    delta = nested["primary_paired_deltas_vs_tuned_A"]
    e1 = f"Nested {folds}/9 folds; internal gate={internal}. " + "; ".join(f'{panel} {100 * row["delta"]:+.3f} pp [{100 * row["ci95"][0]:+.3f}, {100 * row["ci95"][1]:+.3f}]' for panel, row in delta.items())
    e2 = "; ".join(f'{panel}: A {locked["models"]["tuned_course"]["development"][panel]["correct"]}/{locked["models"]["tuned_course"]["development"][panel]["n"]}, C {locked["models"]["augmented"]["development"][panel]["correct"]}/{locked["models"]["augmented"]["development"][panel]["n"]}' for panel in PANELS)
    e2 += f'; development gate={locked["development_gate_passed"]}; fits={locked["model_fits"]}'
    e3 = "; ".join(f'{panel}: fixed {row["corrected_errors"]}, new errors {row["new_errors"]}, persistent {row["persistent_errors"]}; net {row["net_correct_gain"]:+d}' for panel, row in errors.items())
    rows = [
        {"id": "E0", "status": "complete", "what": "Audit approved inputs; reproduce original validated V4", "why": "Establish input contracts and a trustworthy fallback", "result": e0 + f'; audited training-origin rows={audit["training_rows"]}',
         "做了什么": "审计获准输入，并复核原 V4 validated", "为什么": "确认数据边界和保底模型的可复现性", "结果": e0 + f'；审计训练来源记录 {audit["training_rows"]} 条'},
        {"id": "E1", "status": "complete" if folds == 9 else "partial", "what": "Complete the missing nested fold without changing the locked protocol", "why": "Estimate the selection procedure using all three complete partitions", "result": e1,
         "做了什么": "按冻结协议补缺失嵌套折", "为什么": "用三套完整划分评估选择流程", "结果": e1},
        {"id": "E2", "status": locked["status"], "what": "Fit/export locked A and C once, then confirm on three development panels", "why": "Measure transfer after internal selection without dev retuning", "result": e2,
         "做了什么": "锁定 A/C 各拟合与导出一次，在三个开发面板确认", "为什么": "检验内部收益是否迁移，且不按开发结果再调参", "结果": e2},
        {"id": "E3", "status": "complete", "what": "Summarize paired errors, length slices, intervals, and controlled comparisons", "why": "Explain where changes occur and quantify conditional uncertainty", "result": e3,
         "做了什么": "汇总错误变化、长度切片、区间与受控对照", "为什么": "说明改进或退步发生在哪里，并量化条件不确定性", "结果": e3},
        {"id": "E4", "status": "pending_conditional_stage" if internal and locked["development_gate_passed"] else "not_started_gate_failed",
         "what": "Conditionally migrate the fixed lexical adaptation to V4", "why": "Require internal and development protection gates before spending the conditional budget",
         "result": "Prerequisites passed; consult the separately approved E4 protocol and execution record." if internal and locked["development_gate_passed"] else "Not trained: at least one E1/E2 prerequisite failed. This is a gate decision, not a completed V4 experiment.",
         "做了什么": "有条件地把固定词法适配迁移到 V4", "为什么": "先通过内部与开发保护门槛，再使用条件预算",
         "结果": "前置门槛通过；执行情况以单独批准的 E4 协议和记录为准。" if internal and locked["development_gate_passed"] else "未训练：E1/E2 至少一项前置门槛失败。这是停止决策，不是完成了 V4 训练实验。"},
        {"id": "E5", "status": "pending_separate_stage", "what": "Freeze the final model and its provenance", "why": "Keep validated scoring separate from the full-data submission artifact", "result": "Read E5 stage output; E3 does not select or refit a model.",
         "做了什么": "冻结最终模型及其来源", "为什么": "分开开发评估模型和全数据提交模型", "结果": "以 E5 阶段产物为准；E3 不选择或重拟合模型。"},
        {"id": "E6", "status": "pending_separate_stage", "what": "Verify deployment interfaces and CPU runtime", "why": "Check the course inference contract on approved/synthetic inputs", "result": "Read E6 stage output; a timing fixture is not the teacher test set.",
         "做了什么": "验证部署接口与 CPU 耗时", "为什么": "在获准或合成输入上检查课程推理要求", "结果": "以 E6 阶段产物为准；计时样例不是教师测试集。"},
    ]
    for row in rows:
        key = "v4_gate" if row["id"] == "E4" else "freeze" if row["id"] == "E5" else "runtime" if row["id"] == "E6" else None
        if key and key in later_stages:
            evidence = later_stages[key]
            row["status"] = evidence.get("status", "recorded_see_stage_evidence")
            if key == "v4_gate":
                course_check, nbc_check = evidence["course_protection"], evidence["NBC_protection"]
                counts = (f'Course {course_check["A_correct"]}→{course_check["C_correct"]}/{course_check["n"]}; '
                          f'NBC {nbc_check["A_true_positives"]}→{nbc_check["C_true_positives"]}/{nbc_check["n"]}; '
                          f'allowed extra errors/misses {course_check["maximum_extra_errors"]}/{nbc_check["maximum_extra_misses"]}.')
                row["result"] = evidence["reason"] + " " + counts + f' Model fits={evidence["model_fits"]}.'
                row["结果"] = ("E4 已有条件授权；前置门槛失败，未启动训练：" + ", ".join(evidence["failed_prerequisites"]) + "。" + counts + " 模型拟合次数为 0。") if not evidence["prerequisites_passed"] else "前置门槛通过；此文件仅记录检查，尚不代表执行或完成 V4 训练。"
            elif key == "freeze":
                row["result"] = f'Decision={evidence.get("decision")}; fitting rows={evidence.get("training_rows")}; new training={evidence.get("new_training_performed")}; original models overwritten={evidence.get("original_models_overwritten")}. The 810/951 score belongs to the validated sibling, not the submission fitted on all 4,800 rows.'
                row["结果"] = f'决策={evidence.get("decision")}；拟合行数={evidence.get("training_rows")}；新增训练={evidence.get("new_training_performed")}；覆盖原模型={evidence.get("original_models_overwritten")}。810/951 属于 validated 模型，不是拟合全部 4800 行的提交模型的独立开发成绩。'
            else:
                timings = []
                for threads in sorted({record["threads"] for record in evidence.get("runs", [])}):
                    values = [record["total_wall_seconds"] for record in evidence["runs"] if record["threads"] == threads]
                    timings.append(f'{threads} threads, {len(values)} repeats: {min(values):.2f}–{max(values):.2f}s wall')
                row["result"] = "; ".join(timings) + f'; PyTorch {evidence.get("torch")}; target passed={evidence.get("all_runs_below_target")}. Approved 8,000-row fixture only; server hardware equivalence is unknown.'
                row["结果"] = "; ".join(timings) + f'；PyTorch {evidence.get("torch")}；通过本地目标={evidence.get("all_runs_below_target")}。仅使用获准的 8000 行计时样例，服务器硬件等价性未知。'
    return rows


def write_tables(output, rows):
    english = "# Approved experiments: what, why, and results\n\nE0 is foundation work; the six requested experiment IDs are E1–E6. Results are drawn from saved evidence. Training-only OOF, reused development, and leaderboard performance are distinct. No new leaderboard result exists. E4 remains conditional on E1/E2 gates; a failed gate is not a completed training experiment.\n\n| Stage | Status | What was done | Why | Result |\n|---|---|---|---|---|\n"
    chinese = "# 获准实验：做了什么、为什么、结果\n\nE0 是基础准备，用户确认的六个实验编号为 E1–E6。以下数字来自保存的实验结果。训练内部 OOF、复用开发集和排行榜成绩分别记录；本轮没有新排行榜成绩。E4 须通过 E1/E2 前置门槛；门槛失败停止不等于完成了训练。\n\n| 阶段 | 状态 | 做了什么 | 为什么 | 结果 |\n|---|---|---|---|---|\n"
    for row in rows:
        english += "| " + " | ".join(str(row[key]).replace("|", "/") for key in ("id", "status", "what", "why", "result")) + " |\n"
        chinese += "| " + " | ".join(str(row[key]).replace("|", "/") for key in ("id", "status", "做了什么", "为什么", "结果")) + " |\n"
    (output / "EXPERIMENT_RESULTS_EN.md").write_text(english)
    (output / "EXPERIMENT_RESULTS_ZH.md").write_text(chinese)


def run(run_root, package, output, figure_output=None):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    run_root, package, output = Path(run_root).resolve(strict=True), Path(package).resolve(strict=True), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("E3 output must be new or empty.")
    output.mkdir(parents=True, exist_ok=True)
    figure_output = Path(figure_output).resolve() if figure_output else output / "figures"
    accesses = []
    evidence = {key: read_named(run_root, RUN_INPUTS, key, accesses) for key in RUN_INPUTS}
    history = {key: read_named(package, HISTORICAL_INPUTS, key, accesses) for key in HISTORICAL_INPUTS}
    later_stages = {key: read_named(run_root, OPTIONAL_STAGE_INPUTS, key, accesses)
                    for key, relative in OPTIONAL_STAGE_INPUTS.items() if (run_root / relative).exists()}
    baseline, nested, locked = evidence["baseline"], evidence["nested"], evidence["locked"]
    if locked["status"] != "complete" or locked["model_fits"] != 2:
        raise ValueError("E3 requires the completed two-fit locked comparison.")
    grouped = align_predictions(evidence["locked_predictions"], evidence["baseline_predictions"])
    for model in MODELS:
        for panel in PANELS:
            if model == "original_v4_validated":
                metric = baseline["panels"]["legacy_dev" if panel == "external_dev" else panel]
            else:
                metric = locked["models"][model]["development"][panel]
            if sum(row["correct"] for row in grouped[model, panel]) != metric["correct"]:
                raise ValueError("Saved prediction correctness differs from the aggregate result: " + model + "/" + panel)
    errors, examples, intervals = error_analysis(grouped)
    slices = slice_analysis(grouped)
    progression = historical_progression(history)
    master = list(progression)
    for panel, metric in baseline["panels"].items():
        master.append(metric_row("E0", baseline["model"], "reused development", "external_dev" if panel == "legacy_dev" else panel,
                                 metric, RUN_INPUTS["baseline"], 3849))
    for row in nested["policies"]:
        for panel in ("course", "new", "old"):
            master.append(metric_row("E1_nested", row["policy"], "nested training-only OOF; mean over partitions", panel,
                                     row[panel], RUN_INPUTS["nested"]))
    for row in nested["regularization_curves"]:
        for panel in ("course", "new", "old"):
            master.append(metric_row("E1_fixed_grid", f'{row["representation"]};C={row["C"]};tau={row["threshold"]}',
                                     "fixed training-only OOF; mean over partitions", panel, row[panel], RUN_INPUTS["nested"]))
    for name, candidate in locked["models"].items():
        for panel, metric in candidate["development"].items():
            master.append(metric_row("E2", name, "reused development", panel, metric, RUN_INPUTS["locked"], 3730 if name == "tuned_course" else 5730))
    dump(output / "results_master.json", {"records": master, "new_leaderboard_result": None,
         "scope": "Historical and new evidence retain their own evaluation levels, data versions, and denominators."})
    write_csv(output / "results_master.csv", master)
    dump(output / "error_transitions.json", errors)
    write_csv(output / "error_review_queue.csv", examples)
    dump(output / "development_paired_intervals.json", intervals)
    write_csv(output / "publisher_length_slices.csv", slices)
    dump(output / "historical_progression.json", progression)
    figures = make_figures(figure_output, progression, nested, locked, baseline, slices, intervals)
    table = stages_table(evidence["audit"], baseline, nested, locked, errors, intervals, later_stages)
    dump(output / "experiment_table.json", table)
    write_tables(output, table)
    dump(output / "input_access_log.json", accesses)
    result = {"status": "complete", "stage": "E3", "new_model_fits": 0, "new_model_predictions": 0,
              "error_transitions": errors, "development_paired_intervals": intervals,
              "master_records": len(master), "figure_count": len(figures),
              "final_test_accessed": False, "manual_error_cause_review": "not performed",
              "saved_predictions_match_aggregate_correct_counts": True,
              "historical_in_sample_diagnostics": [
                  {"representation": row["representation"], "C": row["C"],
                   "available_folds": row.get("in_sample_diagnostics_available_folds"),
                   "expected_folds": row.get("in_sample_diagnostics_expected_folds"),
                   "complete": row.get("in_sample_diagnostics_complete")}
                  for row in evidence["nested_summary"]["fixed_feature_results"]
              ],
              "limitations": ["Development panels have already been reused; these are descriptive confirmations.",
                              "Group bootstrap is conditional on available groups and saved predictions.",
                              "Course/legacy event dependencies beyond normalized text are unknown.",
                              "Length slices and secondary comparisons are not multiplicity adjusted.",
                              "Historical in-sample diagnostic logs cover eight folds only; the OOF plots use all nine completed folds.",
                              "Original V4 and locked V3 candidates differ in architecture and training data.",
                              "No hidden-test or new leaderboard result is available."],
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    dump(output / "results.json", result)
    print(json.dumps({key: result[key] for key in ("status", "stage", "master_records", "figure_count", "cpu_seconds")}), flush=True)
    return result


def refresh_stages(run_root, output):
    """Refresh only the evidence table after later stages; no bootstrap or plots."""
    run_root, output = Path(run_root).resolve(strict=True), Path(output).resolve(strict=True)
    accesses = []
    evidence = {key: read_named(run_root, RUN_INPUTS, key, accesses)
                for key in ("audit", "baseline", "nested", "locked")}
    later = {key: read_named(run_root, OPTIONAL_STAGE_INPUTS, key, accesses)
             for key, relative in OPTIONAL_STAGE_INPUTS.items() if (run_root / relative).exists()}
    errors = read_named(output, {"errors": "error_transitions.json"}, "errors", accesses)
    intervals = read_named(output, {"intervals": "development_paired_intervals.json"}, "intervals", accesses)
    table = stages_table(evidence["audit"], evidence["baseline"], evidence["nested"], evidence["locked"], errors, intervals, later)
    dump(output / "experiment_table.json", table)
    write_tables(output, table)
    dump(output / "stage_table_refresh.json", {"status": "complete", "new_model_fits": 0,
         "new_model_predictions": 0, "bootstrap_rerun": False, "plots_rerun": False,
         "analytical_metrics_changed": False, "stage_evidence_available": list(later),
         "reads": accesses})
    print(json.dumps({"stage_table_refreshed": True, "later_stages": list(later)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--refresh-stages", action="store_true", help="Refresh only the existing bilingual stage table.")
    args = parser.parse_args()
    if args.refresh_stages:
        refresh_stages(args.run, args.output)
    else:
        run(args.run, args.package, args.output, args.figure_output)


if __name__ == "__main__":
    main()
