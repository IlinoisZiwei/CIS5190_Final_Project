# DragonThree · CIS 5190 Option B

[English](README.md) · [Overleaf 报告](https://www.overleaf.com/project/6aea3dd3f59e426e6b892e27)

任务：根据清理后的 URL 最后一段文字判断 FoxNews 或 NBC。最终保留 V4。

1. [01 数据采集、清理与分析](01_Data_Collection_and_Cleaning.ipynb)
2. [02 历史实验与结果](02_Historical_Experiments_and_Results.ipynb)
3. [03 V1–V5、E0–E6 与最终决定](03_Model_Versions_and_Final_Evaluation.ipynb)

[Experiment table](experiments/New%20Experiments/E3/EXPERIMENT_RESULTS_ZH.md) · [Data manifest](data/manifests/approved_inputs.json) · [File list](FILES_TO_SHARE.md)

依次从 Notebook 01 至 03 查看采集清理、V1–V5、历史研究及 E1–E6。运行 bash setup.sh 安装依赖。当前可离线查看保存的结果，数据单元及模型复现需要另行取得获准输入。归档采集源码用于说明历史流程，运行前需适配路径和输入。

[V4 提交接口源码](03_Model_Versions_and_Final_Evaluation.ipynb#v4-submission-and-inference). 原 E6 仍需单独取得三文件提交包及匹配权重。

[组员审阅数据及字段说明](data/team_review/README.md)提供新采集训练、配对训练和配对开发三份精简 CSV。重跑原实验仍需对应的获准原始快照。

[实验环境记录](configs/experiment_environment.json)列出当时的 Python 和依赖版本。[部署依赖锁](configs/requirements-deploy.lock.txt)对应独立部署环境。Notebook 安装由根目录的 setup.sh 读取 requirements.lock.txt。

## 代码导航

| 想做什么 | 看哪个文件 | Notebook 03 章节 |
|---|---|---|
| 准备 V1–V5 的新训练 | [train_historical.py](scripts/train_historical.py), [models_v1_v5.py](src/dragonthree/models_v1_v5.py) | [Historical training entry point](03_Model_Versions_and_Final_Evaluation.ipynb#historical-training-entry-point) |
| 理解 CPU 预算和工作进程保护 | [run_budgeted.py](scripts/run_budgeted.py), [worker_bootstrap.py](scripts/worker_bootstrap.py) | [Historical E1 launch procedure](03_Model_Versions_and_Final_Evaluation.ipynb#historical-e1-launch-procedure) |
| 检查数据清单、URL 处理和结果读取 | [project_io.py](src/dragonthree/project_io.py) | [1. Review setup and input boundaries](03_Model_Versions_and_Final_Evaluation.ipynb#1-review-setup-and-input-boundaries) |
| 核对 E0 数据审计与 V4 基线 | [audit_and_baseline.py](src/dragonthree/audit_and_baseline.py) | [Audit approved inputs and establish the V4 baseline](03_Model_Versions_and_Final_Evaluation.ipynb#audit-approved-inputs-and-establish-the-v4-baseline) |
| 查看 E1 嵌套验证和折内规则 | [e1_validation.py](src/dragonthree/e1_validation.py), [v3_validation.py](src/dragonthree/v3_validation.py) | [Continue nested group-disjoint validation](03_Model_Versions_and_Final_Evaluation.ipynb#continue-nested-group-disjoint-validation) |
| 查看 E2 锁定比较及导出模板 | [e2_locked_eval.py](src/dragonthree/e2_locked_eval.py), [lexical_model.py](src/dragonthree/lexical_model.py) | [Compare the locked course-only and augmented candidates](03_Model_Versions_and_Final_Evaluation.ipynb#compare-the-locked-course-only-and-augmented-candidates) |
| 理解 E3 误差分析和区间 | [analyze_results.py](src/dragonthree/analyze_results.py) | [Summarize errors, confidence intervals and slices](03_Model_Versions_and_Final_Evaluation.ipynb#summarize-errors-confidence-intervals-and-slices) |
| 核对 E5–E6 的冻结与部署检查 | [v4_release_checks.py](src/dragonthree/v4_release_checks.py) | [Record the frozen V4 and verify its deployment interface](03_Model_Versions_and_Final_Evaluation.ipynb#record-the-frozen-v4-and-verify-its-deployment-interface) |

[清理前输入与本地清理流程](data/before_cleaning/README.md)说明如何在 Notebook 01 重跑课程清理，并在 data/generated/course_cleaning/ 查看生成的 CSV。

[自采数据离线复现](01_Data_Collection_and_Cleaning.ipynb#offline-replay--self-collected-training-and-development-tables)可生成三份精简表，结果写入 data/generated/self_collected_replay/。

[自采训练数据离线重建](01_Data_Collection_and_Cleaning.ipynb#offline-rebuild--supplemental-new-training-from-verified-candidates)从待核验清单、缓存、去重参照和审核决定重新生成 new_train.csv。

[组员复现指南](data/TEAM_REPRODUCTION.md)列出 Notebook 01 每条流程的输入、取得方式和输出位置。
