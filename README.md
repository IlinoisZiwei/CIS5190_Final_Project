# DragonThree · CIS 5190 Option B

[中文](README.zh-CN.md) · [Overleaf report](https://www.overleaf.com/project/6aea3dd3f59e426e6b892e27)

Classify FoxNews or NBC using cleaned text from the final URL segment. V4 was retained.

1. [01 Data collection, cleaning and analysis](01_Data_Collection_and_Cleaning.ipynb)
2. [02 Historical experiments and evidence](02_Historical_Experiments_and_Results.ipynb)
3. [03 V1–V5, E0–E6 and final decision](03_Model_Versions_and_Final_Evaluation.ipynb)

[Experiment table](experiments/New%20Experiments/E3/EXPERIMENT_RESULTS_EN.md) · [Data manifest](data/manifests/approved_inputs.json) · [File list](FILES_TO_SHARE.md)

Read notebooks 01 to 03 in order. They cover collection and cleaning, V1–V5, historical follow-up work, and E1–E6. Run bash setup.sh to install dependencies. Saved results can be reviewed offline. Data-dependent cells and model reproduction require approved inputs obtained separately. Historical collection source is provided for review and requires path and input adaptation before execution.

[V4 submission interface source](03_Model_Versions_and_Final_Evaluation.ipynb#v4-submission-and-inference). The original E6 benchmark requires the separate three-file submission package and matching weights.

[Team review data and field guide](data/team_review/README.md) provides reduced new-training, paired-training and paired-development CSVs. The original experiment loaders require their original approved inputs.

[Recorded experiment environment](configs/experiment_environment.json) documents the original Python and package environment. The [deployment lock](configs/requirements-deploy.lock.txt) records the separate deployment dependencies. Use the root requirements.lock.txt with setup.sh for the notebook environment.

## Code navigation

| What you want to do | Files to read | Notebook 03 section |
|---|---|---|
| Review the V1–V5 training entry point | [train_historical.py](scripts/train_historical.py), [models_v1_v5.py](src/dragonthree/models_v1_v5.py) | [Historical training entry point](03_Model_Versions_and_Final_Evaluation.ipynb#historical-training-entry-point) |
| Understand CPU supervision and worker safeguards | [run_budgeted.py](scripts/run_budgeted.py), [worker_bootstrap.py](scripts/worker_bootstrap.py) | [Historical E1 launch procedure](03_Model_Versions_and_Final_Evaluation.ipynb#historical-e1-launch-procedure) |
| Inspect input manifests, URL handling and result loading | [project_io.py](src/dragonthree/project_io.py) | [1. Review setup and input boundaries](03_Model_Versions_and_Final_Evaluation.ipynb#1-review-setup-and-input-boundaries) |
| Review the E0 data audit and V4 baseline | [audit_and_baseline.py](src/dragonthree/audit_and_baseline.py) | [Audit approved inputs and establish the V4 baseline](03_Model_Versions_and_Final_Evaluation.ipynb#audit-approved-inputs-and-establish-the-v4-baseline) |
| Review E1 nested validation and fold rules | [e1_validation.py](src/dragonthree/e1_validation.py), [v3_validation.py](src/dragonthree/v3_validation.py) | [Continue nested group-disjoint validation](03_Model_Versions_and_Final_Evaluation.ipynb#continue-nested-group-disjoint-validation) |
| Review the E2 locked comparison and export template | [e2_locked_eval.py](src/dragonthree/e2_locked_eval.py), [lexical_model.py](src/dragonthree/lexical_model.py) | [Compare the locked course-only and augmented candidates](03_Model_Versions_and_Final_Evaluation.ipynb#compare-the-locked-course-only-and-augmented-candidates) |
| Understand E3 error analysis and intervals | [analyze_results.py](src/dragonthree/analyze_results.py) | [Summarize errors, confidence intervals and slices](03_Model_Versions_and_Final_Evaluation.ipynb#summarize-errors-confidence-intervals-and-slices) |
| Review E5–E6 release and deployment checks | [v4_release_checks.py](src/dragonthree/v4_release_checks.py) | [Record the frozen V4 and verify its deployment interface](03_Model_Versions_and_Final_Evaluation.ipynb#record-the-frozen-v4-and-verify-its-deployment-interface) |

[Before-cleaning course inputs and local pipeline](data/before_cleaning/README.md) explain how Notebook 01 rebuilds separate output CSVs under data/generated/course_cleaning/.

[Offline replay of the self-collected data](01_Data_Collection_and_Cleaning.ipynb#offline-replay--self-collected-training-and-development-tables) generates the three reduced tables under data/generated/self_collected_replay/.

[Supplemental candidate-to-training rebuild](01_Data_Collection_and_Cleaning.ipynb#offline-rebuild--supplemental-new-training-from-verified-candidates) replays the saved verification queue and review decisions locally.

[Team reproduction guide](data/TEAM_REPRODUCTION.md) lists the inputs for each Notebook 01 branch and which files teammates obtain separately.
