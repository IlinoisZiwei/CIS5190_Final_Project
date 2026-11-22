# Team reproduction guide

Start at the project root. Use Python 3.12, run bash setup.sh, and select the project environment in [Notebook 01](../01_Data_Collection_and_Cleaning.ipynb). Its input check reports which branches can run. Open the existing notebook outputs and summaries for a read-only review.

| Branch | Inputs | Generated output |
|---|---|---|
| Course cleaning | Obtain the original 3,849 training and 951 development URLs through the course channel. Place the unchanged files at data/before_cleaning/course_original_train.csv and data/before_cleaning/course_dev.csv. | Three CSVs under data/generated/course_cleaning/ |
| Event-pair replay and selected new training export | Place the team's replay companion inputs under data/before_cleaning/self_collected/replay_inputs/. The comparison CSVs are included in data/team_review/. | new_train.csv, paired_train.csv and paired_dev.csv under data/generated/self_collected_replay/ |
| Supplemental assembly from cached verification | Place the supplied rebuild artifacts under data/before_cleaning/self_collected/new_training_rebuild/. Separately obtain the approved dedup_reference_urls.csv from the full local project and put it in that folder. The replay_inputs/new_metadata.jsonl snapshot is the final comparison. | new_train.csv under data/generated/new_training_rebuild/ |

The local companion archive contains the two self-collected input groups and preserves their repository-relative paths. Where attached separately, extract it from the project root. The public-facing share list includes these instructions and the three reduced CSVs. The local companion requires a separate team distribution decision.

The deduplication projection contains URLs from six approved references: course original training, course development, the course/legacy training ledger, legacy development, paired training and paired development. It stays separate because it includes course and historical data. Use the same historical reference to reproduce the selection exactly.

[Notebook 02](../02_Historical_Experiments_and_Results.ipynb) and [Notebook 03](../03_Model_Versions_and_Final_Evaluation.ipynb) document saved results. Full model reproduction depends on separately authorized snapshots, historical caches, the E2 frozen input manifest and matching checkpoint artifacts.
