# Course inputs before cleaning

These unchanged course-provided URL snapshots support the local Notebook 01 demonstration.

| Input | Rows | Role |
|---|---:|---|
| course_original_train.csv | 3,849 | Raw training records to clean |
| course_dev.csv | 951 | Separate course development reference for the historical near-duplicate filter |

Run the “Start here” section of [Notebook 01](../../01_Data_Collection_and_Cleaning.ipynb). It generates course_clean.csv, course_exclusions.csv and cleaning_summary.csv under data/generated/course_cleaning/. The original inputs stay in this folder. Keep course-provided snapshots and row-level derivatives out of Git unless their distribution is approved through the course.

The [self-collected candidate snapshots](self_collected/README.md) are a separate local branch of the project data lineage.
