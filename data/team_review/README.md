# DragonThree team review data

These UTF-8 CSV files are reduced views of the team-collected data. They keep the row order and split of each approved snapshot. Existing experiment loaders still expect the original snapshots under data/approved. These review tables are not substitutes for those inputs.

| File | Rows | Role |
|---|---:|---|
| [new_train.csv](new_train.csv) | 2,000 | New training pool |
| [paired_train.csv](paired_train.csv) | 800 | Event-paired training |
| [paired_dev.csv](paired_dev.csv) | 200 | Event-paired development |

## New training pool fields

| Field | Meaning |
|---|---|
| record_id | Record identifier in the new pool |
| source_url | Source news URL |
| publisher_label | Publisher label as recorded |
| model_text | Cleaned URL title text used as model input |
| group_id | Group identifier used in the new-data workflow |
| published_month | Publication month as recorded |
| batch_id | Collection batch identifier |

## Event-paired fields

| Field | Meaning |
|---|---|
| event_group | Shared news-event identifier |
| publisher | Recorded publisher |
| split | Original split label (paired_train or paired_confirmation) |
| url | News URL associated with the paired record |
| text | Cleaned text for modeling |

The event-group identifiers are disjoint between the paired training and development tables.
