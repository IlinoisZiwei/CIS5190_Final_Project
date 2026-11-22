# Self-collected candidates before final cleaning

These local JSONL snapshots record candidate news URLs collected by the team. Candidate files may overlap and each historical discovery step may already have performed URL consolidation. Verification, selection, event matching and final dataset splits happened later.

| File | Candidate rows | Role |
|---|---:|---|
| `supplemental/discovery_records.jsonl` | 143,556 | Main supplemental sitemap discovery, after URL consolidation. |
| `supplemental/special_discovery_records.jsonl` | 4,676 | Supplemental discovery for special page formats. |
| `supplemental/extra_discovery_records.jsonl` | 1,564 | Additional discovery candidates before final verification and selection. |
| `event_paired/discovered_metadata.jsonl` | 30,717 | Publisher URL candidates before event proposal, review and splitting. |

[Team review CSVs](../../team_review/README.md) are derived from the later approved snapshots. These candidate files are not read by the Notebook 01 course-cleaning demonstration. Replaying the historical candidate-to-dataset workflow also requires verification caches and review decisions.
