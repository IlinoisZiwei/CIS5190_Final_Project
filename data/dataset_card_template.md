---
language:
- en
task_categories:
- text-classification
tags:
- news-source-classification
- educational-research
- url-slugs
pretty_name: DragonThree CIS5190 Approved Research Data
configs:
- config_name: course_clean
  default: true
  data_files:
  - split: train
    path: data/course_clean.csv
  - split: validation
    path: data/course_dev.csv
- config_name: course_original
  data_files:
  - split: train
    path: data/course_original_train.csv
  - split: validation
    path: data/course_dev.csv
- config_name: legacy_external
  data_files:
  - split: internal_validation
    path: data/legacy_train.csv
  - split: reused_development
    path: data/legacy_dev.csv
- config_name: paired_external
  data_files:
  - split: internal_validation
    path: data/paired_train.csv
  - split: reused_development
    path: data/paired_dev.csv
- config_name: selected_new_original_annotations
  data_files:
  - split: training_pool
    path: data/new_metadata.jsonl
- config_name: audited_pool_ledger
  data_files:
  - split: approved_pool
    path: data/audited_training.jsonl
- config_name: historical_training_annotations
  data_files:
  - split: historical_training_pool
    path: data/training_metadata.csv
- config_name: course_combined_audit_only
  data_files:
  - split: audit_only
    path: data/course_all.csv
- config_name: course_exclusions_audit_only
  data_files:
  - split: excluded
    path: data/course_exclusions.csv
- config_name: official_provenance_audit_only
  data_files:
  - split: provenance_only
    path: data/official_provenance.csv
---

# DragonThree approved research data

Packaging status: @BUILD_STATUS@

This package preserves 13 approved source files for the UPenn CIS5190 Option B project: predicting FoxNews or NBC from cleaned news URL slug text. It contains original fields, annotations, and recorded provenance. It is a local research package; building it does not upload it to Hugging Face or GitHub. No sealed self-built test, teacher hidden test, mixed external metadata, or unapproved collection pool is included.

The full local project maintains the dataset packaging tool. It verifies frozen rows, columns and byte digests, then copies approved snapshots without changing their order or fields. The generated package includes a build manifest and source index. For the reduced tables supplied in the Desktop share, see the [team review data guide](team_review/README.md).

## Data roles and overlap

| File | Rows | Role in the latest E1/E2 study |
|---|---:|---|
| course_original_train.csv | 3,849 | Historical original course training; not the latest cleaned fitting table |
| course_clean.csv | 3,730 | Course fitting pool |
| course_dev.csv | 951 | Reused course development; never fitting in E1/E2 |
| course_all.csv | 4,800 | Original training plus development; audit only |
| course_exclusions.csv | 119 | Excluded original training records and reasons; audit only |
| legacy_train.csv | 1,566 | Historical external training pool; internal validation only in the latest study |
| legacy_dev.csv | 306 | Reused external development |
| paired_train.csv | 800 | Historical event-paired training pool; internal validation only in the latest study |
| paired_dev.csv | 200 | Reused event-paired development |
| new_metadata.jsonl | 2,000 | Selected new pool with original annotations; consult audited ledger for latest approved annotations |
| audited_training.jsonl | 8,096 | Combined pool ledger: course 3,730, new 2,000, legacy 1,566, paired 800; not all fitting rows |
| training_metadata.csv | 5,296 | Historical course-plus-legacy training annotations; not the latest fit instruction |
| official_provenance.csv | 3,000 | Sources for new 2,000 and paired 1,000; includes paired development; provenance only |

Several files are overlapping representations of the same records. Do not concatenate all configurations or interpret their row counts as distinct examples. In the latest study only the course and new origins may enter fitting. The old-panel origins (legacy and paired) support internal selection/validation. Development and audit-only configurations must remain outside fitting and feature discovery.

The Hugging Face configuration names and explicit relative file mappings preserve these roles rather than relying on filename auto-detection. `course_clean` is the default configuration. Nonstandard split names such as `internal_validation` and `provenance_only` are intentional. The format follows [Hugging Face's data-file configuration documentation](https://huggingface.co/docs/hub/datasets-data-files-configuration).

## Model inputs and labels

Model text uses the last nonempty URL path component, percent-decoded, with a trailing HTML/PHP-style extension removed. Hyphens and underscores separate tokens; tokens containing digits are removed. Hostnames, other path components, article bodies, and preserved page headlines are not model features. Publisher labels are FoxNews and NBC; model class IDs are FoxNews=0 and NBC=1.

The metadata retains information such as headline, topic, date/month, page form, group identifiers, collection batch, verification details, and original source references when present in the corresponding source file. Retaining a field does not authorize using it as a predictor. The exact field names and schemas remain in `build_manifest.json`; different configurations are not forced into a shared schema.

## Provenance and annotation status

Course URLs came from the instructor-provided project data. The legacy, paired, and new pools were assembled in earlier project work from publisher URLs and stored with the historical project's verification records. `official_provenance.csv` preserves available discovery/canonical URLs, response status, collection timestamps, and response digests for the new and paired pools. A digest column is historical evidence; building this release does not fetch or re-hash remote pages.

Original annotations and later audited annotations are both retained where they exist. They must not be silently treated as interchangeable. E0's saved audit found no URL-derived label/preprocessing mismatches and no exact cleaned-text overlap with the approved development panels. This does not prove the absence of all semantic near-duplicates or shared real-world events. Existing group assignments are preserved, and the prepared review queue does not establish an independent human review was completed.

## Evaluation boundaries and limitations

E1 uses three repeated, three-fold grouped outer partitions with fold-local inner selection. E2 fits two specifications that were locked before the new development evaluation. Development panels have prior reuse and are not fresh independent test sets. Internal results and reused-development results must be labeled separately. The group-bootstrap intervals are conditional on reused training records and saved partitions.

The selected pool and publisher-dependent URL formats may introduce topic, time, length, and template differences. Accuracy is evidence about these project samples, not a general measure of political bias or journalistic quality. No final-test or leaderboard score is supplied by this data package.

## Distribution and maintenance

No new license is asserted for instructor materials, publisher content, or historical annotations. Their redistribution terms are not established by this package. This card is ready for a local review of a potential dataset release; publication is a separate action and was not performed by the builder.

The builder requires only Python's standard library, reads an explicit allowlist, checks the frozen E0 hashes, refuses an existing output directory, and does not train, collect, install, download, or publish anything. For this approved run, invoke it through the project's CPU-budget supervisor; installing dependencies does not invoke it automatically.
