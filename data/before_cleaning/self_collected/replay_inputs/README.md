# Local replay inputs for Notebook 01

- eligible_pairs.jsonl: 879 verified eligible event candidates, in the historical order.
- pair_review_decisions.json: recorded accept/reject decisions and topic overrides used for the final event split.
- new_metadata.jsonl: 2,000 verified, selected supplemental training records, before projection to the team-review CSV fields.

The [offline replay section](../../../../01_Data_Collection_and_Cleaning.ipynb#offline-replay--self-collected-training-and-development-tables) describes the two starting points. The raw discovery snapshots one directory above document earlier work. These three local inputs contain additional fields and are excluded from the Git share list.
