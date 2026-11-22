# Local supplemental rebuild inputs

Notebook 01 replays the original assembly from three historical ordered URL waitlists, 3,252 cached verification records, 21 topic overrides and the saved related-event decisions. The deduplication CSV is a URL-only projection of six approved reference snapshots. Saved exclusions and grouping edges provide intermediate checkpoints.

The [Notebook section](../../../../01_Data_Collection_and_Cleaning.ipynb#offline-rebuild--supplemental-new-training-from-verified-candidates) describes and runs the checks. The expected full selected snapshot is in the sibling replay_inputs folder. The verified CSV is saved separately in data/generated/new_training_rebuild/. These inputs stay local.
