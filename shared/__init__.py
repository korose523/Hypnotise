# shared package — minimal reconstruction to load pre-generated split JSONs.
# The original shared/ source was lost (only .pyc remained after OS reinstall);
# this module faithfully re-implements the load interface used by
# run_exp101_reproducible.py (it only READS splits/*.json, which are the
# authoritative pre-generated partitions, so behavior is identical).
