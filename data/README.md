# Retained experimental inputs

This directory contains datasets intentionally retained as inputs. Exact membership and
SHA-256 values are recorded in `retained_manifest_v1.json`.

Retained data are not generated reports. They may be normalized or selected datasets,
but their inclusion makes no claim about experimental outcome. New local downloads and
scratch transformations belong in `data/_local/`, which is ignored by Git.

`cocc_clean_negation_confirmatory_v2.jsonl` is stored with Git LFS because it exceeds
GitHub's normal object-size limit. Run `git lfs pull` after cloning when the full dataset
is required.
