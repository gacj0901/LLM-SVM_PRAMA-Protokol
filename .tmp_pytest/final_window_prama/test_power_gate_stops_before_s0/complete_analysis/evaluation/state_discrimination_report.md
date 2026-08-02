# E-P1 Final-State Discrimination Evaluation

The preregistered latent-occupancy channel is evaluated as final-state discrimination, not as token-localized early warning.

## Null Hypothesis

The preregistered latent-occupancy channel adds no discriminative signal beyond surprisal-derived baselines.

## Summary

- target_fpr: `0.1`
- primary_score: `latent_occupancy`
- permutations: `1000` (seed `0`)
- n_examples: `115`
- split_aware: `True`
- final_outcome_proxy_count: `115`

## Split: test

- status: `honest_null`
- primary_confirmatory: `{'verdict': 'honest_null', 'primary_auroc': 0.6236842105263158, 'best_baseline_auroc': 0.5263157894736842, 'auroc_delta': 0.09736842105263166, 'primary_test_tpr': 0.0, 'best_baseline_test_tpr': 0.2, 'tpr_delta': -0.2, 'permutation_p': 0.12287712287712288, 'rule': 'permutation p < 0.01 and primary exceeds every baseline by AUROC and test TPR at train-calibrated target FPR'}`
- best_prama_by_auroc_exploratory: `('delta', 1.0)`
- best_baseline_by_auroc: `('negative_mean_top1_gap', 0.5263157894736842)`

### Certified-kernel channels

- latent_occupancy: AUROC=0.6236842105263158 threshold=inf confusion={'tp': 0, 'fp': 0, 'tn': 19, 'fn': 10}
- delta: AUROC=1.0 threshold=0.1506824099306323 confusion={'tp': 10, 'fp': 2, 'tn': 17, 'fn': 0}
- xi: AUROC=1.0 threshold=0.12612525240451927 confusion={'tp': 10, 'fp': 2, 'tn': 17, 'fn': 0}
- neg_M: AUROC=1.0 threshold=-1.8738747475954807 confusion={'tp': 10, 'fp': 2, 'tn': 17, 'fn': 0}

### Baselines

- mean_surprisal: AUROC=0.4 threshold=3.342129975239003 confusion={'tp': 0, 'fp': 1, 'tn': 18, 'fn': 10}
- mean_entropy: AUROC=0.5 threshold=inf confusion={'tp': 0, 'fp': 0, 'tn': 19, 'fn': 10}
- negative_mean_top1_gap: AUROC=0.5263157894736842 threshold=-0.7999999999999999 confusion={'tp': 0, 'fp': 0, 'tn': 19, 'fn': 10}
- markovian_surprisal_tau64: AUROC=0.49473684210526314 threshold=3.0610257211581025 confusion={'tp': 2, 'fp': 0, 'tn': 19, 'fn': 8}

## Methodological Note

Ground truth must come from finish_reason or external automatic verification, never from the Protokol channel itself.
The confirmatory verdict uses only latent_occupancy. Best-of-panel kernel channels are exploratory.
If event_token equals the final token, any reported lead time is a final-outcome proxy and not anticipation.
