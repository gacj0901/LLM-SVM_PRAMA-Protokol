# E-P1 Authoritative Verdict

- verdict: `honest_null`
- terminal: `True`
- detail: At least one confirmatory D6 signal gate failed.

## Preregistered gates

- C3: `{'check': 'C3 degeneration', 'passed': True, 'detail': 'r_Δω = -0.092, r_deg = -0.037, separation = -0.0545 (s_min = 0.01), absolute threshold r_star = 0.5, branch = absolute', 'r_delta_omega': -0.09194754007715046, 'r_degenerate': -0.03741779308614284, 'separation': -0.05452974699100762, 's_min': 0.01, 'r_star': 0.5, 'branch': 'absolute'}`
- power: `{'passed': True, 'eligible_sessions': 115, 'eligible_positives': 40, 'eligible_negatives': 75, 'test_positives': 10, 'minimum': 1, 'total_collected_n': 120, 'extension_block': 100, 'maximum_total_n': 1000, 'next_total_n': None, 'extension_allowed': False, 'scores_computed_at_gate': False}`
- permutation: `{'passed': False, 'p': 0.12287712287712288, 'threshold': 0.01, 'permutations': 1000}`
- auroc_vs_all_four_baselines: `{'passed': True, 'by_baseline': {'mean_surprisal': 0.22368421052631582, 'mean_entropy': 0.12368421052631584, 'negative_mean_top1_gap': 0.09736842105263166, 'markovian_surprisal_tau64': 0.1289473684210527}}`
- tpr_at_fpr_vs_all_four_baselines: `{'passed': False, 'by_baseline': {'mean_surprisal': 0.0, 'mean_entropy': 0.0, 'negative_mean_top1_gap': 0.0, 'markovian_surprisal_tau64': -0.2}, 'target_fpr': 0.1, 'threshold_source': 'train'}`
