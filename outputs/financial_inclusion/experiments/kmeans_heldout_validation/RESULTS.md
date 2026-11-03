# Component 02: K-Means held-out validation

## Objective and preserved baseline

This experiment assesses whether the existing 5D_k5 partition can be reproduced when a new K-Means estimator is fitted only on training respondents and predicts held-out respondents. The production K-Means artifact, saved labels, five dimensions, profile names, dashboard, and supervised models were not modified.

Inspection confirmed the exact feature order: `score_connectivity`, `score_financial_access`, `score_digital_financial_usage`, `score_digital_safety`, `score_financial_resilience`. The experiment reads these columns from the 42-column analytical dataset and verifies that all respondent IDs and all five scores match the clustering input file exactly. Scores are used as their existing raw [0, 1] values without scaling. The saved estimator uses `k=5`, `init="k-means++"`, `n_init=10`, `max_iter=300`, `tol=0.0001`, Lloyd's algorithm, and `random_state=42`. The cohort has 56,503 unique respondents in 75 economies; the saved labels cover the same cohort and the saved model exactly reproduces them.

The reported historical full-data silhouette was 0.6313 (from the existing evaluation procedure, which samples up to 5,000 rows for silhouette). The existing Davies–Bouldin score was 0.8046 and Calinski–Harabasz score 78,662.38. Existing repeated-seed ARI and K-Means/GMM ARI were both reported as 1.0. These are preserved baseline results.

## Method

For seeds 42, 123, 2025, 2026, and 2027, respondents were split 80:20 without stratification; clustering is unsupervised, so no target labels were used to construct the split. Each new K-Means model was fitted on the 45,202 training respondents only and predicted the 11,301 held-out respondents. The fixed K-Means configuration was preserved; the repeated split seed was also used as the estimator seed.

Test silhouette was calculated over all held-out respondents using their raw five-dimensional scores and the training-only model's predicted clusters. ARI compares those predictions to the existing full-data reference labels for the same test respondents. The reference solution was trained using the full cohort, including these test respondents, so ARI measures agreement with that reference partition. It is not ground-truth accuracy and the comparison is not fully independent of the test respondents.

New cluster IDs are arbitrary. For descriptive cluster-level tables, IDs were aligned to the saved model using a Hungarian assignment that minimizes Euclidean distance between the newly fitted training centroids and the existing reference centroids. This uses training-only centroids and reference centroids, not the test reference labels. ARI itself was calculated before alignment because it is permutation-invariant.

## Findings

| Seed | Test silhouette | ARI vs full-data reference |
|---:|---:|---:|
| 42 | 0.6326 | 1.0000 |
| 123 | 0.6301 | 1.0000 |
| 2025 | 0.6315 | 1.0000 |
| 2026 | 0.6323 | 1.0000 |
| 2027 | 0.6318 | 1.0000 |
| **Mean** | **0.6317** | **1.0000** |
| **SD** | **0.0010** | **0.0000** |
| **Min–max** | **0.6301–0.6326** | **1.0000–1.0000** |

The test silhouette is very consistent across splits and close to the historical full-data value. Each training-only partition matched the reference partition perfectly up to label permutation (ARI 1.0). Test cluster sizes were also identical to the reference labels in each split after alignment, as expected from the perfect partition agreement. Across the repeated test sets, mean proportions were 17.5% for cluster 0, 41.8% for cluster 1, 16.3% for cluster 2, 15.5% for cluster 3, and 8.9% for cluster 4. The full-data proportions were 17.9%, 41.7%, 15.9%, 15.5%, and 8.9%, respectively.

The five profile patterns were retained in the held-out assignments. Mean scores averaged over held-out samples and seeds were:

| Profile | Connectivity | Financial access | Digital usage | Digital safety | Resilience |
|---|---:|---:|---:|---:|---:|
| Digitally Active, Safety Gap | 0.763 | 1.000 | 1.000 | 0.000 | 0.535 |
| Connected and Digitally Included | 0.952 | 1.000 | 1.000 | 1.000 | 0.592 |
| Connected, Financially Excluded | 0.909 | 0.000 | 0.000 | 1.000 | 0.433 |
| Multi-Dimensional Inclusion Gap | 0.701 | 0.000 | 0.000 | 0.000 | 0.384 |
| Access Without Digital Usage | 0.830 | 1.000 | 0.000 | 0.531 | 0.443 |

The expected weaknesses are visible in the calculated values: cluster 0's safety score is 0; cluster 1 is high across all five dimensions; cluster 2 has zero financial-access and usage scores; cluster 3 has zero scores in access, usage, and safety alongside lower connectivity and resilience; and cluster 4 has access 1.0 with usage 0.0. These descriptive patterns agree with the saved profile names.

Mean distance to the nearest training-only centroid ranged from 0.236 (cluster 1) to 0.605 (cluster 4), averaged across seeds and clusters. The mean gap between the nearest and second-nearest centroid ranged from 0.437 (cluster 4) to 0.822 (cluster 1). A small gap indicates more ambiguous assignments; these raw Euclidean distances are descriptive and should not be interpreted as calibrated confidence probabilities.

## Interpretation and limits

The experiment provides strong evidence that the saved five-profile partition is reproducible from an 80% respondent sample for these data and this cohort. It does not establish that the profiles are ground truth, that the solution generalizes to new economies or future survey waves, or that the five dimensions are causal. The reference ARI is inherently a comparison to labels produced by a full-data model, not an independent validation target. A separate economy-held-out or temporal validation would be needed to assess transfer beyond the current respondent cohort.

## Files

- `validate_kmeans_holdout.py`: complete repeated hold-out experiment; writes only within this directory.
- `heldout_metrics_by_seed.csv`, `heldout_metric_summary.csv`: test silhouette and reference ARI.
- `cluster_distribution_comparison.csv`: full-data, test-reference, and held-out distributions by seed.
- `dimension_profile_comparison_by_seed.csv`, `dimension_profile_comparison_summary.csv`: means and medians by profile and dimension.
- `centroid_distance_summary.csv`: nearest-centroid distances and nearest-versus-second-nearest margins by seed and cluster.
- `cluster_distribution_comparison.png`, `dimension_profile_comparison.png`, `repeated_holdout_metrics.png`: visual summaries.
- `experiment_details.json`: feature order, estimator configuration, and comparison interpretation.

## Reproduction

From the repository root:

```powershell
python outputs/financial_inclusion/experiments/kmeans_heldout_validation/validate_kmeans_holdout.py
```

The script needs the existing processed clustering input, saved reference labels, validated model, and profile metadata. It does not write to any of those baseline paths.
