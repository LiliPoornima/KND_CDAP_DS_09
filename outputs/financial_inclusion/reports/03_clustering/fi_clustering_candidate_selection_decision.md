# Final Clustering Model Selection — Component 02

## Final decision
Selected candidate: 5D_k5
Algorithm: KMeans
Scaling: Raw
Number of clusters: 5
Final model selected: Yes

## Research objective
Identify interpretable digital financial inclusion profiles using five
dimensions: connectivity, financial access, digital financial usage,
digital safety, and financial resilience.

## Selection rationale
The 5D_k5 candidate was selected after reviewing clustering quality,
stability, algorithm agreement, cluster sizes, and profile interpretability.

- Respondents: 56,503
- Economies: 75
- Silhouette score: 0.6313
- Davies–Bouldin score: 0.8046
- Calinski–Harabasz score: 78662.38
- Smallest cluster: 5,036 respondents (8.91%)
- Largest cluster: 23,574 respondents (41.72%)
- Stability: mean and minimum pairwise ARI were both 1.0 across 10 seed runs.
- KMeans–GMM agreement: ARI = 1.0.

## Comparison with alternatives
5D_k4 was simpler and had a more balanced minimum cluster size, but its
silhouette score was lower and it did not isolate the access-without-usage
profile as a separate cluster.

5D_k6 achieved stronger internal clustering metrics, but produced two
smaller clusters (4.17% and 4.74%) and had lower KMeans–GMM agreement
(ARI = 0.8855). Its additional complexity was not preferred for the
current research objective.

The selected model does not have the best value on every metric. The
decision prioritises interpretability and research relevance alongside
stability and internal clustering quality.

## Key profile of interest
Cluster 4 contains 5,036 respondents (8.91%) with financial access but
without digital financial usage according to the observed cluster profile.
This is a descriptive profile, not evidence of causality.

## Limitations
- Five-dimensional complete-case eligibility limits the analysis to
  56,503 respondents across 75 economies.
- Digital safety is represented by the available phone-protection measure
  and is not a comprehensive measure of digital safety.
- Economy representation across clusters does not establish equal
  representation or equal generalisability across economies.
- Cluster profiles are descriptive and do not establish causal relationships.
- The 3D and 5D candidates use different eligible populations and should
  not be compared as if they represent the same population.

## Completed downstream validation

- Updated the supervised-learning dataset and target audits to use the selected `5D_k5` cluster labels.
- Verified that the supervised dataset contains 56,503 respondents with no missing target labels or duplicate respondent IDs.
- Confirmed the target distribution across five clusters: cluster 0 = 10,107; cluster 1 = 23,574; cluster 2 = 9,005; cluster 3 = 8,781; cluster 4 = 5,036.
- Evaluated Majority Baseline, Logistic Regression, Random Forest, and XGBoost using a stratified train/test split.
- Logistic Regression achieved the highest balanced accuracy (0.3783) and macro-F1 (0.3624), while XGBoost achieved the highest accuracy (0.4729).
- These supervised results measure how well the selected cluster labels can be predicted from the available demographic and socioeconomic predictors; they do not independently establish clustering quality.

## Supervised-learning evaluation findings

Three classifiers were evaluated against a majority-class baseline using a stratified 80/20 train-test split and demographic and socioeconomic predictors.

- **Logistic Regression** achieved the highest balanced accuracy (0.3783) and macro-F1 (0.3624), indicating comparatively better performance across the five cluster classes.
- **XGBoost** achieved the highest accuracy (0.4729) and weighted F1 (0.4243), but its performance was uneven across clusters.
- **Random Forest** achieved an accuracy of 0.3544, balanced accuracy of 0.3080, and macro-F1 of 0.3031.
- The majority baseline achieved an accuracy of 0.4172, balanced accuracy of 0.2000, and macro-F1 of 0.1178.

Cluster-level evaluation showed that Cluster 1 was most successfully identified by XGBoost (recall = 0.83; F1 = 0.65). In contrast, XGBoost recall was low for Cluster 0 (0.15) and Cluster 4 (0.12). Logistic Regression performed more evenly and achieved recall values of 0.27 for Cluster 0 and 0.37 for Cluster 4.

These results suggest that the available demographic and socioeconomic predictors do not distinguish all cluster profiles equally well. Supervised classification evaluates the predictability of the derived cluster labels; it is not independent evidence of clustering validity or causal relationships.

## Remaining actions

- Complete the final review of the five cluster profile names and descriptions.
- Verify consistency among the selected-model decision, cluster profile reports, label metadata, and supervised-learning results.
- Document the complete-case eligibility and economy-coverage limitations in the final Component 02 report.
- Prepare the final Component 02 results and reproducibility checklist.
