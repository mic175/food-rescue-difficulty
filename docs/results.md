# Results and limitations

Three different sources of metrics appear in this repository: the published research paper, the 2025 capstone report, and the newly added synthetic demo. They use different data or protocols and must be kept distinct.

## Published research

Table 2 of the WWW 2024 paper, PDF p. 4 / proceedings p. 4689, reports:

| Algorithm | Easy mean ROC-AUC | Easy SD | Hard mean ROC-AUC | Hard SD |
| --- | --- | --- | --- | --- |
| Hybrid algorithm | 0.710 | 0.023 | 0.685 | 0.041 |
| Baseline 1 | 0.543 | 0.024 | 0.495 | 0.025 |
| Baseline 2 | 0.709 | 0.037 | 0.563 | 0.000 |

These are the original researchers' reported values over 10 random seeds. The public repository has not reproduced them.

## Capstone observations

The report's LightGBM entries are:

| Approach | Task | Accuracy | F1 | ROC-AUC | PR-AUC | PDF page |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline 1 | Easy | 0.914 | 0.066 | 0.692 | 0.142 | 7 |
| Baseline 1 | Hard | 0.909 | 0.043 | 0.405 | 0.031 | 7 |
| Baseline 2, raw | Easy | 0.638 | 0.231 | 0.703 | 0.140 | 8 |
| Baseline 2, raw | Hard | 0.048 | 0.074 | 0.356 | 0.028 | 8 |
| Final algorithm, raw | Easy | 0.638 | 0.231 | 0.703 | 0.140 | 9 |
| Final algorithm, raw | Hard | 0.048 | 0.074 | 0.356 | 0.028 | 9 |

The report displays zero standard deviation for these entries. The supplied files do not include run-level logs to verify the repeated trials. For Baseline 1 hard rescues, the SVM row reports ROC-AUC 0.641, above the LightGBM value 0.405. The report's prose about LightGBM being highest should therefore not be applied to all tasks.

## Source disagreements

- The presentation includes the paper's 0.710 / 0.685 headline results, while the capstone report's local tables differ. The headline values are attributed to the publication in this repository.
- The Baseline 2 and final-algorithm tables in the report are identical across all models. They could reflect shared runs or duplicated reporting, but the supplied evidence cannot determine the cause.
- The report describes later tuning improvements without supplying a final verified numeric result. This repository does not invent an improved score.
- Presentation slide 10 lists completed AWS integration, while report Section 4.4 calls it future work. No deployment code or AWS infrastructure files were supplied. The repository records AWS integration as unverified future work.
- The active hybrid script's random split does not enforce its queried expert test IDs. The local results cannot be assumed to use the paper's expert-held-out evaluation.

## Interpreting the metrics

Accuracy can hide poor minority-class performance. The Baseline 1 report entries have high accuracy and very low F1, so accuracy alone does not support a strong difficulty predictor. ROC-AUC below 0.5 is a reason to investigate the positive-class definition, label mapping, score direction, data alignment, and evaluation protocol. It is not proof of a specific bug.

PR-AUC depends strongly on class prevalence. Compare runs only when the target task and evaluation set are comparable. The archived scripts and demo calculate trapezoidal area under the precision-recall curve, which differs from average precision.

## Demo results

`examples/synthetic-demo-metrics.json` records an actual offline demo run using fictional data. It verifies that the public example can produce both classifiers' metrics. It does not establish utility for Food Rescue Hero or improvement over a real-data baseline.

## Project scope

The supplied capstone focuses on data preparation, debugging, model comparison, and analysis. The original paper also includes stakeholder studies and explanation methods, but implementations of those studies, LIME/LLM explanations, and a production integration are not present in this archive.
