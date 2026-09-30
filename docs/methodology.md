# Methodology

## Prediction tasks

The reference paper defines two separate binary classifiers. A rescue can be easy, hard, or undetermined according to domain experts. The archived scripts expect original database labels of `-1` for easy and `1` for hard, with other records treated as negative for each task.

| Original category | Easy classifier target | Hard classifier target |
| --- | --- | --- |
| Easy | 1 | 0 |
| Hard | 0 | 1 |
| Undetermined | 0 | 0 |

The public demo uses an explicit synthetic label `0` for the third category and rejects missing or unexpected labels. Its mapping uses the original series directly, avoiding sequential in-place replacements.

## Historical data preparation

`notebooks/build_dataset.ipynb` reads operational PostgreSQL tables and joins rescue, route, donation, participant, address, household, and weather information. The supplied notebook contains exploratory code, intermediate CSV dependencies, and repeated feature-building cells. It is retained as a historical record.

Feature families in the source include:

| Family | Examples |
| --- | --- |
| Weather | Temperature, precipitation, visibility, humidity, wind speed |
| Geography | Donor, recipient and volunteer coordinates; volunteer-to-donor and volunteer-to-recipient distances |
| Donation | Quantity, food categories, dietary restrictions, household information |
| History | Participant experience, previous rescue counts, previous ratings |
| Timing | Publication year, month, day and hour |
| Other operational information | Phone-presence indicators and recurrence information in exploratory code |

Only information available at the time of prediction belongs in the final feature set. Historical aggregates should exclude the current rescue and all future events. The notebook's completion-time and distance/time experiments are exploratory proxies and must not be used as pre-rescue features.

## Language model and pseudo-labels

The published hybrid method first trains a language model on volunteer comments and their four-level ratings, then adapts it to expert difficulty labels. It uses the difficulty model to assign soft labels to additional historical comments. Expert labels take precedence where available, and a tabular model learns from the augmented training data.

Baseline 2 skips rating training. Baseline 1 uses only expert-labeled tabular records. The paper evaluates six tabular predictor families and selects LightGBM using validation performance. The capstone archive also includes classifier-based variants, balancing, and changes to training and splitting, so it is not an exact implementation of every paper setting.

The supplied hybrid script tokenizes with `bert-base-cased` but initializes its active text-training function with `distilbert-base-cased`. That differs from a simple claim that one BERT configuration was used throughout. The active training code also limits the number of text training and evaluation examples.

## Evaluation design

For a valid real-data experiment:

1. Reserve a fixed expert-labeled test set before training the language or tabular models.
2. Exclude those rescue IDs from rating training, difficulty training, pseudo-label generation used for fitting, and model selection.
3. Define and retain a separate training/validation split. For a temporal experiment, use only earlier records to predict later rescues.
4. Fit imputers, scalers, oversampling, and feature selection only on training data.
5. Choose hyperparameters and thresholds using validation data.
6. Evaluate once on the held-out expert labels and report class prevalence, ROC-AUC, PR-AUC, F1, and accuracy with a clear positive class.

The paper's headline results use mean and standard deviation over 10 trials. Historical source comments are not enough to establish that the supplied archive followed the same protocol. See [reproduction](reproduction.md).

## Public offline demonstration

The demo generates 1,000 fictional rows by default with six artificial pre-rescue features. An invented noisy scoring rule produces three difficulty categories. A stratified split allocates 60% to training, 20% to validation, and 20% to testing.

For each binary task it fits Logistic Regression and Random Forest pipelines. Median imputation and standardization fit only on training rows. It selects an F1 threshold on validation data and leaves the fitted pipeline unchanged when testing. It writes the seed, dependency versions, split sizes, metrics, and fictional rescue-ID split membership.

The demo does not implement text training, pseudo-labeling, weather retrieval, a real database export, or production inference.

## Source

Shi et al., *Predicting and Presenting Task Difficulty for Crowdsourcing Food Rescue Platforms*, WWW 2024, Sections 3 and 4. [DOI](https://doi.org/10.1145/3589334.3648155). The capstone report and archived source provide the implementation context.
