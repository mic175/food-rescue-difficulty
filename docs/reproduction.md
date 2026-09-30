# Reproduction and original-code prerequisites

## Supported public command

From the repository root, with Python 3.12:

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py --output-dir outputs/demo
python -m unittest discover -s tests -v
```

This command runs offline after dependency installation. It uses fictional data and needs neither PostgreSQL nor a Hugging Face download.

## Historical research environment

The files in `archive/` preserve the supplied Python scripts byte for byte. They are research records with unresolved assumptions, not the supported demo entry point. `notebooks/build_dataset.ipynb` retains the supplied source cells while clearing outputs and execution metadata.

The original experiments need:

- Permission to access the original operational PostgreSQL database, including the expected `public` tables and `derivative.train_annot` / `derivative.test_annot` tables.
- A reviewed tabular feature export named `all_rescues_info-noorg.csv`, which was not supplied in the ZIP.
- A local `database.yaml` for the scripts, following `archive/database.yaml.example`. The notebook also supports `DATABASE_URL`. Credentials must stay outside Git.
- A separate research environment with PyTorch, Transformers, Datasets, Evaluate, Accelerate, LightGBM, imbalanced-learn, SQLAlchemy, psycopg2, PyYAML, SciPy, plotting libraries, and Jupyter.
- Network access for language-model weights and metric downloads, suitable compute, and private storage for checkpoints and generated CSVs.

The supplied dependency snapshot records versions including NumPy 1.24.4, pandas 2.0.3, scikit-learn 1.3.2, LightGBM 4.5.0, Transformers 4.46.3, and SQLAlchemy 2.0.37. It also contains Ubuntu desktop/system packages and does not list PyTorch or Accelerate. **Do not install `archive/requirements-original.txt` as a portable environment.** An exact working research environment cannot be recovered from that snapshot alone.

Run archived scripts from `archive/` only after reviewing them and providing approved private inputs. The feature notebook expects its inputs relative to the working directory. It is not a restart-and-run-all notebook as supplied.

## Issues to resolve before claiming reproduction

| Source | Observed issue | Required review |
| --- | --- | --- |
| `archive/baseline1.py`, `create_difficulty_csv` | For easy labels, the function first replaces `-1` with `1`, then replaces every non-`-1` value with `0`, including the newly assigned positives. | Map from an unchanged copy of the original labels and rebuild cached label CSVs. |
| `archive/baseline1.py`, `split_tabular` | The `raw` and `time` branches both use the same 2022-03-01 cutoff. | Define the intended experimental split explicitly. |
| `archive/bert-rating-label-geo.py`, `split_tabular` | It queries expert test IDs but the active branch performs a random stratified split of `data` and returns before the expert-ID logic. | Reserve the expert test set outside all training and pseudo-labeling stages. |
| Hybrid text training | Tokenizer is `bert-base-cased`; model initialization is `distilbert-base-cased`. Training/evaluation examples are truncated to 2,000 / 1,000. | Make the model/tokenizer configuration and data-volume decision explicit. |
| Hybrid training orchestration | The fine-tuned model is saved to a shared `ckpt-{split}-finetune` path while the main loop later requests a seed-specific `ckpt-{split}-seed{seed}-fast` path. | Align checkpoint writers and readers and isolate every trial's artifacts. |
| Archived text evaluation | Both text-training workflows use the supplied difficulty `test` data for language-model evaluation. | Introduce a separate validation set and keep expert test labels outside all development decisions. |
| Cached files and historical aggregates | Cached CSVs, annotations, and history features depend on earlier workspace state. | Record input versions, rebuild features, validate joins, and exclude current/future outcomes from history. |

These observations come from code inspection. The full private-data training pipeline was not executed here. The original algorithms have not been silently rewritten or represented as validated.

## Verified in this repository

- Syntax parsing for all three archived Python scripts.
- Output removal from every public notebook code cell.
- Successful demo execution with separate training, validation, and test rows.
- Tests for both label mappings, input validation, duplicate-ID rejection, deterministic splitting, and split isolation.
- Public file selection excludes the original annotation CSVs and executed notebook.

To establish real reproducibility, supply an approved dataset version, verified dependency lock, explicit expert test-ID list, repaired research entry point, run-level logs, and aggregate results. Compare those results against the source papers only after the protocols match.
