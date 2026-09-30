# Historical research scripts

These scripts are copied unchanged from the supplied capstone code archive. Comments describing a script as a paper baseline do not establish that its current implementation reproduces the paper.

| File | Historical purpose |
| --- | --- |
| `baseline1.py` | Expert-labeled tabular model comparison |
| `baseline2.py` | Difficulty-trained BERT pseudo-labeling and tabular model comparison |
| `bert-rating-label-geo.py` | Rating training, difficulty fine-tuning, pseudo-labeling, tabular experiments, and temporal analysis |
| `database.yaml.example` | Placeholder PostgreSQL configuration |
| `requirements-original.txt` | Original machine's dependency snapshot; contains nonportable system packages |

Start with [reproduction and known issues](../docs/reproduction.md) before executing these files. They require approved private inputs. The public runnable entry point is [the synthetic demo](../scripts/run_demo.py).
