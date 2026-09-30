# Data

No real Food Rescue Hero operational records are distributed in this public repository. The original annotation CSVs each contain 500 rows and include rescue IDs, volunteer ratings, free-text comments, labels, and notes. They are excluded along with executed notebook outputs and generated data exports.

Access to the original PostgreSQL data and any redistribution must be arranged with the research lab and data provider. A placeholder configuration is available at `archive/database.yaml.example`; real credentials must stay outside Git.

## Original annotation schema

| Field | Meaning |
| --- | --- |
| `rescue_id` | Operational rescue identifier |
| `rating` | Volunteer rating |
| `text` | Volunteer comment |
| `label` | Annotation or transformed difficulty label |
| Notes column | Additional annotation notes; column names differ between exports |

The supplied `stateB` CSVs have binary `0.0` / `1.0` values. They must not be assumed to be the unchanged `-1` / `1` original expert labels expected by some scripts. The original label transformation should be verified with the data owner.

## Synthetic demo schema

`scripts/run_demo.py` generates fictional data in memory. Its IDs start with `synthetic_`. No real IDs, comments, addresses, phone numbers, names, or contact information are copied into it.

| Field | Synthetic meaning |
| --- | --- |
| `travel_km` | Invented travel distance |
| `quantity_units` | Arbitrary donation quantity units |
| `rain_mm` | Invented precipitation |
| `prior_donor_rescues` | Invented donor history count |
| `prior_recipient_rescues` | Invented recipient history count |
| `volunteer_experience_days` | Invented volunteer experience |
| `label` | `-1` easy, `0` other, or `1` hard, from an invented noisy rule |

Outputs go into the ignored `outputs/` directory. The checked-in example contains aggregate metrics only. The synthetic dataset does not represent the distribution, volume, or behavior of the real platform.
