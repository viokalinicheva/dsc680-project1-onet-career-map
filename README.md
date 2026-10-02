# Beyond the Job Title: Mapping Data Science Skills and Career Pathways

Job titles do not always explain what a person actually does. I built this project to compare the O*NET profile for Data Scientists with other U.S. occupations using measured skills, knowledge, and work activities. The result is a reproducible career map that shows which roles are most similar, where meaningful differences appear, and how stable those conclusions remain when reasonable analytical choices change.

## Question I wanted to answer

Which occupations are most similar to Data Scientists, what separates the closest roles, and do the conclusions remain stable when the filtering, distance measure, or feature weighting changes?

## At a glance

| Item | Verified result |
|---|---|
| Primary analysis | 882 occupations × 109 standardized features |
| Sensitivity analysis | 910 occupations × 109 standardized features |
| Closest occupation | Statisticians |
| Other nearby roles | Business Intelligence Analysts, Financial Quantitative Analysts, Data Warehousing Specialists, and Biostatisticians |
| First two PCA components | 46.3% of total variation |
| Components needed for 80% | 14 |
| Best clustering result | `k = 2`, silhouette score `0.209` |
| Software tools identified | 87 total, including 55 Hot Technology and 21 In Demand labels |

## Why this project matters

Career advice often relies on titles, salary rankings, or broad assumptions about transferable skills. Those shortcuts can hide important differences between occupations. A comparison based on the work itself can give students and career changers a clearer starting point, as long as the results are treated as descriptive guidance rather than a prediction of anyone's potential.

## Data

I used the [O*NET 31.0 database](https://www.onetcenter.org/database.html) from the U.S. Department of Labor's Employment and Training Administration. The analysis combines 109 Importance ratings from four domains:

- Essential Skills
- Transferable Skills
- Knowledge
- Work Activities

The primary analysis applies the documented reliability and suppression policy. A second, less restrictive dataset retains flagged estimates so I can test whether the main findings depend on that cleaning decision.

The raw O*NET archive is not included in this repository. The verified August 2026 archive has this SHA-256 value:

```text
b8d9bc5bcf3ed90fcacf386ad40171a46d63cebb47acc250c2ae88a9f3448a23
```

## What I did

1. Audited the source files, keys, expected ranges, duplicates, and reliability flags before analysis.
2. Preserved the raw inputs and documented how suppressed or flagged estimates were handled.
3. Standardized all 109 features so measures on different scales could be compared fairly.
4. Used principal component analysis to explore broad occupational structure.
5. Ranked similar occupations in the full standardized feature space rather than from a two-dimensional chart.
6. Tested the findings under different suppression, distance, and domain-weighting choices.
7. Evaluated clustering solutions before deciding how much meaning to assign to them.
8. Connected the occupational results to O*NET software information to create a practical learning roadmap.

## Main findings

Statisticians were the closest occupation to Data Scientists. Business Intelligence Analysts, Financial Quantitative Analysts, Data Warehousing Specialists, and Biostatisticians were also among the nearest roles.

The ranking was highly stable. Retaining flagged estimates preserved nine of the ten closest occupations, and the complete rankings across 881 shared non-target occupations had a Spearman correlation of `0.99996`. Cosine distance and equal domain weighting each preserved eight of the top ten occupations.

The first two principal components explained 46.3% of the variation, while 14 components were required to reach 80%. I therefore used PCA for exploration and communication, not as the space for the final similarity ranking.

The best clustering result had a silhouette score of only `0.209`. I interpreted it as a broad occupational divide rather than evidence of sharply separated career families.

## Repository contents

- `career_pathways_analysis.ipynb` — executed notebook with data intake, cleaning, validation, analysis, figures, and exported tables
- `analysis_pipeline.py` — reproducible Python version of the complete analytical workflow
- `requirements.txt` — verified Python package versions
- `README.md` — project overview, findings, limitations, and reproduction instructions

Generated results are written to the directory supplied through `ONET_RUN_DIR`. Raw data and generated outputs are excluded from version control.

## Reproduce the analysis

1. Clone the repository and enter the project folder.
2. Download and extract the O*NET 31.0 Excel database.
3. Create the Python environment and set the three paths shown below.
4. Run the script, or open the notebook and run its cells from top to bottom.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ONET_DATA_DIR="/absolute/path/to/db_31_0_excel"
export ONET_PROJECT_ROOT="$PWD"
export ONET_RUN_DIR="$PWD/analysis_output"

python analysis_pipeline.py
```

The pipeline validates the schema, checks ranges and duplicate keys, applies the documented suppression policy, standardizes the features, runs PCA and occupational similarity analyses, evaluates alternative analytical choices, tests clustering, and exports the result tables and figures.

## Tools

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, adjustText, openpyxl, and Jupyter Notebook

## Responsible use and limitations

This project compares occupations, not people. Similarity does not measure an individual's ability, hiring probability, expected salary, or chance of career success. O*NET ratings summarize occupations at a national level and may not represent every employer, location, specialty, or rapidly changing technology. The results should support exploration and better questions, not screen candidates or limit someone's choices.

## Data attribution and modifications

O*NET 31.0 Database content is used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution to the U.S. Department of Labor, Employment and Training Administration. O*NET is a trademark of USDOL/ETA. See the official [O*NET database license](https://www.onetcenter.org/license_db.html).

This project modifies the source data through suppression-based filtering, feature standardization, principal component analysis, similarity ranking, clustering, and an analyst-created grouping of software tools. These modifications are not endorsed by or affiliated with USDOL/ETA.
