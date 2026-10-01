# Beyond the Job Title: Mapping Data Science Skills and the Closest Career Paths

This repository contains the reproducible Python analysis for DSC 680 Project 1. The project compares the O*NET profile for Data Scientists (15-2051.00) with other U.S. occupations using 109 Importance ratings from Essential Skills, Transferable Skills, Knowledge, and Work Activities.

## Research question

Which occupations are most similar to Data Scientists, which measured characteristics distinguish the target from nearby roles, and how stable are those conclusions when reasonable analytical choices change?

## Files

- `DSC680_Project_1_Analysis.ipynb`: executed notebook with data intake, cleaning, validation, modeling, figures, and exported result tables.
- `analysis_pipeline.py`: reproducible script containing the same analytical workflow.
- `requirements.txt`: Python package versions used for the verified run.

The report, presentation, and Q&A are submitted separately through Blackboard and are intentionally not stored in this code repository.

## Data

Download the O*NET 31.0 Excel database from the [O*NET Resource Center](https://www.onetcenter.org/database.html). The verified August 2026 archive has this SHA-256 value:

```text
b8d9bc5bcf3ed90fcacf386ad40171a46d63cebb47acc250c2ae88a9f3448a23
```

Place the extracted Excel files in a local folder and set `ONET_DATA_DIR` to that folder. The raw O*NET archive is not included in this repository.

## Reproduce the analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ONET_DATA_DIR="/path/to/db_31_0_excel"
export DSC680_PROJECT_ROOT="$PWD"
export DSC680_RUN_DIR="$PWD/analysis_output"
python analysis_pipeline.py
```

The script validates the source schema, checks ranges and duplicate keys, applies the documented suppression policy, standardizes all features, runs PCA and occupational similarity models, tests sensitivity to distance and weighting choices, validates clustering, and exports tables and figures.

## Verified results

- Primary matrix: 882 occupations by 109 features.
- Full sensitivity matrix: 910 occupations by 109 features.
- Closest occupation to Data Scientists: Statisticians.
- Top-10 overlap after retaining flagged estimates: 9 of 10.
- PCA variance explained by the first two components: 46.3%.
- Selected clustering solution: k = 2 with silhouette = 0.209, interpreted only as a broad divide.

## Interpretation boundary

Occupational similarity is descriptive. It is not a measure of an individual's potential, hiring probability, expected pay, or causal career success.

## Data attribution and modifications

O*NET 31.0 Database content is used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution to the U.S. Department of Labor, Employment and Training Administration. O*NET is a trademark of USDOL/ETA. See the official [O*NET database license](https://www.onetcenter.org/license_db.html).

This project modifies the source data through suppression-based filtering, feature standardization, PCA, similarity ranking, clustering, and an analyst-created grouping of software tools. These modifications are not endorsed by or affiliated with USDOL/ETA.
