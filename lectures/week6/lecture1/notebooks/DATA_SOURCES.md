# Lecture 6.1 data and attribution

All data load from files beside the notebook. The notebook does not download datasets or install packages.

- `tips.csv`: local copy of the restaurant tips dataset already used in CME538 Week 5. The notebook retains the first 243 observations to preserve the earlier teaching sample. Total bill and tip are in US dollars. [Dataset and description](https://rdrr.io/cran/reshape2/man/tips.html), [seaborn data copy](https://github.com/mwaskom/seaborn-data/blob/master/tips.csv).
- `height_weight_train.csv`: existing local height/weight teaching data, with weight in pounds and height in inches. The original collection and sampling provenance are not recorded in the supplied file. The notebook makes no population-representativeness claim. The separate `height_weight_test.csv` is not used by this notebook.
- `air_quality.csv`: local copy of R's `airquality` measurements for daily New York air quality in 1973. Ozone is in ppb, wind in mph and temperature in degrees Fahrenheit. The examples use complete rows. [Data dictionary](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/airquality.html).
- `nba.csv`: existing course-local NBA per-game statistics. The raw file has 708 rows and 530 player identifiers. Some traded players have a `TOT` season total plus team-specific rows. The canonical notebook retains the original 708 rows for the descriptive fits used in the slides. These rows are not independent player-season observations. Lecture 6.3 keeps one season record per player before splitting; the `_new_new` notebook also uses that reduced sample. The original source URL and retrieval date were not recorded with this local file. No new measurements were added. Assists (`AST`), three-point attempts (`3PA`) and points (`PTS`) are per-game averages. These contemporaneous features support a descriptive relationship rather than a pre-season forecast.
- Correlation clouds and salary examples are constructed with seed 538. Salary uses illustrative units, not recorded salaries.
- Anscombe's quartet is a constructed statistical example attributed to F. J. Anscombe, “Graphs in Statistical Analysis” (1973). The numerical values are embedded in the notebook.

## Code attribution

Adapted code retains the upstream attribution: Copyright (c) 2026, UC Berkeley Data 100. The applicable BSD 2-Clause notice is included in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). This notice does not grant a new licence for separately sourced datasets.
