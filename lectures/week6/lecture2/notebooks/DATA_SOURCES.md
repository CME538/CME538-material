# Lecture 6.2 data and attribution

All data load from files beside the notebook. The notebook does not download datasets or install packages.

- `auto-mpg.csv`: existing local Auto MPG data. Missing horsepower entries marked `?` are converted to missing numeric values. [Original dataset and dictionary](https://archive.ics.uci.edu/dataset/9/auto+mpg).
- `air_quality.csv`: local R `airquality` data for New York in 1973. [Data dictionary](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/airquality.html). Log ozone requires positive values; the complete-case subset used here has positive ozone measurements.
- `ice_cream.csv`: existing small sweetness/overall-rating teaching table. The original collection provenance is not recorded in the supplied file. Treat the rating units as those of this example rather than a calibrated measurement scale.
- `titanic.csv`: existing Titanic passenger data used for category, age and text examples. [Dataset description](https://www.kaggle.com/competitions/titanic/data). The notebook extracts passenger titles, replaces missing ages and illustrates a fare cap. The final passenger case study also explores embarkation-port categories, reference categories and missingness.
- `flight_feature_case_study.csv`: existing deterministic sample of 8,000 scheduled New York City flights in 2013, prepared earlier for CME538 by joining flight, carrier and hourly weather tables. The earlier preparation notes report replacing one impossible wind-speed value with missing. [Flight source](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236), [hourly weather source](https://mesonet.agron.iastate.edu/request/download.phtml). The file is unchanged in this revision. It lacks weather issue-time and publication-latency fields, so the notebook treats it as retrospective preprocessing data rather than claiming verified real-time availability. Outcomes stay outside the feature matrix.
- The small people, education, ticket, country, breed and date examples are constructed teaching tables, not population samples. The peak-time sequences in the canonical waveform case are also constructed; they contain no patient observations and do not establish a diagnostic rule.

## Code attribution

Adapted code retains the upstream attribution: Copyright (c) 2026, UC Berkeley Data 100. The applicable BSD 2-Clause notice is included in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). This notice does not grant a new licence for separately sourced datasets.
