# Lecture 6.3 data and attribution

All data load from files beside the notebook. The notebook does not download datasets or install packages.

- `nba.csv`: existing course-local NBA per-game statistics. The raw file has 708 rows and 530 player identifiers. Some traded players have a `TOT` season total plus team-specific rows. The notebook retains `TOT` for those players and the single row for other players. The original source URL and retrieval date were not recorded with this local file. No new measurements were added. Assists (`AST`), three-point attempts (`3PA`) and points (`PTS`) are per-game averages. These contemporaneous features support a descriptive relationship rather than a pre-season forecast.
- Polynomial-response, feature-selection, patient-group and time-series examples are constructed with seed 538. Patient fingerprints and labels are simulated and contain no real patient data.
- The time example measures rolling one-step prediction with the previous observed outcomes available. It does not evaluate a fixed-origin multi-step forecast.
- `spatial_temperature_demo.csv`: 400 synthetic sensor locations on a 20 by 20 grid, spaced 0.5 km apart within a 10 km square. The visible notebook code generates a smooth temperature surface with independent Gaussian measurement noise using seed 538. Columns are east and north coordinates in kilometres and temperature in degrees Celsius. These course-authored data contain no real locations or observed temperatures. Both evaluation designs use 280 training locations and 120 evaluated locations, with the geographic split reserving the eastern area at east >= 7 km.
- The spatial case compares interpolation within a sensor network with transfer to an unobserved area at the same time. It does not estimate future temperatures, establish a causal effect of location, or isolate autocorrelation as the sole reason for the error difference. Method reference: [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html). Further reading: [Mahoney et al., Assessing the performance of spatial cross-validation approaches for models of spatially structured data](https://arxiv.org/abs/2303.07334).

## Code attribution

Adapted code retains the upstream attribution: Copyright (c) 2026, UC Berkeley Data 100. The applicable BSD 2-Clause notice is included in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). This notice does not grant a new licence for separately sourced datasets.
