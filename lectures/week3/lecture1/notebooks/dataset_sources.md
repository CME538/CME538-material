Lecture 3.1 dataset details

California housing

File: california_housing.csv
Original data: 1990 US census block groups, compiled for Pace and Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters 33, 291-297, 1997.
Documentation: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
Archive: https://ndownloader.figshare.com/files/5976036
Archive SHA-256: aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681
Retrieved: 21 September 2026.

The local CSV preserves CaliforniaHousing/cal_housing.data byte for byte, with a CSV filename. No rows or values changed. The original file has no header. Its nine columns, in order, are longitude, latitude, median housing age, total rooms, total bedrooms, population, households, median income, and median house value. Home values use 1990 US dollars. The file contains 20,640 block groups. Credit the original compilers and StatLib when reusing the extract. The archive does not include a separate licence statement.

The notebook reads columns 0, 1, and 8. Repeated coordinates reflect limited coordinate precision. Average the reported block-group median values at each unique coordinate to produce 12,590 plotting locations. These averages are not pooled medians of individual homes. No individual-home observations or population weights enter this calculation.

The nearest-location comparison uses distance on a sphere. Each location contributes one directed nearest-neighbour pair. Reciprocal pairs occur twice. A single permutation with seed 538 supplies the comparison. Locations sort by latitude, then longitude. The median absolute gaps are $22,300 for observed values and $97,950 for shuffled values. The comparison is descriptive, without a significance test or a causal interpretation. The highest recorded home value is 500,001. The map labels its upper colour limit as $500k+.

USGS earthquake catalogue

File: usgs_california_nevada_2025_01.csv
Catalogue: https://earthquake.usgs.gov/earthquakes/search/
Query: https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2025-01-01&endtime=2025-02-01&minlatitude=32&maxlatitude=37&minlongitude=-121&maxlongitude=-114&eventtype=earthquake&orderby=time-asc
Query documentation: https://earthquake.usgs.gov/fdsnws/event/1/
Magnitude definitions: https://www.usgs.gov/programs/earthquake-hazards/magnitude-types
Negative magnitudes: https://www.usgs.gov/faqs/how-can-earthquake-have-a-negative-magnitude
Retrieved: 21 September 2026.

The local file preserves the complete CSV response, including event identifiers, review status, magnitude type, uncertainties, and source networks. Cite USGS and the contributing seismic networks when reusing records. See the USGS data-use policy: https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits

The geographic rectangle spans 32 to 37 degrees north and 121 to 114 degrees west. This window includes parts of California and Nevada, not either complete state. The query covers January 2025 in UTC, with the service's inclusive end bound at 1 February 00:00. No returned row occurs at the end bound. The response has 1,391 events, below the service's 20,000-event limit.

The notebook selects magType = ml and status = reviewed. This leaves 1,323 events, including 40 below magnitude zero. Other magnitude definitions and two automatic local-magnitude events stay in the source file but outside this comparison. The deliberately incorrect nonnegative filter creates a separate 1,283-row comparison. The original selection stays unchanged.

The logarithmic scale graphic is a mathematical illustration of local magnitude, not a plot of recorded seismograms or a conversion of catalogue magnitudes into measured amplitudes. A negative magnitude alone is not a reason to delete an event. Network sensitivity and reporting thresholds limit catalogue completeness for small events.
