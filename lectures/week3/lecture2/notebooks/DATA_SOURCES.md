# Lecture 3.2 local data

All required examples use these local files. Reading the notebook does not make a network request.

| Files | Provenance and classroom role |
| --- | --- |
| `groceries_dataset.csv`, `.tsv`, `.psv` | Existing course fixture in three delimiters. Its original acquisition and licence were not recorded alongside the files; no new source or licence is asserted. The copies are unchanged. |
| `e202027712.0.wav` | Existing NOAA GLERL course file, with repeated forecast headers and data blocks. Only its first block is parsed. Column units are not inferred from the numbers. [Source organisation](https://www.glerl.noaa.gov/res/glcfs/). |
| `tweet.json` | Existing 2017 Twitter Developer example. Used as a historical nested document, not as a claim about the current API. |
| `books.xml` | Existing course copy of the [Microsoft sample book catalogue](https://learn.microsoft.com/en-us/previous-versions/windows/desktop/ms762271(v=vs.85)). |
| `NEON_hyperspectral_dataset.h5` | Existing NEON SJER reflectance subset. Scaling, ignored-data code, wavelengths and map coordinates are read from its metadata. [NEON HDF5 tutorial](https://www.neonscience.org/resources/learning-hub/tutorials/neon-refl-h5-py). |
| `civmin_response.html`, `civmin_response.json` | HTML and response metadata recovered from saved outputs in the pre-revision Lecture 3.2 notebook. The HTTP Date header is **13 September 2024, 04:54:02 GMT**. Original request URL: <https://civmin.utoronto.ca/>. The helper reconstructs a response object; it does not pretend to issue a new request. The saved Unicode text is re-encoded to illustrate `.content`; these are not archived wire bytes. |
| `coingecko_prices.json` | Reconstructed from the four-coin, four-currency table saved in the pre-revision notebook. Values retain the **displayed precision only**. No response date was preserved with that table, so none is asserted. This is a frozen teaching snapshot, not current market information or an exact archived API body. Original endpoint: <https://api.coingecko.com/api/v3/simple/price>. |
| `coingecko_coins_excerpt.json` | The ten coin identifiers, symbols, and names printed by the reference notebook's `/coins/list` example. This is an excerpt, not a complete or current coin list. |
| `coingecko_bitcoin_history.json` | The original notebook's 365 dated Bitcoin chart summaries, 15 September 2023 through 13 September 2024, plus five timestamp/price pairs whose timestamps also appear in the saved table. Values retain the precision in the embedded Plotly trace. The full market-chart response was not saved. The file therefore labels the data as daily summaries and a timestamped preview, not an archived API response. Original route: `/coins/bitcoin/market_chart`, USD, 365 days. |
| `noaa_archive_excerpt.html` | The first 1,000 HTML characters preserved in the reference notebook's NOAA request output. Source: <https://www.glerl.noaa.gov/emf/waves/GLERL-Donelan-Archive/2021/>. The excerpt ends during the next row and is not the full archive listing. |
| `books_page.html` | Public Books to Scrape listing page downloaded **20 September 2026** from <https://books.toscrape.com/catalogue/category/books_1/index.html>. Contains 20 book records. This site is explicitly a scraping sandbox: its prices and ratings are demonstration values, not real retail evidence. Used as local HTML, not executed in a browser. |

The 404 response is constructed in `utils.py` solely to demonstrate status handling; no failing remote request is sent. External HTML remains text parsed by BeautifulSoup. A web page layout and an API's requirements can change independently of this frozen classroom example.

Restoration source: <https://github.com/CME538/CME538-material/blob/master/lectures/week3/lecture2/notebooks/week3_lecture2.ipynb>. Downloaded 21 September 2026. SHA-256: `0dee0426cfe333d4f912acd73a68cef11f4101dc19fd24ff6d458a111f9a4a41`. This file matches the instructor's local reference byte for byte.

The reference candlestick code assigned `min` to `high` and `max` to `low`. The restored extract corrects those field names without changing any numeric values. For 364 of the 365 dates, all four values are identical. The final date contains different first and last values. The notebook preserves these daily summaries, plots the last sampled price per date, and aggregates monthly first, maximum, minimum, and last values for the interactive candlestick chart. Neither plot claims full intraday coverage. Both endpoint months are partial.

The original HTTP portrait URL returns 404 as of the restoration check. The byte-content example now uses the existing tweet PNG, retaining the binary-resource lesson without a broken download. Prepared request headers come from Requests defaults, not the archived CivMin transaction.
