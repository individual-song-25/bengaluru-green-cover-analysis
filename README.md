# 🌳 Bangalore Green Cover & Urban Heat Dashboard

An interactive Streamlit web app for exploring ward-level vegetation, heat, and urban density across Bangalore — with actionable tree planting recommendations derived from satellite imagery.

---

## What It Does

- Displays a choropleth map of all BBMP wards coloured by NDVI, LST, NDBI, Priority Score, or K-Means cluster
- Lets users filter by Corporation zone and priority tier
- Shows per-ward metrics and tailored tree planting recommendations on click
- Ranks the top 15 most urgent wards for intervention
- Provides a full data table with CSV export

---

## Data Sources

| Data | Source |
|---|---|
| NDVI & NDBI | Sentinel-2 SR via Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`) |
| Land Surface Temperature | Landsat-8 Collection 2 via GEE (`LANDSAT/LC08/C02/T1_L2`) |
| Ward boundaries | BBMP ward GeoJSON — [opencity.in](https://data.opencity.in/dataset/bbmp-ward-information) |
| Analysis period | March–June, 2020–2025 (pre-monsoon, peak heat stress) |

Spectral indices were exported from GEE as zonal mean statistics per ward polygon at 30m resolution, then loaded as static CSVs.

---

## Project Structure

```
project-folder/
│
├── app.py                           # Main Streamlit application
├── bangalore_heat_project_model.ipynb    # GEE analysis notebook (data pipeline)
├── bangalore_priority_zones.geojson      # BBMP ward boundaries + computed features
│
├── bangalore_ward_stats_2020.csv         # GEE export – March–June 2020
├── bangalore_ward_stats_2021.csv         # GEE export – March–June 2021
├── bangalore_ward_stats_2022.csv         # GEE export – March–June 2022
├── bangalore_ward_stats_2023.csv         # GEE export – March–June 2023
├── bangalore_ward_stats_2024.csv         # GEE export – March–June 2024
├── bangalore_ward_stats_2025.csv         # GEE export – March–June 2025
│
└── README.md
```

---

## Analysis Notebook

`bangalore_heat_project_model.ipynb` is the core data pipeline that generates all inputs for the Streamlit app. Run it in Jupyter with an active GEE session before running the app for the first time, or whenever you want to update the data.

**What the notebook does, cell by cell:**

| Step | What happens |
|---|---|
| GEE authentication & initialisation | Connects to your GEE project |
| Load ward boundaries | Reads the KML ward file into a GeoDataFrame |
| Sentinel-2 collection | Filters March–June 2020–2025, masks clouds via QA60, takes median composite |
| NDVI calculation | `(B8 − B4) / (B8 + B4)` — visualised with a custom legend |
| Landsat-8 LST | Masks clouds via QA_PIXEL, applies official C2 scale factors, converts to °C |
| NDBI calculation | Computed from Landsat SR_B6 (SWIR) and SR_B5 (NIR) |
| Urban Index | `NDBI − NDVI` — combines both into a single urbanisation signal |
| Zonal extraction | `reduceRegions` at 30m scale — extracts mean NDVI, LST, NDBI, Urban Index per ward |
| Priority Score | Normalises each index 0–100, weights: 50% NDVI deficit + 30% LST + 20% NDBI |
| K-Means optimisation | Tests k=2 to 30, plots silhouette scores to find the best k |
| K-Means clustering | Fits k=3 clusters on StandardScaler-normalised features, assigns priority labels |
| Cluster map | Visualises High / Medium / Low priority wards in geemap |
| Summary output | Prints ward counts, average metrics, and plain-language recommendations |
| Export | Saves `bangalore_priority_zones.geojson` and `bangalore_tree_priority_wards.csv` |

**To run the notebook:**
```bash
pip install earthengine-api geemap geopandas scikit-learn matplotlib pandas
jupyter notebook bangalore_heat_project_model.ipynb
```

You will need a Google account with GEE access. Run `ee.Authenticate()` in the second cell if credentials are not already saved locally.

---

## Setup & Installation

**Requirements:** Python 3.9+

1. Clone or download this repository

2. Install dependencies:
```bash
pip install streamlit folium streamlit-folium geopandas scikit-learn pandas numpy
```

3. Make sure all required files are in the same folder as `app.py` (see Project Structure above)

4. Run the app:
```bash
streamlit run app.py
```

---

## Methodology

### Spectral Indices

| Index | Formula | What it measures |
|---|---|---|
| NDVI | (B8 − B4) / (B8 + B4) | Vegetation density and health |
| NDBI | (B11 − B8) / (B11 + B8) | Built-up / impervious surface density |
| LST | Landsat ST_B10 × 0.00341802 + 149 − 273.15 | Land surface temperature in °C |

All three use pre-monsoon (March–June) median composites averaged across 2020–2025 to capture the period of peak urban heat stress.

### Priority Score

Each ward receives a priority score (0–100) calculated as:

```
Priority Score = 50% × (1 − NDVI_norm) + 30% × LST_norm + 20% × NDBI_norm
```

Where each index is min-max normalised to 0–100 across all wards before weighting. A higher score means the ward needs more urgent tree planting intervention.

**Thresholds:**
- 0–33 → Low Priority
- 34–66 → Medium Priority
- 67–100 → High Priority

### K-Means Clustering

Wards are grouped into 3 clusters using K-Means on StandardScaler-normalised NDVI, LST, and NDBI values. Cluster labels are assigned by sorting cluster centroids by mean NDVI (highest NDVI = Low Priority, lowest = High Priority).

---

## Re-exporting GEE Data

To update the ward statistics with a new year or date range, run the GEE export script (`gee_export.py` if included, or the standalone Python block from the project documentation) with your Google Earth Engine credentials:

```bash
python gee_export.py
```

Tasks will appear in the [GEE Tasks tab](https://code.earthengine.google.com). Once complete, download the CSVs from Google Drive and replace the existing files in the project folder.

---

## Limitations

- Ward boundary file covers 281 of Bangalore's 369 wards — inner-city wards may be missing depending on the GeoJSON source used
- Recommendations are based on index thresholds and will be refined in collaboration with domain experts
- Mixed-pixel effects at ward boundaries can affect zonal mean accuracy for small or irregular wards
- LST values represent surface radiometric temperature, not air temperature

---

## Author

**Adhvaith Jaishankar** 
