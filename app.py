"""
Bangalore Green Cover & Urban Heat Analysis
Interactive Streamlit app – ward-level tree planting priority explorer
"""

import io
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
import streamlit as st
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# PAGE CONFIG
st.set_page_config(
    page_title="Bangalore Green Cover Analysis",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a5c1a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f8f0;
        border-left: 4px solid #2e8b57;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .metric-title {
        font-size: 0.78rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a5c1a;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #888;
    }
    .rec-card {
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.7rem;
        border: 1px solid #eee;
    }
    .rec-high {
        background: #fff5f5;
        border-left: 5px solid #e53e3e;
    }
    .rec-med {
        background: #fffaf0;
        border-left: 5px solid #dd6b20;
    }
    .rec-low {
        background: #f0fff4;
        border-left: 5px solid #38a169;
    }
    .legend-box {
        background: white;
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #ddd;
        font-size: 0.85rem;
    }
    .ward-detail-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# DATA LOADING & SIMULATION

@st.cache_data
def load_and_enrich_data():
    """Load complete GeoJSON, simulate spectral indices, run KMeans."""
    gdf = gpd.read_file("bangalore_priority_zones.geojson")
    gdf = gdf.to_crs(epsg=4326)

    # Load real GEE-exported values
    import pandas as pd
    import os
    dfs = []
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        path = f"bangalore_ward_stats_{year}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["year"] = year
            dfs.append(df)
    stats_df = pd.concat(dfs, ignore_index=True)
    stats_df = stats_df.groupby("ward_name")[["NDVI_mean","NDBI_mean","LST_mean"]].mean().reset_index()
    stats_df["ward_name"] = stats_df["ward_name"].str.strip()
    gdf["ward_name"] = gdf["ward_name"].str.strip()

# Drop old index column from CSV if present
    stats_df = stats_df.drop(columns=[c for c in stats_df.columns if c.startswith("Unnamed")])
    
    # Drop existing spectral columns from gdf to avoid merge conflicts
    cols_to_drop = [c for c in ["NDVI_mean","LST_mean","NDBI_mean","UrbanIndex_mean",
                                 "NDVI_norm","LST_norm","NDBI_norm","Priority_Score",
                                 "Priority_Tier","Cluster","Cluster_Label"] 
                    if c in gdf.columns]
    gdf = gdf.drop(columns=cols_to_drop)

    gdf["ward_name"] = gdf["ward_name"].str.strip()
    stats_df["ward_name"] = stats_df["ward_name"].str.strip()

    gdf = gdf.merge(
        stats_df[["ward_name", "NDVI_mean", "NDBI_mean", "LST_mean"]],
        on="ward_name",
        how="left",
    )

    # Fill any wards missing from the export with city median
    for col in ["NDVI_mean", "LST_mean", "NDBI_mean"]:
        gdf[col] = gdf[col].fillna(gdf[col].median())

    gdf["NDVI_mean"]       = gdf["NDVI_mean"].round(3)
    gdf["LST_mean"]        = gdf["LST_mean"].round(1)
    gdf["NDBI_mean"]       = gdf["NDBI_mean"].round(3)
    gdf["UrbanIndex_mean"] = (gdf["NDBI_mean"] - gdf["NDVI_mean"]).round(3)

    # Priority Score (higher = needs more trees)
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min()) * 100

    gdf["NDVI_norm"] = normalize(gdf["NDVI_mean"])
    gdf["LST_norm"] = normalize(gdf["LST_mean"])
    gdf["NDBI_norm"] = normalize(gdf["NDBI_mean"])

    gdf["Priority_Score"] = (
        (100 - gdf["NDVI_norm"]) * 0.50 +
        gdf["LST_norm"] * 0.30 +
        gdf["NDBI_norm"] * 0.20
    ).round(1)

    # KMeans clustering
    features = ["NDVI_mean", "LST_mean", "NDBI_mean"]
    X = gdf[features].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    gdf["Cluster"] = km.fit_predict(X_sc)

    # Map clusters to human labels
    cluster_means = gdf.groupby("Cluster")[features].mean()
    # Sort by NDVI descending → most green first
    order = cluster_means["NDVI_mean"].sort_values(ascending=False).index.tolist()
    label_map = {
        order[0]: "Low Priority – Already Green",
        order[1]: "Medium Priority",
        order[2]: "High Priority – Needs Trees",
    }
    gdf["Cluster_Label"] = gdf["Cluster"].map(label_map)

    # Priority tier
    gdf["Priority_Tier"] = pd.cut(
        gdf["Priority_Score"],
        bins=[0, 33, 66, 100],
        labels=["Low Priority", "Medium Priority", "High Priority"],
    )

    # Clean ward name display (strip leading number)
    gdf["ward_display"] = gdf["ward_name"].apply(
        lambda x: " ".join(x.split(" - ")[1:]) if " - " in str(x) else str(x)
    )

    return gdf


# COLOURS
PRIORITY_COLORS = {
    "High Priority – Needs Trees": "#e53e3e",
    "Medium Priority": "#dd6b20",
    "Low Priority – Already Green": "#38a169",
}

LAYER_PALETTES = {
    "Priority Score": {
        "column": "Priority_Score",
        "vmin": 0, "vmax": 100,
        "colormap": ["#2e8b57", "#8bc34a", "#ffeb3b", "#ff9800", "#e53e3e"],
        "legend_labels": ["0 – Low", "25", "50", "75", "100 – High"],
    },
    "NDVI (Tree Cover)": {
        "column": "NDVI_mean",
        "vmin": 0.05, "vmax": 0.75,
        "colormap": ["#8B0000", "#FF0000", "#FFA500", "#FFFF00", "#90EE90", "#228B22"],
        "legend_labels": ["<0.1 Barren", "0.2", "0.4", "0.6", ">0.7 Dense"],
    },
    "LST – Heat (°C)": {
        "column": "LST_mean",
        "vmin": 26, "vmax": 43,
        "colormap": ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#ff6b35", "#d7191c"],
        "legend_labels": ["26°C Cool", "30°C", "33°C", "36°C", "40°C", ">42°C Hot"],
    },
    "NDBI (Built-up)": {
        "column": "NDBI_mean",
        "vmin": -0.25, "vmax": 0.55,
        "colormap": ["#2E8B57", "#FFFF00", "#FF8C00", "#FF0000", "#8B0000"],
        "legend_labels": ["Vegetation", "Bare soil", "Low built", "Moderate", "Dense built"],
    },
    "Cluster (K-Means)": {
        "column": "Cluster_Label",
        "type": "categorical",
        "colormap": PRIORITY_COLORS,
    },
}


def value_to_hex(val, vmin, vmax, palette):
    """Interpolate a scalar value to hex color from a list of hex stops."""
    colors_rgb = [tuple(int(c[i:i+2], 16) for i in (1, 3, 5)) for c in palette]
    t = np.clip((val - vmin) / (vmax - vmin), 0, 1)
    idx = t * (len(colors_rgb) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(colors_rgb) - 1)
    frac = idx - lo
    r = int(colors_rgb[lo][0] + frac * (colors_rgb[hi][0] - colors_rgb[lo][0]))
    g = int(colors_rgb[lo][1] + frac * (colors_rgb[hi][1] - colors_rgb[lo][1]))
    b = int(colors_rgb[lo][2] + frac * (colors_rgb[hi][2] - colors_rgb[lo][2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def add_fill_colors(gdf, layer_key):
    """Compute fill color per row and store as _fill_color column."""
    cfg = LAYER_PALETTES[layer_key]
    col = cfg["column"]
    gdf = gdf.copy()
    if cfg.get("type") == "categorical":
        color_map = cfg["colormap"]
        gdf["_fill_color"] = gdf[col].map(color_map).fillna("#cccccc")
    else:
        vmin, vmax = cfg["vmin"], cfg["vmax"]
        palette = cfg["colormap"]
        gdf["_fill_color"] = gdf[col].apply(
            lambda v: value_to_hex(float(v), vmin, vmax, palette)
        )
    return gdf


# MAP BUILDER
def build_map(gdf_filtered, layer_key, selected_ward=None):
    """Build a Folium map – color embedded directly in each GeoJSON feature."""
    center_lat = gdf_filtered.geometry.to_crs(epsg=32643).centroid.to_crs(epsg=4326).y.mean()
    center_lng = gdf_filtered.geometry.to_crs(epsg=32643).centroid.to_crs(epsg=4326).x.mean()

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # Embed _fill_color into the GDF before serialising to GeoJSON.
    # style_function reads it from feature["properties"] – no dict lookup needed.
    gdf_colored = add_fill_colors(gdf_filtered, layer_key)
    geojson_data = json.loads(gdf_colored.to_json())

    tooltip_fields = ["ward_display", "Corporation", "Priority_Score",
                       "NDVI_mean", "LST_mean", "NDBI_mean", "Cluster_Label"]
    tooltip_aliases = ["Ward:", "Corporation:", "Priority Score:",
                        "NDVI:", "LST (°C):", "NDBI:", "Category:"]

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            "fillColor": feature["properties"].get("_fill_color", "#cccccc"),
            "color": "#222",
            "weight": 0.7,
            "fillOpacity": 0.75,
        },
        highlight_function=lambda feature: {
            "weight": 2.5,
            "color": "#ffffff",
            "fillOpacity": 0.92,
        },
        tooltip=GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
            labels=True,
            style="font-size:12px;",
        ),
        name="Wards",
    ).add_to(m)

    # Highlight selected ward
    if selected_ward:
        sel = gdf_filtered[gdf_filtered["ward_display"] == selected_ward]
        if not sel.empty:
            folium.GeoJson(
                json.loads(sel.to_json()),
                style_function=lambda x: {
                    "fillColor": "#f6e05e", "color": "#d69e2e",
                    "weight": 3, "fillOpacity": 0.9,
                },
                name="Selected Ward",
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# RECOMMENDATIONS
def get_recommendations(row):
    """Return a list of action strings for a ward based on its metrics."""
    recs = []
    ndvi = row["NDVI_mean"]
    lst = row["LST_mean"]
    ndbi = row["NDBI_mean"]
    priority = row["Priority_Score"]

    if priority >= 66:
        recs.append("🚨 **Immediate Action:** Plant native canopy trees in road medians, empty lots, and park edges.")
        if lst > 38:
            recs.append("🌡️ **Heat Mitigation:** Deploy reflective surfaces and green roofs to reduce the urban heat island effect (current avg LST {:.1f}°C).".format(lst))
        if ndbi > 0.3:
            recs.append("🏗️ **Built-up Density High (NDBI {:.2f}):** Mandate green building standards and sky gardens in new constructions.".format(ndbi))
        recs.append("📋 **Target:** Increase NDVI from {:.2f} to above 0.30 within 3 years through structured planting drives.".format(ndvi))
    elif priority >= 33:
        if ndvi < 0.30:
            recs.append("🌳 **Tree Augmentation:** Supplement existing vegetation with fast-growing shade trees along arterial roads.")
        recs.append("🛡️ **Preservation Order:** Protect all existing mature trees from removal due to construction or road-widening.")
        recs.append("🌿 **Green Corridors:** Develop linear parks connecting open spaces to enhance ecological connectivity.")
        if lst > 33:
            recs.append("🌊 **Water Bodies:** Consider creating or restoring small water features to provide localised cooling.")
    else:
        recs.append("✅ **Maintain Green Cover:** Current NDVI ({:.2f}) is healthy – enforce tree protection bylaws.".format(ndvi))
        recs.append("🔄 **Monitoring:** Bi-annual satellite NDVI monitoring to detect any green cover loss.")
        recs.append("🌱 **Biodiversity:** Introduce native understory planting to increase species diversity without large-scale canopy changes.")

    return recs


# SIDEBAR
with st.sidebar:
    st.markdown("## 🌳 Filters & Settings")
    st.markdown("---")

    gdf = load_and_enrich_data()

    # Corporation filter
    corps = ["All Corporations"] + sorted(gdf["Corporation"].dropna().unique().tolist())
    selected_corp = st.selectbox("🏛️ Corporation Zone", corps)

    # Priority tier filter
    priority_opts = ["All Priorities", "High Priority", "Medium Priority", "Low Priority"]
    selected_priority = st.selectbox("⚠️ Priority Tier", priority_opts)

    # Layer selector
    layer_key = st.selectbox(
        "🗺️ Map Layer",
        list(LAYER_PALETTES.keys()),
    )

    st.markdown("---")
    st.markdown("### 📌 Ward Drill-Down")
    ward_list = sorted(gdf["ward_display"].unique().tolist())
    selected_ward = st.selectbox("Select a ward to highlight:", ["None"] + ward_list)
    if selected_ward == "None":
        selected_ward = None

    st.markdown("---")
    st.markdown("### 📖 Layer Explanations")

    with st.expander("🌿 NDVI – Vegetation Index"):
        st.markdown("""
**What exactly is NDVI?**

- The NDVI (or Normalized Difference Vegetation Index) is a spectral index that measures the presence of healthy green vegetation.
- It uses the difference between near-infrared (which vegetation strongly reflects) and red light (which vegetation absorbs) to calculate a value between -1 and +1.
- Higher NDVI values (closer to +1) indicate denser, healthier vegetation,
- Data source: Sentinel-2 Band 8 / Band 4
- Temporal window used (e.g. Jan–Dec 2024, cloud-free median)
        """)

    with st.expander("🌡️ LST – Land Surface Temperature"):
        st.markdown("""
**What exactly is LST?**

- The LST (or Land Surface Temperature) is an estimate of the temperature of the Earth's surface, derived from thermal infrared satellite data.
- Units: degrees Celsius (°C)
- Data source: Landsat-8 Band 10 (ST_B10), scale factor applied to convert to °C
- Higher LST values indicate hotter surfaces, often due to impervious materials and lack of vegetation
        """)

    with st.expander("🏗️ NDBI – Built-up Index"):
        st.markdown("""
**What exactly is NDBI?**

- The NDBI (or Normalized Difference Built-up Index) is a spectral index that highlights artificial surfaces.
- It uses the difference between shortwave infrared (which built-up areas reflect) and near-infrared (which they absorb) to calculate a value typically between -0.25 and +0.55.
- Higher NDBI values indicate more built-up, impervious surfaces, which often correlate with higher temperatures and lower vegetation.
- Data source: Sentinel-2 Band 11 / Band 8
- Relationship to impervious surface coverage
        """)

    with st.expander("🎯 Priority Score – How It's Calculated"):
        st.markdown("""

- Formula: **Score = 50% × (1 − NDVI_norm) + 30% × LST_norm + 20% × NDBI_norm**
- Each index is normalised 0–100 across all wards before weighting
- Higher score = ward needs more urgent tree planting intervention
- Thresholds: 0–33 Low, 34–66 Medium, 67–100 High Priority
        """)

    with st.expander("🔵 K-Means Clustering"):
        st.markdown("""

- Unsupervised clustering on NDVI + LST + NDBI (StandardScaler normalised)
- 3 clusters mapped to Low / Medium / High priority categories
- The clusters are assigned by sorting cluster centroids by NDVI (most green = Low Priority)
- Limitations: clusters are relative to this dataset only.
        """)

    st.markdown("---")
    st.markdown(
        "<small>**Data note:** Spectral indices derived from Google Earth Engine · "
        "Sentinel-2 & Landsat-8 · March–June median composite 2020–2025.</small>",
        unsafe_allow_html=True,
    )


# FILTER
gdf_view = gdf.copy()
if selected_corp != "All Corporations":
    gdf_view = gdf_view[gdf_view["Corporation"] == selected_corp]
if selected_priority != "All Priorities":
    gdf_view = gdf_view[gdf_view["Priority_Tier"] == selected_priority]


# HEADER
st.markdown('<div class="main-header">🌳 Bangalore Green Cover & Urban Heat Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ward-level NDVI · Land Surface Temperature · Built-up Index · Tree Planting Priorities</div>', unsafe_allow_html=True)


# KPI METRICS ROW
col1, col2, col3, col4, col5 = st.columns(5)
hp = (gdf_view["Priority_Tier"] == "High Priority").sum()
mp = (gdf_view["Priority_Tier"] == "Medium Priority").sum()
lp = (gdf_view["Priority_Tier"] == "Low Priority").sum()
avg_ndvi = gdf_view["NDVI_mean"].mean()
avg_lst = gdf_view["LST_mean"].mean()

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Wards Shown</div>
        <div class="metric-value">{len(gdf_view)}</div>
        <div class="metric-sub">of {len(gdf)} total</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="metric-card" style="border-color:#e53e3e; background:#fff5f5;">
        <div class="metric-title">🔴 High Priority</div>
        <div class="metric-value" style="color:#e53e3e;">{hp}</div>
        <div class="metric-sub">Urgent planting needed</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""<div class="metric-card" style="border-color:#dd6b20; background:#fffaf0;">
        <div class="metric-title">🟠 Medium Priority</div>
        <div class="metric-value" style="color:#dd6b20;">{mp}</div>
        <div class="metric-sub">Augmentation needed</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Avg NDVI</div>
        <div class="metric-value">{avg_ndvi:.2f}</div>
        <div class="metric-sub">{"🟢 Healthy" if avg_ndvi > 0.35 else "🟡 Moderate" if avg_ndvi > 0.22 else "🔴 Low"}</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""<div class="metric-card" style="border-color:#dd6b20;">
        <div class="metric-title">Avg LST</div>
        <div class="metric-value">{avg_lst:.1f}°C</div>
        <div class="metric-sub">{"🔴 Very hot" if avg_lst > 38 else "🟠 Hot" if avg_lst > 34 else "🟡 Moderate"}</div>
    </div>""", unsafe_allow_html=True)


# MAIN CONTENT
map_col, panel_col = st.columns([3, 2])

with map_col:
    st.markdown(f"#### 🗺️ {layer_key} – {selected_corp}")
    if len(gdf_view) == 0:
        st.warning("No wards match the current filters.")
    else:
        m = build_map(gdf_view, layer_key, selected_ward)
        map_data = st_folium(m, width=720, height=540, returned_objects=["last_object_clicked"])

        # Legend
        cfg = LAYER_PALETTES[layer_key]
        if cfg.get("type") == "categorical":
            legend_html = " &nbsp; ".join(
                [f'<span style="display:inline-block;width:12px;height:12px;background:{v};border-radius:2px;margin-right:4px"></span>{k.replace(" – ", " – ")}'
                 for k, v in cfg["colormap"].items()]
            )
        else:
            legend_html = " &nbsp; ".join(
                [f'<span style="display:inline-block;width:12px;height:12px;background:{cfg["colormap"][i]};border-radius:2px;margin-right:4px"></span>{cfg["legend_labels"][i]}'
                 for i in range(min(len(cfg["colormap"]), len(cfg["legend_labels"])))]
            )
        st.markdown(f'<div class="legend-box">{legend_html}</div>', unsafe_allow_html=True)


with panel_col:
    # Tabs: Selected ward detail | Top priorities | Data table
    tab1, tab2, tab3 = st.tabs(["📋 Ward Detail", "🏆 Top Priorities", "📊 Data Table"])

    with tab1:
        if selected_ward and selected_ward != "None":
            ward_row = gdf_view[gdf_view["ward_display"] == selected_ward]
            if not ward_row.empty:
                r = ward_row.iloc[0]
                priority_label = str(r["Priority_Tier"])
                priority_class = "rec-high" if "High" in priority_label else "rec-med" if "Medium" in priority_label else "rec-low"

                st.markdown(f'<div class="ward-detail-header">{r["ward_display"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**Zone:** {r['Corporation']}  |  **Assembly:** {r['ac']}")

                # Metric chips
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("NDVI", f"{r['NDVI_mean']:.2f}")
                c2.metric("LST", f"{r['LST_mean']:.1f}°C")
                c3.metric("NDBI", f"{r['NDBI_mean']:.2f}")
                c4.metric("Score", f"{r['Priority_Score']:.0f}")

                st.markdown(f'<div class="rec-card {priority_class}"><strong style="color:#1a202c;">Priority: {priority_label}</strong></div>', unsafe_allow_html=True)

                st.markdown("##### 📌 Actionable Recommendations")
                recs = get_recommendations(r)
                for rec in recs:
                    st.markdown(f"- {rec}")

                # Temperature estimate after canopy addition
                trees_needed = max(0, int((0.35 - r["NDVI_mean"]) / 0.001 * 50))
                temp_drop = max(0, round((0.35 - r["NDVI_mean"]) * 6, 1))
                if r["NDVI_mean"] < 0.35:
                    st.info(f"🌡️ Planting ~**{trees_needed:,} trees** could lower average temperature by **{temp_drop}°C** over 5 years.")
            else:
                st.info("Selected ward not visible under current filters.")
        else:
            st.info("👈 Select a ward in the sidebar to view detailed recommendations.")

            # Show city-wide summary
            st.markdown("##### City-Wide Overview")
            st.markdown(f"""
- **{hp}** wards need urgent tree planting intervention
- **{mp}** wards require moderate augmentation  
- **{lp}** wards have healthy green cover
- Avg temperature in high-priority wards: **{gdf[gdf['Priority_Tier']=='High Priority']['LST_mean'].mean():.1f}°C**
- Avg temperature in low-priority wards: **{gdf[gdf['Priority_Tier']=='Low Priority']['LST_mean'].mean():.1f}°C**
- **Estimated temperature gap:** {gdf[gdf['Priority_Tier']=='High Priority']['LST_mean'].mean() - gdf[gdf['Priority_Tier']=='Low Priority']['LST_mean'].mean():.1f}°C between worst and best wards
            """)

    with tab2:
        st.markdown("##### Top 15 Wards Needing Trees")
        top15 = gdf_view.nlargest(15, "Priority_Score")[
            ["ward_display", "Corporation", "Priority_Score", "NDVI_mean", "LST_mean", "NDBI_mean", "Cluster_Label"]
        ].rename(columns={
            "ward_display": "Ward", "Corporation": "Zone",
            "Priority_Score": "Score", "NDVI_mean": "NDVI",
            "LST_mean": "LST°C", "NDBI_mean": "NDBI",
            "Cluster_Label": "Category"
        })

    st.dataframe(
            top15,
            use_container_width=True,
            height=430,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=100,
                    format="%d",
                ),
            }
        )

    with tab3:
        st.markdown("##### All Visible Wards")
        display_df = gdf_view[[
            "ward_display", "Corporation", "Priority_Tier",
            "Priority_Score", "NDVI_mean", "LST_mean", "NDBI_mean",
        ]].rename(columns={
            "ward_display": "Ward", "Corporation": "Zone",
            "Priority_Tier": "Priority", "Priority_Score": "Score",
            "NDVI_mean": "NDVI", "LST_mean": "LST°C", "NDBI_mean": "NDBI",
        }).sort_values("Score", ascending=False).reset_index(drop=True)

        st.dataframe(display_df, use_container_width=True, height=430)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="bangalore_ward_priority.csv",
            mime="text/csv",
        )


# ABOUT SECTION
st.markdown("---")
with st.expander("ℹ️ About this Dashboard"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
### About the Project

- As we all know, the Bengaluru heat has been unbearable during the summer over the past few years. It wasn't always like this - and I wanted to do something about it.
- Did you know that over the last 50 years, Bengaluru has lost 88% of its green cover? Not just that, Land Surface Temperatures have gone up by 7.9°C from 1973 to 2015!
- There are so many different things we can do in our own houses to try and make ourselves cooler - but why not try something that will just make things better for everyone?
- Is all this development really worth it if it comes at the cost of our sanity? Let's make a change!
        """)
        st.markdown("""
### Data Sources

- Sentinel-2 SR (via Google Earth Engine) – NDVI, NDBI
- Landsat-8 Collection 2 (via GEE) – Land Surface Temperature
- BBMP Ward Boundaries – https://data.opencity.in/dataset/bbmp-ward-information
- Analysis period: March-June of 2020-2025 (pre-monsoon season with highest heat stress)
        """)
    with col_b:
        st.markdown("""
### Methodology

- Pre-monsoon (March–June) median composite to capture peak heat stress period
- Cloud masking via Sentinel-2 SCL band and Landsat QA_PIXEL band
- Zonal mean per ward polygon at 30m resolution using GEE reduceRegions
- K-Means (k=3) on StandardScaler-normalised NDVI, LST, NDBI
- Priority Score weighted: 50% vegetation deficit, 30% heat, 20% built-up density
        """)
        st.markdown("""
### Limitations

- Temporal resolution of satellite data
- Mixed-pixel effects at ward boundaries
- Recommendations are currently just based on my knowledge. Will collaborate with researchers and other experts to refine them.
        """)

st.markdown(
    "<small>Priority Score = 50% × (1−NDVI) + 30% × LST + 20% × NDBI · "
    "Data source: Google Earth Engine · Ward boundaries: BBMP</small>",
    unsafe_allow_html=True,
)
