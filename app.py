import os
import ee
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon, MultiPolygon
from dotenv import load_dotenv

load_dotenv()
BANDS = [f'A{str(i).zfill(2)}' for i in range(64)]
BUFFER_METERS = 100

# ---------- Cached setup (runs once, not on every UI interaction) ----------
@st.cache_resource
def init_ee():
    # Cloud: read from Streamlit secrets; Local: read from .env
    try:
        project_id = st.secrets["PROJECT_ID"]
    except (KeyError, FileNotFoundError):
        project_id = os.getenv("PROJECT_ID")
    if project_id:
        project_id = project_id.strip()

    # Try service account key first (Streamlit Cloud), fall back to default credentials
    try:        
        service_account_key = st.secrets["EARTH_ENGINE_SERVICE_ACCOUNT_KEY"]
        import json
        key_dict = json.loads(service_account_key)
        credentials = ee.ServiceAccountCredentials(key_dict["client_email"], key_data=service_account_key)
        ee.Initialize(credentials, project=project_id)
    except (KeyError, FileNotFoundError):
        ee.Initialize(project=project_id)

    return True

@st.cache_resource
def load_model():
    model = joblib.load("dataset/mine_clustered_classifier.joblib")
    scaler = joblib.load("dataset/feature_clustered_scaler.joblib")
    return model, scaler

@st.cache_data
def get_myanmar_regions():
    gaul = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level1')
    names = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Myanmar')).aggregate_array('ADM1_NAME').getInfo()
    return sorted(names)

@st.cache_data
def get_latest_year():
    col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    years = col.aggregate_array('system:time_start').map(lambda t: ee.Date(t).get('year')).distinct().sort().getInfo()
    return max(years)

# ---------- Geometry helpers ----------
def geojson_to_shapely(geom_info):
    gtype = geom_info['type']
    if gtype == 'Polygon':
        c = geom_info['coordinates']
        return Polygon(c[0], c[1:] if len(c) > 1 else None)
    elif gtype == 'MultiPolygon':
        return MultiPolygon([Polygon(p[0], p[1:] if len(p) > 1 else None) for p in geom_info['coordinates']])
    elif gtype == 'GeometryCollection':
        polys = []
        for sub in geom_info['geometries']:
            if sub['type'] == 'Polygon':
                c = sub['coordinates']
                polys.append(Polygon(c[0], c[1:] if len(c) > 1 else None))
            elif sub['type'] == 'MultiPolygon':
                polys += [Polygon(p[0], p[1:] if len(p) > 1 else None) for p in sub['coordinates']]
        return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    raise ValueError(f"Unexpected geometry type: {gtype}")

def get_region_geometry(region_name):
    gaul = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level1')
    matched = gaul.filter(ee.Filter.And(ee.Filter.eq('ADM0_NAME', 'Myanmar'), ee.Filter.eq('ADM1_NAME', region_name)))
    feature = matched.first()
    geom_info = feature.geometry().getInfo()
    bounds = feature.geometry().bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]; lats = [p[1] for p in bounds]
    return geom_info, (min(lats), max(lats), min(lons), max(lons))

def build_grid_within_region(region_name, spacing_m):
    geom_info, (min_lat, max_lat, min_lon, max_lon) = get_region_geometry(region_name)
    shapely_poly = geojson_to_shapely(geom_info)
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * np.cos(np.radians((min_lat + max_lat) / 2))
    lats = np.arange(min_lat, max_lat, spacing_m / m_per_deg_lat)
    lons = np.arange(min_lon, max_lon, spacing_m / m_per_deg_lon)
    return [(lat, lon) for lat in lats for lon in lons if shapely_poly.contains(Point(lon, lat))]


def build_grid_around_point(center_lat, center_lon, radius_km, spacing_m):
    """Generate grid points within a circular radius of a coordinate."""
    radius_m = radius_km * 1000
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * np.cos(np.radians(center_lat))

    lat_span = radius_m / m_per_deg_lat
    lon_span = radius_m / m_per_deg_lon

    lats = np.arange(center_lat - lat_span, center_lat + lat_span, spacing_m / m_per_deg_lat)
    lons = np.arange(center_lon - lon_span, center_lon + lon_span, spacing_m / m_per_deg_lon)

    center_point = Point(center_lon, center_lat)
    points = []
    for lat in lats:
        for lon in lons:
            # Approximate distance check in meters (good enough at this scale)
            dist_m = np.sqrt(
                ((lat - center_lat) * m_per_deg_lat) ** 2 +
                ((lon - center_lon) * m_per_deg_lon) ** 2
            )
            if dist_m <= radius_m:
                points.append((lat, lon))

    return points


# ---------- Scan + classify ----------
def extract_grid_embeddings(points, year, progress_callback=None, chunk_size=500):
    img = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
        .filterDate(f'{year}-01-01', f'{year}-12-31').mosaic().select(BANDS)
    all_results = []
    total_chunks = (len(points) + chunk_size - 1) // chunk_size

    for i in range(total_chunks):
        chunk = points[i * chunk_size:(i + 1) * chunk_size]
        features = [ee.Feature(ee.Geometry.Point([lon, lat]).buffer(BUFFER_METERS),
                    {'lat': lat, 'lon': lon}) for lat, lon in chunk]
        fc = ee.FeatureCollection(features)
        reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
        try:
            result = reduced.getInfo()
            for feat in result['features']:
                props = feat['properties']
                if props.get(BANDS[0]) is not None:
                    row = {'lat': props['lat'], 'lon': props['lon']}
                    row.update({b: props.get(b) for b in BANDS})
                    all_results.append(row)
        except Exception:
            continue
        if progress_callback:
            progress_callback((i + 1) / total_chunks)

    return pd.DataFrame(all_results)

def cluster_flagged_points(df, eps_m=300):
    if len(df) == 0:
        return df
    coords_rad = np.radians(df[['lat', 'lon']].values)
    db = DBSCAN(eps=eps_m / 6371000, min_samples=1, metric='haversine').fit(coords_rad)
    df = df.copy()
    df['site_cluster'] = db.labels_
    sites = df.groupby('site_cluster').agg(
        lat=('lat', 'mean'), lon=('lon', 'mean'),
        mine_probability=('mine_probability', 'max'),
        n_grid_points=('site_cluster', 'count')
    ).reset_index(drop=True)
    return sites

# ---------- Expansion analysis (per confirmed candidate site) ----------
NDVI_THRESHOLD = 0.3
SITE_RADIUS_M = 500
EXPANSION_YEARS = list(range(2016, 2027))

def compute_ndvi_stats(lat, lon, year):
    """Compute NDVI statistics for a site in a given year (matches export_expansion.py)."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(SITE_RADIUS_M).bounds()

    start = f"{year}-01-01"
    end = f"{year}-12-31"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )

    count = collection.size()
    if count.getInfo() == 0:
        return None

    image = collection.median()
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    disturbed = ndvi.lt(NDVI_THRESHOLD).rename("NDVI")

    result = ee.Dictionary({
        "ndvi_mean": ndvi.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e8
        ).get("NDVI"),
        "ndvi_std": ndvi.reduceRegion(
            reducer=ee.Reducer.stdDev(), geometry=region, scale=10, maxPixels=1e8
        ).get("NDVI"),
        "disturbed_ha": ee.Number(disturbed.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=10, maxPixels=1e8
        ).get("NDVI")).multiply(0.01),
        "total_ha": ee.Number(disturbed.reduceRegion(
            reducer=ee.Reducer.count(), geometry=region, scale=10, maxPixels=1e8
        ).get("NDVI")).multiply(0.01),
        "image_count": count,
    }).getInfo()

    if result.get("ndvi_mean") is None:
        return None
    return result

def analyze_expansion(lat, lon, start_year=2016, end_year=None):
    """Compare disturbed area (NDVI < 0.3) between start and end year using Sentinel-2."""
    if end_year is None:
        end_year = get_latest_year()
    start_stats = compute_ndvi_stats(lat, lon, start_year)
    end_stats = compute_ndvi_stats(lat, lon, end_year)
    if start_stats is None or end_stats is None:
        return {'bare_area_start_ha': None, 'bare_area_end_ha': None,
                'area_change_ha': None, 'pct_change': None, 'expanding': 'no_data'}
    start_area = start_stats['disturbed_ha']
    end_area = end_stats['disturbed_ha']
    change = end_area - start_area
    pct = (change / start_area * 100) if start_area > 0 else None
    return {'bare_area_start_ha': round(start_area, 2), 'bare_area_end_ha': round(end_area, 2),
            'area_change_ha': round(change, 2), 'pct_change': round(pct, 1) if pct else None,
            'expanding': change > 0.5}

# ---------- True-color before/after snapshots ----------
def mask_landsat_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
    return image.updateMask(mask)

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)

def get_landsat_true_color(region, start, end):
    """Harmonized L5/L7/L8/L9 true-color composite, properly scaled to reflectance."""
    def scale_landsat(img):
        return img.multiply(0.0000275).add(-0.2)

    def prep(collection, bands):
        return (collection
                .filterBounds(region).filterDate(start, end)
                .filter(ee.Filter.calendarRange(1, 3, 'month'))
                .map(mask_landsat_clouds)
                .map(lambda img: scale_landsat(img.select(bands, ['BLUE', 'GREEN', 'RED']))))

    l5 = prep(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'), ['SR_B1', 'SR_B2', 'SR_B3'])
    l7 = prep(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'), ['SR_B1', 'SR_B2', 'SR_B3'])
    l8 = prep(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'), ['SR_B2', 'SR_B3', 'SR_B4'])
    l9 = prep(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'), ['SR_B2', 'SR_B3', 'SR_B4'])

    merged = l5.merge(l7).merge(l8).merge(l9)
    return merged.median().clip(region), merged.size()

def get_sentinel2_true_color(region, start, end):
    """Sentinel-2 true-color composite, scaled to reflectance."""
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.calendarRange(1, 3, 'month'))
           .map(mask_s2_clouds)
           .map(lambda img: img.select(['B4', 'B3', 'B2'], ['RED', 'GREEN', 'BLUE']).divide(10000)))
    return col.median().clip(region), col.size()

def get_site_snapshots(lat, lon, before_year, after_year, radius_m=500):
    """Return (before_url, after_url) true-color thumbnail URLs for a site.
    Before uses Landsat (better historical coverage); after uses Sentinel-2 (higher resolution)."""
    region = ee.Geometry.Point([lon, lat]).buffer(radius_m).bounds()

    before_start = ee.Date.fromYMD(before_year, 1, 1)
    before_end = ee.Date.fromYMD(before_year, 3, 31)
    after_start = ee.Date.fromYMD(after_year, 1, 1)
    after_end = ee.Date.fromYMD(after_year, 3, 31)

    before_img, before_count = get_landsat_true_color(region, before_start, before_end)
    after_img, after_count = get_sentinel2_true_color(region, after_start, after_end)

    thumb_params = {'region': region, 'dimensions': 400, 'format': 'png',
                     'bands': ['RED', 'GREEN', 'BLUE'], 'min': 0, 'max': 0.3}

    before_url, after_url = None, None
    try:
        if before_count.getInfo() > 0:
            before_url = before_img.getThumbURL(thumb_params)
    except Exception:
        before_url = None
    try:
        if after_count.getInfo() > 0:
            after_url = after_img.getThumbURL(thumb_params)
    except Exception:
        after_url = None

    return before_url, after_url

def render_site_snapshots(snapshots, report_df):
    """Display before/after image pairs for each analyzed site."""
    if not snapshots:
        return
    st.subheader("Before / After satellite snapshots")
    for i, snap in enumerate(snapshots):
        label = f"Site {i + 1} — ({snap['lat']:.5f}, {snap['lon']:.5f})"
        with st.expander(label, expanded=(i == 0)):
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"Before ({snap['before_year']})")
                if snap['before_url']:
                    st.image(snap['before_url'])
                else:
                    st.info("No cloud-free imagery available for this year.")
            with c2:
                st.caption(f"After ({snap['after_year']})")
                if snap['after_url']:
                    st.image(snap['after_url'])
                else:
                    st.info("No cloud-free imagery available for this year.")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Rare Earth Mine Monitor", layout="wide")
st.title("Rare Earth Mine Detection & Expansion Monitor")

init_ee()
model, scaler = load_model()

with st.sidebar:
    st.header("Scan settings")
    regions = get_myanmar_regions()
    region_name = st.selectbox("Myanmar state/region", regions, index=regions.index("Kachin") if "Kachin" in regions else 0)
    spacing = st.slider("Grid spacing (meters)", 200, 1000, 500, step=100)
    threshold = st.slider("Mine probability threshold", 0.0, 1.0, 0.5, step=0.05)
    run_scan_btn = st.button("Run region scan", type="primary")

if run_scan_btn:
    with st.spinner(f"Building grid for {region_name}..."):
        points = build_grid_within_region(region_name, spacing)
        st.info(f"{len(points)} grid points to scan")

    progress_bar = st.progress(0.0, text="Extracting embeddings...")
    year = get_latest_year()
    df = extract_grid_embeddings(points, year, progress_callback=lambda p: progress_bar.progress(p, text=f"Scanning... {int(p*100)}%"))
    progress_bar.empty()

    if len(df) == 0:
        st.warning("No valid embeddings extracted for this region.")
    else:
        X_scaled = scaler.transform(df[BANDS].values)
        df['mine_probability'] = model.predict_proba(X_scaled)[:, 1]
        df['flagged'] = df['mine_probability'] >= threshold

        candidates = df[df['flagged']]
        st.success(f"Scanned {len(df)} points, flagged {len(candidates)}")

        sites = cluster_flagged_points(candidates)
        st.session_state['sites'] = sites
        st.session_state['region_name'] = region_name


st.divider()
st.header("Check a specific coordinate")

col1, col2, col3 = st.columns(3)
input_lat = col1.number_input("Latitude", value=25.65463, format="%.6f")
input_lon = col2.number_input("Longitude", value=98.26000, format="%.6f")
input_radius_km = col3.number_input("Search radius (km)", min_value=0.5, max_value=20.0, value=3.0, step=0.5)

check_coord_btn = st.button("Check this location", type="primary")

if check_coord_btn:
    points = build_grid_around_point(input_lat, input_lon, input_radius_km, spacing_m=200)
    st.info(f"{len(points)} grid points to scan")

    year = get_latest_year()
    progress_bar = st.progress(0.0, text="Extracting embeddings...")
    df = extract_grid_embeddings(
        points, year,
        progress_callback=lambda p: progress_bar.progress(p, text=f"Scanning... {int(p*100)}%")
    )
    progress_bar.empty()

    if len(df) == 0:
        st.session_state['coord_result'] = {'status': 'no_data'}
    else:
        X_scaled = scaler.transform(df[BANDS].values)
        df['mine_probability'] = model.predict_proba(X_scaled)[:, 1]
        df['flagged'] = df['mine_probability'] >= threshold

        candidates = df[df['flagged']]
        if len(candidates) == 0:
            st.session_state['coord_result'] = {'status': 'no_mines', 'lat': input_lat, 'lon': input_lon, 'radius_km': input_radius_km}
        else:
            sites = cluster_flagged_points(candidates)
            st.session_state['coord_result'] = {
                'status': 'found',
                'lat': input_lat, 'lon': input_lon, 'radius_km': input_radius_km,
                'sites': sites, 'all_df': df
            }

if 'coord_result' in st.session_state:
    result = st.session_state['coord_result']
    if result['status'] == 'no_data':
        st.warning("No valid AlphaEarth data in this area — check the point isn't over water, or try a different year.")
    elif result['status'] == 'no_mines':
        st.success(f"No mine-like sites detected within {result['radius_km']}km of ({result['lat']}, {result['lon']}).")
    elif result['status'] == 'found':
        sites = result['sites']
        st.warning(f"Found {len(sites)} candidate site(s) within {result['radius_km']}km")

        m = folium.Map(location=[result['lat'], result['lon']], zoom_start=13, tiles=None)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri, Maxar, Earthstar Geographics',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        folium.TileLayer('OpenStreetMap', name='Streets', overlay=False, control=True).add_to(m)
        folium.LayerControl().add_to(m)
        folium.Marker([result['lat'], result['lon']], icon=folium.Icon(color='blue'),
                      popup="Your search point").add_to(m)
        folium.Circle([result['lat'], result['lon']], radius=result['radius_km'] * 1000,
                      color='blue', fill=False, dash_array='5').add_to(m)
        for _, row in sites.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']], radius=8, color='red', fill=True,
                popup=f"Probability: {row['mine_probability']:.3f}"
            ).add_to(m)
        st_folium(m, width=900, height=500, key="coord_check_map")

        st.dataframe(sites.sort_values('mine_probability', ascending=False))

        st.subheader("Environmental impact for detected site(s)")
        baseline_year = st.number_input("Baseline year for comparison", min_value=2016, max_value=2026, value=2016, key="coord_baseline")

        if st.button("Run expansion analysis on detected site(s)"):
            results = []
            snapshots = []
            end_year = get_latest_year()
            progress = st.progress(0.0)
            for i, (_, row) in enumerate(sites.iterrows()):
                expansion = analyze_expansion(row['lat'], row['lon'], start_year=baseline_year, end_year=end_year)
                results.append({
                    'lat': row['lat'], 'lon': row['lon'],
                    'mine_probability': row['mine_probability'],
                    **expansion
                })
                before_url, after_url = get_site_snapshots(row['lat'], row['lon'], baseline_year, end_year)
                snapshots.append({
                    'lat': row['lat'], 'lon': row['lon'],
                    'before_year': baseline_year, 'after_year': end_year,
                    'before_url': before_url, 'after_url': after_url
                })
                progress.progress((i + 1) / len(sites))
            progress.empty()

            report_df = pd.DataFrame(results)
            st.session_state['coord_report'] = report_df
            st.session_state['coord_snapshots'] = snapshots

        if 'coord_report' in st.session_state:
            report_df = st.session_state['coord_report']
            st.dataframe(report_df)
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download this report as CSV", csv,
                               f"impact_report_{result['lat']}_{result['lon']}.csv", "text/csv",
                               key="coord_download")
            render_site_snapshots(st.session_state.get('coord_snapshots', []), report_df)



if 'sites' in st.session_state:
    sites = st.session_state['sites']
    st.subheader(f"Candidate sites in {st.session_state['region_name']} ({len(sites)} distinct clusters)")

    m = folium.Map(location=[sites['lat'].mean(), sites['lon'].mean()], zoom_start=10)
    for _, row in sites.iterrows():
        color = 'red' if row['mine_probability'] > 0.8 else 'orange'
        folium.CircleMarker(
            location=[row['lat'], row['lon']], radius=6, color=color, fill=True,
            popup=f"Probability: {row['mine_probability']:.3f}<br>Grid points: {row['n_grid_points']}"
        ).add_to(m)
    st_folium(m, width=900, height=500)

    st.dataframe(sites.sort_values('mine_probability', ascending=False))

    st.subheader("Environmental impact analysis")
    col1, col2 = st.columns(2)
    start_year = col1.number_input("Baseline year", min_value=2016, max_value=2026, value=2016)
    max_sites = col2.number_input("Max sites to analyze (limits runtime)", min_value=1, max_value=len(sites), value=min(10, len(sites)))

    if st.button("Run expansion analysis on candidate sites"):
        results = []
        snapshots = []
        end_year = get_latest_year()
        progress = st.progress(0.0)
        top_sites = sites.sort_values('mine_probability', ascending=False).head(max_sites)

        for i, (_, row) in enumerate(top_sites.iterrows()):
            expansion = analyze_expansion(row['lat'], row['lon'], start_year=start_year, end_year=end_year)
            results.append({
                'lat': row['lat'], 'lon': row['lon'],
                'mine_probability': row['mine_probability'],
                **expansion
            })
            before_url, after_url = get_site_snapshots(row['lat'], row['lon'], start_year, end_year)
            snapshots.append({
                'lat': row['lat'], 'lon': row['lon'],
                'before_year': start_year, 'after_year': end_year,
                'before_url': before_url, 'after_url': after_url
            })
            progress.progress((i + 1) / len(top_sites))

        report_df = pd.DataFrame(results)
        st.session_state['report'] = report_df
        st.session_state['report_snapshots'] = snapshots
        progress.empty()

if 'report' in st.session_state:
    report_df = st.session_state['report']
    st.subheader("Environmental impact report")
    st.dataframe(report_df)

    n_expanding = (report_df['expanding'] == True).sum()
    st.metric("Sites showing expansion", f"{n_expanding} / {len(report_df)}")

    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download report as CSV", csv, "environmental_impact_report.csv", "text/csv")

    render_site_snapshots(st.session_state.get('report_snapshots', []), report_df)