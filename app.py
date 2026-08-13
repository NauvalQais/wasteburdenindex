import streamlit as st
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static
import json

st.set_page_config(
    page_title="Klasifikasi Pengelolaan Sampah",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model_decision_tree.joblib")

model = load_model()

# Load GeoJSON
@st.cache_data
def load_geojson():
    with open("3273-kota-bandung-level-kewilayahan.json", "r") as f:
        data = json.load(f)
    
    wilayah_list = []
    if 'features' in data:
        for feature in data['features']:
            props = feature.get('properties', {})
            wilayah_list.append({
                'nama': props.get('nama_wilayah', ''),
                'id': props.get('id_wilayah', ''),
                'geometry': feature.get('geometry')
            })
    return wilayah_list, data

# Inisialisasi session state
if 'wilayah_status' not in st.session_state:
    st.session_state.wilayah_status = {}

if 'wilayah_params' not in st.session_state:
    st.session_state.wilayah_params = {}

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False

# Konstanta label
LABEL_CONFIG = {
    "KRITIS": {
        "color": "#E74C3C", 
        "icon": "KRITIS",
        "desc": "Pengelolaan sampah dalam kondisi kritis. Diperlukan tindakan segera."
    },
    "WASPADA": {
        "color": "#F39C12", 
        "icon": "WASPADA",
        "desc": "Pengelolaan sampah perlu perhatian lebih. Risiko meningkat."
    },
    "AMAN": {
        "color": "#27AE60", 
        "icon": "AMAN",
        "desc": "Pengelolaan sampah berjalan dengan baik."
    },
}

DEFAULT_PARAMS = {
    "rasio_angkut": 0.75,
    "rasio_diolah": 0.50,
    "rasio_sisa": 0.25,
    "indeks_jarak": 0.40
}

# Sidebar
with st.sidebar:
    st.markdown("### Pengaturan Peta")
    show_boundary = st.checkbox("Tampilkan Batas Wilayah", value=True)
    show_labels = st.checkbox("Tampilkan Label Wilayah", value=True)
    st.markdown("---")
    st.markdown("**Keterangan Warna**")
    st.markdown("**Merah**: Kritis")
    st.markdown("**Kuning**: Waspada")
    st.markdown("**Hijau**: Aman")
    st.markdown("**Abu-abu**: Belum Diprediksi")
    
    if st.button("Reset Semua Data", use_container_width=True):
        st.session_state.wilayah_status = {}
        st.session_state.wilayah_params = {}
        st.session_state.prediction_result = None
        st.session_state.show_prediction = False
        st.rerun()

# Header
st.markdown(
     """
    <head>
        <meta name="dicoding:email" content="rizki123.rs1@gmail.com">
        <meta name="dicoding:email" content="andinn1410@gmail.com">
        <meta name="dicoding:email" content="nauvaall.qa@gmail.com">
    </head>
    """,
    unsafe_allow_html=True
)
st.title("Klasifikasi Pengelolaan Sampah")
st.markdown("Kota Bandung — Berbasis Waste Burden Index (WBI)")
st.divider()

# Layout 2 kolom
col_left, col_right = st.columns([1, 1.2])

# Load data
wilayah_list, geojson_data = load_geojson()
wilayah_names = [w['nama'] for w in wilayah_list if w['nama']]

# ==================== KOLOM KIRI ====================
with col_left:
    with st.expander("Panduan Pengisian Fitur", expanded=False):
        st.markdown("""
        | Fitur | Keterangan | Rentang |
        |---|---|---|
        | **Rasio Angkut** | Proporsi sampah yang berhasil diangkut | 0.0 – 1.0 |
        | **Rasio Diolah** | Proporsi sampah yang berhasil diolah | 0.0 – 1.0 |
        | **Rasio Sisa** | Proporsi sampah yang tersisa | 0.0 – 1.0 |
        | **Indeks Jarak** | Indeks jarak tempuh TPS ke TPA | 0.0 – 1.0 |
        """)
    
    if wilayah_names:
        selected_wilayah = st.selectbox("Pilih Wilayah:", wilayah_names)
        
        saved_params = st.session_state.wilayah_params.get(selected_wilayah, DEFAULT_PARAMS.copy())
        
        st.markdown("### Input Data")
        col1, col2 = st.columns(2)
        
        with col1:
            rasio_angkut = st.number_input(
                "Rasio Angkut", min_value=0.0, max_value=1.0, step=0.01,
                value=saved_params["rasio_angkut"], key=f"angkut_{selected_wilayah}"
            )
            rasio_diolah = st.number_input(
                "Rasio Diolah", min_value=0.0, max_value=1.0, step=0.01,
                value=saved_params["rasio_diolah"], key=f"diolah_{selected_wilayah}"
            )
        
        with col2:
            rasio_sisa = st.number_input(
                "Rasio Sisa", min_value=0.0, max_value=1.0, step=0.01,
                value=saved_params["rasio_sisa"], key=f"sisa_{selected_wilayah}"
            )
            indeks_jarak = st.number_input(
                "Indeks Jarak", min_value=0.0, max_value=1.0, step=0.01,
                value=saved_params["indeks_jarak"], key=f"jarak_{selected_wilayah}"
            )
        
        # Tombol Klasifikasi
        if st.button("Klasifikasi dan Prediksi", type="primary", use_container_width=True):
            fitur = np.array([[rasio_angkut, rasio_diolah, rasio_sisa, indeks_jarak]])
            
            try:
                prediksi = model.predict(fitur)[0]
                label = str(prediksi).strip().upper()
                
                if label not in LABEL_CONFIG:
                    label = "WASPADA"
                
                st.session_state.wilayah_status[selected_wilayah] = label
                st.session_state.wilayah_params[selected_wilayah] = {
                    "rasio_angkut": rasio_angkut,
                    "rasio_diolah": rasio_diolah,
                    "rasio_sisa": rasio_sisa,
                    "indeks_jarak": indeks_jarak
                }
                
                st.session_state.prediction_result = {
                    "label": label,
                    "cfg": LABEL_CONFIG[label],
                    "params": {
                        "rasio_angkut": rasio_angkut,
                        "rasio_diolah": rasio_diolah,
                        "rasio_sisa": rasio_sisa,
                        "indeks_jarak": indeks_jarak
                    }
                }
                st.session_state.show_prediction = True
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        # TAMPILAN HASIL KLASIFIKASI
        if st.session_state.show_prediction and st.session_state.prediction_result:
            res = st.session_state.prediction_result
            label = res["label"]
            cfg = res["cfg"]
            
            st.markdown("---")
            st.markdown("### Hasil Klasifikasi")
            
            if label == "KRITIS":
                st.error(f"**STATUS: {label}**")
            elif label == "WASPADA":
                st.warning(f"**STATUS: {label}**")
            else:
                st.success(f"**STATUS: {label}**")
            
            st.write(cfg['desc'])
            
            if hasattr(model, "predict_proba"):
                st.write("**Probabilitas:**")
                fitur = np.array([[res["params"]["rasio_angkut"], res["params"]["rasio_diolah"], 
                                   res["params"]["rasio_sisa"], res["params"]["indeks_jarak"]]])
                proba = model.predict_proba(fitur)[0]
                classes = [str(c).upper() for c in model.classes_]
                
                for c, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                    st.progress(p, text=f"{c}: {p*100:.1f}%")
            
            st.write("**Ringkasan Input:**")
            cols = st.columns(4)
            values = [
                ("Rasio Angkut", res["params"]["rasio_angkut"]),
                ("Rasio Diolah", res["params"]["rasio_diolah"]),
                ("Rasio Sisa", res["params"]["rasio_sisa"]),
                ("Indeks Jarak", res["params"]["indeks_jarak"]),
            ]
            for col, (label, val) in zip(cols, values):
                col.markdown(
                    f"""<div style="background:#f5f5f5;border-radius:8px;padding:8px 10px;text-align:center;">
                        <div style="font-size:12px;color:#555;">{label}</div>
                        <div style="font-size:16px;font-weight:600;margin-top:2px;">{val:.3f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            
            st.write("**Rekomendasi:**")
            params = res["params"]
            rekomendasi = []
            if params["rasio_sisa"] > 0.3:
                rekomendasi.append("Rasio sisa tinggi — tambah frekuensi pengangkutan")
            if params["rasio_angkut"] < 0.7:
                rekomendasi.append("Rasio angkut rendah — evaluasi armada")
            if params["rasio_diolah"] < 0.4:
                rekomendasi.append("Rasio diolah rendah — tingkatkan kapasitas pengolahan")
            if params["indeks_jarak"] > 0.7:
                rekomendasi.append("Jarak ke TPA jauh — optimasi rute")
            if not rekomendasi:
                rekomendasi.append("Semua indikator dalam kondisi baik")
            
            for r in rekomendasi:
                st.write(r)

# ==================== KOLOM KANAN (PETA) ====================
with col_right:
    st.markdown("### Peta Wilayah Kota Bandung")
    
    # Buat peta
    center_lat = -6.9146
    center_lon = 107.6098
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Tambahkan tile alternatif
    folium.TileLayer('openstreetmap').add_to(m)
    
    def get_wilayah_color(wilayah_name):
        status = st.session_state.wilayah_status.get(wilayah_name, "")
        if status == "KRITIS":
            return "#E74C3C"
        elif status == "WASPADA":
            return "#F39C12"
        elif status == "AMAN":
            return "#27AE60"
        else:
            return "#95A5A6"
    
    if show_boundary and geojson_data:
        for feature in geojson_data.get('features', []):
            props = feature.get('properties', {})
            wilayah_name = props.get('nama_wilayah', '')
            
            if not wilayah_name:
                continue
                
            fill_color = get_wilayah_color(wilayah_name)
            params = st.session_state.wilayah_params.get(wilayah_name, {})
            status_text = st.session_state.wilayah_status.get(wilayah_name, 'Belum diprediksi')
            
            popup_html = f"""
            <div style="min-width: 180px;">
                <b>{wilayah_name}</b><br>
                Status: {status_text}<br>
                <hr>
                Rasio Angkut: {params.get('rasio_angkut', '-')}<br>
                Rasio Diolah: {params.get('rasio_diolah', '-')}<br>
                Rasio Sisa: {params.get('rasio_sisa', '-')}<br>
                Indeks Jarak: {params.get('indeks_jarak', '-')}
            </div>
            """
            
            opacity = 0.6 if fill_color != "#95A5A6" else 0.3
            
            folium.GeoJson(
                feature,
                name=wilayah_name,
                style_function=lambda x, color=fill_color, op=opacity: {
                    'fillColor': color,
                    'color': '#2C3E50',
                    'weight': 1.5,
                    'fillOpacity': op,
                },
                tooltip=wilayah_name,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)
            
            if show_labels and wilayah_name:
                try:
                    geom = feature.get('geometry', {})
                    if geom.get('type') == 'Polygon':
                        coords = geom['coordinates'][0]
                        if coords and len(coords) > 0:
                            lats = [c[1] for c in coords]
                            lons = [c[0] for c in coords]
                            center_lat_label = sum(lats) / len(lats)
                            center_lon_label = sum(lons) / len(lons)
                            
                            folium.Marker(
                                location=[center_lat_label, center_lon_label],
                                icon=folium.DivIcon(
                                    html=f'<div style="font-size: 10px; font-weight: bold; background: white; padding: 2px 6px; border-radius: 4px; border: 1px solid {fill_color};">{wilayah_name}</div>'
                                )
                            ).add_to(m)
                except:
                    pass
    
    # Legend
    legend_html = '''
    <div style="position: fixed; bottom: 30px; right: 30px; background: white; padding: 8px 12px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); font-size: 11px; z-index: 1000;">
        <b>Status</b><br>
        <span style="color:#e74c3c;">■</span> Kritis<br>
        <span style="color:#f39c12;">■</span> Waspada<br>
        <span style="color:#27ae60;">■</span> Aman<br>
        <span style="color:#95a5a6;">■</span> Belum
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # PAKAI INI - folium_static bukan st_folium
    folium_static(m, width=700, height=550)
    
    st.caption("Klik area untuk melihat detail | Warna berubah setelah klasifikasi")

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray;'>Developed by Masoem University – Fakultas Teknik</div>",
    unsafe_allow_html=True
)