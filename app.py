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

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1rem;
        color: #222;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem;
        color: #888;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.7rem;
    }
    [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.1rem;
    }
    [data-testid="stMarkdownContainer"] h3 {
        font-size: 0.95rem;
    }
    [data-testid="stMarkdownContainer"] h4 {
        font-size: 0.9rem;
    }
    [data-testid="stMarkdownContainer"] p {
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
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

# Nilai maksimum TOTAL_JARAK_BULAT pada data latih (km)
MAX_TOTAL_JARAK_BULAT = 91.0

DEFAULT_PARAMS = {
    "input": 7.0,
    "angkut": 6.0,
    "diolah": 0.0,
    "sisa": 1.0,
    "jarak_bulat": 63.0
}

def derive_features(vol_input, angkut, diolah, sisa, jarak_bulat):
    """Konversi input mentah -> 4 fitur yang dipakai model."""
    return {
        "rasio_angkut": float(angkut / vol_input) if vol_input else 0.0,
        "rasio_diolah": float(diolah / angkut) if angkut else 0.0,
        "rasio_sisa": float(sisa / vol_input) if vol_input else 0.0,
        "indeks_jarak": float(jarak_bulat / MAX_TOTAL_JARAK_BULAT),
    }

def balance_sisa(vol_input, angkut, diolah, sisa):
    """Koreksi SISA mengikuti pipeline training (selisih neraca > 3 m3)."""
    sisa_calc = max(vol_input - angkut - diolah, 0.0)
    if abs(sisa - sisa_calc) > 3:
        return sisa_calc, True
    return sisa, False

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
        | Fitur | Keterangan | Satuan |
        |---|---|---|
        | **INPUT** | Volume sampah masuk | m³ |
        | **ANGKUT** | Volume sampah yang berhasil diangkut | m³ |
        | **DIOLAH** | Volume sampah yang berhasil diolah | m³ |
        | **SISA** | Volume sampah tersisa (dikoreksi otomatis bila neraca selisih > 3 m³) | m³ |
        | **TOTAL_JARAK_BULAT** | Jarak bulat rute TPS ke TPA | km |

        Fitur model dihitung otomatis dari input mentah:
        `rasio_angkut = ANGKUT/INPUT`, `rasio_diolah = DIOLAH/ANGKUT`,
        `rasio_sisa = SISA/INPUT`, `indeks_jarak = TOTAL_JARAK_BULAT/{max_jarak}`.
        """.format(max_jarak=int(MAX_TOTAL_JARAK_BULAT)))
    
    if wilayah_names:
        selected_wilayah = st.selectbox("Pilih Wilayah:", wilayah_names)
        
        saved_params = st.session_state.wilayah_params.get(selected_wilayah, DEFAULT_PARAMS.copy())
        
        st.markdown("### Input Data Mentah")
        col1, col2 = st.columns(2)
        
        with col1:
            vol_input = st.number_input(
                "INPUT (m³)", min_value=0.0, step=0.5,
                value=saved_params["input"], key=f"input_{selected_wilayah}"
            )
            angkut = st.number_input(
                "ANGKUT (m³)", min_value=0.0, step=0.5,
                value=saved_params["angkut"], key=f"angkut_{selected_wilayah}"
            )
            diolah = st.number_input(
                "DIOLAH (m³)", min_value=0.0, step=0.5,
                value=saved_params["diolah"], key=f"diolah_{selected_wilayah}"
            )
        
        with col2:
            sisa_input = st.number_input(
                "SISA (m³)", min_value=0.0, step=0.5,
                value=saved_params["sisa"], key=f"sisa_{selected_wilayah}"
            )
            jarak_bulat = st.number_input(
                "TOTAL_JARAK_BULAT (km)", min_value=0.0, step=1.0,
                value=saved_params["jarak_bulat"], key=f"jarak_{selected_wilayah}"
            )
        
        sisa_eff, dikoreksi = balance_sisa(vol_input, angkut, diolah, sisa_input)
        sisa_calc = max(vol_input - angkut - diolah, 0.0)
        
        if vol_input <= 0:
            st.warning("INPUT harus lebih dari 0 agar rasio dapat dihitung.")
        elif dikoreksi:
            st.info(
                f"Neraca tidak seimbang (selisih {abs(sisa_input - sisa_calc):.2f} m³) — "
                f"SISA dikoreksi otomatis menjadi {sisa_eff:.2f} m³."
            )
        
        fitur = derive_features(vol_input, angkut, diolah, sisa_eff, jarak_bulat)
        
        with st.expander("Fitur Turunan (dihitung otomatis)", expanded=False):
            fitur_values = [
                ("Rasio Angkut", fitur["rasio_angkut"]),
                ("Rasio Diolah", fitur["rasio_diolah"]),
                ("Rasio Sisa", fitur["rasio_sisa"]),
                ("Indeks Jarak", fitur["indeks_jarak"]),
            ]
            fitur_cols = st.columns(4)
            for fcol, (flabel, fval) in zip(fitur_cols, fitur_values):
                fcol.metric(flabel, f"{fval:.3f}")
            st.caption(
                f"Indeks Jarak = TOTAL_JARAK_BULAT / {int(MAX_TOTAL_JARAK_BULAT)} "
                "(maksimum data latih); boleh > 1.0 bila jarak melebihi data latih."
            )
        
        # Tombol Klasifikasi
        if st.button("Klasifikasi dan Prediksi", type="primary", use_container_width=True):
            if vol_input <= 0:
                st.error("INPUT harus lebih dari 0.")
            else:
                try:
                    X = np.array([[
                        fitur["rasio_angkut"],
                        fitur["rasio_diolah"],
                        fitur["rasio_sisa"],
                        fitur["indeks_jarak"],
                    ]])
                    prediksi = model.predict(X)[0]
                    label = str(prediksi).strip().upper()
                    
                    if label not in LABEL_CONFIG:
                        label = "WASPADA"
                    
                    params_raw = {
                        "input": vol_input,
                        "angkut": angkut,
                        "diolah": diolah,
                        "sisa": sisa_eff,
                        "jarak_bulat": jarak_bulat,
                    }
                    
                    st.session_state.wilayah_status[selected_wilayah] = label
                    st.session_state.wilayah_params[selected_wilayah] = params_raw
                    
                    st.session_state.prediction_result = {
                        "label": label,
                        "cfg": LABEL_CONFIG[label],
                        "params": params_raw,
                        "fitur": fitur,
                        "sisa_awal": sisa_input,
                        "sisa_dikoreksi": dikoreksi,
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
            params = res["params"]
            fitur_res = res["fitur"]
            
            st.markdown("---")
            st.markdown("#### Hasil Klasifikasi")

            if label == "KRITIS":
                st.error(f"**STATUS: {label}**")
            elif label == "WASPADA":
                st.warning(f"**STATUS: {label}**")
            else:
                st.success(f"**STATUS: {label}**")

            st.write(cfg["desc"])

            if res.get("sisa_dikoreksi"):
                st.info(
                    f"Nilai SISA dikoreksi dari {res['sisa_awal']:.2f} m\u00b3 "
                    f"menjadi {params['sisa']:.2f} m\u00b3."
                )

            rekomendasi = []
            if fitur_res["rasio_sisa"] > 0.3:
                rekomendasi.append("Rasio sisa tinggi \u2014 tambah frekuensi pengangkutan")
            if fitur_res["rasio_angkut"] < 0.7:
                rekomendasi.append("Rasio angkut rendah \u2014 evaluasi armada")
            if fitur_res["rasio_diolah"] < 0.4:
                rekomendasi.append("Rasio diolah rendah \u2014 tingkatkan kapasitas pengolahan")
            if fitur_res["indeks_jarak"] > 0.7:
                rekomendasi.append("Jarak ke TPA jauh \u2014 optimasi rute")
            if not rekomendasi:
                rekomendasi.append("Semua indikator dalam kondisi baik")

            st.markdown("**Ringkasan Input Mentah:**")
            raw_cols = st.columns(5)
            raw_values = [
                ("INPUT (m\u00b3)", params["input"]),
                ("ANGKUT (m\u00b3)", params["angkut"]),
                ("DIOLAH (m\u00b3)", params["diolah"]),
                ("SISA (m\u00b3)", params["sisa"]),
                ("Jarak (km)", params["jarak_bulat"]),
            ]
            for rcol, (rlab, rval) in zip(raw_cols, raw_values):
                rcol.metric(rlab, f"{rval:.2f}")

            st.markdown("**Fitur Model:**")
            fit_cols = st.columns(4)
            fit_values = [
                ("Rasio Angkut", fitur_res["rasio_angkut"]),
                ("Rasio Diolah", fitur_res["rasio_diolah"]),
                ("Rasio Sisa", fitur_res["rasio_sisa"]),
                ("Indeks Jarak", fitur_res["indeks_jarak"]),
            ]
            for fcol, (flab, fval) in zip(fit_cols, fit_values):
                fcol.metric(flab, f"{fval:.3f}")

            st.markdown("**Rekomendasi:**")
            for r in rekomendasi:
                st.write(f"- {r}")

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
            
            if params:
                f = derive_features(
                    params["input"], params["angkut"], params["diolah"],
                    params["sisa"], params["jarak_bulat"]
                )
                raw_html = (
                    f"INPUT: {params['input']:.2f} m³<br>"
                    f"ANGKUT: {params['angkut']:.2f} m³<br>"
                    f"DIOLAH: {params['diolah']:.2f} m³<br>"
                    f"SISA: {params['sisa']:.2f} m³<br>"
                    f"Jarak: {params['jarak_bulat']:.0f} km"
                )
                fitur_html = (
                    f"Rasio Angkut: {f['rasio_angkut']:.3f}<br>"
                    f"Rasio Diolah: {f['rasio_diolah']:.3f}<br>"
                    f"Rasio Sisa: {f['rasio_sisa']:.3f}<br>"
                    f"Indeks Jarak: {f['indeks_jarak']:.3f}"
                )
            else:
                raw_html = "Input mentah belum diisi"
                fitur_html = "-"
            
            popup_html = f"""
            <div style="min-width: 200px;">
                <b>{wilayah_name}</b><br>
                Status: {status_text}<br>
                <hr>
                {raw_html}
                <hr>
                {fitur_html}
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