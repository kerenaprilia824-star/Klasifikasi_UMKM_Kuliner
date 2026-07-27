
import streamlit as st
import pandas as pd
from pathlib import Path
from joblib import load

# Tetapkan urutan fitur persis seperti yang diminta agar UI mengikuti susunan ini
FEATURE_COLS = [
    "pendidikan", "lama_usaha", "lokasi_usaha", "media_penjualan", "hari_berjualan",
    "jumlah_karyawan", "omzet_bulanan", "transaksi_per_hari", "modal_awal",
    "pencatatan_keuangan", "sumber_modal", "penggunaan_teknologi", "memiliki_izin",
    "aktif_promosi_medsos", "memiliki_target", "pernah_pelatihan",
]

# Page config
st.set_page_config(page_title="Klasifikasi UMKM", page_icon="🏪")

# Load model
@st.cache_resource
def load_model(model_name):
    """Load model dari file lokal"""
    # Cek beberapa lokasi: file di root proyek, lalu di results/metrics
    candidates = [Path(f"{model_name}.pkl"), Path("results") / "metrics" / f"{model_name}.pkl"]
    for model_path in candidates:
        if model_path.exists():
            return load(model_path)

    st.error(
        f"Model {model_name}.pkl tidak ditemukan. Dicari di: {', '.join(str(p) for p in candidates)}"
    )
    return None

# Sidebar
st.sidebar.header("⚙️ Model")
model_choice = st.sidebar.radio("Pilih Model:",
    ("Random Forest", "Decision Tree"))

model_name = "model_random_forest" if "Random Forest" in model_choice else "model_decision_tree"
model = load_model(model_name)

if model is None:
    st.stop()

# Title
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0.25em;'>Klasifikasi Tingkat Pertumbuhan UMKM Kuliner di Kota Manado</h1>",
    unsafe_allow_html=True,
)
st.markdown("Masukkan data UMKM Anda untuk mengetahui klasifikasi tingkat pertumbuhan UMKM Kuliner di Kota Manado")

# Input form
st.header("📝 Data UMKM")

field_options = {
    'pendidikan': ("Pendidikan", ["SD / SMP", "SMA / SMK", "Diploma (D1/D2/D3)", "Sarjana (S1) atau lebih tinggi"]),
    'lama_usaha': ("Lama Usaha", ["Kurang dari 6 bulan", "6 bulan - 1 tahun", "1 - 3 tahun", "Lebih dari 3 tahun"]),
    'lokasi_usaha': ("Lokasi", ["Warung / kios / ruko permanen", "Stand / booth / gerobak", "Rumahan (diantar / diambil sendiri)", "Titipan di kantin / toko lain", "Online (tanpa lokasi fisik tetap)"]),
    'media_penjualan': ("Media Penjualan", ["Offline saja (tatap muka langsung)", "WhatsApp / Instagram / TikTok / Facebook", "GoFood / ShopeeFood / GrabFood", "Marketplace (Tokopedia, Shopee, dll)"]),
    'hari_berjualan': ("Hari Berjualan", ["1 - 2 hari", "3 - 4 hari", "5 - 6 hari", "Setiap hari (7 hari)"]),
    'jumlah_karyawan': ("Jumlah Karyawan", ["Hanya saya sendiri", "2 - 3 orang", "4 - 5 orang", "Lebih dari 5 orang"]),
    'omzet_bulanan': ("Omzet Bulanan", ["Kurang dari Rp 1.000.000", "Rp 1.000.000 - Rp 5.000.000", "Lebih dari Rp 5.000.000"]),
    'transaksi_per_hari': ("Transaksi/Hari", ["1 - 5 transaksi", "6 - 15 transaksi", "16 - 30 transaksi", "Lebih dari 30 transaksi"]),
    'modal_awal': ("Modal Awal", ["Kurang dari Rp 500.000", "Rp 500.000 - Rp 2.000.000", "Rp 2.000.000 - Rp 5.000.000", "Rp 5.000.000 - Rp 10.000.000"]),
    'pencatatan_keuangan': ("Pencatatan Keuangan", ["Tidak pernah mencatat", "Kadang-kadang mencatat", "Selalu mencatat (secara manual / buku)", "Selalu mencatat (menggunakan aplikasi / digital)"]),
    'sumber_modal': ("Sumber Modal", ["Modal sendiri / tabungan pribadi", "Bantuan keluarga / orang tua"]),
    'penggunaan_teknologi': ("Penggunaan Teknologi", ["Tidak menggunakan aplikasi apapun", "Aplikasi chat bisnis (WhatsApp Business)", "Platform pesan antar online (GoFood, ShopeeFood, dll)", "Media sosial untuk promosi (Instagram, TikTok, Facebook)", "Aplikasi pembukuan / keuangan (BukuKas, BukuWarung, dll)"]),
    'memiliki_izin': ("Memiliki Izin", ["TIDAK", "YA"]),
    'aktif_promosi_medsos': ("Promosi Media Sosial", ["1 - Sangat Tidak Setuju", "2 - Tidak Setuju", "3 - Netral", "4 - Setuju", "5 - Sangat Setuju"]),
    'memiliki_target': ("Memiliki Target Penjualan", ["1 - Sangat Tidak Setuju", "2 - Tidak Setuju", "3 - Netral", "4 - Setuju", "5 - Sangat Setuju"]),
    'pernah_pelatihan': ("Pernah Pelatihan Wirausaha", ["1 - Sangat Tidak Setuju", "2 - Tidak Setuju", "3 - Netral", "4 - Setuju", "5 - Sangat Setuju"]),
}

# Helper untuk membuat selectbox dengan opsi numerik yang sesuai
def make_select(key):
    label, opts = field_options[key]
    # tambahkan placeholder pertama supaya tidak ada pilihan default
    choices = [f"Pilih {label}..."] + opts
    return st.selectbox(label, list(range(len(choices))), format_func=lambda i, _choices=choices: _choices[i])

# Helper untuk mendapatkan feature importance dari model
def get_feature_importance(model, top_n=5):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    top_importances = importances.sort_values(ascending=False).head(top_n)
    return top_importances

# Render fields dua per baris mengikuti urutan FEATURE_COLS, pastikan 'pernah_pelatihan' terakhir
widgets = {}
for i in range(0, len(FEATURE_COLS), 2):
    left = FEATURE_COLS[i]
    right = FEATURE_COLS[i+1] if i+1 < len(FEATURE_COLS) else None
    if right:
        c1, c2 = st.columns(2)
        with c1:
            widgets[left] = make_select(left)
        with c2:
            widgets[right] = make_select(right)
    else:
        # terakhir (odd count) — tampil full width
        widgets[left] = make_select(left)

# Predict button
if st.button("🔮 PREDIKSI", type="primary", use_container_width=True):
    # Prepare data from widgets in the same order as FEATURE_COLS
    values = []
    for f in FEATURE_COLS:
        sel = widgets[f]
        mapped = sel - 1  # because widget index 0 = placeholder
        if mapped < 0:
            st.error(f"Silakan pilih nilai untuk '{field_options[f][0]}' sebelum memprediksi.")
            st.stop()
        values.append(mapped)

    X = pd.DataFrame([values], columns=FEATURE_COLS)

    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Results
    st.success("Prediksi Berhasil!")
    st.header("📊 Hasil Prediksi")

    class_names = ["Rendah", "Sedang", "Tinggi"]
    pred_label = class_names[prediction]

    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("Tingkat Adopsi", pred_label, f"{probabilities[prediction]:.1%}")

    with col2:
        colors = {"Rendah": "🔴", "Sedang": "🟠", "Tinggi": "🟢"}
        st.markdown(f"### {colors[pred_label]} {pred_label}")

    # Tingkat Keyakinan Model - Versi ramah untuk orang awam
    st.subheader("🎯 Tingkat Keyakinan Model")
    st.write("Berikut adalah seberapa yakin model terhadap setiap kategori hasil klasifikasi:")

    for name, prob in zip(class_names, probabilities):
        if prob >= 0.5:
            emoji = "✅"
            interpretation = "Sangat mungkin"
        elif prob >= 0.3:
            emoji = "⚠️"
            interpretation = "Mungkin"
        else:
            emoji = "❌"
            interpretation = "Tidak kemungkinan besar"

        col_prob1, col_prob2 = st.columns([2, 3])
        with col_prob1:
            st.write(f"**{name}**")
        with col_prob2:
            st.progress(prob)
            st.caption(f"{prob:.0%} {emoji} {interpretation}")

    # Faktor Paling Berpengaruh - Versi ramah untuk orang awam
    st.subheader("Faktor-Faktor Kunci yang Mempengaruhi Hasil Ini")
    st.write("Model mengidentifikasi faktor-faktor di bawah ini sebagai yang paling penting dalam menentukan hasil klasifikasi Anda:")

    factor_explanations = {
        "pendidikan": "Pendidikan mempengaruhi pemahaman terhadap teknologi dan manajemen bisnis",
        "lama_usaha": "Semakin lama usaha berjalan, semakin banyak pengalaman yang dimiliki",
        "lokasi_usaha": "Lokasi yang strategis sangat penting untuk aksesibilitas pelanggan",
        "media_penjualan": "Semakin banyak channel penjualan, semakin besar potensi menjangkau pelanggan",
        "hari_berjualan": "Frekuensi berjualan mempengaruhi konsistensi dan omzet harian",
        "jumlah_karyawan": "Lebih banyak karyawan biasanya terkait dengan skala bisnis yang lebih besar",
        "omzet_bulanan": "Omzet saat ini adalah indikator kesehatan bisnis Anda",
        "transaksi_per_hari": "Jumlah transaksi menunjukkan seberapa banyak pelanggan tertarik pada produk Anda",
        "modal_awal": "Modal yang lebih besar memungkinkan investasi lebih baik di awal",
        "pencatatan_keuangan": "Pencatatan yang baik membantu kontrol keuangan dan perencanaan",
        "sumber_modal": "Sumber modal mempengaruhi kualitas dan kelancaran bisnis",
        "penggunaan_teknologi": "Teknologi digital adalah kunci untuk memperluas jangkauan dan efisiensi",
        "memiliki_izin": "Izin resmi menunjukkan kredibilitas dan legalitas bisnis",
        "aktif_promosi_medsos": "Promosi aktif di media sosial sangat penting untuk menarik pelanggan baru",
        "memiliki_target": "Memiliki target penjualan membantu fokus dan motivasi bisnis",
        "pernah_pelatihan": "Pelatihan meningkatkan keterampilan dan pengetahuan dalam menjalankan bisnis",
    }

    importance = get_feature_importance(model, top_n=5)
    if importance is not None:
        st.write("\n**Faktor-faktor terpenting (Top 5):**\n")
        for i, (feature, score) in enumerate(importance.items(), 1):
            explanation = factor_explanations.get(feature, "Faktor penting untuk klasifikasi")
            st.write(f"**{i}. {feature.replace('_', ' ').title()}**")
            st.write(f"*{explanation}*")
            st.progress(min(score / importance.max(), 1.0))
            st.write("")  # Spasi

    # Interpretation
    interpretations = {
        "Rendah": "UMKM perlu pelatihan dasar teknologi",
        "Sedang": "Sudah baik, bisa tingkatkan ke level lanjut",
        "Tinggi": "Sudah maju, siap untuk inovasi"
    }
    st.info(f"💡 **Rekomendasi**: {interpretations[pred_label]}")

# Footer
st.markdown("---")
st.caption("Made with Streamlit | Model: Random Forest & Decision Tree | Data: UMKM Manado")
