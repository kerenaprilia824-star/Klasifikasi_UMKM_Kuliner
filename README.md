---
title: Prediksi Adopsi Teknologi UMKM
emoji: 🏪
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.48.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

## 🍜 Klasifikasi Tingkat Pertumbuhan UMKM Kuliner di Kota Manado

### Perbandingan Algoritma Random Forest dan Decision Tree dengan Pendekatan Explainable AI (XAI)

---

## 📋 Deskripsi Proyek

Proyek ini merupakan implementasi penelitian skripsi yang membandingkan performa algoritma **Random Forest** dan **Decision Tree** dalam mengklasifikasikan tingkat pertumbuhan UMKM kuliner di Kota Manado. Penelitian menggunakan pendekatan **Explainable AI (XAI)** dengan metode SHAP dan LIME untuk menginterpretasikan hasil model.

---

## 🗂️ Struktur Folder

```text
skripsi_umkm_kuliner_manado/
│
├── 📁 data/
│   ├── raw/              → Data mentah (hasil survei/kuesioner)
│   ├── processed/        → Data setelah preprocessing
│   └── external/         → Data pendukung (BPS, Dinas Koperasi, dll)
│
├── 📁 notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_decision_tree.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 05_perbandingan_model.ipynb
│   └── 06_explainability_xai.ipynb
│
├── 📁 src/
│   ├── preprocessing/    → Script pembersihan & transformasi data
│   ├── models/           → Script training model
│   ├── explainability/   → Script SHAP & LIME
│   ├── visualization/    → Script pembuatan grafik
│   └── utils/            → Fungsi-fungsi pembantu
│
├── 📁 results/
│   ├── figures/          → Grafik & visualisasi (PNG, SVG)
│   ├── metrics/          → Hasil evaluasi model (CSV, JSON)
│   └── reports/          → Laporan ringkasan
│
├── 📁 docs/              → Dokumentasi tambahan
├── 📁 tests/             → Unit testing
│
├── requirements.txt      → Daftar library Python
├── .env.example          → Template variabel environment
├── .gitignore
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Clone / Buka Proyek

```bash
cd skripsi_umkm_kuliner_manado
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Jupyter Notebook

```bash
jupyter notebook
```

### 5. Urutan Pengerjaan Notebook

1. `01_exploratory_data_analysis.ipynb` → Eksplorasi data awal
2. `02_preprocessing.ipynb` → Pembersihan & persiapan data
3. `03_decision_tree.ipynb` → Pemodelan Decision Tree
4. `04_random_forest.ipynb` → Pemodelan Random Forest
5. `05_perbandingan_model.ipynb` → Perbandingan kedua model
6. `06_explainability_xai.ipynb` → Analisis XAI (SHAP & LIME)

---

## 🎯 Label Klasifikasi

| Label | Kategori | Keterangan |
| --- | --- | --- |
| 0 | Rendah | UMKM dengan pertumbuhan stagnan/menurun |
| 1 | Sedang | UMKM dengan pertumbuhan moderat |
| 2 | Tinggi | UMKM dengan pertumbuhan pesat |

---

## 📊 Fitur (Variabel) yang Digunakan

| Fitur | Keterangan |
| --- | --- |
| `omzet_bulanan` | Rata-rata omzet per bulan (Rp) |
| `jumlah_karyawan` | Jumlah tenaga kerja |
| `lama_usaha` | Lama usaha berjalan (tahun) |
| `jenis_produk` | Jenis kuliner yang dijual |
| `lokasi_usaha` | Kecamatan lokasi usaha |
| `media_pemasaran` | Penggunaan media sosial/online |
| `modal_awal` | Modal awal pendirian (Rp) |
| `akses_pinjaman` | Akses ke kredit/pinjaman usaha |
| `pelatihan_usaha` | Pernah ikut pelatihan UMKM |
| `tingkat_pendidikan` | Pendidikan terakhir pemilik |

---

## 🧠 Metode Explainability

- **SHAP (SHapley Additive exPlanations)**: Mengukur kontribusi setiap fitur terhadap prediksi model secara global dan lokal.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Menjelaskan prediksi individual dengan model surrogate yang sederhana.

---

## 📈 Metrik Evaluasi

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve
- Cross-Validation Score

---
