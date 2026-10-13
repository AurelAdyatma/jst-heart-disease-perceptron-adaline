# TB1-JST-PrediksiPenyakitJantung

**Tugas Besar 1 – Jaringan Syaraf Tiruan (IT6053)**  
Analisis dan Implementasi Klasifikasi Penyakit Jantung Menggunakan Perceptron dan ADALINE

---

## Identitas Kelompok

| NIM | Nama |
|-----|------|
| 71241076 | Klemens Aurel Adyatma |
| 71241081 | Christopher Edbert Wibowo Siauw |
| 71241087 | Darren Malvino Gunawan |
| 71241090 | Yohanes Septiano |
| 71241122 | Kevin Nathanael Hariyanto |

**Kasus**: Klasifikasi Biner Prediksi Penyakit Jantung  
**Dataset**: UCI Heart Disease Cleveland Dataset

---

## Struktur Repository

```
TB1-JST-PrediksiPenyakitJantung/
|
|-- TB1_PrediksiPenyakitJantung.ipynb   # Notebook utama (kode + penjelasan + output)
|-- TB1_PrediksiPenyakitJantung.py      # Script Python mandiri
|-- TB1_PrediksiPenyakitJantung.pdf     # Laporan PDF (5-8 halaman)
|-- Presentasi_TB1.pdf                  # Slide presentasi
|-- dataset.csv                         # Dataset yang digunakan (297 baris, 14 kolom)
|-- requirements.txt                    # Dependensi Python
|-- README.md                           # File ini
|-- create_dataset.py                   # Script download dataset
|-- plot_perceptron_curves.png          # Kurva misklasifikasi Perceptron per epoch
|-- plot_adaline_curves.png             # Kurva loss J ADALINE per epoch
|-- plot_confusion_matrix.png           # Confusion matrix semua konfigurasi
|-- plot_accuracy_comparison.png        # Perbandingan akurasi latih vs uji
|-- plot_decision_boundary.png          # Decision boundary (proyeksi PCA)
```

---

## Dataset

**Sumber**: [UCI Machine Learning Repository – Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease)  
**DOI**: https://doi.org/10.24432/C52P4X  
**Referensi Kaggle**: https://www.kaggle.com/datasets/aavigan/cleveland-clinic-heart-disease-dataset

### Deskripsi Kolom

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| age | Numerik | Usia pasien (tahun) |
| sex | Biner | Jenis kelamin (1=pria, 0=wanita) |
| cp | Kategorikal | Tipe nyeri dada (1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic) |
| trestbps | Numerik | Tekanan darah istirahat (mmHg) |
| chol | Numerik | Kolesterol serum (mg/dl) |
| fbs | Biner | Gula darah puasa > 120 mg/dl (1=benar, 0=salah) |
| restecg | Kategorikal | Hasil EKG istirahat (0=normal, 1=ST-T wave abnormality, 2=LV hypertrophy) |
| thalach | Numerik | Detak jantung maksimum yang dicapai |
| exang | Biner | Angina akibat olahraga (1=ya, 0=tidak) |
| oldpeak | Numerik | Depresi ST akibat olahraga relatif terhadap istirahat |
| slope | Kategorikal | Kemiringan puncak segmen ST olahraga (1=upsloping, 2=flat, 3=downsloping) |
| ca | Numerik | Jumlah pembuluh mayor yang terfluoroskopi (0-3) |
| thal | Kategorikal | Thalassemia (3=normal, 6=fixed defect, 7=reversible defect) |
| **target** | **Label** | **0=tidak terindikasi, 1=terindikasi penyakit jantung** |

**Setelah cleaning (hapus missing values)**: 297 sampel  
**Distribusi kelas**: 160 tidak sakit (53.9%), 137 sakit (46.1%)  
**Label model**: -1 = tidak sakit, +1 = sakit (kelas positif)

---

## Instalasi dan Cara Menjalankan

### Prasyarat
- Python 3.12+
- pip

### Langkah 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/TB1-JST-PrediksiPenyakitJantung.git
cd TB1-JST-PrediksiPenyakitJantung
```

### Langkah 2: Install Dependensi
```bash
pip install -r requirements.txt
```

### Langkah 3: Jalankan Notebook
```bash
jupyter notebook TB1_PrediksiPenyakitJantung.ipynb
```

### Langkah 4 (opsional): Jalankan Script Python Mandiri
```bash
python TB1_PrediksiPenyakitJantung.py
```

### Versi Python dan Seed
- **Python**: 3.12
- **Random Seed**: 42 (digunakan di semua langkah)
- **numpy**: >= 1.24.0
- **pandas**: >= 2.0.0
- **scikit-learn**: >= 1.3.0
- **matplotlib**: >= 3.7.0

---

## Ringkasan Hasil Eksperimen

| No | Model | eta | Std | Acc Latih | Acc Uji | Loss/Err Akhir | P | R | F1 |
|----|-------|-----|-----|-----------|---------|----------------|---|---|-----|
| 1 | Perceptron | 0.001 | Ya | 0.7862 | 0.8167 | 60 misklas | 0.7879 | 0.8214 | 0.8043 |
| 2 | Perceptron | 0.010 | Ya | 0.7862 | 0.8167 | 60 misklas | 0.7879 | 0.8214 | 0.8043 |
| 3 | Perceptron | 0.100 | Ya | 0.7862 | 0.8167 | 60 misklas | 0.7879 | 0.8214 | 0.8043 |
| 4 | Perceptron | 0.010 | Tidak | 0.5253 | 0.5167 | 88 misklas | - | - | - |
| 5 | ADALINE | 0.001 | Ya | 0.7862 | 0.8333 | 0.4407 J | 0.8000 | 0.8571 | 0.8276 |
| 6 | ADALINE | 0.010 | Ya | 0.8312 | **0.8667** | 0.2696 J | **0.8846** | 0.8214 | **0.8519** |
| 7 | ADALINE | 0.100 | Ya | 0.8523 | 0.8500 | 0.2439 J | 0.8800 | 0.7857 | 0.8302 |
| 8 | ADALINE | 0.010 | Tidak | *(divergen)* | 0.4667 | *NaN* | - | - | - |

**Konfigurasi terbaik**: ADALINE, eta=0.01, standardisasi=Ya (Acc uji=86.7%, F1=0.8519)

---

## Referensi

1. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386–408. https://doi.org/10.1037/h0042519

2. Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. *IRE WESCON Convention Record, Part 4*, 96–104. https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf

3. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

4. Raschka, S. (n.d.). Adaline: Adaptive Linear Neuron Classifier. *mlxtend Documentation*. https://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/

---

## Bantuan AI

Antigravity AI (Google DeepMind) digunakan untuk:
- Membantu debugging masalah encoding Windows
- Menyempurnakan dokumentasi docstring kode
- Menghasilkan kerangka laporan

Seluruh logika implementasi Perceptron dan ADALINE ditulis secara mandiri sesuai aturan pembaruan dalam panduan tugas. Semua keputusan desain (split ratio, eta, epoch) ditentukan oleh tim.

---

## Kontribusi Anggota

| NIM | Nama | Kontribusi |
|-----|------|-----------|
| 71241076 | Klemens Aurel Adyatma | Koordinasi, analisis perbandingan, laporan |
| 71241081 | Christopher Edbert Wibowo Siauw | Implementasi Perceptron, perhitungan manual |
| 71241087 | Darren Malvino Gunawan | Implementasi ADALINE, visualisasi |
| 71241090 | Yohanes Septiano | Preprocessing data, evaluasi metrik |
| 71241122 | Kevin Nathanael Hariyanto | Rancangan eksperimen, slide presentasi |
