# Laporan Tugas Besar 1 – Jaringan Syaraf Tiruan (IT6053)
**Analisis dan Implementasi Klasifikasi Data Menggunakan Perceptron dan ADALINE**

**Kelompok:**
1. 71241076 – Klemens Aurel Adyatma
2. 71241081 – Christopher Edbert Wibowo Siauw
3. 71241087 – Darren Malvino Gunawan
4. 71241090 – Yohanes Septiano
5. 71241122 – Kevin Nathanael Hariyanto

---

## 1. Masalah dan Tujuan
Penyakit jantung merupakan salah satu penyebab utama kematian di dunia. Deteksi dini yang akurat sangat penting untuk mencegah komplikasi dan memberikan perawatan yang tepat. Masalah ini diformulasikan sebagai *supervised learning* (pembelajaran terawasi) dengan tujuan membangun model Jaringan Syaraf Tiruan sederhana (Perceptron dan ADALINE) yang dapat memprediksi klasifikasi biner pasien (sakit jantung atau sehat) berdasarkan data metrik kesehatan klinis mereka.

## 2. Dataset serta Kaitannya dengan Tugas 1
Pada Tugas 1, kelompok telah mengidentifikasi bahwa memprediksi penyakit jantung membutuhkan banyak parameter klinis (seperti tekanan darah, kolesterol, usia, dsb). Pada Tugas Besar 1 ini, analisis tersebut dilanjutkan menggunakan **Cleveland Heart Disease Dataset** dari *UCI Machine Learning Repository*. 

*   **Izin dan Sumber**: Dataset bersifat *Public Domain/Open Source* untuk tujuan riset akademis, disediakan oleh UCI Machine Learning Repository.
*   **Jumlah Observasi**: 303 sampel asli. Setelah menghapus sampel dengan *missing values*, tersisa **297 observasi** valid.
*   **Fitur**: 13 fitur numerik dan kategorikal yang telah dikonversi ke numerik: usia (`age`), jenis kelamin (`sex`), tipe nyeri dada (`cp`), tekanan darah istirahat (`trestbps`), kolesterol (`chol`), gula darah puasa (`fbs`), EKG (`restecg`), detak jantung maksimal (`thalach`), angina (`exang`), depresi ST (`oldpeak`), kemiringan ST (`slope`), jumlah pembuluh (`ca`), thalasemia (`thal`).
*   **Label Target**: Dua kelas yaitu **-1 (Kelas Negatif)** untuk pasien tidak terindikasi penyakit jantung, dan **+1 (Kelas Positif)** untuk pasien yang terindikasi memiliki penyakit jantung.
*   **Distribusi Kelas**: 160 sampel kelas -1 (53.9%) dan 137 sampel kelas +1 (46.1%). Distribusi cukup seimbang.

## 3. Artikel dan Kode Pendukung
*   **Artikel Ilmiah Utama (Perceptron)**: Rosenblatt, F. (1958). *The perceptron*. Artikel ini menjadi referensi landasan teori mengenai aturan *hard-threshold* dan pembaruan penambahan matriks.
*   **Artikel Ilmiah Utama (ADALINE)**: Widrow, B., & Hoff, M. E. (1960). *Adaptive switching circuits*. Menjadi rujukan utama konsep pembaruan bobot berbasis fungsi *linear* dan fungsi biaya *Gradient Descent*.
*   **Kode Pendukung**: *Raschka, S. (mlxtend)*. Dokumentasi *Adaline Classifier* digunakan sebagai rujukan logika penulisan implementasi *Batch Gradient Descent* di Python (murni NumPy). Kontribusinya ada pada konseptualisasi array NumPy, namun implementasi kode tugas ini ditulis ulang sepenuhnya secara mandiri (from scratch).
*   **Dataset Sumber**: Janosi, A., dkk. (1988). *Heart Disease Dataset*. (DOI: 10.24432/C52P4X).
*   **Bantuan AI**: Antigravity AI (Google) digunakan sebagai asisten kopilot untuk mendebug kendala *encoding* UTF-8 pada *console* Windows, serta merapikan komentar struktur sintaks kode Python agar selaras dengan spesifikasi algoritma dari rubrik tanpa mengambil alih logika matematika mandiri.

## 4. Model Neuron dan Aturan Pembelajaran
Kedua model menggunakan dasar persamaan matematika neuron yang sama:
*   **Net Input**: $z = \mathbf{w}^T \mathbf{x} + b$
*   **Prediksi Kelas**: $\hat{y} = +1$ jika $z \geq 0$, dan $\hat{y} = -1$ jika $z < 0$.

**A. Perceptron**
*   **Fungsi Objektif**: Meminimalkan jumlah kesalahan klasifikasi.
*   **Aturan Pembaruan**: Pembaruan terjadi **hanya ketika model salah memprediksi** kelas.
*   **Error**: $e = y - \hat{y}$ (selisih keluaran ambang/threshold).
*   **Update**: $\Delta = \eta (y - \hat{y})$. Bobot diperbarui menjadi $\mathbf{w} \leftarrow \mathbf{w} + \Delta\mathbf{x}$, dan bias $b \leftarrow b + \Delta$.

**B. ADALINE (Adaptive Linear Neuron)**
*   **Fungsi Objektif**: Meminimalkan jarak selisih rata-rata (Loss J) menggunakan *Batch Gradient Descent*. $J = \frac{1}{2n} \sum_{i=1}^{n} (y_i - z_i)^2$.
*   **Aturan Pembaruan**: Pembaruan dihitung menggunakan seluruh data latih setiap epoch.
*   **Error**: $e = y - z$ (selisih dengan keluaran **linear**).
*   **Update**: $\mathbf{w} \leftarrow \mathbf{w} + \eta \frac{\mathbf{X}^T \mathbf{e}}{n}$, dan $b \leftarrow b + \eta \cdot \text{mean}(e)$.

## 5. Penyiapan Data serta Rancangan Eksperimen
*   **Pembagian Data (Split)**: Data diacak dan dibagi dengan proporsi **80% Latih (237 sampel)** dan **20% Uji (60 sampel)**. Pembagian menggunakan teknik *stratified* (distribusi kelas terjaga) dengan *random seed = 42* agar hasil keterulangan pasti (*reproducible*).
*   **Standardisasi**: Fitur numerik distandarisasi (Z-score, *mean* = 0, *std* = 1). Parameter *scaling* dihitung (di-fit) murni **hanya dari data latih** untuk mencegah *data leakage*, kemudian diaplikasikan (transform) ke data latih dan data uji.
*   **Inisialisasi**: Semua parameter bobot ($\mathbf{w}$) dan bias ($b$) selalu diinisialisasi dari $0.0$.
*   **Rancangan 8 Konfigurasi Minimum (Maksimal 50 Epoch):**
    1. Perceptron, $\eta=0.001$, Standardisasi = Ya
    2. Perceptron, $\eta=0.010$, Standardisasi = Ya
    3. Perceptron, $\eta=0.100$, Standardisasi = Ya
    4. Perceptron, $\eta=0.010$, Standardisasi = Tidak
    5. ADALINE, $\eta=0.001$, Standardisasi = Ya
    6. ADALINE, $\eta=0.010$, Standardisasi = Ya
    7. ADALINE, $\eta=0.100$, Standardisasi = Ya
    8. ADALINE, $\eta=0.010$, Standardisasi = Tidak

## 6. Hasil
Berikut adalah tabel pelaporan hasil dari data uji setelah 50 epoch pelatihan:

| No | Model | $\eta$ | Std | Acc Latih | Acc Uji | Loss/Err Akhir | Precision | Recall | F1 Score |
|----|-------|-------|-----|-----------|---------|----------------|-----------|--------|----------|
| 1 | Perceptron | 0.001 | Ya | 0.7862 | 0.8167 | 60 misklas | 0.7879 | 0.8214 | 0.8043 |
| 2 | Perceptron | 0.010 | Ya | 0.7862 | 0.8167 | 60 misklas | 0.7879 | 0.8214 | 0.8043 |
| 3 | Perceptron | 0.100 | Ya | 0.7862 | 0.8167 | 60 misklas | 0.7879 | 0.8214 | 0.8043 |
| 4 | Perceptron | 0.010 | Tidak | 0.5253 | 0.5167 | 88 misklas | - | - | - |
| 5 | ADALINE | 0.001 | Ya | 0.7862 | 0.8333 | 0.4407 (Loss J) | 0.8000 | 0.8571 | 0.8276 |
| 6 | ADALINE | 0.010 | Ya | 0.8312 | **0.8667** | 0.2696 (Loss J) | **0.8846** | 0.8214 | **0.8519** |
| 7 | ADALINE | 0.100 | Ya | 0.8523 | 0.8500 | 0.2439 (Loss J) | 0.8800 | 0.7857 | 0.8302 |
| 8 | ADALINE | 0.010 | Tidak | Divergen | 0.4667 | NaN | - | - | - |

*(Visualisasi matriks kebingungan, batas keputusan PCA, kurva J, dan kurva salah klasifikasi dapat dilihat di repositori dalam format .PNG / IPython Notebook)*.

Kondisi Epoch (10, 30, 50) untuk Konfigurasi 1, 2, 3 (Perceptron): Semua jumlah misklasifikasi pada epoch 10 adalah 62, pada epoch 30 adalah 55, dan pada epoch 50 adalah 60.

Bobot dan Bias Akhir Konfigurasi Terbaik (ADALINE, $\eta=0.01$, Std=Ya):
`b_akhir = -0.031666`
`w[age]=0.0398, w[sex]=0.0766, w[cp]=0.1127, w[trestbps]=0.0436, w[chol]=0.0220, w[fbs]=-0.0078, w[restecg]=0.0489, w[thalach]=-0.0860, w[exang]=0.1026, w[oldpeak]=0.1065, w[slope]=0.0837, w[ca]=0.1212, w[thal]=0.1529`.

## 7. Pembahasan dan Keterbatasan

*   **Apakah data dapat dipisahkan secara linear?** 
    Tidak. Kurva pelatihan Perceptron tidak pernah menyentuh 0 salah klasifikasi hingga epoch ke-50 (mengambang pada rentang 55-62 kesalahan per epoch dari 237 data latih). Selain itu, fungsi Loss J dari ADALINE juga tidak turun sampai angka nol sempurna (terhenti di 0.24). Hal ini menegaskan bahwa terdapat batas-batas di mana data positif dan negatif saling tumpang tindih (*overlap*) secara ruang (*non-linearly separable*).
*   **Perbedaan penggunaan Error pada Bobot:**
    Perceptron bergantung murni pada *Hard Threshold* ($y - \hat{y}$). Jika sampel sudah benar diprediksi, jaringan merasa "puas" dan pembaruan bobot dihentikan untuk sampel tersebut, meskipun sebenarnya jarak sampel tersebut sangat dekat dengan *Decision Boundary* (kurang *robust*). Sebaliknya, ADALINE terus mengevaluasi fungsi linier. ADALINE akan tetap memodifikasi margin dan merapikan hiperbidang bobot guna memperkecil rata-rata loss keseluruhan sekalipun akurasi latih tidak berubah banyak.
*   **Pengaruh Standardisasi:**
    Standardisasi mutlak diperlukan untuk ADALINE dalam dataset rekam medis. Fitur klinis seperti *Cholesterol* dapat bernilai 300+, sedangkan *sex* hanya bernilai 0 dan 1. Tanpa standardisasi, gradien dari fitur besar langsung meledak (*exploding gradients*), menyebabkan ADALINE divergen ke Loss `NaN` bahkan pada epoch ke-1 (Eksperimen 8).
*   **Fenomena $\eta$ pada Perceptron (Akurasi Sama = Proses Belajar Sama?):**
    Hasil Eksperimen 1, 2, dan 3 menghasilkan urutan misklasifikasi dan performa metrik yang *tepat sama*. Jika bobot diinisialisasi pada titik $0$ dengan data set urutan absolut sama, kenaikan nilai $\eta$ yang merupakan suatu tetapan positif linier hanya akan memperbesar *skala absolut matriks (scaling)* batas hiperbidang tanpa merubah arah *vektor normal*. Oleh karena itu, rasio prediksinya konstan sama. Ini berarti proses belajar esensinya sama meskipun skala bobot (yang tercetak di memori) berbeda secara proporsional, membuktikan teori konvergensi *Novikoff*.
*   **Keterbatasan:**
    Pendekatan JST 1 *layer* hanya dapat menghasilkan solusi fungsi klasifikasi garis datar (linear). Hal ini tidak cocok untuk struktur medis biologis yang memiliki variabel perancu silang tingkat lanjut. Evaluasi uji 60 sampel juga dianggap terlalu sedikit sehingga rentan pada *variance error*.

## 8. Kesimpulan
Klasifikasi dataset Cleveland menggunakan Jaringan Syaraf Tiruan menunjukkan bahwa model ADALINE dengan pendekatan *Batch Gradient Descent* terbukti lebih superior, *robust*, dan konvergen (Akurasi Uji: 86.67%, F1 Score: 85.19%) dibanding model Perceptron *Threshold* (Akurasi Uji: 81.67%) untuk jenis data *non-linearly separable*. Standardisasi (Z-Score) mutlak wajib diimplementasikan untuk mencegah ledakan gradien. Meskipun terbilang konvergen dalam menurunkan fungsi J, kedua model dibatasi oleh kemampuan mendeteksi keputusan *linear* belaka, sehingga pengembangan *Multi-Layer Perceptron (MLP)* disarankan untuk tahap riset diagnosis kardiovaskular berikutnya.

## 9. Daftar Pustaka
1. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386–408. https://doi.org/10.1037/h0042519
2. Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. *IRE WESCON Convention Record, Part 4*, 96–104. https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf
3. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease Dataset. *UCI Machine Learning Repository*. https://doi.org/10.24432/C52P4X
4. Raschka, S. (tanpa tahun). Adaline: Adaptive Linear Neuron Classifier. Dokumentasi *mlxtend*. https://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/

---
**LAMPIRAN: Rincian Perhitungan Manual (2 Langkah)**

*(Catatan: Ini adalah dua iterasi pembaruan pertama dari kode Python)*
**1. Perceptron ($\eta=0.01$)**
*   Inisialisasi: $\mathbf{w} = 0, b = 0$
*   Langkah 1 (Sampel 0, kelas $y=1$): Net Input $z = 0$. Karena $z \geq 0$, diprediksi $\hat{y} = 1$. Karena $\hat{y} = y$, Error $= 0$. Tidak ada pembaruan bobot.
*   Langkah 2 (Sampel 1, kelas $y=-1$): Net Input $z = 0$. Diprediksi $\hat{y} = 1$. Salah klasifikasi. Error $= \eta (y - \hat{y}) = 0.01(-1 - 1) = -0.02$.
    *Update*: $w_{baru} = w + \Delta \mathbf{x}$, dan $b_{baru} = b - 0.02$.

**2. ADALINE ($\eta=0.01$, 5 sampel pertama sebagai ilustrasi batch)**
*   Inisialisasi: $\mathbf{w} = 0, b = 0$
*   Iterasi Batch 1: Seluruh $z = 0$, $e = y - z = y$. Loss $J$ awal $= 0.5$.
    Bobot diupdate dari rata-rata $X^T e$ 5 sampel. $\mathbf{w}$ bergeser dari $0$ menjadi matriks bernilai pecahan desimal, bias $b_{baru}$ di-update dari mean error sampel.
*   Iterasi Batch 2: Evaluasi $z$ dari bobot $\mathbf{w}_{baru}$. Error $e$ mengecil. Terjadi reduksi fungsi objektif $J$ (Gradient Descent).
*(Perhitungan skalar desimal panjang dilampirkan terpisah pada file Jupyter Notebook kelompok).*
