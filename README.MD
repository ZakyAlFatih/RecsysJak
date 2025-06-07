# Laporan Proyek Machine Learning - Zaky Al Fatih Nata Imam

---

## Domain Project

Proyek ini bertujuan membangun sistem rekomendasi film menggunakan dataset (https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) dari Kaggle. Latar belakangnya adalah bertambahnya jumlah film yang tersedia, sehingga pengguna kesulitan menemukan film sesuai preferensi mereka. Sistem rekomendasi membantu pengguna menemukan film baru yang relevan berdasarkan riwayat tontonan atau kesamaan karakteristik film.

Tanpa sistem rekomendasi, pengguna akan kesulitan mencari film secara manual atau melewatkan film berkualitas. Sistem ini meningkatkan pengalaman menonton, kepuasan pengguna, dan engagement platform penyedia film.

Dua pendekatan utama dalam sistem rekomendasi yang digunakan adalah:

- **Collaborative Filtering**: merekomendasikan film berdasarkan preferensi pengguna lain yang memiliki selera serupa (Sarwar et al., 2001).
- **Content-Based Filtering**: merekomendasikan film berdasarkan kemiripan karakteristik film yang pernah disukai pengguna (Pazzani & Billsus, 2007).

---

## Business Understanding

### Problem Statements

- Bagaimana membantu pengguna menemukan film sesuai preferensi genre dari ribuan film?
- Bagaimana memberikan rekomendasi berdasarkan histori tontonan dan rating pengguna, serta preferensi pengguna lain yang mirip?

### Goals

- Membangun sistem rekomendasi film berbasis **Content-Based Filtering (CBF)** yang merekomendasikan berdasarkan kesamaan genre.
- Membangun sistem rekomendasi film berbasis **Collaborative Filtering (CF)** menggunakan Deep Learning berdasarkan histori rating dan pola pengguna.

### Solution Statements

#### Content-Based Filtering (CBF)

- Gunakan TF-IDF untuk bobot fitur genre.
- Hitung kemiripan antar film menggunakan Cosine Similarity.
- Hasilkan rekomendasi Top-N film paling mirip genre dengan film yang dipilih.

#### Collaborative Filtering (CF) dengan Deep Learning

- Encode `userId` dan `movieId` menjadi representasi numerik.
- Bangun model Deep Learning dengan embedding layers untuk mempelajari representasi laten pengguna dan film.
- Hitung skor kecocokan lewat dot product embedding + bias.
- Gunakan sigmoid untuk output skor dalam rentang [0,1].
- Latih model dengan data rating, evaluasi pakai MAE dan RMSE.
- Hasilkan rekomendasi Top-N film yang diprediksi disukai pengguna.

---

## Data Understanding

Dataset utama berasal dari Kaggle:
(https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

### Informasi Dataset

Dataset yang digunakan adalah The Movies Dataset. Informasi mengenai dataset film ini dijelaskan sebagai berikut:

Sumber: Dataset tersedia di Kaggle

Pemilik Dataset: Rounak Banik

Lisensi: CC0 (Public Domain)

Kategori: Movies & TV Shows

Format dan Ukuran Berkas: ZIP Version 7 dengan ukuran sekitar 943.76 MB

Jumlah File dalam Dataset: Terdiri dari 7 file CSV

Dari informasi tersebut, diketahui bahwa file-file dalam dataset ini berisi metadata untuk sekitar 45.000 film yang tercakup dalam Full MovieLens Dataset. Data film tersebut merupakan film-film yang dirilis pada atau sebelum Juli 2017. Informasi yang tersedia meliputi pemeran, kru, kata kunci plot, anggaran, pendapatan, poster, tanggal rilis, bahasa, perusahaan produksi, negara asal, jumlah suara di TMDB, serta rata-rata rating.

Selain itu, dataset ini juga mencakup file yang berisi sekitar 26 juta rating yang diberikan oleh 270.000 pengguna untuk seluruh 45.000 film tersebut. Rating yang digunakan memiliki skala 1 hingga 5 dan berasal dari situs resmi GroupLens.

Berikut penjelasan singkat mengenai file-file dalam kumpulan data ini:

movie_metadata.csv: File utama berisi metadata film untuk 45.000 film dalam dataset Full MovieLens. Fitur yang tersedia antara lain poster, latar belakang, anggaran, pendapatan, tanggal rilis, bahasa, negara produksi, dan perusahaan produksi.

keywords.csv: Berisi kata kunci plot film dalam bentuk objek JSON yang dirangkai.

credits.csv: Informasi tentang pemeran dan kru untuk semua film, disajikan dalam format objek JSON yang dirangkai.

links.csv: File yang memuat ID TMDB dan IMDB dari seluruh film dalam Full MovieLens Dataset.

links_small.csv: Memuat ID TMDB dan IMDB untuk sebagian kecil sekitar 9.000 film dari dataset lengkap.

rating_small.csv: Bagian kecil dari dataset rating, terdiri dari 100.000 rating yang diberikan oleh 700 pengguna untuk 9.000 film.

Statistik Umum Data set
![Statistik](assets/statistik.png)
Gambar di atas menunjukkan total data yang terdapat dalam berbagai file dataset film.

Dalam proyek ini, hanya dua file CSV yang digunakan, yaitu ratings_small.csv (disimpan dalam variabel ratings) dan movies_metadata.csv (disimpan dalam variabel movies).

Variabel utama yang digunakan:

| Dataset             | Variabel                                    | Keterangan           |
| ------------------- | ------------------------------------------- | -------------------- |
| movies_metadata.csv | id, genres, title, vote_average, vote_count | Metadata film        |
| ratings_small.csv   | userId, movieId, rating, timestamp          | Data rating pengguna |

- **movies_metadata.csv**: Detail 45.436 film.
- **ratings_small.csv**: 100.004 rating dari 671 pengguna terhadap 9.066 film.

- Rating terbanyak: 4.0 (28.7%), 3.0 (20.1%), 5.0 (15.1%).
- Genre paling banyak: Drama, Comedy (>13.000 film).
- Film dengan skor tertinggi: _Dilwale Dulhania Le Jayenge_.
- Film dengan rating terbanyak: _Terminator 3: Rise of the Machines_ (324 rating, rata-rata 4.25).

### Eksplorasi Data (EDA)

1. Analisis Univariate
   Pada tahap ini, dilakukan analisis statistik terhadap variabel movies dan ratings untuk memahami distribusi, pola, dan karakteristik data. Selain itu, grafik digunakan untuk memvisualisasikan distribusi genre dan rating film, serta menggambarkan hubungan antar fitur dalam dataset.

   Deskripsi Variabel

   Pada tahap ini, akan disajikan penjelasan mengenai variabel-variabel yang digunakan, yaitu:
   a. Ratings (file ratings_small.csv)
   Penjelasan mengenai variabel tersebut.

---

## Data Preparation

Langkah-langkah utama:

- Pilih kolom relevan (`id`, `genres`, `title`, `vote_average`, `vote_count`) dan rename `id` jadi `movieId`.
- Hapus missing values dan data duplikat pada metadata film.
- Ubah kolom `genres` dari JSON string ke list nama genre.
- Gabungkan data rating dengan metadata film berdasarkan `movieId`.
- Hapus kolom tidak diperlukan (`timestamp`, `vote_average`, `vote_count`).
- Sampling 20.000 data untuk pemodelan.
- Encode `userId` dan `movieId` menjadi integer berurutan.
- Normalisasi rating ke rentang [0,1].
- Bagi data jadi training (80%) dan validasi (20%).

---

## Modeling

### 1. Content-Based Filtering (CBF)

- Ubah `genres` ke string genre terpisah koma.
- Gunakan `TfidfVectorizer` untuk membuat matriks TF-IDF genre.
- Hitung cosine similarity antar film.
- Fungsi rekomendasi mencari film mirip berdasarkan genre, hasil Top-N.

**Kelebihan:**

- Tidak perlu data rating pengguna lain.
- Rekomendasi transparan dan spesifik.

**Kekurangan:**

- Terbatas pada fitur genre.
- Bisa overspecialize, kurang eksplorasi.
- Sulit untuk item baru tanpa fitur.

### 2. Collaborative Filtering (CF) dengan Deep Learning

- Model `RecommenderNet` dengan embedding layer untuk user dan movie.
- Output prediksi rating memakai dot product embedding + bias dan sigmoid.
- Kompilasi dengan Adam optimizer, loss MSE, metrik MAE dan RMSE.
- Latih dengan early stopping.
- Fungsi rekomendasi prediksi rating film yang belum ditonton, ambil Top-N tertinggi.

**Kelebihan:**

- Tangkap pola non-linear kompleks.
- Dapat temukan kesamaan laten.
- Performa meningkat dengan data besar.

**Kekurangan:**

- Butuh data interaksi banyak.
- Model kurang transparan (black box).
- Pelatihan mahal secara komputasi.

---

## Evaluation

### Content-Based Filtering

- Metrik: **Precision**
- Definisi: Rasio rekomendasi yang relevan (genre sama persis).
- Hasil: Precision 100% untuk film _Made in Hong Kong_ (10/10 rekomendasi relevan).

### Collaborative Filtering (Deep Learning)

- Metrik: **MAE** dan **RMSE**
- Hasil setelah 10 epoch (early stopping):

| Data       | Loss (MSE) | MAE    | RMSE   |
| ---------- | ---------- | ------ | ------ |
| Training   | 0.0058     | 0.0337 | 0.0447 |
| Validation | 0.0537     | 0.1735 | 0.2233 |

- Visualisasi metrik menunjukkan penurunan MAE dan RMSE pada training, stabil di validasi, menandakan model belajar dengan baik namun perlu awasi overfitting.

---

## Referensi

- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). _Item-based collaborative filtering recommendation algorithms_. Proceedings of the 10th international conference on World Wide Web, 285-295.
- Pazzani, M. J., & Billsus, D. (2007). _Content-based recommendation systems_. The adaptive web, 325-341.

---

## Catatan

Laporan ini adalah ringkasan lengkap proses mulai dari business understanding, data understanding, data preparation, pemodelan, hingga evaluasi performa sistem rekomendasi film yang dibangun.
