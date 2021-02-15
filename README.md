# Customer churn Prediction
Tujuan project ini adalah mempersiapkan data sekaligus membuat model prediksi yang tepat untuk menentukan pelanggan akan berhenti berlangganan (Churn) atau tidak.

## Data Cleansing
Langkah-langkah data cleansing:
- Mencari ID pelanggan (Nomor telphone) yang valid

- Mengatasi data-data yang masih kosong (Missing Values)

Gambar missing value

Gambar missing value handled

Setelah dianalisis lebih lanjut, ternyata masih ada Missing Values dari data yang sudah validkan Id Number pelanggannya. Missing values terdapat pada kolom Churn, tenure, MonthlyCharges & TotalCharges. Setelah ditangani dengan cara penghapusan rows dan pengisian rows dengan nilai tertentu, terbukti sudah tidak ada missing values lagi pada data, terbukti dari jumlah missing values masing-masing variable yang bernilai 0.

- Mengatasi Nilai-Nilai Pencilan (Outlier) dari setiap Variable
Mendeteksi Pencilan dari suatu Nilai (Outlier) salah satunya bisa melihat plot dari data tersebut menggunakan Box Plot. Boxplot merupakan ringkasan distribusi sampel yang disajikan secara grafis yang bisa menggambarkan bentuk distribusi data (skewness), ukuran tendensi sentral dan ukuran penyebaran (keragaman) data pengamatan. 

Gambar print outlier

Gambar Box 1

Gambar Box 2

Gambar Box 3

Nilai outlier tersebut ditangani dengan cara merubah nilainya ke nilai Maximum & Minimum dari interquartile range (IQR).

Gambar print outlier handled

- Menstandarisasi Nilai dari Variable
Mendeteksi apakah ada nilai-nilai dari variable kategorik yang tidak standard. Hal ini biasanya terjadi dikarenakan kesalahan input data. Perbedaan istilah menjadi salah satu faktor yang sering terjadi, untuk itu dibutuhkan standarisasi dari data yang sudah terinput.

Gambar nilai tidak standar

Nilai tersebut distandarkan dengan pola terbanyaknya, dengan syarat tanpa mengubah maknanya.

Gambar nilai standar

Library yang digunakan:
- Pandas
- Matplotlib
- Seaborn

