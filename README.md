# Customer churn Prediction
Tujuan project ini adalah mempersiapkan data sekaligus membuat model prediksi yang tepat untuk menentukan pelanggan akan berhenti berlangganan (Churn) atau tidak.

## Library
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Xgboost
- Pickle

## Data Cleansing

Langkah-langkah data cleansing:
- Mencari ID pelanggan (Nomor telphone) yang valid

- Mengatasi data-data yang masih kosong (Missing Values)

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Missing%20value.png)
![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Missing%20value%20handled.png)

Setelah dianalisis lebih lanjut, ternyata masih ada Missing Values dari data yang sudah validkan Id Number pelanggannya. Missing values terdapat pada kolom Churn, tenure, MonthlyCharges & TotalCharges. Setelah ditangani dengan cara penghapusan rows dan pengisian rows dengan nilai tertentu, terbukti sudah tidak ada missing values lagi pada data, terbukti dari jumlah missing values masing-masing variable yang bernilai 0.

- Mengatasi Nilai-Nilai Pencilan (Outlier) dari setiap Variable
Mendeteksi Pencilan dari suatu Nilai (Outlier) salah satunya bisa melihat plot dari data tersebut menggunakan Box Plot. Boxplot merupakan ringkasan distribusi sampel yang disajikan secara grafis yang bisa menggambarkan bentuk distribusi data (skewness), ukuran tendensi sentral dan ukuran penyebaran (keragaman) data pengamatan. 

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Data%20ada%20outlier.png)

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Boxplot%20tenure.png)
![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Boxplot%20monthly%20charge.png)
![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Boxplot%20total%20charge.png)

Nilai outlier tersebut ditangani dengan cara merubah nilainya ke nilai Maximum & Minimum dari interquartile range (IQR).

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/outlier%20handled.png)

- Menstandarisasi Nilai dari Variable
Mendeteksi apakah ada nilai-nilai dari variable kategorik yang tidak standard. Hal ini biasanya terjadi dikarenakan kesalahan input data. Perbedaan istilah menjadi salah satu faktor yang sering terjadi, untuk itu dibutuhkan standarisasi dari data yang sudah terinput.

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Nilai%20tidak%20standar.png)

Nilai tersebut distandarkan dengan pola terbanyaknya, dengan syarat tanpa mengubah maknanya.

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/nilai%20standar.png)

## Pemodelan Machine Learning
- Eksploratory Data Analysis (EDA)
Exploratory Data Analysis memungkinkan analyst memahami isi data yang digunakan, mulai dari distribusi, frekuensi, korelasi dan lainnya. Pada umumnya EDA dilakukan dengan beberapa cara yaitu Univariat, Bivariat, dan Multivariat Analysis. Analisis dalam project ini dilakukan dengan melihat persebaran:
  - Prosentase persebaran data Churn dan tidaknya dari seluruh data
  
![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/Persentase%20pelanggan%20Churn.png)

Dari grafik diatas, diketahui bahwa sebaran data secara kesuluruhan customer tidak melakukan churn, dengan detil Churn sebanyak 26% dan No Churn sebanyak 74%.
  
  - Persebarang data dari variable predictor terhadap label (Churn)

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/EDA%20predictor%20terhadap%20churn.png)

MonthlyCharges ada kecenderungan semakin kecil nilai biaya bulanan yang dikenakan, semakin kecil juga kecenderungan untuk melakukan Churn. Untuk TotalCharges terlihat tidak ada kecenderungan apapun terhadap Churn customers. Untuk tenure ada kecenderungan semakin lama berlangganan customer, semakin kecil kecenderungan untuk melakukan Churn.

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/EDA%20predictor%20kategorik%20terhadap%20churn.png)

Tidak ada perbedaan yang signifikan untuk orang melakukan churn dilihat dari faktor jenis kelamin dan layanan telfonnya. Akan tetapi ada kecenderungan bahwa orang yang melakukan churn adalah orang-orang yang tidak memiliki partner, orang-orang yang statusnya adalah senior citizen, orang-orang yang mempunyai layanan streaming TV, orang-orang yang mempunyai layanan Internet dan orang-orang yang tagihannya paperless.

- Melakukan Data Pre-Processing
- Melakukan Pemodelan Machine Learning
- Menentukan Model Terbaik
