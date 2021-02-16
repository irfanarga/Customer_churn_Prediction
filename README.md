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
### Mencari ID pelanggan (Nomor telphone) yang valid

### Mengatasi data-data yang masih kosong (Missing Values)

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Missing%20value.png)
![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Missing%20value%20handled.png)

Setelah dianalisis lebih lanjut, ternyata masih ada Missing Values dari data yang sudah validkan Id Number pelanggannya. Missing values terdapat pada kolom Churn, tenure, MonthlyCharges & TotalCharges. Setelah ditangani dengan cara penghapusan rows dan pengisian rows dengan nilai tertentu, terbukti sudah tidak ada missing values lagi pada data, terbukti dari jumlah missing values masing-masing variable yang bernilai 0.

### Mengatasi Nilai-Nilai Pencilan (Outlier) dari setiap Variable
Mendeteksi Pencilan dari suatu Nilai (Outlier) salah satunya bisa melihat plot dari data tersebut menggunakan Box Plot. Boxplot merupakan ringkasan distribusi sampel yang disajikan secara grafis yang bisa menggambarkan bentuk distribusi data (skewness), ukuran tendensi sentral dan ukuran penyebaran (keragaman) data pengamatan. 

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Data%20ada%20outlier.png)

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Boxplot%20tenure.png)
![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Boxplot%20monthly%20charge.png)
![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Boxplot%20total%20charge.png)

Nilai outlier tersebut ditangani dengan cara merubah nilainya ke nilai Maximum & Minimum dari interquartile range (IQR).

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/outlier%20handled.png)

### Menstandarisasi Nilai dari Variable
Mendeteksi apakah ada nilai-nilai dari variable kategorik yang tidak standard. Hal ini biasanya terjadi dikarenakan kesalahan input data. Perbedaan istilah menjadi salah satu faktor yang sering terjadi, untuk itu dibutuhkan standarisasi dari data yang sudah terinput.

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/Nilai%20tidak%20standar.png)

Nilai tersebut distandarkan dengan pola terbanyaknya, dengan syarat tanpa mengubah maknanya.

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/Data_Cleansing/nilai%20standar.png)

## Pemodelan Machine Learning
### Eksploratory Data Analysis (EDA)
Exploratory Data Analysis memungkinkan analyst memahami isi data yang digunakan, mulai dari distribusi, frekuensi, korelasi dan lainnya. Pada umumnya EDA dilakukan dengan beberapa cara yaitu Univariat, Bivariat, dan Multivariat Analysis. Analisis dalam project ini dilakukan dengan melihat persebaran:
  - Prosentase persebaran data Churn dan tidaknya dari seluruh data
  
![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/Persentase%20pelanggan%20Churn.png)

Dari grafik diatas, diketahui bahwa sebaran data secara kesuluruhan customer tidak melakukan churn, dengan detil Churn sebanyak 26% dan No Churn sebanyak 74%.
  
  - Persebaran data dari variable predictor terhadap label (Churn)

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/EDA%20predictor%20terhadap%20churn.png)

MonthlyCharges ada kecenderungan semakin kecil nilai biaya bulanan yang dikenakan, semakin kecil juga kecenderungan untuk melakukan Churn. Untuk TotalCharges terlihat tidak ada kecenderungan apapun terhadap Churn customers. Untuk tenure ada kecenderungan semakin lama berlangganan customer, semakin kecil kecenderungan untuk melakukan Churn.

![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/EDA%20predictor%20kategorik%20terhadap%20churn.png)

Tidak ada perbedaan yang signifikan untuk orang melakukan churn dilihat dari faktor jenis kelamin dan layanan telfonnya. Akan tetapi ada kecenderungan bahwa orang yang melakukan churn adalah orang-orang yang tidak memiliki partner, orang-orang yang statusnya adalah senior citizen, orang-orang yang mempunyai layanan streaming TV, orang-orang yang mempunyai layanan Internet dan orang-orang yang tagihannya paperless.

### Data Pre-Processing
Beberapa hal yang dilakukan dalam tahap ini yaitu:
   - Menghapus kolom yang tidak digunakan dalam pemodelan seperti customerID & UpdatedAt
   - Mengubah semua bentuk data menjadi numerik dengan Encoding Data
   - Membagi dataset menjadi 2 bagian yaitu 70% training dan 30% testing

### Melakukan Pemodelan Machine Learning
1. Logistic Regression (Default)
   - Performa model training
   
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/LR%20matrix%20train.png)
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/LR%20matrix%20plot%20train.png)
   
   Dari data training terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 80%, dengan detil tebakan churn yang sebenernya benar churn adalah      638, tebakan tidak churn yang sebenernya tidak churn adalah 3237, tebakan tidak churn yang sebenernya benar churn adalah 652 dan tebakan churn yang sebenernya tidak churn        adalah 338.
   
   - Performa model testing
   
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/LR%20matrix%20tes.png)
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/LR%20matrix%20plot%20test.png)

   Dari data testing terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 79%, dengan detil tebakan churn yang sebenernya benar churn adalah        264, tebakan tidak churn yang sebenernya tidak churn adalah 1392, tebakan tidak churn yang sebenernya benar churn adalah 282 dan tebakan churn yang sebenernya tidak churn        adalah 146.

2. Random Forest Classifier (Default)
   - Performa model training
   
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/RDF%20matrix%20train.png)
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/RDF%20matrix%20plot%20train.png)
   
   Dari data training terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 100%, dengan detil tebakan churn yang sebenernya benar churn adalah      1278, tebakan tidak churn yang sebenernya tidak churn adalah 3566, tebakan tidak churn yang sebenernya benar churn adalah 12 dan tebakan churn yang sebenernya tidak churn        adalah 9.
   
   - Performa model testing
   
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/RDF%20matrix%20test.png)
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/RDF%20matrix%20plot%20test.png)
   
   Dari data testing terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 78%, dengan detil tebakan churn yang sebenernya benar churn adalah        262, tebakan tidak churn yang sebenernya tidak churn adalah 1360, tebakan tidak churn yang sebenernya benar churn adalah 284 dan tebakan churn yang sebenernya tidak churn        adalah 179.
   
3. Gradient Boosting Classifier (Default)
   - Performa model training
   
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/GBT%20matrix%20train.png)
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/GBT%20matrix%20plot%20train.png)
   
   Dari data training terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 82%, dengan detil tebakan churn yang sebenernya benar churn adalah      684, tebakan tidak churn yang sebenernya tidak churn adalah 3286, tebakan tidak churn yang sebenernya benar churn adalah 606 dan tebakan churn yang sebenernya tidak churn        adalah 289.
   
   - Performa model testing
   
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/GBT%20matrix%20test.png)
   ![](https://github.com/irfanarga/Customer_churn_Prediction/blob/master/images/pemodelan/GBT%20matrix%20plot%20test.png)
   
   Dari data testing terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 79%, dengan detil tebakan churn yang sebenernya benar churn adalah        261, tebakan tidak churn yang sebenernya tidak churn adalah 1394, tebakan tidak churn yang sebenernya benar churn adalah 285 dan tebakan churn yang sebenernya tidak churn        adalah 145.

### Menentukan Model Terbaik
Model yang baik adalah model yang mampu memberikan performa bagus di fase training dan testing model.
Ada beberapa kondisi model antara lain:
- Over-Fitting adalah suatu kondisi dimana model mampu memprediksi dengan sangat baik di fase training, akan tetapi tidak mampu memprediksi sama baiknya di fase testing.
- Under-Fitting adalah suatu kondisi dimana model kurang mampu memprediksi dengan baik di fase training, akan tetapi mampu memprediksi dengan baik di fase testing.
- Appropriate-Fitting adalah suatu kondisi dimana model mampu memprediksi dengan baik di fase training maupun di fase testing.

Berdasarkan pemodelan yang telah dilakukan dengan menggunakan Logistic Regression, Random Forest dan Extreme Gradiant Boost, maka dapat disimpulkan untuk memprediksi pelanggan Churn dengan menggunakan dataset ini, model terbaiknya adalah menggunakan algortima Logistic Regression. Hal ini dikarenakan performa dari model Logistic Regression cenderung mampu memprediksi sama baiknya di fase training maupun testing (akurasi training 80%, akurasi testing 79%), dilain sisi algoritma lainnya cenderung Over-Fitting performanya. Akan tetapi hal ini tidak menetapkan bahwasannya jika untuk melakukan pemodelan apapun maka digunakan Logistic Regression, namun tetap harus melakukan banyak percobaan model untuk menentukan mana yang terbaik.

