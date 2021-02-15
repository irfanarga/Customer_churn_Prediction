import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns = 50

#import dataset
df_load = pd.read_csv('dqlab_telco.csv')

#Tampilkan jumlah baris dan kolom
print(df_load.shape)

#Tampilkan 5 data teratas
print(df_load.head(5))

#Jumlah ID yang unik
print(df_load.customerID.nunique())

#Memfilter ID Number Pelanggan
"""
Kriteria:
- Panjang karakter adalah 11-12.
- Terdiri dari Angka Saja, tidak diperbolehkan ada karakter selain angka
- Diawali dengan angka 45 2 digit pertama.
"""

df_load['valid_id']=df_load['customerID'].astype(str).str.match(r'(45\d{9,10})')
df_load=(df_load[df_load['valid_id']==True]).drop('valid_id', axis=1)
print('Hasil jumlah ID Customer yang terfilter adalah',df_load['customerID'].count())

#Memfilter Duplikasi ID Number Pelanggan
"""
Penyebab duplikasi:
- inserting melebihi satu kali dengan nilai yang sama tiap kolomnya
- inserting beda periode pengambilan data
"""

#Menghapus nilai duplikat
df_load.drop_duplicates()
#Menghapus nilai duplikat ID yang diurutkan berdasarkan Periode
df_load=df_load.sort_values('UpdatedAt',ascending=False).drop_duplicates(['customerID'])
print('Hasil jumlah ID Customer yang sudah dihilangkan duplikasinya (distinct) adalah',df_load['customerID'].count())

#Mengatasi Missing Values dengan Penghapusan Rows
"""
Di asumsikan data modeller hanya mau menerima data yang benar ada flag churn nya atau tidak.
"""

print('Total missing values data dari kolom Churn',df_load['Churn'].isnull().sum())

#Menghapus semua missing value pada kolom churn
df_load.dropna(subset=['Churn'],inplace=True)

print('Total Rows dan kolom Data setelah dihapus data Missing Values adalah',df_load.shape)

#Mengatasi missing value dengan pengisian nilai tertentu pada kolom numerik
"""
Diasumsikan data modeller meminta pengisian missing values dengan kriteria:
- Tenure pihak data modeller meminta setiap rows yang memiliki missing values untuk Lama berlangganan di isi dengan 11
- Variable yang bersifat numeric selain Tenure di isi dengan median dari masing-masing variable tersebut
"""

print('Status Missing Values:',df_load.isnull().values.any())
print('\nJumlah Missing Values masing-masing kolom, adalah:')
print(df_load.isnull().sum().sort_values(ascending=False))

#handling missing values Tenure fill with 11
df_load['tenure'].fillna(11, inplace=True)

#Loop
#Handling missing values num vars (except Tenure)
for col_name in list(['MonthlyCharges','TotalCharges']):
	median=df_load[col_name].median()
	df_load[col_name].fillna(median, inplace=True)

print('\nJumlah Missing Values setelah di imputer datanya, adalah:')
print(df_load.isnull().sum().sort_values(ascending=False))

#Mendeteksi adanya Outlier (Boxplot)
print('\nPersebaran data sebelum ditangani Outlier: ')
print(df_load[['tenure','MonthlyCharges','TotalCharges']].describe())

#Membuat Box Plot
#Masukkan variabel numerik
plt.figure()
sns.boxplot(x=df_load['tenure'])
plt.show()

plt.figure()
sns.boxplot(x=df_load['MonthlyCharges'])
plt.show()

plt.figure()
sns.boxplot(x=df_load['TotalCharges'])
plt.show()

#Mengatasi outlier dengan metode Inter Quartile Range (IQR)
"""
Tentukan:
- Nilai Minimum dan Maximum data di tolerir
- Ubah Nilai yg di luar Range Minumum & Maximum ke dalam nilai Minimum dan Maximum
"""

Q1 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.25)
Q3 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.75)
IQR = Q3 - Q1
maximum = Q3 + (1.5*IQR)
print('Nilai Maximum dari masing-masing Variable adalah: ')
print(maximum)
minimum = Q1 - (1.5*IQR)
print('\nNilai Minimum dari masing-masing Variable adalah: ')
print(minimum)
more_than = (df_load > maximum)
lower_than = (df_load < minimum)
df_load = df_load.mask(more_than, maximum, axis=1)
df_load = df_load.mask(lower_than, minimum, axis=1)
print('\nPersebaran data setelah ditangani Outlier: ')
print(df_load[['tenure','MonthlyCharges','TotalCharges']].describe())

#Mendeteksi Nilai yang tidak Standar
#Loop
for col_name in list(['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']):
	print('\nUnique Values Count \033[1m'+'Before Standardized \033[0m Variable',col_name)
	print(df_load[col_name].value_counts())

#Menstandarisasi Variable Kategorik
df_load = df_load.replace(['Wanita','Laki-Laki','Churn','Iya'],['Female','Male','Yes','Yes'])
# Masukkan variable
for col_name in list(['gender','Dependents','Churn']):
	print('\nUnique Values Count \033[1m' + 'After Standardized \033[0mVariable',col_name)
	print(df_load[col_name].value_counts())

