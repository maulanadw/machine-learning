import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.write("""
# Prediksi Harga Rumah App

Tampilkan prediksi *harga perumahan*!

""")
st.write('---')

# Load dan tampilkan datasets
# Membaca dataset
dataset = load_boston()
df = pd.DataFrame(dataset.data)
# Tetapkan kolom ke df
df.columns = dataset.feature_names
# Tetapkan variabel harga rumah
df["HARGA"] = dataset.target

# Tampilkan data tabel
if st.checkbox('Tampilkan dataset sebagai tabel'):
	st.dataframe(df)

# Cek, "Target" vs setiap variabel
if st.checkbox(''):
	checked_variable = st.selectbox(
		'Pilih satu variabel:',
		FeatureNames
		)
	# Plot
	fig, ax = plt.subplots(figsize=(5, 3))
	ax.scatter(x=df[checked_variable], y=df["HARGA"])
	plt.xlabel(checked_variable)
	plt.ylabel("HARGA")
	st.pyplot(fig)

# Explanatory Data variable
FeatureNames = [\
              #-- "Tingkat kejadian kejahatan per unit populasi menurut kota"
              "CRIM",\
              #-- "Persentase rumah seluas 25000 kaki persegi"
              'ZN',\
              #-- "Persentase luas lahan non-ritel menurut kota"
              'INDUS',\
              #-- "Indeks untuk sungai Charles: 0 dekat, 1 jauh"
              'CHAS',\
              #-- "Konsentrasi senyawa nitrogen"
              'NOX',\
              #-- "Rata-rata jumlah kamar per tempat tinggal"
              'RM',\
              #-- "Persentase bangunan yang dibangun sebelum 1940"
              'AGE',\
              #-- 'Jarak tertimbang dari lima pusat pekerjaan'
              "DIS",\
              ##-- "Indeks untuk akses mudah ke jalan raya"
              'RAD',\
              ##-- "Tarif pajak per $100,000"
              'TAX',\
              ##-- "Persentase siswa dan guru di setiap kota"
              'PTRATIO',\
              ##-- "1000(Bk - 0,63)^2, di mana Bk adalah persentase orang kulit hitam"
              'B',\
              ##-- "Persentase penduduk kelas bawah"
              'LSTAT'
              ]

"""
## Preprocessing
"""
# Pilih variabel yang TIDAK akan digunakan
Features_chosen = []
Features_NonUsed = st.multiselect(
	'Pilih variabel yang TIDAK akan digunakan', 
	FeatureNames)

df = df.drop(columns=Features_NonUsed)

left_column, right_column = st.beta_columns(2)
bool_log = left_column.radio(
			'Lakukan transformasi logaritma?', 
			('Tidak','Ya')
			)

df_log, Log_Features = df.copy(), []
if bool_log == 'Ya':
	Log_Features = right_column.multiselect(
					'Pilih variabel, lakukan transformasi logaritma', 
					df.columns
					)
	# Lakukan transformasi logaritma
	df_log[Log_Features] = np.log(df_log[Log_Features])

left_column, right_column = st.beta_columns(2)
bool_std = left_column.radio(
			'Lakukan standarisasi?', 
			('Tidak','Ya')
			)

df_std = df_log.copy()
if bool_std == 'Ya':
	Std_Features_chosen = []
	Std_Features_NonUsed = right_column.multiselect(
					'Pilih variabel yang TIDAK akan distandarisasi (variabel categorical)', 
					df_log.drop(columns=["HARGA"]).columns
					)
	for name in df_log.drop(columns=["HARGA"]).columns:
		if name in Std_Features_NonUsed:
			continue
		else:
			Std_Features_chosen.append(name)
	# Lakukan standarisasi
	sscaler = preprocessing.StandardScaler()
	sscaler.fit(df_std[Std_Features_chosen])
	df_std[Std_Features_chosen] = sscaler.transform(df_std[Std_Features_chosen])

"""
### Split dataset
"""
left_column, right_column = st.beta_columns(2)

# test size
test_size = left_column.number_input(
				'Validasi ukuran dataset (rate: 0.0-1.0):',
				min_value=0.0,
				max_value=1.0,
				value=0.2,
				step=0.1,
				 )

# random_seed
random_seed = right_column.number_input('Set random seed (0-1):',
							  value=0, step=1,
							  min_value=0)

# split dataset
X_train, X_val, Y_train, Y_val = train_test_split(
	df_std.drop(columns=["HARGA"]), 
	df_std['HARGA'], 
	test_size=test_size, 
	random_state=random_seed
	)


# Train model

regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Prediksi training data dan validasi

Y_pred_train = regressor.predict(X_train)
Y_pred_val = regressor.predict(X_val)

# Transformasi logaritma terbalik jika perlu
if "HARGA" in Log_Features:
	Y_pred_train, Y_pred_val = np.exp(Y_pred_train), np.exp(Y_pred_val)
	Y_train, Y_val = np.exp(Y_train), np.exp(Y_val)


## Tampilkan hasilnya
### Cek skor R2 (Indikator validasi)

R2 = r2_score(Y_val, Y_pred_val)
st.write(f'Skor R2: {R2:.2f}')

"""
### Plot (result)
"""
left_column, right_column = st.beta_columns(2)
show_train = left_column.radio(
				'Tampilkan training dataset:', 
				('Ya','Tidak')
				)
show_val = right_column.radio(
				'Tampilkan validasi dataset:', 
				('Ya','Tidak')
				)

# default axis range
y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_val), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val])) 

# interactive axis range
left_column, right_column = st.beta_columns(2)
x_min = left_column.number_input('x_min:',value=0,step=1)
x_max = right_column.number_input('x_max:',value=y_max,step=1)
left_column, right_column = st.beta_columns(2)
y_min = left_column.number_input('y_min:',value=0,step=1)
y_max = right_column.number_input('y_max:',value=y_max,step=1)


fig = plt.figure(figsize=(3, 3))
if show_train == 'Ya':
	plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
if show_val == 'Ya':
	plt.scatter(Y_val, Y_pred_val,lw=0.1,color="b",label="validasi data")
plt.xlabel("HARGA",fontsize=8)
plt.ylabel("PREDIKSI HARGA",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
st.pyplot(fig)