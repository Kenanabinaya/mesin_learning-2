import pickle
import streamlit as st

model = pickle.load(open('estimasi_mobil.sav', 'rb'))

st.title('estimasi harga mobil bekas')

year = st.number_input('input tahun mobil')
mileage = st.number_input('input km mobil')
tax = st.number_input('input pajak mobil')
mpg = st.number_input('input konsumsi BBM mobil')
engineSize = st.number_input('input Engine Size')
Transmission_options = ['Manual', 'Automatic']
Transmission = st.selectbox('Masukkan jenis Transmission', Transmission_options)
if Transmission == 'Manual':
    Transmission = 1
else:
    Transmission = 2
fuelType_options = ['Petrol', 'Disel']
fuelType = st.selectbox('Masukkan jenis Bahan Bakar', fuelType_options)
if fuelType == 'Petrol':
    fuelType = 1
else:
    fuelType = 2
type_options = ['A1', 'A2']
type = st.selectbox('Masukkan model kendaraan', type_options)
if type == 'A1':
    type = 1
else:
    type = 2

predict = ''

if st.button('estimasi harga'):
    predict = model.predict(
        [[year, mileage, tax, mpg, engineSize,]]
    )
    st.write('estimasi harga mobil bekas dalam ponds : ', predict)
    st.write('estimasi harga mobil bekas dalam IDR (juta) :', predict*19000)
