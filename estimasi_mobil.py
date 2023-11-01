import pickle
import streamlit as st

model = pickle.load(open('estimasi_mobil.sav', 'rb'))

st.title('estimasi harga mobil bekas')

year = st.number_input('input tahun mobil')
mileage = st.number_input('input km mobil')
tax = st.number_input('input pajak mobil')
mpg = st.number_input('input konsumsi BBM mobil')
engineSize = st.number_input('input Engine Size')
Transmission =st.number_input('input transmission')
fueltype =st.number_input('input fueltype')
type = st.number_input('input model')

predict = ''

if st.button('estimasi harga'):
    predict = model.predict(
        [[year, mileage, tax, mpg, engineSize,]]
    )
    st.write('estimasi harga mobil bekas dalam ponds : ', predict)
    st.write('estimasi harga mobil bekas dalam IDR (juta) :', predict*19000)
