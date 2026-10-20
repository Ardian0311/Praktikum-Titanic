import os
import pickle

import pandas as pd
import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Survival Titanic",
    page_icon="🚢",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load model, scaler dan encoder (di-cache agar tidak reload terus)
@st.cache_resource
def load_artifacts():
    def _load(name):
        with open(os.path.join(BASE_DIR, name), "rb") as f:
            return pickle.load(f)

    model = _load("model.pkl")
    scaler = _load("scaler.pkl")
    le_sex = _load("le_sex.pkl")
    le_embarked = _load("le_embarked.pkl")
    return model, scaler, le_sex, le_embarked


try:
    model, scaler, le_sex, le_embarked = load_artifacts()
    artifacts_loaded = True
    load_error = None
except Exception as e:
    artifacts_loaded = False
    load_error = e

# Header
st.title("🚢 Prediksi Survival Penumpang Titanic")
st.markdown(
    """
    Aplikasi ini memprediksi **peluang seorang penumpang selamat** dalam
    tragedi Titanic, menggunakan model **Random Forest Classifier**
    yang telah dilatih pada dataset Titanic.
    """
)

if not artifacts_loaded:
    st.error(
        "Gagal memuat file model (model.pkl, scaler.pkl, le_sex.pkl, le_embarked.pkl). "
        "Pastikan keempat file ada di folder yang sama dengan app.py dan sudah "
        "ter-push ke GitHub."
    )
    st.exception(load_error)
    st.stop()

st.divider()

# Form input data penumpang
st.subheader("Masukkan Data Penumpang")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Kelas Tiket (Pclass)",
        options=[1, 2, 3],
        index=2,
        help="1 = Kelas 1 (mewah), 2 = Kelas 2, 3 = Kelas 3 (ekonomi)"
    )
    sex = st.radio("Jenis Kelamin", options=["male", "female"], horizontal=True)
    age = st.slider("Umur", min_value=0, max_value=100, value=30)
    fare = st.number_input("Harga Tiket (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

with col2:
    sibsp = st.number_input("Jumlah Saudara/Pasangan di Kapal (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Jumlah Orang Tua/Anak di Kapal (Parch)", min_value=0, max_value=10, value=0)
    embarked = st.selectbox(
        "Pelabuhan Keberangkatan (Embarked)",
        options=["S", "C", "Q"],
        help="S = Southampton, C = Cherbourg, Q = Queenstown"
    )

family_size = sibsp + parch + 1
st.caption(f"Total anggota keluarga di kapal (FamilySize): **{family_size}**")

st.divider()

# Prediksi
if st.button("Prediksi Sekarang", type="primary", use_container_width=True):

    # Encode input sesuai encoder yang dipakai saat training
    sex_encoded = le_sex.transform([sex])[0]
    embarked_encoded = le_embarked.transform([embarked])[0]

    # Urutan kolom harus sama dengan saat training:
    # Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked_encoded,
        "FamilySize": family_size
    }])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.success(f"**SELAMAT** - Peluang selamat: {probability[1] * 100:.2f}%")
    else:
        st.error(f"**TIDAK SELAMAT** - Peluang tidak selamat: {probability[0] * 100:.2f}%")

    # Detail probabilitas
    prob_df = pd.DataFrame({
        "Status": ["Tidak Selamat", "Selamat"],
        "Probabilitas (%)": [probability[0] * 100, probability[1] * 100]
    })
    st.bar_chart(prob_df.set_index("Status"))

    with st.expander("Lihat data input yang digunakan model"):
        st.dataframe(input_data)

st.divider()
st.caption(
    "Model: Random Forest Classifier • Dataset: Titanic (891 penumpang) • "
    "Dibuat dengan Streamlit"
)
