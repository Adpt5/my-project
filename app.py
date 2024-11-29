import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Fungsi untuk memuat dan memproses data
def load_and_preprocess():
    st.title("Data Preprocessing")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Membaca dataset
        if uploaded_file.name.endswith("csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("Data Preview", data.head())

        # Menghapus missing values atau mengisi missing values
        st.subheader("Handling Missing Data")
        missing_option = st.radio("Pilih Penanganan Missing Data", ["Hapus", "Isi dengan rata-rata/median"])
        if missing_option == "Hapus":
            data = data.dropna()
        elif missing_option == "Isi dengan rata-rata/median":
            for column in data.select_dtypes(include=[np.number]).columns:
                data[column] = data[column].fillna(data[column].mean())

        st.write("Data setelah penanganan missing values:", data.head())

        # Pisahkan kolom numerik dan non-numerik
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns

        # Korelasi hanya pada kolom numerik
        st.write("Correlation Matrix")
        if numerical_columns.size > 0:
            correlation_matrix = data[numerical_columns].corr()
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(fig)
        else:
            st.write("Tidak ada kolom numerik untuk korelasi.")

        # Grafik Distribusi
        st.subheader("Distribusi Fitur Numerik")
        if numerical_columns.size > 0:
            feature = st.selectbox("Pilih Fitur untuk Distribusi", numerical_columns)
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data[feature], kde=True)
            st.pyplot(fig)

        st.session_state.data = data
        return data

# Fungsi untuk melatih model machine learning
def model_training():
    st.title("Model Training and Evaluation")

    # Memilih dataset yang sudah diupload
    if 'data' not in st.session_state:
        st.warning("Harap upload dataset terlebih dahulu di halaman Data Preprocessing.")
        return

    data = st.session_state.data

    # Memilih kolom target dan fitur
    target_column = st.selectbox("Pilih Kolom Target", data.columns)
    feature_columns = [col for col in data.columns if col != target_column]

    # Pisahkan kolom numerik dan non-numerik
    X = data[feature_columns]
    numerical_columns = X.select_dtypes(include=[np.number]).columns
    non_numerical_columns = X.select_dtypes(exclude=[np.number]).columns

    # Memisahkan fitur (X) dan target (y)
    y = data[target_column]

    # Membagi dataset menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menyusun pipeline untuk preprocessing data
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_columns),
                      ('cat', OneHotEncoder(handle_unknown='ignore'), non_numerical_columns)
                     ])

    # Pemilihan Model
    model_option = st.radio("Pilih Algoritma Machine Learning", ["Logistic Regression", "Random Forest"])

    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier()

    # Membuat pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Melatih model menggunakan pipeline
    pipeline.fit(X_train, y_train)

    # Evaluasi Model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Simpan pipeline ke session state
    st.session_state.pipeline = pipeline
    st.session_state.target_column = target_column
    st.session_state.feature_columns = feature_columns

# Fungsi untuk visualisasi interaktif
def interactive_visualization():
    st.title("Interactive Visualization")

    # Memeriksa apakah pipeline tersedia di session state
    if 'pipeline' not in st.session_state:
        st.warning("Harap latih model terlebih dahulu di halaman Model Training.")
        return

    # Ambil pipeline dari session state
    pipeline = st.session_state.pipeline

    # Memastikan data tersedia
    if 'data' not in st.session_state:
        st.warning("Harap upload dataset terlebih dahulu di halaman Data Preprocessing.")
        return

    data = st.session_state.data
    target_column = st.session_state.target_column
    feature_columns = st.session_state.feature_columns

    # Visualisasi interaktif menggunakan Plotly
    st.subheader("Visualisasi Interaktif")
    selected_feature = st.selectbox("Pilih Fitur untuk Visualisasi", feature_columns)
    fig = px.scatter(data_frame=data, x=selected_feature, y=target_column, title=f"{selected_feature} vs {target_column}")
    st.plotly_chart(fig)

    # Prediksi Interaktif
    st.subheader("Prediksi Menggunakan Model")
    input_features = {}

    for col in feature_columns:
        if col in data.select_dtypes(include=[np.number]).columns:
            # Input untuk fitur numerik
            input_features[col] = st.number_input(f"Masukkan nilai untuk {col}", value=0.0)
        else:
            # Input untuk fitur kategorikal
            input_features[col] = st.selectbox(f"Pilih nilai untuk {col}", options=data[col].unique())

    # Konversi input ke DataFrame
    input_df = pd.DataFrame([input_features])

    # Prediksi menggunakan pipeline
    prediction = pipeline.predict(input_df)
    st.write(f"Prediksi untuk data ini: {prediction[0]}")

# Fungsi login
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "admin" and password == "password":
            st.session_state.logged_in = True
            st.sidebar.success("Login berhasil!")
        else:
            st.sidebar.error("Username atau Password salah!")

# Fungsi untuk halaman About
def about():
    st.title("Tentang Aplikasi")
    st.write("""
    **Pembuat Aplikasi**: Adipati Sulaiman  
    Selamat datang di aplikasi ini! Dibuat untuk memudahkan Anda dalam mengeksplorasi dan melatih model machine learning, aplikasi ini menyediakan berbagai fitur canggih untuk:
    
    - **Pemrosesan Data**: Membersihkan dan mempersiapkan dataset agar siap untuk analisis dan pemodelan.
    - **Pelatihan Model**: Menyediakan berbagai algoritma machine learning untuk membangun model prediktif yang akurat.
    - **Visualisasi Interaktif**: Menampilkan grafik dan visualisasi dinamis yang memungkinkan Anda untuk lebih memahami data dan hasil model dengan mudah.

    Dengan antarmuka yang intuitif dan mudah digunakan, aplikasi ini bertujuan untuk memberikan pengalaman yang menyenangkan bagi para profesional dan pemula dalam bidang data science dan machine learning.
    """)


# Fungsi utama untuk menyusun halaman
def main():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login()
    else:
        st.sidebar.title("Menu")
        page = st.sidebar.radio("Pilih Halaman", ["Data Preprocessing", "Model Training", "Interactive Visualization", "About"])

        if page == "Data Preprocessing":
            load_and_preprocess()
        elif page == "Model Training":
            model_training()
        elif page == "Interactive Visualization":
            interactive_visualization()
        elif page == "About":
            about()

if __name__ == "__main__":
    main()
