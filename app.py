

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import time

df = pd.read_csv('ptkai_label.csv')
x = df['clean_review']
y = df['label']

model = pickle.load(open('sentiment.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf.pkl', 'rb'))

x_pred = vectorizer.transform(x.values.astype('U')).toarray()
y_pred = model.predict(x_pred)
acc = accuracy_score(y_pred,y)
acc = round((acc*100),2)

st.set_page_config(
    page_title = "Access By KAI Review Sentiment Analysis", 
    page_icon = ":🚆:"
)

st.title('PT.KAI Apps Review Sentiment Analysis')

tab1,tab2,tab3,tab4 = st.tabs(['PT.KAI Apps Information','Sentiment Analysis Model Information','Single Prediction','multi Prediction'])
with tab1:
    st.title('PT.KAI Apps')
    st.image('access-kai.png')
    st.write('')
    st.write('Access by KAI adalah aplikasi super yang berfokus pada pemesanan tiket kereta api serta dikembangkan dan diterbitkan oleh PT Kereta Api Indonesia sebagai KAI Access sejak 2014. KAI Access resmi berubah nama menjadi Access by KAI setelah adanya peluncuran oleh PT Kereta Api Indonesia pada 7 Juli 2023 di Stasiun Gambir, Jakarta Pusat.\n \n Diluncurkan pada tanggal 4 September 2014 sebagai KAI Access, mulanya hanya menawarkan fitur pemesanan tiket kereta api baik jarak jauh maupun menengah, tetapi kini Access by KAI melayani pembelian, ubah jadwal, pembatalan, transfer tiket berbagai hingga layanan perkeretaapian PT KAI maupun anak perusahaannya dan akan melayani layanan hotel, logistik, pulsa, serta paket data.')
    st.write('')
    st.write('**Fitur Unggulan Access by KAI:**')
    lst = ['Pembelian Tiket Kereta','Trip Planner','Reservasi Hotel','Live Tracking','Loyalty Point']
    s = ''
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
with tab2:
    st.title('Reviews Sentiment Anaylsis Model')
    st.write('')
    st.write('Model ini dirancang untuk menganalisis sentimen dari ulasan pengguna aplikasi Access by KAI. Dengan menggunakan model ini, pengguna dapat memahami apakah ulasan yang diberikan oleh pengguna lain bersifat positif atau negatif. Model analisis sentimen ini dibuat menggunakan algoritma Multinomial Naive Bayes, yang merupakan salah satu metode yang efektif dalam pengolahan teks dan analisis sentimen.')
    st.write('')
    st.write('**DataSet:**')
    st.write('Dataset yang digunakan untuk melatih model ini adalah data ulasan aplikasi Access by KAI yang diambil langsung dari Google Playstore. Data ini mencakup berbagai ulasan dari pengguna, yang kemudian diproses dan dianalisis untuk membangun model yang akurat dan andal.')
    st.image('distribusi_label.png', caption='Distribusi Jumlah Ulasan Sesuai Sentimen')
    st.write('Pada Gambar di atas memperlihatkan dataset yang berisi ulasan aplikasi access by kai dari user sejumlah 1466 ulasan terbagi menjadi 1125 ulasan negatif dan 341 ulasan positif. Data ini nantinya akan dipisah menjadi data latih dan data uji. Pada penelitian ini dataset akan dipecah dalam bentuk 90% data latih dan 10% data uji.')
    st.write('')
    st.write('**Hasil Uji:**')
    col1, col2 = st.columns(2)
    col1.image('cf-kai.png', use_column_width=True)
    col2.image('Hasilprediksi_ptkai.png',width=435)
    st.write('')
    st.write('Dari 147 ulasan yang termasuk dalam data uji, model ini memprediksi bahwa 140 ulasan memiliki sentimen negatif dan 7 ulasan memiliki sentimen positif. Deskripsi ini menggambarkan kemampuan model dalam mengkategorikan sentimen ulasan pengguna dengan cukup baik.')
with tab3:
    st.title('Single-Predict Model Demo')
    coms = st.text_input('Enter your review about the access by kai app')

    submit = st.button('Predict')

    if submit:
        start = time.time()
        # Transform the input text using the loaded TF-IDF vectorizer
        transformed_text = vectorizer.transform([coms]).toarray()
        #st.write('Transformed text shape:', transformed_text.shape)  # Debugging statement
        # Reshape the transformed text to 2D array
        transformed_text = transformed_text.reshape(1, -1)
        #st.write('Reshaped text shape:', transformed_text.shape)  # Debugging statement
        # Make prediction
        prediction = model.predict(transformed_text)
        end = time.time()
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')

        print(prediction[0])
        if prediction[0] == 1:
            st.title("😆 :green[**Sentimen review anda positif**]")
        else:
            st.title("🤬 :red[**Sentimen review anda negatif**]")
with tab4:
    st.title('Multi-Predict Model Demo')
    sample_csv = df.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

    st.write("")
    st.download_button("Download CSV Example", data=sample_csv, file_name='sample_review.csv', mime='text/csv')

    st.write("")
    st.write("")
    file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        conv_df = vectorizer.transform(uploaded_df['clean_review'].values.astype('U')).toarray()
        prediction_arr = model.predict(conv_df)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            if prediction == 1:
                result = "Sentimen positif"
            else:
                result = "Sentimen Negatif"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)
