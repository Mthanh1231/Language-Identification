import streamlit as st
import pandas as pd
import re
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
from docx import Document
import fitz  # PyMuPDF

# CSS tuỳ chỉnh để chỉnh giao diện theo yêu cầu
st.markdown("""
    <style>
        /* Nền tổng thể */
        body {
            background-color: #1e1e1e;
            color: #f0f2f5;
        }
        
        /* Tiêu đề chính */
        .css-18e3th9 {
            color: #5A9BD5 !important;
        }
        
        /* Vùng nhập văn bản */
        .stTextArea textarea {
            background-color: #2e2e2e;
            color: white;
            border: 1px solid #5A9BD5;
        }
        
        /* Nút 'Detect Language' */
        .stButton button {
            background-color: #0073E6;
            color: white;
        }
        .stButton button:hover {
            background-color: #005BB5;
            color: white;
        }
        
        /* Khung kết quả dự đoán */
        .result-box {
            background-color: #d1e7dd;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
        }
        
        /* Lịch sử dịch thuật */
        .stDataFrame {
            background-color: #2e2e2e;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Đường dẫn file lịch sử dự đoán
history_file_path = r"C:\Users\GIGABYTE\OneDrive\Máy tính\quản trị dự án\detection_history.csv"

# Đọc dữ liệu từ tập tin CSV
raw = pd.read_csv('C:/Users/GIGABYTE/Downloads/dataset.csv')
X = raw['Text']
y = raw['language']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Định nghĩa lớp tiền xử lý
class TextCleaner(BaseEstimator, TransformerMixin):
    def clean_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self.clean_text(text) for text in X]

# Tạo một pipeline
pipeline = Pipeline([
    ('text_cleaner', TextCleaner()),
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 2), min_df=1e-2)),
    ('classifier', MultinomialNB())
])

# Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# Hàm lưu kết quả dự đoán vào file lịch sử
def save_prediction_history(source_name, predicted_language):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(history_file_path):
        history_df = pd.read_csv(history_file_path)
    else:
        history_df = pd.DataFrame(columns=['Timestamp', 'Source', 'Predicted Language'])
    new_record = {'Timestamp': timestamp, 'Source': source_name, 'Predicted Language': predicted_language}
    history_df = pd.concat([history_df, pd.DataFrame([new_record])], ignore_index=True)
    history_df.to_csv(history_file_path, index=False)

# Hàm đọc file DOCX
def load_text_from_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Hàm đọc file PDF
def load_text_from_pdf(file_path):
    pdf_text = []
    with fitz.open(file_path) as pdf:
        for page in pdf:
            pdf_text.append(page.get_text())
    return '\n'.join(pdf_text)

# Hàm đọc nội dung từ URL
def load_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load content from URL: {e}")
        return None

# Giao diện ứng dụng
st.title("Language Detection Application")
input_type = st.selectbox("Select Input Type", ("Text", "DOCX File", "PDF File", "URL"))

if input_type == "Text":
    input_text = st.text_area("Enter text here:")
    if st.button("Detect Language"):
        if input_text.strip():
            predicted_language = pipeline.predict([input_text])[0]
            st.markdown(f'<div class="result-box">Predicted Language: {predicted_language}</div>', unsafe_allow_html=True)
            save_prediction_history("Text Input", predicted_language)
        else:
            st.warning("Please enter text for detection.")

elif input_type == "DOCX File":
    uploaded_file = st.file_uploader("Upload DOCX file", type="docx")
    if st.button("Detect Language"):
        if uploaded_file:
            text = load_text_from_docx(uploaded_file)
            predicted_language = pipeline.predict([text])[0]
            st.markdown(f'<div class="result-box">Predicted Language: {predicted_language}</div>', unsafe_allow_html=True)
            save_prediction_history("DOCX File", predicted_language)
        else:
            st.warning("Please upload a DOCX file to detect its language.")

elif input_type == "PDF File":
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    if st.button("Detect Language"):
        if uploaded_file:
            text = load_text_from_pdf(uploaded_file)
            predicted_language = pipeline.predict([text])[0]
            st.markdown(f'<div class="result-box">Predicted Language: {predicted_language}</div>', unsafe_allow_html=True)
            save_prediction_history("PDF File", predicted_language)
        else:
            st.warning("Please upload a PDF file to detect its language.")

elif input_type == "URL":
    url = st.text_input("Enter URL:")
    if st.button("Detect Language"):
        if url.strip():
            text = load_text_from_url(url)
            if text:
                predicted_language = pipeline.predict([text])[0]
                st.markdown(f'<div class="result-box">Predicted Language: {predicted_language}</div>', unsafe_allow_html=True)
                save_prediction_history("URL", predicted_language)
            else:
                st.error("Unable to retrieve text from the provided URL.")
        else:
            st.warning("Please enter a valid URL for detection.")

# Phần kiểm tra lịch sử dịch thuật
st.subheader("Translation History")
if os.path.exists(history_file_path):
    history_df = pd.read_csv(history_file_path)
    st.dataframe(history_df)
else:
    st.write("No translation history found.")
