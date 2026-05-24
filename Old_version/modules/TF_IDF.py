import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Tải danh sách stopword tiếng Bồ Đào Nha (chỉ cần chạy lần đầu)
try:
    stopwords.words('portuguese')
except LookupError:
    print(">>> Đang tải danh sách stop words tiếng Bồ Đào Nha...")
    nltk.download('stopwords')
    print("    -> Tải thành công.")

def preprocess_text(text):
    """
    Hàm để làm sạch và chuẩn hóa văn bản tiếng Bồ Đào Nha.
    """
    if not isinstance(text, str):
        return ""
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ các ký tự đặc biệt, số
    text = re.sub(r'[^a-zA-Záàâãéèêíïóôõöúçñ\s]', '', text)
    # Tokenization và loại bỏ stop words
    portuguese_stopwords = set(stopwords.words('portuguese'))
    words = text.split()
    meaningful_words = [w for w in words if w not in portuguese_stopwords and len(w) > 2]
    return " ".join(meaningful_words)

def get_top_tfidf_words(corpus, n_top=25):
    """
    Sử dụng TF-IDF để trích xuất các từ khóa quan trọng nhất từ một tập hợp văn bản.
    Trả về một danh sách các tuple (word, tfidf_score).
    """
    # max_features giới hạn số lượng từ trong từ điển để tăng tốc
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3)) # Thêm ngram_range để bắt cả cụm 2 từ
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Tính tổng điểm TF-IDF cho mỗi từ/cụm từ
    sum_tfidf = tfidf_matrix.sum(axis=0)
    words_freq = [(word, sum_tfidf[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n_top]
