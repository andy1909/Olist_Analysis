# review_topic_analysis.py
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from googletrans import Translator
import time

# Import các công cụ từ file TF_IDF.py
# Đảm bảo file TF_IDF.py nằm trong cùng thư mục
try:
    from TF_IDF import preprocess_text, get_top_tfidf_words
except ImportError:
    print("LỖI: Không thể import từ file TF_IDF.py. Vui lòng đảm bảo file này tồn tại trong cùng thư mục.")
    # Định nghĩa các hàm rỗng để tránh lỗi khi import thất bại
    def preprocess_text(text): return text
    def get_top_tfidf_words(corpus, n_top=25): return []

def analyze_reviews_by_score(df, score, n_top=50):
    """
    Phân tích và trích xuất các từ khóa hàng đầu cho một nhóm điểm đánh giá cụ thể.

    Args:
        df (pd.DataFrame): DataFrame chứa cột 'review_score' và 'cleaned_comment'.
        score (int): Điểm số cần phân tích (ví dụ: 1 hoặc 5).
        n_top (int): Số lượng từ khóa hàng đầu cần trích xuất.

    Returns:
        tuple: Một tuple chứa (DataFrame từ khóa, Danh sách tuple từ khóa gốc).
               Trả về (None, None) nếu không có bình luận cho điểm số đó.
    """
    print(f"\n>>> Đang phân tích các đánh giá có điểm = {score}...")

    # Lọc lấy các bình luận của nhóm điểm số này
    reviews_corpus = df[df['review_score'] == score]['cleaned_comment']

    if reviews_corpus.empty or reviews_corpus.isnull().all():
        print(f"    -> Không có bình luận nào hợp lệ cho điểm = {score}.")
        return None, None

    # Lấy các từ khóa hàng đầu bằng TF-IDF
    top_words = get_top_tfidf_words(reviews_corpus, n_top=n_top)

    if not top_words:
        print(f"    -> Không thể trích xuất từ khóa nào cho điểm = {score}.")
        return None, None

    # Dịch các từ khóa sang tiếng Anh
    translated_map = _translate_words(top_words)

    # Tạo DataFrame kết quả để dễ dàng xử lý
    df_keywords = pd.DataFrame(top_words, columns=['keyword_pt', 'tfidf_score'])
    df_keywords['translation_en'] = df_keywords['keyword_pt'].map(translated_map)

    return df_keywords, top_words

def create_wordcloud(word_freq, title, filename, translation_map=None):
    """
    Tạo và lưu hình ảnh Word Cloud từ danh sách tần suất từ.

    Args:
        word_freq (list): Danh sách các tuple (từ, điểm số).
        title (str): Tiêu đề của biểu đồ.
        filename (str): Đường dẫn đầy đủ để lưu file ảnh.
        translation_map (dict, optional): Một dictionary để dịch các từ trước khi vẽ.
    """
    if not word_freq:
        print(f"    ! Cảnh báo: Không có từ nào để tạo Word Cloud cho '{title}'.")
        return

    # Nếu có bản đồ dịch, tạo một dict mới với các từ đã được dịch
    if translation_map:
        # Dùng .get(word, word) để nếu không tìm thấy bản dịch, nó sẽ giữ lại từ gốc
        display_freq = {translation_map.get(word, word): score for word, score in word_freq}
    else:
        display_freq = dict(word_freq)

    try:
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',
            collocations=False # Ngăn WordCloud tự tạo cụm từ
        ).generate_from_frequencies(display_freq)

        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20)

        plt.savefig(filename)
        plt.close()
        print(f"    -> Đã lưu Word Cloud tại: {filename}")
    except Exception as e:
        print(f"    ! Lỗi khi tạo Word Cloud cho '{title}': {e}")


def _translate_words(top_words_list):
    """
    Hàm "private" để dịch một danh sách các từ khóa.
    Được gọi bên trong `analyze_reviews_by_score`.
    """
    if not top_words_list:
        return {}

    translator = Translator()
    translated_dict = {}
    print("    -> Đang dịch các từ khóa quan trọng...")

    count = 0
    for word, score in top_words_list:
        try:
            # Dịch từng từ
            translated = translator.translate(word, src='pt', dest='en')
            translated_dict[word] = translated.text

            # Thêm độ trễ để tránh bị API block
            # Có thể tăng nếu gặp lỗi "Too Many Requests"
            time.sleep(0.3)
            count += 1
            if count % 10 == 0:
                print(f"       ... đã dịch {count}/{len(top_words_list)} từ.")

        except Exception as e:
            translated_dict[word] = "[Translation Error]"
            print(f"       ! Lỗi khi dịch từ '{word}': {e}")

    return translated_dict
