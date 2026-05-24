import os
import pandas as pd
import warnings
from gensim.models import Word2Vec
import multiprocessing
import time
from googletrans import Translator

# Import hàm tiền xử lý từ file TF_IDF.py
try:
    from modules.TF_IDF import preprocess_text
except ImportError:
    print("LỖI: Không thể import từ file TF_IDF.py.")
    def preprocess_text(text): return text

# Cấu hình
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'RawData')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_word2vec_model():
    """
    Huấn luyện mô hình Word2Vec trên toàn bộ kho bình luận và lưu lại.
    """
    print("====== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH WORD2VEC ======")

    # 1. Nạp và tiền xử lý toàn bộ bình luận
    print(">>> [1] Đang nạp và tiền xử lý văn bản...")
    review_path = os.path.join(RAW_DATA_DIR, 'olist_order_reviews_dataset.csv')
    df = pd.read_csv(review_path, usecols=['review_comment_message'])
    df.dropna(subset=['review_comment_message'], inplace=True)

    # Tiền xử lý và tách các bình luận thành danh sách các từ
    # [['produto', 'bom', 'entrega', 'rapida'], ['nao', 'recebi', 'ainda'], ...]
    corpus = [preprocess_text(comment).split() for comment in df['review_comment_message']]

    print(f"    -> Đã xử lý {len(corpus)} bình luận để huấn luyện.")

    # 2. Huấn luyện mô hình Word2Vec
    print(">>> [2] Đang huấn luyện mô hình Word2Vec (có thể mất vài phút)...")

    # Các tham số quan trọng:
    # vector_size: Số chiều của vector từ (100-300 là phổ biến)
    # window: Cửa sổ ngữ cảnh (số từ xung quanh một từ để học)
    # min_count: Bỏ qua các từ xuất hiện quá ít
    # workers: Tận dụng đa lõi CPU để tăng tốc
    model = Word2Vec(
        sentences=corpus,
        vector_size=150,
        window=5,
        min_count=5,
        workers=multiprocessing.cpu_count() - 1
    )

    print("    -> Huấn luyện hoàn tất.")

    # 3. Lưu mô hình đã huấn luyện
    model_path = os.path.join(MODEL_DIR, 'word2vec_reviews.model')
    model.save(model_path)
    print(f"\n✅ Mô hình Word2Vec đã được lưu tại: {model_path}")

    return model

def analyze_topics_with_word2vec(model):
    """
    Sử dụng mô hình Word2Vec đã huấn luyện để tìm các từ liên quan đến các chủ đề chính.
    """
    print("\n====== BẮT ĐẦU PHÂN TÍCH CHỦ ĐỀ VỚI WORD2VEC ======")

    # --- Định nghĩa các "từ khóa hạt nhân" cho các chủ đề chúng ta quan tâm ---
    # Dựa trên kết quả TF-IDF, chúng ta đã biết các chủ đề này rất quan trọng
    seed_keywords = {
        'positive_delivery': ['rápido', 'antes', 'prazo'], # Giao hàng nhanh/trước hạn
        'positive_product': ['bom', 'ótimo', 'excelente', 'qualidade'], # Sản phẩm tốt
        'negative_delivery': ['recebi', 'atraso', 'ainda', 'chegou'], # Không nhận được/chậm trễ
        'negative_product': ['errado', 'veio', 'diferente', 'problema'] # Sản phẩm sai/có vấn đề
    }

    all_related_words = []

    for topic, keywords in seed_keywords.items():
        print(f">>> [Phân tích chủ đề]: {topic}")
        try:
            # Tìm 20 từ có vector gần nhất với TỔNG HỢP vector của các từ khóa hạt nhân
            # most_similar sẽ tìm các từ có độ tương đồng cosine cao nhất
            similar_words = model.wv.most_similar(positive=keywords, topn=20)

            # Tạo DataFrame cho chủ đề này
            df_topic = pd.DataFrame(similar_words, columns=['keyword_pt', 'similarity_score'])
            df_topic['topic'] = topic
            all_related_words.append(df_topic)

            print(f"    -> Các từ liên quan nhất: {[word for word, score in similar_words[:5]]}")

        except KeyError as e:
            print(f"    ! Cảnh báo: Một trong các từ khóa hạt nhân không có trong từ điển: {e}")
            continue

    # 4. Ghép các kết quả và lưu file CSV
    if all_related_words:
        final_df = pd.concat(all_related_words, ignore_index=True)

        # Thêm cột dịch thuật
        translator = Translator()
        print("\n>>> [Dịch thuật] Đang dịch các từ khóa...")

        # Dịch các từ duy nhất để tiết kiệm thời gian
        unique_words = final_df['keyword_pt'].unique()
        translation_map = {}
        for word in unique_words:
            try:
                translated = translator.translate(word, src='pt', dest='en')
                translation_map[word] = translated.text
                time.sleep(0.3)
            except Exception:
                translation_map[word] = "[Translation Error]"

        final_df['translation_en'] = final_df['keyword_pt'].map(translation_map)

        output_path = os.path.join(OUTPUT_DIR, 'word_embeddings_topic_analysis.csv')
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Phân tích hoàn tất. Dữ liệu đã lưu tại: {output_path}")

if __name__ == "__main__":
    # Bước 1: Huấn luyện mô hình (hoặc nạp nếu đã có)
    model_path = os.path.join(MODEL_DIR, 'word2vec_reviews.model')
    if os.path.exists(model_path):
        print("Đang nạp mô hình Word2Vec đã có...")
        w2v_model = Word2Vec.load(model_path)
    else:
        w2v_model = train_word2vec_model()

    # Bước 2: Phân tích chủ đề bằng mô hình đã có
    analyze_topics_with_word2vec(w2v_model)
