# visualize_embeddings.py
import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'Outputs')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'Charts')
sns.set_theme(style="whitegrid")

def create_aggregated_wordclouds(csv_path, n_top=50):
    """
    Đọc file CSV, gộp các chủ đề thành 'Positive' và 'Negative',
    và tạo 2 Word Cloud tổng hợp.
    """
    print("====== STARTING AGGREGATED EMBEDDINGS VISUALIZATION ======")

    if not os.path.exists(csv_path):
        print(f"LỖI: Không tìm thấy file {csv_path}. Hãy chạy 'word_embeddings_analysis.py' trước.")
        return

    # 1. Đọc dữ liệu từ file CSV
    df_topics = pd.read_csv(csv_path)
    print(f">>> [1] Đã nạp thành công file phân tích chủ đề.")

    # --- THAY ĐỔI LOGIC: GỘP CHỦ ĐỀ ---

    # 2. Xử lý nhóm TÍCH CỰC (POSITIVE)
    print(">>> [2] Đang tạo Word Cloud tổng hợp cho nhóm 'Positive'...")
    df_positive = df_topics[df_topics['topic'].str.startswith('positive')].copy()

    # Do có thể có các từ trùng lặp giữa các chủ đề con (ví dụ 'ótimo' trong cả delivery và product),
    # chúng ta nên nhóm lại và lấy điểm similarity cao nhất cho mỗi từ.
    df_positive_agg = df_positive.groupby('translation_en')['similarity_score'].max().sort_values(ascending=False).head(n_top)

    if not df_positive_agg.empty:
        positive_freq = df_positive_agg.to_dict()

        wordcloud_pos = WordCloud(
            width=1200, height=600, background_color='white', colormap='viridis', collocations=False
        ).generate_from_frequencies(positive_freq)

        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title(' ', fontsize=20)

        save_path_pos = os.path.join(CHARTS_DIR, 'wordcloud_embedding_AGGREGATED_positive.png')
        plt.savefig(save_path_pos, bbox_inches='tight')
        plt.close()
        print(f"    -> Đã lưu tại: {save_path_pos}")

    # 3. Xử lý nhóm TIÊU CỰC (NEGATIVE)
    print("\n>>> [3] Đang tạo Word Cloud tổng hợp cho nhóm 'Negative'...")
    df_negative = df_topics[df_topics['topic'].str.startswith('negative')].copy()

    # Tương tự, nhóm lại và lấy điểm cao nhất
    df_negative_agg = df_negative.groupby('translation_en')['similarity_score'].max().sort_values(ascending=False).head(n_top)

    if not df_negative_agg.empty:
        negative_freq = df_negative_agg.to_dict()

        wordcloud_neg = WordCloud(
            width=1200, height=600, background_color='white', colormap='plasma', collocations=False
        ).generate_from_frequencies(negative_freq)

        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title(' ', fontsize=20)

        save_path_neg = os.path.join(CHARTS_DIR, 'wordcloud_embedding_AGGREGATED_negative.png')
        plt.savefig(save_path_neg, bbox_inches='tight')
        plt.close()
        print(f"    -> Đã lưu tại: {save_path_neg}")
    else:
        print("    -> Không tìm thấy dữ liệu cho nhóm 'Negative' để vẽ.")


    print("\n====== AGGREGATED VISUALIZATION COMPLETE ======")


if __name__ == "__main__":
    # Đường dẫn đến file kết quả từ Word Embeddings
    embeddings_analysis_file = os.path.join(OUTPUT_DIR, 'word_embeddings_topic_analysis.csv')

    # Gọi hàm để thực hiện
    create_aggregated_wordclouds(embeddings_analysis_file)
