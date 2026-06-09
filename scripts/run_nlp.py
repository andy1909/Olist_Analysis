# scripts/run_nlp.py
import os
import time
import pandas as pd
import warnings
import multiprocessing
try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

try:
    from googletrans import Translator
except Exception:
    class Translator:
        def translate(self, text, src='pt', dest='en'):
            class Translation:
                def __init__(self, t):
                    self.text = t
            return Translation(text)

from src.nlp import preprocess_text, create_wordcloud

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def train_word2vec_model():
    """
    Train Gensim Word2Vec on the review comments dataset.
    """
    print("\n====== TRAINING WORD2VEC MODEL ======")
    review_path = os.path.join(RAW_DATA_DIR, 'olist_order_reviews_dataset.csv')
    if not os.path.exists(review_path):
        print(f"    ! Error: Review dataset not found at {review_path}")
        return None

    print(">>> [1] Loading and preprocessing text corpus...")
    df = pd.read_csv(review_path, usecols=['review_comment_message'])
    df.dropna(subset=['review_comment_message'], inplace=True)

    corpus = [preprocess_text(comment).split() for comment in df['review_comment_message']]
    print(f"    - Preprocessed {len(corpus)} reviews for training.")

    print(">>> [2] Training Word2Vec model (this may take a moment)...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=150,
        window=5,
        min_count=5,
        workers=max(1, multiprocessing.cpu_count() - 1)
    )
    print("    - Training complete.")

    model_path = os.path.join(MODEL_DIR, 'word2vec_reviews.model')
    model.save(model_path)
    print(f"    - Saved Word2Vec model to: {model_path}")
    return model

def analyze_topics_with_word2vec(model):
    """
    Find words most similar to predefined seed keywords for different sentiment categories.
    """
    print("\n====== ANALYZING TOPICS WITH WORD2VEC ======")
    seed_keywords = {
        'positive_delivery': ['rápido', 'antes', 'prazo'],
        'positive_product': ['bom', 'ótimo', 'excelente', 'qualidade'],
        'negative_delivery': ['recebi', 'atraso', 'ainda', 'chegou'],
        'negative_product': ['errado', 'veio', 'diferente', 'problema']
    }

    all_related_words = []

    for topic, keywords in seed_keywords.items():
        print(f">>> [Topic Analysis] Topic: {topic}")
        # Filter keywords that exist in model vocabulary
        valid_keywords = [w for w in keywords if w in model.wv]
        if not valid_keywords:
            print(f"    ! Warning: None of the seed keywords for {topic} exist in vocab.")
            continue
        try:
            similar_words = model.wv.most_similar(positive=valid_keywords, topn=20)
            df_topic = pd.DataFrame(similar_words, columns=['keyword_pt', 'similarity_score'])
            df_topic['topic'] = topic
            all_related_words.append(df_topic)
            print(f"    - Most similar words: {[word for word, score in similar_words[:5]]}")
        except Exception as e:
            print(f"    ! Error calculating similarities for {topic}: {e}")

    if all_related_words:
        final_df = pd.concat(all_related_words, ignore_index=True)
        translator = Translator()
        print("\n>>> [Translation] Translating unique keywords to English...")
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
        output_path = os.path.join(PROCESSED_DATA_DIR, 'word_embeddings_topic_analysis.csv')
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Topic analysis saved to: {output_path}")
        return output_path
    return None

def create_aggregated_wordclouds(csv_path, n_top=50):
    """
    Read the output CSV, aggregate topics into 'Positive' and 'Negative', and generate WordClouds.
    """
    print("\n====== CREATING AGGREGATED WORD CLOUDS ======")
    if not os.path.exists(csv_path):
        print(f"    ! Error: Analysis file not found at {csv_path}")
        return

    df_topics = pd.read_csv(csv_path)
    
    # 1. POSITIVE CLOUD
    print(">>> [WordCloud] Building Positive cloud...")
    df_positive = df_topics[df_topics['topic'].str.startswith('positive')].copy()
    df_positive_agg = df_positive.groupby('translation_en')['similarity_score'].max().sort_values(ascending=False).head(n_top)
    
    if not df_positive_agg.empty:
        positive_freq = df_positive_agg.to_dict()
        save_path = os.path.join(FIGURES_DIR, 'wordcloud_embedding_AGGREGATED_positive.png')
        create_wordcloud(list(positive_freq.items()), 'Positive Feedback Embeddings', save_path)
    else:
        print("    - No positive topic data found.")

    # 2. NEGATIVE CLOUD
    print(">>> [WordCloud] Building Negative cloud...")
    df_negative = df_topics[df_topics['topic'].str.startswith('negative')].copy()
    df_negative_agg = df_negative.groupby('translation_en')['similarity_score'].max().sort_values(ascending=False).head(n_top)
    
    if not df_negative_agg.empty:
        negative_freq = df_negative_agg.to_dict()
        save_path = os.path.join(FIGURES_DIR, 'wordcloud_embedding_AGGREGATED_negative.png')
        create_wordcloud(list(negative_freq.items()), 'Negative Feedback Embeddings', save_path)
    else:
        print("    - No negative topic data found.")

def main():
    print("====================================================")
    print("STARTING NLP EMBEDDINGS AND TOPIC ANALYSIS RUNNER")
    print("====================================================")
    
    if not HAS_GENSIM:
        print("\n[WARNING] Gensim package is not installed.")
        print("This package cannot compile/install on Python 3.14+ due to C-API changes.")
        print("To run Word2Vec training and NLP analysis, please run this script in an environment with Python 3.8 - 3.12.")
        print("\n====================================================")
        print("NLP ANALYSIS SKIPPED (ENVIRONMENT COMPATIBILITY)")
        print("====================================================")
        return
        
    model_path = os.path.join(MODEL_DIR, 'word2vec_reviews.model')
    if os.path.exists(model_path):
        print(f"Loading pre-trained Word2Vec model from {model_path}...")
        w2v_model = Word2Vec.load(model_path)
    else:
        w2v_model = train_word2vec_model()

    if w2v_model:
        # Run similarity analysis
        csv_path = analyze_topics_with_word2vec(w2v_model)
        
        # Draw WordClouds
        if csv_path:
            create_aggregated_wordclouds(csv_path)
            
    print("\n====================================================")
    print("NLP ANALYSIS COMPLETE")
    print("====================================================")

if __name__ == "__main__":
    main()
