# src/nlp.py
import re
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
try:
    from googletrans import Translator
    HAS_GOOGLETRANS = True
except Exception:
    HAS_GOOGLETRANS = False
    print(">>> [NLP] Warning: googletrans import failed (commonly due to 'cgi' module removal in Python 3.13+).")
    print("    - Falling back to returning original text for translations.")
    class Translator:
        def translate(self, text, src='pt', dest='en'):
            class Translation:
                def __init__(self, t):
                    self.text = t
            return Translation(text)

# Download nltk Portuguese stopwords if not available
try:
    stopwords.words('portuguese')
except LookupError:
    print(">>> [NLP] Downloading Portuguese stopwords...")
    nltk.download('stopwords')
    print("    - Download complete.")

def preprocess_text(text):
    """
    Clean and normalize Portuguese text (lowercase, regex cleaning, stopword removal).
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Záàâãéèêíïóôõöúçñ\s]', '', text)
    # Tokenize and remove stopwords
    portuguese_stopwords = set(stopwords.words('portuguese'))
    words = text.split()
    meaningful_words = [w for w in words if w not in portuguese_stopwords and len(w) > 2]
    return " ".join(meaningful_words)

def get_top_tfidf_words(corpus, n_top=25):
    """
    Use TF-IDF Vectorizer to extract top keywords/ngrams from a text corpus.
    """
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    sum_tfidf = tfidf_matrix.sum(axis=0)
    words_freq = [(word, sum_tfidf[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n_top]

def analyze_reviews_by_score(df, score, n_top=50):
    """
    Analyze customer reviews for a given score, returning top TF-IDF keywords and English translations.
    """
    print(f"\n>>> [NLP] Analyzing reviews with rating = {score}...")
    reviews_corpus = df[df['review_score'] == score]['cleaned_comment']

    if reviews_corpus.empty or reviews_corpus.isnull().all():
        print(f"    - No valid reviews found for score = {score}.")
        return None, None

    top_words = get_top_tfidf_words(reviews_corpus, n_top=n_top)
    if not top_words:
        print(f"    - Failed to extract keywords for score = {score}.")
        return None, None

    translated_map = _translate_words(top_words)

    df_keywords = pd.DataFrame(top_words, columns=['keyword_pt', 'tfidf_score'])
    df_keywords['translation_en'] = df_keywords['keyword_pt'].map(translated_map)

    return df_keywords, top_words

def create_wordcloud(word_freq, title, filename, translation_map=None):
    """
    Create and save a WordCloud from a word frequency list/dictionary.
    """
    if not word_freq:
        print(f"    ! Warning: Word frequencies list empty for cloud '{title}'.")
        return

    if translation_map:
        display_freq = {translation_map.get(word, word): score for word, score in word_freq}
    else:
        display_freq = dict(word_freq)

    try:
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',
            collocations=False
        ).generate_from_frequencies(display_freq)

        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20)

        plt.savefig(filename)
        plt.close()
        print(f"    - Saved WordCloud: {filename}")
    except Exception as e:
        print(f"    ! Error creating WordCloud for '{title}': {e}")

def _translate_words(top_words_list):
    """
    Translate Portuguese keywords to English via Google Translate API (with rate-limiting delay).
    """
    if not top_words_list:
        return {}

    translator = Translator()
    translated_dict = {}
    print("    - Translating top keywords...")

    count = 0
    for word, score in top_words_list:
        try:
            translated = translator.translate(word, src='pt', dest='en')
            translated_dict[word] = translated.text
            time.sleep(0.3)  # Anti-blocking sleep
            count += 1
            if count % 10 == 0:
                print(f"       ... translated {count}/{len(top_words_list)} words.")
        except Exception as e:
            translated_dict[word] = "[Translation Error]"
            print(f"       ! Error translating '{word}': {e}")

    return translated_dict
