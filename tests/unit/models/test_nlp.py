# tests/unit/models/test_nlp.py
"""
Unit tests for src/models/nlp.py

Covers:
    - preprocess_text         : text cleaning and stopword removal
    - get_top_tfidf_words     : TF-IDF keyword extraction
    - create_wordcloud        : WordCloud file generation

External translation calls (_translate_words / Translator) are not tested here
because they require a live network connection.  Integration-level translation
tests live in tests/integration/.
"""
import os
import shutil
import tempfile
import unittest

from src.models.nlp import preprocess_text, get_top_tfidf_words, create_wordcloud


class TestPreprocessText(unittest.TestCase):
    """Tests for preprocess_text()."""

    def test_lowercases_all_characters(self):
        result = preprocess_text("PRODUTO BOM ENTREGA RÁPIDA")
        self.assertEqual(result, result.lower())

    def test_preserves_accented_portuguese_characters(self):
        # ã, é, ç, ó must not be stripped by the regex
        result = preprocess_text("ótimo produto qualidade excelente")
        self.assertIn('ótimo', result)

    def test_removes_digits_and_special_characters(self):
        result = preprocess_text("produto! #123 @entrega")
        self.assertNotIn('!', result)
        self.assertNotIn('#', result)
        self.assertNotIn('@', result)
        self.assertNotIn('123', result)

    def test_removes_common_portuguese_stopwords(self):
        # 'de', 'do', 'da', 'que' are all Portuguese stopwords
        result = preprocess_text("entrega do produto de qualidade que chegou")
        tokens = result.split()
        for stopword in ('do', 'de', 'que'):
            self.assertNotIn(stopword, tokens, msg=f"Stopword '{stopword}' was not removed.")

    def test_removes_tokens_shorter_than_three_characters(self):
        # single-letter and two-letter tokens must be removed
        result = preprocess_text("o produto foi entregue na casa")
        tokens = result.split()
        for token in tokens:
            self.assertGreater(len(token), 2,
                               msg=f"Short token '{token}' was not filtered out.")

    def test_returns_empty_string_when_input_is_none(self):
        self.assertEqual(preprocess_text(None), "")

    def test_returns_empty_string_when_input_is_not_a_string(self):
        self.assertEqual(preprocess_text(42), "")
        self.assertEqual(preprocess_text(['lista']), "")

    def test_returns_empty_string_for_empty_input(self):
        self.assertEqual(preprocess_text(""), "")

    def test_handles_text_with_only_stopwords_returning_empty_or_minimal_output(self):
        # All words are stopwords → result should be empty or very short
        result = preprocess_text("de do da que")
        self.assertEqual(result.strip(), "")


class TestGetTopTfidfWords(unittest.TestCase):
    """Tests for get_top_tfidf_words()."""

    def setUp(self):
        self.corpus = [
            "produto muito bom qualidade excelente entrega rápida",
            "entrega rápida produto chegou antes prazo",
            "qualidade produto excelente recomendo muito",
            "produto errado veio diferente cor problema",
            "atraso entrega produto ainda não chegou",
        ]

    def test_returns_list_of_word_score_tuples(self):
        result = get_top_tfidf_words(self.corpus, n_top=5)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_respects_n_top_limit(self):
        result = get_top_tfidf_words(self.corpus, n_top=3)
        self.assertLessEqual(len(result), 3)

    def test_scores_are_positive_floats(self):
        result = get_top_tfidf_words(self.corpus, n_top=10)
        for _, score in result:
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0.0)

    def test_results_are_sorted_descending_by_score(self):
        result = get_top_tfidf_words(self.corpus, n_top=10)
        scores = [score for _, score in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_extracts_meaningful_keywords_from_corpus(self):
        # 'produto' appears in every document → high TF, low IDF → may or may not top
        # 'rápida' is distinctive → should appear somewhere in top results
        result = get_top_tfidf_words(self.corpus, n_top=20)
        words = [w for w, _ in result]
        # At least one known domain keyword must be present
        domain_keywords = {'produto', 'entrega', 'qualidade', 'rápida', 'atraso'}
        found = domain_keywords.intersection(set(words))
        self.assertGreater(len(found), 0,
                           msg=f"None of {domain_keywords} found in top words: {words}")


class TestCreateWordcloud(unittest.TestCase):
    """Tests for create_wordcloud()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.tmp_dir, 'test_wordcloud.png')
        self.word_freq = [
            ('fast', 0.9), ('delivery', 0.8), ('product', 0.7),
            ('quality', 0.6), ('excellent', 0.5),
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_saves_png_file_to_specified_path(self):
        create_wordcloud(self.word_freq, title='Test Cloud', filename=self.output_path)
        self.assertTrue(
            os.path.exists(self.output_path),
            msg="WordCloud PNG was not created at the expected path."
        )

    def test_output_file_is_non_empty(self):
        create_wordcloud(self.word_freq, title='Test Cloud', filename=self.output_path)
        file_size = os.path.getsize(self.output_path)
        self.assertGreater(file_size, 0, msg="WordCloud PNG file is empty.")

    def test_does_not_raise_when_word_freq_is_empty(self):
        """Empty frequency list should log a warning but never crash."""
        try:
            create_wordcloud([], title='Empty Cloud', filename=self.output_path)
        except Exception as e:
            self.fail(f"create_wordcloud raised unexpectedly with empty input: {e}")


if __name__ == '__main__':
    unittest.main()
