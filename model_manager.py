import logging
import re
from typing import Dict, List, Optional

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "SemanticGenerator"
FALLBACK_MODEL_NAMES = []  # не используются

# Модель для эмбеддингов
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Шаблоны вопросов
QUESTION_TEMPLATES = [
    "Что из текста верно относительно «{topic}»?",
    "Какое утверждение о «{topic}» соответствует содержанию?",
    "Что говорится в тексте о «{topic}»?",
    "Какой факт о «{topic}» подтверждается текстом?",
]

FALLBACK_DISTRACTORS = [
    "Неверное утверждение",
    "Выдуманный факт",
    "Данные отсутствуют",
]

OPTION_LETTERS = ["А", "Б", "В", "Г"]


def _ensure_nltk_resources() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


class ModelManager:
    def __init__(self, model_name: str = None, **kwargs):
        _ensure_nltk_resources()
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.model_name = f"SemanticGenerator ({EMBEDDING_MODEL_NAME})"
            LOGGER.info("Загружена эмбеддинг-модель: %s", self.model_name)
        except Exception as e:
            LOGGER.error("Ошибка загрузки модели: %s", e)
            raise RuntimeError(f"Не удалось загрузить модель: {e}")

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text, language="russian")
        return [s.strip() for s in sentences if len(s.strip()) > 30]

    @staticmethod
    def _extract_topic(sentence: str) -> str:
        words = sentence.split()
        stop_words = {"и", "в", "на", "с", "по", "к", "у", "о", "из", "за", "над", "под", "для", "без", "это", "этот"}
        filtered = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        if not filtered:
            return " ".join(words[:5])
        topic = " ".join(filtered[:5])
        return topic[:70]

    def _select_key_sentences(self, sentences: List[str], top_k: int = 3) -> List[str]:
        if not sentences:
            return []
        if len(sentences) <= top_k:
            return sentences

        embeddings = self.embedder.encode(sentences)
        full_emb = np.mean(embeddings, axis=0).reshape(1, -1)
        similarities = cosine_similarity(embeddings, full_emb).flatten()
        lengths = np.array([len(s) for s in sentences])
        length_norm = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-6)
        scores = 0.7 * similarities + 0.3 * length_norm
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [sentences[i] for i in top_indices]

    def _generate_distractors(self, correct_sentence: str, other_sentences: List[str], num_needed: int = 3) -> List[str]:
        if not other_sentences:
            return FALLBACK_DISTRACTORS[:num_needed]

        correct_emb = self.embedder.encode([correct_sentence])[0].reshape(1, -1)
        other_embs = self.embedder.encode(other_sentences)
        similarities = cosine_similarity(correct_emb, other_embs).flatten()

        # берём непохожие (чем меньше сходство, тем лучше, но не слишком)
        # отсортируем по убыванию сходства и возьмём те, что ниже порога 0.6
        sorted_idx = np.argsort(similarities)[::-1]
        candidates = []
        for idx in sorted_idx:
            if similarities[idx] < 0.6:
                candidates.append(other_sentences[idx])
            if len(candidates) >= num_needed:
                break
        while len(candidates) < num_needed:
            candidates.append(FALLBACK_DISTRACTORS[len(candidates) % len(FALLBACK_DISTRACTORS)])
        return candidates[:num_needed]

    def generate_questions(self, chunk_text: str, num_questions: int = 2) -> List[Dict]:
        if not chunk_text or len(chunk_text.strip()) < 50:
            return []

        sentences = self._split_sentences(chunk_text)
        if len(sentences) < 2:
            return []

        key_sentences = self._select_key_sentences(sentences, top_k=num_questions)
        if not key_sentences:
            return []

        questions = []
        for idx, key_sent in enumerate(key_sentences[:num_questions]):
            topic = self._extract_topic(key_sent)
            template = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)]
            question_text = template.format(topic=topic)

            correct_text = key_sent[:120]
            correct_option = f"А) {correct_text}"

            other_sentences = [s for s in sentences if s != key_sent]
            distractors = self._generate_distractors(key_sent, other_sentences, num_needed=3)
            distractor_options = [
                f"Б) {distractors[0][:120]}",
                f"В) {distractors[1][:120]}",
                f"Г) {distractors[2][:120]}",
            ]
            options = [correct_option] + distractor_options

            questions.append({
                "question": question_text,
                "options": options,
                "correct": "А"
            })

        return questions