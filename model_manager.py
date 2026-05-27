import logging
import re
from typing import Dict, List, Optional

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "SemanticGenerator"
FALLBACK_MODEL_NAMES = []

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

STOP_WORDS = {"и", "в", "на", "с", "по", "к", "у", "о", "из", "за", "над", "под", "для", "без", "это", "этот", "то", "также", "который", "которая", "которые", "его", "её", "их", "быть", "не", "ни", "что", "как", "так", "вот", "все"}

def _ensure_nltk():
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk()

class ModelManager:
    def __init__(self, model_name: str = None, **kwargs):
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.model_name = f"SemanticGenerator ({EMBEDDING_MODEL_NAME})"
            LOGGER.info("Загружена модель: %s", self.model_name)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки: {e}")

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text, language="russian")
        return [s.strip() for s in sentences if len(s.strip()) > 50]

    @staticmethod
    def _extract_topic(sentence: str) -> str:
        """Извлекает ключевую тему (первые 3-5 значимых слов)."""
        words = sentence.split()
        filtered = [w for w in words if w.lower() not in STOP_WORDS and len(w) > 2]
        if not filtered:
            return " ".join(words[:4])
        topic = " ".join(filtered[:4])
        if len(topic) > 70:
            topic = topic[:67] + "..."
        return topic

    @staticmethod
    def _extract_answer_phrase(sentence: str, max_words: int = 20) -> str:
        """Извлекает короткую ответную фразу (первые max_words слов предложения)."""
        words = sentence.split()
        if len(words) <= max_words:
            return sentence
        return " ".join(words[:max_words])

    @staticmethod
    def _determine_question_type(sentence: str) -> str:
        s_low = sentence.lower()
        if "это" in s_low or "называется" in s_low or "является" in s_low:
            return "what"
        if "происходит в" in s_low or "находится в" in s_low or "расположен" in s_low:
            return "where"
        if "выделяется" in s_low or "образуется" in s_low or "превращается" in s_low:
            return "what_happens"
        return "general"

    def _build_question(self, question_type: str, topic: str) -> str:
        if question_type == "what":
            return f"Что такое «{topic}»?"
        if question_type == "where":
            return f"Где происходит «{topic}»?"
        if question_type == "what_happens":
            return f"Что происходит при «{topic}»?"
        return f"Какое утверждение о «{topic}» верно?"

    def _get_distractors(self, sentences: List[str], correct_phrase: str, num: int = 3) -> List[str]:
        """Выбирает короткие фразы из других предложений, не слишком похожие на правильный ответ."""
        if not sentences:
            return ["Неверно", "Ошибка", "Выдумка"]
        # Эмбеддинги правильного ответа
        correct_emb = self.embedder.encode([correct_phrase])[0].reshape(1, -1)
        # Для каждого предложения берём первые 6-8 слов
        short_phrases = []
        for s in sentences:
            words = s.split()
            if len(words) > 8:
                short = " ".join(words[:8])
            else:
                short = s
            short_phrases.append(short)
        # Вычисляем сходство
        candidates_emb = self.embedder.encode(short_phrases)
        sims = cosine_similarity(correct_emb, candidates_emb).flatten()
        # Берём индексы с наименьшим сходством
        indices = np.argsort(sims)[:num]
        return [short_phrases[i] for i in indices]

    def generate_questions(self, chunk_text: str, num_questions: int = 2) -> List[Dict]:
        if not chunk_text or len(chunk_text.strip()) < 100:
            return []

        sentences = self._split_sentences(chunk_text)
        if len(sentences) < 2:
            return []

        # Эмбеддинги предложений
        embeddings = self.embedder.encode(sentences)
        full_emb = np.mean(embeddings, axis=0).reshape(1, -1)
        sims = cosine_similarity(embeddings, full_emb).flatten()
        # Комбинируем с длиной предложения
        lengths = np.array([len(s) for s in sentences])
        if lengths.max() > lengths.min():
            length_norm = (lengths - lengths.min()) / (lengths.max() - lengths.min())
        else:
            length_norm = np.ones_like(lengths)
        scores = 0.7 * sims + 0.3 * length_norm
        top_indices = np.argsort(scores)[-num_questions:][::-1]

        questions = []
        for idx in top_indices:
            sent = sentences[idx]
            topic = self._extract_topic(sent)
            q_type = self._determine_question_type(sent)
            question_text = self._build_question(q_type, topic)

            correct_phrase = self._extract_answer_phrase(sent, max_words=12)
            if len(correct_phrase) < 15:
                correct_phrase = " ".join(sent.split()[:8])

            other_sentences = [sentences[i] for i in range(len(sentences)) if i != idx]
            distractors = self._get_distractors(other_sentences, correct_phrase, num=3)

            options = [
                f"А) {correct_phrase}",
                f"Б) {distractors[0]}",
                f"В) {distractors[1]}",
                f"Г) {distractors[2]}",
            ]

            questions.append({
                "question": question_text,
                "options": options,
                "correct": "А"
            })

        return questions