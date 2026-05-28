import json
import logging
import re
from typing import Dict, List, Optional
import numpy as np
import nltk
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)

# Константы для совместимости с app.py
MODEL_NAME = "Llama3_8b_Q3"
FALLBACK_MODEL_NAMES = []  # не используется

# Путь к скачанной модели
MODEL_PATH = "./models/saiga_llama3_8b.Q3_K_S.gguf"

# Параметры генерации
GENERATION_KWARGS = {
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
    "stop": ["<|end|>", "<|user|>", "<|assistant|>", "\n```\n"],
}

# Модель для эмбеддингов (для fallback)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Стоп-слова для fallback
STOP_WORDS = {"и", "в", "на", "с", "по", "к", "у", "о", "из", "за", "над", "под", "для", "без", "это", "этот", "то", "также", "который", "которая", "которые", "его", "её", "их", "быть", "не", "ни", "что", "как", "так", "вот", "все"}

OPTION_LETTERS = ["А", "Б", "В", "Г"]
CORRECT_LETTER_MAP = {
    "A": "А", "B": "Б", "C": "В", "D": "Г",
    "А": "А", "Б": "Б", "В": "В", "Г": "Г",
    "а": "А", "б": "Б", "в": "В", "г": "Г",
}

# -------------------- Утилиты для LLM --------------------
def _normalize_correct_letter(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    first = value[0]
    return CORRECT_LETTER_MAP.get(first, CORRECT_LETTER_MAP.get(value.upper(), ""))

def _normalize_options(options: List) -> List[str]:
    """Приводит варианты к единому формату с А), Б), В), Г)."""
    norm = []
    for i, opt in enumerate(options):
        opt_str = str(opt).strip()
        # Убираем существующий префикс типа "А) ", "А.", "А"
        opt_str = re.sub(r"^[АБВГABCD]\)?\s*", "", opt_str)
        norm.append(f"{OPTION_LETTERS[i]}) {opt_str}")
    while len(norm) < 4:
        norm.append(f"{OPTION_LETTERS[len(norm)]}) ...")
    return norm[:4]

def _validate_questions(items: List[Dict]) -> List[Dict]:
    """Проверяет и чистит каждый вопрос."""
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        opts = item.get("options", [])
        cor = _normalize_correct_letter(item.get("correct", ""))
        if not q or not isinstance(opts, list) or len(opts) < 2:
            continue
        if cor not in OPTION_LETTERS:
            continue
        opts = _normalize_options(opts)
        valid.append({
            "question": q,
            "options": opts,
            "correct": cor,
        })
    return valid

def _extract_json_from_text(raw: str) -> Optional[List[Dict]]:
    """Извлекает JSON из ответа модели."""
    raw = raw.strip()
    # Убираем маркеры кода
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        json_str = raw[start:end+1]
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            # Пробуем очистить от висячих запятых
            cleaned = re.sub(r",\s*]", "]", json_str)
            try:
                data = json.loads(cleaned)
                if isinstance(data, list):
                    return data
            except:
                pass
    return None

# -------------------- Fallback (семантический генератор) --------------------
def _ensure_nltk():
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
_ensure_nltk()

class ModelManager:
    def __init__(self, model_name: str = None, **kwargs):
        # Инициализация LLM
        try:
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,           # контекст 2K для экономии памяти
                n_threads=4,
                verbose=False,
            )
            self.model_name = "saiga_llama3_8b_Q3"
            LOGGER.info("LLM модель загружена: %s", self.model_name)
        except Exception as e:
            LOGGER.error("Не удалось загрузить LLM: %s, работаем только в fallback-режиме", e)
            self.llm = None
            self.model_name = "FallbackOnly"

        # Инициализация эмбеддинг-модели для fallback
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            LOGGER.info("Эмбеддинг-модель загружена")
        except Exception as e:
            LOGGER.error("Не удалось загрузить эмбеддинг-модель: %s", e)
            self.embedder = None

    # -------------------- LLM генерация --------------------
    @staticmethod
    def _build_prompt(chunk_text: str, num_questions: int) -> str:
        return (
            "<|system|>\n"
            "Ты — помощник учителя. Составь учебные вопросы с выбором ответа по тексту.\n"
            "Каждый вопрос должен иметь ровно 4 варианта: А), Б), В), Г).\n"
            "Выдай ТОЛЬКО JSON-массив без лишних слов.\n"
            "Пример:\n"
            '[{"question": "Что такое фотосинтез?", "options": ["А) процесс дыхания", "Б) процесс образования глюкозы", "В) поглощение воды", "Г) выделение углекислого газа"], "correct": "Б"}]\n'
            "<|user|>\n"
            f"Текст:\n{chunk_text}\n\n"
            f"Составь {num_questions} вопрос(ов) по этому тексту. Ответь JSON.\n"
            "<|assistant|>\n"
        )

    def _generate_llm_questions(self, chunk_text: str, num_questions: int) -> List[Dict]:
        if not self.llm:
            return []
        prompt = self._build_prompt(chunk_text, num_questions)
        # Пробуем несколько температур
        for temp in [0.7, 0.85, 0.6]:
            try:
                response = self.llm(prompt, temperature=temp, **{k:v for k,v in GENERATION_KWARGS.items() if k != 'temperature'})
                raw = response["choices"][0]["text"].strip()
                LOGGER.debug(f"LLM raw output (temp={temp}): {raw[:500]}")
                parsed = _extract_json_from_text(raw)
                if parsed:
                    questions = _validate_questions(parsed)
                    if questions:
                        LOGGER.info(f"LLM сгенерировала {len(questions)} вопросов при temp={temp}")
                        return questions[:num_questions]
            except Exception as e:
                LOGGER.warning(f"Ошибка при temp={temp}: {e}")
        return []

    # -------------------- Fallback (семантический) --------------------
    def _split_sentences(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text, language="russian")
        return [s.strip() for s in sentences if len(s.strip()) > 50]

    def _extract_topic(self, sentence: str) -> str:
        words = sentence.split()
        filtered = [w for w in words if w.lower() not in STOP_WORDS and len(w) > 2]
        if not filtered:
            return " ".join(words[:4])
        topic = " ".join(filtered[:4])
        if len(topic) > 70:
            topic = topic[:67] + "..."
        return topic

    def _extract_answer_phrase(self, sentence: str, max_words: int = 20) -> str:
        words = sentence.split()
        if len(words) <= max_words:
            return sentence
        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

    def _determine_question_type(self, sentence: str) -> str:
        s_low = sentence.lower()
        if "это" in s_low or "называется" in s_low or "является" in s_low:
            return "what"
        if "происходит в" in s_low or "находится в" in s_low or "расположен" in s_low:
            return "where"
        if "выделяется" in s_low or "образуется" in s_low or "превращается" in s_low:
            return "what_happens"
        return "general"

    def _build_question_text(self, qtype: str, topic: str) -> str:
        if qtype == "what":
            return f"Что такое «{topic}»?"
        if qtype == "where":
            return f"Где происходит «{topic}»?"
        if qtype == "what_happens":
            return f"Что происходит при «{topic}»?"
        return f"Какое утверждение о «{topic}» верно?"

    def _get_distractors(self, sentences: List[str], correct_phrase: str, num: int = 3) -> List[str]:
        if not self.embedder:
            return ["Неверно", "Ошибка", "Выдумка"]
        correct_emb = self.embedder.encode([correct_phrase])[0].reshape(1, -1)
        short_phrases = []
        for s in sentences:
            words = s.split()
            if len(words) > 10:
                short = " ".join(words[:10])
            else:
                short = s
            short_phrases.append(short)
        if not short_phrases:
            return ["Неверно"] * num
        candidates_emb = self.embedder.encode(short_phrases)
        sims = cosine_similarity(correct_emb, candidates_emb).flatten()
        indices = np.argsort(sims)[:num]
        return [short_phrases[i] for i in indices]

    def _semantic_fallback(self, chunk_text: str, num_questions: int) -> List[Dict]:
        """Генерация вопросов без LLM (улучшенный семантический метод)."""
        if not self.embedder:
            return []
        sentences = self._split_sentences(chunk_text)
        if len(sentences) < 2:
            return []
        # Эмбеддинги предложений
        embeddings = self.embedder.encode(sentences)
        full_emb = np.mean(embeddings, axis=0).reshape(1, -1)
        sims = cosine_similarity(embeddings, full_emb).flatten()
        lengths = np.array([len(s) for s in sentences])
        length_norm = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-6)
        scores = 0.7 * sims + 0.3 * length_norm
        top_indices = np.argsort(scores)[-num_questions:][::-1]

        questions = []
        for idx in top_indices:
            sent = sentences[idx]
            topic = self._extract_topic(sent)
            qtype = self._determine_question_type(sent)
            question_text = self._build_question_text(qtype, topic)
            correct_phrase = self._extract_answer_phrase(sent, max_words=20)
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

    # -------------------- Основной метод генерации --------------------
    def generate_questions(self, chunk_text: str, num_questions: int = 2) -> List[Dict]:
        """Сначала пытается LLM, при ошибке – semantic fallback."""
        if not chunk_text or len(chunk_text.strip()) < 100:
            return []

        # Попытка LLM
        llm_questions = self._generate_llm_questions(chunk_text, num_questions)
        if llm_questions:
            return llm_questions

        # Fallback
        LOGGER.warning("LLM не удалась, используем семантический fallback для фрагмента")
        return self._semantic_fallback(chunk_text, num_questions)