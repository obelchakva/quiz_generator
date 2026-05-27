import json
import logging
import re
from typing import Dict, List, Optional

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _ensure_nltk_resources() -> None:
    resources = ["punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def load_ground_truth(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_tokens(text: str) -> List[str]:
    _ensure_nltk_resources()
    tokens = word_tokenize(text.lower(), language="russian")
    return [token for token in tokens if re.match(r"\w+", token)]


def _match_questions_by_similarity(
    generated: List[Dict],
    ground_truth: List[Dict],
    embedding_model: SentenceTransformer,
) -> List[tuple]:
    """
    Жадное сопоставление: для каждого сгенерированного вопроса выбираем
    самый похожий эталонный, который ещё не был использован.
    Возвращает список пар (gen_idx, truth_idx).
    """
    if not generated or not ground_truth:
        return []

    # Эмбеддинги вопросов
    gen_texts = [item.get("question", "") for item in generated]
    truth_texts = [item.get("question", "") for item in ground_truth]
    gen_emb = embedding_model.encode(gen_texts)
    truth_emb = embedding_model.encode(truth_texts)

    sim_matrix = cosine_similarity(gen_emb, truth_emb)

    used_truth = set()
    pairs = []

    # Жадное присваивание: для каждого generated берём лучший ещё не использованный truth
    for gen_idx in range(len(generated)):
        best_sim = -1
        best_truth_idx = -1
        for truth_idx in range(len(ground_truth)):
            if truth_idx in used_truth:
                continue
            sim = sim_matrix[gen_idx, truth_idx]
            if sim > best_sim:
                best_sim = sim
                best_truth_idx = truth_idx
        if best_truth_idx != -1:
            pairs.append((gen_idx, best_truth_idx))
            used_truth.add(best_truth_idx)
    return pairs


def evaluate_bleu_matched(
    generated: List[Dict],
    ground_truth: List[Dict],
    pairs: List[tuple],
) -> Dict[str, float]:
    """Вычисляет BLEU только для сопоставленных пар."""
    if not pairs:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    smoother = SmoothingFunction().method1
    weights = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }
    scores = {k: [] for k in weights}

    for gen_idx, truth_idx in pairs:
        gen_question = generated[gen_idx].get("question", "")
        truth_question = ground_truth[truth_idx].get("question", "")
        candidate = _normalize_tokens(gen_question)
        reference = _normalize_tokens(truth_question)
        if not candidate or not reference:
            for key in scores:
                scores[key].append(0.0)
            continue
        for key, weight in weights.items():
            value = sentence_bleu([reference], candidate, weights=weight, smoothing_function=smoother)
            scores[key].append(float(value))

    return {metric: round(sum(vals) / len(vals), 4) if vals else 0.0 for metric, vals in scores.items()}


def evaluate_cosine_similarity_matched(
    generated: List[Dict],
    ground_truth: List[Dict],
    pairs: List[tuple],
    embedding_model: Optional[SentenceTransformer] = None,
) -> float:
    """Среднее косинусное сходство между сопоставленными вопросами."""
    if not pairs:
        return 0.0

    model = embedding_model or SentenceTransformer(EMBEDDING_MODEL_NAME)
    similarities = []
    for gen_idx, truth_idx in pairs:
        gen_q = generated[gen_idx].get("question", "")
        truth_q = ground_truth[truth_idx].get("question", "")
        if not gen_q or not truth_q:
            similarities.append(0.0)
            continue
        emb = model.encode([gen_q, truth_q])
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        similarities.append(float(sim))
    return round(sum(similarities) / len(similarities), 4) if similarities else 0.0


def evaluate_answer_accuracy_matched(
    generated: List[Dict],
    ground_truth: List[Dict],
    pairs: List[tuple],
) -> float:
    """Доля пар, в которых буква правильного ответа совпадает."""
    if not pairs:
        return 0.0
    correct = 0
    for gen_idx, truth_idx in pairs:
        gen_correct = str(generated[gen_idx].get("correct", "")).strip().upper()
        truth_correct = str(ground_truth[truth_idx].get("correct", "")).strip().upper()
        if gen_correct == truth_correct:
            correct += 1
    return round(correct / len(pairs), 4)


def run_full_evaluation(
    gen_questions: List[Dict],
    gt_path: str,
    embedding_model: Optional[SentenceTransformer] = None,
) -> Dict[str, float]:
    """Основная функция оценки с сопоставлением по семантике."""
    ground_truth = load_ground_truth(gt_path)
    if not gen_questions or not ground_truth:
        return {
            "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0,
            "cosine_similarity": 0.0, "answer_accuracy": 0.0,
            "generated_count": len(gen_questions),
            "ground_truth_count": len(ground_truth),
            "matched_pairs": 0,
        }

    # Загружаем эмбеддинг-модель, если не передали
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Сопоставляем вопросы
    pairs = _match_questions_by_similarity(gen_questions, ground_truth, embedding_model)

    # Метрики по сопоставленным парам
    bleu_scores = evaluate_bleu_matched(gen_questions, ground_truth, pairs)
    cosine_score = evaluate_cosine_similarity_matched(gen_questions, ground_truth, pairs, embedding_model)
    answer_acc = evaluate_answer_accuracy_matched(gen_questions, ground_truth, pairs)

    results = {
        **bleu_scores,
        "cosine_similarity": cosine_score,
        "answer_accuracy": answer_acc,
        "generated_count": len(gen_questions),
        "ground_truth_count": len(ground_truth),
        "matched_pairs": len(pairs),
    }

    print("Evaluation metrics (semantic matching):")
    for key, value in results.items():
        print(f"- {key}: {value}")

    return results