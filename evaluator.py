import json
import logging
import re
from typing import Dict, List, Optional

import nltk
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


def evaluate_bleu(generated_questions: List[Dict], ground_truth_questions: List[Dict]) -> Dict[str, float]:
    if not generated_questions or not ground_truth_questions:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    smoother = SmoothingFunction().method1
    total = min(len(generated_questions), len(ground_truth_questions))

    scores = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}
    weights = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }

    for index in range(total):
        candidate = _normalize_tokens(generated_questions[index].get("question", ""))
        reference = _normalize_tokens(ground_truth_questions[index].get("question", ""))

        if not candidate or not reference:
            for key in scores:
                scores[key].append(0.0)
            continue

        for key, weight in weights.items():
            value = sentence_bleu([reference], candidate, weights=weight, smoothing_function=smoother)
            scores[key].append(float(value))

    return {metric: round(sum(values) / len(values), 4) if values else 0.0 for metric, values in scores.items()}


def evaluate_cosine_similarity(
    generated_questions: List[Dict],
    ground_truth_questions: List[Dict],
    embedding_model: Optional[SentenceTransformer] = None,
) -> float:
    if not generated_questions or not ground_truth_questions:
        return 0.0

    if len(generated_questions) != len(ground_truth_questions):
        LOGGER.warning("Generated and ground truth question counts differ; comparing by index on overlap.")

    total = min(len(generated_questions), len(ground_truth_questions))
    model = embedding_model or SentenceTransformer(EMBEDDING_MODEL_NAME)

    generated_texts = [generated_questions[index].get("question", "") for index in range(total)]
    truth_texts = [ground_truth_questions[index].get("question", "") for index in range(total)]

    generated_embeddings = model.encode(generated_texts)
    truth_embeddings = model.encode(truth_texts)

    similarities = []
    for index in range(total):
        similarity = cosine_similarity(
            generated_embeddings[index].reshape(1, -1),
            truth_embeddings[index].reshape(1, -1),
        )[0][0]
        similarities.append(float(similarity))

    return round(sum(similarities) / len(similarities), 4) if similarities else 0.0


def evaluate_answer_accuracy(generated_questions: List[Dict], ground_truth_questions: List[Dict]) -> float:
    if not generated_questions or not ground_truth_questions:
        return 0.0

    total = min(len(generated_questions), len(ground_truth_questions))
    matches = 0

    for index in range(total):
        generated_correct = str(generated_questions[index].get("correct", "")).strip().upper()
        ground_truth_correct = str(ground_truth_questions[index].get("correct", "")).strip().upper()
        if generated_correct == ground_truth_correct:
            matches += 1

    return round(matches / total, 4) if total else 0.0


def run_full_evaluation(
    gen_questions: List[Dict],
    gt_path: str,
    embedding_model: Optional[SentenceTransformer] = None,
) -> Dict[str, float]:
    ground_truth = load_ground_truth(gt_path)

    bleu_scores = evaluate_bleu(gen_questions, ground_truth)
    cosine_score = evaluate_cosine_similarity(gen_questions, ground_truth, embedding_model=embedding_model)
    answer_accuracy = evaluate_answer_accuracy(gen_questions, ground_truth)

    results = {
        **bleu_scores,
        "cosine_similarity": cosine_score,
        "answer_accuracy": answer_accuracy,
        "generated_count": len(gen_questions),
        "ground_truth_count": len(ground_truth),
    }

    print("Evaluation metrics:")
    for key, value in results.items():
        print(f"- {key}: {value}")

    return results
