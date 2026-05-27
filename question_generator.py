import csv
import io
import json
import logging
from typing import Callable, Dict, List, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from text_processor import split_into_chunks

LOGGER = logging.getLogger(__name__)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _deduplicate_questions(
    questions: List[Dict],
    threshold: float = 0.9,
    embedding_model: Optional[SentenceTransformer] = None,
) -> List[Dict]:
    if not questions:
        return []

    if embedding_model is None:
        unique_questions: List[Dict] = []
        seen = set()
        for question in questions:
            key = question.get("question", "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique_questions.append(question)
        return unique_questions

    texts = [
        " | ".join(
            [
                question.get("question", ""),
                *question.get("options", []),
                question.get("correct", ""),
            ]
        )
        for question in questions
    ]
    embeddings = embedding_model.encode(texts)

    unique_questions: List[Dict] = []
    unique_embeddings = []

    for idx, question in enumerate(questions):
        current_embedding = embeddings[idx].reshape(1, -1)

        if not unique_embeddings:
            unique_questions.append(question)
            unique_embeddings.append(current_embedding)
            continue

        similarities = [
            cosine_similarity(current_embedding, existing_embedding)[0][0]
            for existing_embedding in unique_embeddings
        ]

        if max(similarities) <= threshold:
            unique_questions.append(question)
            unique_embeddings.append(current_embedding)

    return unique_questions


def generate_questions_for_text(
    text: str,
    model_manager,
    questions_per_chunk: int = 2,
    progress_callback: Optional[Callable[[float], None]] = None,
    warn_callback: Optional[Callable[[str], None]] = None,
    info_callback: Optional[Callable[[str], None]] = None,
    embedding_model: Optional[SentenceTransformer] = None,
) -> List[Dict]:
    """Split source text, generate question cards, and remove duplicates."""
    chunks = split_into_chunks(text)
    all_questions: List[Dict] = []

    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks):
        generated = model_manager.generate_questions(chunk, num_questions=questions_per_chunk)
        chunk_message = f"Чанк {index + 1}/{total_chunks}: получено {len(generated)} вопрос(а)."
        LOGGER.info(chunk_message)

        if info_callback is not None:
            info_callback(chunk_message)

        if not generated and warn_callback is not None:
            warn_callback(
                f"Чанк {index + 1}: модель не вернула валидный JSON. "
                "Попробуйте другую модель или уменьшите число вопросов на чанк."
            )

        all_questions.extend(generated)

        if progress_callback is not None and total_chunks > 0:
            progress_callback((index + 1) / total_chunks)

    deduplicated = _deduplicate_questions(all_questions, embedding_model=embedding_model)
    LOGGER.info("Total generated: %s, after deduplication: %s", len(all_questions), len(deduplicated))
    return deduplicated


def save_questions_to_json(questions: List[Dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(questions, file, ensure_ascii=False, indent=2)


def save_questions_to_csv(questions: List[Dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["question", "option_a", "option_b", "option_c", "option_d", "correct"],
        )
        writer.writeheader()

        for item in questions:
            options = item.get("options", ["", "", "", ""])
            writer.writerow(
                {
                    "question": item.get("question", ""),
                    "option_a": options[0] if len(options) > 0 else "",
                    "option_b": options[1] if len(options) > 1 else "",
                    "option_c": options[2] if len(options) > 2 else "",
                    "option_d": options[3] if len(options) > 3 else "",
                    "correct": item.get("correct", ""),
                }
            )


def questions_to_csv_bytes(questions: List[Dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["question", "option_a", "option_b", "option_c", "option_d", "correct"],
    )
    writer.writeheader()

    for item in questions:
        options = item.get("options", ["", "", "", ""])
        writer.writerow(
            {
                "question": item.get("question", ""),
                "option_a": options[0] if len(options) > 0 else "",
                "option_b": options[1] if len(options) > 1 else "",
                "option_c": options[2] if len(options) > 2 else "",
                "option_d": options[3] if len(options) > 3 else "",
                "correct": item.get("correct", ""),
            }
        )

    return buffer.getvalue().encode("utf-8")
