import json
import logging
from pathlib import Path
from typing import List

import streamlit as st
from sentence_transformers import SentenceTransformer

from evaluator import EMBEDDING_MODEL_NAME, run_full_evaluation
from model_manager import FALLBACK_MODEL_NAMES, MODEL_NAME, ModelManager
from question_generator import (
    generate_questions_for_text,
    questions_to_csv_bytes,
    save_questions_to_csv,
    save_questions_to_json,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
SAMPLE_PATH = TEST_DATA_DIR / "sample.txt"


@st.cache_resource
def get_model_manager(selected_model: str) -> ModelManager:
    st.info(
        "Загрузка LLM на CPU. Первый запуск может занять несколько минут и требует интернет. "
        "При недоступности основной модели используется резервная fallback-модель."
    )
    return ModelManager(model_name=selected_model, fallback_model_names=FALLBACK_MODEL_NAMES)


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    st.info("Загрузка embedding-модели для оценки качества.")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _questions_to_json_bytes(questions):
    return json.dumps(questions, ensure_ascii=False, indent=2).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="Interactive Quiz Generator", layout="wide")
    st.title("Интерактивный генератор учебных карточек")
    st.write(
        "Загрузите текстовый файл на русском языке, сгенерируйте тестовые карточки и экспортируйте результат. "
        "Поддержка PDF/DOCX в разработке."
    )

    with st.sidebar:
        st.header("Настройки")
        model_options = [MODEL_NAME] + [m for m in FALLBACK_MODEL_NAMES if m != MODEL_NAME]
        selected_model = st.selectbox("Модель генерации", options=model_options, index=0)
        questions_per_chunk = st.slider("Вопросов на фрагмент текста", min_value=1, max_value=3, value=2, step=1)
        st.caption("Каждый фрагмент текста содержит 1200 символов (3-5 предложений).")
        generate_clicked = st.button("Сгенерировать", type="primary")
        clear_clicked = st.button("Очистить результаты")
        

    if clear_clicked:
        st.session_state.pop("generated_questions", None)
        st.session_state.pop("chunk_logs", None)
        st.success("Результаты очищены. Можно запустить новую генерацию.")

    uploaded_file = st.file_uploader("Загрузите .txt файл", type=["txt"])
    if uploaded_file is not None:
        st.session_state["last_uploaded_file"] = uploaded_file.name

    if generate_clicked:
        if uploaded_file is None:
            st.error("Пожалуйста, загрузите .txt файл перед генерацией.")
        else:
            try:
                text = uploaded_file.getvalue().decode("utf-8")
                if not text.strip():
                    st.error("Файл пуст. Загрузите непустой текст.")
                    return

                model_manager = get_model_manager(selected_model)
                st.caption(f"Загружена модель: `{model_manager.model_name}`")

                progress_bar = st.progress(0)
                status = st.empty()
                chunk_log_box = st.empty()
                chunk_logs: List[str] = []
                status.info("Генерация вопросов по фрагментам...")

                def _update_progress(value: float) -> None:
                    progress_bar.progress(min(max(value, 0.0), 1.0))

                def _warn(message: str) -> None:
                    st.warning(message)

                def _info(message: str) -> None:
                    chunk_logs.append(message)
                    chunk_log_box.info("\n".join(chunk_logs))

                questions = generate_questions_for_text(
                    text=text,
                    model_manager=model_manager,
                    questions_per_chunk=questions_per_chunk,
                    progress_callback=_update_progress,
                    warn_callback=_warn,
                    info_callback=_info,
                )

                if not questions:
                    st.warning(
                        "Не удалось сгенерировать валидные JSON-вопросы. "
                        "Попробуйте уменьшить число вопросов на фрагмент текста."
                    )
                    return

                st.session_state["generated_questions"] = questions
                st.session_state["chunk_logs"] = chunk_logs
                status.success(f"Готово! Сгенерировано вопросов: {len(questions)}")
            except Exception as error:  # pylint: disable=broad-except
                LOGGER.exception("Generation failed")
                st.error(f"Ошибка генерации: {error}")

    questions = st.session_state.get("generated_questions", [])
    if not questions:
        return
    

    st.subheader("Сгенерированные карточки")
    for index, item in enumerate(questions, start=1):
        with st.expander(f"Вопрос {index}"):
            st.markdown(f"**{item.get('question', '')}**")
            for option in item.get("options", []):
                st.write(option)
            st.write(f"Правильный ответ: **{item.get('correct', '')}**")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Экспорт в JSON",
            data=_questions_to_json_bytes(questions),
            file_name="generated_questions.json",
            mime="application/json",
        )
    with col2:
        st.download_button(
            label="Экспорт в CSV",
            data=questions_to_csv_bytes(questions),
            file_name="generated_questions.csv",
            mime="text/csv",
        )

    save_dir = PROJECT_ROOT / "exports"
    save_dir.mkdir(exist_ok=True)
    save_questions_to_json(questions, str(save_dir / "generated_questions.json"))
    save_questions_to_csv(questions, str(save_dir / "generated_questions.csv"))

    # Кнопка оценки качества показывается только для предопределённого тестового файла
    if SAMPLE_PATH.exists():
        # Показываем кнопку только если загружен файл sample.txt (проверяем имя загруженного файла)
        show_eval_button = False
        if uploaded_file is not None and uploaded_file.name == "sample.txt":
            show_eval_button = True
        # Если пользователь уже сгенерировал вопросы для sample.txt, тоже можно показать кнопку
        elif st.session_state.get("last_uploaded_file") == "sample.txt":
            show_eval_button = True
        
        if show_eval_button:
            if st.button("Оценить качество (только для sample.txt)"):
                try:
                    model_manager = get_model_manager(selected_model)
                    embedding_model = get_embedding_model()
                    sample_text = SAMPLE_PATH.read_text(encoding="utf-8")

                    st.info("Генерируем вопросы для test_data/sample.txt и рассчитываем метрики...")
                    eval_questions = generate_questions_for_text(
                        text=sample_text,
                        model_manager=model_manager,
                        questions_per_chunk=questions_per_chunk,
                        embedding_model=embedding_model,
                    )

                    if not eval_questions:
                        st.warning("Для sample.txt не удалось получить вопросы. Метрики недоступны.")
                    else:
                        metrics = run_full_evaluation(
                            gen_questions=eval_questions,
                            questions_per_chunk=questions_per_chunk,
                            embedding_model=embedding_model,
                        )
                        st.subheader("Результаты оценки")
                        st.table(metrics)
                except Exception as error:
                    LOGGER.exception("Evaluation failed")
                    st.error(f"Ошибка при оценке качества: {error}")


if __name__ == "__main__":
    main()
