# Interactive Quiz Generator (Russian, Offline After First Run)

## Project Overview
This project generates multiple-choice educational cards from Russian text files using a local LLM. It is designed for coursework in AI and Data Science and works fully offline after model download on first launch.

Core capabilities:
- `.txt` input processing (PDF/DOCX extraction stubs included for future extension)
- question generation with local `ai-forever/ruGPT-3.5-1.3B` on CPU
- Streamlit web interface with card-like display
- JSON/CSV export
- quality evaluation with BLEU, cosine similarity, and answer accuracy

## Requirements
- Python 3.10+
- 8+ GB RAM recommended
- ~5 GB free disk space for model caches
- Internet connection required only for first launch (model downloads)

## Installation
```bash
# Option 1: clone a repository that contains this folder
# git clone <repo_url>

cd quiz_generator
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
streamlit run app.py
```

On first launch, models are downloaded automatically:
- `ai-forever/ruGPT-3.5-1.3B` (around 1.5 GB)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (around 0.5 GB)

## How to Use
1. Run `streamlit run app.py`.
2. Upload a Russian `.txt` file.
3. Choose questions per chunk (1-3).
4. Click **Сгенерировать**.
5. Review generated cards and export to JSON or CSV.

## Quality Evaluation
If `test_data/sample.txt` and `test_data/ground_truth.json` exist:
1. Generate questions in the app.
2. Click **Оценить качество**.
3. The app computes:
   - BLEU-1, BLEU-2, BLEU-3, BLEU-4
   - cosine similarity of question embeddings
   - answer accuracy by index matching

## Notes
- The app is CPU-only and does not require CUDA.
- Hugging Face cache location is typically `~/.cache/huggingface/`.
- If the model returns invalid JSON, the app skips invalid outputs and shows warnings.
