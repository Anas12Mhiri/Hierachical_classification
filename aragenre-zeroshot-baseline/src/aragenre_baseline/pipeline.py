"""End-to-end zero-shot baseline on the AraGenre dev set."""
import argparse
import time
from pathlib import Path

from tqdm import tqdm

from . import config
from .embedding import embed_texts, load_embedder, snap_to_candidate
from .qwen import build_prediction_prompt, call_qwen, extract_explanation, load_qwen
from .translation import (build_candidate_text_for_translation, load_translator,
                          translate_all)
from .utils import evaluate, hierarchical_macro_f1, load_json, save_json


def run(data_dir=config.DATA_DIR, out_dir=config.OUT_DIR, run_limit=config.RUN_LIMIT):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("Loading data...")
    dev_texts = load_json(data_dir / config.DEV_TEXT_FILE_NAME)
    dev_gold = load_json(data_dir / config.DEV_GOLD_FILE_NAME)
    candidates = load_json(data_dir / config.DEV_DEFINITIONS_FILE_NAME)
    gold_by_id = {row["id"]: row for row in dev_gold}

    if run_limit is not None:
        dev_texts = dev_texts[:run_limit]

    print(f"Dev examples : {len(dev_texts)}")
    print(f"Candidates   : {len(candidates)}")
    for c in candidates:
        print(f"  {c['specific_genre']} ({c['broad_genre']})")

    # ---- Step 1: translate Arabic -> English ----
    translator = load_translator()
    print("\nTranslating candidate definitions...")
    candidates_en = translate_all(
        [build_candidate_text_for_translation(c) for c in candidates],
        translator, "Translating candidates",
    )
    candidates_with_en = [
        {**c, "specific_genre_definition": candidates_en[i]}
        for i, c in enumerate(candidates)
    ]

    print("\nTranslating dev texts...")
    dev_en = translate_all([r["text"] for r in dev_texts], translator, "Translating dev texts")

    # ---- Step 2/3: load models, embed candidates ----
    qwen_tokenizer, qwen_model = load_qwen()
    embedder = load_embedder()
    print("Embedding candidate definitions...")
    candidate_embs = embed_texts(embedder, candidates_en)

    # ---- Step 4: classify ----
    detailed, submission = [], []
    print("\nClassifying dev examples...")
    for i, row in enumerate(tqdm(dev_texts, desc="Classifying dev")):
        text_en = dev_en[i]

        prompt = build_prediction_prompt(text_en, candidates_with_en)
        try:
            qwen_output = call_qwen(prompt, qwen_tokenizer, qwen_model)
        except Exception as e:
            qwen_output = f"ERROR: {e!r}"

        explanation = extract_explanation(qwen_output)
        pred_emb = embedder.encode(explanation, convert_to_numpy=True, show_progress_bar=False)
        text_emb = embedder.encode(text_en, convert_to_numpy=True, show_progress_bar=False)

        chosen, similarity = snap_to_candidate(text_emb, pred_emb, candidate_embs, candidates)
        gold = gold_by_id[row["id"]]

        detailed.append({
            "id": row["id"],
            "text_ar": row["text"],
            "text_en": text_en,
            "gold_broad_genre": gold["broad_genre"],
            "predicted_broad_genre": chosen["broad_genre"],
            "gold_specific_genre": gold["specific_genre"],
            "predicted_specific_genre": chosen["specific_genre"],
            "ensemble_similarity": similarity,
            "extracted_explanation": explanation,
            "raw_qwen_output": qwen_output,
        })
        submission.append({
            "id": row["id"],
            "broad_genre": chosen["broad_genre"],
            "specific_genre": chosen["specific_genre"],
        })

        print(f"[{i+1}/{len(dev_texts)}] Gold: {gold['specific_genre']} | "
              f"Pred: {chosen['specific_genre']} | Sim: {similarity:.4f}")
        if config.SLEEP_BETWEEN_CALLS > 0:
            time.sleep(config.SLEEP_BETWEEN_CALLS)

    # ---- Step 5: save & evaluate ----
    detailed_path = out_dir / config.DETAILED_OUT_NAME
    submission_path = out_dir / config.SUBMISSION_OUT_NAME
    save_json(detailed_path, detailed)
    save_json(submission_path, submission)

    gs = [p["gold_specific_genre"] for p in detailed]
    ps = [p["predicted_specific_genre"] for p in detailed]
    gb = [p["gold_broad_genre"] for p in detailed]
    pb = [p["predicted_broad_genre"] for p in detailed]

    evaluate("Specific Genre Prediction", gs, ps)
    evaluate("Broad Genre Prediction", gb, pb)
    hierarchical_macro_f1(gb, pb, gs, ps)

    print("\nSaved:")
    print(detailed_path)
    print(submission_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--out-dir", default=config.OUT_DIR)
    p.add_argument("--limit", type=int, default=config.RUN_LIMIT,
                   help="Only process the first N dev examples (smoke test).")
    args = p.parse_args()
    run(args.data_dir, args.out_dir, args.limit)


if __name__ == "__main__":
    main()
