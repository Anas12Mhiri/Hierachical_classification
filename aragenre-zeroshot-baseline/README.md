# AraGenre Zero-Shot Baseline

Zero-shot Arabic genre classification baseline for the **AraGenre** shared task
(broad + specific genre). No training is involved.

## Pipeline

1. **Translate** Arabic texts and candidate genre definitions to English (Google Translate via `deep-translator`).
2. **Qwen3-4B** reads each translated text (with the candidate genres as hints) and writes a genre explanation.
3. **sentence-t5-xxl** embeds the raw translated text and Qwen's explanation.
4. **Ensemble snap:** `0.4 * sim(text, genre) + 0.6 * sim(explanation, genre)`; the best-scoring candidate gives both the broad and specific genre.

Metric: Hierarchical Macro F1 = mean(Broad Macro F1, Specific Macro F1).

## Layout

```
.
├── run_baseline.py              # CLI entry point
├── src/aragenre_baseline/
│   ├── config.py                # paths, models, devices, weights
│   ├── utils.py                 # I/O, cosine, metrics
│   ├── translation.py           # AR -> EN
│   ├── qwen.py                  # prompt + generation
│   ├── embedding.py             # embeddings + candidate matching
│   └── pipeline.py              # end-to-end run
├── notebooks/baseline.ipynb     # cleaned notebook (no outputs)
├── baseline_outputs/            # predictions (git-ignored)
├── requirements.txt
└── .gitignore
```

## Usage

```bash
pip install -r requirements.txt
python run_baseline.py --data-dir /path/to/public_data_train_dev --out-dir baseline_outputs
# quick smoke test
python run_baseline.py --data-dir /path/to/public_data_train_dev --limit 5
```

The data dir must contain `dev.json`, `dev_gold.json`, `dev_genre_definitions.json`.
The data dir can also be set with `ARAGENRE_DATA_DIR`.

**Hardware:** the defaults put Qwen on `cuda:1` and the embedder on `cuda:0` (two GPUs).
For a single GPU, set both devices to `cuda:0` in `src/aragenre_baseline/config.py`.
Google Translate needs internet access.

## Output

`baseline_outputs/baseline_zeroshot_dev_submission.json` — `[{id, broad_genre, specific_genre}, ...]`
`baseline_outputs/baseline_zeroshot_dev_predictions_detailed.json` — adds gold labels, translations, Qwen output, similarity.
