"""Sentence embeddings and cosine matching to candidate genres."""
from sentence_transformers import SentenceTransformer

from . import config
from .utils import cosine_matrix


def load_embedder():
    print(f"\nLoading sentence-t5-xxl embedder ({config.EMBED_DEVICE})...")
    return SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBED_DEVICE)


def embed_texts(embedder, texts):
    return embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )


def snap_to_candidate(text_emb, pred_emb, candidate_embs, candidates):
    """Weighted ensemble of raw-text similarity and Qwen-explanation similarity."""
    sims_text = cosine_matrix(text_emb.reshape(1, -1), candidate_embs)[0]
    sims_pred = cosine_matrix(pred_emb.reshape(1, -1), candidate_embs)[0]
    final_sims = config.WEIGHT_TEXT * sims_text + config.WEIGHT_PRED * sims_pred
    best_idx = int(final_sims.argmax())
    return candidates[best_idx], float(final_sims[best_idx])
