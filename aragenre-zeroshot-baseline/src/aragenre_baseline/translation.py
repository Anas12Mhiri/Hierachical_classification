"""Arabic -> English translation via Google Translate (deep-translator)."""
from deep_translator import GoogleTranslator
from tqdm import tqdm

from . import config


def load_translator():
    print("Using Google Translator...")
    return GoogleTranslator(source="ar", target="en")


def translate_batch(texts, translator):
    results = []
    for text in texts:
        try:
            results.append(translator.translate(text))
        except Exception as e:
            print(f"Translation error: {e}")
            results.append(text)  # fall back to the original text
    return results


def translate_all(texts, translator, desc="Translating"):
    results = []
    bs = config.TRANSLATION_BATCH_SIZE
    for i in tqdm(range(0, len(texts), bs), desc=desc):
        results.extend(translate_batch(texts[i:i + bs], translator))
    return results


def build_candidate_text_for_translation(c):
    return (
        f"Genre: {c['specific_genre']}. "
        f"Definition: {c['specific_genre_definition']}. "
        f"Broad genre: {c['broad_genre']}. "
        f"Broad genre definition: {c['broad_genre_definition']}."
    )
