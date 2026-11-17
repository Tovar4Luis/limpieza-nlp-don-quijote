import argparse
from pathlib import Path

import pandas as pd
import spacy
from nltk.stem import SnowballStemmer
from spacy.cli import download

BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR / "don_quijote.txt"
OUTPUT = BASE_DIR / "don_quijote_normalizado.txt"
COMPARISON = BASE_DIR / "comparacion_stemming_lematizacion.csv"
MODEL = "es_core_news_sm"


def load_spanish_model():
    """Carga el modelo español; lo instala en la primera ejecución."""
    try:
        return spacy.load(MODEL, disable=["parser", "ner"])
    except OSError:
        download(MODEL)
        return spacy.load(MODEL, disable=["parser", "ner"])


def main():
    parser = argparse.ArgumentParser(
        description="Tokeniza, normaliza y lematiza Don Quijote en español."
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--comparison", type=Path, default=COMPARISON)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8-sig")
    # Excluye el prólogo técnico y la licencia añadidos por Project Gutenberg.
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***"
    if start_marker in text and end_marker in text:
        text = text.split(start_marker, 1)[1].split(end_marker, 1)[0]
    nlp = load_spanish_model()
    stemmer = SnowballStemmer("spanish")

    # El modelo recibe bloques para procesar el libro sin exceder max_length.
    blocks = [text[i:i + 80000] for i in range(0, len(text), 80000)]
    normalized = []
    comparison = []
    total_tokens = 0
    stopword_count = 0
    punctuation_count = 0
    first_tokens = []

    for doc in nlp.pipe(blocks, batch_size=4):
        for token in doc:
            total_tokens += 1
            if len(first_tokens) < 20:
                first_tokens.append(token.text)
            if token.is_stop:
                stopword_count += 1
            if token.is_punct or token.is_space or not token.text.strip():
                punctuation_count += 1
                continue

            lemma = token.lemma_.lower().strip()
            if not lemma:
                continue
            normalized.append(lemma)
            comparison.append({
                "Original": token.text,
                "Stemming (corte)": stemmer.stem(token.text),
                "Lematización (spaCy)": lemma,
                "¿Coinciden?": stemmer.stem(token.text) == lemma,
            })

    args.output.write_text(" ".join(normalized) + "\n", encoding="utf-8")
    pd.DataFrame(comparison).to_csv(args.comparison, index=False, encoding="utf-8-sig")

    print(f"Texto cargado: {args.input.name} ({len(text)} caracteres)")
    print(f"Tokenización: {total_tokens} tokens")
    print(f"Primeros tokens: {first_tokens}")
    print(f"Stop words detectadas: {stopword_count}")
    print(f"Puntuación/espacios descartados: {punctuation_count}")
    print(f"Tokens normalizados y lematizados: {len(normalized)}")
    print("Ejemplos de stemming y lematización:")
    print(pd.DataFrame(comparison).head(10).to_string(index=False))
    print(f"Texto procesado: {args.output}")
    print(f"Comparación guardada: {args.comparison}")


if __name__ == "__main__":
    main()
