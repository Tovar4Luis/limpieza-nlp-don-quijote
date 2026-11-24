import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacy
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from spacy.cli import download

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "don_quijote.txt"
DEFAULT_OUTPUT = BASE_DIR / "don_quijote_normalizado.txt"


def load_model():
    try:
        nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    except OSError:
        download("es_core_news_sm")
        nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def vector_plot(model, output, title):
    words = [word for word, _ in model.wv.most_common(60)]
    vectors = [model.wv[word] for word in words]
    points = PCA(n_components=2, random_state=42).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(points[:, 0], points[:, 1], s=24, alpha=0.7)
    for word, (x, y) in zip(words, points):
        ax.annotate(word, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set(title=title, xlabel="Componente principal 1", ylabel="Componente principal 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Carga, limpia, lematiza y vectoriza Don Quijote.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8-sig")
    start = "*** START OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***"
    if start in text and end in text:
        text = text.split(start, 1)[1].split(end, 1)[0]

    nlp = load_model()
    blocks = (text[i:i + 80000] for i in range(0, len(text), 80000))
    sentences = []
    normalized = []
    total_tokens = 0
    for doc in nlp.pipe(blocks, batch_size=4):
        for sent in doc.sents:
            lemmas = []
            for token in sent:
                total_tokens += 1
                if token.is_stop or token.is_punct or token.is_space:
                    continue
                lemma = re.sub(r"[^\wáéíóúüñ]", "", token.lemma_.lower().strip(), flags=re.UNICODE)
                if lemma:
                    lemmas.append(lemma)
                    normalized.append(lemma)
            if lemmas:
                sentences.append(lemmas)

    args.output.write_text(" ".join(normalized) + "\n", encoding="utf-8")
    print(f"Texto cargado: {args.input.name} ({len(text):,} caracteres)")
    print(f"Tokens analizados: {total_tokens:,}; lemas conservados: {len(normalized):,}")
    print(f"Texto limpio guardado: {args.output.name}")

    for sg, name, title in [
        (1, "espacio_word2vec_skipgram.png", "Don Quijote: Word2Vec Skip-gram (PCA)"),
        (0, "espacio_word2vec_cbow.png", "Don Quijote: Word2Vec CBOW (PCA)"),
    ]:
        model = Word2Vec(
            sentences=sentences, vector_size=100, window=5, min_count=5,
            workers=1, sg=sg, seed=42, epochs=15,
        )
        output = BASE_DIR / name
        vector_plot(model, output, title)
        print(f"Representación vectorial guardada: {output.name}")


if __name__ == "__main__":
    main()
