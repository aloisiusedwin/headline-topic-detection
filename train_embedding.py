import gensim
import multiprocessing
import os
import requests
import argparse
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import sys

BASE_DIR = "."
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "embeddings", "idwiki_word2vec.model")
DEFAULT_EXTRACTED_PATH = os.path.join(BASE_DIR, "artifacts", "embedding", "idwiki_extracted.txt")
DEFAULT_DUMP_PATH = os.path.join(BASE_DIR, "artifacts", "embedding", "idwiki_dump.xml.bz2")


def download(link, file_name):
    """Download file dengan progress bar sederhana."""
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()


def get_id_wiki(dump_path):
    """Download Wikipedia dump jika belum ada."""
    if not os.path.isfile(dump_path):
        url = 'https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2'
        download(url, dump_path)

    return gensim.corpora.WikiCorpus(dump_path, dictionary={})


def extract_text(extracted_path, id_wiki, stem):
    """Ekstraksi teks mentah dari Wikipedia dump."""
    os.makedirs(os.path.dirname(extracted_path), exist_ok=True)

    if os.path.isfile(extracted_path):
        print("Extracted text file already exists. Skipping...")
        return

    stemmer = None
    if stem:
        print("Warning: Using stemmer will slow the process significantly.")
        stemmer = StemmerFactory().create_stemmer()

    with open(extracted_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(id_wiki.get_texts()):
            text = " ".join(text)
            if stemmer:
                text = stemmer.stem(text)

            f.write(text + "\n")

            if i % (10 if stem else 1000) == 0:
                print(f"{i} articles processed...")

        print("Total extracted:", i)


def build_model(extracted_path, model_path, dim):
    """Train Word2Vec dan simpan hasilnya."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    sentences = gensim.models.word2vec.LineSentence(extracted_path)

    id_w2v = gensim.models.word2vec.Word2Vec(
        sentences,
        vector_size=dim,
        workers=multiprocessing.cpu_count() - 1,
    )
    
    id_w2v.save(model_path)
    return id_w2v


def main(args):
    model_path = args.model_path
    extracted_path = args.extracted_path
    dump_path = args.dump_path
    dim = args.dim
    stem = args.stem

    id_wiki = get_id_wiki(dump_path)

    print("Extracting text...")
    extract_text(extracted_path, id_wiki, stem)

    print("Building Word2Vec model...")
    build_model(extracted_path, model_path, dim)

    print("Saved model:", model_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "y", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Word2Vec: Generate word2vec model for Bahasa Indonesia"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--extracted_path",
        type=str,
        default=DEFAULT_EXTRACTED_PATH,
        help="Path to save extracted text"
    )
    parser.add_argument(
        "--dump_path",
        type=str,
        default=DEFAULT_DUMP_PATH,
        help="Wikipedia dump path"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=300,
        help="Embedding size"
    )
    parser.add_argument(
        "--stem",
        default=False,
        type=str2bool,
        help="Use stemming (default false)"
    )

    args = parser.parse_args()
    main(args)
