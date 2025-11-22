📰 Identifikasi Topik Berita Berdasarkan Judul
Menggunakan Word Embedding (Word2Vec) dan Long Short-Term Memory (LSTM)

Repository ini berisi implementasi lengkap sistem klasifikasi topik berita berdasarkan judul berita dengan memanfaatkan Word Embedding berbasis Word2Vec Wikipedia Indonesia dan model LSTM/GRU. Pipeline dibangun modular melalui beberapa notebook terpisah yang mencakup proses dari pengecekan dataset hingga inference akhir.

📌 Fitur Utama

Word Embedding hasil pelatihan Word2Vec Wikipedia Indonesia

Integrasi embedding matrix ke model PyTorch

Arsitektur LSTM dan GRU

Stratified K-Fold Cross Validation

Penanganan imbalance dataset (undersampling)

Pipeline modular: database → preprocess → embedding → training → inference

Pemilihan model terbaik otomatis (LSTM atau GRU)

🗂️ Struktur Repository
headline-topic-detection/
│
├── dataset/
│   └── indonesian-news-title.csv
│
├── artifacts/
│   ├── vocab/                # word2idx.pkl
│   ├── labels/               # label encoder
│   ├── embedding/            # embedding matrix, dump wiki, extract text
│   ├── config/               # konfigurasi model akhir
│   └── model_final/          # model terbaik
│
├── embeddings/
│   └── idwiki_word2vec.model (+ .npy)
│
├── database.ipynb            # pengecekan dataset + undersampling
├── preprocess.ipynb          # preprocessing & encoding token
├── embedding.ipynb           # membuat embedding matrix
├── training.ipynb            # training LSTM & GRU + K-Fold
├── inference.ipynb           # prediksi headline baru
├── train_embedding.py        # pelatihan Word2Vec Wikipedia
└── README.md
⚙️ Instalasi

Gunakan Python 3.10 / 3.11.

pip install -r requirements.txt

Jika muncul error NumPy 2.x:

pip install "numpy<2"
▶️ Pipeline Pengolahan
1. database.ipynb — Pengecekan Data & Undersampling

Menampilkan distribusi kelas

Menyeimbangkan dataset (undersampling)

Menyimpan dataset final

2. preprocess.ipynb — Preprocessing & Encoding

Normalisasi teks (lowercase, regex)

Tokenisasi

Pembuatan vocabulary (word2idx)

Padding

Encoding label

Output utama:

word2idx.pkl

label_encoder.pkl

X.npy, y.npy

3. embedding.ipynb — Word2Vec Integration

Memuat Word2Vec Wikipedia

Membangun embedding matrix berdasarkan vocabulary

Menyimpan embedding_matrix.npy

4. training.ipynb — Training Model

Arsitektur LSTM dan GRU

Stratified K-Fold

Fine-tuning embedding

Penyimpanan model terbaik

Output:

final_model.pth

config.json (berisi jenis model terbaik: lstm/gru)

5. inference.ipynb — Prediksi Headline Baru

Memuat config

Memilih model terbaik (otomatis)

Encoding teks → prediksi kategori

🧪 Contoh Input–Output

Input:

"Putri KW Enggan Terbebani SEA Games dan World Tour Finals"

Output:

sport
🏗️ Arsitektur Model
Input → Word2Vec Embedding → LSTM/GRU → Dense → Softmax
📊 Evaluasi Model

Menggunakan Stratified K-Fold untuk menjaga distribusi label tiap fold.

Metrik:

Accuracy

Precision (weighted)

Recall (weighted)

F1-score (weighted)

Model terbaik dipilih berdasarkan rata-rata F1-score.

📎 Rencana Pengembangan

Penambahan data augmentation (EDA / synonym replacement / back-translation)

Pembandingan berbagai skema embedding (random vs Word2Vec)

Ekspor model ke ONNX (?)
