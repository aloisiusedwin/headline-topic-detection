# 📰 Identifikasi Topik Berita Berdasarkan Judul
### Menggunakan Word Embedding (Word2Vec), LSTM, dan GRU

Repository ini berisi implementasi sistem klasifikasi topik berita berdasarkan judul (headline) menggunakan Word2Vec embedding Wikipedia Indonesia dan model LSTM/GRU dengan Stratified K-Fold cross validation.  
Pipeline bersifat modular, mulai dari analisis dataset, preprocessing, embedding, augmentasi, training, hingga inference.

---

## 📌 Fitur Utama

- Word Embedding hasil pelatihan **Word2Vec Wikipedia Indonesia**
- Integrasi embedding matrix ke dalam model PyTorch
- Dua arsitektur model: **LSTM & GRU**
- **Stratified K-Fold Cross Validation**
- Penanganan imbalance dataset (undersampling)
- **Data augmentation** berbasis paraphrase IndoT5 dan backtranslate
- Pipeline lengkap: database → preprocess → embedding → augmentation → training → inference
- Pemilihan model terbaik otomatis berdasarkan metrik F1-score

---

## 🗂️ Struktur Repository

```
headline-topic-detection/
│
├── dataset/
│   ├── indonesian-news-title.csv
│   ├── indonesian-news-title-balanced.csv
│   └── indonesian-news-title-augmented.csv
│
├── artifacts/
│   ├── vocab/                # word2idx.pkl
│   ├── labels/               # label_encoder.pkl
│   ├── embedding/            # embedding_matrix.npy + wiki dump
│   ├── config/               # konfigurasi model terbaik
│   └── model_final/          # final_model.pth
│
├── embeddings/
│   └── idwiki_word2vec.model (+ files .npy)
│
├── augmentation.ipynb        # pembuatan data augmentasi
├── database.ipynb            # analisis dataset & undersampling
├── preprocess.ipynb          # normalisasi & encoding token
├── embedding.ipynb           # pembuatan embedding matrix
├── training.ipynb            # training LSTM & GRU (K-Fold)
├── inference.ipynb           # prediksi headline baru
├── train_embedding.py        # pelatihan Word2Vec Wikipedia
└── README.md
```

---

## ⚙️ Instalasi

Gunakan Python **3.10 atau 3.11**.

```bash
pip install -r requirements.txt
```

Jika NumPy versi terbaru membuat error:

```bash
pip install "numpy<2"
```

---

# ▶️ Pipeline Pengolahan

Flowchart pipeline :

```mermaid
flowchart TD
    A[Dataset Asli] --> B[Analisis & Balancing]
    B --> C[Preprocessing]
    C --> D[Tokenisasi & Vocabulary]
    D --> E[Word2Vec Embedding Matrix]
    E --> F[Augmentasi Dataset]
    F --> G[Training LSTM & GRU (K-Fold)]
    G --> H[Evaluasi & Pemilihan Model Terbaik]
    H --> I[Inference]
```

---

# 1️⃣ `database.ipynb` — Analisis & Undersampling

- Menampilkan distribusi kelas awal
- Menangani imbalance dengan undersampling
- Menyimpan dataset: `indonesian-news-title-balanced.csv`

---

# 2️⃣ `preprocess.ipynb` — Preprocessing & Encoding

Fungsi:

- Lowercase & normalisasi teks
- Tokenisasi
- Pembuatan vocabulary (`word2idx.pkl`)
- Padding
- Encoding label (`label_encoder.pkl`)

Output:

```
X.npy
y.npy
word2idx.pkl
label_encoder.pkl
```

---

# 3️⃣ `embedding.ipynb` — Word2Vec Integration

- Memuat model Word2Vec dari folder `embeddings/`
- Membuat embedding matrix (300 dimensi)
- Menyimpan `embedding_matrix.npy`

---

# 4️⃣ `augmentation.ipynb` — Data Augmentation

Metode augmentasi:

- **Paraphrase IndoT5-base-paraphrase**
- **Back-Translate opus-mt-id-en dan opus-mt-en-id**

Contoh:

```
Original     : Kepala Desainer Mobil Hyundai Mengundurkan Diri
Paraphrase   : Kepala arsitek untuk mobil Hyundai mengundurkan diri 
```

Dataset yang dihasilkan:  
`indonesian-news-title-augmented.csv`

---

# 5️⃣ `training.ipynb` — Training LSTM & GRU

Fitur:

- **Stratified K-Fold** (k = 5)
- Fine-tuning embedding matrix
- Arsitektur LSTM dan GRU diuji secara paralel
- Dipilih model terbaik berdasarkan macro F1-score
- Disimpan dalam folder `artifacts/model_final/`

Output:

```
final_model.pth
config.json
```

`config.json` menentukan apakah model terbaik adalah LSTM atau GRU.

---

# 6️⃣ `inference.ipynb` — Prediksi Headline Baru

Contoh penggunaan:

```python
predict("Putri KW Enggan Terbebani SEA Games dan World Tour Finals")
```

Output:

```
sport
```

---

# 📊 Evaluasi Model

## Distribusi Dataset Balanced

```
Setiap kategori = 2436 sampel
Total dataset  = 21.924 sampel
```

## Distribusi Dataset Balanced-Augmented

```
Setiap kategori = 9744 sampel
Total dataset  = 87.696 sampel
```

---

## Tabel Hasil Eksperimen

```
+-----------------+-----------+-----------+
| Dataset         | LSTM (F1) | GRU (F1)  |
+-----------------+-----------+-----------+
| Non-Augmented   |   0.817   |   0.814   |
| Augmented       |   0.926   |   0.927   |
+-----------------+-----------+-----------+
```
---

# 🏗️ Arsitektur Model

```
Input → Word2Vec Embedding → LSTM/GRU → Dense Layer → Softmax
```

---

# 📎 Peluang Pengembangan

- Menambah teknik augmentasi lain (EDA, contextual augmentation)
- Membandingkan embedding lain (FastText, IndoBERT)
- Hyperparameter tuning otomatis (Optuna)
- Menyediakan model dalam format ONNX

---

