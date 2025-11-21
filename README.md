# 📰 Identifikasi Topik Berita Berdasarkan Judul

### Menggunakan Word Embedding dan Long Short-Term Memory (LSTM)

Repository ini berisi implementasi sistem klasifikasi topik berita berdasarkan judul (headline). Model utama menggunakan Word Embedding + LSTM, dengan perbandingan GRU sebagai baseline. Proyek tersusun modular dalam beberapa notebook yang merepresentasikan pipeline penuh: pengecekan database, preprocessing, training, dan inference.

---

## 📌 Fitur Utama

* Klasifikasi 9 kategori topik berita
* Word Embedding custom (dilatih dari awal)
* Arsitektur LSTM dan GRU
* K-Fold Cross Validation
* Penanganan dataset imbalance melalui undersampling
* Notebook modular dan mudah direproduksi

---

## 🗂️ Struktur Repository

```
headline-topic-detection/
│
├── dataset/
│   └── indonesian-news-title.csv
│
├── database.ipynb        # Exploratory data + undersampling
├── preprocess.ipynb      # Cleaning, tokenizing, encoding
├── training.ipynb        # Training LSTM & GRU + K-Fold
├── inference.ipynb       # Prediksi headline
│
├── models/               # Model tersimpan
└── README.md
```

---

## ⚙️ Instalasi

Gunakan Python 3.10 / 3.11. Disarankan pakai virtual environment.

```
pip install -r requirements.txt
```

Jika muncul error NumPy 2.x:

```
pip install "numpy<2"
```

---

## ▶️ Cara Menjalankan Pipeline

### **1) database.ipynb — Cek Data & Undersampling**

Notebook ini digunakan untuk:

* Membaca dataset asli
* Menampilkan distribusi kategori
* Melakukan undersampling agar dataset seimbang
* Menyimpan dataset hasil balancing

---

### **2) preprocess.ipynb — Preprocessing & Encoding**

Tahap-tahap preprocessing:

* Lowercase dan normalisasi karakter
* Tokenisasi
* Pembuatan vocabulary (word2idx)
* Padding sequence
* Encoding label menjadi angka

Output:

* `X.npy` — tokenized padded input
* `y.npy` — encoded label
* `word2idx.json`

---

### **3) training.ipynb — Training Model (LSTM & GRU)**

Notebook ini berisi:

* Dataloader
* Definisi arsitektur LSTM dan GRU
* Loop training
* K-Fold Cross Validation
* Perhitungan metrik:

  * Accuracy
  * Precision (weighted)
  * Recall (weighted)
  * F1-score (weighted)

Output:

* `lstm_model_foldX.pt`
* `gru_model_foldX.pt`
* Rangkuman hasil evaluasi

---

### **4) inference.ipynb — Prediksi Headline Baru**

Digunakan untuk memprediksi topik dari judul berita.

Contoh penggunaan:

```
predict("Harga BBM naik akibat tekanan global")
```

Output:

```
Kategori: finance
```

---

## 🧪 Contoh Input–Output

**Input:**

```
"Rupiah melemah terhadap dolar AS akibat sentimen global"
```

**Output (LSTM):**

```
finance
```

---

## 🏗️ Arsitektur Model (Ringkas)

```
Input text → Embedding layer → LSTM → Dense layer → Softmax → Output kelas
```

---

## 📊 Evaluasi

Evaluasi dilakukan menggunakan **Stratified K-Fold** agar distribusi label tiap fold tetap seimbang.

Metrik yang digunakan:

* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1-score (weighted)

Setiap model dievaluasi di tiap fold, kemudian dirata-ratakan.

---

## 📎 Rencana Pengembangan

* Penambahan data augmentation (Perbandingan sebelum dan sesudah)
* 4 skema evaluasi (LSTM & GRU dengan 2 Word Embedding)

---