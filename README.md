# Eksperimen SML Arya Prasetya

Repository ini berisi eksperimen dan otomatisasi preprocessing untuk submission akhir kelas **Membangun Sistem Machine Learning** Dicoding. Dataset yang digunakan adalah **dair-ai/emotion** untuk tugas **multiclass text classification** dengan 6 label emosi: sadness, joy, love, anger, fear, dan surprise.

## Struktur Repository

```bash
Eksperimen_SML_Arya-prasetya/
├── .github/
│   └── workflows/
│       └── preprocessing.yml
├── emotion_raw/
│   └── merged_training.pkl
├── preprocessing/
│   ├── Eksperimen_Arya-prasetya.ipynb
│   └── automate_Arya-prasetya.py
├── namadataset_preprocessing/
│   ├── full_preprocessed.csv
│   ├── train_preprocessed.csv
│   ├── val_preprocessed.csv
│   ├── test_preprocessed.csv
│   ├── label_mapping.json
│   ├── preprocessing_metadata.json
│   └── split_summary.json
├── requirements.txt
└── README.md
```

## Deskripsi Tahapan

### 1. Eksperimen Manual

Notebook `preprocessing/Eksperimen_Arya-prasetya.ipynb` digunakan untuk:

- memuat dataset raw,
- melakukan exploratory data analysis (EDA),
- melakukan preprocessing teks secara manual,
- menyimpan dataset hasil preprocessing.

### 2. Otomatisasi Preprocessing

Script `preprocessing/automate_Arya-prasetya.py` digunakan untuk:

- membaca raw dataset dari file `merged_training.pkl`,
- membersihkan teks,
- menghapus stopwords,
- menghapus duplikasi,
- membagi dataset menjadi train, validation, dan test,
- menyimpan output preprocessing ke folder `namadataset_preprocessing/`.

### 3. Workflow GitHub Actions

Workflow `.github/workflows/preprocessing.yml` digunakan untuk menjalankan preprocessing secara otomatis setiap kali workflow dipicu, lalu menghasilkan dataset preprocessing terbaru.

## Cara Menjalankan Secara Lokal

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan notebook eksperimen

Buka file berikut di Jupyter Notebook atau VS Code:

```bash
preprocessing/Eksperimen_Arya-prasetya.ipynb
```

### 3. Jalankan preprocessing otomatis

```bash
python preprocessing/automate_Arya-prasetya.py \
  --input_path emotion_raw/merged_training.pkl \
  --output_dir namadataset_preprocessing
```

## Output Preprocessing

Output preprocessing yang dihasilkan:

- `full_preprocessed.csv`
- `train_preprocessed.csv`
- `val_preprocessed.csv`
- `test_preprocessed.csv`
- `label_mapping.json`
- `preprocessing_metadata.json`
- `split_summary.json`

## Dataset

Dataset yang digunakan mengacu pada emotion dataset dari `dair-ai/emotion_dataset`, yang disediakan untuk tugas emotion classification berbasis teks.
