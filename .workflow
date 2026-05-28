name: Preprocessing Emotion Dataset

on:
  push:
    branches: [main]
    paths:
      - "preprocessing/**"
      - "emotion_raw/**"
      - ".github/workflows/preprocessing.yml"
  workflow_dispatch:

jobs:
  preprocess:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas scikit-learn nltk

      - name: Run automated preprocessing
        run: |
          python preprocessing/automate_Arya-prasetya.py \
            --input_path emotion_raw/merged_training.pkl \
            --output_dir namadataset_preprocessing

      - name: Upload preprocessing artifacts
        uses: actions/upload-artifact@v4
        with:
          name: emotion-preprocessed-data
          path: namadataset_preprocessing/

      - name: Commit updated preprocessing outputs
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add namadataset_preprocessing/*
          git diff --cached --quiet || git commit -m "Auto-update preprocessed dataset"
          git push
