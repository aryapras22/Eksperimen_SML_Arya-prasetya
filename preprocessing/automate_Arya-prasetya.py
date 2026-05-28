import os
import re
import json
import pickle
import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered_words)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "text" not in df.columns or "emotions" not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'text' dan 'emotions'.")

    df = df[["text", "emotions"]].dropna().reset_index(drop=True)
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_clean"] = df["text_clean"].apply(remove_stopwords)

    df = df.drop_duplicates(subset=["text_clean", "emotions"]).reset_index(drop=True)
    df["clean_word_count"] = df["text_clean"].apply(lambda x: len(x.split()))

    return df


def save_label_mapping(df: pd.DataFrame, output_dir: str):
    labels = sorted(df["emotions"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)


def save_metadata(df: pd.DataFrame, output_dir: str):
    metadata = {
        "num_rows": int(len(df)),
        "num_classes": int(df["emotions"].nunique()),
        "classes": sorted(df["emotions"].unique().tolist()),
        "avg_clean_word_count": float(df["clean_word_count"].mean()),
    }

    with open(os.path.join(output_dir, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def split_and_save(
    df: pd.DataFrame,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    final_df = df[["text_clean", "emotions"]].copy()

    train_df, test_df = train_test_split(
        final_df,
        test_size=test_size,
        random_state=random_state,
        stratify=final_df["emotions"],
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df["emotions"],
    )

    train_df.to_csv(os.path.join(output_dir, "train_preprocessed.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_preprocessed.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_preprocessed.csv"), index=False)

    summary = {
        "train_shape": list(train_df.shape),
        "val_shape": list(val_df.shape),
        "test_shape": list(test_df.shape),
    }

    with open(os.path.join(output_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return train_df, val_df, test_df


def main(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_pickle(input_path)
    processed_df = preprocess_dataframe(df)

    processed_df.to_csv(os.path.join(output_dir, "full_preprocessed.csv"), index=False)

    save_label_mapping(processed_df, output_dir)
    save_metadata(processed_df, output_dir)
    split_and_save(processed_df, output_dir)

    print("Preprocessing selesai.")
    print(f"Output tersimpan di: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated preprocessing for Emotion Dataset"
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path ke merged_training.pkl"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Folder output preprocessing"
    )
    args = parser.parse_args()

    main(args.input_path, args.output_dir)
