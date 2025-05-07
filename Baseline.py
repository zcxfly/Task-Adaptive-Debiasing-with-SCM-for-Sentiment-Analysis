import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import sys
import time
import torch
import math
import html
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, \
    logging as hf_logging
import os

try:
    from tqdm.auto import tqdm

    tqdm.pandas(desc="Pandas Apply")
    use_tqdm = True
except ImportError:
    use_tqdm = False
    print("Info: 'tqdm' library not found. Progress bars will not be shown.")
    print("You can install it via 'pip install tqdm' for progress updates.")

print(f"PyTorch available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dict_path = "/content/drive/MyDrive/dataset/FullDictionaries.csv"
amazon_appliances_path = "/content/drive/MyDrive/dataset/amazon_Appliances_5_jiadian.json"
amazon_fashion_path = "/content/drive/MyDrive/dataset/AMAZON_FASHION_5_shishang.json"
amazon_beauty_path = "/content/drive/MyDrive/dataset/All_Beauty_5_meizhuang.json"
amazon_pet_path = "/content/drive/MyDrive/dataset/Pet_Supplies_5_sampled.json"
movie_path = "/content/drive/MyDrive/dataset/Movie Reviews_train.tsv"
twitter1_path = "/content/drive/MyDrive/dataset/train-twitter.tsv"
twitter2_path = "/content/drive/MyDrive/dataset/test-twitter.tsv"

lookup_key_col = 'preprocessed word 4 (minus one trailing s)'
sociability_dict_col = 'Sociability dictionary'
sociability_dir_col = 'Sociability direction'
morality_dict_col = 'Morality dictionary'
morality_dir_col = 'Morality direction'
ability_dict_col = 'Ability dictionary'
ability_dir_col = 'Ability direction'
agency_dict_col = 'Agency dictionary'
agency_dir_col = 'Agency direction'

print(f"Loading SCM dictionary from: {dict_path}")
try:
    df_scm = pd.read_csv(
        dict_path,
        header=0
    )
    print(f"Successfully loaded SCM dictionary with {len(df_scm)} entries.")
except FileNotFoundError:
    print(f"ERROR: SCM Dictionary file not found at '{dict_path}'. Please check the path.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading SCM dictionary file '{dict_path}': {e}")
    sys.exit(1)

print("\nLoading and preparing datasets...")
num_labels = 5  # We aim for 5 classes (1 to 5)


# Twitter pre-processing
def preprocess_tweet(text: str) -> str:
    """Clean raw tweet text: remove URLs, mentions, hashtags (# sign but keep word), RTs, HTML entities, punctuation, and extra spaces."""
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"rt\s+", "", text)  # Remove retweet tokens
    text = text.replace("#", "")  # Remove hashtag symbol but keep the word
    text = re.sub(r"&amp;", "and", text)  # Replace HTML ampersand
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation/special chars
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


try:
    # --- Amazon ---
    print(f"Loading Amazon Appliances: {amazon_appliances_path}")
    df_amazon_Appliances = pd.read_json(amazon_appliances_path, lines=True)
    print(f"Loading Amazon Fashion: {amazon_fashion_path}")
    df_amazon_fashion = pd.read_json(amazon_fashion_path, lines=True)
    print(f"Loading Amazon Beauty: {amazon_beauty_path}")
    df_amazon_beauty = pd.read_json(amazon_beauty_path, lines=True)
    print(f"Loading Amazon Pet: {amazon_pet_path}")
    df_amazon_pet = pd.read_json(amazon_pet_path, lines=True)

    df_amazon_Appliances['source'] = 'Appliances'
    df_amazon_fashion['source'] = 'Fashion'
    df_amazon_beauty['source'] = 'Beauty'
    df_amazon_pet['source'] = 'Pet'
    df_amazon = pd.concat([df_amazon_Appliances, df_amazon_fashion, df_amazon_beauty, df_amazon_pet], ignore_index=True)
    df_amazon['label_5class'] = df_amazon['overall'].astype(int)
    if 'reviewText' not in df_amazon.columns and 'review' in df_amazon.columns:
        df_amazon.rename(columns={'review': 'reviewText'}, inplace=True)
    elif 'reviewText' not in df_amazon.columns:
        print("ERROR: Could not find 'reviewText' or 'review' column in Amazon data.")
        sys.exit(1)
    if 'review_id' not in df_amazon.columns:
        df_amazon['review_id'] = 'amazon_' + df_amazon.index.astype(str)
    print(f"Combined Amazon datasets ({len(df_amazon)} rows). Labels 1-5.")

    print("Reducing Movie dataset to 20% (stratified by label) to reduce runtime…")
    df_amazon = (
        df_amazon.groupby('label_5class', group_keys=False)
        .apply(lambda x: x.sample(frac=0.2, random_state=42))
        .reset_index(drop=True)
    )
    print(f"Movie dataset reduced to {len(df_amazon)} rows.")

    # --- Movie ---
    print(f"Loading Movie Reviews: {movie_path}")
    df_movie = pd.read_csv(movie_path, sep='\t', header=0)
    df_movie.rename(
        columns={'Phrase': 'review_text_movie', 'Sentiment': 'original_sentiment', 'SentenceId': 'review_id'},
        inplace=True)
    df_movie['review_id'] = 'movie_' + df_movie['review_id'].astype(str) + '_' + df_movie['PhraseId'].astype(str)
    df_movie['label_5class'] = df_movie['original_sentiment'] + 1
    print(f"Loaded Movie dataset ({len(df_movie)} rows). Mapped original 0-4 labels to 1-5.")

    # Reduce Movie dataset to 20% to speed up processing ***
    print("Reducing Movie dataset to 20% (stratified by label) to reduce runtime…")
    df_movie = (
        df_movie.groupby('label_5class', group_keys=False)
        .apply(lambda x: x.sample(frac=0.2, random_state=42))
        .reset_index(drop=True)
    )
    print(f"Movie dataset reduced to {len(df_movie)} rows.")

    # --- Twitter ---
    print(f"Loading Twitter: {twitter1_path}")
    df_twitter1 = pd.read_csv(
        twitter1_path, sep='\t', header=None, quoting=3, engine='python', on_bad_lines='skip'
    )
    print(f"Loading Twitter: {twitter2_path}")
    df_twitter2 = pd.read_csv(
        twitter2_path, sep='\t', header=None, quoting=3, engine='python', on_bad_lines='skip'
    )
    df_twitter = pd.concat([df_twitter1, df_twitter2], ignore_index=True)

    df_twitter.columns = ['tweet_id', 'user', 'original_label', 'text']
    df_twitter['review_id'] = 'tweet_' + df_twitter['tweet_id'].astype(str)

    # Replace placeholder with NaN
    df_twitter['text'] = df_twitter['text'].replace("Not Available", np.nan)
    df_twitter.dropna(subset=['text'], inplace=True)

    print("Applying Twitter text preprocessing…")
    df_twitter['clean_text'] = df_twitter['text'].astype(str).apply(preprocess_tweet)
    df_twitter = df_twitter[df_twitter['clean_text'] != ""]

    # Map original labels (-2 to 2) to 1‑5
    label_map_twitter = {-2: 1, -1: 2, 0: 3, 1: 4, 2: 5}
    df_twitter['label_5class'] = df_twitter['original_label'].map(label_map_twitter)
    if df_twitter['label_5class'].isnull().any():
        print(
            f"Warning: {df_twitter['label_5class'].isnull().sum()} Twitter rows had labels outside -2 to 2 and were dropped.")
        df_twitter.dropna(subset=['label_5class'], inplace=True)
    df_twitter['label_5class'] = df_twitter['label_5class'].astype(int)
    print(f"Loaded, cleaned, and preprocessed Twitter dataset ({len(df_twitter)} rows). Labels mapped to 1-5.")

except FileNotFoundError as e:
    print("\nERROR: Dataset file not found. Please check the path in the error message below.")
    print(e)
    sys.exit(1)
except Exception as e:
    print(f"\nERROR processing dataset files: {e}")
    sys.exit(1)
print("All datasets loaded, labels standardized, and required preprocessing completed successfully.")

# Build the SCM lookup dictionary
print("\nBuilding SCM lookup dictionary…")
if lookup_key_col not in df_scm.columns:
    print(f"ERROR: The specified lookup key column '{lookup_key_col}' not found in the SCM dictionary header.")
    print(f"Available columns are: {df_scm.columns.tolist()}")
    sys.exit(1)

df_scm.dropna(subset=[lookup_key_col], inplace=True)
df_scm[lookup_key_col] = df_scm[lookup_key_col].astype(str)

try:
    lookup_scm = pd.Series(
        list(zip(
            df_scm[sociability_dict_col], df_scm[sociability_dir_col],
            df_scm[morality_dict_col], df_scm[morality_dir_col],
            df_scm[ability_dict_col], df_scm[ability_dir_col],
            df_scm[agency_dict_col], df_scm[agency_dir_col]
        )),
        index=df_scm[lookup_key_col]
    ).to_dict()
    print(f"Built lookup dictionary with {len(lookup_scm)} entries.")
except KeyError as e:
    print(f"ERROR: Column '{e}' not found in SCM dictionary. Check column names in Section 1.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR building lookup dictionary: {e}")
    sys.exit(1)

del df_scm  # Free memory


# Define SCM Calculation and Coverage Functions
def calculate_coverage(df, text_column, lookup_dict, dataset_name):
    """Calculates token and type coverage of a dictionary for a given DataFrame column."""
    print(f"\nCalculating dictionary coverage for '{dataset_name}' ({text_column})…")
    start_time = time.time()
    total_tokens = 0
    matched_tokens = 0
    unique_words_in_data = set()
    matched_unique_words = set()

    if text_column not in df.columns:
        print(f"  Error: Column '{text_column}' not found in the DataFrame for {dataset_name}.")
        return

    texts_to_process = df[text_column].fillna('').astype(str).tolist()

    iterable = texts_to_process
    desc = f"Coverage for {dataset_name}"
    if use_tqdm:
        iterable = tqdm(texts_to_process, desc=desc, total=len(texts_to_process), leave=False)

    for text in iterable:
        tokens = re.findall(r'[a-zA-Z]+', text.lower())
        for token in tokens:
            total_tokens += 1
            unique_words_in_data.add(token)
            if token in lookup_dict:
                matched_tokens += 1
                matched_unique_words.add(token)

    duration = time.time() - start_time
    token_coverage = (matched_tokens / total_tokens * 100) if total_tokens else 0
    type_coverage = (len(matched_unique_words) / len(unique_words_in_data) * 100) if unique_words_in_data else 0

    print(f"Coverage calculation for '{dataset_name}' completed in {duration:.2f} seconds.")
    print(f"  Total Tokens processed: {total_tokens:,}")
    print(f"  Unique Word Types found: {len(unique_words_in_data):,}")
    print(f"  Dictionary Tokens Matched: {matched_tokens:,} ({token_coverage:.2f}%)")
    print(f"  Dictionary Types Matched (Vocabulary Coverage): {len(matched_unique_words):,} ({type_coverage:.2f}%)")


def compute_scm_scores(text, lookup_dict):
    """Computes Warmth (Sociability + Morality) and Competence (Ability + Agency) for a text."""
    tokens = re.findall(r'[a-zA-Z]+', str(text).lower())
    sum_sociability = cnt_sociability = 0
    sum_morality = cnt_morality = 0
    sum_ability = cnt_ability = 0
    sum_agency = cnt_agency = 0
    matched_tokens_count = 0

    for tok in tokens:
        if tok in lookup_dict:
            matched_tokens_count += 1
            s_dict, s_dir, m_dict, m_dir, ab_dict, ab_dir, ag_dict, ag_dir = lookup_dict[tok]
            try:
                if pd.notna(s_dict) and int(s_dict) == 1 and pd.notna(s_dir):
                    sum_sociability += float(s_dir);
                    cnt_sociability += 1
                if pd.notna(m_dict) and int(m_dict) == 1 and pd.notna(m_dir):
                    sum_morality += float(m_dir);
                    cnt_morality += 1
                if pd.notna(ab_dict) and int(ab_dict) == 1 and pd.notna(ab_dir):
                    sum_ability += float(ab_dir);
                    cnt_ability += 1
                if pd.notna(ag_dict) and int(ag_dict) == 1 and pd.notna(ag_dir):
                    sum_agency += float(ag_dir);
                    cnt_agency += 1
            except (ValueError, TypeError):
                pass  # Ignore malformed entries

    warmth_score = (sum_sociability + sum_morality) / (cnt_sociability + cnt_morality) if (
                cnt_sociability + cnt_morality) else 0.0
    competence_score = (sum_ability + sum_agency) / (cnt_ability + cnt_agency) if (cnt_ability + cnt_agency) else 0.0

    return warmth_score, competence_score, matched_tokens_count


def apply_scm_to_dataframe(df, text_column_name, lookup_dict_param):
    """Applies SCM scoring to a DataFrame and appends new columns."""
    if text_column_name not in df.columns:
        print(f"ERROR: Text column '{text_column_name}' not found in the DataFrame for SCM calculation.")
        return df

    total_rows = len(df)
    print(f"\nCalculating SCM scores for column '{text_column_name}' in DataFrame ({total_rows} rows)…")

    if use_tqdm:
        results = df[text_column_name].progress_apply(lambda txt: compute_scm_scores(txt, lookup_dict_param))
    else:
        print("Processing… (this may take a while without tqdm)")
        results = df[text_column_name].apply(lambda txt: compute_scm_scores(txt, lookup_dict_param))

    df['warmth_scm'] = [res[0] for res in results]
    df['competence_scm'] = [res[1] for res in results]
    df['matched_token_count'] = [res[2] for res in results]

    try:
        any_match = (df['matched_token_count'] > 0).sum()
        non_zero_warmth = (df['warmth_scm'] != 0).sum()
        non_zero_competence = (df['competence_scm'] != 0).sum()
        if total_rows > 0:
            print(f"  Rows with any matched tokens: {any_match}/{total_rows} ({(any_match / total_rows) * 100:.2f}%)")
            print(
                f"  Rows with non‑zero Warmth: {non_zero_warmth}/{total_rows} ({(non_zero_warmth / total_rows) * 100:.2f}%)")
            print(
                f"  Rows with non‑zero Competence: {non_zero_competence}/{total_rows} ({(non_zero_competence / total_rows) * 100:.2f}%)")
    except KeyError:
        print("  Warning: Could not calculate post‑processing statistics due to missing columns.")
    return df


# Calculate Coverage & SCM Scores for all datasets
datasets = {
    "Amazon": {"df": df_amazon, "text_col": "reviewText"},
    "Movie": {"df": df_movie, "text_col": "review_text_movie"},
    "Twitter": {"df": df_twitter, "text_col": "clean_text"}  # Use pre‑processed tweet text
}

for name, data in datasets.items():
    calculate_coverage(data["df"], data["text_col"], lookup_scm, name)

for name, data in datasets.items():
    datasets[name]["df"] = apply_scm_to_dataframe(data["df"], data["text_col"], lookup_scm)

df_amazon = datasets["Amazon"]["df"]
df_movie = datasets["Movie"]["df"]
df_twitter = datasets["Twitter"]["df"]

print("\nSCM Score Calculation and Coverage Check Completed for all datasets.")


# 7. Baseline Model Training Function
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels) if not isinstance(labels, (list, np.ndarray)) else labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(self.encodings[key][idx]) for key in self.encodings.keys()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1_macro': f1, 'precision_macro': precision, 'recall_macro': recall}


def batch_tokenize(tokenizer, texts, max_len, batch_size=1000, dataset_name="Unknown"):
    encodings = {'input_ids': [], 'attention_mask': []}
    print(f"Tokenizing {len(texts)} texts for {dataset_name} in batches of {batch_size}…")
    iterator = range(0, len(texts), batch_size)
    iterator = tqdm(iterator, desc=f"Tokenizing {dataset_name}", leave=False) if use_tqdm else iterator

    for i in iterator:
        batch = texts[i:i + batch_size]
        processed_batch = [str(text) if pd.notna(text) else "" for text in batch]
        try:
            encoded_batch = tokenizer(
                processed_batch,
                truncation=True,
                padding='max_length',
                max_length=max_len,
                return_tensors=None,
            )
            encodings['input_ids'].extend(encoded_batch['input_ids'])
            encodings['attention_mask'].extend(encoded_batch['attention_mask'])
        except Exception as e:
            print(f"\nError tokenizing batch starting at index {i} for {dataset_name}. Error: {e}")
            num_skipped = len(processed_batch)
            pad_id = tokenizer.pad_token_id or 0
            encodings['input_ids'].extend([[pad_id] * max_len] * num_skipped)
            encodings['attention_mask'].extend([[0] * max_len] * num_skipped)

    if len(encodings['input_ids']) != len(texts):
        print(f"FATAL ERROR during tokenization for {dataset_name}: Length mismatch.")
        sys.exit(1)

    print(f"Tokenization for {dataset_name} complete.")
    return encodings


def train_and_evaluate_baseline(df, text_col, label_col, dataset_name, model_name, max_len, num_epochs=3,
                                train_batch_size=16, eval_batch_size=32):
    print(f"\n Starting Baseline Training for {dataset_name} ")
    base_output_dir = f'./results_baseline_{dataset_name.lower()}_5class'
    log_dir = f'./logs_baseline_{dataset_name.lower()}_5class'
    final_model_dir = f"{base_output_dir}/final_model"

    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    if label_col not in df.columns or text_col not in df.columns:
        print(f"ERROR: Missing columns in DataFrame for {dataset_name}. Skipping training.")
        return None

    df['model_label'] = df[label_col] - 1  # Map 1‑5 to 0‑4

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df[text_col].tolist(), df['model_label'].tolist(), test_size=0.2, random_state=42,
        stratify=df['model_label'].tolist()
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_encodings = batch_tokenize(tokenizer, train_texts, max_len, dataset_name=f"{dataset_name} Train")
    val_encodings = batch_tokenize(tokenizer, val_texts, max_len, dataset_name=f"{dataset_name} Val")
    test_encodings = batch_tokenize(tokenizer, test_texts, max_len, dataset_name=f"{dataset_name} Test")

    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    print(f"Loading pre‑trained model: {model_name}…")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5).to(device)

    training_args = TrainingArguments(
        output_dir=base_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=log_dir,
        logging_steps=100,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        disable_tqdm=not use_tqdm,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print(f"--- Training started for {dataset_name} ---")
    train_result = trainer.train()
    print(f"--- Training finished for {dataset_name} ---")

    # trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    print("Evaluating on validation set…")
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    # trainer.log_metrics("validation", {"val_" + k: v for k, v in val_results.items()})
    trainer.save_metrics("validation", {"val_" + k: v for k, v in val_results.items()})

    print("Evaluating on test set…")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    # trainer.log_metrics("test", {"test_" + k: v for k, v in test_results.items()})
    trainer.save_metrics("test", {"test_" + k: v for k, v in test_results.items()})

    print(f"===== Finished Baseline Training for {dataset_name} =====")
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {k.replace('eval_', ''): v for k, v in test_results.items()}


# Run Baseline Training for All Datasets
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 256
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256

all_results = {}

for name, data in datasets.items():
    print(f"\n{'=' * 30} PROCESSING DATASET: {name} {'=' * 30}")
    df_current = data["df"]
    text_col = data["text_col"]
    label_col = 'label_5class'

    if df_current.empty or df_current[text_col].isnull().all():
        print(f"Skipping {name} due to empty or invalid text column.")
        continue

    results = train_and_evaluate_baseline(
        df=df_current.copy(),
        text_col=text_col,
        label_col=label_col,
        dataset_name=name,
        model_name=MODEL_NAME,
        max_len=MAX_LEN,
        num_epochs=NUM_EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE
    )
    if results:
        all_results[name] = results

# 9. Display Final Results Summary and Data Samples

print(" Overall Baseline Training Summary (Test Set Metrics)")

if not all_results:
    print("No models were trained successfully.")
else:
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    cols_order = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'loss', 'runtime', 'samples_per_second',
                  'steps_per_second']
    results_df = results_df[[c for c in cols_order if c in results_df.columns]]
    display(results_df)

    summary_path = "./baseline_test_results_summary.csv"
    try:
        results_df.to_csv(summary_path)
        print(f"Test results summary saved to {summary_path}")
    except Exception as e:
        print(f"Error saving summary: {e}")

# print("\n--- Displaying DataFrame samples with SCM scores ---")

# print("\n--- Amazon Sample ---")
# print(df_amazon[['reviewText', 'label_5class', 'warmth_scm', 'competence_scm', 'matched_token_count']].head().to_string())

# print("\n--- Movie Sample ---")
# print(df_movie[['review_text_movie', 'label_5class', 'warmth_scm', 'competence_scm', 'matched_token_count']].head().to_string())

# print("\n--- Twitter Sample ---")
# print(df_twitter[['clean_text', 'label_5class', 'warmth_scm', 'competence_scm', 'matched_token_count']].head().to_string())

print("\nScript finished.")
