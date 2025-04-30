import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import sys
import time


try:
    from tqdm.auto import tqdm
    tqdm.pandas()
    use_tqdm = True
except ImportError:
    use_tqdm = False


dict_path = "C:/Users/zcx99/Desktop/try/Task-Adaptive-Debiasing-with-SCM-for-Sentiment-Analysis/FullDictionaries.csv"
amazon_appliances_path = "C:/Users/zcx99/Desktop/try/Task-Adaptive-Debiasing-with-SCM-for-Sentiment-Analysis/dataset/amazon_Appliances_5_jiadian.json"
amazon_fashion_path = "C:/Users/zcx99/Desktop/try/Task-Adaptive-Debiasing-with-SCM-for-Sentiment-Analysis/dataset/AMAZON_FASHION_5_shishang.json"
amazon_beauty_path = "C:/Users/zcx99/Desktop/try/Task-Adaptive-Debiasing-with-SCM-for-Sentiment-Analysis/dataset/All_Beauty_5_meizhuang.json"
movie_path = "C:/Users/zcx99/Desktop/try/Task-Adaptive-Debiasing-with-SCM-for-Sentiment-Analysis/dataset/Movie Reviews_train.tsv"
twitter_path = "C:/Users/zcx99/Desktop/try/Task-Adaptive-Debiasing-with-SCM-for-Sentiment-Analysis/dataset/train-twitter.tsv"


lookup_key_col = 'preprocessed word 4 (minus one trailing s)'
sociability_dict_col = 'Sociability dictionary'
sociability_dir_col = 'Sociability direction'
morality_dict_col = 'Morality dictionary'
morality_dir_col = 'Morality direction'
ability_dict_col = 'Ability dictionary'
ability_dir_col = 'Ability direction'
agency_dict_col = 'Agency dictionary'
agency_dir_col = 'Agency direction'



try:
    df_scm = pd.read_csv(
        dict_path,
        header=0
    )
    print("Successfully loaded SCM dictionary.")
except FileNotFoundError:
    print(f"ERROR: SCM Dictionary file not found at '{dict_path}'. Please check the path.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading SCM dictionary file '{dict_path}': {e}")
    sys.exit(1)


try:
   #print(f"Loading Amazon Appliances: {amazon_appliances_path}")
    df_amazon_Appliances = pd.read_json(amazon_appliances_path, lines=True)

    #print(f"Loading Amazon Fashion: {amazon_fashion_path}")
    df_amazon_fashion = pd.read_json(amazon_fashion_path, lines=True)

    #print(f"Loading Amazon Beauty: {amazon_beauty_path}")
    df_amazon_beauty  = pd.read_json(amazon_beauty_path, lines=True)

    df_amazon_Appliances['source'] = 'Appliances'
    df_amazon_fashion['source'] = 'Fashion'
    df_amazon_beauty['source']  = 'Beauty'
    df_amazon = pd.concat([df_amazon_Appliances, df_amazon_fashion, df_amazon_beauty],
                          ignore_index=True)
    #print(f"Combined Amazon datasets ({len(df_amazon)} rows).")

    #print(f"Loading Movie Reviews: {movie_path}")
    df_movie = pd.read_csv(movie_path, sep='\t', header=0)
    df_movie.rename(columns={'Phrase': 'review_text_movie'}, inplace=True)
    #print(f"Loaded Movie dataset ({len(df_movie)} rows). Renamed 'Phrase' to 'review_text_movie'.")

    #print(f"Loading Twitter: {twitter_path}")
    df_twitter = pd.read_csv(
        twitter_path,
        sep='\t',
        header=None,
        quoting=3,
        engine='python',
        on_bad_lines='skip'
    )
    df_twitter.columns = ['tweet_id', 'user', 'label', 'text'] # Assign column names
    df_twitter['text'] = df_twitter['text'].replace("Not Available", np.nan)
    df_twitter.dropna(subset=['text'], inplace=True)
    #print(f"Loaded and cleaned Twitter dataset ({len(df_twitter)} rows).")

except FileNotFoundError as e:
     print(f"\nERROR: Dataset file not found. Please check the path in the error message below.")
     print(e)
     sys.exit(1)
except Exception as e:
     print(f"\nERROR processing dataset files: {e}")
     sys.exit(1)
print("All datasets loaded successfully.")



# lookup dictionary
print("\nBuilding SCM lookup dictionary...")
if lookup_key_col not in df_scm.columns:
    print(f"ERROR: The specified lookup key column '{lookup_key_col}' does not exist in the SCM dictionary header.")
    print(f"Available columns are: {df_scm.columns.tolist()}")
    sys.exit(1)


df_scm.dropna(subset=[lookup_key_col], inplace=True)

df_scm[lookup_key_col] = df_scm[lookup_key_col].astype(str)

df_scm_indexed = df_scm.set_index(lookup_key_col, drop=False)

lookup_scm = {}

for word_key, row_data in df_scm_indexed.iterrows():
    lookup_scm[str(word_key)] = row_data.to_dict()

print(f"Built lookup dictionary with {len(lookup_scm)} entries.")

# def calculate_coverage(df, text_column, lookup_dict, dataset_name):
#     """Calculates token and type coverage of a dictionary for a given DataFrame column."""
#     print(f"\nCalculating dictionary coverage for '{dataset_name}' ({text_column})...")
#     start_time = time.time()
#     total_tokens = 0
#     matched_tokens = 0
#     unique_words_in_data = set()
#     matched_unique_words = set()
#
#     # Check if the text column exists
#     if text_column not in df.columns:
#         print(f"  Error: Column '{text_column}' not found in the DataFrame for {dataset_name}.")
#         return
#
#     # Use tqdm for progress if available
#     iterable = df[text_column]
#     if use_tqdm:
#         iterable = tqdm(df[text_column], desc=f"Coverage for {dataset_name}", total=len(df))
#
#     for text in iterable:
#         # Same tokenization as in compute_scm_scores
#         tokens = re.findall(r'[a-zA-Z]+', str(text).lower())
#         for token in tokens:
#             total_tokens += 1
#             unique_words_in_data.add(token)
#             if token in lookup_dict:
#                 matched_tokens += 1
#                 matched_unique_words.add(token)
#
#     end_time = time.time()
#     duration = end_time - start_time
#
#     # Calculate percentages
#     token_coverage = (matched_tokens / total_tokens * 100) if total_tokens > 0 else 0
#     type_coverage = (len(matched_unique_words) / len(unique_words_in_data) * 100) if unique_words_in_data else 0
#
#     print(f"Coverage calculation for '{dataset_name}' completed in {duration:.2f} seconds.")
#     print(f"  Total Tokens processed: {total_tokens:,}")
#     print(f"  Unique Word Types found: {len(unique_words_in_data):,}")
#     print(f"  Dictionary Tokens Matched: {matched_tokens:,} ({token_coverage:.2f}%)")
#     print(f"  Dictionary Types Matched (Vocabulary Coverage): {len(matched_unique_words):,} ({type_coverage:.2f}%)")
#
# # --- Calculate Coverage for each dataset ---
# calculate_coverage(df_amazon, 'reviewText', lookup_scm, "Amazon Reviews")
# calculate_coverage(df_movie, 'review_text_movie', lookup_scm, "Movie Reviews")
# calculate_coverage(df_twitter, 'text', lookup_scm, "Twitter Data")
# # ------------------------------------------



# compute SCM scores
def compute_scm_scores(text, lookup_dict):
    tokens = re.findall(r'[a-zA-Z]+', str(text).lower())

    sum_sociability, cnt_sociability = 0.0, 0
    sum_morality, cnt_morality = 0.0, 0
    sum_ability, cnt_ability = 0.0, 0
    sum_agency, cnt_agency = 0.0, 0
    matched_tokens_count = 0

    for tok in tokens:
        if tok in lookup_dict:
            matched_tokens_count += 1
            info = lookup_dict[tok]

            try:
                # Sociability
                if pd.notna(info.get(sociability_dict_col)) and int(info[sociability_dict_col]) == 1:
                    dir_val = info.get(sociability_dir_col)
                    if pd.notna(dir_val):
                        sum_sociability += float(dir_val)
                        cnt_sociability += 1
                # Morality
                if pd.notna(info.get(morality_dict_col)) and int(info[morality_dict_col]) == 1:
                    dir_val = info.get(morality_dir_col)
                    if pd.notna(dir_val):
                        sum_morality += float(dir_val)
                        cnt_morality += 1
                # Ability
                if pd.notna(info.get(ability_dict_col)) and int(info[ability_dict_col]) == 1:
                    dir_val = info.get(ability_dir_col)
                    if pd.notna(dir_val):
                        sum_ability += float(dir_val)
                        cnt_ability += 1
                # Agency
                if pd.notna(info.get(agency_dict_col)) and int(info[agency_dict_col]) == 1:
                    dir_val = info.get(agency_dir_col)
                    if pd.notna(dir_val):
                        sum_agency += float(dir_val)
                        cnt_agency += 1
            except (ValueError, TypeError, KeyError) as e:
                pass

    warmth_score = 0.0
    total_warm_count = cnt_sociability + cnt_morality
    if total_warm_count > 0:
        warmth_score = (sum_sociability + sum_morality) / total_warm_count

    competence_score = 0.0
    total_comp_count = cnt_ability + cnt_agency
    if total_comp_count > 0:
        competence_score = (sum_ability + sum_agency) / total_comp_count

    return warmth_score, competence_score, matched_tokens_count


def apply_scm_to_dataframe(df, text_column_name):

    if text_column_name not in df.columns:
        print(f"ERROR: Text column '{text_column_name}' not found in the DataFrame for SCM calculation.")
        return

    total_rows = len(df)
    print(f"\nCalculating SCM scores for column '{text_column_name}' in DataFrame ({total_rows} rows)...")

    if use_tqdm:
        results = df[text_column_name].progress_apply(lambda txt: compute_scm_scores(txt, lookup_scm))
    else:
        print("Processing... (this may take a while without tqdm)")
        results = df[text_column_name].apply(lambda txt: compute_scm_scores(txt, lookup_scm))

    df['warmth_scm'] = [res[0] for res in results]
    df['competence_scm'] = [res[1] for res in results]
    df['matched_token_count'] = [res[2] for res in results]

    print(f"Finished processing '{text_column_name}'.")
    try:
        any_match = (df['matched_token_count'] > 0).sum()
        non_zero_warmth = (df['warmth_scm'] != 0).sum()
        non_zero_competence = (df['competence_scm'] != 0).sum()

        if total_rows > 0:
             print(f"  Rows with any matched tokens: {any_match}/{total_rows} ({(any_match/total_rows)*100:.2f}%)")
             print(f"  Rows with non-zero Warmth: {non_zero_warmth}/{total_rows} ({(non_zero_warmth/total_rows)*100:.2f}%)")
             print(f"  Rows with non-zero Competence: {non_zero_competence}/{total_rows} ({(non_zero_competence/total_rows)*100:.2f}%)")
        else:
             print("  DataFrame is empty, no statistics to calculate.")
    except KeyError:
        print("  Warning: Could not calculate post-processing statistics due to missing columns.")






apply_scm_to_dataframe(df_amazon, 'reviewText')

apply_scm_to_dataframe(df_movie, 'review_text_movie')

apply_scm_to_dataframe(df_twitter, 'text')

print(" SCM Score Calculation Complete.")



print("\n--- First 5 Amazon Reviews ---")
amazon_cols_to_show = ['reviewText', 'warmth_scm', 'competence_scm', 'matched_token_count']
print(df_amazon[amazon_cols_to_show].head(5).to_string())

print("\n--- Amazon - Example Rows with Highest Warmth ---")
print(df_amazon.sort_values(by='warmth_scm', ascending=False)[amazon_cols_to_show].head(5).to_string())

print("\n--- Amazon - Example Rows with Lowest Warmth ---")
print(df_amazon[df_amazon['warmth_scm'] < 0].sort_values(by='warmth_scm', ascending=True)[amazon_cols_to_show].head(5).to_string())


# print("\n--- Score Distributions (Amazon Dataset) ---")
# try:
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(1, 2, 1)
#     sns.histplot(df_amazon[df_amazon['warmth_scm'] != 0]['warmth_scm'], kde=True)
#     plt.title('Amazon Warmth Score Distribution (non-zero)')
#
#     plt.subplot(1, 2, 2)
#     sns.histplot(df_amazon[df_amazon['competence_scm'] != 0]['competence_scm'], kde=True)
#     plt.title('Amazon Competence Score Distribution (non-zero)')
#
#     plt.tight_layout()
#     plt.show()
#
# except ImportError:
#      print("\nMatplotlib or Seaborn not installed. Cannot generate plots. Use 'pip install matplotlib seaborn'")
# except Exception as e:
#      print(f"\nCould not generate distribution plots: {e}")

