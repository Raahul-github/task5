

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ------------------------------
# Configuration
# ------------------------------
CSV_PATH = 'C:/Users/Asus/Desktop/complaints.csv'  # <-- set this to the downloaded CSV
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Mapping of product keywords to target labels (user requested categories)
LABEL_MAP = {
    'credit reporting': 0,
    'credit repair': 0,
    'debt collection': 1,
    'consumer loan': 2,
    'mortgage': 3,
    'mortgage loan': 3,
}

LABEL_NAMES = {
    0: 'Credit reporting, repair, or other',
    1: 'Debt collection',
    2: 'Consumer Loan',
    3: 'Mortgage'
}

# ------------------------------
# Helpers
# ------------------------------

def detect_label_from_product(product_value: str):
    if not isinstance(product_value, str):
        return None
    pv = product_value.lower()
    for kw, label in LABEL_MAP.items():
        if kw in pv:
            return label
    return None


def choose_text_column(df: pd.DataFrame):
    # prefer these columns if present
    candidates = [
        'Consumer complaint narrative', 'consumer_complaint_narrative',
        'complaint_what_happened', 'Complaint', 'complaint', 'Narrative'
    ]
    for c in candidates:
        if c in df.columns:
            # return if there is at least some non-null values
            if df[c].notnull().sum() > 0:
                return c
    # fallback: try to combine 'Issue' + 'Company public response' or similar
    if 'Issue' in df.columns:
        return 'Issue'
    # as last resort use 'Product' or 'Sub-product' (not ideal but something)
    if 'Product' in df.columns:
        return 'Product'
    raise ValueError('No suitable text column found in CSV. Please inspect your file and update the script.')


# ------------------------------
# Load data
# ------------------------------
print('Loading CSV:', CSV_PATH)
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Download the CSV and set CSV_PATH accordingly.")

df = pd.read_csv(CSV_PATH, low_memory=False)
print('Original rows:', len(df))

# ------------------------------
# Create labels
# ------------------------------
# Attempt to map existing "Product" column to our 4 labels
if 'Product' in df.columns:
    df['label'] = df['Product'].apply(detect_label_from_product)
else:
    df['label'] = None

# If dataset already has a 'product' or a more explicit mapping, adjust above mapping.
# Keep only rows for which we have target labels
before = len(df)
df = df[df['label'].notnull()].copy()
after = len(df)
print(f'Kept {after} rows ({after/before:.1%}) after mapping Product -> labels')

if after == 0:
    raise ValueError('No rows mapped to the requested labels. Please update LABEL_MAP to match your CSV Product values.')

# ------------------------------
# Choose text column for features
# ------------------------------
text_col = choose_text_column(df)
print('Using text column:', text_col)

# create a text field
# sometimes narratives are truncated; fallback to Issue + Sub-issue combination
if text_col not in df.columns:
    raise ValueError('Chosen text column not in dataframe')

# create final text column
df['text'] = df[text_col].fillna('')

# optionally concatenate 'Issue' or 'Sub-issue' if exists and text is short
if 'Issue' in df.columns:
    df['text'] = (df['text'].astype(str) + ' ' + df['Issue'].astype(str)).str.strip()

# basic EDA
print('\nTarget distribution:')
print(df['label'].value_counts().sort_index().rename(index=LABEL_NAMES))

# Add simple features
df['text_len'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].astype(str).apply(lambda s: len(s.split()))

print('\nText length stats:')
print(df[['text_len','word_count']].describe())

# show top n-grams function
from sklearn.feature_extraction.text import CountVectorizer

def top_ngrams(corpus, n=20, ngram_range=(1,1)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=5000)
    X = vec.fit_transform(corpus)
    counts = np.asarray(X.sum(axis=0)).ravel()
    freqs = sorted(zip(vec.get_feature_names_out(), counts), key=lambda x: -x[1])[:n]
    return freqs

print('\nTop unigrams overall:')
print(top_ngrams(df['text'].astype(str).values, n=20, ngram_range=(1,1)))
print('\nTop bigrams overall:')
print(top_ngrams(df['text'].astype(str).values, n=20, ngram_range=(2,2)))

# ------------------------------
# Train/test split
# ------------------------------
X = df['text'].astype(str).values
y = df['label'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print(f'X_train: {len(X_train)}, X_test: {len(X_test)}')

# ------------------------------
# Preprocessing + Model pipelines
# ------------------------------
# Tfidf settings
tfidf_params = {
    'ngram_range': (1,2),
    'max_df': 0.9,
    'min_df': 3,
    'max_features': 20000,
}

vectorizer = TfidfVectorizer(**tfidf_params, stop_words='english')

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced'),
    'MultinomialNB': MultinomialNB(),
}

results = {}

for name, clf in models.items():
    print('\nTraining pipeline for', name)
    pipe = Pipeline([('tfidf', vectorizer), ('clf', clf)])
    # quick cross-val
    scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro')
    print('Cross-val F1-macro:', scores, 'mean:', scores.mean())
    # fit on full train
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print('Test accuracy:', acc)
    print('Test F1-macro:', f1)
    print('Classification report:\n', classification_report(y_test, y_pred, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]))
    results[name] = {'pipe': pipe, 'accuracy': acc, 'f1_macro': f1}

# ------------------------------
# Compare models
# ------------------------------
print('\nModel comparison:')
for name, r in results.items():
    print(name, 'Acc:', r['accuracy'], 'F1-macro:', r['f1_macro'])

# pick best by f1_macro
best_name = max(results.keys(), key=lambda n: results[n]['f1_macro'])
best_pipe = results[best_name]['pipe']
print('\nBest model:', best_name)

# Confusion matrix for best
from sklearn.metrics import ConfusionMatrixDisplay

y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
print('\nConfusion matrix:\n', cm)

# save best model
model_out = 'best_consumer_complaint_model.joblib'
joblib.dump({'pipe': best_pipe, 'label_names': LABEL_NAMES}, model_out)
print('Saved best model to', model_out)

# ------------------------------
# Prediction function
# ------------------------------

def predict_texts(texts, model_file=model_out):
    data = joblib.load(model_file)
    pipe = data['pipe']
    label_names = data['label_names']
    preds = pipe.predict(texts)
    return [label_names[int(p)] for p in preds]

# example predictions
examples = [
    "I pulled my credit report and found an error with an account that isn't mine. The credit bureau won't remove it.",
    "I received a call from a collector about a debt I paid. They threaten to sue and the amount seems wrong.",
    "I applied for a personal loan but the lender says my application was rejected with no reason.",
    "The mortgage servicer has been charging incorrect escrow amounts and is not responding." 
]

print('\nExample predictions:')
for t, p in zip(examples, predict_texts(examples)):
    print('-', p, '->', t[:120], '...')

# ------------------------------
# Save evaluation report
# ------------------------------
report = classification_report(y_test, y_pred_best, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())], output_dict=True)
with open('evaluation_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print('Saved evaluation_report.json')

print('\nDone. Inspect saved files: best_consumer_complaint_model.joblib, evaluation_report.json')
