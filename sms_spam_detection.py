import os
import re
import urllib.request
import ssl
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

data_dir = r"C:\Users\Ramakrishna\.gemini\antigravity\scratch\sms_spam_project\data"
os.makedirs(data_dir, exist_ok=True)
data_file = os.path.join(data_dir, "SMSSpamCollection")

if not os.path.exists(data_file):
    print("Downloading dataset from UCI repository...")
    ssl_context = ssl._create_unverified_context()
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = os.path.join(data_dir, "sms.zip")
    try:
        with urllib.request.urlopen(url, context=ssl_context) as resp, open(zip_path, "wb") as f:
            f.write(resp.read())
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
    except Exception as e:
        print("Direct UCI zip failed, trying mirror:", e)
        mirror_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        with urllib.request.urlopen(mirror_url, context=ssl_context) as resp, open(data_file, "wb") as f:
            f.write(resp.read())

print("Loading dataset...")
df = pd.read_csv(data_file, sep='\t', names=['label', 'message'], header=None)

# 1. Dataset Handling
print("\n" + "="*50)
print("ACTIVITY 1: DATASET HANDLING")
print("="*50)
print(f"Original shape: {df.shape[0]} rows, {df.shape[1]} columns")
missing_vals = df.isnull().sum()
print("Missing values per column:\n", missing_vals)

duplicates_count = df.duplicated().sum()
print(f"Duplicate messages detected: {duplicates_count}")
df_cleaned = df.drop_duplicates().copy()
print(f"Cleaned dataset shape after removing duplicates: {df_cleaned.shape[0]} rows")

# Convert labels into numerical values: Spam = 1, Ham = 0
df_cleaned['label_num'] = df_cleaned['label'].map({'ham': 0, 'spam': 1})
print("Sample labeled rows:")
print(df_cleaned[['label', 'label_num', 'message']].head(5))

# 2. Dataset Analysis
print("\n" + "="*50)
print("ACTIVITY 2: DATASET ANALYSIS & PRIOR PROBABILITIES")
print("="*50)
total_msgs = len(df_cleaned)
spam_count = (df_cleaned['label_num'] == 1).sum()
ham_count = (df_cleaned['label_num'] == 0).sum()

p_spam = spam_count / total_msgs
p_ham = ham_count / total_msgs

print(f"Total messages (N): {total_msgs}")
print(f"Ham count: {ham_count} | P(Ham) = {p_ham:.5f} ({p_ham*100:.2f}%)")
print(f"Spam count: {spam_count} | P(Spam) = {p_spam:.5f} ({p_spam*100:.2f}%)")

# Save class distribution chart
plt.figure(figsize=(7, 5))
bars = plt.bar(['Ham (Genuine)', 'Spam'], [ham_count, spam_count], color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
plt.title('SMS Spam Collection Class Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Number of Messages', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval} ({yval/total_msgs*100:.1f}%)", ha='center', va='bottom', fontweight='bold')
chart_path = r"C:\Users\Ramakrishna\.gemini\antigravity\scratch\sms_spam_project\class_distribution.png"
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Class distribution chart saved to: {chart_path}")

# 3. Word-Based Probability Analysis
print("\n" + "="*50)
print("ACTIVITY 3: WORD-BASED PROBABILITY ANALYSIS & BAYES THEOREM")
print("="*50)
suspicious_words = ['free', 'prize', 'win', 'urgent', 'offer']

def contains_word(text, word):
    return bool(re.search(r'\b' + re.escape(word) + r'\b', str(text).lower()))

word_stats = []
for w in suspicious_words:
    mask_w = df_cleaned['message'].apply(lambda msg: contains_word(msg, w))
    count_w = mask_w.sum()
    count_spam_and_w = (mask_w & (df_cleaned['label_num'] == 1)).sum()
    count_ham_and_w = (mask_w & (df_cleaned['label_num'] == 0)).sum()
    
    # Probabilities
    p_w = count_w / total_msgs
    p_w_given_spam = count_spam_and_w / spam_count if spam_count > 0 else 0
    p_w_given_ham = count_ham_and_w / ham_count if ham_count > 0 else 0
    
    # Empirical conditional probability
    empirical_p_spam_given_w = count_spam_and_w / count_w if count_w > 0 else 0
    
    # Bayes' Theorem: P(Spam | W) = [P(W | Spam) * P(Spam)] / P(W)
    bayes_p_spam_given_w = (p_w_given_spam * p_spam) / p_w if p_w > 0 else 0
    
    word_stats.append({
        'Word': w,
        'Count_Total': count_w,
        'Count_in_Spam': count_spam_and_w,
        'Count_in_Ham': count_ham_and_w,
        'P(Word)': p_w,
        'P(Word|Spam)': p_w_given_spam,
        'P(Word|Ham)': p_w_given_ham,
        'Empirical_P(Spam|Word)': empirical_p_spam_given_w,
        'Bayes_P(Spam|Word)': bayes_p_spam_given_w,
        'Match': np.isclose(empirical_p_spam_given_w, bayes_p_spam_given_w)
    })

word_df = pd.DataFrame(word_stats)
print(word_df[['Word', 'Count_Total', 'Count_in_Spam', 'Count_in_Ham', 'P(Word)', 'P(Word|Spam)', 'Empirical_P(Spam|Word)', 'Bayes_P(Spam|Word)', 'Match']].to_string(index=False))

# 4. Random Variable Analysis
print("\n" + "="*50)
print("ACTIVITY 4: RANDOM VARIABLE ANALYSIS")
print("="*50)
p = p_spam
e_y = p
var_y = p * (1 - p)
std_y = np.sqrt(var_y)
print(f"Random Variable Y ~ Bernoulli(p = {p:.5f})")
print(f"Expectation E(Y) = P(Spam) = {e_y:.5f}")
print(f"Variance Var(Y) = p * (1 - p) = {var_y:.5f}")
print(f"Standard Deviation Std(Y) = sqrt(Var(Y)) = {std_y:.5f}")

# 5. Data Preparation
print("\n" + "="*50)
print("ACTIVITY 5: DATA PREPARATION (STRATIFIED SPLIT & COUNTVECTORIZER)")
print("="*50)
X = df_cleaned['message']
y = df_cleaned['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)} ({len(X_train)/len(df_cleaned)*100:.1f}%)")
print(f"Testing set size:  {len(X_test)} ({len(X_test)/len(df_cleaned)*100:.1f}%)")
print(f"Train Spam Proportion: {y_train.mean():.4f}")
print(f"Test Spam Proportion:  {y_test.mean():.4f}")

vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Vocabulary size (features extracted): {len(vectorizer.vocabulary_)}")

# 6. Model Building
print("\n" + "="*50)
print("ACTIVITY 6: MODEL BUILDING (MULTINOMIAL NAIVE BAYES)")
print("="*50)
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("MultinomialNB model trained successfully.")

# 7. Model Evaluation
print("\n" + "="*50)
print("ACTIVITY 7: MODEL EVALUATION")
print("="*50)
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall   : {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-score : {f1:.4f}")
print("\nConfusion Matrix:")
tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives (TN - Ham correctly identified): {tn}")
print(f"  False Positives (FP - Ham incorrectly identified as Spam): {fp}")
print(f"  False Negatives (FN - Spam incorrectly identified as Ham): {fn}")
print(f"  True Positives (TP - Spam correctly identified): {tp}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], digits=4))

# 8. Real-time Testing
print("\n" + "="*50)
print("ACTIVITY 8: REAL-TIME TESTING")
print("="*50)
sample_messages = [
    "Congratulations! You have won a free prize. Claim now.",
    "Hey bro, are we still meeting in the library for statistics study?",
    "URGENT! Your mobile account has won 1000 cash bonus. Call now to claim your offer.",
    "Can you please pick up milk on your way home?"
]

for msg in sample_messages:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "Spam" if pred == 1 else "Ham"
    print(f"\nMessage: \"{msg}\"")
    print(f"Prediction: {label}")
    print(f"P(Ham | Message)  = {prob[0]:.4f}")
    print(f"P(Spam | Message) = {prob[1]:.4f}")
