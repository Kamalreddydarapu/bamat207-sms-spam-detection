import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# BAMAT207: Applications of Probability and Statistics\n",
    "## Project Activity: Probabilistic SMS Spam Detection\n",
    "**Topic:** Probability, Random Variables & Naive Bayes Classification  \n",
    "**Institution:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology  \n",
    "**Dataset:** UCI Machine Learning Repository - SMS Spam Collection (5,574 messages)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### Activity 1: Dataset Handling & Preprocessing"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import re\n",
    "import urllib.request\n",
    "import ssl\n",
    "import zipfile\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n\n",
    "# Ensure dataset is downloaded\n",
    "data_dir = 'data'\n",
    "os.makedirs(data_dir, exist_ok=True)\n",
    "data_file = os.path.join(data_dir, 'SMSSpamCollection')\n\n",
    "if not os.path.exists(data_file):\n",
    "    ssl_context = ssl._create_unverified_context()\n",
    "    try:\n",
    "        url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'\n",
    "        zip_path = os.path.join(data_dir, 'sms.zip')\n",
    "        with urllib.request.urlopen(url, context=ssl_context) as resp, open(zip_path, 'wb') as f:\n",
    "            f.write(resp.read())\n",
    "        with zipfile.ZipFile(zip_path, 'r') as z:\n",
    "            z.extractall(data_dir)\n",
    "    except Exception:\n",
    "        mirror_url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'\n",
    "        with urllib.request.urlopen(mirror_url, context=ssl_context) as resp, open(data_file, 'wb') as f:\n",
    "            f.write(resp.read())\n\n",
    "# Load dataset\n",
    "df = pd.read_csv(data_file, sep='\\t', names=['label', 'message'], header=None)\n",
    "print('Original dataset dimensions:', df.shape)\n",
    "print('Missing value count:\\n', df.isnull().sum())\n\n",
    "# Remove duplicates\n",
    "initial_len = len(df)\n",
    "df_cleaned = df.drop_duplicates().copy()\n",
    "print(f'Duplicate records removed: {initial_len - len(df_cleaned)}')\n",
    "print(f'Cleaned dataset dimensions: {df_cleaned.shape}')\n\n",
    "# Convert labels into numerical values (Spam = 1, Ham = 0)\n",
    "df_cleaned['label_num'] = df_cleaned['label'].map({'ham': 0, 'spam': 1})\n",
    "df_cleaned.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### Activity 2: Dataset Analysis & Prior Probabilities"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "total_msgs = len(df_cleaned)\n",
    "spam_count = (df_cleaned['label_num'] == 1).sum()\n",
    "ham_count = (df_cleaned['label_num'] == 0).sum()\n\n",
    "p_spam = spam_count / total_msgs\n",
    "p_ham = ham_count / total_msgs\n\n",
    "print(f'Total messages (N) = {total_msgs}')\n",
    "print(f'P(Spam) = {spam_count} / {total_msgs} = {p_spam:.5f} ({p_spam*100:.2f}%)')\n",
    "print(f'P(Ham)  = {ham_count} / {total_msgs} = {p_ham:.5f} ({p_ham*100:.2f}%)')\n\n",
    "# Class distribution bar chart\n",
    "plt.figure(figsize=(7, 4.5))\n",
    "colors = ['#2ecc71', '#e74c3c']\n",
    "bars = plt.bar(['Ham (0)', 'Spam (1)'], [ham_count, spam_count], color=colors, width=0.45, edgecolor='black')\n",
    "plt.title('SMS Dataset Class Distribution (BAMAT207)', fontsize=13, fontweight='bold')\n",
    "plt.xlabel('Message Category', fontsize=11)\n",
    "plt.ylabel('Count', fontsize=11)\n",
    "for bar in bars:\n",
    "    h = bar.get_height()\n",
    "    plt.text(bar.get_x() + bar.get_width()/2.0, h + 50, f'{h} ({h/total_msgs*100:.2f}%)', ha='center', fontweight='bold')\n",
    "plt.grid(axis='y', linestyle='--', alpha=0.6)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### Activity 3: Word-Based Probability Analysis & Bayes' Theorem Verification"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "suspicious_words = ['free', 'prize', 'win', 'urgent', 'offer']\n",
    "word_results = []\n\n",
    "def contains_word(text, w):\n",
    "    return bool(re.search(r'\\b' + re.escape(w) + r'\\b', str(text).lower()))\n\n",
    "for w in suspicious_words:\n",
    "    mask = df_cleaned['message'].apply(lambda msg: contains_word(msg, w))\n",
    "    count_w = mask.sum()\n",
    "    count_spam_w = (mask & (df_cleaned['label_num'] == 1)).sum()\n",
    "    count_ham_w = (mask & (df_cleaned['label_num'] == 0)).sum()\n",
    "    \n",
    "    p_w = count_w / total_msgs\n",
    "    p_w_given_spam = count_spam_w / spam_count\n",
    "    empirical_p_spam_w = count_spam_w / count_w if count_w > 0 else 0\n",
    "    bayes_p_spam_w = (p_w_given_spam * p_spam) / p_w if p_w > 0 else 0\n",
    "    \n",
    "    word_results.append({\n",
    "        'Word': w,\n",
    "        'Total Msgs Containing Word': count_w,\n",
    "        'Spam Msgs': count_spam_w,\n",
    "        'Ham Msgs': count_ham_w,\n",
    "        'P(Word)': round(p_w, 5),\n",
    "        'P(Word | Spam)': round(p_w_given_spam, 5),\n",
    "        'Empirical P(Spam | Word)': round(empirical_p_spam_w, 5),\n",
    "        'Bayes P(Spam | Word)': round(bayes_p_spam_w, 5),\n",
    "        'Theorem Verified': np.isclose(empirical_p_spam_w, bayes_p_spam_w)\n",
    "    })\n\n",
    "word_df = pd.DataFrame(word_results)\n",
    "word_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### Activity 4: Random Variable Analysis"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "p = p_spam\n",
    "e_y = p\n",
    "var_y = p * (1 - p)\n",
    "std_y = np.sqrt(var_y)\n\n",
    "print('Bernoulli Random Variable Y (1=Spam, 0=Ham):')\n",
    "print(f'Parameter p = P(Spam) = {p:.5f}')\n",
    "print(f'Expectation E(Y) = p = {e_y:.5f}')\n",
    "print(f'Variance Var(Y) = p * (1 - p) = {var_y:.5f}')\n",
    "print(f'Standard Deviation = sqrt(Var(Y)) = {std_y:.5f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### Activity 5 & 6: Data Preparation and Multinomial Naive Bayes Model Building"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df_cleaned['message']\n",
    "y = df_cleaned['label_num']\n\n",
    "# 80-20 Stratified Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.20, random_state=42, stratify=y\n",
    ")\n",
    "print(f'Training Set: {len(X_train)} samples ({y_train.mean():.4f} spam ratio)')\n",
    "print(f'Testing Set:  {len(X_test)} samples ({y_test.mean():.4f} spam ratio)')\n\n",
    "# CountVectorizer feature extraction\n",
    "vectorizer = CountVectorizer(lowercase=True, stop_words='english')\n",
    "X_train_vec = vectorizer.fit_transform(X_train)\n",
    "X_test_vec = vectorizer.transform(X_test)\n",
    "print(f'Feature dimensions (Vocabulary size): {len(vectorizer.vocabulary_)}')\n\n",
    "# Train Multinomial Naive Bayes\n",
    "model = MultinomialNB()\n",
    "model.fit(X_train_vec, y_train)\n",
    "print('MultinomialNB model training complete.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### Activity 7: Model Evaluation & Performance Metrics"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test_vec)\n\n",
    "acc = accuracy_score(y_test, y_pred)\n",
    "prec = precision_score(y_test, y_pred)\n",
    "rec = recall_score(y_test, y_pred)\n",
    "f1 = f1_score(y_test, y_pred)\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "tn, fp, fn, tp = cm.ravel()\n\n",
    "print(f'Accuracy : {acc:.4f} ({acc*100:.2f}%)')\n",
    "print(f'Precision: {prec:.4f} ({prec*100:.2f}%)')\n",
    "print(f'Recall   : {rec:.4f} ({rec*100:.2f}%)')\n",
    "print(f'F1-score : {f1:.4f}')\n\n",
    "print('\\nConfusion Matrix breakdown:')\n",
    "print(f'  True Negatives (Ham correctly identified)      : {tn}')\n",
    "print(f'  False Positives (Ham incorrectly flagged spam) : {fp}')\n",
    "print(f'  False Negatives (Spam undetected / missed)     : {fn}')\n",
    "print(f'  True Positives (Spam correctly blocked)        : {tp}')\n\n",
    "print('\\nFull Classification Report:\\n')\n",
    "print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'], digits=4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### Activity 8: Real-Time SMS Message Classification Testing"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def classify_sms(message_text):\n",
    "    vec = vectorizer.transform([message_text])\n",
    "    prediction = model.predict(vec)[0]\n",
    "    probs = model.predict_proba(vec)[0]\n",
    "    label = 'Spam' if prediction == 1 else 'Ham'\n",
    "    \n",
    "    print(f'Input Message: \"{message_text}\"')\n",
    "    print(f'Prediction: {label}')\n",
    "    print(f'P(Ham | Message)  = {probs[0]:.4f}')\n",
    "    print(f'P(Spam | Message) = {probs[1]:.4f}')\n",
    "    print('-'*50)\n\n",
    "# Activity Example\n",
    "classify_sms('Congratulations! You have won a free prize. Claim now.')\n",
    "classify_sms('Hey, are we still meeting for lunch tomorrow?')\n",
    "classify_sms('URGENT! Call 09061701461 to claim your £1000 cash offer now.')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open(r'C:\Users\Ramakrishna\.gemini\antigravity\scratch\sms_spam_project\sms_spam_detection.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=2)

print('Successfully created sms_spam_detection.ipynb')
