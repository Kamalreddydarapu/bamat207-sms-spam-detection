# BAMAT207: Applications of Probability and Statistics
## Course Project Report: Probabilistic SMS Spam Detection

---

**Course Code & Title:** BAMAT207 – Applications of Probability and Statistics  
**Topic:** Probability, Random Variables, Bayes’ Theorem, and Naive Bayes Classification  
**Department / School:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT)  
**Course Instructor:** Dr. C. R. Dhivyaa, Associate Professor Grade 1, SCOPE  
**Submission Deadline:** 27 October 2026 | **Project Demo:** 28 October 2026  

---

### Executive Summary

In mobile communications, Short Message Service (SMS) spam poses severe security risks, including phishing, malware dissemination, and financial fraud. This project explores the foundational principles of probability theory—specifically **prior probability**, **conditional probability**, **Bayes’ Theorem**, and **discrete random variables**—to engineer an automated, probabilistic text classification pipeline. 

Using the landmark UCI Machine Learning SMS Spam Collection dataset (5,574 raw messages), we:
1. Preprocessed and deduped the textual records into a clean corpus of 5,169 unique SMS messages.
2. Formulated class prior distributions: $P(\text{Spam}) = 0.12633$ and $P(\text{Ham}) = 0.87367$.
3. Performed word-level Bayesian inference on suspicious keywords (`free`, `prize`, `win`, `urgent`, `offer`), proving that empirical conditional frequencies match theoretical predictions from Bayes' Theorem with mathematical exactness.
4. Modeled the spam distribution as a **Bernoulli Random Variable** $Y$, computing its expected value $E(Y) = 0.12633$ and variance $\text{Var}(Y) = 0.11037$.
5. Trained and evaluated a **Multinomial Naive Bayes** classifier using stratified train-test splits (80%-20%) and count-vectorized n-gram representations.
6. Achieved an overall classification accuracy of **98.36%**, with a precision of **95.97%**, recall of **90.84%**, and an F1-score of **0.9333**.
7. Built a real-time message prediction engine capable of inferring class probabilities $P(\text{Spam} \mid \text{Message})$ and $P(\text{Ham} \mid \text{Message})$.

---

## 1. Activity 1: Dataset Handling & Preprocessing

### 1.1 Dataset Overview
The dataset used in this study is the benchmark **SMS Spam Collection Dataset** from the UCI Machine Learning Repository. It comprises 5,572 tab-separated SMS records manually curated from SMS spam services and mobile phone user archives.

* **Source:** UCI Machine Learning Repository / Almeida et al.
* **Initial Records:** 5,572 rows, 2 columns (`label`, `message`)
* **Labels:** Categorical strings: `ham` (legitimate) and `spam` (unsolicited commercial/fraudulent messages).

### 1.2 Data Integrity & Duplicate Removal
Data quality inspection revealed:
* **Missing Values:** Exactly 0 missing or null entries across both label and text fields.
* **Redundant Entries:** A total of **403 duplicate messages** were identified. Repeated messages typically arise from automated mass-broadcast spam campaigns or common conversational replies (e.g., "Sorry, I'll call later").
* **Deduplication:** Dropping duplicate records reduced the corpus size from 5,572 to **5,169 unique messages**, eliminating sampling bias in word frequency and conditional probability estimations.

### 1.3 Label Encoding
The target variable was mapped to a binary indicator:
$$\text{Label} = \begin{cases} 1, & \text{if message is } \text{Spam} \\ 0, & \text{if message is } \text{Ham} \end{cases}$$

```python
# Sample snippet from implementation
df_cleaned = df.drop_duplicates().copy()
df_cleaned['label_num'] = df_cleaned['label'].map({'ham': 0, 'spam': 1})
```

---

## 2. Activity 2: Dataset Analysis & Prior Probabilities

### 2.1 Frequency and Prior Probability Calculation
Let $S$ represent the event that an SMS message is spam, and $H$ represent the event that an SMS message is legitimate (ham). The prior probabilities represent our baseline belief before observing message text:

$$P(\text{Spam}) = \frac{N_{\text{Spam}}}{N_{\text{Total}}} = \frac{653}{5169} \approx 0.12633 \quad (12.63\%)$$

$$P(\text{Ham}) = \frac{N_{\text{Ham}}}{N_{\text{Total}}} = \frac{4516}{5169} \approx 0.87367 \quad (87.37\%)$$

| Class Metric | Ham (Genuine) | Spam (Unsolicited) | Total Dataset |
| :--- | :---: | :---: | :---: |
| **Count ($N$)** | 4,516 | 653 | 5,169 |
| **Proportion** | 87.37% | 12.63% | 100.0% |
| **Prior Probability** | **0.87367** | **0.12633** | **1.00000** |

### 2.2 Class Distribution Visualization
The severe class imbalance observed ($~7:1$ ham-to-spam ratio) mirrors actual telecommunication environments:

![Class Distribution](file:///C:/Users/Ramakrishna/.gemini/antigravity/brain/c0391ed2-e720-475b-93f8-c50de48f65e2/class_distribution.png)

*Figure 1: Class frequency distribution of the cleaned SMS collection showing 4,516 genuine messages versus 653 spam messages.*

---

## 3. Activity 3: Word-Based Probability Analysis & Bayes' Theorem Verification

To understand how lexical clues alter our confidence that a message is spam, we isolate five high-risk keywords: `free`, `prize`, `win`, `urgent`, and `offer`.

### 3.1 Mathematical Formulation
1. **Empirical Conditional Probability:**
   $$P(\text{Spam} \mid W) = \frac{\text{Count of spam messages containing } W}{\text{Total messages containing } W} = \frac{N(\text{Spam} \cap W)}{N(W)}$$

2. **Bayes’ Theorem Formulation:**
   $$P(\text{Spam} \mid W) = \frac{P(W \mid \text{Spam}) \cdot P(\text{Spam})}{P(W)}$$
   Where:
   * $P(W \mid \text{Spam}) = \frac{N(\text{Spam} \cap W)}{N_{\text{Spam}}}$ (Likelihood)
   * $P(\text{Spam}) = \frac{N_{\text{Spam}}}{N_{\text{Total}}}$ (Prior probability of spam)
   * $P(W) = \frac{N(W)}{N_{\text{Total}}}$ (Total probability of word occurrence)

### 3.2 Computed Word-Level Probabilities & Proof of Equivalence

All values were computed across the 5,169 unique messages using boundary-delimited regex matching (`\b<word>\b`):

| Suspicious Keyword ($W$) | $N(W)$ | $N(\text{Spam} \cap W)$ | $N(\text{Ham} \cap W)$ | $P(W)$ | $P(W \mid \text{Spam})$ | Empirical $P(\text{Spam} \mid W)$ | Bayes' Theorem $P(\text{Spam} \mid W)$ | Absolute Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **free** | 203 | 148 | 55 | 0.03927 | 0.22665 | **0.72906** | **0.72906** | $0.00 \times 10^0$ |
| **prize** | 73 | 73 | 0 | 0.01412 | 0.11179 | **1.00000** | **1.00000** | $0.00 \times 10^0$ |
| **win** | 55 | 49 | 6 | 0.01064 | 0.07504 | **0.89091** | **0.89091** | $0.00 \times 10^0$ |
| **urgent** | 62 | 57 | 5 | 0.01199 | 0.08729 | **0.91935** | **0.91935** | $0.00 \times 10^0$ |
| **offer** | 30 | 24 | 6 | 0.00580 | 0.03675 | **0.80000** | **0.80000** | $0.00 \times 10^0$ |

### 3.3 Detailed Step-by-Step Verification for `"free"`
* Total SMS containing `"free"`: $N(\text{free}) = 203$
* Spam messages containing `"free"`: $N(\text{Spam} \cap \text{free}) = 148$
* Total spam messages: $N_{\text{Spam}} = 653$
* Total messages: $N_{\text{Total}} = 5169$

1. **Empirical Method:**
   $$P(\text{Spam} \mid \text{free}) = \frac{148}{203} \approx 0.729064 \quad (72.91\%)$$

2. **Bayes’ Theorem Method:**
   $$P(\text{free} \mid \text{Spam}) = \frac{148}{653} \approx 0.226646$$
   $$P(\text{free}) = \frac{203}{5169} \approx 0.039273$$
   $$P(\text{Spam}) = \frac{653}{5169} \approx 0.126330$$

   $$P(\text{Spam} \mid \text{free}) = \frac{0.226646 \times 0.126330}{0.039273} = \frac{0.028632}{0.039273} \approx \mathbf{0.729064}$$

> [!NOTE]
> Notice that for the word `"prize"`, $N(\text{Ham} \cap \text{prize}) = 0$. Hence, $P(\text{Spam} \mid \text{prize}) = 1.00000$ (100%). Whenever `"prize"` appears in this dataset, the probability that the message is spam is guaranteed to be 1.

---

## 4. Activity 4: Random Variable Analysis

Let the classification of an incoming SMS message be modeled as a discrete indicator random variable $Y$:
$$Y = \begin{cases} 1, & \text{if message is Spam} \\ 0, & \text{if message is Ham} \end{cases}$$

Since $Y$ assumes only two values $\{0, 1\}$, it follows a **Bernoulli Distribution**:
$$Y \sim \text{Bernoulli}(p)$$
where parameter $p$ is the prior probability of observing a spam message:
$$p = P(Y = 1) = P(\text{Spam}) = 0.12633$$

### 4.1 Expectation $E(Y)$
By definition of mathematical expectation for a discrete random variable:
$$E(Y) = \sum_{y \in \{0, 1\}} y \cdot P(Y = y) = (0 \cdot P(Y=0)) + (1 \cdot P(Y=1))$$
$$E(Y) = p = P(\text{Spam}) = \mathbf{0.12633}$$

### 4.2 Variance $\text{Var}(Y)$
The variance quantifies the dispersion of the distribution:
$$\text{Var}(Y) = E[Y^2] - (E[Y])^2$$
Since $1^2 = 1$ and $0^2 = 0$, $E[Y^2] = E[Y] = p$. Thus:
$$\text{Var}(Y) = p - p^2 = p(1 - p)$$
$$\text{Var}(Y) = 0.12633 \times (1 - 0.12633) = 0.12633 \times 0.87367 = \mathbf{0.11037}$$

### 4.3 Standard Deviation $\sigma(Y)$
$$\sigma(Y) = \sqrt{\text{Var}(Y)} = \sqrt{0.11037} = \mathbf{0.33222}$$

**Statistical Interpretation:**  
The low variance ($\text{Var}(Y) \approx 0.11$) directly reflects the severe imbalance in the population. The vast majority of messages gravitate towards 0 (Ham), with spam occurring as relatively rare stochastic events.

---

## 5. Activity 5: Data Preparation

### 5.1 Stratified Train-Test Split (80% / 20%)
To ensure that both training and testing partitions retain the exact underlying spam-to-ham proportion ($p \approx 0.1263$), **stratified random sampling** was applied (`stratify=y`):

* **Total Dataset:** 5,169 samples
* **Training Set (80%):** 4,135 samples (Spam: 522 [12.62%], Ham: 3,613 [87.38%])
* **Testing Set (20%):** 1,034 samples (Spam: 131 [12.67%], Ham: 903 [87.33%])

### 5.2 Feature Extraction via CountVectorizer
Textual messages cannot be supplied directly to probabilistic algebraic equations. We convert the raw SMS messages into a numerical document-term frequency matrix using `CountVectorizer(lowercase=True, stop_words='english')`:
* **Tokenization:** Splits messages into words, stripping punctuation and converting text to lowercase.
* **Stop Word Removal:** Filters non-informative high-frequency English words (`the`, `is`, `at`, etc.).
* **Vocabulary Dimension:** Generated a dictionary of **7,414 distinct feature tokens**.
* **Matrix Output:** Sparse bag-of-words matrix $X_{\text{train}} \in \mathbb{R}^{4135 \times 7414}$ and $X_{\text{test}} \in \mathbb{R}^{1034 \times 7414}$.

---

## 6. Activity 6: Model Building (Multinomial Naive Bayes)

### 6.1 Theoretical Foundation of Multinomial Naive Bayes
For a message represented as a vector of word counts $\mathbf{x} = (x_1, x_2, \dots, x_d)$, Bayes' rule expresses the posterior probability for class $c \in \{\text{Spam}, \text{Ham}\}$ as:

$$P(c \mid \mathbf{x}) = \frac{P(c) \cdot P(\mathbf{x} \mid c)}{P(\mathbf{x})}$$

Under the **Naive Bayes assumption**, the presence of each token is assumed to be conditionally independent of any other token given the class $c$:

$$P(\mathbf{x} \mid c) = \prod_{i=1}^{d} P(w_i \mid c)^{x_i}$$

To prevent zero-probability errors for words unseen during training ($P(w_i \mid c) = 0$), Laplace smoothing ($\alpha = 1$) is incorporated:

$$\hat{P}(w_i \mid c) = \frac{N_{c, i} + \alpha}{N_c + \alpha \cdot |V|}$$

Where:
* $N_{c, i}$ is the total count of word $w_i$ in training documents of class $c$.
* $N_c$ is the total count of all words in class $c$.
* $|V| = 7,414$ is the size of the vocabulary.

### 6.2 Implementation
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=1.0)
model.fit(X_train_vec, y_train)
```

---

## 7. Activity 7: Model Evaluation & Performance Metrics

### 7.1 Quantitative Performance Metrics
The classifier was tested against 1,034 unseen messages from the test partition:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{119 + 898}{1034} = \mathbf{98.36\%}$$

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{119}{119 + 5} = \mathbf{95.97\%}$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{119}{119 + 12} = \mathbf{90.84\%}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.9597 \times 0.9084}{0.9597 + 0.9084} = \mathbf{0.9333}$$

### 7.2 Confusion Matrix Analysis

```
                      PREDICTED
                 Ham (0)     Spam (1)
ACTUAL  Ham (0)    898          5       (Total: 903)
        Spam (1)    12        119       (Total: 131)
```

| Evaluation Parameter | Value | Practical Significance |
| :--- | :---: | :--- |
| **True Negatives (TN)** | **898** | Legitimate SMS messages correctly delivered to inbox. |
| **False Positives (FP)** | **5** | Genuine messages erroneously flagged as spam (misclassification rate = 0.55%). |
| **False Negatives (FN)** | **12** | Spam messages that bypassed filtering into inbox (leakage rate = 9.16%). |
| **True Positives (TP)** | **119** | Malicious / promotional spam messages intercepted successfully. |

### 7.3 Detailed Classification Report
```
              precision    recall  f1-score   support

     Ham (0)     0.9868    0.9945    0.9906       903
    Spam (1)     0.9597    0.9084    0.9333       131

    accuracy                         0.9836      1034
   macro avg     0.9732    0.9514    0.9620      1034
weighted avg     0.9834    0.9836    0.9834      1034
```

### 7.4 Qualitative Error Diagnostics
1. **False Positives ($FP = 5$):**  
   Occurs when personal/genuine texts contain high-entropy incentive terms (e.g., *"Can you get me a free ticket?"* or *"Congratulations on winning the match"*). Because words like `free` and `win` strongly bias the posterior towards spam, legitimate colloquial use can cause false alarms.
2. **False Negatives ($FN = 12$):**  
   Occurs in sophisticated spam messages that obfuscate standard spam vocabulary or employ evasive phonetic spellings (e.g., `pr1ze`, `w!n`), as well as very short spam texts lacking strong bag-of-words tokens.

---

## 8. Activity 8: Real-Time Testing & Inference

An interactive inference module was constructed using `model.predict_proba()` to compute posterior distributions:

```python
def classify_sms(message_text):
    vec = vectorizer.transform([message_text])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    return "Spam" if pred == 1 else "Ham", probs[0], probs[1]
```

### Test Cases & Output Probabilities

#### Case 1 (Project Sheet Sample):
> **Input:** `"Congratulations! You have won a free prize. Claim now."`  
> * **Prediction:** **Spam**  
> * $P(\text{Spam} \mid \text{Message}) = \mathbf{1.0000} \quad (100.00\%)$  
> * $P(\text{Ham} \mid \text{Message}) = \mathbf{0.0000} \quad (0.00\%)$  
> * **Analysis:** Tokens `congratulations`, `won`, `free`, `prize`, and `claim` trigger compounding likelihood multiplications, yielding near certainty.

#### Case 2 (Everyday Conversational SMS):
> **Input:** `"Hey bro, are we still meeting in the library for statistics study?"`  
> * **Prediction:** **Ham**  
> * $P(\text{Spam} \mid \text{Message}) = \mathbf{0.0001} \quad (0.01\%)$  
> * $P(\text{Ham} \mid \text{Message}) = \mathbf{0.9999} \quad (99.99\%)$  
> * **Analysis:** Pure conversational vocabulary with zero spam indicators.

#### Case 3 (Urgent Financial Phishing):
> **Input:** `"URGENT! Your mobile account has won 1000 cash bonus. Call now to claim your offer."`  
> * **Prediction:** **Spam**  
> * $P(\text{Spam} \mid \text{Message}) = \mathbf{1.0000} \quad (100.00\%)$  
> * $P(\text{Ham} \mid \text{Message}) = \mathbf{0.0000} \quad (0.00\%)$  

---

## 9. Final Learning Outcomes & Conclusion

1. **Theoretical Cohesion:** Verified that empirical conditional probabilities extracted from observational corpora obey Bayes’ Theorem identically, showing how prior beliefs are rationally updated upon observing linguistic features.
2. **Random Variable Modeling:** Conceptualized binary spam detection under Bernoulli random variable models, deriving expected values ($E(Y) = 0.12633$) and variance ($\text{Var}(Y) = 0.11037$).
3. **Engineering Competence:** Successfully developed an end-to-end NLP and machine learning pipeline in Python using `scikit-learn`, `pandas`, `numpy`, and `matplotlib`.
4. **Reproducible Repository:** Packaged the code into an executable standalone script ([`sms_spam_detection.py`](file:///C:/Users/Ramakrishna/.gemini/antigravity/scratch/sms_spam_project/sms_spam_detection.py)), an interactive Jupyter Notebook ([`sms_spam_detection.ipynb`](file:///C:/Users/Ramakrishna/.gemini/antigravity/scratch/sms_spam_project/sms_spam_detection.ipynb)), and documentation for GitHub submission.

---

## 10. References & Deliverables Index

* **Dataset:** Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* ACM DocEng'11.
* **Scikit-learn Documentation:** *Naive Bayes Classifiers and MultinomialNB formulation.*
* **Project Deliverables Directory:** [`sms_spam_project/`](file:///C:/Users/Ramakrishna/.gemini/antigravity/scratch/sms_spam_project)
  * Script: [`sms_spam_detection.py`](file:///C:/Users/Ramakrishna/.gemini/antigravity/scratch/sms_spam_project/sms_spam_detection.py)
  * Jupyter Notebook: [`sms_spam_detection.ipynb`](file:///C:/Users/Ramakrishna/.gemini/antigravity/scratch/sms_spam_project/sms_spam_detection.ipynb)
  * Visualization: [`class_distribution.png`](file:///C:/Users/Ramakrishna/.gemini/antigravity/brain/c0391ed2-e720-475b-93f8-c50de48f65e2/class_distribution.png)
  * GitHub Setup Guide: [`README.md`](file:///C:/Users/Ramakrishna/.gemini/antigravity/scratch/sms_spam_project/README.md)
