# BAMAT207: Applications of Probability and Statistics - Project
## Probabilistic SMS Spam Detection System

### Project Details
- **Course Code & Title:** BAMAT207 Applications of Probability and Statistics
- **School:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT)
- **Topic:** Probability, Random Variables, Bayes' Theorem, and Machine Learning
- **Dataset:** UCI SMS Spam Collection Dataset (5,574 messages)

---

### Repository Structure
```
sms_spam_project/
├── data/
│   └── SMSSpamCollection          # Cleaned dataset from UCI ML repository
├── class_distribution.png         # Matplotlib class distribution visualization
├── sms_spam_detection.py          # Complete executable Python script
├── sms_spam_detection.ipynb       # Jupyter Notebook for Google Colab / VS Code
├── README.md                      # Project documentation and setup guide
└── Project_Report.pdf / .md       # Complete academic project report
```

---

### How to Run

1. **Clone or Download the Repository:**
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd sms_spam_project
   ```

2. **Install Required Dependencies:**
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

3. **Run the Python Pipeline:**
   ```bash
   python sms_spam_detection.py
   ```

4. **Run via Jupyter / Google Colab:**
   - Open `sms_spam_detection.ipynb` in VS Code or upload it to [Google Colaboratory](https://colab.research.google.com).
   - Execute all cells sequentially.

---

### Submission Checklist
- [x] Python Script (`sms_spam_detection.py`) & Jupyter Notebook (`sms_spam_detection.ipynb`)
- [x] Cleaned dataset summary (5,169 unique SMS messages, 0 missing values)
- [x] Class distribution chart (`class_distribution.png`)
- [x] Probability calculations for 5 suspicious words (`free`, `prize`, `win`, `urgent`, `offer`) verified with Bayes' Theorem
- [x] Bernoulli Random Variable Analysis ($E(Y)$, $Var(Y)$)
- [x] Stratified 80/20 train-test split & CountVectorizer
- [x] Multinomial Naive Bayes model training
- [x] Comprehensive evaluation: Confusion Matrix, Accuracy (98.36%), Precision (95.97%), Recall (90.84%), F1-Score (0.9333)
- [x] Real-time inference testing on custom SMS messages
- [x] GitHub Repository updated and prepared for submission link
