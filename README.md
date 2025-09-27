# 📘 README for `consumer_comp.py`

## Overview
`composer_cpm.py` is a machine learning pipeline for **classifying consumer complaints** into specific categories (e.g., credit reporting, debt collection, consumer loans, mortgages).  

The script:
- Loads a **consumer complaint dataset (CSV)**  
- Maps product categories to predefined labels  
- Cleans and processes complaint text  
- Extracts features using **TF-IDF (with unigrams and bigrams)**  
- Trains and compares two models:  
  - **Logistic Regression**  
  - **Multinomial Naive Bayes**  
- Evaluates models using accuracy, F1-score, and confusion matrix  
- Saves the **best model** and an **evaluation report**  
- Provides a **prediction function** for new complaints  

---

## 🔧 Requirements
Install the dependencies before running the script:

```bash
pip install numpy pandas scikit-learn joblib matplotlib seaborn
```

---

## 📂 Input Data
- You must provide a CSV file of consumer complaints.  
- Update the script variable `CSV_PATH` with the path to your dataset:
  ```python
  CSV_PATH = "C:/Users/Asus/Desktop/complaints.csv"
  ```

### Expected Columns
The script automatically detects a suitable text column. It prefers:
- `Consumer complaint narrative`
- `consumer_complaint_narrative`
- `complaint_what_happened`
- `Complaint`  
If not found, it falls back to `Issue` or `Product`.

The `Product` column is used to map complaints to one of 4 categories:
- **0** → Credit reporting, repair, or other  
- **1** → Debt collection  
- **2** → Consumer Loan  
- **3** → Mortgage  

---

## ▶️ How to Run
Run the script directly:

```bash
python consumer_comp.py
```

It will:
1. Load and clean the data  
2. Train Logistic Regression and Naive Bayes models  
3. Compare their performance  
4. Save:  
   - `best_consumer_complaint_model.joblib` → the best model  
   - `evaluation_report.json` → evaluation metrics  

---

## 📊 Outputs
1. **Model performance** (printed to console):
   - Accuracy
   - F1-macro score
   - Classification report  
   - Confusion matrix

2. **Saved files**:
   - `best_consumer_complaint_model.joblib` (trained model)
   - `evaluation_report.json` (evaluation metrics)

---

## 🔮 Example Predictions
At the end of the run, the script tests a few sample complaints:

```text
- Credit reporting, repair, or other -> I pulled my credit report and found an error ...
- Debt collection -> I received a call from a collector about a debt ...
- Consumer Loan -> I applied for a personal loan but the lender says ...
- Mortgage -> The mortgage servicer has been charging incorrect escrow ...
```

---

## 📌 Using the Saved Model
You can load the model and classify new texts using the provided function:

```python
from composer_cpm import predict_texts

examples = [
    "My credit report shows incorrect information.",
    "A debt collector is harassing me."
]

predictions = predict_texts(examples)
print(predictions)
# ['Credit reporting, repair, or other', 'Debt collection']
```

---

## 🚀 Next Steps
- Expand `LABEL_MAP` to include more categories.  
- Experiment with different models (e.g., SVM, Random Forest).  
- Deploy the model as an API (Flask/FastAPI).  
