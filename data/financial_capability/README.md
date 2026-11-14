# Component 04 – Explainable Financial Well-being Assessment System for Bank Loan Decision Support

## 1. Component Identity & Dataset Source
**Component 04** is an independent Financial Well-being Assessment System designed to assist a **Bank Officer** during a loan application review. It evaluates a loan applicant's financial well-being (`Low`, `Medium`, `High`), explains the model prediction using **SHAP**, profiles four financial capability dimensions (**Saving, Budgeting, Basic Financial Knowledge, and Financial Goal Planning**), and generates a supplementary assessment report.

- **Dataset**: CFPB National Financial Well-Being Survey (NFWBS) 2016 public-use file (`6,394` raw records; `6,389` valid target records; `31` approved predictor features).
- **Source Page**: https://www.consumerfinance.gov/data-research/financial-well-being-survey-data/
- **Public-Use CSV**: https://www.consumerfinance.gov/documents/5614/NFWBS_PUF_2016_data.csv

Save the CSV as `data/financial_capability/raw/NFWBS_PUF_2016_data.csv` (raw file is gitignored).

To re-run the training and evaluation pipeline from the repository root:

```powershell
python -m backend.src.financial_capability.fc_run_pipeline
```

---

## 2. Diagram A – Independent Component 04 System Workflow

```mermaid
flowchart LR
    S1["1. CFPB Financial Well-Being Survey (NFWBS 2016)<br/>• 6,394 raw / 6,389 valid rows<br/>• Saving, Budgeting, Knowledge, Goals & Context"]
    S2["2. Data Preparation<br/>• Clean data & special codes (-1, 98, 99)<br/>• Select 31 approved input features<br/>• Create FWBscore target (Low ≤49, Med 50–57, High ≥58)<br/>• Prevent target leakage"]
    S3["3. Model Training & Evaluation<br/>• Random Forest & XGBoost (5-fold CV)<br/>• Selected model: XGBoost<br/>• Compare Accuracy, Precision, Recall, F1, ROC-AUC"]
    S4["4. Explainability (SHAP)<br/>• Global feature importance<br/>• Individual applicant explanation<br/>• Saving habit, budget control, knowledge, goal planning"]
    S5["5. Financial Profile & Bank Officer Guidance<br/>• 4 Dimensions: Saving, Budgeting, Knowledge, Goal Planning<br/>• Strengths & Gap Areas<br/>• Bank Officer Practical Guidance & Follow-up Questions"]
    S6["6. Financial Well-being Assessment Report<br/>• Predicted level (High / Medium / Low)<br/>• Key SHAP factors & resilience indicators<br/>• Downloadable PDF & TXT report"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6
```

---

## 3. Diagram B – How Component 04 Supports the Bank Loan Scenario

```mermaid
flowchart LR
    Customer["Customer (Loan Applicant)<br/>Submits Loan Application"]
    BankCheck["Bank's Normal Loan Check (Existing Process)<br/>• Identity & Income Verification<br/>• Credit History / Score<br/>• Repayment Capacity (Income/Expenses, DTI)"]
    Comp04["Component 04 (Independent Financial Well-Being Assessment)<br/>• Assesses financial well-being (High / Medium / Low)<br/>• Explains key factors via SHAP<br/>• Identifies capability strengths & gaps<br/>• Generates supplementary report & officer guidance"]
    Officer["Bank Officer (Decision Support)<br/>Reviews normal loan check and Component 04 report separately"]
    Decision["Final Loan Decision by Bank Officer<br/>✔ Approve<br/>✔ Approve with Conditions<br/>✔ or Reject"]

    Customer --> BankCheck
    Customer --> Comp04
    BankCheck --> Officer
    Comp04 --> Officer
    Officer --> Decision
```

> **Important Governance Notice:** Component 04 does not automatically approve or reject the loan. It provides an additional financial well-being report to help the bank officer make a more informed decision under existing bank policies. Component 04 is completely independent of Components 01, 02, and 03.


