Loan Approval Prediction
A machine learning project for predicting whether a loan application will be approved or rejected using classification models, with particular attention to class imbalance, reliable evaluation, and model interpretability.
Project type: Supervised Machine Learning · Binary Classification · Exploratory Data Analysis
Main notebook: loan_approval_prediction.ipynb

Table of Contents
- Overview
- Objectives
- Dataset
- Workflow
- Models and Imbalance Handling
- Evaluation Metrics
- Tech Stack
- Getting Started
- Project Structure
- Key Considerations
- Future Improvements
Overview
Loan approval decisions depend on multiple applicant characteristics. This project explores how machine learning can learn patterns from historical loan applications and classify an application as:
- 1 — Approved
- 0 — Rejected
The notebook builds and compares Logistic Regression and Decision Tree models. It also investigates class imbalance, tunes hyperparameters using cross-validation, evaluates performance on a held-out test set, and examines which features influence model predictions.

Important: This is an educational machine learning project, not a production lending system. Its predictions should not be used to make real credit decisions without rigorous validation, fairness assessment, explainability, security, and regulatory review.
Objectives
- Explore the dataset and inspect its distributions, correlations, and missing values.
- Prepare numerical and categorical features in a consistent preprocessing pipeline.
- Prevent data leakage by fitting preprocessing and resampling steps only on training data/folds.
- Compare baseline models with class-weighted and SMOTE-based approaches.
- Tune model hyperparameters with stratified cross-validation.
- Evaluate both classes rather than relying on accuracy alone.
- Visualize Precision–Recall curves and explore classification-threshold tuning.
- Inspect model coefficients, tree feature importances, and decision-tree rules.
Dataset
The notebook is configured to load loan_approval_dataset.csv from the current working directory.

The notebook references the Kaggle dataset:

Loan Approval Prediction Dataset — Kaggle

Download the CSV and place it beside the notebook, or update DATA_PATH in the notebook to point to its location.

The notebook attempts to identify a target column named loan_status or Loan_Status (case-insensitive). It maps common positive labels such as approved, y, yes, and 1 to 1; other labels become 0. ID columns named loan_id or id are dropped.

The notebook also notes compatibility with another commonly used Kaggle loan dataset whose target is Loan_Status with Y/N labels and missing values. Column names and the target labels should still be checked when using a different dataset.
Workflow
1. Load and clean data — read the CSV, trim whitespace from column names and text values, and inspect the dataset.
2. Define the target — convert loan status to a binary target and remove ID columns.
3. Exploratory Data Analysis (EDA) — inspect missing values, numerical distributions, correlations, and a credit-score-related feature when available.
4. Train/test split — use an 80/20 stratified split with random_state=42.
5. Preprocessing — median-impute and standardize numerical features; most-frequent-impute and one-hot encode categorical features.
6. Baseline training — train a Logistic Regression model without imbalance handling.
7. Imbalance experiments — compare class weights and SMOTE for Logistic Regression and Decision Tree models.
8. Hyperparameter tuning — use stratified 5-fold cross-validation and macro F1 scoring.
9. Model comparison — compare models on the held-out test set and with cross-validation.
10. Threshold analysis — plot Precision–Recall curves and explore an alternative probability threshold.
11. Interpretation — examine Logistic Regression coefficients, Decision Tree feature importances, and a visualization of the tree.
Models and Imbalance Handling
Logistic Regression
A linear classification baseline. The notebook also evaluates a class-weighted version and tunes the regularization parameter C.
Decision Tree
A rule-based model capable of learning non-linear splits. Baseline experiments use a maximum depth of 5; a separate search tunes tree depth, minimum leaf size, and split criterion.
Class weights
class_weight="balanced" adjusts the class penalty based on the observed class frequencies. This can help the model pay more attention to a minority class.
SMOTE
Synthetic Minority Over-sampling Technique creates synthetic minority-class examples. In the notebook, SMOTE is placed in an imbalanced-learn pipeline so resampling happens within training folds rather than on the held-out test data.

Implementation note: SMOTE is applied after one-hot encoding in this notebook. That can create fractional values in encoded categorical features; approaches such as SMOTENC may be more appropriate for mixed numerical/categorical data and are worth evaluating separately.
Evaluation Metrics
The notebook reports several complementary metrics:
Metric	What it measures
Precision	Of the applications predicted as a class, how many actually belong to that class?
Recall	Of the applications that actually belong to a class, how many did the model identify?
F1-score	Harmonic mean of precision and recall.
Macro F1	F1 calculated independently for each class and then averaged equally.
ROC-AUC	How well predicted scores rank positive examples above negative examples across thresholds.
PR-AUC / Average Precision	Summarizes the precision–recall trade-off across thresholds.
Confusion matrix	Shows correct and incorrect predictions for each class.


Accuracy is not the sole evaluation criterion because class imbalance can make it misleading. The notebook also reports per-class precision, recall, and F1, with explicit labels for rejected (0) and approved (1) applications.

The threshold-tuning section chooses a threshold that maximizes approved-class F1 on the test set for exploration. For an unbiased final estimate, select the threshold using a validation set or cross-validation, then evaluate it once on untouched test data.
Tech Stack
- Python
- Jupyter Notebook
- NumPy and Pandas — data manipulation
- Matplotlib and Seaborn — visualizations
- Scikit-learn — preprocessing, model training, tuning, and evaluation
- imbalanced-learn — SMOTE and resampling pipelines
Getting Started
1. Clone or download the project
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
If you downloaded the files directly, open a terminal in the project directory instead.
2. Create a virtual environment (recommended)
python -m venv .venv
Activate it:

Windows (PowerShell):
.venv\Scripts\Activate.ps1
macOS / Linux:
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
4. Add the dataset
Place loan_approval_dataset.csv in the project directory. If your file is stored elsewhere, update the DATA_PATH variable near the beginning of the notebook.
5. Run the notebook
jupyter notebook loan_approval_prediction.ipynb
Run the cells from top to bottom. The notebook prints model reports and displays plots throughout the workflow.
Project Structure
.
├── loan_approval_prediction.ipynb   # Data analysis, modeling, evaluation
├── loan_approval_dataset.csv        # Dataset (download separately)
└── README.md                        # Project documentation
Key Considerations
- Data leakage: preprocessing is encapsulated in a ColumnTransformer pipeline. SMOTE is placed inside an imbalanced-learn pipeline for the SMOTE experiments.
- Reproducibility: the notebook uses random_state=42 for its split and randomized model steps.
- No fabricated results: model performance depends on the dataset and execution environment. Run the notebook and use its actual output when reporting final metrics.
- Threshold choice is contextual: false approvals and false rejections can have very different consequences. A threshold should reflect the real decision costs and be selected without using the final test set.
- Fairness and privacy: lending decisions are high-impact. Before any real-world use, evaluate disparate performance across relevant groups, comply with applicable laws, protect applicant data, and provide appropriate human oversight and explanations.
Future Improvements
- Evaluate additional models such as Random Forest and gradient boosting.
- Compare SMOTE with SMOTENC for mixed feature types.
- Calibrate predicted probabilities and assess calibration quality.
- Select decision thresholds using validation data and explicit business costs.
- Add fairness and subgroup-performance analysis where legally and ethically appropriate.
- Package the best pipeline into a reusable prediction function or API.
- Add automated tests, dependency version pinning, and a reproducible experiment log.
