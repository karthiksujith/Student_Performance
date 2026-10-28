# Student Performance Prediction

## Project Overview

The **Student Performance Prediction** project uses Machine Learning to predict whether a student is likely to **Pass** or be **At Risk** based on their study habits, academic history, family support, and other relevant factors.

The project uses a **Logistic Regression** model and a **Streamlit** web application to make predictions through a simple, interactive user interface.

## 🎯 Objectives

- Analyze student data and understand final grade distributions.
- Identify factors that may be associated with student performance.
- Build a machine learning classification model.
- Predict whether a student is likely to pass or be at risk.
- Evaluate model performance using classification metrics.
- Deploy an interactive web application using Streamlit.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine learning and model evaluation
- **Joblib** — Saving and loading the trained model
- **Streamlit** — Interactive web application

## 📂 Project Structure

```text
student-performance-prediction/
│
├── data/
│   └── student.csv
│
├── models/
│   └── student_performance_model.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

**File descriptions:**

- `data/student.csv` — Dataset containing student information and final grades.
- `models/student_performance_model.pkl` — Saved machine learning pipeline.
- `train_model.py` — Script for data exploration, preprocessing, model training, and evaluation.
- `app.py` — Streamlit application for predicting student performance.
- `requirements.txt` — Project dependencies.
- `README.md` — Project documentation.

## 📊 Dataset

The project uses a student performance dataset stored in `data/student.csv`, with fields separated by semicolons (`;`).

The target variable is `G3`, which represents the student's final grade.

### Target Classification

| Final Grade (`G3`) | Prediction |
|---|---|
| 10 or above | Pass |
| Below 10 | At Risk |

The target variable is created from the final grade during model training.

### Features Used

The model uses the following 10 input features:

| Feature | Description |
|---|---|
| `studytime` | Weekly study time |
| `failures` | Number of past class failures |
| `absences` | Number of school absences |
| `schoolsup` | Extra educational support from school |
| `famsup` | Extra educational support from family |
| `paid` | Paid extra classes |
| `higher` | Desire to pursue higher education |
| `Medu` | Mother's education level |
| `Fedu` | Father's education level |
| `traveltime` | Travel time to school |

## ⚙️ Machine Learning Workflow

1. **Data loading:** Load the CSV dataset using Pandas.
2. **Exploratory data analysis:** Examine the dataset's shape, columns, missing values, duplicate rows, descriptive statistics, and final grade distribution.
3. **Target creation:** Convert final grades into two classes: `Pass` and `At Risk`.
4. **Feature selection:** Select 10 relevant student-related features.
5. **Train-test split:** Split the dataset into 80% training data and 20% testing data using stratified sampling.
6. **Preprocessing:**
   - Apply `StandardScaler` to numerical features.
   - Apply `OneHotEncoder` to categorical features.
7. **Model training:** Train a Logistic Regression classifier with `class_weight='balanced'`.
8. **Model evaluation:** Evaluate predictions using accuracy, recall, F1-score, a classification report, and a confusion matrix.
9. **Model saving:** Save the complete preprocessing and classification pipeline using Joblib.
10. **Web deployment:** Load the saved model into Streamlit to generate predictions from user inputs.

## 🤖 Model Details

**Algorithm:** Logistic Regression

The model is implemented using a Scikit-learn pipeline that combines preprocessing and classification.

Key configuration:

- `max_iter=1000` — Allows additional iterations for model convergence.
- `class_weight='balanced'` — Adjusts class weights according to their frequencies in the training data.
- `StandardScaler` — Standardizes numerical features.
- `OneHotEncoder` — Encodes categorical features while ignoring unseen categories during prediction.

### Evaluation Metrics

The model is evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Recall:** Proportion of actual `At Risk` students correctly identified.
- **F1-score:** Harmonic mean of precision and recall for the `At Risk` class.
- **Classification report:** Provides precision, recall, F1-score, and support for each class.
- **Confusion matrix:** Shows correct and incorrect predictions for both classes.

Actual evaluation scores depend on the dataset and training run.

## 💻 Setup


### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

Create a `requirements.txt` file containing:

```text
pandas
matplotlib
scikit-learn
streamlit
joblib
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Project

### Step 1: Train the model

Ensure the dataset exists at `data/student.csv` and the `models/` directory exists.

Run:

```bash
python train_model.py
```

This script explores the data, trains the model, displays evaluation metrics, and saves the trained pipeline to:

```text
models/student_performance_model.pkl
```

### Step 2: Launch the Streamlit application

Run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL.

Open the URL in your browser.

### Step 3: Make a prediction

1. Enter the student's weekly study time.
2. Specify past class failures and school absences.
3. Select school support, family support, paid classes, and higher education preferences.
4. Enter the parents' education levels.
5. Select the student's travel time to school.
6. Click **Predict Performance**.

The application displays the predicted class (`PASS` or `AT RISK`) and the estimated probability of being at risk.

## 🌐 Streamlit Application Features

- Simple, interactive user interface.
- Input controls for all 10 model features.
- Real-time predictions using the saved model.
- Pass and at-risk results displayed with visual indicators.
- Estimated probability of being at risk.

## ⚠️ Limitations

- Logistic Regression may not capture complex relationships between student characteristics and academic outcomes.
- The model's predictions are estimates and may not accurately represent every student's circumstances.
- An at-risk prediction should not be treated as a definitive judgment about a student's ability or future.
- The probability displayed by the application is a model estimate and is not necessarily a calibrated real-world probability.

## 🔮 Future Improvements

- Compare Logistic Regression with Random Forest, Decision Tree, and other classification algorithms.
- Add visualizations and more detailed evaluation metrics to the Streamlit dashboard.
- Improve probability calibration and assess performance on independent datasets.
- Perform feature importance analysis and model interpretation.
- Add automated data validation and improved error handling.
- Refactor the training and prediction scripts into reusable functions to improve code readability, maintainability, and testing.
- Deploy the application to a cloud hosting platform.
