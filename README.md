# Big Data Classification of Hospital Readmissions

Predicting 30-day hospital readmission with **Apache Spark and machine learning**, using 30,000 records and 12 variables. This academic project explores clinical and demographic features, builds classification pipelines, and compares Logistic Regression, Decision Tree, and Random Forest models.

## Project overview

The workflow includes data quality checks, exploratory analysis, feature engineering, model training, and evaluation with PySpark MLlib.

- Check missing values, duplicate patient IDs, and numerical outliers using the IQR method.
- Visualize the target distribution, numerical correlations, and readmission rates by chronic condition and discharge destination.
- Extract systolic blood pressure and encode categorical features with `StringIndexer` and `OneHotEncoder`.
- Split records into 70% training and 30% testing data with seed `123456`.
- Tune each classifier with a four-combination parameter grid and three-fold cross-validation, selecting by weighted F1.
- Compare accuracy, weighted precision, weighted recall, weighted F1, and ROC AUC.

## Repository contents

| File | Description |
| --- | --- |
| [Final_Phase_Hospital_Readmission.ipynb](Final_Phase_Hospital_Readmission.ipynb) | Complete analysis and model training notebook, including original outputs |
| [hospital_readmissions_30k.csv](hospital_readmissions_30k.csv) | Input dataset: 30,000 rows and 12 columns |
| [Hospital_Readmission_Report.pdf](docs/Hospital_Readmission_Report.pdf) | Academic report describing methods, findings, and references |
| [requirements.txt](requirements.txt) | Python dependencies for running the notebook |

The notebook and CSV stay together because the notebook loads the dataset using a relative filename.

## Dataset

The report cites the [Hospital Readmission Prediction (Synthetic Dataset)](https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset) by S. Siddharth on Kaggle (2025). The dataset is described as synthetic in that citation; the uploaded CSV's provenance has not been independently verified.

Features include age, gender, blood pressure, cholesterol, BMI, diabetes, hypertension, medication count, length of stay, and discharge destination. `patient_id` is an identifier and is excluded from model features. The target, `readmitted_30_days`, is encoded as `Yes = 1` and `No = 0`.

## Run locally

Use **Python 3.11 or 3.12** and **Java 17**, with Java available on `PATH`. The dependency versions below are a suggested compatible setup; the original notebook does not provide an environment lockfile.

```bash
git clone https://github.com/TumadhirX/Big-Data-Classification-of-Hospital-Readmissions.git
cd Big-Data-Classification-of-Hospital-Readmissions
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab Final_Phase_Hospital_Readmission.ipynb
```

On Windows, activate with `.venv\Scripts\activate` instead. Run the notebook cells in order from the repository directory. Spark runs locally using `local[*]`; cross-validation may take several minutes and uses the available CPU cores.

## Reported results

These values come from the supplied notebook outputs and report, rather than a new training run.

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8764 | 0.7681 | 0.8764 | 0.8187 | 0.5884 |
| Decision Tree | 0.8751 | 0.7680 | 0.8751 | 0.8180 | 0.4349 |
| Random Forest | 0.8764 | 0.7681 | 0.8764 | 0.8187 | 0.5831 |

The target is imbalanced. Weighted scores can conceal weak performance on readmitted patients, and the reported AUC values show limited discrimination. The notebook does not report positive-class recall or a confusion matrix. Cross-validation folds and classifier randomness are not explicitly seeded, so reruns may differ. This is an academic analysis, not a validated clinical prediction system.

## Contributors

The accompanying report credits Tala Alroeis, Tumadhir Alyahya, Layan Almarshud, Leena Alsaif, and Nora Alotaibi, from Data Science and Analytics at Princess Nourah University, Riyadh, Saudi Arabia.

## Usage rights

No project license has been specified. Refer to the original dataset source for its usage terms.
