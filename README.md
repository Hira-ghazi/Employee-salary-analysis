# Employee Salary Analysis Dashboard

A premium Streamlit dashboard built from the actual Employee Salaries records and the notebook's cleaning workflow. The app uses the generated cleaned CSV for every dashboard calculation, retains the supplied Linear Regression estimator, and does not retrain it.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The included `data/cleaned_employee_salary.csv` is already generated. Reproduce it from the bundled source file with:

```bash
python clean_data.py
```

`data/Employee_Salaries_raw.csv` contains the 10,291 original dataset rows. The uploaded workspace CSV had task text before its dataset header; only the actual CSV body is included here.

## Cleaning workflow

`clean_data.py` matches the notebook: 80/20 split with random state 42; numeric median and categorical mode imputation fitted on training rows; exact duplicate removal on the imputed training frame; IQR clipping of overtime and longevity pay using training bounds; then one-hot encoding and train-column alignment for the model preprocessing artifact. The cleaned analysis CSV recombines retained train rows and the test rows, preserving salary targets and readable source features. No source rows are removed beyond the notebook's 446 exact training duplicates.

The dashboard uses this cleaned CSV for KPIs, charts, filtering, observations, the complete data table, and CSV download. The data contains Grade but no job title, years of experience, or education. Grade is displayed only as a job-level classification proxy; absent fields are not fabricated.

## Existing model

`models/linear_regression_model.joblib` is the supplied trained model. `models/inference_preprocessing.json` stores the notebook-matched one-hot feature order, training scaler mean/scale, and IQR bounds. Prediction applies those saved transformations and calls the existing model. It never retrains the estimator.

The notebook's held-out evaluation values are MAE $7,617.37 and R? 0.867568. The actual/predicted sample table contains the ten rows saved in the notebook output; those rows are not represented as the full test set.

Gender values in the source are `F` and `M`; the filter displays Female and Male and maps back to those exact stored categories. Department codes are filtered against the `Department` column. The cleaned data has 79 records for CCL (County Council) and Female.
