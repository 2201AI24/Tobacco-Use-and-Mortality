import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------
# Load model and preprocessor
# -----------------------
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

le_dict = preprocessor['label_encoders']
scaler = preprocessor['scaler']

# -----------------------
# Model columns
# -----------------------
model_cols = [
    'icd10_code', 'icd10_diagnosis_admissions', 'diagnosis_type_admissions',
    'metric_admissions', 'sex', 'value_admissions', 'year',
    'smoking_related_admissions', 'icd10_diagnosis_fatalities',
    'diagnosis_type_fatalities', 'metric_fatalities', 'tobacco_price_index',
    'retail_prices_index', 'tobacco_price_index_relative_to_retail_price_index',
    'real_households_disposable_income', 'affordability_of_tobacco_index',
    'household_expenditure_on_tobacco', 'household_expenditure_total',
    'expenditure_on_tobacco_as_a_percentage_of_expenditure',
    'all_pharmacotherapy_prescriptions',
    'nicotine_replacement_therapy_nrt_prescriptions',
    'bupropion_zyban_prescriptions', 'varenicline_champix_prescriptions',
    'net_ingredient_cost_of_all_pharmacotherapies',
    'net_ingredient_cost_of_nicotine_replacement_therapies_nrt',
    'net_ingredient_cost_of_bupropion_zyban',
    'net_ingredient_cost_of_varenicline_champix', 'method', '16_and_over',
    '16_24', '25_34', '35_49', '50_59', '60_and_over', 'smoker_16_24',
    'smoker_25_34', 'smoker_35_49', 'smoker_50_59', 'smoker_60_over',
    'tobacco_affordability', 'tobacco_exp_ratio'
]

numeric_cols = [c for c in model_cols if c not in ['sex', 'method']]
categorical_cols = ['sex', 'method']

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Mortality Prediction App", layout="wide")
st.title("🩺 Tobacco Data – Mortality Prediction")
st.markdown("### Enter the input data to predict the mortality class")

# -----------------------
# Numeric inputs (10 per row)
# -----------------------
user_input = {}

num_cols_per_row = 10
num_groups = [numeric_cols[i:i + num_cols_per_row] for i in range(0, len(numeric_cols), num_cols_per_row)]

for group in num_groups:
    cols = st.columns(len(group))
    for i, col in enumerate(group):
        user_input[col] = cols[i].number_input(col, value=0.0, format="%.4f")

# -----------------------
# Categorical inputs
# -----------------------
st.markdown("### Categorical Inputs")
cat_cols = st.columns(len(categorical_cols))

user_input['sex'] = cat_cols[0].selectbox("sex", ["Male", "Female", "Unknown"])
user_input['method'] = cat_cols[1].selectbox("method", ["Survey", "Other", "Unknown"])

# -----------------------
# Predict Button
# -----------------------
st.markdown("---")
center_col = st.columns(3)[1]
with center_col:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

if predict_btn:
    input_df = pd.DataFrame([user_input])

    # Encode categorical variables safely
    for col in categorical_cols:
        le = le_dict.get(col, None)
        if le is not None:
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            input_df[col] = le.transform(input_df[col])
        else:
            input_df[col] = input_df[col].astype('category').cat.codes

    # Ensure alignment
    for col in model_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_cols]

    # Scale features
    input_df_scaled = scaler.transform(input_df) if scaler is not None else input_df

    # Make prediction
    pred_class = model.predict(input_df_scaled)[0]
    pred_prob = model.predict_proba(input_df_scaled)[:, 1][0]

    st.success(f"✅ Predicted Mortality Class: **{pred_class}**")
    st.info(f"📊 Probability of Mortality Class 1: **{pred_prob:.4f}**")
