# 🚭 Tobacco Use and Mortality (2004–2015)

## 📘 Project Overview

Smoking is a major public health concern and a leading cause of preventable disease and death.  
This project analyzes the relationship between **tobacco use** and **mortality rates** in **England (2004–2015)** using **machine learning** and **data science** techniques.

The study explores smoking prevalence, hospital admissions, prescriptions, and socioeconomic factors to predict and understand the impact of tobacco on mortality.

The goal is to answer key analytical questions such as:

* How does tobacco use influence mortality rates over time?  
* Which demographic and socioeconomic factors contribute most to smoking-related deaths?  
* Can predictive models forecast mortality risk based on tobacco-related indicators?

---
## 🧾 Dataset Overview

The project utilizes **five datasets** covering various aspects of tobacco use and its impact on health and mortality in **England (2004–2015).**

---

### 🏥 Admissions Dataset  
**Records:** 2,079 | **Columns:** 7  
Contains hospital admission statistics for diseases linked to smoking, categorized by year, diagnosis type, ICD10 codes, sex, and number of admissions.  
➡️ Provides insights into smoking-related disease prevalence and hospitalization trends.

---

### ⚰️ Fatalities Dataset  
**Records:** 1,749 | **Columns:** 7  
Includes annual data on observed deaths caused by smoking-related diseases.  
➡️ Serves as the **primary indicator** of tobacco-attributable mortality trends.

---

### 💰 Metrics Dataset  
**Records:** 36 | **Columns:** 9  
Tracks economic indicators such as **tobacco price index**, **household expenditure**, **disposable income**, and **affordability of tobacco**.  
➡️ Reflects the **socioeconomic factors** influencing tobacco consumption and mortality.

---

### 💊 Prescriptions Dataset  
**Records:** 11 | **Columns:** 9  
Provides data on pharmacotherapy prescriptions for smoking cessation (e.g., **Nicotine Replacement Therapy**, **Bupropion**, **Varenicline**) and their associated net costs.  
➡️ Offers insights into **treatment efforts** and **public health initiatives** targeting smokers.

---

### 🚬 Smokers Dataset  
**Records:** 84 | **Columns:** 9  
Reports smoking prevalence by **age group**, **sex**, and **survey method** from **1974–2014**.  
➡️ Helps analyze **long-term demographic trends** and **behavioral patterns** in smoking across populations.

---
---

## 🎯 Objectives of the Analysis

1. **Understand Mortality Trends** – Analyze how tobacco use and related health metrics changed between 2004–2015.  
2. **Identify Key Risk Factors** – Determine the strongest predictors of smoking-related deaths.  
3. **Build Predictive Models** – Use supervised learning algorithms to forecast mortality based on tobacco use.  
4. **Provide Policy Insights** – Offer data-driven recommendations for reducing smoking-related health impacts.

---

## 🧹 Data Preprocessing

| Step               | Description                                                                 |
| ------------------ | --------------------------------------------------------------------------- |
| Data Cleaning      | Handled missing values, outliers, and inconsistencies                       |
| Data Integration   | Merged multiple data sources (mortality, smoking, demographics, etc.)       |
| Feature Engineering| Created derived features (e.g., duration of smoking, affordability index)   |
| Encoding           | Converted categorical variables to numeric values                           |
| Scaling            | Standardized continuous numerical variables                                 |
| Train-Test Split   | 80% training and 20% testing datasets                                       |

After preprocessing, a comprehensive and consistent dataset was obtained for modeling and analysis.

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights discovered during EDA:

* **Smoking-related hospital admissions:** Highest among **ages 35–59**.  
* **Gender trends:** Men show higher smoking prevalence and mortality.  
* **Economic impact:** Higher tobacco prices correlate with reduced smoking prevalence.  
* **Health behavior:** Increased prescription use coincides with decreasing mortality.  
* **Temporal trends:** Gradual decline in smoking rates post-2010 due to policy interventions.  

**Summary:**  
Smoking prevalence, hospital admissions, and affordability are the most influential factors in determining mortality rates.

<img width="1236" height="670" alt="image" src="https://github.com/user-attachments/assets/9190dc4b-e68a-46d9-8c2e-6ccfbe39fc4f" />
<img width="1237" height="678" alt="image" src="https://github.com/user-attachments/assets/72ebf37e-12f2-4d7a-991d-44fa23360069" />
<img width="1230" height="690" alt="image" src="https://github.com/user-attachments/assets/1e727a5d-a264-4af3-9718-fad18a35c15f" />

---

## 📈 Correlation Analysis

**Top Positive Correlations:**
* Smoking Prevalence ↔ Mortality  
* Hospital Admissions ↔ Mortality  

**Top Negative Correlations:**
* Tobacco Price Index ↔ Smoking Rate  
* Income Level ↔ Mortality  

**Interpretation:**  
As tobacco prices rise and affordability drops, smoking prevalence and related mortality decrease — reflecting effective public health interventions.
<img width="1188" height="1045" alt="image" src="https://github.com/user-attachments/assets/d8eb564e-8038-415d-89b8-240f3ac73d09" />
<img width="1963" height="1175" alt="image" src="https://github.com/user-attachments/assets/fad975e0-a1a6-4d40-bd03-5083191b462a" />

---

## 🤖 Model Development

### Algorithms Used

* Logistic Regression  
* Decision Tree Classifier  
* Random Forest Classifier  
* Gradient Boosting (XGBoost, LightGBM)  
* Support Vector Machine (SVM)  
* Neural Network (TensorFlow/Keras)

**Train-Test Split:** 80% training, 20% testing  
**Validation:** k-Fold Cross-Validation  
**Hyperparameter Tuning:** Grid Search / Random Search  

---

## 🧪 Model Evaluation

| Model                   | Accuracy | Comments                                      |
| ------------------------ | -------- | --------------------------------------------- |
| **Random Forest**        | **99%**  | Strong predictive accuracy, robust performance|
| **XGBoost**              | 98%      | Excellent accuracy, slightly overfitted       |
| **Gradient Boosting**    | 98%      | Strong but computationally heavier            |
| **Logistic Regression**  | 92%      | Good baseline model                           |
| **SVM**                  | 95%      | Stable but slower on large feature sets       |
| **Neural Network**       | 97%      | High accuracy, longer training time required  |

---

## 🏆 Best Model: Random Forest Classifier

**Test Accuracy:** 99%  
**Precision:** 0.99  
**Recall:** 0.99  
**F1-Score:** 0.99  
**ROC-AUC:** 1.0  

### Top Influential Features:
1. Smoking Prevalence  
2. Hospital Admissions  
3. Tobacco Price Index  
4. Income / Affordability Ratio  
5. Age Group  
6. Gender (Male/Female)  
7. Prescription Rates  
8. Annual Tobacco Expenditure  

📊 **Interpretation:**  
Higher smoking prevalence and hospital admissions directly increase mortality risk.  
Rising tobacco prices and health interventions (like prescriptions) correlate with lower death rates.
<img width="2088" height="1114" alt="image" src="https://github.com/user-attachments/assets/2e99c82f-2004-44dd-a594-6e005964d858" />
<img width="1499" height="1242" alt="image" src="https://github.com/user-attachments/assets/86989f27-60ef-45c3-91f1-c8e56974d44a" />

---

## 🧰 Tools & Technologies Used

| Tool / Library            | Purpose                                      |
| -------------------------- | -------------------------------------------- |
| 🐍 **Python**              | Core programming and modeling                |
| 📓 **Jupyter Notebook**    | Interactive data analysis                    |
| 💾 **VS Code**             | Development environment                      |
| 📊 **Pandas, NumPy**       | Data manipulation and processing             |
| 📈 **Matplotlib, Seaborn** | Visualization and EDA                        |
| 🤖 **Scikit-learn, XGBoost, LightGBM** | Machine learning algorithms         |
| 🔦 **SHAP, LIME**          | Model explainability                         |
| 🧮 **SQL**                 | Data extraction and querying                 |
| 🌐 **Flask / Streamlit**   | Model deployment (optional)                  |

---

## ⚙️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Tobacco-Use-and-Mortality
.git
cd Tobacco-Use-and-Mortality

```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook or Script

```bash
jupyter notebook tobacoo.ipynb
```

or

```bash
python app.py
```

---

## 📚 Results Summary

* **Best Model:** Random Forest (99% Accuracy)  
* **Key Predictors:** Smoking prevalence, hospital admissions, affordability index  

**Findings:**
- Tobacco use is strongly correlated with mortality.  
- Increased tobacco prices lead to reduced smoking and deaths.  
- Healthcare access and cessation prescriptions show a positive public health impact.  
<img width="2900" height="1587" alt="image" src="https://github.com/user-attachments/assets/8cdc2a29-69be-4f07-9abb-65703c596e67" />

---

## 🔮 Conclusion & Future Scope

The analysis concludes that **tobacco use is a critical predictor of mortality.**  
Machine learning models can effectively forecast risk and support **data-driven public health policy.**

**Future Enhancements:**
- Implement **SHAP** for explainability dashboards.  
- Develop a real-time **Streamlit dashboard** for mortality predictions.  
- Include **time-series forecasting** for future trend prediction.  
- Expand dataset to include **global tobacco and mortality data** for comparison.


---

## 👨‍💻 Author

**M.Umesh Chandra**
📧 *[[metlaumeshchandra2005.email@example.com](mailto:metlaumeshchandra2005.email@example.com)]*
💼 *Data Analyst & Machine Learning Enthusiast*

---

⭐ **If you like this project, don’t forget to star the repo!**
