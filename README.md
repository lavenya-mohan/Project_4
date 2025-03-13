![image](https://github.com/user-attachments/assets/f249b9c0-599d-4908-81ac-001afbca5805)

# Project: Predicting Future Loss Claims for Car Insurance

## By Group 8 Team Members:
- **Lavenya Mohanasundaram**
- **Mehmet Akif Cingöz**
- **Nicholas Voltin**

## Objective
The primary objective of this project is to predict whether an insurance applicant is likely to file a claim in the future based on their loss history and application data. This predictive analysis helps insurance companies assess risk and tailor their pricing strategies more effectively.

## Tableau Dashboard
For a detailed visual summary of our project's findings, view our interactive Tableau dashboard:
[View Tableau Dashboard](https://public.tableau.com/app/profile/lavenya2766/viz/Project4Dash_17417641495290/ClaimPredictorSummary?publish=yes)

## Data Utilization
The project leverages historical data involving loss history and demographic information. While the exact datasets used were not specified, they typically include factors such as prior claims, driving records, and demographic data.

## Methodology
- **Data Preparation**: Data was prepared by selecting relevant features that could influence the likelihood of future claims, including cleaning data, handling missing values, and transforming variables as needed.
- **Feature Engineering**: Key features were engineered and selected based on their predictive power and relevance to insurance claims, including variables like the number of major and minor violations, number of vehicles, coverage limits, credit scores, and historical loss amounts.
- **Predictive Modeling**: Various algorithms were evaluated, with a decision tree model being the final choice due to its ability to handle non-linear relationships and provide interpretable results.

# Machine Learning Model Performance Summary

The following table summarizes the performance of different algorithms on our feature set:

| Algorithm          | Jaccard | F1-score | Logloss |
|--------------------|---------|----------|---------|
| KNN                | 0.0     | 0.93507  | NA      |
| Decision Tree      | 0.998849| 0.99995  | NA      |
| SVM                | 0.0     | 0.93507  | NA      |
| Logistic Regression| 0.204519| 0.878617 | 0.677313|

### Observations:
- **Decision Tree** demonstrates


## Final Model Recommendation
The project recommends deploying a decision tree model, which uses features such as past claims, type of coverage, and demographic characteristics to predict future claims. This model helps in identifying potential high-risk customers, significantly impacting the insurance company's risk management and pricing policy.


## Technical Implementation
- A **Flask web application** was developed to demonstrate the model's utility. Users can input relevant data through a web form, which is then processed by the model to predict the likelihood of a claim.
- The backend setup involves loading a trained Decision Tree model, handling form data using pandas DataFrame for correct feature representation, and dynamically rendering results using Flask and HTML.

## Impact
The predictive model allows insurance companies to better understand and predict customer behavior, leading to optimized pricing strategies and improved profitability by adequately pricing the risk associated with each insured individual or household.

This project showcases the practical application of machine learning in the insurance industry, emphasizing risk assessment and predictive analytics to support decision-making processes.
