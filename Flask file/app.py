from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd  # Import pandas for proper DataFrame handling

app = Flask(__name__)

# Load the trained Decision Tree model
model_filename = 'ml_model.pkl'  # Ensure this model file exists in your directory
model = pickle.load(open(model_filename, 'rb'))

# Feature descriptions for rendering the form fields dynamically
feature_descriptions = {
    'cnt_majr_viol': 'Total number of major violations in household',
    'cnt_minr_viol': 'Total number of minor violations in household',
    'cnt_lic_susp': 'Number of drivers with a suspended license in household',
    'prior_bi': 'Bodily Injury Coverage Individual Limit with insurer',
    'veh_lease_cnt': 'Number of vehicles leased in household',
    'veh_own_cnt': 'Number of vehicles owned in household',
    'veh_w_coll_cnt': 'Number of vehicles with Collision Coverage',
    'veh_w_comp_cnt': 'Number of vehicles with Comprehensive Coverage',
    'veh_w_ers_cnt': 'Number of vehicles with Emergency Road Service Coverage',
    'curnt_bi_low': 'Bodily Injury Coverage Individual Limit applied for',
    'curnt_bi_upp': 'Bodily Injury Coverage Occurrence Limit applied for',
    'credit_score': 'Household credit score',
    'premium': 'Premium amount paid',
    'loss_amount': 'Loss amount due to past claims',
    'loss_ratio': 'Loss ratio percentage',
    'fault_claims': 'Number of fault claims in the last 5 years',
    'not_at_fault_claims': 'Number of not-at-fault claims in the last 5 years'
}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features and convert them to float, handling them via DataFrame
            data = {feature: float(request.form[feature]) for feature in feature_descriptions.keys()}
            features_df = pd.DataFrame([data])

            # Make prediction
            prediction_result = model.predict(features_df)
            prediction = 'Future Claim Likely' if prediction_result[0] == 1 else 'No Future Claim'
        except Exception as e:
            prediction = f'Error processing the form: {str(e)}'

    return render_template('form.html', prediction=prediction, feature_descriptions=feature_descriptions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
