import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

# Train a dummy Random Forest model on startup to simulate the notebook's ML model
X_dummy = np.random.rand(1000, 5) 
X_dummy[:, 0] *= 2000000  # Annual Income
X_dummy[:, 1] *= 1000000  # Total Debt
X_dummy[:, 2] = X_dummy[:, 2] * 600 + 300  # Credit Score (300-900)
X_dummy[:, 3] *= 5000000  # Loan Amount
X_dummy[:, 4] = X_dummy[:, 4] * 114 + 6  # Duration (6-120 months)

# Dummy Target: Quant Score between 300 and 900
# Higher income, higher credit score -> good target
# Higher debt, higher loan amount -> bad target
y_dummy = (X_dummy[:, 2] * 0.6) + (X_dummy[:, 0] / (X_dummy[:, 1] + 1)) * 50 - (X_dummy[:, 3] / 100000) - X_dummy[:, 4] * 0.5
y_dummy = np.clip(y_dummy, 300, 900)

rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_dummy, y_dummy)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Expected inputs
        # income, debt, credit_score, loan_amount, duration
        income = float(data.get('income', 0).replace(',', ''))
        debt = float(data.get('debt', 0).replace(',', ''))
        credit_score = float(data.get('credit_score', 0))
        loan_amount = float(data.get('loan_amount', 0).replace(',', ''))
        duration = float(data.get('duration', 0))

        # Make prediction
        features = np.array([[income, debt, credit_score, loan_amount, duration]])
        score = rf_model.predict(features)[0]
        score = int(np.clip(score, 300, 900))
        
        # Determine strict logic
        if score > 700:
            risk_label = "Low Probability of Default"
            color = "emerald"
            recommendation = "Approval recommended at standard prime rate. No additional collateral required for this exposure level."
        elif score > 550:
            risk_label = "Moderate Risk"
            color = "amber"
            recommendation = "Approval suggested with increased interest rate or additional collateral to mitigate intermediate risk."
        else:
            risk_label = "High Risk"
            color = "red"
            recommendation = "Decline recommended. High probability of default detected based on current financial profile."
            
        idr = round(income / (debt + 1), 2)
        idr_status = "emerald" if idr > 1.5 else "amber" if idr > 0.8 else "red"

        return jsonify({
            'score': score,
            'risk_label': risk_label,
            'color': color,
            'recommendation': recommendation,
            'insights': [
                {'time': 'Current', 'text': f'Income to Debt Ratio (IDR) calculated: <span class="text-{idr_status}-600 font-bold">{idr}</span>.'},
                {'time': 'Current', 'text': f'Base Credit Score initialized at: <span class="text-primary font-bold">{int(credit_score)}</span>'},
                {'time': 'Current', 'text': f'Model execution completed successfully.'}
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
