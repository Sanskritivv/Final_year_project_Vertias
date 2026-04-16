import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from database import (
    init_db,
    save_application,
    get_all_applications,
    get_analytics,
    get_or_create_client_application,
    update_client_document,
    set_client_iris_verified,
    mark_client_digilocker,
    create_support_ticket,
    get_support_tickets
)
from iris_engine.verifier import IrisVerifier, decode_image

# Configure paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'web', 'templates')
STATIC_DIR = os.path.join(BASE_DIR, '..', 'web', 'static')
UPLOAD_DIR = os.path.join(STATIC_DIR, 'uploads')

app = Flask(__name__,
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)
app.secret_key = os.environ.get('SECRET_KEY', 'veritas-demo-key-change-in-prod')
CORS(app)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Demo credentials
DEMO_CREDENTIALS = {
    'admin': {'password': 'admin123', 'role': 'company'},
    'client': {'password': 'client123', 'role': 'client'},
}

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_role' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def company_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('user_role') != 'company':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def client_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('user_role') != 'client':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_onboarding_next_step(app_data):
    """
    Determines the next step in the onboarding flow.
    Returns: endpoint_name
    """
    if not app_data.get('doc_step_complete'):
        return 'client_documents'
    if not app_data.get('iris_verified'):
        return 'client_iris'
    if app_data.get('status') != 'Submitted':
        return 'client_review'
    return 'client_dashboard'

def onboarding_enforcer(f):
    """
    Redirects user to the correct onboarding step if they haven't finished.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('user_role') != 'client':
            return redirect(url_for('login'))
        
        username = session.get('username')
        app_data = get_or_create_client_application(username)
        
        target = get_onboarding_next_step(app_data)
        
        # If the user is trying to access a page they aren't ready for, redirect.
        # But allow them to stay on the page if it IS the target step.
        if request.endpoint != target and target != 'client_dashboard':
            # Special case: allow dashboard if fully submitted
            if app_data.get('status') == 'Submitted' and request.endpoint == 'client_dashboard':
                 return f(*args, **kwargs)
            return redirect(url_for(target))
            
        return f(*args, **kwargs)
    return decorated

# Initialize Database
init_db()

# Load ML components
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')
ENCODER_PATH = os.path.join(BASE_DIR, 'encoders.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, 'features.joblib')

model = None

# Initialize Iris Verifier
print("Initializing Iris Recognition Engine...")
iris_verifier = IrisVerifier()

# Logic to map Credit Score to Loan Grade
def score_to_grade(score):
    if score >= 750: return 'A'
    if score >= 700: return 'B'
    if score >= 650: return 'C'
    if score >= 600: return 'D'
    if score >= 550: return 'E'
    return 'F'

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'company')

        user = DEMO_CREDENTIALS.get(username)
        if user and user['password'] == password:
            session['user_role'] = user['role']
            session['username'] = username
            redirect_url = '/dashboard' if user['role'] == 'company' else '/client_dashboard'
            return jsonify({'ok': True, 'redirect': redirect_url})
        # Allow demo bypass: any non-empty credentials route by role selection
        if username and password:
            session['user_role'] = role
            session['username'] = username
            redirect_url = '/dashboard' if role == 'company' else '/client_dashboard'
            return jsonify({'ok': True, 'redirect': redirect_url})
        return jsonify({'ok': False, 'error': 'Please enter your credentials.'}), 401

    return render_template('login.html')

@app.route('/dashboard')
@company_required
def dashboard():
    return render_template('dashboard.html', metrics=get_analytics())

@app.route('/client_dashboard')
@onboarding_enforcer
def client_dashboard():
    username = session.get('username', 'client')
    app_data = get_or_create_client_application(username)
    # Fetch all applications for the main view
    all_apps = get_all_applications()
    # Filter for this specific client (Demo logic: in real world, filter would be in SQL)
    client_apps = [a for a in all_apps if a['customer_name'] == username or a['customer_name'] == 'Rajesh Kumar'] # Rajesh is dummy for client role
    return render_template('client_dashboard.html', applications=client_apps, client_app=app_data)

@app.route('/client_onboarding')
@client_required
def client_onboarding():
    username = session.get('username')
    app_data = get_or_create_client_application(username)
    target = get_onboarding_next_step(app_data)
    # If the target is the dashboard, it means onboarding is done but we might need 
    # to show the review page if status isn't submitted yet.
    # get_onboarding_next_step already handles this logic.
    return redirect(url_for(target))

@app.route('/client_documents')
@onboarding_enforcer
def client_documents():
    username = session.get('username', 'client')
    app_data = get_or_create_client_application(username)
    return render_template('client_documents.html', client_app=app_data)

@app.route('/client_iris')
@onboarding_enforcer
def client_iris():
    return render_template('client_iris.html')

@app.route('/client_review')
@onboarding_enforcer
def client_review():
    username = session.get('username', 'client')
    app_data = get_or_create_client_application(username)
    return render_template('client_review.html', client_app=app_data)

@app.route('/client_support')
@client_required
def client_support():
    username = session.get('username', 'client')
    tickets = get_support_tickets(username)
    return render_template('client_support.html', tickets=tickets)

@app.route('/underwriting')
@company_required
def underwriting():
    return render_template('underwriting.html', metrics=get_analytics())

@app.route('/credit_risk')
@company_required
def credit_risk():
    return render_template('credit_risk.html')

@app.route('/analytics')
@company_required
def analytics():
    return render_template('analytics.html', metrics=get_analytics())

@app.route('/reports')
@company_required
def reports():
    return render_template('reports.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/applications')
@login_required
def api_applications():
    return jsonify(get_all_applications())

@app.route('/api/client/application')
@client_required
def api_client_application():
    username = session.get('username', 'client')
    return jsonify(get_or_create_client_application(username))

@app.route('/api/client/document-upload', methods=['POST'])
@client_required
def api_client_document_upload():
    username = session.get('username', 'client')
    document_key = request.form.get('document_key', '').strip()
    file = request.files.get('file')

    if not document_key:
        return jsonify({'error': 'Document key is required.'}), 400
    if not file or not file.filename:
        return jsonify({'error': 'Please select a file to upload.'}), 400

    safe_name = secure_filename(file.filename)
    storage_name = f"{username}_{document_key}_{safe_name}"
    save_path = os.path.join(UPLOAD_DIR, storage_name)
    file.save(save_path)

    try:
        updated = update_client_document(username, document_key, safe_name)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({'ok': True, 'application': updated})

@app.route('/api/client/digilocker', methods=['POST'])
@client_required
def api_client_digilocker():
    username = session.get('username', 'client')
    updated = mark_client_digilocker(username)
    return jsonify({'ok': True, 'application': updated})

@app.route('/api/client/iris/verify', methods=['POST'])
@client_required
def api_client_iris_verify():
    username = session.get('username', 'client')
    
    data = request.get_json(silent=True) or {}
    image_base64 = data.get('image')
    
    if not image_base64:
        return jsonify({'ok': False, 'error': 'NO_IMAGE', 'message': 'No image data received.'}), 400

    try:
        # Decode and Verify
        image_bgr = decode_image(image_base64)
        result = iris_verifier.verify(image_bgr)
        
        if result['ok']:
            updated = set_client_iris_verified(username)
            return jsonify({'ok': True, 'application': updated, 'message': result['message']})
        else:
            return jsonify({'ok': False, 'error': result['error'], 'message': result['message']}), 400
            
    except Exception as e:
        print(f"Iris verification error: {e}")
        return jsonify({'ok': False, 'error': 'SERVER_ERROR', 'message': 'An error occurred during biometric processing.'}), 500

@app.route('/api/client/submit', methods=['POST'])
@client_required
def api_client_submit():
    username = session.get('username', 'client')
    # Update status to Submitted
    from database import submit_client_onboarding
    updated = submit_client_onboarding(username)
    
    # Also create a formal application entry if needed
    # (In this schema, we just transition the existing onboarding record to 'Submitted')
    
    return jsonify({'ok': True, 'application': updated})

@app.route('/api/client/support', methods=['GET', 'POST'])
@client_required
def api_client_support():
    username = session.get('username', 'client')
    if request.method == 'GET':
        return jsonify(get_support_tickets(username))

    data = request.get_json(silent=True) or {}
    subject = (data.get('subject') or '').strip()
    message = (data.get('message') or '').strip()
    if not subject or not message:
        return jsonify({'error': 'Subject and message are required.'}), 400
    ticket_id = create_support_ticket(username, subject, message)
    return jsonify({'ok': True, 'ticket_id': ticket_id, 'tickets': get_support_tickets(username)})

@app.route('/api/analytics')
@login_required
def api_analytics():
    return jsonify(get_analytics())

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        
        # Extract inputs from UI
        income = float(str(data.get('income', 0)).replace(',', ''))
        debt = float(str(data.get('debt', 0)).replace(',', ''))
        credit_score = int(data.get('credit_score', 300))
        loan_amnt = float(str(data.get('loan_amount', 0)).replace(',', ''))
        duration = int(data.get('duration', 12))
        age = int(data.get('age', 30))
        emp_length = float(data.get('emp_length', 5.0))
        customer_name = data.get('customer_name', 'Guest User')

        if model:
            grade = score_to_grade(credit_score)
            
            # Feature engineering
            features = {
                'person_age': age,
                'person_income': income,
                'person_home_ownership': 'RENT', # Default
                'person_emp_length': emp_length,
                'loan_intent': 'PERSONAL', # Default
                'loan_grade': grade,
                'loan_amnt': loan_amnt,
                'loan_int_rate': 11.0, # Default
                'loan_percent_income': loan_amnt / income if income > 0 else 0,
                'cb_person_default_on_file': 'N', # Default
                'cb_person_cred_hist_length': int(duration/12)
            }
            
            # Encode categorical features
            input_df = pd.DataFrame([features])
            for col, le in encoders.items():
                if col in input_df.columns:
                    # Handle unseen labels by mapping to first class if necessary
                    try:
                        input_df[col] = le.transform(input_df[col])
                    except:
                        input_df[col] = 0

            # Reorder columns to match feature_names
            input_df = input_df[feature_names]
            
            # Prediction
            prediction_prob = model.predict_proba(input_df)[0]
            # Prob of default is second element (1)
            default_prob = prediction_prob[1]
            
            # Convert default probability to a score 0-1000 (higher is better)
            score = int((1 - default_prob) * 1000)
        else:
            # Fallback dummy logic
            score = 750
            default_prob = 0.1

        # Determine label and color
        if score > 800:
            label, color = "Minimal Risk", "emerald"
        elif score > 700:
            label, color = "Low Risk", "green"
        elif score > 500:
            label, color = "Medium Risk", "amber"
        else:
            label, color = "High Risk", "red"

        # Save to database
        app_id = save_application({
            'customer_name': customer_name,
            'income': income,
            'debt': debt,
            'credit_score': credit_score,
            'loan_amount': loan_amnt,
            'duration': duration,
            'risk_score': score,
            'risk_label': label
        })

        return jsonify({
            'score': score,
            'risk_label': label,
            'color': color,
            'application_id': app_id,
            'recommendation': "Automatic approval recommended" if score > 700 else "Requires manual review",
            'insights': [
                {'text': f'Monthly Income Coverage: {round(income/(loan_amnt/duration if duration > 0 else 1), 2)}x'},
                {'text': f'Derived Credit Grade: {score_to_grade(credit_score)}'},
                {'text': f'Exposure Level: {round((loan_amnt/income)*100, 1)}% of annual income'}
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
