# Veritas - AI-Powered Credit Risk Underwriting

Veritas is a professional fintech solution designed to streamline the credit underwriting process using machine learning. It provides an executive dashboard for monitoring portfolio health and a client portal for seamless loan applications.

## Project Structure

```text
veritas/
├── backend/            # Flask API and ML logic
│   └── app.py          # Main backend entry point
├── web/                # Frontend assets
│   ├── templates/      # HTML templates (Jinja2)
│   └── static/         # CSS, JS, and Images
├── data/               # Datasets and Research
│   ├── credit_risk_dataset.csv
│   └── research_notebook.ipynb
├── scripts/            # Internal utility scripts
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Features

- **Executive Dashboard**: Real-time visualization of approval rates, pending queues, and portfolio health.
- **AI Prediction Engine**: Advanced ensemble learning approach utilizing XGBoost, CatBoost, and LightGBM for state-of-the-art accuracy in credit risk assessment.
- **Client Portal**: Modern, multi-step application flow with document management and biometric (Iris recognition) simulations.
- **Dark Mode Support**: Fully responsive UI with high-end aesthetics.

## Model Performance & Research

The underwriting engine is powered by an ensemble of gradient-boosted decision trees, achieving significantly higher predictive power than standard linear models.

### Metrics Comparison

| Model | Accuracy | Precision | Recall | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **94.43%** | 97.83% | **76.05%** | 99.53% |
| **CatBoost** | 94.33% | 98.52% | 75.00% | 99.69% |
| **LightGBM** | 93.91% | 98.69% | 72.91% | 99.73% |
| **Ensemble (Voting)** | 93.61% | **99.44%** | 70.98% | **99.89%** |
| KNN | 79.43% | 55.65% | 26.13% | 94.22% |

### Key Risk Indicators (Feature Importance)

The model identifies the following factors as the most critical predictors of credit risk:

1. **Loan-to-Income Ratio** (23.3%) - The primary driver of default probability.
2. **Interest Rate** (15.5%) - Higher rates correlate strongly with higher risk profiles.
3. **Home Ownership (Rent)** (12.4%) - Applicants currently renting show a distinct statistical variance.
4. **Annual Income** (10.1%) - Direct correlation with repayment capacity.

> [!TIP]
> The **Ensemble Voting Classifier** is recommended for production environments where **Precision** is critical (minimizing False Positives/bad loans), while **XGBoost** offers the best balance for overall default detection (**Recall**).

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sanskritivv/Final_year_project_Vertias.git
   cd Final_year_project_Vertias
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Start the Flask server:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to `http://127.0.0.1:5000`

---
*Created as a Final Year Project by the Veritas Team.*
