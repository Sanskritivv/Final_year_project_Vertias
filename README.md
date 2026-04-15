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
- **AI Prediction Engine**: Uses a Random Forest model to assess credit risk based on income, debt, and credit history.
- **Client Portal**: Modern, multi-step application flow with document management and biometric (Iris recognition) simulations.
- **Dark Mode Support**: Fully responsive UI with high-end aesthetics.

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
