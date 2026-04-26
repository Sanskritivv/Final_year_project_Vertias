import sqlite3
import os
from datetime import datetime
import json

DB_PATH = os.path.join(os.path.dirname(__file__), 'veritas.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    print("Initializing database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create applications table with expanded features
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        application_id TEXT UNIQUE,
        customer_name TEXT,
        income REAL,
        debt REAL,
        credit_score INTEGER,
        loan_amount REAL,
        duration INTEGER,
        risk_score REAL,
        risk_label TEXT,
        status TEXT,
        priority TEXT,
        age INTEGER,
        emp_length REAL,
        home_ownership TEXT,
        loan_intent TEXT,
        loan_grade TEXT,
        int_rate REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Simple migration: check if 'age' exists, if not, recreate or alter
    try:
        cursor.execute("SELECT age FROM applications LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating applications table...")
        cursor.execute("DROP TABLE IF EXISTS applications")
        # Re-run create
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id TEXT UNIQUE,
            customer_name TEXT,
            income REAL,
            debt REAL,
            credit_score INTEGER,
            loan_amount REAL,
            duration INTEGER,
            risk_score REAL,
            risk_label TEXT,
            status TEXT,
            priority TEXT,
            age INTEGER,
            emp_length REAL,
            home_ownership TEXT,
            loan_intent TEXT,
            loan_grade TEXT,
            int_rate REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS client_applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        application_id TEXT UNIQUE,
        estimated_loan REAL DEFAULT 1250000,
        doc_step_complete INTEGER DEFAULT 0,
        iris_verified INTEGER DEFAULT 0,
        final_review_ready INTEGER DEFAULT 0,
        status TEXT DEFAULT 'Draft',
        documents_json TEXT DEFAULT '{}',
        has_logged_in INTEGER DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS client_support_tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        subject TEXT NOT NULL,
        message TEXT NOT NULL,
        status TEXT DEFAULT 'Open',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Migration for client_applications: add has_logged_in
    try:
        cursor.execute("SELECT has_logged_in FROM client_applications LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating client_applications table...")
        cursor.execute("ALTER TABLE client_applications ADD COLUMN has_logged_in INTEGER DEFAULT 0")
    
    # Check if we need to seed
    cursor.execute('SELECT COUNT(*) FROM applications')
    if cursor.fetchone()[0] == 0:
        print("Seeding database with sample applications...")
        samples = [
            ('VNB-90488', 'Rajesh Kumar', 1200000, 450000, 785, 850000, 36, 785, 'Low Risk', 'Waiting', 'High', 34, 12.0, 'MORTGAGE', 'PERSONAL', 'A', 11.5),
            ('VNB-90512', 'Ananya Sharma', 1500000, 200000, 812, 1500000, 48, 812, 'Minimal Risk', 'Accepted', 'Normal', 29, 8.0, 'RENT', 'EDUCATION', 'A', 10.2),
            ('VNB-90399', 'Vikram Singh', 600000, 550000, 645, 420000, 24, 645, 'High Risk', 'Rejected', 'Critical', 45, 1.5, 'RENT', 'MEDICAL', 'D', 14.8),
            ('VNB-90555', 'Priya Verma', 850000, 100000, 790, 150000, 12, 790, 'Low Risk', 'Accepted', 'Normal', 24, 4.0, 'RENT', 'PERSONAL', 'B', 11.0)
        ]
        cursor.executemany('''
        INSERT INTO applications (application_id, customer_name, income, debt, credit_score, loan_amount, duration, risk_score, risk_label, status, priority, age, emp_length, home_ownership, loan_intent, loan_grade, int_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', samples)
    
    conn.commit()
    
    # Normalize existing statuses for the new workflow
    cursor.execute("UPDATE applications SET status = 'Accepted' WHERE status = 'Approved'")
    cursor.execute("UPDATE applications SET status = 'Waiting' WHERE status IN ('Pending', 'Under Review', 'In Progress')")
    conn.commit()
    
    conn.close()
    print("Database ready.")

def save_application(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    app_id = f"VNB-{os.urandom(2).hex().upper()}"
    
    cursor.execute('''
    INSERT INTO applications (application_id, customer_name, income, debt, credit_score, loan_amount, duration, risk_score, risk_label, status, priority, age, emp_length, home_ownership, loan_intent, loan_grade, int_rate)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        app_id,
        data.get('customer_name', 'Guest User'),
        data.get('income'),
        data.get('debt'),
        data.get('credit_score'),
        data.get('loan_amount'),
        data.get('duration'),
        data.get('risk_score'),
        data.get('risk_label'),
        data.get('status', 'Waiting'),
        data.get('priority', 'Normal'),
        data.get('age'),
        data.get('emp_length'),
        data.get('home_ownership', 'RENT'),
        data.get('loan_intent', 'PERSONAL'),
        data.get('loan_grade', 'B'),
        data.get('int_rate', 11.0)
    ))
    
    conn.commit()
    conn.close()
    return app_id

def update_application_status(app_id, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE applications SET status = ? WHERE application_id = ?',
        (status, app_id)
    )
    conn.commit()
    conn.close()
    return True

def delete_application(app_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM applications WHERE application_id = ?', (app_id,))
    conn.commit()
    conn.close()
    return True

def get_all_applications():
    conn = get_db_connection()
    apps = conn.execute('SELECT * FROM applications ORDER BY created_at DESC').fetchall()
    conn.close()
    return [dict(app) for app in apps]

def get_analytics():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM applications')
    total = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM applications WHERE status = "Accepted"')
    approved = cursor.fetchone()[0]
    
    cursor.execute('SELECT SUM(loan_amount) FROM applications')
    total_volume = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT COUNT(*) FROM applications WHERE status = "Waiting"')
    pending = cursor.fetchone()[0]
    
    # 1. Risk Label Distribution
    cursor.execute('SELECT risk_label, COUNT(*) FROM applications GROUP BY risk_label')
    risk_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    # 2. Status Volume Distribution
    cursor.execute('SELECT status, SUM(loan_amount) FROM applications GROUP BY status')
    status_volume = {row[0]: row[1] for row in cursor.fetchall()}
    
    # 3. Monthly Trend (last 6 months)
    cursor.execute("SELECT strftime('%m', created_at) as month, SUM(loan_amount) FROM applications GROUP BY month ORDER BY month DESC LIMIT 6")
    monthly_trend = {row[0]: row[1] for row in cursor.fetchall()}
    
    # 4. Grade Distribution
    cursor.execute('SELECT loan_grade, COUNT(*) FROM applications GROUP BY loan_grade')
    grade_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        'total_applications': total,
        'approval_rate': (approved / total * 100) if total > 0 else 0,
        'total_volume': total_volume,
        'pending_count': pending,
        'risk_distribution': risk_dist,
        'status_volume': status_volume,
        'monthly_trend': monthly_trend,
        'grade_distribution': grade_dist
    }

def _default_documents():
    return {
        "proof_identity": {"label": "Proof of Identity", "required": True, "status": "not_started", "filename": ""},
        "proof_income": {"label": "Proof of Income", "required": True, "status": "not_started", "filename": ""},
        "proof_address": {"label": "Address Proof", "required": True, "status": "not_started", "filename": ""},
        "bank_statements": {"label": "Bank Statements", "required": True, "status": "not_started", "filename": ""},
        "tax_returns": {"label": "Tax Returns", "required": True, "status": "not_started", "filename": ""},
        "employment_verification": {"label": "Employment Verification", "required": True, "status": "not_started", "filename": ""}
    }

def _compute_doc_step_complete(documents):
    required_docs = [doc for doc in documents.values() if doc.get("required")]
    return int(all(doc.get("status") == "uploaded" for doc in required_docs))

def _compute_final_review_ready(doc_step_complete, iris_verified):
    return int(bool(doc_step_complete) and bool(iris_verified))

def _get_or_create_client_application_row(conn, username):
    cursor = conn.cursor()
    row = cursor.execute(
        "SELECT * FROM client_applications WHERE username = ?",
        (username,)
    ).fetchone()
    if row:
        return row

    app_id = f"VRT-{os.urandom(3).hex().upper()}"
    documents = _default_documents()
    cursor.execute(
        '''
        INSERT INTO client_applications (username, application_id, documents_json)
        VALUES (?, ?, ?)
        ''',
        (username, app_id, json.dumps(documents))
    )
    conn.commit()
    return cursor.execute(
        "SELECT * FROM client_applications WHERE username = ?",
        (username,)
    ).fetchone()

def get_or_create_client_application(username):
    conn = get_db_connection()
    row = _get_or_create_client_application_row(conn, username)
    conn.close()
    app = dict(row)
    app["documents"] = json.loads(app.get("documents_json") or "{}")
    return app

def update_client_document(username, document_key, filename):
    conn = get_db_connection()
    row = _get_or_create_client_application_row(conn, username)
    app = dict(row)
    documents = json.loads(app.get("documents_json") or "{}")

    if document_key not in documents:
        conn.close()
        raise ValueError("Unknown document type")

    documents[document_key]["status"] = "uploaded"
    documents[document_key]["filename"] = filename

    doc_step_complete = _compute_doc_step_complete(documents)
    final_review_ready = _compute_final_review_ready(doc_step_complete, app.get("iris_verified", 0))
    status = "Ready for Review" if final_review_ready else "In Progress"

    conn.execute(
        '''
        UPDATE client_applications
        SET documents_json = ?, doc_step_complete = ?, final_review_ready = ?, status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE username = ?
        ''',
        (json.dumps(documents), doc_step_complete, final_review_ready, status, username)
    )
    conn.commit()
    conn.close()
    return get_or_create_client_application(username)

def set_client_iris_verified(username):
    conn = get_db_connection()
    row = _get_or_create_client_application_row(conn, username)
    app = dict(row)
    documents = json.loads(app.get("documents_json") or "{}")
    doc_step_complete = _compute_doc_step_complete(documents)
    final_review_ready = _compute_final_review_ready(doc_step_complete, 1)
    status = "Ready for Review" if final_review_ready else "In Progress"

    conn.execute(
        '''
        UPDATE client_applications
        SET iris_verified = 1, final_review_ready = ?, status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE username = ?
        ''',
        (final_review_ready, status, username)
    )
    conn.commit()
    conn.close()
    return get_or_create_client_application(username)

def mark_client_digilocker(username):
    conn = get_db_connection()
    row = _get_or_create_client_application_row(conn, username)
    app = dict(row)
    documents = json.loads(app.get("documents_json") or "{}")

    for key, doc in documents.items():
        if doc.get("required"):
            doc["status"] = "uploaded"
            if not doc.get("filename"):
                doc["filename"] = f"{key}.pdf"

    doc_step_complete = _compute_doc_step_complete(documents)
    final_review_ready = _compute_final_review_ready(doc_step_complete, app.get("iris_verified", 0))
    status = "Ready for Review" if final_review_ready else "In Progress"

    conn.execute(
        '''
        UPDATE client_applications
        SET documents_json = ?, doc_step_complete = ?, final_review_ready = ?, status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE username = ?
        ''',
        (json.dumps(documents), doc_step_complete, final_review_ready, status, username)
    )
    conn.commit()
    conn.close()
    return get_or_create_client_application(username)

def submit_client_onboarding(username):
    conn = get_db_connection()
    row = _get_or_create_client_application_row(conn, username)
    app = dict(row)
    
    # Only allow submission if documents and iris are ready
    if app.get("final_review_ready"):
        conn.execute(
            '''
            UPDATE client_applications
            SET status = 'Submitted', updated_at = CURRENT_TIMESTAMP
            WHERE username = ?
            ''',
            (username,)
        )
        conn.commit()
    
    conn.close()
    return get_or_create_client_application(username)

def create_support_ticket(username, subject, message):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO client_support_tickets (username, subject, message, status)
        VALUES (?, ?, ?, 'Open')
        ''',
        (username, subject, message)
    )
    ticket_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return ticket_id

def get_support_tickets(username):
    conn = get_db_connection()
    rows = conn.execute(
        '''
        SELECT id, subject, message, status, created_at
        FROM client_support_tickets
        WHERE username = ?
        ORDER BY created_at DESC
        ''',
        (username,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def handle_client_login(username):
    """
    Logic for conditional login flow:
    - On first login: has_logged_in = 0 -> 1.
    - On returning login: skip documentation step.
    """
    conn = get_db_connection()
    row = _get_or_create_client_application_row(conn, username)
    app = dict(row)
    
    if app.get('has_logged_in', 0) == 1:
        # Returning user: auto-complete documentation if not already done
        if not app.get('doc_step_complete'):
            print(f"Returning user {username}: Skipping documentation step.")
            # Use existing mark_client_digilocker logic to "fake" doc completion
            # (Or just set the bit directly)
            conn.execute(
                "UPDATE client_applications SET doc_step_complete = 1 WHERE username = ?",
                (username,)
            )
            conn.commit()
    else:
        # First login: mark that they have logged in now
        print(f"First-time user {username}: Enforcing full sequence.")
        conn.execute(
            "UPDATE client_applications SET has_logged_in = 1 WHERE username = ?",
            (username,)
        )
        conn.commit()
    
    conn.close()
    return True

if __name__ == "__main__":
    init_db()
