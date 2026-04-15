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
    
    # Create applications table
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
    
    # Check if we need to seed
    cursor.execute('SELECT COUNT(*) FROM applications')
    if cursor.fetchone()[0] == 0:
        print("Seeding database with sample applications...")
        samples = [
            ('VNB-90488', 'Rajesh Kumar', 1200000, 450000, 785, 850000, 36, 785, 'Low Risk', 'Under Review', 'High'),
            ('VNB-90512', 'Ananya Sharma', 1500000, 200000, 812, 1500000, 48, 812, 'Minimal Risk', 'Pending', 'Normal'),
            ('VNB-90399', 'Vikram Singh', 600000, 550000, 645, 420000, 24, 645, 'High Risk', 'Escalated', 'Critical'),
            ('VNB-90555', 'Priya Verma', 850000, 100000, 790, 150000, 12, 790, 'Low Risk', 'Pending', 'Normal')
        ]
        cursor.executemany('''
        INSERT INTO applications (application_id, customer_name, income, debt, credit_score, loan_amount, duration, risk_score, risk_label, status, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', samples)
    
    conn.commit()
    conn.close()
    print("Database ready.")

def save_application(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    app_id = f"VNB-{os.urandom(2).hex().upper()}"
    
    cursor.execute('''
    INSERT INTO applications (application_id, customer_name, income, debt, credit_score, loan_amount, duration, risk_score, risk_label, status, priority)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        'Pending',
        'Normal'
    ))
    
    conn.commit()
    conn.close()
    return app_id

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
    
    cursor.execute('SELECT COUNT(*) FROM applications WHERE status = "Approved"')
    approved = cursor.fetchone()[0]
    
    cursor.execute('SELECT SUM(loan_amount) FROM applications')
    total_volume = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT COUNT(*) FROM applications WHERE status = "Under Review"')
    pending = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_applications': total,
        'approval_rate': (approved / total * 100) if total > 0 else 0,
        'total_volume': total_volume,
        'pending_count': pending
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

if __name__ == "__main__":
    init_db()
