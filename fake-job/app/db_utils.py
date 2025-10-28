import sqlite3
# from db_utils import insert_job


def create_connection():
    conn = sqlite3.connect('jobs.db')
    return conn

def create_table():
    conn = create_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            company_profile TEXT,
            description TEXT,
            requirements TEXT,
            benefits TEXT,
            result TEXT
        );
    ''')
    conn.commit()
    conn.close()


def insert_job(title, company_profile, description, requirements, benefits, result):
    conn = create_connection()
    conn.execute('''
        INSERT INTO jobs (title, company_profile, description, requirements, benefits, result)
        VALUES (?, ?, ?, ?, ?, ?);
    ''', (title, company_profile, description, requirements, benefits, result))
    conn.commit()
    conn.close()

def fetch_all_jobs():
    conn = create_connection()
    cur = conn.cursor()
    cur.execute('SELECT title, company_profile, result FROM jobs')
    rows = cur.fetchall()
    conn.close()
    return rows

def job_stats():
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT result, COUNT(*) FROM jobs GROUP BY result")
    result_counts = cur.fetchall()
    conn.close()
    return dict(result_counts)
