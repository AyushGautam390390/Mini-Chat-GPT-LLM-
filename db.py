import sqlite3

conn   = sqlite3.connect("chat.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    TEXT,
        session_id TEXT,
        role       TEXT,
        message    TEXT,
        timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

def save_message(user_id, session_id, role, message):
    cursor.execute(
        "INSERT INTO chats (user_id, session_id, role, message) VALUES (?, ?, ?, ?)",
        (user_id, session_id, role, message)
    )
    conn.commit()

def load_memory(user_id, session_id, limit=10):
    cursor.execute(
        "SELECT role, message FROM chats WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
        (user_id, session_id, limit)
    )
    return cursor.fetchall()[::-1]