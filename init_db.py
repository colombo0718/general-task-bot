import os, sqlite3

def init_user_db(user_id: str, base_dir="databases"):
    # 確保目錄存在
    os.makedirs(base_dir, exist_ok=True)

    # 以 user_id 命名 DB 檔案
    db_path = os.path.join(base_dir, f"{user_id}.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 初始化表結構（這裡先示範記帳用）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item TEXT NOT NULL,
        price REAL NOT NULL,
        date TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS todos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        due_date TEXT,
        status TEXT DEFAULT 'open'
    );
    """)
    # created_at DATETIME DEFAULT CURRENT_TIMESTAMP

    conn.commit()
    conn.close()
    return db_path

if __name__ == "__main__":
    # 測試建立一個使用者 DB
    user_id = "Ud1fdd990474ff4799c27e156df7aeb27"
    db_path = init_user_db(user_id)
    print(f"資料庫已建立：{db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO expenses (item, price, date) VALUES (?, ?, ?)",
        ("午餐", 100, "2025-09-18")
    )

    conn.commit()
    conn.close()
