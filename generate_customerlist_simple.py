# generate_customerlist_simple.py
import pyodbc
from pathlib import Path

# ===== 把這四個改成你的實際連線資訊 =====
DRIVER = "ODBC Driver 17 for SQL Server"
SERVER = "www.cwsoft.com.tw,1226"
USER   = "Pos"
PWD    = "sql2wsxCFT^3edc"
# =====================================

def main():
    conn_str = (
        f"DRIVER={{{DRIVER}}};"
        f"SERVER={SERVER};"
        f"DATABASE=POSConfig;"
        f"UID={USER};"
        f"PWD={PWD}"
    )

    query = """
        SELECT 資料庫名
        FROM 分店設定檔
        WHERE ISNULL(停用, 0) = 0
        ORDER BY 主索引 ASC
    """

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute(query)

    db_names = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    out_path = Path("customerlist.txt")
    with out_path.open("w", encoding="utf-8") as f:
        for name in db_names:
            f.write(str(name).strip() + "\n")

    print(f"已將 {len(db_names)} 家公司寫入 {out_path.resolve()}")

if __name__ == "__main__":
    main()
