#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyodbc
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def safe_json_default(obj: Any) -> Any:
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def make_error(message: str, **extra: Any) -> Dict[str, Any]:
    payload = {
        "ok": False,
        "error": message,
    }
    payload.update(extra)
    return payload


def normalize_two_part_name(name: str) -> Tuple[str, str]:
    cleaned = name.strip().replace("[", "").replace("]", "")
    if "." in cleaned:
        parts = cleaned.split(".", 1)
        schema_name = parts[0].strip() or "dbo"
        object_name = parts[1].strip()
    else:
        schema_name = "dbo"
        object_name = cleaned
    return schema_name, object_name


READ_ONLY_SQL_PATTERN = re.compile(r"^\s*(?:WITH\b[\s\S]+?\bSELECT\b|SELECT\b)", re.IGNORECASE)
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|TRUNCATE|CREATE|EXEC|EXECUTE|GRANT|REVOKE|DENY|DBCC|BACKUP|RESTORE|USE)\b",
    re.IGNORECASE,
)


def validate_read_only_sql(sql: str) -> Optional[str]:
    normalized = sql.strip().rstrip(";").strip()
    if not normalized:
        return "SQL 不能是空的"

    if not READ_ONLY_SQL_PATTERN.match(normalized):
        return "只允許 SELECT 或 WITH ... SELECT 查詢"

    if FORBIDDEN_SQL_PATTERN.search(normalized):
        return "偵測到非唯讀語法，已拒絕執行"

    return None


class DbIntrospector:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.conn = self._connect()

    def _connect(self) -> pyodbc.Connection:
        server = os.getenv("MSSQL_SERVER", "").strip()
        database = os.getenv("MSSQL_DATABASE", "").strip()
        username = os.getenv("MSSQL_USERNAME", "").strip()
        password = os.getenv("MSSQL_PASSWORD", "").strip()
        driver = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server").strip()
        trusted = os.getenv("MSSQL_TRUSTED_CONNECTION", "").strip().lower()

        print(f"[DEBUG] .env path: {ENV_PATH}", flush=True)
        print(f"[DEBUG] .env exists: {ENV_PATH.exists()}", flush=True)
        print(f"[DEBUG] MSSQL_SERVER={server}", flush=True)
        print(f"[DEBUG] MSSQL_DATABASE={database}", flush=True)
        print(f"[DEBUG] MSSQL_DRIVER={driver}", flush=True)
        print(f"[DEBUG] trusted={trusted}", flush=True)

        if not server or not database:
            raise RuntimeError("缺少 MSSQL_SERVER 或 MSSQL_DATABASE。請確認 .env 是否存在且有載入。")

        if trusted in {"yes", "true", "1"}:
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Encrypt=no;"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
                f"Login Timeout=5;"
            )
        else:
            if not username or not password:
                raise RuntimeError("若未使用 Trusted Connection，需提供 MSSQL_USERNAME 與 MSSQL_PASSWORD。")
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
                f"Encrypt=no;"
                f"TrustServerCertificate=yes;"
                f"Login Timeout=5;"
            )

        print("[DEBUG] connecting to SQL Server...", flush=True)
        conn = pyodbc.connect(conn_str, autocommit=True)
        print("[DEBUG] connected.", flush=True)
        return conn

    def query_all(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()

        try:
            cursor.timeout = self.timeout
        except Exception:
            pass

        cursor.execute(sql, params)

        columns = [col[0] for col in cursor.description] if cursor.description else []
        rows = cursor.fetchall() if cursor.description else []

        results: List[Dict[str, Any]] = []
        for row in rows:
            item = {}
            for i, col_name in enumerate(columns):
                item[col_name] = row[i]
            results.append(item)
        return results

    def query_one(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        rows = self.query_all(sql, params)
        return rows[0] if rows else None

    def ping(self) -> Dict[str, Any]:
        row = self.query_one(
            """
            SELECT
                @@SERVERNAME AS server_name,
                DB_NAME() AS database_name,
                SYSTEM_USER AS login_name
            """
        )
        return {
            "ok": True,
            "type": "ping",
            "result": row or {},
        }

    def get_database_info(self) -> Dict[str, Any]:
        row = self.query_one(
            """
            SELECT
                DB_NAME() AS database_name,
                @@SERVERNAME AS server_name,
                SYSTEM_USER AS login_name
            """
        )
        return row or {}

    def get_procedure_definition(self, full_name: str) -> Dict[str, Any]:
        schema_name, object_name = normalize_two_part_name(full_name)

        obj = self.query_one(
            """
            SELECT
                p.object_id,
                s.name AS schema_name,
                p.name AS object_name,
                p.create_date,
                p.modify_date,
                m.definition
            FROM sys.procedures p
            INNER JOIN sys.schemas s ON p.schema_id = s.schema_id
            LEFT JOIN sys.sql_modules m ON p.object_id = m.object_id
            WHERE s.name = ? AND p.name = ?
            """,
            (schema_name, object_name),
        )

        if not obj:
            return make_error("找不到 Stored Procedure", requested=full_name)

        return {
            "ok": True,
            "type": "procedure",
            "database": self.get_database_info(),
            "object": {
                "schema_name": obj["schema_name"],
                "object_name": obj["object_name"],
                "full_name": f'{obj["schema_name"]}.{obj["object_name"]}',
                "create_date": obj["create_date"],
                "modify_date": obj["modify_date"],
                "definition": obj["definition"] or "",
            },
        }

    def search_objects(self, keyword: str, limit: int = 50) -> Dict[str, Any]:
        if limit <= 0:
            limit = 50
        if limit > 500:
            limit = 500

        name_pattern = f"%{keyword}%"
        body_pattern = f"%{keyword}%"

        rows = self.query_all(
            f"""
            SELECT TOP {limit}
                s.name AS schema_name,
                o.name AS object_name,
                o.type_desc,
                o.create_date,
                o.modify_date,
                CASE WHEN m.definition LIKE ? THEN 1 ELSE 0 END AS matched_in_definition,
                CASE WHEN o.name LIKE ? THEN 1 ELSE 0 END AS matched_in_name
            FROM sys.objects o
            INNER JOIN sys.schemas s
                ON o.schema_id = s.schema_id
            LEFT JOIN sys.sql_modules m
                ON o.object_id = m.object_id
            WHERE o.name LIKE ?
               OR m.definition LIKE ?
            ORDER BY
                CASE WHEN o.name LIKE ? THEN 0 ELSE 1 END,
                o.type_desc,
                s.name,
                o.name
            """,
            (body_pattern, name_pattern, name_pattern, body_pattern, name_pattern),
        )

        return {
            "ok": True,
            "type": "search",
            "database": self.get_database_info(),
            "keyword": keyword,
            "count": len(rows),
            "results": rows,
        }
    def get_table_schema(self, full_name: str) -> Dict[str, Any]:
        schema_name, object_name = normalize_two_part_name(full_name)

        rows = self.query_all(
            """
            SELECT
                c.name AS column_name,
                t.name AS data_type,
                c.is_nullable,
                c.max_length
            FROM sys.columns c
            JOIN sys.types t ON c.user_type_id = t.user_type_id
            JOIN sys.tables tb ON c.object_id = tb.object_id
            JOIN sys.schemas s ON tb.schema_id = s.schema_id
            WHERE s.name = ? AND tb.name = ?
            ORDER BY c.column_id
            """,
            (schema_name, object_name),
        )

        if not rows:
            return make_error("找不到 Table", requested=full_name)

        return {
            "ok": True,
            "type": "table",
            "table": f"{schema_name}.{object_name}",
            "columns": rows,
        }

    def execute_read_only_query(self, sql: str, limit: int = 100) -> Dict[str, Any]:
        error = validate_read_only_sql(sql)
        if error:
            return make_error(error, type="query", sql=sql)

        if limit <= 0:
            limit = 100
        if limit > 1000:
            limit = 1000

        wrapped_sql = f"""
        SELECT TOP {limit} *
        FROM (
            {sql.strip().rstrip(';')}
        ) AS src
        """
        rows = self.query_all(wrapped_sql)

        return {
            "ok": True,
            "type": "query",
            "database": self.get_database_info(),
            "limit": limit,
            "row_count": len(rows),
            "sql": sql,
            "rows": rows,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="唯讀 MSSQL introspection 工具")
    parser.add_argument(
        "command",
        choices=["ping", "proc", "search", "table", "query"],
        help="要執行的命令",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="",
        help="目標名稱，例如 dbo.刷條碼 / 關鍵字",
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=15)
    return parser

def main() -> int:
        parser = build_parser()
        args = parser.parse_args()

        try:
            db = DbIntrospector(timeout=args.timeout)

            if args.command == "ping":
                result = db.ping()
            elif args.command == "proc":
                result = db.get_procedure_definition(args.target)
            elif args.command == "search":
                result = db.search_objects(args.target, limit=args.limit)
            elif args.command == "table":
                result = db.get_table_schema(args.target)
            elif args.command == "query":
                result = db.execute_read_only_query(args.target, limit=args.limit)
            else:
                result = make_error("不支援的命令", command=args.command)

        except pyodbc.Error as ex:
            result = make_error("資料庫連線或查詢失敗", detail=str(ex))
        except Exception as ex:
            result = make_error("執行失敗", detail=str(ex))

        if args.pretty:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=safe_json_default), flush=True)
        else:
            print(json.dumps(result, ensure_ascii=False, default=safe_json_default), flush=True)

        return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
