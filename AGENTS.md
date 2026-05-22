# AGENTS.md

## Project context
This repository is a VB.NET + MSSQL project.

When the user asks:
- 這支 .vb 用到哪些 stored procedures
- DataTable 的資料怎麼來
- 某個欄位是從哪個 SP / SQL / table 來的
- 某段程式最後查到哪張表

You must not guess.
Always inspect the code first, then use the local DB tool to verify.

## Required workflow for VB.NET files
1. Read the target .vb file and nearby related files first.
2. Look for:
   - SqlCommand
   - CommandType.StoredProcedure
   - SqlDataAdapter / DataAdapter
   - DataTable / DataSet / Fill / FillSchema
   - ExecuteReader / ExecuteScalar / ExecuteNonQuery
   - helper methods that build SQL
   - shared DB utility classes
   - string-built SQL
3. If a stored procedure name is found, run:
   python tools/db_introspect.py proc <SP_NAME> --pretty
4. If a table name is found and schema is needed, run:
   python tools/db_introspect.py table <SCHEMA.TABLE> --pretty
5. If object relationships are unclear, run:
   python tools/db_introspect.py refs <OBJECT_NAME> --pretty
   python tools/db_introspect.py search <KEYWORD> --pretty
   Note: when using `tools/db_introspect.py`, run it in a non-sandbox environment. If it fails in sandbox, retry outside sandbox before concluding the DB/tool is broken.
6. In the final answer, always include:
   - call chain
   - related stored procedures
   - how the DataTable gets populated
   - likely source tables
   - uncertainties, if any

## Behavior rules
- Trace code first, then use the DB tool.
- Do not assume a stored procedure unless the code or DB search supports it.
- If dynamic SQL is used, explicitly say so.
- If a shared helper is involved, trace the helper before concluding.
- If there are multiple possible stored procedures, list them with confidence notes.
- Prefer grounded answers over fast guesses.
- Available database names are recorded in `tools/database_names.txt`. When the user asks for database names, available switch targets, or references `各資料庫設定`, read that file first before inferring names from memory.
- When the user explicitly says `切換資料庫到 <DB_NAME>`, directly update `MSSQL_DATABASE` in `.env` to `<DB_NAME>` and report that the switch is complete.
- For database switching, do not search broadly first if `.env` exists at the repo root; update the repo-root `.env` file directly.
- If `.env` text appears garbled in terminal output, read and write the file as UTF-8 and match the `MSSQL_DATABASE=` line by key name instead of relying on terminal-rendered characters.
- For `tools/db_introspect.py` commands, prefer a single consistent approved command prefix and command shape whenever possible so repeated permission prompts are minimized.
- When a matching approved prefix already exists, reuse that same PowerShell + `py -X utf8 tools/db_introspect.py ...` invocation format instead of introducing alternate command forms.
- Before modifying any code or project files, ask the user for confirmation first.
- Before running any action that updates, inserts, deletes, or otherwise writes data to the database, ask the user for confirmation first.
- Run `tools/db_introspect.py` outside sandbox by default. Do not treat a sandbox-only DB connection failure as proof that the database, credentials, or script are broken.
- When reading SQL Server schema metadata, do not assume `sys.columns.max_length` is the character length for `nvarchar` / `nchar`. In SQL Server, that value is stored in bytes. For Unicode columns, use `INFORMATION_SCHEMA.COLUMNS.CHARACTER_MAXIMUM_LENGTH` or convert `max_length / 2` before concluding the real column length.
- When column length accuracy matters for foreign keys, temp tables, procedure parameters, or migration SQL, verify the actual character length explicitly before answering or generating SQL.
- If `apply_patch` fails because the patch is too large, Windows sandbox limits are hit, or a similar oversized-edit/tooling error occurs, prefer splitting the file changes into smaller patches or rebuilding the file in smaller sections instead of retrying the same large patch.
- When the user asks for a `.vb` implementation draft, explain the approach in chat by default and do not create a separate concept/notes document unless the user explicitly asks for one.
- For WinForms `.Designer.vb` files, do not assume a garbled designer is only an encoding issue. First distinguish whether the problem is:
  - file encoding only, or
  - mojibake/corrupted control identifiers and `InitializeComponent` content.
- If a WinForms designer file has corrupted identifiers, broken `InitializeComponent` syntax, or the VS designer reports that generated code was manually modified, prefer rebuilding the `.Designer.vb` file with clean control names instead of trying to patch individual garbled lines.
- When adding or removing WinForms controls, keep the `Designer` declarations, `InitializeComponent` initialization, container `Controls.Add(...)`, and `Friend WithEvents` list synchronized in the same change to avoid designer breakage.

## SQL design note
- The point-system SQL design drafts for the advanced marketing module are stored under `C:\Users\USER\Documents\SQL Server Management Studio\進階行銷系統\點數系統`.
- Current draft files:
  - `01_點數系統_資料表.sql`
  - `02_點數系統_SP.sql`
- These files are design drafts only. Do not assume they are already deployed to the active database.
