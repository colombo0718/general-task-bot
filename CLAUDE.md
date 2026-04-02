# CLAUDE.md

This file is for LLMs, coding agents, and automation tools that need to understand this repository quickly.

## One-Sentence Summary

`general-task-bot` is a LINE webhook service that uses LLM extractors plus config-driven task definitions to convert natural-language messages into backend API calls, with optional human confirmation and scheduled execution.

## Primary Entry Point

- [main.py](/c:/Users/USER/general-task-bot/main.py)

If you only have time to read one file, read `main.py` first.

## Mental Model

This repo is not organized around hardcoded business logic per feature.
It is organized around a reusable execution engine:

- `prompts*.ini` tells the LLM what to extract
- `mission*.json` tells the app how to classify intents and which action to run
- `main.py` orchestrates the runtime flow

Most new features should be implemented by editing prompt and mission config first, and only editing Python when the framework itself needs new behavior.

## Runtime Flow

The request lifecycle in [main.py](/c:/Users/USER/general-task-bot/main.py) is:

1. Receive LINE webhook at `/callback/<oaid>`
2. Resolve LINE credentials from `oa_registry.json`
3. Parse incoming text message
4. Rebuild `customerlist.txt` by running [generate_customerlist_simple.py](/c:/Users/USER/general-task-bot/generate_customerlist_simple.py)
5. Check whether the user is responding to a pending confirmation flow
6. Check whether the user is requesting todo list viewing
7. Extract `run_at` if the message implies delayed execution
8. Run the mission classifier using `classify_tree.prompt_key`
9. Resolve a task from the classifier output
10. Extract task fields with `gather_fields()`
11. Optionally post-process a field with `match_pool` candidate matching
12. Build the API URL with `build_command()`
13. Depending on `human_check`, either:
    wait for confirmation,
    auto-confirm,
    or execute immediately
14. Execute the backend API via `execute_command()`
15. Reply to the user on LINE

## Important Files

- [main.py](/c:/Users/USER/general-task-bot/main.py)
  Main Flask app and orchestration logic.
- [prompts_system.ini](/c:/Users/USER/general-task-bot/prompts_system.ini)
  Shared extractors used across modes.
- [prompts.ini](/c:/Users/USER/general-task-bot/prompts.ini)
  Default task prompts.
- [prompts_pos.ini](/c:/Users/USER/general-task-bot/prompts_pos.ini)
  POS-mode prompts.
- [prompts_store.ini](/c:/Users/USER/general-task-bot/prompts_store.ini)
  Store-mode prompts.
- [prompts_cs.ini](/c:/Users/USER/general-task-bot/prompts_cs.ini)
  Customer-support prompts.
- [mission.json](/c:/Users/USER/general-task-bot/mission.json)
  Default task definitions.
- [mission_pos.json](/c:/Users/USER/general-task-bot/mission_pos.json)
  POS task definitions.
- [mission_store.json](/c:/Users/USER/general-task-bot/mission_store.json)
  Store task definitions.
- [mission_cs.json](/c:/Users/USER/general-task-bot/mission_cs.json)
  Customer-support task definitions.
- [todo_list.py](/c:/Users/USER/general-task-bot/todo_list.py)
  SQLite helpers for scheduled tasks.
- [todo_worker.py](/c:/Users/USER/general-task-bot/todo_worker.py)
  Background executor for scheduled tasks.
- [generate_customerlist_simple.py](/c:/Users/USER/general-task-bot/generate_customerlist_simple.py)
  Rebuilds customer candidate names from SQL Server.
- [TODO.md](/c:/Users/USER/general-task-bot/TODO.md)
  Current implementation plan, especially for clarification flows.

## Config System

### Prompt Files

`prompts*.ini` stores extractor prompts under `[extractors]`.
Each extractor is expected to return a tightly constrained value, often:

- an enum token
- a single number
- a date or datetime
- `true` / `false`
- `null`

Agents should preserve this strictness. The runtime expects machine-readable outputs, not free-form prose.

### Mission Files

Each `mission*.json` file defines:

- `classify_tree`
  Maps classifier outputs to `task_id`
- `tasks`
  Defines the fields required to execute each task

Each task usually contains:

- `description`
- `human_check`
- `fields`
- `action`

Field metadata can include:

- `prompt_key`
- `required`
- `reference`
- `match_pool`

## Human Confirmation Model

`human_check` has three meanings:

- `false`
  Execute directly.
- `true`
  Always wait for explicit user confirmation.
- `auto`
  Store a pending command and allow an implicit default path when the follow-up reply is ambiguous.

Pending confirmation state is stored in the in-memory `todo_command` dict inside [main.py](/c:/Users/USER/general-task-bot/main.py).

## Name Matching

Name extraction is not trusted blindly.
When a field specifies `match_pool`, the runtime:

1. Loads a candidate list
2. Compares extracted text against candidates
3. Uses Chinese pinyin similarity plus plain string similarity
4. Replaces the extracted value with the best candidate when confidence is high enough

This is especially important for customer names affected by ASR mistakes, homophones, or informal phrasing.

## Scheduling

Delayed tasks are written to `todo_list.db`.
The schema and insert/list logic live in [todo_list.py](/c:/Users/USER/general-task-bot/todo_list.py).
Execution is handled by [todo_worker.py](/c:/Users/USER/general-task-bot/todo_worker.py).

Current model:

- `main.py` detects delayed execution intent
- A pending record is inserted into sqlite
- `todo_worker.py` periodically scans for due tasks and POSTs them

## Modes

The program chooses prompt and mission files from command-line arguments:

- `python main.py`
  Uses `prompts.ini` and `mission.json`
- `python main.py pos`
  Uses `prompts_pos.ini` and `mission_pos.json`
- `python main.py store`
  Uses `prompts_store.ini` and `mission_store.json`
- `python main.py cs`
  Uses `prompts_cs.ini` and `mission_cs.json`

Optional second argument sets the port.

## External Dependencies and Local Secrets

This repo depends on local/private resources that may not exist in a clean checkout:

- `.env`
  Holds LLM provider API keys such as `GROQ_API_KEY`, `HF_TOKEN`, `OPENROUTER_API_KEY`
- `oa_registry.json`
  Maps `oaid` to LINE channel credentials
- SQL Server connectivity
  Required by [generate_customerlist_simple.py](/c:/Users/USER/general-task-bot/generate_customerlist_simple.py)

Do not assume the repo is runnable without these files.

## Editing Guidance for Agents

- Start by checking whether the change belongs in config or framework code.
- Prefer editing `prompts*.ini` and `mission*.json` for new business tasks.
- Edit `main.py` only when the runtime needs a new capability.
- Preserve extractor output contracts.
- Be careful with confirmation flows in `todo_command`; they are stateful and user-facing.
- Be careful with text encoding when editing Chinese content.
- Do not remove or overwrite ignored private files such as `oa_registry.json`.

## Current Work In Progress

[TODO.md](/c:/Users/USER/general-task-bot/TODO.md) currently describes a planned clarification feature:

- return top-N candidate names instead of only one best match
- detect low-confidence name matches in `gather_fields()`
- add a `pending_clarification` state to `todo_command`
- let `callback()` continue execution after the user selects the intended company
- ideally use LINE Quick Reply instead of plain text

If a task mentions clarification, ambiguity, top-N matching, or Quick Reply, read `TODO.md` before changing code.

## Suggested Reading Order for a New Agent

1. [main.py](/c:/Users/USER/general-task-bot/main.py)
2. [mission_cs.json](/c:/Users/USER/general-task-bot/mission_cs.json) or the mission file relevant to your mode
3. [prompts_cs.ini](/c:/Users/USER/general-task-bot/prompts_cs.ini) or the prompt file relevant to your mode
4. [docs/mission_json_guide.md](/c:/Users/USER/general-task-bot/docs/mission_json_guide.md)
5. [docs/prompts_ini_guide.md](/c:/Users/USER/general-task-bot/docs/prompts_ini_guide.md)
6. [TODO.md](/c:/Users/USER/general-task-bot/TODO.md)

## Safe Assumptions

- The project is intended for Traditional Chinese usage.
- The repo may be in active local development and may contain uncommitted changes.
- Mission and prompt files are business-critical and often more important than helper scripts.
- A feature request may only require config changes rather than Python changes.
