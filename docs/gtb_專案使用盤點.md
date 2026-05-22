# GTB 專案使用盤點

最後更新：2026-04-14
掃描範圍：`C:\Users\pos\Desktop`

## 1) 目前主要使用 GTB 的專案（以你最新口徑為準）

1. `C:\Users\pos\Desktop\cwsoft-aquan-manager`
- 用途：阿全經理 / 全葳小幫手
- 啟動目標：`gtb.py --conf pos`

2. `C:\Users\pos\Desktop\cwsoft-ai-customer-service`
- 用途：AI 客服
- 啟動目標：`gtb.py --conf cs --mode shadow`

3. `C:\Users\pos\Desktop\cwsoft-clerk`
- 用途：公司進階行銷方案中的客戶 Line OA 助理（櫃台/行銷情境）
- 啟動目標：依專案 conf 設定（待補正式啟動指令）

## 2) 仍可看到的歷史或周邊啟動點（不列為主要專案）

1. 專案路徑：`C:\Users\pos\Desktop\cwsoft-linebot-service`
- 證據：[tasks.json](C:/Users/pos/Desktop/cwsoft-linebot-service/.vscode/tasks.json)
- 目前啟動方式：`py -3 main.py pos`
- 執行目錄：`C:\Users\pos\Desktop\general-task-bot`
- 判定：歷史啟動配置，仍是舊入口（`main.py`）

2. 啟動腳本：`C:\Users\pos\Desktop\start_all.bat`
- 證據：[start_all.bat](C:/Users/pos/Desktop/start_all.bat)
- 目前啟動方式：`cd /d C:\Users\pos\Desktop\general-task-bot && py -3 main.py pos`
- 判定：正在啟動 GTB（舊入口 `main.py`）

## 3) 盤點結論（當前）

1. 目前主要 GTB 專案：3 個
- `C:\Users\pos\Desktop\cwsoft-aquan-manager`
- `C:\Users\pos\Desktop\cwsoft-ai-customer-service`
- `C:\Users\pos\Desktop\cwsoft-clerk`

2. 其他引用 GTB 的啟動設定，先視為歷史或待切換
- `cwsoft-linebot-service/.vscode/tasks.json`
- `C:\Users\pos\Desktop\start_all.bat`

3. 結構更新（2026-04-14）
- `cwsoft-pos-knowledge-core` 不再獨立運行，已整合進 `C:\Users\pos\Desktop\cwsoft-ai-customer-service`
- `C:\Users\pos\Desktop\cwsoft-liff-pages` 已建立（已 pull）

4. 三個 chatbot 的定位差異（2026-04-14）
- `cwsoft-aquan-manager`：面向 POS/門市管理流程，強調指令執行與營運任務處理。
- `cwsoft-ai-customer-service`：面向客服問答，包含基礎知識庫回覆與客服場景對話。
- `cwsoft-clerk`：面向行銷方案客戶的 Line OA 助理場景，服務對象與任務流不同於前兩者。

## 4) 下一步建議

1. 將 `cwsoft-linebot-service/.vscode/tasks.json` 的「阿全總管（GTB）」改為 `gtb.py` 啟動（避免誤走舊入口）。
2. 將 `start_all.bat` 的 `main.py pos` 改為 `gtb.py --conf pos`（避免混用）。
3. 為 `cwsoft-aquan-manager`、`cwsoft-ai-customer-service`、`cwsoft-clerk` 各補一份專用啟動 task（固定 port 與 conf）。
