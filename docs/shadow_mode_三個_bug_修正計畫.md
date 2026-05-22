# Shadow Mode 三個 Bug 修正計畫

- 發現日期：2026-04-27
- 影響範圍：`gtb.py` shadow mode（`--mode shadow`）+ `admin_server.py`
- 狀態：待確認後實作

---

## 背景

AI 客服影子模式（cwsoft-ai-customer-service）目前已可收訊息、生成 AI 草稿、由工程師在管理介面發送回覆。
但有三個問題回報：

1. 群組訊息沒進入影子模式
2. 使用者傳圖片不會進來
3. 工程師從影子介面發送訊息時跳出 bug，訊息有傳出去（LINE OA 後台可確認），但影子介面沒記錄到

---

## Bug 1：群組訊息沒進來

### 根本原因

`gtb.py` webhook handler 第 527 行：

```python
profile = line_bot_api.get_profile(user_id)
```

這個 API 只適用於 1-on-1 聊天（user source）。當 LINE OA 被加入群組後，收到的訊息 source type 是 `group`，此時呼叫 `get_profile` 會拋出例外（LINE API 對群組成員有不同的查詢路徑）。

例外沒有被 catch，整個 event 處理流程中斷，訊息不會被存進 shadow DB。

### 預計修正

在讀取 profile 前，先判斷 source type，改用對應的 API：

```python
source_type = getattr(event.source, "type", "user")
try:
    if source_type == "group":
        profile = line_bot_api.get_group_member_profile(event.source.group_id, user_id)
    elif source_type == "room":
        profile = line_bot_api.get_room_member_profile(event.source.room_id, user_id)
    else:
        profile = line_bot_api.get_profile(user_id)
    display_name = profile.display_name
except Exception:
    display_name = user_id  # fallback，不讓 profile 失敗中斷整個 event
```

### 對其他 chatbot 的影響

非 shadow mode 的流程也有同樣的 `get_profile` 呼叫，同樣受影響。
但目前其他 chatbot 都是 1-on-1 場景，不太會收到 group source，所以實際上不會出問題。
修正後兩種 mode 都受益，無副作用。

---

## Bug 2：使用者傳圖片不進來

### 根本原因

`gtb.py` webhook handler 的 event filter（第 520 行）：

```python
if not (getattr(event, "message", None) and getattr(event.message, "text", None)):
    continue
```

圖片訊息的 `event.message.text` 是 `None`，所以整個 event 被 `continue` 跳過，完全不進入後續處理，shadow DB 沒有記錄。

### 預計修正

把 filter 從「只允許 text」改成「允許 text 和 image」，image 的 `user_message` 存 `[圖片]` 作為佔位符：

```python
if not getattr(event, "message", None):
    continue
msg_type = getattr(event.message, "type", "")
if msg_type not in ("text", "image"):
    continue

user_text = event.message.text.strip() if msg_type == "text" else "[圖片]"
```

### 未包含在本次修正的事項

- 圖片實際內容的下載與顯示（見下方「後續：圖片顯示完整方案」）
- 目前只做到「知道使用者傳了圖片」這一層

### 對其他 chatbot 的影響

非 shadow mode 的 chatbot 目前對圖片也是無聲跳過，行為不變。
修正後 shadow mode 能捕到圖片事件，其他 chatbot 不受影響。

---

## 後續：圖片顯示完整方案

Bug 2 的修正只存了 `[圖片]` 佔位字串，要在管理介面真正顯示圖片需要以下改動。

### 背景限制

LINE 的圖片透過 Content API 取得，但**只在事件發生後短時間內有效**，無法事後補拉。
因此必須在 webhook handler 收到圖片事件時，立刻下載並存到本機。

### 需要改動的地方

**（1）shadow DB schema：新增 `image_path` 欄位**

在 `init_user_message_table` 的 schema 加入：

```sql
image_path TEXT
```

同樣需要對舊 table 補欄位（ALTER TABLE + try-except，同 Bug 3 的做法）。

**（2）gtb.py：收到圖片時立刻下載**

在 shadow mode 的 `save_shadow_message` 前，先下載圖片：

```python
image_path = None
if msg_type == "image":
    message_id_line = event.message.id
    img_dir = os.path.join(PROJECT_DIR, "database", "images")
    os.makedirs(img_dir, exist_ok=True)
    img_file = os.path.join(img_dir, f"{message_id_line}.jpg")
    try:
        img_resp = requests.get(
            f"https://api-data.line.me/v2/bot/message/{message_id_line}/content",
            headers={"Authorization": f"Bearer {channel_access_token}"},
            timeout=10
        )
        img_resp.raise_for_status()
        with open(img_file, "wb") as f:
            f.write(img_resp.content)
        image_path = f"images/{message_id_line}.jpg"
    except Exception as e:
        print(f"[SHADOW] 圖片下載失敗：{e}")
```

`save_shadow_message` 也需要新增 `image_path` 參數，並在 INSERT 時寫入。

**（3）admin_server.py：新增 route 提供圖片檔案**

```python
@app.route("/admin/image/<filename>")
def serve_image(filename):
    img_dir = os.path.join(BASE_DIR, "database", "images")
    return send_from_directory(img_dir, filename)
```

同時 `get_messages` API 的 SELECT 需要包含 `image_path` 欄位並回傳。

**（4）admin/app.js：renderCard 顯示圖片**

在「客戶訊息」區塊，根據 `image_path` 決定渲染方式：

```javascript
const customerContent = m.image_path
  ? `<img src="/admin/image/${m.image_path.split('/').pop()}"
         style="max-width:100%;border-radius:8px;">`
  : escHtml(m.user_message || "")
```

### 改動總覽

| 改動 | 檔案 |
|------|------|
| 新增 `image_path` 欄位 + migration | `gtb.py` `init_user_message_table` |
| webhook 收圖片時下載到本機 | `gtb.py` shadow mode 區塊 |
| `save_shadow_message` 帶入 `image_path` | `gtb.py` |
| 新增 `/admin/image/<filename>` route | `admin_server.py` |
| `get_messages` 回傳 `image_path` | `admin_server.py` |
| `renderCard` 顯示圖片 | `admin/app.js` |

### 對其他 chatbot 的影響

圖片下載邏輯包在 `if SHADOW_MODE:` 區塊內，不影響其他 chatbot。
新增的 `image_path` 欄位由 `save_shadow_message` 帶入，該函數只在 shadow mode 被呼叫。

---

## Bug 3：工程師傳訊息跳出 bug，沒有記錄到

### 根本原因

`admin_server.py` 的 `send_reply` 流程：

1. 呼叫 LINE Push Message API → **成功**（LINE OA 後台有記錄）
2. 執行 `UPDATE {table} SET staff_reply = ? WHERE id = ?` → **失敗**

失敗原因：shadow DB 裡的 user message table 是在較早的時間點建立的，那時候 `init_user_message_table` 的 schema 還沒有 `staff_reply` 這個欄位。

`CREATE TABLE IF NOT EXISTS` 的特性是：table 已存在就直接跳過，不會補新欄位。所以舊 table 一直缺 `staff_reply`，UPDATE 時 SQLite 丟 `no such column: staff_reply`。

LINE push 已在 DB 失敗之前完成，所以訊息傳出去了，但 DB 沒更新，前端收到錯誤 response，彈出 alert。

### 預計修正

分兩個地方修：

**（A）gtb.py `init_user_message_table`**

在 `CREATE TABLE` 之後加一行嘗試補欄位，讓未來建立的 table 和已存在的舊 table 都有這個欄位：

```python
try:
    cur.execute(f"ALTER TABLE {table} ADD COLUMN staff_reply TEXT")
except Exception:
    pass  # 欄位已存在時 SQLite 會拋例外，正常忽略
conn.commit()
```

效果：下次有新訊息進來觸發 `init_user_message_table` 時，舊 table 會自動補上欄位。

**（B）admin_server.py `send_reply`**

在 UPDATE 之前加同樣的 migration，避免工程師試圖回覆時因 table 尚未接過新訊息而還沒補欄位就先失敗：

```python
try:
    conn.execute(f"ALTER TABLE {table} ADD COLUMN staff_reply TEXT")
    conn.commit()
except Exception:
    pass
conn.execute(
    f"UPDATE {table} SET staff_reply = ? WHERE id = ?",
    (text, message_id)
)
conn.commit()
```

### 為什麼要改兩個地方

只改 `gtb.py` 的話，需要等該 user 的新訊息進來才會觸發補欄位。
如果工程師在新訊息進來之前就點發送，依然會失敗。
`admin_server.py` 那邊加一層保險，確保 UPDATE 之前欄位一定存在。

### 對其他 chatbot 的影響

`init_user_message_table` 只在 shadow mode 被呼叫，其他 chatbot 不使用 shadow mode，所以不受影響。
`admin_server.py` 是獨立服務，與 gtb.py 的其他使用場景無關。

---

## 修正總結

| Bug | 修改檔案 | 修改位置 |
|-----|---------|---------|
| 群組訊息沒進來 | `gtb.py` | webhook handler，profile API 呼叫處 |
| 圖片不進來 | `gtb.py` | webhook handler，event filter 條件 |
| 傳訊息沒記錄 | `gtb.py` + `admin_server.py` | `init_user_message_table` + `send_reply` |

三個修正都在 shadow mode 的路徑內，不影響 gtb.py 正常模式下的其他 chatbot 行為。
