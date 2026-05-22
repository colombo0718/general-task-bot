#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LINE Messaging API - 訊息拉取工具
"""
import requests
import json
import sys

# TOKEN 和配置
CHANNEL_ACCESS_TOKEN = "pECFyW0nU67uld3MRDM+RmmNP6+GQVAvOUjDJptV5I0u4Ze5P4YZk4nTxYbSxndl6buVetJrp1D2NclLqx7hpSL7htdDN9NH7wAt0evpBYGnZjAEcIWauKleAqZXu3g5fyrkqP1y7yFA6XbiN6xa1QdB04t89/1O/w1cDnyilFU="
LINE_API_BASE = "https://api.line.biz"

def get_user_profile(user_id):
    """查詢用戶 Profile"""
    url = f"{LINE_API_BASE}/v2/bot/profile/{user_id}"
    headers = {
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"[DEBUG] GET {url}")
        print(f"[DEBUG] Status: {resp.status_code}")
        print(f"[DEBUG] Response: {resp.text}\n")
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"[ERROR] {e}\n")
        return None

def test_api():
    """測試 API 連接"""
    print("=" * 60)
    print("LINE Messaging API 測試")
    print("=" * 60)
    print(f"\nChannel Access Token: {CHANNEL_ACCESS_TOKEN[:20]}...\n")

    # 測試 1: 查詢 Bot 本身的 Profile
    print("\n[Test 1] 查詢 Bot Profile")
    url = f"{LINE_API_BASE}/v2/bot/profile"
    headers = {
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"ERROR: {e}")

    # 測試 2: 查詢指定用戶
    print("\n[Test 2] 查詢用戶 Profile (需要 User ID)")
    print("NOTE: LINE Messaging API 本身 NOT 支援【查詢歷史訊息】")
    print("  - 只能接收 webhook（新訊息時觸發）")
    print("  - 只能發送訊息")
    print("  - 需要用戶 ID 才能取得用戶資訊")
    print("\n實際上要拉歷史訊息，LINE 目前的方案是：")
    print("  1. 在 LINE Official Account Manager 後台手動導出")
    print("  2. 維護一個 webhook receiver 來記錄所有新訊息")
    print("  3. 使用「LINE Bot SDK」的其他 API（如果有）")

if __name__ == "__main__":
    test_api()
