# Quick fix script - run ONCE
import pandas as pd
from sqlalchemy import create_engine
from datetime import date
import requests
import json

ENGINE_STR = (
    "mssql+pyodbc://@BEELINK/Stock"
    "?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
)
engine = create_engine(ENGINE_STR)
API_KEY = "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"

# Get today's Top-20
today_str = '2026-01-02'
top20 = pd.read_sql(f"""
    SELECT DISTINCT Symbol 
    FROM dbo.DailyVolatilityAlerts 
    WHERE ReportDate = '{today_str}' AND ABS(ZScore) >= 2.0
""", engine)["Symbol"].tolist()

print(f"🔄 Backfilling sentiment for {len(top20)} symbols on {today_str}")

for sym in top20[:5]:  # Test 5 first
    url = "https://api.perplexity.ai/chat/completions"
    resp = requests.post(url, 
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": f"{sym} sentiment (-1 to 1). JSON: {{\"score\":0.3,\"label\":\"Bullish\"}}"}]
        }
    )
    
    if resp.status_code == 200:
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            data = json.loads(content)
            score = data.get("score", 0.0)
            label = data.get("label", "Neutral")
        except:
            score, label = 0.0, "Neutral"
    else:
        score, label = 0.0, "Error"
    
    # Safe insert
    row = {
        "ReportDate": today_str,
        "Symbol": sym,
        "AvgSentiment": float(score),
        "PosCount": 1 if score > 0 else 0,
        "NegCount": 1 if score < 0 else 0,
        "NeuCount": 1 if score == 0 else 0,
        "ArticleCount": 1,
        "SentimentLabel": label,
        "InsertedAt": pd.Timestamp.now()
    }
    
    pd.DataFrame([row]).to_sql("DailySymbolSentiment", engine, if_exists="append", index=False)
    print(f"  {sym}: {score:.2f} ({label})")

print("✅ Backfill complete - rerun sp_Top20_WithSentiment_Long")
