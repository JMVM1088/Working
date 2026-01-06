#!/usr/bin/env python3
"""
🚀 COMPLETE DAILY TRADING PIPELINE
1. Generate volatility alerts (SP)
2. Perplexity sentiment (Top-20)
3. Options flow aggregation (Barchart)
4. Final filtered Top-20 dashboard
5. Save all to SQL
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date, datetime, timedelta
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# =============================
# CONFIG (update these)
# =============================
ENGINE_STR = (
    "mssql+pyodbc://@BEELINK/Stock"
    "?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
)
PERPLEXITY_API_KEY = "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"  # .env file
UNUSUAL_OPTIONS_TABLE = "dbo.UsualOptions"  # Your Barchart table name

engine = create_engine(ENGINE_STR)
today = date.today()

print(f"🚀 Daily Pipeline: {today}")

# =============================
# 1. STEP 1: Volatility Alerts (SP call)
# =============================
def run_volatility_pipeline():
    """Call your existing SP"""
    with engine.begin() as conn:
        conn.execute(text("EXEC dbo.sp_GenerateDailyVolatilityAlerts"))
        print("✅ Volatility alerts generated")

# =============================
# 2. STEP 2: Perplexity Sentiment (Top-20)
# =============================
def get_perplexity_sentiment(symbol):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": f"{symbol} stock sentiment (-1 to +1). JSON: {{\"score\":0.3,\"label\":\"Bullish\"}}"}],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code == 200:
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            data = json.loads(content)
            return data.get("score", 0.0), data.get("label", "Neutral")
        except:
            return 0.0, "Neutral"
    return 0.0, "Neutral"

def run_sentiment_pipeline():
    """Sentiment for Top-20 volatility names"""
    top20_query = """
        SELECT TOP 20 Symbol FROM (
            SELECT Symbol FROM dbo.DailyVolatilityAlerts 
            WHERE ReportDate = (SELECT MAX(ReportDate) FROM dbo.DailyVolatilityAlerts)
              AND ABS(ZScore) >= 2.0
        ) t
    """
    symbols = pd.read_sql(top20_query, engine)["Symbol"].tolist()
    
    sentiment_rows = []
    for sym in symbols:
        score, label = get_perplexity_sentiment(sym)
        sentiment_rows.append({
            "ReportDate": today,
            "Symbol": sym,
            "AvgSentiment": round(score, 4),
            "PosCount": 0,      # Safe defaults
            "NegCount": 0,
            "NeuCount": 0,
            "ArticleCount": 1,  # Perplexity summary = 1
            "SentimentLabel": label,
            "InsertedAt": datetime.now()
        })
        print(f"   {sym}: {score:.2f} ({label})")
    
    df_sentiment = pd.DataFrame(sentiment_rows)
    df_sentiment.to_sql("DailySymbolSentiment", engine, if_exists="append", index=False)
    print(f"✅ Sentiment: {len(symbols)} symbols")

# =============================
# 3. STEP 3: Options Flow (Barchart)
# =============================
def run_options_flow_pipeline():
    """Fixed aggregation with ALL columns"""
    today_str = today.strftime('%Y-%m-%d')
    
    agg_query = f"""
    INSERT INTO dbo.DailyOptionsFlow (BusinessDate, Symbol, CallVol, PutVol, CallOI, PutOI,
        TotalVol, TotalOI, VolOverOI, CallPutVolRatio, CallPutOIRatio, NetDeltaVol, FlowScore)
    SELECT 
        '{today_str}' AS BusinessDate,
        u.Symbol,
        SUM(CASE WHEN u.Type = 'Call' THEN u.Volume ELSE 0 END) AS CallVol,
        SUM(CASE WHEN u.Type = 'Put' THEN u.Volume ELSE 0 END) AS PutVol,
        SUM(CASE WHEN u.Type = 'Call' THEN u.[Open Int] ELSE 0 END) AS CallOI,
        SUM(CASE WHEN u.Type = 'Put' THEN u.[Open Int] ELSE 0 END) AS PutOI,
        SUM(u.Volume) AS TotalVol,
        SUM(u.[Open Int]) AS TotalOI,
        SUM(u.Volume)*1.0/NULLIF(SUM(u.[Open Int]),0) AS VolOverOI,
        -- ✅ FIXED: CallPutVolRatio (volume ratio)
        SUM(CASE WHEN u.Type = 'Call' THEN u.Volume ELSE 0 END)*1.0/
        NULLIF(SUM(CASE WHEN u.Type = 'Put' THEN u.Volume ELSE 0 END),0) AS CallPutVolRatio,
        -- ✅ FIXED: CallPutOIRatio (open interest ratio)
        SUM(CASE WHEN u.Type = 'Call' THEN u.[Open Int] ELSE 0 END)*1.0/
        NULLIF(SUM(CASE WHEN u.Type = 'Put' THEN u.[Open Int] ELSE 0 END),0) AS CallPutOIRatio,
        SUM(u.Volume * u.Delta) AS NetDeltaVol,
        AVG(u.[Vol/OI]) AS FlowScore
    FROM {UNUSUAL_OPTIONS_TABLE} u
    WHERE u.BusinessDate = '{today_str}' AND u.[Vol/OI] > 5.0
    GROUP BY u.BusinessDate, u.Symbol  -- ✅ Added BusinessDate
    HAVING SUM(u.Volume) >= 1000
    """
    
    with engine.begin() as conn:
        result = conn.execute(text(agg_query))
        print(f"✅ Options flow: {result.rowcount} symbols inserted")


# =============================
# 4. STEP 4: Final Dashboard (SP call)
# =============================
def run_dashboard():
    """Generate final Top-20"""
    with engine.begin() as conn:
        conn.execute(text("EXEC dbo.sp_DailyVolatilityMaster"))
    print("✅ Dashboard complete")

# =============================
# MAIN EXECUTION
# =============================
if __name__ == "__main__":
    if not PERPLEXITY_API_KEY:
        print("❌ Set PERPLEXITY_API_KEY env var")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 STARTING DAILY PIPELINE")
    print("=" * 60)
    
    try:
       # run_volatility_pipeline()
       # run_sentiment_pipeline()
        run_options_flow_pipeline()
        run_dashboard()
        
        print("\n🎉 PIPELINE COMPLETE!")
        print(f"📊 Results saved for {today}")
        print("📈 Check: EXEC dbo.sp_Top20_WithSentiment_Long")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        sys.exit(1)
