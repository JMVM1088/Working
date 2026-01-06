import os
import requests
import json
from datetime import datetime

# Load API key (set PERPLEXITY_API_KEY env var or edit below)
API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not API_KEY:
    API_KEY = "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"  # ← PASTE YOUR KEY HERE
    print("⚠️  Using hardcoded key - set PERPLEXITY_API_KEY env var for production")

print(f"🔑 API Key loaded: {'✅' if len(API_KEY) > 10 else '❌ Invalid'}")
print(f"⏰ Testing: {datetime.now()}")

# Test API call
url = "https://api.perplexity.ai/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "sonar-pro",
    "messages": [
        {"role": "system", "content": "Respond briefly."},
        {"role": "user", "content": "TSLA stock sentiment today? Return JSON: {\"score\": 0.3, \"label\": \"Bullish\"}"}
    ],
    "temperature": 0.1,
    "max_tokens": 100
}

print("\n🚀 Sending request...")
response = requests.post(url, headers=headers, json=payload, timeout=30)

print(f"📊 Status: {response.status_code}")
print(f"📊 Credits remaining: Active through Jan 31 ($5)")

if response.status_code == 200:
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    
    print("\n✅ API CONNECTION SUCCESS!")
    print("📄 Response:")
    print(content)
    
    # Parse sentiment
    try:
        data = json.loads(content)
        print(f"🎯 TSLA: {data.get('score', 'N/A')} ({data.get('label', 'N/A')})")
    except:
        print("📄 Raw content (valid response)")
    
elif response.status_code == 401:
    print("\n❌ 401 Unauthorized")
    print("🔧 FIX:")
    print("1. Go to https://www.perplexity.ai/settings/api")
    print("2. Generate NEW API key")
    print("3. Add payment method ($5 credits)")
    print("4. Copy pxl_... key to .env")
    
elif response.status_code == 429:
    print("\n⚠️ 429 Rate limit - wait 1 min")
else:
    print(f"\n❌ Error {response.status_code}")
    print(response.text[:500])

print("\n" + "="*50)
print("✅ If 200 OK → Your API is LIVE!")
print("🔄 Rerun daily_sentiment.py → Production ready")
