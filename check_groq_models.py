import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("GROQ_API_KEY")

resp = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"}
)
models = resp.json()["data"]
ids = sorted(m["id"] for m in models)

print(f"Total models available to this key: {len(ids)}\n")
for m in ids:
    print(" -", m)

print("\nHas llama-3.3-70b-versatile:", "llama-3.3-70b-versatile" in ids)
print("Has llama-3.1-8b-instant:", "llama-3.1-8b-instant" in ids)